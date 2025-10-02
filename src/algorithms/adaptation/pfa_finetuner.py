# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.algorithms.pfa import (
    instance_recalibrate, compute_prototypes,
    distance_logits, l2_normalize, pairwise_sqdist, vtan_normalize
)
from src.models.encoders.graph_encoder import GraphEncoder
from src.models.aggregators.local_branch import LocalBranch
from src.models.aggregators.sym_branch import SymBranch
from src.models.aggregators.global_branch import GlobalBranch
from src.models.aggregators.fusion import FusionHead
from src.data.collate import collate_graphs

# ===== backbone: graph -> 128-d representation =====
class FeatureExtractor(nn.Module):
    def __init__(self, node_dim=13, enc_hid=64, enc_out=64, num_heads=4, embed_dim=128):
        super().__init__()
        self.encoder = GraphEncoder(
            in_dim=node_dim, hid_dim=enc_hid, out_dim=enc_out,
            num_heads=num_heads, dropout=0.1, use_ln=True
        )
        self.local = LocalBranch(in_dim=enc_out, embed_dim=embed_dim)
        self.sym = SymBranch(in_dim=enc_out, embed_dim=embed_dim)
        self.glob = GlobalBranch(in_dim=enc_out, embed_dim=embed_dim)
        self.fusion = FusionHead(embed_dim=embed_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        enc_out = self.encoder(batch)
        H = enc_out["H"]
        bp = batch["batch_ptr"]
        pairs = batch.get("pairs", None)
        v_loc, _ = self.local(H, bp)
        v_sym, _ = self.sym(H, bp, pairs=pairs)
        v_glob, _ = self.glob(H, bp)
        h, _ = self.fusion(v_loc, v_sym, v_glob)  # [B, D]
        return torch.nan_to_num(h)

# ===== utils =====
def _to_cuda_batch(collated: Dict, device: torch.device) -> Dict:
    out = {}
    to = lambda x: (x if torch.is_tensor(x) else torch.as_tensor(x)).to(device)
    out["nodes"] = to(collated.get("nodes", collated.get("x"))).float()
    out["edge_index"] = to(collated["edge_index"]).long()
    out["batch_ptr"] = to(collated.get("batch_ptr", [0, int(out["nodes"].size(0))])).long()
    out["pairs"] = to(collated["pairs"]).long() if collated.get("pairs") is not None else None
    if "y" in collated:
        out["y"] = to(collated["y"]).long().view(-1)
    return out

def _set_bn_eval(m: nn.Module):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

def _compaction_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.nan_to_num(z)
    if z.numel() == 0 or y is None or y.numel() == 0:
        return z.new_tensor(0.0)
    loss = z.new_tensor(0.0)
    for c in y.unique():
        m = (y == c)
        if int(m.sum()) == 0:
            continue
        mu = z[m].mean(0, keepdim=True)
        loss = loss + (z[m] - mu).pow(2).sum(dim=1).mean()
    return loss / max(1, int(y.unique().numel()))

def _margin_hinge_sq(protos: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    C = int(protos.size(0))
    if C <= 1:
        return protos.new_tensor(0.0)
    D2 = pairwise_sqdist(protos, protos)
    m2 = margin * margin
    loss = 0.0
    cnt = 0
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            loss += torch.clamp(m2 - D2[i, j], min=0.0)
            cnt += 1
    return loss / max(1, cnt)

def _coral(ps: torch.Tensor, pq: torch.Tensor) -> torch.Tensor:
    if ps.numel() == 0 or pq.numel() == 0:
        return ps.new_tensor(0.0)
    def _mean_cov(x: torch.Tensor):
        mu = x.mean(0, keepdim=True)
        xc = x - mu
        C = (xc.t() @ xc) / max(1, x.size(0) - 1)
        return mu.squeeze(0), C
    mu1, C1 = _mean_cov(ps)
    mu2, C2 = _mean_cov(pq)
    return (mu1 - mu2).pow(2).mean() + (C1 - C2).pow(2).mean()

# ===== Episode-level adaptation (statistics alignment only) =====
class PFAMetaAdapter:
    def __init__(self, fx: FeatureExtractor, device: torch.device, cfg_finetune: Dict, cfg_pfa: Dict):
        self.fx = fx.to(device); self.device = device

        self.do_ft = bool(cfg_finetune.get("enabled", True))
        self.reload_per_episode = bool(cfg_finetune.get("reload_per_episode", True))
        self.bn_eval_mode = bool(cfg_finetune.get("bn_eval_mode", True))

        self.inner_epochs = int(cfg_finetune.get("inner_epochs", 10))

        self.cls_gamma = float(cfg_finetune.get("cls_gamma", 16.0))
        self.adapt_gamma = bool(cfg_finetune.get("adaptive_gamma", True))
        self.gamma_floor = float(cfg_finetune.get("gamma_floor", 14.0))
        self.gamma_ceil = float(cfg_finetune.get("gamma_ceil", 36.0))
        self.gamma_alpha = float(cfg_finetune.get("gamma_alpha", 2.0))

        lw = cfg_finetune.get("loss_weights", {})
        self.lw_con = float(lw.get("lambda_con", 0.0))
        self.lw_comp = float(lw.get("lambda_comp", 1.0))
        self.lw_margin = float(lw.get("lambda_margin", 0.1))
        self.lw_align = float(lw.get("lambda_align", 0.2))
        self.proto_margin = float(cfg_finetune.get("proto_margin", 0.5))

        self.lr = float(cfg_finetune.get("lr", 1.0e-3))
        self.wd = float(cfg_finetune.get("weight_decay", 1.0e-4))
        self.optim_name = (cfg_finetune.get("optimizer", "adamw") or "adamw").lower()
        self.grad_clip = float(cfg_finetune.get("grad_clip", 1.0))
        self.skip_nonfinite = bool(cfg_finetune.get("skip_nonfinite", True))

        self.use_recalib = bool(cfg_finetune.get("use_recalib", True))
        self.proto_weighted = bool(cfg_finetune.get("proto_weighted", True))
        self.recalib_tau = float(cfg_finetune.get("recalib_tau", 0.35))
        self.recalib_sim = (cfg_finetune.get("recalib_sim", "rbf") or "rbf")
        self.recalib_rbf_gamma = float(cfg_finetune.get("recalib_rbf_gamma", 0.5))
        self.recalib_exclude_self_train = bool(cfg_finetune.get("recalib_exclude_self_train", False))
        self.recalib_exclude_self_eval = bool(cfg_finetune.get("exclude_self", True))
        self.recalib_blend = float(cfg_finetune.get("recalib_blend", 0.5))

        self.use_vtan = bool(cfg_pfa.get("use_vtan", True))
        self.reproj_query = False
        self.scale_by_dim = bool(cfg_pfa.get("scale_by_dim", False))
        self.norm_after = bool(cfg_pfa.get("norm_after_reproj", False))

        self._backbone_state: Dict[str, torch.Tensor] = {}

    def cache_backbone_state(self, loadable: Dict[str, torch.Tensor]):
        self._backbone_state = {k: v.clone().detach() for k, v in loadable.items()}

    def reload_backbone(self):
        if self._backbone_state:
            self.fx.load_state_dict(self._backbone_state, strict=False)

    @staticmethod
    def _adaptive_gamma_from_logits(logits: torch.Tensor, floor: float, ceil: float, alpha: float) -> float:
        v = float(torch.nan_to_num(logits).var().detach().cpu().item())
        g = floor + alpha * math.log1p(max(0.0, v))
        return float(max(floor, min(ceil, g)))

    def _build_optimizer(self):
        params = [p for p in self.fx.parameters() if p.requires_grad]
        if self.optim_name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
        elif self.optim_name == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        else:
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=0.9)

    def _encode_indices(self, ds, idx_list: List[int], require_grad: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        recs = [ds[i] for i in idx_list]
        batch = collate_graphs(recs)
        b = _to_cuda_batch(batch, self.device)
        if self.bn_eval_mode:
            self.fx.apply(_set_bn_eval)
        if require_grad:
            self.fx.train(True)
            h = self.fx(b)
        else:
            self.fx.eval()
            with torch.no_grad():
                h = self.fx(b)
        y = b.get("y", None)
        return torch.nan_to_num(h), y

    # ================== Main episode routine ==================
    def run_episode(self, ds, S_idx: List[int], Q_idx: List[int], prog_desc: str = "") -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1) Reload backbone at the start of each episode
        if self.reload_per_episode:
            self.reload_backbone()

        opt = self._build_optimizer() if self.do_ft else None

        # Best snapshot (in-memory only)
        best_loss = float("inf")
        best_epoch = -1
        best_state: Optional[Dict[str, torch.Tensor]] = None
        last_dbg: Dict[str, float] = {}

        # 2) Inner-loop optimization
        pbar = tqdm(range(self.inner_epochs), desc=prog_desc, leave=False, dynamic_ncols=True)
        for epoch in pbar:
            # Encode support & query (training phase uses gradients)
            Zs_raw, Ys = self._encode_indices(ds, S_idx, require_grad=self.do_ft)
            Zq_raw, _  = self._encode_indices(ds, Q_idx, require_grad=self.do_ft)

            # Support recalibration + residual blend
            if self.use_recalib and Zs_raw.size(0) > 0:
                Zs_rc_raw, w_list, _ = instance_recalibrate(
                    Zs_raw, Ys,
                    tau=self.recalib_tau, sim=self.recalib_sim,
                    rbf_gamma=self.recalib_rbf_gamma,
                    exclude_self=self.recalib_exclude_self_train
                )
                alpha = float(self.recalib_blend)
                Zs_rc = (1.0 - alpha) * Zs_raw + alpha * Zs_rc_raw
            else:
                Zs_rc, w_list = Zs_raw, None

            # Prototypes
            P = compute_prototypes(Zs_rc, Ys, w_list if self.proto_weighted else None)

            # Statistic alignment losses
            loss_comp = 0.5 * _compaction_loss(Zs_raw, Ys) + 0.5 * _compaction_loss(Zs_rc, Ys)
            loss_margin = _margin_hinge_sq(P, margin=self.proto_margin)
            loss_align = _coral(Zs_rc, Zq_raw)
            loss = self.lw_comp * loss_comp + self.lw_margin * loss_margin + self.lw_align * loss_align

            # Optimize & record best
            if self.do_ft:
                if not torch.isfinite(loss) and self.skip_nonfinite:
                    cur_loss_val = float("inf")
                    last_dbg = {"loss": float("nan")}
                else:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.fx.parameters(), self.grad_clip)
                    opt.step()
                    cur_loss_val = float(loss.item())
                    last_dbg = {
                        "loss": cur_loss_val,
                        "lcmp": float(loss_comp.item()),
                        "lmar": float(loss_margin.item()),
                        "lali": float(loss_align.item()),
                    }

                # Best model (strictly by minimal loss)
                if cur_loss_val < best_loss:
                    best_loss = cur_loss_val
                    best_epoch = epoch + 1
                    # Store on CPU to save GPU memory
                    best_state = {k: v.detach().cpu().clone() for k, v in self.fx.state_dict().items()}

            else:
                last_dbg = {"loss": 0.0}

            pbar.set_postfix(loss=f"{last_dbg['loss']:.4f}")

        # 3) Inference: load best model (if finetuning enabled and a snapshot exists)
        if self.do_ft and best_state is not None:
            self.fx.load_state_dict(best_state, strict=False)

        # ===== Inference stage =====
        with torch.no_grad():
            Zs_full, Ys_full = self._encode_indices(ds, S_idx, require_grad=False)
            Zq, Yq = self._encode_indices(ds, Q_idx, require_grad=False)

            if self.use_recalib and Zs_full.size(0) > 0:
                Zs_full_rc_raw, w_full, _ = instance_recalibrate(
                    Zs_full, Ys_full,
                    tau=self.recalib_tau, sim=self.recalib_sim,
                    rbf_gamma=self.recalib_rbf_gamma,
                    exclude_self=self.recalib_exclude_self_eval
                )
                Zs_full_rc = Zs_full_rc_raw  # use pure recalibration at inference
            else:
                Zs_full_rc, w_full = Zs_full, None

            P0 = compute_prototypes(Zs_full_rc, Ys_full, w_full if self.proto_weighted else None)

            if self.use_vtan:
                Zs_adp, Zq_adp, (mu, std) = vtan_normalize(Zs_full_rc, Zq)
            else:
                Zs_adp, Zq_adp = Zs_full_rc, Zq
                mu = std = torch.tensor(0.0, device=self.device)

            if self.norm_after:
                Zs_adp = l2_normalize(Zs_adp, dim=-1)
                Zq_adp = l2_normalize(Zq_adp, dim=-1)
                P = l2_normalize(P0, dim=-1)
            else:
                P = P0

            logits = distance_logits(Zq_adp, P, gamma=self.cls_gamma, scale_by_dim=self.scale_by_dim)
            gamma_used = self.cls_gamma
            if self.adapt_gamma:
                g = self._adaptive_gamma_from_logits(logits, self.gamma_floor, self.gamma_ceil, self.gamma_alpha)
                logits = distance_logits(Zq_adp, P, gamma=g, scale_by_dim=self.scale_by_dim)
                gamma_used = g

            if P.size(0) == 2:
                proto_sep_sq = float(((P[0] - P[1]) ** 2).sum().item())
            else:
                proto_sep_sq = float(pairwise_sqdist(P, P).mean().item())

            dbg = dict(last_dbg)
            dbg.update(dict(
                gamma=float(gamma_used),
                w_norm=0.0,
                proto_sep_sq=proto_sep_sq,
                vtan_mu_min=float(mu.min()) if torch.is_tensor(mu) and getattr(mu, "numel", lambda:0)() > 0 else 0.0,
                vtan_mu_max=float(mu.max()) if torch.is_tensor(mu) and getattr(mu, "numel", lambda:0)() > 0 else 0.0,
                vtan_std_min=float(std.min()) if torch.is_tensor(std) and getattr(std, "numel", lambda:0)() > 0 else 0.0,
                vtan_std_max=float(std.max()) if torch.is_tensor(std) and getattr(std, "numel", lambda:0)() > 0 else 0.0,
                best_loss=float(best_loss if math.isfinite(best_loss) else 0.0),
                best_epoch=int(best_epoch if best_epoch >= 0 else 0),
            ))
            return logits, dbg
