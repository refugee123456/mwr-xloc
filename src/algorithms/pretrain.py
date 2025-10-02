# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from src.models.encoders.graph_encoder import GraphEncoder
from src.models.aggregators.local_branch import LocalBranch
from src.models.aggregators.sym_branch import SymBranch
from src.models.aggregators.global_branch import GlobalBranch
from src.models.aggregators.fusion import FusionHead
from src.models.heads.metric_head import MetricHead
from src.losses.metric import metric_ce_loss
from src.losses.contrastive import contrastive_loss

class PretrainNet(nn.Module):
    """
    GraphEncoder -> (Local/Sym/Global) -> Fusion -> MetricHead
    - Classification: distance-softmax + cross-entropy
    - Contrastive: applied directly on the fused graph-level feature h_G (no projection head)
    """
    def __init__(
        self,
        node_dim: int = 13,
        enc_hid: int = 64,
        enc_out: int = 64,
        num_heads: int = 4,
        embed_dim: int = 128,
        num_classes: int = 2,
        metric_gamma: float = 10.0,
    ):
        super().__init__()
        self.encoder = GraphEncoder(
            in_dim=node_dim, hid_dim=enc_hid, out_dim=enc_out,
            num_heads=num_heads, dropout=0.1, use_ln=True
        )
        self.local = LocalBranch(in_dim=enc_out, embed_dim=embed_dim)
        self.sym   = SymBranch(in_dim=enc_out, embed_dim=embed_dim)
        self.glob  = GlobalBranch(in_dim=enc_out, embed_dim=embed_dim)
        self.fusion = FusionHead(embed_dim=embed_dim)

        self.head = MetricHead(
            num_classes=num_classes,
            feat_dim=embed_dim,
            gamma_init=metric_gamma,
            learnable_gamma=True,
            per_class_gamma=False,
            norm_by_dim=True,
        )

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            logits: [B, C]
            dists : None (placeholder; MetricHead internally uses distances to form logits)
            h_G   : [B, D] fused graph-level feature
        """
        enc = self.encoder(batch)  # {"H": [N, enc_out], ...}
        H = enc["H"]
        bp = batch["batch_ptr"]
        pairs = batch.get("pairs", None)

        v_loc,  _ = self.local(H, bp)
        v_sym,  _ = self.sym(H, bp, pairs=pairs)
        v_glob, _ = self.glob(H, bp)
        h_G, _ = self.fusion(v_loc, v_sym, v_glob)  # [B, D]

        logits, _ = self.head(h_G, return_dist=False)
        return logits, None, h_G


@torch.no_grad()
def proto_bootstrap_once(model: PretrainNet, batch: Dict, momentum: float = 0.0):
    """
    One-time warm start / EMA update for MetricHead prototypes using
    the current batch fused features + labels.
    """
    model.eval()
    logits, _, h = model(batch)
    y = batch["y"].long()
    model.head.init_from_means(h, y, momentum=momentum)


def compute_losses(
    model: PretrainNet,
    batch: Dict,
    contrastive_cfg: Dict,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main loss = CE; optionally add λ * contrastive(h_G, y).
    """
    y = batch["y"].long()
    logits, _, h = model(batch)

    loss_ce = metric_ce_loss(logits, y, reduction="mean")
    loss = loss_ce

    c_method = (contrastive_cfg.get("method", "none") or "none").lower()
    if c_method != "none":
        lam = float(contrastive_cfg.get("weight", 0.1))
        margin = float(contrastive_cfg.get("margin", 0.2))
        tau = float(contrastive_cfg.get("tau", 1.0))
        loss_ctr = contrastive_loss(h, y, method=c_method, margin=margin, tau=tau)
        loss = loss + lam * loss_ctr
    else:
        loss_ctr = torch.tensor(0., device=logits.device)

    scalars = dict(loss=float(loss.item()), ce=float(loss_ce.item()), ctr=float(loss_ctr.item()))
    return loss, scalars, logits.detach(), y.detach(), h.detach()
