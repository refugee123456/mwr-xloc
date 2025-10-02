# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, List
from pathlib import Path
import os, csv, math, json, torch
from tqdm.auto import tqdm

from src.utils.config import load_yaml, ensure_dir
from src.utils.seed import set_seed
from src.evaluation.metrics import binary_metrics_from_logits
from src.evaluation.reporter import Reporter
from src.data.datasets.mwr_dataset import MWRGraphDataset
from src.data.episodic_sampler import EpisodeSampler
from src.algorithms.adaptation.pfa_finetuner import FeatureExtractor, PFAMetaAdapter


# ---------- ckpt loader ----------
def _load_backbone_from_ckpt(fx: FeatureExtractor, ckpt_path: str) -> Tuple[int, int, int, Dict[str, torch.Tensor]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    cur = fx.state_dict()
    loadable = {}
    for k, v in state.items():
        if k.startswith("encoder.") or k.startswith("local.") or k.startswith("sym.") \
            or k.startswith("glob.") or k.startswith("fusion."):
            if k in cur and cur[k].shape == v.shape:
                loadable[k] = v
    miss = set(cur.keys()) - set(loadable.keys())
    extra = set(state.keys()) - set(loadable.keys())
    fx.load_state_dict(loadable, strict=False)
    return len(loadable), len(miss), len(extra), loadable


def _mean_ci(vals: List[float], alpha: float = 0.05):
    n = max(1, len(vals))
    t = torch.tensor(vals, dtype=torch.float32)
    mean = float(t.mean().item()) if n > 0 else 0.0
    sd   = float(t.std(unbiased=True).item()) if n > 1 else 0.0
    z = 1.96  # Normal approximation
    ci = z * sd / math.sqrt(n) if n > 1 else 0.0
    return dict(mean=mean, ci_low=mean - ci, ci_high=mean + ci, std=sd, n=n)


# ---------- runner ----------
def run_from_yaml(cfg_path: str):
    cfg = load_yaml(cfg_path)
    main = cfg["main"]; model_cfg = cfg["model"]
    meta_cfg = cfg["meta"]; ft_cfg = cfg.get("finetune", {})
    pfa_cfg = cfg.get("pfa", {})

    set_seed(int(main.get("seed", 42)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(main.get("out_dir", "outputs/meta_eval"))
    ensure_dir(out_dir)
    reporter = Reporter(out_dir)
    reporter.print_env(torch.__version__, torch.cuda.is_available(), str(device))
    reporter.print_config(cfg)

    # dataset (target domain)
    root = main.get("data_root", "") or os.environ.get("MWR_DATASET_OUT", "data/processed/lung")
    ds = MWRGraphDataset(root, split="test", keep_unknown=False)
    if len(ds) == 0:
        raise RuntimeError("Empty target dataset split=test")

    # backbone & ckpt
    fx = FeatureExtractor(
        node_dim=model_cfg.get("node_dim", 13),
        enc_hid=model_cfg.get("enc_hid", 64),
        enc_out=model_cfg.get("enc_out", 64),
        num_heads=model_cfg.get("num_heads", 4),
        embed_dim=model_cfg.get("embed_dim", 128)
    ).to(device).eval()
    loaded, missing, extra, loadable = _load_backbone_from_ckpt(fx, main["ckpt"])
    print(f"[ckpt] loaded (strict=False): matched~{loaded} missing={missing} unused_in_ckpt={extra if isinstance(extra, int) else len(extra)}")

    adapter = PFAMetaAdapter(fx, device, ft_cfg, pfa_cfg)
    adapter.cache_backbone_state(loadable)  # Reset before each episode

    # episodic sampler
    nway = int(meta_cfg.get("way", 2))
    kshot = int(meta_cfg.get("shot", 5))
    qnum = int(meta_cfg.get("query", 15))
    episodes = int(meta_cfg.get("episodes", 50))
    es = EpisodeSampler(ds, n_way=nway, k_shot=kshot, q_query=qnum, seed=0)

    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    per_ep_rows: List[Dict] = []

    pbar = tqdm(range(episodes), desc="Meta-test", dynamic_ncols=True)
    for epi in pbar:
        epi_map = es.sample_episode()  # {c: (S_idx, Q_idx)}
        S_idx = []; Q_idx = []; Q_y = []
        for c, (S, Q) in epi_map.items():
            S_idx += S; Q_idx += Q
            Q_y += [int(c)] * len(Q)
        Q_y = torch.tensor(Q_y, device=device, dtype=torch.long)

        logits, dbg = adapter.run_episode(ds, S_idx, Q_idx, prog_desc=f"ft(ep{epi+1})")
        all_logits.append(logits.detach())
        all_y.append(Q_y.detach())

        # per-episode metrics (with AUC)
        mets = binary_metrics_from_logits(logits, Q_y, thr=0.5, with_auc=True)
        acc  = float(mets.get('acc', 0.0))
        mcc  = float(mets.get('mcc', 0.0))
        spec = float(mets.get('spec', 0.0))
        sens = float(mets.get('sens', 0.0))
        auc  = float(mets.get('auc', 0.0))
        loss = float(dbg.get('loss', 0.0))

        # --- NEW: gmean ---
        gmean = math.sqrt(max(0.0, spec * sens))

        # progress bar
        pbar.set_postfix(
            acc=f"{acc:.3f}", mcc=f"{mcc:.3f}", loss=f"{loss:.4f}",
            gamma=f"{dbg['gamma']:.2f}", gmean=f"{gmean:.3f}"
        )
        if bool(main.get("debug", True)):
            print(
                f"[episode {epi+1}] acc={acc:.4f} mcc={mcc:.4f} loss={loss:.4f} "
                f"spec={spec:.4f} sens={sens:.4f} auc={auc:.4f} gmean={gmean:.4f} "
                f"| gamma={dbg['gamma']:.2f} proto-sep(sq)={dbg.get('proto_sep_sq',0.0):.4f} "
                f"best_loss={dbg.get('best_loss', 0.0):.4f} best_epoch={dbg.get('best_epoch',0)}"
            )

        # save row
        per_ep_rows.append(dict(
            episode=epi+1, acc=acc, mcc=mcc, spec=spec, sens=sens, auc=auc, gmean=gmean,
            loss=loss, gamma=float(dbg['gamma']),
            proto_sep_sq=float(dbg.get('proto_sep_sq', 0.0)),
            best_loss=float(dbg.get('best_loss', 0.0)),
            best_epoch=int(dbg.get('best_epoch', 0)),
        ))

    # ===== Summary (aggregate + distribution) =====
    # 1) Aggregate all queries together
    L = torch.cat(all_logits, 0)
    Y = torch.cat(all_y, 0)
    mets = binary_metrics_from_logits(L, Y, thr=0.5, with_auc=True)

    # --- NEW: aggregate gmean ---
    spec_agg = float(mets.get('spec', 0.0))
    sens_agg = float(mets.get('sens', 0.0))
    mets['gmean'] = math.sqrt(max(0.0, spec_agg * sens_agg))

    reporter.log_test(dict(loss=0.0), mets)

    # 2) Per-episode distribution stats (mean and 95% CI)
    keys = ["acc", "mcc", "spec", "sens", "auc", "gmean", "loss", "gamma", "proto_sep_sq"]
    summary = {}
    for k in keys:
        vals = [float(r[k]) for r in per_ep_rows]
        summary[k] = _mean_ci(vals, alpha=0.05)

    # Write CSV
    csv_path = out_dir / "episodes_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["episode"] + keys + ["best_loss", "best_epoch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_ep_rows:
            writer.writerow(r)

    # Write summary text
    txt_path = out_dir / "episodes_summary.txt"
    with open(txt_path, "w") as f:
        f.write("Per-episode metrics mean ± 95% CI (N={}):\n".format(len(per_ep_rows)))
        for k in keys:
            s = summary[k]
            line = f"- {k}: mean={s['mean']:.4f}  95%CI=[{s['ci_low']:.4f}, {s['ci_high']:.4f}]  std={s['std']:.4f}\n"
            f.write(line)

    # Also print summary to console
    print("\n[episode distribution] mean ± 95% CI")
    for k in keys:
        s = summary[k]
        print(f"  {k:>12s}: {s['mean']:.4f}  [ {s['ci_low']:.4f} , {s['ci_high']:.4f} ]  std={s['std']:.4f}")

    reporter.finish()
