# -*- coding: utf-8 -*-
"""
Plot a single LEG episode under four settings:
  (A) No-FT + Recal (no VTAN)
  (B) No-FT + Recal (+VTAN)
  (C) FT + Recal (no VTAN)
  (D) FT + Recal (+VTAN)

Output:
  - 2x2 figure saved to out_dir
  - Episode metrics (acc/mcc/spec/sens/auc/gmean) printed and saved to CSV/TXT
"""
from __future__ import annotations
import os, sys, math, csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- fonts: use Times everywhere ----
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "dejavuserif"

# --- project path ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# --- project imports ---
from src.utils.config import load_yaml, ensure_dir
from src.utils.seed import set_seed
from src.data.datasets.mwr_dataset import MWRGraphDataset
from src.data.episodic_sampler import EpisodeSampler
from src.algorithms.adaptation.pfa_finetuner import FeatureExtractor
from src.algorithms.pfa import (
    instance_recalibrate, compute_prototypes,
    vtan_normalize, distance_logits
)
from src.evaluation.metrics import binary_metrics_from_logits


# ---------------- utils ----------------
def _reduce_2d(X: np.ndarray, method: str = "umap", perplexity: int = 30, seed: int = 0) -> np.ndarray:
    """UMAP (preferred) / t-SNE / PCA -> 2D; return [N, 2]"""
    m = (method or "umap").lower()

    if m == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05,
                             random_state=seed, metric="euclidean").fit_transform(X)
        except Exception:
            print("[warn] UMAP not available; fallback to t-SNE.")
            m = "tsne"

    if m == "tsne":
        try:
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, perplexity=perplexity, init="random",
                        learning_rate="auto", random_state=seed, n_iter=1000).fit_transform(X)
        except Exception:
            print("[warn] t-SNE not available; fallback to PCA.")
            m = "pca"

    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    except Exception:
        if X.shape[1] >= 2:
            return X[:, :2]
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, :X.shape[1]] = X
        return out


@torch.no_grad()
def _encode_indices(fx: FeatureExtractor, device: torch.device, ds, idx_list: List[int]):
    """Encode graphs -> 128-d features (no grad; for visualization/inference)"""
    from src.data.collate import collate_graphs
    recs = [ds[i] for i in idx_list]
    batch = collate_graphs(recs)
    to = lambda x: (x if torch.is_tensor(x) else torch.as_tensor(x, device=device))
    b = {
        "nodes": to(batch.get("nodes", batch.get("x"))).float(),
        "edge_index": to(batch["edge_index"]).long(),
        "batch_ptr": to(batch.get("batch_ptr", [0, int(batch["nodes"].shape[0])])).long(),
        "pairs": to(batch["pairs"]).long() if batch.get("pairs") is not None else None,
        "y": to(batch["y"]).long() if "y" in batch else None,
    }
    fx.eval()
    h = fx(b)  # [B, D]
    y = b.get("y", None)
    return torch.nan_to_num(h), y


def _within_between_2d(X2: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Mean within-class / between-class Euclidean distances on 2D plane."""
    n = X2.shape[0]
    if n <= 2:
        return 0.0, 0.0
    diff = X2[:, None, :] - X2[None, :, :]
    D = np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=-1)))
    iu = np.triu_indices(n, 1)
    D_ut = D[iu]
    same = (y[iu[0]] == y[iu[1]])
    w = D_ut[same]; b = D_ut[~same]
    return (float(w.mean()) if w.size else 0.0,
            float(b.mean()) if b.size else 0.0)


def _plot_panel(ax, XY: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                title: str, within: float, between: float):
    """Paper-style panel with KDE contours + class colors + correctness overlays."""
    from matplotlib.lines import Line2D

    ax.set_title(title, fontsize=14, pad=6)

    # ---- background density contours (soft) ----
    try:
        if XY.shape[0] > 20:
            from scipy.stats import gaussian_kde
            xmin, ymin = XY.min(0) - 0.5
            xmax, ymax = XY.max(0) + 0.5
            xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
            grid = np.vstack([xx.ravel(), yy.ravel()])
            kde = gaussian_kde(XY.T)
            zz = kde(grid).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=5, cmap="YlOrBr", alpha=0.25, antialiased=True)
    except Exception:
        pass  # 没装 scipy 也能正常画散点

    # ---- base: class-colored scatter (no correctness) ----
    healthy = (y_true == 0)
    cancer  = (y_true == 1)
    ax.scatter(XY[healthy, 0], XY[healthy, 1], s=16, marker='o',
               facecolors='C0', edgecolors='none', alpha=0.75)
    ax.scatter(XY[cancer, 0], XY[cancer, 1], s=20, marker='s',
               facecolors='#F2C94C', edgecolors='none', alpha=0.75)  # soft yellow

    # ---- overlays: correctness marks ----
    correct = (y_true == y_pred)
    ax.scatter(XY[correct, 0],  XY[correct, 1],  s=10, marker='.', c='limegreen', alpha=0.95)
    ax.scatter(XY[~correct, 0], XY[~correct, 1], s=26, marker='x', c='red', linewidths=0.9, alpha=0.95)

    # ---- clean axes & legend (4 entries) ----
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)

    handles = [
        Line2D([], [], marker='o', linestyle='None', markersize=6, color='C0', label='True Healthy'),
        Line2D([], [], marker='s', linestyle='None', markersize=6, color='#F2C94C', label='True Cancerous'),
        Line2D([], [], marker='.', linestyle='None', markersize=10, color='limegreen', label='Correct Prediction'),
        Line2D([], [], marker='x', linestyle='None', markersize=6, color='red', label='Incorrect Prediction'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7, frameon=False)

    # distances annotation (bottom-left)
    txt = f"Mean Within-Class Distance: {within:.2f}\nMean Between-Class Distance: {between:.2f}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9,
            va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.65, lw=0))


# ---------------- core ----------------
def run_condition(name: str,
                  fx: FeatureExtractor,
                  cfg_ft: Dict, cfg_common: Dict,
                  ds, S_idx: List[int], Q_idx: List[int],
                  device: torch.device):
    """
    Run one episode under a given setting and return data for plotting + logits for metrics.
    """
    do_ft   = bool(cfg_ft["enabled"])
    use_vtan = bool(cfg_ft["use_vtan"])

    # 1) (no-grad) encode raw features
    Zs_raw, Ys = _encode_indices(fx, device, ds, S_idx)
    Zq_raw, Yq = _encode_indices(fx, device, ds, Q_idx)
    Ys = Ys.view(-1).to(device); Yq = Yq.view(-1).to(device)

    # 2) support recalibration (+ residual blend)
    if bool(cfg_common["use_recalib"]) and Zs_raw.size(0) > 0:
        Zs_rc_raw, _, _ = instance_recalibrate(
            Zs_raw, Ys,
            tau=float(cfg_common["recalib_tau"]),
            sim=str(cfg_common["recalib_sim"]),
            rbf_gamma=float(cfg_common["recalib_rbf_gamma"]),
            exclude_self=bool(cfg_common.get("recalib_exclude_self_train", False))
        )
        alpha = float(cfg_common["recalib_blend"])
        Zs_rc = (1.0 - alpha) * Zs_raw + alpha * Zs_rc_raw
    else:
        Zs_rc = Zs_raw

    # 3) finetune if needed (IMPORTANT: enable grad)
    if do_ft:
        from src.algorithms.adaptation.pfa_finetuner import PFAMetaAdapter as _Adapter
        adapter = _Adapter(
            fx, device,
            {
                "enabled": True,
                "reload_per_episode": True,
                "bn_eval_mode": bool(cfg_common["bn_eval_mode"]),
                "inner_epochs": int(cfg_common["inner_epochs"]),
                "lr": float(cfg_common["lr"]),
                "weight_decay": float(cfg_common["weight_decay"]),
                "optimizer": str(cfg_common["optimizer"]),
                "grad_clip": float(cfg_common["grad_clip"]),
                "skip_nonfinite": bool(cfg_common["skip_nonfinite"]),
                "cls_gamma": float(cfg_common["cls_gamma"]),
                "adaptive_gamma": bool(cfg_common["adaptive_gamma"]),
                "gamma_floor": float(cfg_common["gamma_floor"]),
                "gamma_ceil": float(cfg_common["gamma_ceil"]),
                "gamma_alpha": float(cfg_common["gamma_alpha"]),
                "loss_weights": {
                    "lambda_comp": float(cfg_common["loss_weights"]["lambda_comp"]),
                    "lambda_margin": float(cfg_common["loss_weights"]["lambda_margin"]),
                    "lambda_align": float(cfg_common["loss_weights"]["lambda_align"]),
                },
                "proto_margin": float(cfg_common["proto_margin"]),
                "use_recalib": True,
                "proto_weighted": bool(cfg_common["proto_weighted"]),
                "recalib_sim": str(cfg_common["recalib_sim"]),
                "recalib_rbf_gamma": float(cfg_common["recalib_rbf_gamma"]),
                "recalib_tau": float(cfg_common["recalib_tau"]),
                "recalib_blend": float(cfg_common["recalib_blend"]),
                "recalib_exclude_self_train": bool(cfg_common["recalib_exclude_self_train"]),
                "exclude_self": bool(cfg_common["exclude_self_eval"]),
            },
            {
                "use_vtan": use_vtan,
                "norm_after_reproj": bool(cfg_common["norm_after_reproj"]),
                "scale_by_dim": bool(cfg_common["scale_by_dim"]),
            }
        )
        # run inner-loop with grad enabled
        with torch.enable_grad():
            logits, _ = adapter.run_episode(ds, S_idx, Q_idx, prog_desc=f"{name}")
        y_pred = torch.argmax(logits, dim=1)

        # embeddings used at inference for visualization
        Zs_full, Ys_full = _encode_indices(fx, device, ds, S_idx)
        if bool(cfg_common["use_recalib"]) and Zs_full.size(0) > 0:
            Zs_full_rc_raw, _, _ = instance_recalibrate(
                Zs_full, Ys_full,
                tau=float(cfg_common["recalib_tau"]),
                sim=str(cfg_common["recalib_sim"]),
                rbf_gamma=float(cfg_common["recalib_rbf_gamma"]),
                exclude_self=bool(cfg_common["exclude_self_eval"])
            )
            Zs_full_rc = Zs_full_rc_raw
        else:
            Zs_full_rc = Zs_full
        Zq = Zq_raw
        if use_vtan:
            _, Zq_adp, _ = vtan_normalize(Zs_full_rc, Zq)
        else:
            Zq_adp = Zq
        Z_used = Zq_adp

    else:
        # no-FT: direct inference
        Zs_full_rc = Zs_rc
        Zq = Zq_raw
        if use_vtan:
            _, Zq_adp, _ = vtan_normalize(Zs_full_rc, Zq)
        else:
            Zq_adp = Zq
        P = compute_prototypes(Zs_full_rc, Ys, None)
        gamma = float(cfg_common["cls_gamma"])
        logits = distance_logits(Zq_adp, P, gamma=gamma, scale_by_dim=bool(cfg_common["scale_by_dim"]))
        if bool(cfg_common["adaptive_gamma"]):
            v = float(torch.nan_to_num(logits).var().detach().cpu().item())
            g = float(max(cfg_common["gamma_floor"],
                          min(cfg_common["gamma_ceil"],
                              cfg_common["gamma_floor"] + cfg_common["gamma_alpha"] * math.log1p(max(0.0, v)))))
            logits = distance_logits(Zq_adp, P, gamma=g, scale_by_dim=bool(cfg_common["scale_by_dim"]))
        y_pred = torch.argmax(logits, dim=1)
        Z_used = Zq_adp

    # 4) 2D viz + distance stats
    XY = _reduce_2d(Z_used.detach().cpu().numpy(),
                    method=str(cfg_ft["reducer"]).lower(),
                    perplexity=int(cfg_ft["perplexity"]),
                    seed=int(cfg_ft["seed"]))
    y_true_np = Yq.detach().cpu().numpy().astype(int)
    y_pred_np = y_pred.detach().cpu().numpy().astype(int)
    within, between = _within_between_2d(XY, y_true_np)

    return dict(
        xy=XY, y_true=y_true_np, y_pred=y_pred_np,
        within=within, between=between, title=name,
        logits=logits.detach(), y_true_t=Yq.detach()
    )


# ---------------- main ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser("Plot one LEG episode under 4 settings.")
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    main_cfg = cfg["main"]
    mdl_cfg = cfg["model"]
    epi_cfg = cfg["episode"]
    ft_common = cfg["finetune_common"]
    lw = cfg["loss_weights"]

    # env
    set_seed(int(main_cfg["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(main_cfg["out_dir"]); ensure_dir(out_dir)

    # dataset
    ds = MWRGraphDataset(root=main_cfg["data_root"], split="test", keep_unknown=False)

    # backbone & ckpt
    from src.runners.meta_eval_runner import _load_backbone_from_ckpt
    fx = FeatureExtractor(
        node_dim=mdl_cfg["node_dim"], enc_hid=mdl_cfg["enc_hid"], enc_out=mdl_cfg["enc_out"],
        num_heads=mdl_cfg["num_heads"], embed_dim=mdl_cfg["embed_dim"]
    ).to(device).eval()
    loaded, missing, extra, loadable = _load_backbone_from_ckpt(fx, main_cfg["ckpt"])
    print(f"[ckpt] loaded={loaded}, missing={missing}, extra={extra if isinstance(extra,int) else len(extra)}")

    # sample one episode (deterministic)
    es = EpisodeSampler(ds, n_way=epi_cfg["way"], k_shot=epi_cfg["shot"],
                        q_query=epi_cfg["query"], seed=epi_cfg["sampler_seed"])
    for _ in range(int(epi_cfg.get("episode_index", 0))):
        es.sample_episode()
    epi = es.sample_episode()
    S_idx, Q_idx = [], []
    for _, (S, Q) in epi.items():
        S_idx += S; Q_idx += Q

    # shared params passed to the adapter
    ft_common_full = dict(
        inner_epochs=ft_common["inner_epochs"],
        lr=ft_common["lr"],
        weight_decay=ft_common["weight_decay"],
        optimizer=ft_common["optimizer"],
        grad_clip=ft_common["grad_clip"],
        skip_nonfinite=ft_common["skip_nonfinite"],
        bn_eval_mode=ft_common["bn_eval_mode"],
        reload_per_episode=ft_common["reload_per_episode"],
        cls_gamma=ft_common["cls_gamma"],
        adaptive_gamma=ft_common["adaptive_gamma"],
        gamma_floor=ft_common["gamma_floor"],
        gamma_ceil=ft_common["gamma_ceil"],
        gamma_alpha=ft_common["gamma_alpha"],
        scale_by_dim=ft_common["scale_by_dim"],
        use_recalib=ft_common["use_recalib"],
        proto_weighted=ft_common["proto_weighted"],
        recalib_sim=ft_common["recalib_sim"],
        recalib_rbf_gamma=ft_common["recalib_rbf_gamma"],
        recalib_tau=ft_common["recalib_tau"],
        recalib_blend=ft_common["recalib_blend"],
        recalib_exclude_self_train=ft_common["recalib_exclude_self_train"],
        exclude_self_eval=ft_common["exclude_self_eval"],
        norm_after_reproj=ft_common["norm_after_reproj"],
        loss_weights=lw,
        proto_margin=cfg["loss_weights"]["proto_margin"],
    )

    base_vis = dict(
        reducer=str(main_cfg.get("reducer", "umap")).lower(),
        perplexity=int(main_cfg.get("perplexity", 30)),
        seed=int(main_cfg.get("seed", 42)),
    )
    settings = [
        dict(name="(a) No-FT + Recal (no VTAN)", enabled=False, use_vtan=False),
        dict(name="(b) No-FT + Recal (+VTAN)",  enabled=False, use_vtan=True),
        dict(name="(c) FT + Recal (no VTAN)",    enabled=True,  use_vtan=False),
        dict(name="(d) FT + Recal (+VTAN)",      enabled=True,  use_vtan=True),
    ]

    panels = []
    for s in settings:
        fx.load_state_dict(loadable, strict=False)  # independence among settings
        cond_cfg = dict(enabled=s["enabled"], use_vtan=s["use_vtan"], **base_vis)
        res = run_condition(s["name"], fx, cond_cfg, ft_common_full, ds, S_idx, Q_idx, device)
        panels.append(res)

    # ---- metrics: print + save ----
    rows = []
    print("\n[Episode metrics per setting] (thr=0.5)")
    for p in panels:
        mets = binary_metrics_from_logits(p["logits"], p["y_true_t"], thr=0.5, with_auc=True)
        spec = float(mets.get("spec", 0.0)); sens = float(mets.get("sens", 0.0))
        gmean = math.sqrt(max(0.0, spec * sens))
        row = dict(
            setting=p["title"],
            acc=float(mets.get("acc", 0.0)),
            mcc=float(mets.get("mcc", 0.0)),
            spec=spec,
            sens=sens,
            auc=float(mets.get("auc", 0.0)),
            gmean=gmean,
            within=float(p["within"]),
            between=float(p["between"]),
        )
        rows.append(row)
        print(f"  {row['setting']}: "
              f"ACC={row['acc']:.4f}  MCC={row['mcc']:.4f}  "
              f"SPEC={row['spec']:.4f}  SENS={row['sens']:.4f}  "
              f"AUC={row['auc']:.4f}  G-Mean={row['gmean']:.4f}  "
              f"| within={row['within']:.2f} between={row['between']:.2f}")

    # save CSV/TXT
    with open(out_dir / "episode_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with open(out_dir / "episode_metrics.txt", "w") as f:
        for r in rows:
            f.write(
                f"{r['setting']}: ACC={r['acc']:.4f} MCC={r['mcc']:.4f} "
                f"SPEC={r['spec']:.4f} SENS={r['sens']:.4f} "
                f"AUC={r['auc']:.4f} G-Mean={r['gmean']:.4f} "
                f"| within={r['within']:.2f} between={r['between']:.2f}\n"
            )

    # ---- 2x2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=int(main_cfg.get("figure_dpi", 300)))
    axes = axes.reshape(-1)
    for ax, p in zip(axes, panels):
        _plot_panel(ax, p["xy"], p["y_true"], p["y_pred"], p["title"], p["within"], p["between"])
    plt.tight_layout()
    out_path = Path(main_cfg["out_dir"]) / "leg_episode_viz_4panels.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\n[OK] figure saved to: {out_path}")
    print(f"[OK] metrics saved to: {out_dir/'episode_metrics.csv'}")


if __name__ == "__main__":
    main()
