# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch

# ---------- shape helpers ----------
def _as_1d_labels(y: torch.Tensor) -> torch.Tensor:
    """y -> [N] with values in {0,1}"""
    if y.dim() == 0:
        y = y.view(1)
    if y.dim() == 1:
        return y.long()
    if y.size(-1) == 1:            # [N,1] -> [N]
        return y.squeeze(-1).long()
    if y.size(-1) == 2:            # one-hot -> argmax
        return y.argmax(dim=-1).long()
    return y.reshape(-1).long()

def _as_1d_binary_logit(logits: torch.Tensor) -> torch.Tensor:
    """
    logits -> [N] “positive-class logit”
    - [N]          : as-is
    - [N,1]        : squeeze
    - [N,2]        : logit1 - logit0
    """
    if logits.dim() == 0:
        logits = logits.view(1)
    if logits.dim() == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.squeeze(-1).reshape(-1)
    if logits.size(-1) == 2:
        return (logits[..., 1] - logits[..., 0]).reshape(-1)
    raise ValueError(f"Expect binary logits with last dim 1 or 2, got shape {tuple(logits.shape)}")

def _ensure_prob1(x: torch.Tensor) -> torch.Tensor:
    """
    Convert input (probabilities or logits with shape [N] / [N,1] / [N,2]) to [N] of p(y=1).
    - If [N]: if it looks like probabilities (0..1), return as-is; otherwise treat as logits and apply sigmoid
    - If [N,1]: same as above
    - If [N,2]:
        * If it looks like probabilities (min>=0, max<=1 and each row sums ≈ 1), take x[:,1]
        * Otherwise treat as logits, take sigmoid(logit1 - logit0)
    """
    if x.dim() == 0:
        x = x.view(1)
    # [N] or [N,1]
    if x.dim() == 1 or (x.dim() == 2 and x.size(-1) == 1):
        v = x.reshape(-1)
        if (v.min() >= -1e-6) and (v.max() <= 1 + 1e-6):
            return v
        return torch.sigmoid(v)
    # [N,2]
    if x.dim() == 2 and x.size(-1) == 2:
        # Check if it looks like probabilities
        row_sum = x.sum(dim=-1)
        looks_prob = (x.min() >= -1e-4) and (x.max() <= 1 + 1e-4) and torch.allclose(
            row_sum, torch.ones_like(row_sum), atol=1e-3
        )
        if looks_prob:
            return x[:, 1]
        else:
            return torch.sigmoid(x[:, 1] - x[:, 0])
    raise ValueError(f"Expect binary prob/logits with last dim 1 or 2, got shape {tuple(x.shape)}")

# ---------- core helpers ----------
def _conf_counts(pred: torch.Tensor, y: torch.Tensor) -> Tuple[int,int,int,int]:
    pred = pred.long().reshape(-1)
    y = _as_1d_labels(y)
    assert pred.numel() == y.numel(), f"size mismatch: pred={pred.shape}, y={y.shape}"
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn

def metrics_from_counts(tp:int, tn:int, fp:int, fn:int) -> Dict[str, float]:
    tot  = max(1, tp + tn + fp + fn)
    acc  = (tp + tn) / tot
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    g    = float(np.sqrt(max(0.0, sens * spec)))
    denom = float(np.sqrt(max(1e-12, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    mcc  = ((tp * tn) - (fp * fn)) / denom
    return dict(acc=acc, spec=spec, sens=sens, gmean=g, mcc=mcc, gmean_loss=(1.0 - g))

# ---------- public ----------
def binary_metrics_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    thr: float = 0.5,
    with_auc: bool = False
) -> Dict[str, float]:
    """Binary-classification metrics compatible with [N]/[N,1]/[N,2] logits."""
    logit1d = _as_1d_binary_logit(logits)     # [N]
    prob1   = torch.sigmoid(logit1d)          # [N]
    pred    = (prob1 >= float(thr)).long()    # [N]
    tp, tn, fp, fn = _conf_counts(pred, y)
    m = metrics_from_counts(tp, tn, fp, fn)
    if with_auc:
        m["auc"] = auc_roc_prob(prob1, y)
    return m

def scan_best_threshold_mcc(prob_or_logits: torch.Tensor, y: torch.Tensor) -> Tuple[float, Dict[str,float]]:
    """
    Accepts probabilities or logits with shapes [N]/[N,1]/[N,2], returns the threshold maximizing MCC and the metrics.
    """
    p1 = _ensure_prob1(prob_or_logits)  # [N]
    yy = _as_1d_labels(y)
    p = p1.detach().cpu().numpy().astype(np.float64).ravel()
    t = yy.detach().cpu().numpy().astype(np.int64).ravel()
    cand = np.unique(np.concatenate([p, np.array([0.5], dtype=p.dtype)])).tolist()
    best = (-1.0, 0.5, None)
    for thr in cand:
        pred = (p >= float(thr)).astype(np.int64)
        tp = int(((pred == 1) & (t == 1)).sum())
        tn = int(((pred == 0) & (t == 0)).sum())
        fp = int(((pred == 1) & (t == 0)).sum())
        fn = int(((pred == 0) & (t == 1)).sum())
        m = metrics_from_counts(tp, tn, fp, fn)
        if m["mcc"] > best[0]:
            best = (m["mcc"], float(thr), m)
    return best[1], best[2]

def auc_roc_prob(prob_or_logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Lightweight ROC-AUC:
    - Prefer sklearn when available;
    - Otherwise use a rank-based approximation.
    Input can be probabilities or logits with shapes [N]/[N,1]/[N,2].
    """
    p1 = _ensure_prob1(prob_or_logits).detach().cpu().reshape(-1).numpy()
    t  = _as_1d_labels(y).detach().cpu().numpy()
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(t, p1))
    except Exception:
        pos = p1[t == 1]; neg = p1[t == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        cmp  = (pos[:, None] > neg[None, :]).mean()
        ties = (pos[:, None] == neg[None, :]).mean()
        return float(cmp + 0.5 * ties)
