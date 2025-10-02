# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Sequence
import numpy as np


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / (sd + eps)


def build_breast_13d_features(
    t_d_left: np.ndarray,          # deep, left side [M]
    t_d_right: np.ndarray,         # deep, right side [M]
    t_s_left: np.ndarray,          # surface, left side [M]
    t_s_right: np.ndarray,         # surface, right side [M]
    tref_d1: float, tref_s1: float,    # reference #1 (deep/surface)
    tref_d2: float, tref_s2: float,    # reference #2 (deep/surface)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        nodes:      [N=2M, 13]
        pairs:      [M, 2]  (left_i, right_i) indices
        mirror_idx: [N]     index of the mirror counterpart for each node
    """
    assert t_d_left.shape == t_d_right.shape == t_s_left.shape == t_s_right.shape
    M = t_d_left.size
    # Concatenate into N=2M
    t_d = np.concatenate([t_d_left,  t_d_right], axis=0)   # [N]
    t_s = np.concatenate([t_s_left,  t_s_right], axis=0)   # [N]

    # Mirror index mapping
    left_idx  = np.arange(0, M, dtype=int)
    right_idx = np.arange(M, 2*M, dtype=int)
    mirror_idx = np.concatenate([right_idx, left_idx], axis=0)  # left→right, right→left

    # In-side (same side) mask for local z-score
    side_mask_left  = np.zeros(2*M, dtype=bool);  side_mask_left[left_idx]  = True
    side_mask_right = ~side_mask_left

    # 3~5
    delta_i     = t_d - t_s
    delta_sym_d = t_d - t_d[mirror_idx]
    delta_sym_s = t_s - t_s[mirror_idx]

    # 6~9 w.r.t. the two reference points
    d_ref1 = t_d - float(tref_d1)
    s_ref1 = t_s - float(tref_s1)
    d_ref2 = t_d - float(tref_d2)
    s_ref2 = t_s - float(tref_s2)

    # 10,11 per-side local z-scores (deep/surface computed separately)
    z_d_local = np.empty_like(t_d)
    z_s_local = np.empty_like(t_s)
    z_d_local[side_mask_left]  = _zscore(t_d[side_mask_left])
    z_d_local[side_mask_right] = _zscore(t_d[side_mask_right])
    z_s_local[side_mask_left]  = _zscore(t_s[side_mask_left])
    z_s_local[side_mask_right] = _zscore(t_s[side_mask_right])

    # 12,13 global z-scores (surface and deep separately)
    a_s_global = _zscore(t_s)
    a_d_global = _zscore(t_d)

    # Stack 1..13 as columns
    nodes = np.stack([
        t_d,                # 1
        t_s,                # 2
        delta_i,            # 3
        delta_sym_d,        # 4
        delta_sym_s,        # 5
        d_ref1,             # 6
        s_ref1,             # 7
        d_ref2,             # 8  ← second deep reference
        s_ref2,             # 9  ← second surface reference
        z_d_local,          # 10
        z_s_local,          # 11
        a_s_global,         # 12
        a_d_global,         # 13
    ], axis=1)  # [N, 13]

    pairs = np.stack([left_idx, right_idx], axis=1)  # [M, 2]
    return nodes.astype(np.float32), pairs, mirror_idx

def build_leg_13d_features(
    t_d_left: np.ndarray,
    t_d_right: np.ndarray,
    t_s_left: np.ndarray,
    t_s_right: np.ndarray,
    *,
    tref_d: float,
    tref_s: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    13-D node features for LEG/LUNG-like datasets (no real reference points).

    Feature order (per node i):
      1  t_d[i]                                  (deep temp, normalized)
      2  t_s[i]                                  (surface temp, normalized)
      3  Δt_i          = t_d[i] - t_s[i]
      4  Δt^d_sym      = t_d[i] - t_d[mirror(i)]
      5  Δt^s_sym      = t_s[i] - t_s[mirror(i)]
      6  Δt^d_ref      = t_d[i] - tref_d         (pseudo deep ref)
      7  Δt^s_ref      = t_s[i] - tref_s         (pseudo surface ref)
      8  Δt^d_ref2     = t_d[i] - tref_d         (== 6 by spec)
      9  Δt^s_ref2     = t_s[i] - tref_s         (== 7 by spec)
     10  z_local^d     (z-score inside its side, deep)
     11  z_local^s     (z-score inside its side, surface)
     12  α^s_g         = (t_s[i]-mean(t_s[:])) / std(t_s[:])
     13  α^d_g         = (t_d[i]-mean(t_d[:])) / std(t_d[:])

    NOTE: 8/9 intentionally duplicate 6/7 to follow the user's requirement.
    """
    L = np.asarray(t_d_left,  dtype=np.float32).reshape(-1)
    R = np.asarray(t_d_right, dtype=np.float32).reshape(-1)
    Ls = np.asarray(t_s_left,  dtype=np.float32).reshape(-1)
    Rs = np.asarray(t_s_right, dtype=np.float32).reshape(-1)

    n  = L.shape[0]
    td = np.concatenate([L,  R ], axis=0)
    ts = np.concatenate([Ls, Rs], axis=0)

    feats = np.zeros((2*n, 13), dtype=np.float32)

    md, ms = float(td.mean()), float(ts.mean())
    sd, ss = float(td.std() + 1e-12), float(ts.std() + 1e-12)

    Ld_m, Rd_m = float(L.mean()),  float(R.mean())
    Ld_s, Rd_s = float(L.std() + 1e-12), float(R.std() + 1e-12)
    Ls_m, Rs_m = float(Ls.mean()), float(Rs.mean())
    Ls_s, Rs_s = float(Ls.std() + 1e-12), float(Rs.std() + 1e-12)

    for i in range(2*n):
        mi   = i + n if i < n else i - n      # mirror index
        td_i = float(td[i])
        ts_i = float(ts[i])

        feats[i, 0] = td_i
        feats[i, 1] = ts_i
        feats[i, 2] = td_i - ts_i
        feats[i, 3] = td_i - float(td[mi])
        feats[i, 4] = ts_i - float(ts[mi])
        feats[i, 5] = td_i - float(tref_d)    # Δt^d_ref
        feats[i, 6] = ts_i - float(tref_s)    # Δt^s_ref
        feats[i, 7] = feats[i, 5]             # Δt^d_ref2 (same as 6)
        feats[i, 8] = feats[i, 6]             # Δt^s_ref2 (same as 7)

        if i < n:
            feats[i, 9]  = (td_i - Ld_m) / Ld_s
            feats[i, 10] = (ts_i - Ls_m) / Ls_s
        else:
            feats[i, 9]  = (td_i - Rd_m) / Rd_s
            feats[i, 10] = (ts_i - Rs_m) / Rs_s

        feats[i, 11] = (ts_i - ms) / ss
        feats[i, 12] = (td_i - md) / sd

    pairs = np.stack([np.arange(n), np.arange(n) + n], axis=1).astype(np.int64)
    return feats, pairs, {"tref_d": float(tref_d), "tref_s": float(tref_s)}



def build_lung_13d_features(
    t_d_left: np.ndarray, t_d_right: np.ndarray,
    t_s_left: np.ndarray, t_s_right: np.ndarray,
    tref_d1: float, tref_s1: float, tref_d2: float, tref_s2: float,
):
    """
    13-D node features for Lung, identical definition to Breast:
      0 t_d, 1 t_s, 2 Δi, 3 Δd_sym, 4 Δs_sym,
      5 (t_d - tref_d1), 6 (t_s - tref_s1),
      7 (t_d - tref_d2), 8 (t_s - tref_s2),
      9 z_d (local per-side), 10 z_s (local per-side),
      11 z_s (global), 12 z_d (global)
    Returns: nodes [N,13], pairs [M,2], extras (dict)
    """
    # Reuse the breast constructor to avoid code duplication
    return build_breast_13d_features(
        t_d_left, t_d_right, t_s_left, t_s_right,
        tref_d1=tref_d1, tref_s1=tref_s1, tref_d2=tref_d2, tref_s2=tref_s2,
    )
