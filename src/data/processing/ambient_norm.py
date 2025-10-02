# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd

# -------------------- Existing: normalization based on ambient temperature (kept) -------------------- #
def fit_env_slopes(
    df: pd.DataFrame,
    ambient_col: str,
    deep_cols: Sequence[str],
    surf_cols: Sequence[str],
) -> Dict[str, float]:
    """Linear fit temp ~ ambient for deep/surface groups."""
    x = pd.to_numeric(df[ambient_col], errors="coerce").to_numpy(dtype=float)
    target_env = float(np.nanmean(x))

    def _slope(cols):
        ss = []
        for c in cols:
            y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 3:
                # y ≈ a * x + b; keep only a
                a, b = np.polyfit(x[mask], y[mask], deg=1)
            else:
                a = 0.0
            ss.append(a)
        return float(np.nanmean(ss)) if len(ss) else 0.0

    a_deep = _slope(deep_cols)
    a_surf = _slope(surf_cols)
    return {"a_deep": a_deep, "a_surf": a_surf, "target_env": target_env, "ambient_col": ambient_col}


def apply_env_norm_row(
    row: pd.Series,
    params: Dict[str, float],
    deep_cols: Sequence[str],
    surf_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """t' = t + a*(E_target - E_row)"""
    E = float(row[params["ambient_col"]])
    dt = params["target_env"] - E
    deep = pd.to_numeric(row[deep_cols], errors="coerce").to_numpy(dtype=float)
    surf = pd.to_numeric(row[surf_cols], errors="coerce").to_numpy(dtype=float)
    deep_n = deep + params["a_deep"] * dt
    surf_n = surf + params["a_surf"] * dt
    return deep_n, surf_n

# -------------------- New: normalization based on “reference points” -------------------- #
def fit_ref_slopes(
    df: pd.DataFrame,
    ref_d_col: str,
    ref_s_col: str,
    deep_cols: Sequence[str],
    surf_cols: Sequence[str],
) -> Dict[str, object]:
    """
    For each temperature column y, regress against the in-row reference ref via least squares: y ≈ A*ref + B, keep A only.
    Returns:
      {
        "ref_d_col": ..., "ref_s_col": ...,
        "ref_mean_d": float, "ref_mean_s": float,
        "deep_slopes": {col -> a}, "surf_slopes": {col -> a},
        "ref_col_means": {ref_d_col: mean, ref_s_col: mean}
      }
    """
    ref_d = pd.to_numeric(df[ref_d_col], errors="coerce").to_numpy(dtype=float)
    ref_s = pd.to_numeric(df[ref_s_col], errors="coerce").to_numpy(dtype=float)
    ref_mean_d = float(np.nanmean(ref_d))
    ref_mean_s = float(np.nanmean(ref_s))

    def _fit(cols, ref):
        slopes: Dict[str, float] = {}
        X = np.stack([ref, np.ones_like(ref)], axis=1)  # [N,2]
        for c in cols:
            y = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(ref) & np.isfinite(y)
            if m.sum() >= 3:
                beta, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
                slopes[c] = float(beta[0])
            else:
                slopes[c] = 0.0
        return slopes

    return {
        "ref_d_col": ref_d_col,
        "ref_s_col": ref_s_col,
        "ref_mean_d": ref_mean_d,
        "ref_mean_s": ref_mean_s,
        "deep_slopes": _fit(deep_cols, ref_d),
        "surf_slopes": _fit(surf_cols, ref_s),
        "ref_col_means": {
            ref_d_col: float(np.nanmean(ref_d)),
            ref_s_col: float(np.nanmean(ref_s)),
        },
    }


def apply_ref_norm_row(
    row: pd.Series,
    params: Dict[str, object],
    deep_cols: Sequence[str],
    surf_cols: Sequence[str],
    deep_values_override: np.ndarray | None = None,
    surf_values_override: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference-based normalization: t' = t + a*(ref_mean - ref_row)
    - You may pass deep_values_override / surf_values_override for values after in-row mean imputation;
      otherwise the function reads original values from row[cols].
    """
    ref_d = row[params["ref_d_col"]]
    ref_s = row[params["ref_s_col"]]
    # If a reference value is missing, fill with the column mean
    if not np.isfinite(ref_d):
        ref_d = params["ref_col_means"][params["ref_d_col"]]
    if not np.isfinite(ref_s):
        ref_s = params["ref_col_means"][params["ref_s_col"]]

    ref_d = float(ref_d)
    ref_s = float(ref_s)
    dd = (params["ref_mean_d"] - ref_d)
    ds = (params["ref_mean_s"] - ref_s)

    if deep_values_override is None:
        deep = pd.to_numeric(row[deep_cols], errors="coerce").to_numpy(dtype=float)
    else:
        deep = np.asarray(deep_values_override, dtype=float)
    if surf_values_override is None:
        surf = pd.to_numeric(row[surf_cols], errors="coerce").to_numpy(dtype=float)
    else:
        surf  = np.asarray(surf_values_override, dtype=float)

    deep_slopes = np.array([params["deep_slopes"].get(c, 0.0) for c in deep_cols], dtype=float)
    surf_slopes = np.array([params["surf_slopes"].get(c, 0.0) for c in surf_cols], dtype=float)

    deep_n = (deep + deep_slopes * dd).astype(np.float32)
    surf_n = (surf  + surf_slopes * ds).astype(np.float32)
    return deep_n, surf_n
