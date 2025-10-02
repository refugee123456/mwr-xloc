# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import re
import numpy as np
import pandas as pd

from .base_processor import BaseProcessor, ProcessOutput
from .graph_builder import build_breast_edges
from .node_features_common import build_leg_13d_features


def _sort_by_index(cols: List[str]) -> List[str]:
    def key(c: str) -> int:
        m = re.findall(r"(\d+)", c)
        return int(m[0]) if m else 0
    return sorted(cols, key=key)


def _trimmed_mean(v: np.ndarray, trim: float = 0.25) -> float:
    v = np.sort(v[~np.isnan(v)])
    if v.size == 0:
        return np.nan
    k = int(v.size * trim)
    k = max(0, min(k, v.size - 1))
    if v.size - 2 * k <= 0:
        return float(np.mean(v))
    return float(np.mean(v[k: v.size - k]))


class LegProcessor(BaseProcessor):
    """
    Leg dataset (standing only):
      - per-row group-mean imputation (L/R × deep/surface)
      - pseudo references via trimmed-mean (deep/surface) -- supports different trim ratios
      - linear correction vs pseudo-ref (per-column slopes)
      - 13-D node features (feat#6/7 and #8/9 equal t - pseudo reference)
      - graph: per-side complete + mirror
      - label rule: L_diagnosis & R_diagnosis both 'Normal' -> y=0, otherwise y=1
    """

    def __init__(
        self,
        ambient_col: str = "Room Temp",
        n_points: int = 12,
        trim_deep: float = 0.25,
        trim_surf: float = 0.25,
    ):
        self.ambient_col = ambient_col
        self.n_points = n_points
        self.trim_deep = float(trim_deep)
        self.trim_surf = float(trim_surf)

        self.deep_L: List[str] = []
        self.deep_R: List[str] = []
        self.surf_L: List[str] = []
        self.surf_R: List[str] = []

        # optional diagnosis columns
        self.left_diag_col: str | None = None
        self.right_diag_col: str | None = None

        # slopes & stats for correction
        self.deep_slopes: Dict[str, float] = {}
        self.surf_slopes: Dict[str, float] = {}
        self.ref_mean_deep: float = 0.0
        self.ref_mean_surf: float = 0.0

        # column means as fallback
        self.col_means: Dict[str, float] = {}

    # ----------------- column discovery (standing only) ----------------- #
    def _discover_columns(self, df: pd.DataFrame) -> None:
        rx = lambda p: [c for c in df.columns if re.match(p, c, flags=re.IGNORECASE)]

        # temperatures (standing)
        self.deep_L = _sort_by_index(rx(r"^L\d+_standing$"))
        self.deep_R = _sort_by_index(rx(r"^R\d+_standing$"))
        self.surf_L = _sort_by_index(rx(r"^Skin L\d+_standing$"))
        self.surf_R = _sort_by_index(rx(r"^Skin R\d+_standing$"))

        # sanity
        assert len(self.deep_L) == len(self.deep_R) == self.n_points, \
            f"deep L/R points mismatch or != {self.n_points}"
        assert len(self.surf_L) == len(self.surf_R) == self.n_points, \
            f"surface L/R points mismatch or != {self.n_points}"

        # optional diagnoses (be robust to naming/case/space)
        def _find_one(cands: List[str]) -> str | None:
            for c in df.columns:
                lc = c.strip().lower()
                if lc in cands:
                    return c
            return None

        self.left_diag_col  = _find_one(["l_diagnosis", "left_diagnosis", "l diagnosis", "left diagnosis"])
        self.right_diag_col = _find_one(["r_diagnosis", "right_diagnosis", "r diagnosis", "right diagnosis"])

    # ----------------- helpers ----------------- #
    @staticmethod
    def _impute_row_group_mean(row: pd.Series, cols: List[str], fallback: List[float]) -> np.ndarray:
        """
        Numeric-first fast path: if the columns are already numeric, take values in place; otherwise to_numeric once.
        """
        sub = row[cols]
        vals = sub.to_numpy()
        if np.issubdtype(vals.dtype, np.number):
            vals = vals.astype(float, copy=False)
        else:
            vals = pd.to_numeric(sub, errors="coerce").to_numpy(dtype=float)

        mask = ~np.isnan(vals)
        if mask.any():
            m = float(np.mean(vals[mask]))
            vals[~mask] = m
        else:
            vals = np.array(fallback, dtype=float)
        return vals

    @staticmethod
    def _apply_linear_correct(vec: np.ndarray, slope: List[float], ref_row: float, ref_mean: float) -> np.ndarray:
        return (vec + np.asarray(slope, dtype=float) * (ref_mean - float(ref_row))).astype(np.float32)

    # label = 0 iff both sides are "normal" (case-insensitive); else 1
    def _infer_label(self, row: pd.Series) -> int:
        if self.left_diag_col is None or self.right_diag_col is None:
            return -1  # no labels available

        def _is_normal(v) -> bool:
            if pd.isna(v):
                return False
            s = str(v).strip().lower()
            if s in {"0", "normal", "healthy", "normal.", "normal/healthy"}:
                return True
            return s.startswith("norm")

        l_ok = _is_normal(row[self.left_diag_col])
        r_ok = _is_normal(row[self.right_diag_col])
        return 0 if (l_ok and r_ok) else 1

    # ----------------- fit ----------------- #
    def fit(self, df: pd.DataFrame) -> None:
        # copy to avoid SettingWithCopyWarning
        df = df.copy()

        self._discover_columns(df)

        temp_cols = self.deep_L + self.deep_R + self.surf_L + self.surf_R
        # Unified .loc numeric casting to completely eliminate SettingWithCopyWarning
        for c in temp_cols:
            df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

        self.col_means = {c: float(df[c].mean()) for c in temp_cols}

        deep_arr, surf_arr = [], []
        for _, row in df.iterrows():
            dL = self._impute_row_group_mean(row, self.deep_L, [self.col_means[c] for c in self.deep_L])
            dR = self._impute_row_group_mean(row, self.deep_R, [self.col_means[c] for c in self.deep_R])
            sL = self._impute_row_group_mean(row, self.surf_L, [self.col_means[c] for c in self.surf_L])
            sR = self._impute_row_group_mean(row, self.surf_R, [self.col_means[c] for c in self.surf_R])
            deep_arr.append(_trimmed_mean(np.concatenate([dL, dR]), trim=self.trim_deep))
            surf_arr.append(_trimmed_mean(np.concatenate([sL, sR]), trim=self.trim_surf))

        ref_deep = np.asarray(deep_arr, dtype=float)
        ref_surf = np.asarray(surf_arr, dtype=float)
        self.ref_mean_deep = float(np.nanmean(ref_deep))
        self.ref_mean_surf = float(np.nanmean(ref_surf))

        def fit_slopes(cols: List[str], ref: np.ndarray) -> Dict[str, float]:
            slopes: Dict[str, float] = {}
            x = np.stack([ref, np.ones_like(ref)], axis=1)  # [N,2]
            for c in cols:
                y = df[c].to_numpy(dtype=float)
                ok = ~np.isnan(y) & ~np.isnan(ref)
                if ok.sum() < 3:
                    slopes[c] = 0.0
                    continue
                beta, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
                slopes[c] = float(beta[0])
            return slopes

        self.deep_slopes = fit_slopes(self.deep_L + self.deep_R, ref_deep)
        self.surf_slopes = fit_slopes(self.surf_L + self.surf_R, ref_surf)

    # ----------------- process one row ----------------- #
    def process_row(self, row: pd.Series) -> ProcessOutput:
        dL = self._impute_row_group_mean(row, self.deep_L, [self.col_means[c] for c in self.deep_L])
        dR = self._impute_row_group_mean(row, self.deep_R, [self.col_means[c] for c in self.deep_R])
        sL = self._impute_row_group_mean(row, self.surf_L, [self.col_means[c] for c in self.surf_L])
        sR = self._impute_row_group_mean(row, self.surf_R, [self.col_means[c] for c in self.surf_R])

        ref_d = _trimmed_mean(np.concatenate([dL, dR]), trim=self.trim_deep)
        ref_s = _trimmed_mean(np.concatenate([sL, sR]), trim=self.trim_surf)

        dL = self._apply_linear_correct(dL, [self.deep_slopes[c] for c in self.deep_L], ref_d, self.ref_mean_deep)
        dR = self._apply_linear_correct(dR, [self.deep_slopes[c] for c in self.deep_R], ref_d, self.ref_mean_deep)
        sL = self._apply_linear_correct(sL, [self.surf_slopes[c] for c in self.surf_L], ref_s, self.ref_mean_surf)
        sR = self._apply_linear_correct(sR, [self.surf_slopes[c] for c in self.surf_R], ref_s, self.ref_mean_surf)

        nodes, pairs, _ = build_leg_13d_features(dL, dR, sL, sR, tref_d=ref_d, tref_s=ref_s)
        edge_index = build_breast_edges(num_left=self.n_points)

        y = self._infer_label(row)
        meta = {"y": int(y), "pose": "standing", "ref_d": float(ref_d), "ref_s": float(ref_s),
                "trim_deep": self.trim_deep, "trim_surf": self.trim_surf}
        return ProcessOutput(nodes=nodes, edge_index=edge_index, pairs=pairs, meta=meta)
