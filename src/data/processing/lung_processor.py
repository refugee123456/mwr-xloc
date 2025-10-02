# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Any
import re
import numpy as np
import pandas as pd

from .base_processor import BaseProcessor, ProcessOutput
from .ambient_norm import fit_ref_slopes, apply_ref_norm_row
from .node_features_common import build_lung_13d_features
from .graph_builder import build_breast_edges


def _sort_by_index(cols: List[str]) -> List[str]:
    def key(c: str) -> int:
        m = re.findall(r"(\d+)", str(c))
        return int(m[0]) if m else 0
    return sorted(cols, key=key)


class LungProcessor(BaseProcessor):
    """
    Lung stage-1 preprocessing
    - Missing-value handling: per-row group mean imputation for each side × deep/surface
      (if an entire group is missing, fall back to the column-wise global mean)
    - Normalization: pick a reference point ref_index ∈ {0,1} (T1/T2 and Skin T1/T2),
      and apply per-column linear correction
          t' = t + a * (ref_mean - ref_row)
      (slope a is obtained from least-squares regression y ~ ref; deep/surface are fitted separately)
    - Node features: unified 13-D (same as breast/leg); feat[5/6] use ref#1, feat[7/8] use ref#2
    - Graph: complete graph per side + mirror edges
    - Labels: use the `Diagnosis` column: value 0 → y=0 (healthy); any other non-missing value → y=1; missing → y=-1
    """

    def __init__(self, ref_index: int = 1):
        self.ref_index = int(ref_index)    # 0 -> T1/Skin T1, 1 -> T2/Skin T2

        # Column caches
        self.deep_L: List[str] = []
        self.deep_R: List[str] = []
        self.surf_L: List[str] = []
        self.surf_R: List[str] = []
        self.ref_d_cols: List[str] = []    # e.g., ["T1", "T2"]
        self.ref_s_cols: List[str] = []    # e.g., ["Skin T1", "Skin T2"]

        # Normalization parameters (based on reference points)
        self.ref_params: Dict[str, Any] = {}

        # Column means (fallback)
        self.col_means: Dict[str, float] = {}

        # Label column name (prefer "Diagnosis")
        self.label_col: str | None = None

    # -------- Column discovery --------
    def _discover_columns(self, df: pd.DataFrame) -> None:
        rx = lambda p: [c for c in df.columns if re.match(p, str(c), flags=re.IGNORECASE)]

        # Standard naming (aligned with breast)
        self.deep_L = _sort_by_index(rx(r"^L\d+$"))
        self.deep_R = _sort_by_index(rx(r"^R\d+$"))
        self.surf_L = _sort_by_index(rx(r"^Skin L\d+$"))
        self.surf_R = _sort_by_index(rx(r"^Skin R\d+$"))
        self.ref_d_cols = _sort_by_index(rx(r"^T\d+$"))
        self.ref_s_cols = _sort_by_index(rx(r"^Skin T\d+$"))

        # Be lenient with some possible surface naming variants (if not matched)
        if len(self.surf_L) == 0 or len(self.surf_R) == 0:
            cand_sL = _sort_by_index(rx(r"^(?:Skin\s*)?L(?:skin)?\d+$"))
            cand_sR = _sort_by_index(rx(r"^(?:Skin\s*)?R(?:skin)?\d+$"))
            if len(self.surf_L) == 0 and cand_sL:
                self.surf_L = cand_sL
            if len(self.surf_R) == 0 and cand_sR:
                self.surf_R = cand_sR

        assert len(self.deep_L) == len(self.deep_R), "L/R deep count mismatch"
        assert len(self.surf_L) == len(self.surf_R), "L/R surface count mismatch"
        assert len(self.ref_d_cols) >= 2 and len(self.ref_s_cols) >= 2, "need two deep/surface refs"

        # Label column: prefer "Diagnosis"
        for c in df.columns:
            if str(c).strip().lower() == "diagnosis":
                self.label_col = c
                break
        if self.label_col is None:
            # Fallback (still prefer Diagnosis when present)
            lowers = {str(c).strip().lower(): c for c in df.columns}
            for key in ["diagnosis", "label", "y", "status", "result"]:
                if key in lowers:
                    self.label_col = lowers[key]
                    break

    def _required_temp_cols(self) -> List[str]:
        return self.deep_L + self.deep_R + self.surf_L + self.surf_R + self.ref_d_cols + self.ref_s_cols

    # -------- Per-row group-mean imputation --------
    @staticmethod
    def _impute_row_group_mean(row: pd.Series, cols: List[str], fallback: List[float]) -> np.ndarray:
        vals = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if mask.any():
            mu = float(vals[mask].mean())
            vals[~mask] = mu
        else:
            vals = np.array(fallback, dtype=float)
        return vals

    # -------- Fit (slopes + reference statistics) --------
    def fit(self, df: pd.DataFrame) -> None:
        self._discover_columns(df)

        # Cast all required columns to numeric; also cache column-wise means as fallbacks
        for c in self._required_temp_cols():
            df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")
        self.col_means = {c: float(df[c].mean()) for c in self._required_temp_cols()}

        # Choose reference columns (deep/surface respectively)
        ref_d_col = self.ref_d_cols[self.ref_index]
        ref_s_col = self.ref_s_cols[self.ref_index]

        # Fit slopes based on reference points
        self.ref_params = fit_ref_slopes(
            df,
            ref_d_col=ref_d_col,
            ref_s_col=ref_s_col,
            deep_cols=self.deep_L + self.deep_R,
            surf_cols=self.surf_L + self.surf_R,
        )

    # -------- Process a single row --------
    def process_row(self, row: pd.Series) -> ProcessOutput:
        # 1) Per-row group-mean imputation for side × deep/surface
        dL = self._impute_row_group_mean(row, self.deep_L, [self.col_means[c] for c in self.deep_L])
        dR = self._impute_row_group_mean(row, self.deep_R, [self.col_means[c] for c in self.deep_R])
        sL = self._impute_row_group_mean(row, self.surf_L, [self.col_means[c] for c in self.surf_L])
        sR = self._impute_row_group_mean(row, self.surf_R, [self.col_means[c] for c in self.surf_R])

        # 2) Reference-based normalization (use fitted slopes; pass imputed values to avoid re-reading raw row)
        deep_all, surf_all = apply_ref_norm_row(
            row, self.ref_params,
            deep_cols=self.deep_L + self.deep_R,
            surf_cols=self.surf_L + self.surf_R,
            deep_values_override=np.concatenate([dL, dR]),
            surf_values_override=np.concatenate([sL, sR]),
        )

        # 3) Split back into left/right
        M = len(self.deep_L)
        t_d_left,  t_d_right  = deep_all[:M], deep_all[M:]
        t_s_left,  t_s_right  = surf_all[:M], surf_all[M:]

        # 4) Two reference points (fallback to column means if missing) — used by 13-D features at indices 5/6/7/8
        def _get(c: str) -> float:
            v = row.get(c, np.nan)
            return float(v) if np.isfinite(v) else float(self.col_means[c])

        tref_d1 = _get(self.ref_d_cols[0]);  tref_s1 = _get(self.ref_s_cols[0])
        tref_d2 = _get(self.ref_d_cols[1]);  tref_s2 = _get(self.ref_s_cols[1])

        nodes, pairs, _ = build_lung_13d_features(
            t_d_left, t_d_right, t_s_left, t_s_right,
            tref_d1=tref_d1, tref_s1=tref_s1, tref_d2=tref_d2, tref_s2=tref_s2,
        )

        # 5) Graph
        edge_index = build_breast_edges(num_left=M)

        # 6) Label: from Diagnosis column, 0 → y=0; other non-missing → y=1; missing → y=-1
        y = -1
        if (self.label_col is not None) and (self.label_col in row.index):
            val = row[self.label_col]
            if pd.isna(val):
                y = -1
            else:
                # Try numeric first
                try:
                    f = float(val)
                    y = 0 if abs(f) < 1e-9 else 1
                except Exception:
                    s = str(val).strip().lower()
                    if s in {"0", "healthy", "healthy group", "normal", "control"}:
                        y = 0
                    else:
                        y = 1

        meta = {
            "y": int(y),
            "policy": "row-impute(L/R×deep/surf)+ref-normalize",
            "ref_used_deep": self.ref_params.get("ref_d_col", ""),
            "ref_used_surf": self.ref_params.get("ref_s_col", ""),
        }
        return ProcessOutput(nodes=nodes, edge_index=edge_index, pairs=pairs, meta=meta)
