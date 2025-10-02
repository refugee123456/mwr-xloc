# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List
import re
import numpy as np
import pandas as pd

from .base_processor import BaseProcessor, ProcessOutput
from .ambient_norm import fit_ref_slopes, apply_ref_norm_row
from .node_features_common import build_breast_13d_features
from .graph_builder import build_breast_edges


def _sort_by_index(cols: List[str]) -> List[str]:
    def key(c: str) -> int:
        m = re.findall(r"(\d+)", c)
        return int(m[0]) if m else 0
    return sorted(cols, key=key)


class BreastProcessor(BaseProcessor):
    """
    Breast stage-1 (drop-missing version):
      - DO NOT impute. If any temperature-related column is NaN -> drop the row.
      - Choose ref_index ∈ {0,1} for deep/surface reference; do linear correction:
            t' = t + A * (ref_mean - ref_row)
        where A is the slope of each column regressed on the chosen reference.
      - 13-D node features follow your table strictly:
            feat[5]/[6] use ref#1; feat[7]/[8] use ref#2.
      - Graph: per-side complete + mirror edges.
    """

    def __init__(self, ambient_col: str = "Ambient", ref_index: int = 1):
        self.ambient_col = ambient_col     # kept for compatibility; not used in this policy
        self.ref_index = int(ref_index)    # 0 or 1; default 1 (e.g., T2 & Skin T2)

        # column caches
        self.deep_L: List[str] = []
        self.deep_R: List[str] = []
        self.surf_L: List[str] = []
        self.surf_R: List[str] = []
        self.ref_d_cols: List[str] = []    # e.g., ["T1", "T2"]
        self.ref_s_cols: List[str] = []    # e.g., ["Skin T1", "Skin T2"]
        self.label_col = "Cancer"


        # ref-based normalization params
        self.ref_params: Dict[str, object] = {}

    # -------- column discovery -------- #
    def _discover_columns(self, df: pd.DataFrame) -> None:
        rx = lambda p: [c for c in df.columns if re.match(p, c)]
        self.deep_L = _sort_by_index(rx(r"^L\d+$"))
        self.deep_R = _sort_by_index(rx(r"^R\d+$"))
        self.surf_L = _sort_by_index(rx(r"^Skin L\d+$"))
        self.surf_R = _sort_by_index(rx(r"^Skin R\d+$"))
        self.ref_d_cols = _sort_by_index(rx(r"^T\d+$"))          # need 2
        self.ref_s_cols = _sort_by_index(rx(r"^Skin T\d+$"))     # need 2

        assert len(self.ref_d_cols) >= 2 and len(self.ref_s_cols) >= 2, \
            "breast needs two deep/surface refs"
        assert len(self.deep_L) == len(self.deep_R), "L/R deep count mismatch"
        assert len(self.surf_L) == len(self.surf_R), "L/R surface count mismatch"

    def _required_temp_cols(self) -> List[str]:
        # all temperature-related columns that must be present & non-NaN
        return self.deep_L + self.deep_R + self.surf_L + self.surf_R + self.ref_d_cols + self.ref_s_cols

    # -------- fit (drop rows with any NaN in required columns) -------- #
    def fit(self, df: pd.DataFrame) -> None:
        self._discover_columns(df)

        # ensure numeric
        for c in self._required_temp_cols():
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # choose reference columns for normalization
        ref_d_col = self.ref_d_cols[self.ref_index]
        ref_s_col = self.ref_s_cols[self.ref_index]

        # strict drop-missing: fit on rows that have NO NaN in required temp columns
        fit_df = df.dropna(subset=self._required_temp_cols(), how="any").copy()

        # fit ref-based slopes
        self.ref_params = fit_ref_slopes(
            fit_df, ref_d_col, ref_s_col,
            deep_cols=self.deep_L + self.deep_R,
            surf_cols=self.surf_L + self.surf_R,
        )

    # -------- process one row (drop-missing) -------- #
    def process_row(self, row: pd.Series) -> ProcessOutput:
        needed = self._required_temp_cols()
        # if any NaN -> drop (by raising to let runner skip)
        if row[needed].isna().any():
            raise ValueError("Row contains NaN in temperature columns (drop-missing policy).")

        # reference-based normalization (no overrides; we use row's own values)
        deep_all, surf_all = apply_ref_norm_row(
            row, self.ref_params,
            deep_cols=self.deep_L + self.deep_R,
            surf_cols=self.surf_L + self.surf_R,
        )

        # split back to L/R
        M = len(self.deep_L)
        t_d_left,  t_d_right  = deep_all[:M], deep_all[M:]
        t_s_left,  t_s_right  = surf_all[:M], surf_all[M:]

        # features: use BOTH refs for features (5/6 use ref#1; 7/8 use ref#2)
        tref_d1 = float(row[self.ref_d_cols[0]])
        tref_s1 = float(row[self.ref_s_cols[0]])
        tref_d2 = float(row[self.ref_d_cols[1]])
        tref_s2 = float(row[self.ref_s_cols[1]])

        nodes, pairs, _ = build_breast_13d_features(
            t_d_left, t_d_right, t_s_left, t_s_right,
            tref_d1=tref_d1, tref_s1=tref_s1, tref_d2=tref_d2, tref_s2=tref_s2,
        )

        edge_index = build_breast_edges(num_left=M)

        y = int(row[self.label_col]) if self.label_col in row.index and pd.notna(row[self.label_col]) else -1
        meta = {
            "y": y,
            "drop_policy": "drop-missing",
            "ref_used_deep": self.ref_params.get("ref_d_col", ""),
            "ref_used_surf": self.ref_params.get("ref_s_col", ""),
        }
        return ProcessOutput(nodes=nodes, edge_index=edge_index, pairs=pairs, meta=meta)
