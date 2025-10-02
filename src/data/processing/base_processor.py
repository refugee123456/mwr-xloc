# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


@dataclass
class ProcessOutput:
    nodes: np.ndarray         # [N, F=13]
    edge_index: np.ndarray    # [2, E] (undirected: both directions)
    pairs: np.ndarray         # [P, 2] (mirror pairs: (left_i, right_i))
    meta: Dict[str, Any]      # e.g., {"y": int, "row_id": int}


class BaseProcessor:
    """
    Dataset-agnostic processor interface. A domain-specific subclass must
    implement `fit(df)` and `process_row(row: pd.Series) -> ProcessOutput`.
    """
    def fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def process_row(self, row: pd.Series) -> ProcessOutput:
        raise NotImplementedError
