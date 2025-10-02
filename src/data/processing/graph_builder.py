# src/data/graph_builder.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np

def _env_bool(key: str, default: bool | None = None) -> bool | None:
    v = os.environ.get(key, "").strip().lower()
    if v in {"1", "true", "yes"}:
        return True
    if v in {"0", "false", "no"}:
        return False
    return default

def build_breast_edges(num_left: int, add_self_loops: bool | None = None) -> np.ndarray:
    """
    Build edges for the breast graph:
      - Left-side complete graph + right-side complete graph (without self-loops by default), optionally add self-loops
      - Bidirectional mirror edges between left and right counterparts
      - Return an int64 numpy array with shape [2, E]
    Edge count:
      Without self-loops: E = 2*M*(M-1) + 2*M = 2*M*M
      With self-loops:    E = 2*M*(M-1) + 2*M (mirror) + 2*M (self-loops) = 2*M*M + 2*M
    """
    M = int(num_left)
    N = 2 * M

    if add_self_loops is None:
        add_self_loops = _env_bool("MWR_ADD_SELF_LOOP", default=True)

    edges: list[tuple[int, int]] = []

    # Left complete graph
    for i in range(M):
        for j in range(M):
            if i == j:
                if add_self_loops:
                    edges.append((i, i))
            else:
                edges.append((i, j))

    # Right complete graph
    for i in range(M, N):
        for j in range(M, N):
            if i == j:
                if add_self_loops:
                    edges.append((i, i))
            else:
                edges.append((i, j))

    # Mirror edges (bidirectional)
    for i in range(M):
        edges.append((i, M + i))
        edges.append((M + i, i))

    # -> [2, E]
    edge_index = np.asarray(edges, dtype=np.int64).T
    assert edge_index.shape[0] == 2, f"edge_index bad shape: {edge_index.shape}"
    return edge_index
