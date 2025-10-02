# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np


def collate_graphs(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Concatenate multiple sample graphs into one big graph (preserving sample order).

    Returns:
      nodes      [N_total, 13]
      edge_index [2, E_total]  (indices are offset-adjusted)
      pairs      [M_total, 2]  (indices are offset-adjusted)
      y          [B]
      batch_ptr  [B+1]         (prefix sums; node start/end for each sample)
      ids / paths as lists for traceability
    """
    if len(samples) == 0:
        # Empty-batch fallback (should rarely happen)
        return {
            "nodes": np.zeros((0, 13), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "pairs": np.zeros((0, 2), dtype=np.int64),
            "y": np.zeros((0,), dtype=np.int64),
            "batch_ptr": np.zeros((1,), dtype=np.int64),
            "ids": np.zeros((0,), dtype=np.int64),
            "paths": [],
        }

    N_prefix = [0]
    nodes_list, edges_list, pairs_list, y_list = [], [], [], []
    ids, paths = [], []

    for s in samples:
        nodes = np.asarray(s["nodes"], dtype=np.float32)
        ei = np.asarray(s["edge_index"], dtype=np.int64)
        prs = np.asarray(s["pairs"], dtype=np.int64)
        y = int(s["y"])

        offset = int(N_prefix[-1])
        nodes_list.append(nodes)
        edges_list.append(ei + offset)   # shift edge indices by current offset
        pairs_list.append(prs + offset)  # shift pair indices by current offset
        y_list.append(y)
        ids.append(int(s.get("id", -1)))
        paths.append(str(s.get("path", "")))

        N_prefix.append(offset + nodes.shape[0])

    out = {
        "nodes": np.concatenate(nodes_list, axis=0),
        "edge_index": np.concatenate(edges_list, axis=1) if len(edges_list) else np.zeros((2,0), np.int64),
        "pairs": np.concatenate(pairs_list, axis=0) if len(pairs_list) else np.zeros((0,2), np.int64),
        "y": np.asarray(y_list, dtype=np.int64),
        "batch_ptr": np.asarray(N_prefix, dtype=np.int64),
        "ids": np.asarray(ids, dtype=np.int64),
        "paths": paths,
    }
    return out


def _parse_episode_args(args: Tuple, kwargs: Dict[str, Any]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Unified parsing for episode inputs:
      - Positional:  collate_episode(support_list, query_list)
      - Keywords:    collate_episode(support=..., query=...)
      - Legacy keys: collate_episode(S=..., Q=...)
      - Single dict: collate_episode({"support":..., "query":...}) / {"S":..., "Q":...}
    """
    if "support" in kwargs and "query" in kwargs:
        return kwargs["support"], kwargs["query"]
    if "S" in kwargs and "Q" in kwargs:
        return kwargs["S"], kwargs["Q"]
    if len(args) == 2:
        return args[0], args[1]
    if len(args) == 1 and isinstance(args[0], dict):
        d = args[0]
        if "support" in d and "query" in d:
            return d["support"], d["query"]
        if "S" in d and "Q" in d:
            return d["S"], d["Q"]
    raise TypeError("collate_episode expects (support, query) or keywords support=/query= (or S=/Q=).")


def collate_episode(*args, **kwargs) -> Dict[str, Any]:
    """
    Collate support/query independently with collate_graphs, and return both key sets
    for compatibility with different upper-layer implementations.
    See _parse_episode_args for accepted calling conventions.
    """
    support, query = _parse_episode_args(args, kwargs)
    S = collate_graphs(support)
    Q = collate_graphs(query)
    # Return both naming schemes to avoid upper-layer divergence
    pack = {"support": S, "query": Q, "S": S, "Q": Q}
    return pack
