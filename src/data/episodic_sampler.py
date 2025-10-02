# src/data/episodic_sampler.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


class BalancedRandomSampler:
    """
    Class-balanced random sampler for binary classification (iterable of indices; not a torch Sampler).
    - Balances only among labeled samples (y ∈ {0,1}); unlabeled (-1) are ignored.
    - With a fixed seed, the prefix of the sequence is reproducible.
    """
    def __init__(self, dataset, shuffle: bool = True, seed: int = 0):
        self.ds = dataset
        self.shuffle = shuffle
        self.seed = int(seed)

        pos, neg = [], []
        for i in range(len(dataset)):
            y = int(dataset[i]["y"])
            if y == 1: pos.append(i)
            elif y == 0: neg.append(i)
        self.pos = pos
        self.neg = neg

    def __iter__(self):
        if len(self.pos) == 0 or len(self.neg) == 0:
            # Degenerate case: just yield sequential indices
            yield from range(len(self.ds))
            return

        rs = np.random.RandomState(self.seed)
        pos = np.array(self.pos, dtype=int).copy()
        neg = np.array(self.neg, dtype=int).copy()
        if self.shuffle:
            rs.shuffle(pos)
            rs.shuffle(neg)

        m = min(len(pos), len(neg))
        pos = pos[:m]; neg = neg[:m]

        interleave = np.empty((2*m,), dtype=int)
        interleave[0::2] = pos
        interleave[1::2] = neg
        yield from interleave.tolist()


class EpisodeSampler:
    """
    Binary few-shot episode sampler (balanced support; globally unbalanced query).
    Returns:
        { class_value (int 0/1 or the single class for 1-way): (support_idx_list, query_idx_list) }
    The query set is drawn globally (q_query total) from the remaining labeled samples
    and then partitioned by label, so per-class query counts can be unequal (not balanced).
    """

    def __init__(self, dataset, n_way: int = 2, k_shot: int = 1, q_query: int = 1, seed: int = 0):
        assert n_way in (1, 2), "current EpisodeSampler supports 1-way or 2-way (binary)"
        self.ds = dataset
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.q_query = int(q_query)      # Interpreted as the total number: draw q_query globally
        self.seed = int(seed)

        # Build class → indices mapping (only 0/1)
        pos, neg = [], []
        for i in range(len(dataset)):
            y = int(dataset[i]["y"])
            if y == 1: pos.append(i)
            elif y == 0: neg.append(i)
        self.pool = {1: pos, 0: neg}

        # Monotonic counter to get different seeds per episode while keeping reproducibility
        self._counter = 0

    def _rng(self) -> np.random.RandomState:
        # Use a different seed each call (based on initial seed + counter) for reproducible experiments
        rs = np.random.RandomState(self.seed + self._counter)
        self._counter += 1
        return rs

    def _draw_indices(self, pool: List[int], k: int, rs: np.random.RandomState) -> List[int]:
        """Draw k indices from pool; if insufficient, sample with replacement to fill; prefer no replacement when possible."""
        pool = np.asarray(pool, dtype=int)
        if len(pool) == 0:
            return []
        if len(pool) >= k:
            pick = rs.choice(pool, size=k, replace=False)
        else:
            # If insufficient: first take all without replacement, then fill the rest with replacement
            base = rs.choice(pool, size=len(pool), replace=False)
            extra = rs.choice(pool, size=(k - len(pool)), replace=True)
            pick = np.concatenate([base, extra], axis=0)
        return pick.tolist()

    def sample_episode(self) -> Dict[int, Tuple[List[int], List[int]]]:
        rs = self._rng()

        # 1) Choose participating classes
        if self.n_way == 2:
            classes = [0, 1]
            # Fallback to 1-way if one class has no data
            avail = [c for c in classes if len(self.pool[c]) > 0]
            if len(avail) < 2:
                classes = avail[:1]
        else:
            avail = [c for c in (0, 1) if len(self.pool[c]) > 0]
            if len(avail) == 0:
                raise RuntimeError("not enough data to sample a 1-way episode")
            classes = [rs.choice(avail)]

        # 2) Support: class-balanced, k_shot per participating class
        support: Dict[int, List[int]] = {}
        for c in classes:
            support[c] = self._draw_indices(self.pool[c], self.k_shot, rs)

        # 3) Query: draw q_query globally from the remaining labeled (0/1) samples (unbalanced)
        support_set = set([idx for lst in support.values() for idx in lst])

        # All labeled candidates
        all_known = self.pool[0] + self.pool[1]
        remain = np.array([i for i in all_known if i not in support_set], dtype=int)

        q = self.q_query
        if len(remain) == 0:
            query_global = []
        else:
            replace = len(remain) < q
            query_global = rs.choice(remain, size=(min(q, len(remain)) if not replace else q),
                                     replace=replace).tolist()

        # 4) Keep the original return format: route global queries to their ground-truth classes
        epi: Dict[int, Tuple[List[int], List[int]]] = {}
        # Initialize query containers for participating classes
        for c in classes:
            epi[c] = (support[c], [])  # (support, query)

        # Assign global queries to participating classes only (typical few-shot setting)
        for idx in query_global:
            y = int(self.ds[idx]["y"])
            if y in epi:
                epi[y][1].append(idx)

        return epi
