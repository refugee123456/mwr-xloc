# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os

def _np_load_fast(path):
    return np.load(path, mmap_mode="r")  # read-only memory mapping


class MWRGraphDataset:
    """
    Read Stage-1 outputs (manifest.json + per-sample folder {nodes.npy, edge_index.npy, pairs.npy, meta.json})

    New capabilities:
    - Auto split into train/val/test by ratios and write splits.json for reproducibility
      * Triggered when: the directory has neither splits.json nor split_{split}.txt
      * Config: split_ratios=(0.7,0.15,0.15), stratify=True, seed=42, include_unknown_in_split=True
    - Stratified split: only stratify over labeled samples with y in {0,1}; unknown label (-1) can be
      randomly assigned across splits by ratio or excluded entirely

    Compatible behavior:
    - If splits.json or split_{split}.txt exists, those files fully take precedence
    - keep_unknown: whether to keep samples with y==-1 in the final indices of the current split
      (filtering is applied at the final index stage)

    Args:
    - split: "train" / "val" / "test"
    - keep_unknown: whether to keep y==-1 samples (default True)
    - random_resample: optional class-balanced upsampling on the training set (index-level repetition only)
    - augment: reserved
    - seed: random seed (used for auto split and upsampling)
    - split_ratios: (train, val, test) tuple or None (None means do not auto-split)
    - stratify: whether to stratify by y=0/1 (only for labeled samples)
    - include_unknown_in_split: during auto split, whether to randomly distribute y==-1 samples into the three splits
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        keep_unknown: bool = True,
        random_resample: bool = False,
        augment: bool = False,
        seed: int = 42,
        split_ratios: Optional[Tuple[float, float, float]] = None,
        stratify: bool = True,
        include_unknown_in_split: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.keep_unknown = keep_unknown
        self.random_resample = random_resample and (split == "train")
        self.augment = augment
        self.seed = int(seed)
        self.split_ratios = split_ratios
        self.stratify = bool(stratify)
        self.include_unknown_in_split = bool(include_unknown_in_split)

        mf = self.root / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError(f"manifest.json not found at {mf}")
        with mf.open("r", encoding="utf-8") as f:
            self._manifest: List[Dict[str, Any]] = json.load(f)

        # If there are no split files and ratios are provided, auto-generate splits.json
        self._auto_make_splits_if_needed()

        # Assemble available indices
        base_idx = list(range(len(self._manifest)))
        idx = self._maybe_load_split_indices(base_idx)

        if not self.keep_unknown:
            idx = [i for i in idx if int(self._manifest[i].get("y", -1)) in (0, 1)]

        # Optional: class-balanced upsampling on the training split (index-only)
        if self.random_resample and len(idx) > 0:
            idx = self._upsample_balanced(idx, seed=self.seed)

        self._indices = idx

    # ---------- auto-generate splits.json if needed ----------
    def _auto_make_splits_if_needed(self):
        sp_json = self.root / "splits.json"
        sp_train = self.root / "split_train.txt"
        sp_val = self.root / "split_val.txt"
        sp_test = self.root / "split_test.txt"

        # If any split file exists, do nothing
        if sp_json.exists() or sp_train.exists() or sp_val.exists() or sp_test.exists():
            return

        # If auto split not requested, do nothing
        if self.split_ratios is None:
            return

        r = self.split_ratios
        if not (isinstance(r, (list, tuple)) and len(r) == 3):
            raise ValueError("split_ratios must be a 3-tuple/list, e.g., (0.7, 0.15, 0.15)")

        r = tuple(float(x) for x in r)
        if any(x < 0 for x in r):
            raise ValueError("split_ratios must be non-negative")
        s = sum(r)
        if s == 0:
            raise ValueError("split_ratios sum must be > 0")
        # Normalize to allow inputs like 7, 1.5, 1.5
        ratios = (r[0] / s, r[1] / s, r[2] / s)

        # Prepare ids and labels
        ids = []
        y_list = []
        for pos, item in enumerate(self._manifest):
            rid = item.get("id", None)
            if rid is None:
                # If manifest has no id, use positional index as id
                rid = pos
            try:
                rid = int(rid)
            except Exception:
                rid = pos
            ids.append(rid)
            y_list.append(int(item.get("y", -1)))

        ids = np.asarray(ids)
        y_arr = np.asarray(y_list)

        rs = np.random.RandomState(self.seed)

        def _split_index(arr_idx: np.ndarray, ratios: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            n = arr_idx.size
            if n == 0:
                return arr_idx, arr_idx, arr_idx
            perm = arr_idx[rs.permutation(n)]
            n_tr = int(np.floor(n * ratios[0]))
            n_va = int(np.floor(n * ratios[1]))
            # Remainder goes to test to avoid losing items and to keep union = all
            n_te = n - n_tr - n_va
            tr = perm[:n_tr]
            va = perm[n_tr:n_tr+n_va]
            te = perm[n_tr+n_va:]
            return tr, va, te

        # Stratify: only over y ∈ {0,1}; unknown class distributed by include_unknown_in_split
        idx_pos = np.where(y_arr == 1)[0]
        idx_neg = np.where(y_arr == 0)[0]
        idx_unk = np.where(y_arr == -1)[0]

        if self.stratify:
            tr_p, va_p, te_p = _split_index(idx_pos, ratios)
            tr_n, va_n, te_n = _split_index(idx_neg, ratios)

            tr = np.concatenate([tr_p, tr_n], axis=0)
            va = np.concatenate([va_p, va_n], axis=0)
            te = np.concatenate([te_p, te_n], axis=0)

            if self.include_unknown_in_split and idx_unk.size > 0:
                tr_u, va_u, te_u = _split_index(idx_unk, ratios)
                tr = np.concatenate([tr, tr_u], axis=0)
                va = np.concatenate([va, va_u], axis=0)
                te = np.concatenate([te, te_u], axis=0)
        else:
            all_idx = np.arange(len(self._manifest))
            if not self.include_unknown_in_split:
                all_idx = np.where(y_arr != -1)[0]
            tr, va, te = _split_index(all_idx, ratios)

        # Shuffle within each split (keep counts unchanged)
        def _shuffle(a: np.ndarray):
            if a.size > 0:
                a = a.copy()
                rs.shuffle(a)
            return a
        tr = _shuffle(tr); va = _shuffle(va); te = _shuffle(te)

        # Convert positional indices to the original ids stored in manifest (persist ids)
        pos2id = [None] * len(self._manifest)
        for pos, item in enumerate(self._manifest):
            rid = item.get("id", None)
            if rid is None:
                rid = pos
            try:
                rid = int(rid)
            except Exception:
                rid = pos
            pos2id[pos] = rid

        splits_payload = dict(
            train=[int(pos2id[i]) for i in tr.tolist()],
            val=[int(pos2id[i]) for i in va.tolist()],
            test=[int(pos2id[i]) for i in te.tolist()],
            meta=dict(
                ratios=ratios,
                stratify=self.stratify,
                include_unknown=self.include_unknown_in_split,
                seed=self.seed,
                generated_by="MWRGraphDataset._auto_make_splits_if_needed",
            ),
        )
        with (self.root / "splits.json").open("w", encoding="utf-8") as f:
            json.dump(splits_payload, f, ensure_ascii=False, indent=2)

    # ---------- load split indices if present ----------
    def _maybe_load_split_indices(self, base_idx: List[int]) -> List[int]:
        # Try splits.json
        sp_json = self.root / "splits.json"
        if sp_json.exists():
            try:
                with sp_json.open("r", encoding="utf-8") as f:
                    sp = json.load(f)
                want_ids = set(sp.get(self.split, []))
                if len(want_ids) > 0:
                    # Map manifest raw id to positional index
                    id2pos = {}
                    for pos, item in enumerate(self._manifest):
                        rid = item.get("id", None)
                        if rid is not None:
                            try:
                                id2pos[int(rid)] = pos
                            except Exception:
                                pass
                        else:
                            id2pos[pos] = pos
                    idx = [id2pos[i] for i in want_ids if i in id2pos]
                    if len(idx) > 0:
                        return idx
            except Exception:
                pass

        # Try split_{split}.txt
        sp_txt = self.root / f"split_{self.split}.txt"
        if sp_txt.exists():
            try:
                ids = []
                with sp_txt.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ids.append(int(line))
                        except Exception:
                            pass
                if len(ids) > 0:
                    id2pos = {}
                    for pos, item in enumerate(self._manifest):
                        rid = item.get("id", None)
                        if rid is not None:
                            try:
                                id2pos[int(rid)] = pos
                            except Exception:
                                pass
                        else:
                            id2pos[pos] = pos
                    idx = [id2pos[i] for i in ids if i in id2pos]
                    if len(idx) > 0:
                        return idx
            except Exception:
                pass

        # Default: if no split files, return all (same for all splits)
        return base_idx

    # ---------- class-balanced upsampling for training ----------
    def _upsample_balanced(self, src_idx: List[int], seed: int) -> List[int]:
        pos = [i for i in src_idx if int(self._manifest[i].get("y", -1)) == 1]
        neg = [i for i in src_idx if int(self._manifest[i].get("y", -1)) == 0]
        others = [i for i in src_idx if int(self._manifest[i].get("y", -1)) not in (0, 1)]
        if len(pos) == 0 or len(neg) == 0:
            # Single-class or missing-class: skip balancing
            return src_idx

        rs = np.random.RandomState(seed)
        if len(pos) > len(neg):
            k = len(pos) - len(neg)
            add = rs.choice(neg, size=k, replace=True).tolist()
            neg = neg + add
        elif len(neg) > len(pos):
            k = len(neg) - len(pos)
            add = rs.choice(pos, size=k, replace=True).tolist()
            pos = pos + add

        merged = pos + neg + others
        rs.shuffle(merged)
        return merged

    # ---------- standard Dataset-like interface ----------
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        mi = self._indices[i]
        rec = self._manifest[mi]
        p = Path(rec["path"])

        nodes = _np_load_fast(p / "nodes.npy")
        edge_index = _np_load_fast(p / "edge_index.npy")
        pairs = _np_load_fast(p / "pairs.npy")
        meta_path = p / "meta.json"
        meta = {}
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        return dict(
            id=rec.get("id", mi),
            y=int(rec.get("y", -1)),
            nodes=nodes,
            edge_index=edge_index,
            pairs=pairs,
            meta=meta,
            path=str(p),
            split=self.split,
        )

    # ---------- convenience constructor: auto-discover processed cache root ----------
    @classmethod
    def from_cache(
        cls,
        root: str | Path = "",
        split: str = "train",
        keep_unknown: bool = False,
        random_resample: bool = False,
        augment: bool = False,
        seed: int = 42,
        split_ratios: Optional[Tuple[float, float, float]] = None,
        stratify: bool = True,
        include_unknown_in_split: bool = True,
    ) -> "MWRGraphDataset":
        """
        Priority order to locate the dataset:
          1) Explicit root argument
          2) Env var MWR_DATASET_OUT
          3) Env vars MWR_BREAST_OUT / MWR_LEG_OUT / MWR_LUNG_OUT
          4) Common default paths data/processed/{breast,leg,lung}
        If any candidate contains manifest.json, return a Dataset from that directory.
        If the directory has no split files and split_ratios is provided, auto-generate splits.json.
        """
        candidates: list[Path] = []
        def _add(p):
            if p:
                p = Path(p)
                if p not in candidates:
                    candidates.append(p)

        # 1) explicit root
        _add(root)

        # 2) general env var
        _add(os.environ.get("MWR_DATASET_OUT", ""))

        # 3) per-domain env vars
        for key in ("MWR_BREAST_OUT", "MWR_LEG_OUT", "MWR_LUNG_OUT"):
            _add(os.environ.get(key, ""))

        # 4) common defaults (ordered breast→leg→lung)
        for d in ("data/processed/breast", "data/processed/leg", "data/processed/lung"):
            _add(d)

        chosen: Path | None = None
        for p in candidates:
            if p and (p / "manifest.json").exists():
                chosen = p
                break

        if chosen is None:
            msg = (
                "Cannot locate processed dataset (manifest.json).\n"
                "Tried candidates:\n  - " + "\n  - ".join(str(c) for c in candidates)
                + "\nPlease set data_root in YAML, or export MWR_DATASET_OUT."
            )
            raise FileNotFoundError(msg)

        return cls(
            root=chosen,
            split=split,
            keep_unknown=keep_unknown,
            random_resample=random_resample and (split == "train"),
            augment=augment,
            seed=seed,
            split_ratios=split_ratios,
            stratify=stratify,
            include_unknown_in_split=include_unknown_in_split,
        )
