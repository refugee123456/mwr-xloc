# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Allow: python src/scripts/run_preprocess.py --config ...
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.processing.registry import get_processor


def _load_table(path: str, sheet: str | int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"raw file not found: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p, sheet_name=sheet)
    return pd.read_csv(p)


def _build_processor(name: str, args: Dict[str, Any]) -> Any:
    """Compatible with get_processor returning a class or an already constructed object."""
    obj = get_processor(name)
    # If it's a class, instantiate; if it's an instance, use it directly
    if isinstance(obj, type):
        return obj(**(args or {}))
    # If it's an instance and you still want to set extra params? We don't auto-setattr here to keep it simple and consistent
    return obj


def main() -> None:
    import argparse, yaml

    ap = argparse.ArgumentParser("MWR preprocess runner (multi-dataset)")
    ap.add_argument("--config", type=str, required=True,
                   help="YAML path for a single dataset (breast/lung/leg)")
    ap.add_argument("--limit", type=int, default=None,
                   help="only process first N rows (debug)")
    args = ap.parse_args()

    # Load YAML
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Convention for config structure
    # dataset:
    #   name: breast|lung|leg
    #   csv:  path/to/raw.(csv|xlsx)
    #   sheet: (optional, for xlsx)
    #   id_col: (optional)
    #   processor_args: {...} (optional, passed to processor)
    # output_dir:  data/processed/<name>
    ds_name   = cfg["dataset"]["name"].strip().lower()
    raw_path  = cfg["dataset"]["csv"]
    xlsx_sheet= cfg["dataset"].get("sheet", None)
    id_col    = cfg["dataset"].get("id_col", None)
    proc_args = cfg["dataset"].get("processor_args", {}) or {}
    out_dir   = Path(cfg["output_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read table
    df = _load_table(raw_path, sheet=xlsx_sheet)

    # Build processor
    proc = _build_processor(ds_name, proc_args)

    # Fit
    proc.fit(df)

    # Process row by row
    manifest = []
    rows = df.iterrows()
    if args.limit is not None and args.limit > 0:
        rows = list(rows)[: args.limit]

    ok_cnt = 0
    for idx, row in rows:
        try:
            out = proc.process_row(row)
        except Exception:
            # breast: skip directly if temperature is missing; leg/lung: imputation is handled internally, exceptions are unlikely
            continue

        sample_id = row[id_col] if (id_col and id_col in row) else idx
        sample_dir = out_dir / f"{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.save(sample_dir / "nodes.npy", out.nodes)
        np.save(sample_dir / "edge_index.npy", out.edge_index)
        np.save(sample_dir / "pairs.npy", out.pairs)

        manifest.append({
            "id": int(sample_id) if str(sample_id).isdigit() else str(sample_id),
            "path": str(sample_dir),
            "n_nodes": int(out.nodes.shape[0]),
            **out.meta
        })
        ok_cnt += 1

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] {ds_name}: preprocessed {ok_cnt} / {len(df)} samples → {str(out_dir)}")


if __name__ == "__main__":
    main()
