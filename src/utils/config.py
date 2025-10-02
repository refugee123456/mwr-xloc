# -*- coding: utf-8 -*-
"""
Utils for loading YAML configs and merging CLI overrides.

Usage
-----
cfg = load_config("configs/pretrain.yaml", cli_args=sys.argv[1:])
log = get_logger(__name__)  # from utils.logging
log.info(f"Config:\n{pretty(cfg)}")
"""

from __future__ import annotations
import argparse
import copy
import datetime as dt
import json
import os
import pathlib
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_by_path(d: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set nested dict value by 'a.b.c' path (create sub-dicts if absent)."""
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_override(s: str) -> Tuple[str, Any]:
    """
    Parse 'a.b=1', 'train.lr=1e-3', 'name=abc', 'flag=true' into (key, value).
    Numbers will be cast; 'true/false' -> bool; 'null' -> None; JSON arrays ok.
    """
    if "=" not in s:
        raise ValueError(f"Override '{s}' must be in key=value form.")
    k, v = s.split("=", 1)
    v = v.strip()
    # Try JSON first (supports numbers, booleans, arrays)
    try:
        parsed = json.loads(v)
        return k, parsed
    except Exception:
        pass
    # Fallback: string; but map common literals
    low = v.lower()
    if low in {"true", "false"}:
        return k, low == "true"
    if low in {"none", "null"}:
        return k, None
    # Try to cast numeric (int/float)
    try:
        if any(ch in v for ch in ".eE"):
            return k, float(v)
        return k, int(v)
    except Exception:
        return k, v  # keep as string


def parse_cli_overrides(cli_args: Iterable[str]) -> Dict[str, Any]:
    """
    Accept arguments like:
        --set train.batch_size=16 model.hidden=256 tag='exp1'
    or a flat list without '--set' when you embed this yourself.
    """
    # If user passes '--set ...', parse only after --set
    if "--set" in cli_args:
        idx = list(cli_args).index("--set")
        overrides = list(cli_args)[idx + 1 :]
    else:
        # assume the entire list is overrides (safe for simple scripts)
        overrides = [a for a in cli_args if "=" in a and not a.startswith("--")]
    out: Dict[str, Any] = {}
    for item in overrides:
        k, v = _parse_override(item)
        _set_by_path(out, k, v)
    return out


def deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively update nested dictionaries (like OmegaConf merges)."""
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_config(
    yaml_path: str | os.PathLike,
    cli_args: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Load YAML, merge CLI overrides (dot-path), and append runtime meta fields.
    """
    cfg = load_yaml(yaml_path)

    if cli_args:
        overrides = parse_cli_overrides(cli_args)
        deep_update(cfg, overrides)

    if extra:
        deep_update(cfg, extra)

    # Runtime meta
    now = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.setdefault("_meta", {})
    cfg["_meta"]["timestamp"] = now
    cfg["_meta"]["cwd"] = str(pathlib.Path.cwd())
    return cfg


def pretty(cfg: Mapping[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120)


def ensure_dir(path: str | os.PathLike) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)
