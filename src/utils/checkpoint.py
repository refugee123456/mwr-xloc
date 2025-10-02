# -*- coding: utf-8 -*-
"""
Light-weight checkpoint helpers (unified & backward-compatible).

Preferred API
-------------
save_checkpoint(path, state)
load_checkpoint(path, map_location='cpu') -> dict
load_into(model, ckpt, key='model', strict=True)

Also supports legacy kwargs (model/optimizer/epoch/extra) and will be packed
into a state dict automatically.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional, Union

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


PathLike = Union[str, "os.PathLike[str]"]


def _to_cpu_state_dict_obj(obj: Any) -> Any:
    """Move tensors to cpu and detach if possible; keep non-tensors as is."""
    if not _HAS_TORCH:
        return obj
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            return obj.detach().cpu()
        except Exception:
            return obj
    return obj


def _to_cpu_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_cpu_state_dict_obj(v) for k, v in state_dict.items()}


def _pack_state(
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    epoch: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Pack common fields into a checkpoint state dict."""
    payload: Dict[str, Any] = {}
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if model is not None:
        if hasattr(model, "state_dict"):
            sd = model.state_dict()
        else:
            # allow plain dict-like
            sd = dict(model)
        payload["model"] = _to_cpu_state_dict(sd)
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    return payload


def save_checkpoint(
    path: PathLike,
    state: Optional[Dict[str, Any]] = None,
    *,
    # legacy kwargs (optional): will be packed into `state` if provided
    model: Any | None = None,
    optimizer: Any | None = None,
    epoch: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> None:
    """
    Save checkpoint to `path`.

    Preferred:
        save_checkpoint(path, state)

    Backward compatible:
        save_checkpoint(path, model=..., optimizer=..., epoch=..., extra=...)

    Notes:
        - All tensors in `state` are kept as-is; if you pass model/optimizer via
          kwargs, tensors are moved to CPU for portability.
    """
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)

    if state is None:
        # fall back to legacy packing
        state = _pack_state(model=model, optimizer=optimizer, epoch=epoch, extra=extra)
        if not state:
            # nothing to save -> still write an empty payload to be explicit
            state = {}

    if _HAS_TORCH:
        torch.save(state, path)
    else:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(state, f)


def load_checkpoint(path: PathLike, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    """Load checkpoint dict from `path`."""
    if _HAS_TORCH:
        return torch.load(path, map_location=map_location)
    else:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def load_into(model: Any, ckpt: Dict[str, Any], *, key: str = "model", strict: bool = True) -> None:
    """
    Load `ckpt[key]` into `model`:
      - If the model implements .load_state_dict, we call it.
      - Otherwise we try to set dict items.
    """
    state = ckpt.get(key, {})
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(state, strict=strict)
    else:
        for k, v in state.items():
            try:
                model[k] = v
            except Exception:
                # ignore keys we cannot set on a plain container
                pass
