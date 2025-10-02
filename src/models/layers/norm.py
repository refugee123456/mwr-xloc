# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn


def get_norm(norm: str | None, dim: int) -> nn.Module:
    """
    Lightweight wrapper: return a normalization layer by string key.
      - "layer" / "ln" / None -> LayerNorm / Identity
      - "batch" / "bn"        -> BatchNorm1d
    """
    if norm is None or str(norm).lower() in ("none", ""):
        return nn.Identity()
    key = str(norm).lower()
    if key in ("layer", "ln", "layernorm"):
        return nn.LayerNorm(dim)
    if key in ("batch", "bn", "batchnorm"):
        # Note: assumes input shape is [*, D]; BatchNorm1d expects [B, D]
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unknown norm: {norm!r}")


def init_linear(m: nn.Module):
    """Linear layer initialization: Xavier + zero bias."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NormActDrop(nn.Module):
    """(Norm -> Act -> Dropout) composite block."""
    def __init__(self, dim: int, norm: str = "layer", act: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.norm = get_norm(norm, dim)
        if act is None or act == "identity":
            self.act = nn.Identity()
        elif act.lower() == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown act: {act}")
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(x)))
