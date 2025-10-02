# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from .norm import NormActDrop, init_linear


class _MLPBlock(nn.Module):
    """Linear -> NormActDrop"""
    def __init__(self, in_dim: int, out_dim: int, norm: str = "layer", act: str = "relu", dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.nad = NormActDrop(out_dim, norm=norm, act=act, dropout=dropout)
        self.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nad(self.fc(x))


class _ResBlock(nn.Module):
    """Residual block with two MLP layers (keeps channel size)."""
    def __init__(self, dim: int, hidden: int | None = None, norm: str = "layer", act: str = "relu", dropout: float = 0.0):
        super().__init__()
        h = hidden or dim
        self.fc1 = nn.Linear(dim, h)
        self.nad1 = NormActDrop(h, norm=norm, act=act, dropout=dropout)
        self.fc2 = nn.Linear(h, dim)
        self.nad2 = NormActDrop(dim, norm=norm, act=act, dropout=dropout)
        self.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.nad1(self.fc1(x))
        y = self.nad2(self.fc2(y))
        return x + y


class ResNetMLP(nn.Module):
    """
    Small residual MLP:
      in_dim -> linear expansion -> K residual blocks -> project to out_dim (default 128)
    Recommended defaults: in_dim=64, hidden=256, num_blocks=3, out_dim=128
    """
    def __init__(
        self,
        in_dim: int = 64,
        hidden: int = 256,
        out_dim: int = 128,          # <- Final output is 128-D
        num_blocks: int = 3,
        norm: str = "layer",
        act: str = "relu",
        dropout: float = 0.1,
        last_norm: bool = True,
    ):
        super().__init__()
        self.stem = _MLPBlock(in_dim, hidden, norm=norm, act=act, dropout=dropout)
        self.blocks = nn.Sequential(*[
            _ResBlock(hidden, hidden=hidden, norm=norm, act=act, dropout=dropout)
            for _ in range(max(0, num_blocks))
        ])
        self.head = nn.Linear(hidden, out_dim)
        self.tail_norm = NormActDrop(out_dim, norm=norm, act=act, dropout=dropout) if last_norm else nn.Identity()
        self.apply(init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.head(h)
        h = self.tail_norm(h)
        return h
