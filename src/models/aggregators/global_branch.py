# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

from ..layers.resnet_mlp import ResNetMLP
from .utils import _softmax


def _global_dispersion_scores(H_g: torch.Tensor) -> torch.Tensor:
    """
    Scoring s_i for the spatial aggregation/dispersion branch:
      D_ij = ||h_i - h_j||_2
      Let col-mean = mean_i D_ij
      s_i = mean_j (D_ij - col-mean_j)^2
    """
    n = H_g.size(0)
    if n <= 1:
        return torch.ones((n,), device=H_g.device, dtype=H_g.dtype)

    # [n, n] pairwise Euclidean distances
    D = torch.cdist(H_g, H_g, p=2.0)  # Numerically stable and diag=0
    col_mean = D.mean(dim=0, keepdim=True)           # [1,n]
    s = ((D - col_mean) ** 2).mean(dim=1)            # [n]
    return s


class GlobalBranch(nn.Module):
    """
    Global spatial aggregation branch (Global):
      z_i = ResNetMLP(h_i) -> 128
      s_i = aggregation/dispersion score (see formula above)
      w_i = softmax(s_i / τ_glob)
      f_glob = Σ_i w_i z_i
    """
    def __init__(self,
                 in_dim: int = 64,
                 embed_dim: int = 128,
                 hidden: int = 256,
                 num_blocks: int = 3,
                 tau: float = 0.25,
                 norm: str = "layer",
                 dropout: float = 0.1):
        super().__init__()
        self.tau = float(tau)
        self.proj = ResNetMLP(
            in_dim=in_dim, hidden=hidden, out_dim=embed_dim,
            num_blocks=num_blocks, norm=norm, dropout=dropout, last_norm=True
        )

    def _one_graph(self, H_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(H_g)                            # [n,128]
        s = _global_dispersion_scores(H_g)            # [n]
        w = _softmax(s, self.tau)
        f = (w.unsqueeze(-1) * z).sum(dim=0)
        return f, w

    def forward(self, H: torch.Tensor, batch_ptr: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        outs, ws = [], []
        B = int(batch_ptr.numel() - 1)
        for i in range(B):
            s, e = int(batch_ptr[i].item()), int(batch_ptr[i + 1].item())
            f_g, w_g = self._one_graph(H[s:e])
            outs.append(f_g)
            ws.append(w_g)
        return torch.stack(outs, dim=0), ws
