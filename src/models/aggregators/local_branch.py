# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

from ..layers.resnet_mlp import ResNetMLP
from .utils import _softmax, _mean_pairwise_l2, _seg_loop


class LocalBranch(nn.Module):
    """
    Local metabolism/hotspot branch (Local):
      z_i = ResNetMLP(h_i) -> 128
      s_i = mean_j ||z_i - z_j||_2
      w_i = softmax(s_i / τ_loc)
      f_loc = Σ_i w_i z_i     (graph-level 128-D)
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
        z = self.proj(H_g)                          # [n, 128]
        s = _mean_pairwise_l2(z)                    # [n]
        w = _softmax(s, self.tau)                   # [n]
        f = (w.unsqueeze(-1) * z).sum(dim=0)        # [128]
        return f, w

    def forward(self, H: torch.Tensor, batch_ptr: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args
        ----
        H: [N, 64] node features
        batch_ptr: [B+1] start/end indices per graph
        Returns
        -------
        v_loc: [B, 128]   ws: List[Tensor[n_g]]
        """
        return _seg_loop(H, batch_ptr, self._one_graph)
