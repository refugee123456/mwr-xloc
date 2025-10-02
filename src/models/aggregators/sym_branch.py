# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

from ..layers.resnet_mlp import ResNetMLP
from .utils import _softmax, _mean_pairwise_l2


def _slice_pairs_for_graph(pairs: torch.Tensor, s: int, e: int) -> torch.Tensor:
    """Extract pairs that fall within [s, e) and convert to local indices."""
    if pairs is None or pairs.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=pairs.device if pairs is not None else "cpu")
    m = (pairs[:, 0] >= s) & (pairs[:, 0] < e) & (pairs[:, 1] >= s) & (pairs[:, 1] < e)
    pg = pairs[m] - s
    return pg


class SymBranch(nn.Module):
    """
    Left-right symmetry branch (Sym):
      δ_i = |h_i - h_{π(i)}|  (set to 0 if no paired counterpart)
      z_i = ResNetMLP(δ_i)  -> 128
      s_i = mean_j ||z_i - z_j||_2
      w_i = softmax(s_i / τ_sym)
      f_sym = Σ_i w_i z_i
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
        # δ_i has the same dimension as h_i; project directly to 128 via ResNetMLP
        self.proj = ResNetMLP(
            in_dim=in_dim, hidden=hidden, out_dim=embed_dim,
            num_blocks=num_blocks, norm=norm, dropout=dropout, last_norm=True
        )

    def _one_graph(self, H_g: torch.Tensor, pairs_g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, d = H_g.shape
        delta = torch.zeros((n, d), device=H_g.device, dtype=H_g.dtype)
        if pairs_g.numel() > 0:
            # For each mirror pair (l, r), assign |h_l - h_r| to both sides
            li, ri = pairs_g[:, 0].long(), pairs_g[:, 1].long()
            diff = torch.abs(H_g.index_select(0, li) - H_g.index_select(0, ri))   # [m, d]
            delta.index_copy_(0, li, diff)
            delta.index_copy_(0, ri, diff)

        z = self.proj(delta)                          # [n, 128]
        s = _mean_pairwise_l2(z)
        w = _softmax(s, self.tau)
        f = (w.unsqueeze(-1) * z).sum(dim=0)
        return f, w

    def forward(self,
                H: torch.Tensor,
                batch_ptr: torch.Tensor,
                pairs: torch.Tensor | None = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args
        ----
        H: [N, 64]
        batch_ptr: [B+1]
        pairs: [M,2] (global indices); if missing, treated as no symmetry info
        """
        B = int(batch_ptr.numel() - 1)
        outs, ws = [], []
        device = H.device
        if pairs is None:
            pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        for i in range(B):
            s, e = int(batch_ptr[i].item()), int(batch_ptr[i + 1].item())
            pg = _slice_pairs_for_graph(pairs, s, e)
            f_g, w_g = self._one_graph(H[s:e], pg)
            outs.append(f_g)
            ws.append(w_g)

        return torch.stack(outs, dim=0), ws
