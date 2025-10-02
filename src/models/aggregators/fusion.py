# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn

from ..layers.norm import NormActDrop, init_linear


class FusionHead(nn.Module):
    """
    Three-branch fusion (gated α/β/γ) → 128-D graph representation:
      logits = MLP([v_loc; v_sym; v_glob]) -> 3
      gate = softmax(logits)
      h_G  = α*v_loc + β*v_sym + γ*v_glob
    """
    def __init__(self,
                 embed_dim: int = 128,
                 gate_hidden: int = 64,
                 norm: str = "layer",
                 act: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        d_in = embed_dim * 3
        self.fc1 = nn.Linear(d_in, gate_hidden)
        self.na1 = NormActDrop(gate_hidden, norm=norm, act=act, dropout=dropout)
        self.fc2 = nn.Linear(gate_hidden, 3)
        self.out_ln = nn.LayerNorm(embed_dim)  # Extra stabilization on the output
        self.apply(init_linear)

    def forward(self,
                v_loc: torch.Tensor,   # [B,128]
                v_sym: torch.Tensor,   # [B,128]
                v_glob: torch.Tensor   # [B,128]
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = torch.cat([v_loc, v_sym, v_glob], dim=-1)         # [B, 384]
        logits = self.fc2(self.na1(self.fc1(x)))              # [B, 3]
        gate = torch.softmax(logits, dim=-1)                  # [B, 3]
        alpha, beta, gamma = gate[:, 0:1], gate[:, 1:2], gate[:, 2:3]

        h = alpha * v_loc + beta * v_sym + gamma * v_glob     # [B,128]
        h = self.out_ln(h)
        aux = {"alpha": alpha.squeeze(-1), "beta": beta.squeeze(-1), "gamma": gamma.squeeze(-1), "logits": logits}
        return h, aux
