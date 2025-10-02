# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn


# ---- Low-dimensional (1D) version that can be compiled by torchscript ----
@torch.jit.script
def _segment_softmax_1d(logits: torch.Tensor,
                        segment_ids: torch.Tensor,
                        num_segments: int) -> torch.Tensor:
    """
    Perform per-segment softmax on 1D logits, return [E].
    - logits: [E]
    - segment_ids: [E], each value ∈ [0, num_segments-1]; here we use dst as the segment.
    """
    assert logits.dim() == 1 and segment_ids.dim() == 1 and logits.numel() == segment_ids.numel()
    device = logits.device
    dtype = logits.dtype

    # Numerically stable: per-segment maximum
    base = torch.full((num_segments,), float("-inf"), device=device, dtype=dtype)
    max_per = torch.scatter_reduce(base, 0, segment_ids, logits, reduce="amax", include_self=True)

    stable = logits - max_per.index_select(0, segment_ids)
    exp = torch.exp(stable)

    sum_per = torch.zeros(num_segments, device=device, dtype=dtype)
    sum_per.scatter_add_(0, segment_ids, exp)

    return exp / (sum_per.index_select(0, segment_ids) + 1e-12)


def segment_softmax(logits: torch.Tensor,
                    segment_ids: torch.Tensor,
                    num_segments: int) -> torch.Tensor:
    """
    General segmented softmax:
      - If logits.shape == [E], call the torchscript version (faster)
      - If logits.shape == [E, H], apply softmax column-wise and return [E, H]
    """
    if logits.dim() == 1:
        return _segment_softmax_1d(logits, segment_ids, num_segments)
    if logits.dim() == 2:
        E, H = logits.shape
        outs = []
        for h in range(H):
            outs.append(_segment_softmax_1d(logits[:, h].contiguous(), segment_ids, num_segments))
        return torch.stack(outs, dim=1)
    raise ValueError(f"segment_softmax expects [E] or [E,H], got {tuple(logits.shape)}")


class QKVGATLayer(nn.Module):
    """
    Multi-head graph attention with Q/K/V (score per edge, softmax over dst).

    Parameters
    ----------
    in_dim      : Input dimension F
    out_dim     : **Total output dimension C_out** (not per-head)
    num_heads   : Number of heads H
    residual    : Residual connection (automatically aligns to C_out)
    use_ln      : Whether to use LayerNorm(C_out)
    dropout     : Dropout probability (applied to output)
    return_alpha: Whether to return per-edge attention (shape=[E, H])
    add_self_loops: Whether to add self-loops (default False)

    Shapes
    ------
    Input  nodes: [N, F], edge_index: [2, E] (src, dst)
    Output H: [N, C_out], alpha (optional): [E, H]
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int = 4,
                 residual: bool = True,
                 use_ln: bool = True,
                 dropout: float = 0.0,
                 return_alpha: bool = False,
                 add_self_loops: bool = False):
        super().__init__()
        assert in_dim > 0 and out_dim > 0 and num_heads >= 1

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        # Per-head dim: ceil; finally project back to out_dim with W_out
        self.head_dim = math.ceil(out_dim / num_heads)
        self.inner_dim = self.head_dim * num_heads  # concatenated dim >= out_dim

        # Linear projections
        self.W_Q = nn.Linear(in_dim, self.inner_dim, bias=False)
        self.W_K = nn.Linear(in_dim, self.inner_dim, bias=False)
        self.W_V = nn.Linear(in_dim, self.inner_dim, bias=False)

        # Output projection: H*D -> out_dim (Identity if exactly equal)
        self.W_out = nn.Identity() if self.inner_dim == out_dim else nn.Linear(self.inner_dim, out_dim, bias=True)

        # Residual / normalization / dropout
        self.residual = residual
        self.W_res = nn.Linear(in_dim, out_dim, bias=True) if residual else nn.Identity()
        self.ln = nn.LayerNorm(out_dim) if use_ln else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.return_alpha = return_alpha
        self.add_self_loops = add_self_loops

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [N, H*D] -> [N, H, D]
        N = x.size(0)
        # Ensure divisibility for view; if not divisible (ceil), Linear aligns inner_dim
        return x.view(N, self.num_heads, self.head_dim)

    @torch.no_grad()
    def _maybe_add_self_loops(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if not self.add_self_loops:
            return edge_index
        device = edge_index.device
        self_loops = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
        self_loops = torch.stack([self_loops, self_loops], dim=0)  # [2, N]
        ei = torch.cat([edge_index, self_loops], dim=1)
        # Deduplicate
        ei = torch.unique(ei.t(), dim=0).t().contiguous()
        return ei

    def forward(self,
                nodes: torch.Tensor,         # [N, F]
                edge_index: torch.Tensor,    # [2, E]
                num_nodes: Optional[int] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert nodes.dim() == 2 and edge_index.dim() == 2 and edge_index.size(0) == 2
        N = nodes.size(0) if num_nodes is None else int(num_nodes)

        edge_index = self._maybe_add_self_loops(edge_index, N)

        # Linear maps -> split heads
        Q = self._split_heads(self.W_Q(nodes))  # [N, H, D]
        K = self._split_heads(self.W_K(nodes))  # [N, H, D]
        V = self._split_heads(self.W_V(nodes))  # [N, H, D]

        src, dst = edge_index[0].long(), edge_index[1].long()  # [E]
        q = Q.index_select(0, dst)  # [E, H, D]
        k = K.index_select(0, src)  # [E, H, D]
        v = V.index_select(0, src)  # [E, H, D]

        # Scaled dot-product scores
        score = (q * k).sum(dim=-1) * self.scale        # [E, H]

        # Segment softmax over dst (per head)
        alpha = segment_softmax(score, dst, num_segments=N)  # [E, H]

        # Aggregate to dst
        out = torch.zeros((N, self.num_heads, self.head_dim),
                          device=nodes.device, dtype=nodes.dtype)    # [N, H, D]
        out.index_add_(0, dst, alpha.unsqueeze(-1) * v)              # weighted sum

        # Merge heads + project to out_dim
        H_cat = out.reshape(N, self.inner_dim)                       # [N, H*D]
        H_cat = self.W_out(H_cat)                                    # [N, C_out]

        # Residual + LN + Dropout (Pre-LN style for more stable training)
        if isinstance(self.W_res, nn.Identity) and self.out_dim == self.in_dim:
            H_out = H_cat + nodes
        else:
            H_out = H_cat + self.W_res(nodes)
        H_out = self.ln(H_out)
        H_out = self.dropout(H_out)

        return H_out, (alpha if self.return_alpha else None)
