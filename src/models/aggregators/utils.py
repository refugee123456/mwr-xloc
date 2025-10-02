# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Tuple, List
import torch


def _softmax(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """Numerically stable temperature softmax (along the last dimension)."""
    t = max(1e-6, float(tau))
    return torch.softmax(scores / t, dim=-1)


def _mean_pairwise_l2(z: torch.Tensor) -> torch.Tensor:
    """
    z:[n,d] → s:[n], the average L2 distance of each node to all other nodes; O(n^2) but n is usually small.
    """
    n = z.size(0)
    if n <= 1:
        return torch.ones((n,), device=z.device, dtype=z.dtype)
    sq = (z * z).sum(-1, keepdim=True)           # [n,1]
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dist2 = torch.clamp(sq + sq.t() - 2.0 * (z @ z.t()), min=0.0)  # [n,n]
    dist = torch.sqrt(dist2 + 1e-9)
    dist = dist - torch.diag_embed(torch.diagonal(dist))
    return dist.sum(dim=1) / (n - 1)


def _seg_loop(H: torch.Tensor, batch_ptr, one_graph_fn):
    """
    Generic segmented traversal: slice node features per graph using batch_ptr, feed into one_graph_fn, then stack.
    Compatible with batch_ptr being a numpy.ndarray or torch.Tensor.
    """
    # ---- Key fix: np.ndarray -> torch.LongTensor ----
    if not torch.is_tensor(batch_ptr):
        batch_ptr = torch.as_tensor(batch_ptr, device=H.device, dtype=torch.long)
    else:
        batch_ptr = batch_ptr.to(H.device).long()

    B = int(batch_ptr.numel() - 1)
    outs = []
    for b in range(B):
        s = int(batch_ptr[b].item()); e = int(batch_ptr[b + 1].item())
        Hb = H[s:e]
        outs.append(one_graph_fn(Hb))
    # The specific branch determines how to concatenate; return as-is here, handled by each branch.
    # Most branches expect to return (tensor, aux); if one_graph_fn returns only a tensor, fill None for aux.
    if isinstance(outs[0], tuple):
        xs, auxs = zip(*outs)
        return torch.stack(xs, 0), auxs
    else:
        return torch.stack(outs, 0), None
