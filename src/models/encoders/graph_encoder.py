# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.models.layers.qkv_gat import QKVGATLayer


def _to_tensor(x, device):
    if torch.is_tensor(x): return x.to(device)
    import numpy as np
    if isinstance(x, np.ndarray): return torch.from_numpy(x).to(device)
    if isinstance(x, (list, tuple)): return torch.tensor(x, device=device)
    return x


def _pool_mean(H: torch.Tensor, batch_ptr: torch.Tensor) -> torch.Tensor:
    """Graph-level mean pooling: H:[N,C], batch_ptr:[B+1], return [B,C]."""
    B = int(batch_ptr.numel() - 1)
    out = []
    for i in range(B):
        s, e = int(batch_ptr[i].item()), int(batch_ptr[i + 1].item())
        out.append(H[s:e].mean(dim=0, keepdim=False))
    return torch.stack(out, dim=0) if B > 0 else H.mean(dim=0, keepdim=True)


class GraphEncoder(nn.Module):
    """
    Two stacked QKV-GAT layers:
      - layer1: in_dim -> hid_dim (total dimension)
      - layer2: hid_dim -> out_dim (total dimension)
      - Inside each layer, multi-head splitting with num_heads is used; note out_dim/hid_dim
        are the **total** dimensions, not per-head dimensions.
    """
    def __init__(self,
                 in_dim: int = 13,
                 hid_dim: int = 64,      # total hidden dimension
                 out_dim: int = 64,      # total output dimension
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_ln: bool = True,
                 add_self_loops: bool = False):
        super().__init__()
        assert in_dim > 0 and hid_dim > 0 and out_dim > 0 and num_heads >= 1

        self.layer1 = QKVGATLayer(
            in_dim=in_dim,
            out_dim=hid_dim,        # <- total dimension
            num_heads=num_heads,
            residual=True,
            use_ln=use_ln,
            dropout=dropout,
            return_alpha=False,
            add_self_loops=add_self_loops,
        )
        self.layer2 = QKVGATLayer(
            in_dim=hid_dim,
            out_dim=out_dim,        # <- total dimension
            num_heads=num_heads,
            residual=True,
            use_ln=use_ln,
            dropout=dropout,
            return_alpha=False,
            add_self_loops=add_self_loops,
        )

    # —— Input unification —— #
    def _unify_inputs(self, batch: Any, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return: nodes[N,F], edge_index[2,E](long), batch_ptr[B+1](long)
        Supports dicts produced by collate_graphs; also compatible with {'x':[B,N,F], 'edge_index' or 'adj'}.
        """
        if not isinstance(batch, dict):
            raise TypeError("GraphEncoder forward expects a dict batch.")

        nodes = batch.get("nodes", batch.get("x"))
        if nodes is None:
            raise KeyError("Need 'nodes' (or 'x') in batch dict.")
        nodes = _to_tensor(nodes, device)

        edge_index = batch.get("edge_index")
        batch_ptr = batch.get("batch_ptr")

        if nodes.dim() == 3:  # [B,N,F] -> [BN,F]
            B, N, F = nodes.shape
            nodes = nodes.reshape(B * N, F)
            if edge_index is None and ("adj" in batch):
                adj = _to_tensor(batch["adj"], device).bool()  # [N,N]
                src, dst = torch.nonzero(adj, as_tuple=True)
                edge_index = torch.stack([src, dst], dim=0)
            if batch_ptr is None:
                batch_ptr = torch.arange(0, (B + 1) * N, N, device=device, dtype=torch.long)

        if edge_index is None and ("adj" in batch):
            adj = _to_tensor(batch["adj"], device).bool()
            src, dst = torch.nonzero(adj, as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0)

        if edge_index is None:
            raise KeyError("Need 'edge_index' or 'adj' in batch dict.")

        edge_index = _to_tensor(edge_index, device).long()

        if batch_ptr is None:
            batch_ptr = torch.tensor([0, nodes.size(0)], device=device, dtype=torch.long)
        else:
            batch_ptr = _to_tensor(batch_ptr, device).long()

        return nodes, edge_index, batch_ptr

    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        nodes, edge_index, batch_ptr = self._unify_inputs(batch, device)

        # Two stacked layers
        H, _ = self.layer1(nodes, edge_index, num_nodes=nodes.size(0))  # [N, hid_dim]
        H, _ = self.layer2(H, edge_index, num_nodes=nodes.size(0))      # [N, out_dim]

        # Graph-level pooling
        Q = _pool_mean(H, batch_ptr)                                    # [B, out_dim]
        return {"H": H, "Q": Q}
