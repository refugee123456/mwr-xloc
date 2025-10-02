# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F

def metric_ce_loss(logits: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Cross-entropy loss (with built-in softmax).
    - logits: [N, C]
    - target: [N] (values in 0..C-1)
    """
    return F.cross_entropy(logits, target.long(), reduction=reduction)
