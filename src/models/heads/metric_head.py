# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricHead(nn.Module):
    """
    Distance-softmax classification head: maintains learnable prototypes P:[C,d] and learnable temperature γ
    logits_c = -γ * ||q - P_c||^2 / d  (optionally normalized by d)

    Args:
        num_classes: number of classes C
        feat_dim:    feature dimension d
        gamma_init:  initial γ
        learnable_gamma: whether γ is learnable
        per_class_gamma: whether to use class-wise γ_c
        norm_by_dim: whether to normalize distance by d
        init_std:    std for random initialization of prototypes
        min_gamma/max_gamma: lower/upper bound of γ (max=None means no upper bound)
    """
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        gamma_init: float = 10.0,
        learnable_gamma: bool = True,
        per_class_gamma: bool = False,
        norm_by_dim: bool = True,
        init_std: float = 2e-2,
        min_gamma: float = 1e-4,
        max_gamma: Optional[float] = None,
    ):
        super().__init__()
        assert num_classes >= 2 and feat_dim > 0
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.norm_by_dim = bool(norm_by_dim)
        self.min_gamma = float(min_gamma)
        self.max_gamma = (None if max_gamma is None else float(max_gamma))
        self.learnable_gamma = bool(learnable_gamma)
        self.per_class_gamma = bool(per_class_gamma)

        # Prototype vectors
        P = torch.randn(num_classes, feat_dim) * init_std
        self.prototypes = nn.Parameter(P)  # [C,d]

        # γ: softplus(log_gamma) ensures positivity
        shape = (self.num_classes,) if per_class_gamma else (1,)
        log_g_init = torch.log(torch.tensor(float(gamma_init)))
        self.log_gamma = nn.Parameter(
            log_g_init.expand(shape).clone(), requires_grad=self.learnable_gamma
        )

    # ---- γ related ----
    def _gamma_value(self) -> torch.Tensor:
        g = F.softplus(self.log_gamma) + self.min_gamma
        if self.max_gamma is not None:
            g = torch.clamp(g, max=self.max_gamma)
        return g

    @property
    def gamma(self) -> torch.Tensor:
        return self._gamma_value().detach()

    @torch.no_grad()
    def init_from_means(self, feats: torch.Tensor, y: torch.Tensor, momentum: float = 0.0):
        """
        Warm-start / EMA prototypes using batch means; can be called multiple times.
        feats:[B,d], y:[B]
        """
        assert feats.dim() == 2 and feats.size(1) == self.feat_dim
        device = self.prototypes.device
        feats = feats.to(device); y = y.to(device).long()
        C, d = self.num_classes, self.feat_dim
        means = torch.zeros(C, d, device=device)
        counts = torch.zeros(C, device=device)
        for c in range(C):
            m = (y == c)
            if m.any():
                means[c] = feats[m].mean(0)
                counts[c] = m.sum()
        valid = counts > 0
        if valid.any():
            self.prototypes.data[valid] = momentum * self.prototypes.data[valid] + (1 - momentum) * means[valid]

    # ---- Forward: return logits (and optional distances) ----
    def forward(self, q: torch.Tensor, return_dist: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert q.dim() == 2 and q.size(1) == self.feat_dim, f"q shape {tuple(q.shape)} != [B,{self.feat_dim}]"
        B, d = q.shape
        P = self.prototypes                       # [C,d]
        q2 = (q ** 2).sum(-1, keepdim=True)       # [B,1]
        p2 = (P ** 2).sum(-1).unsqueeze(0)        # [1,C]
        dist2 = q2 + p2 - 2.0 * (q @ P.t())       # [B,C]
        dist = torch.clamp(dist2, min=0.0)
        if self.norm_by_dim:
            dist = dist / float(d)

        g = self._gamma_value()                   # [1] or [C]
        logits = -dist * (g.unsqueeze(0) if g.numel() > 1 else g)
        return (logits, dist) if return_dist else (logits, None)

    def extra_repr(self) -> str:
        typ = "per-class" if self.per_class_gamma else "scalar"
        learn = "learnable" if self.learnable_gamma else "fixed"
        return f"C={self.num_classes}, d={self.feat_dim}, gamma({typ},{learn})"
