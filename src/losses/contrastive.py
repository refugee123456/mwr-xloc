# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F

# --------- helpers ---------
def _pairwise_l2(x: torch.Tensor) -> torch.Tensor:
    """x:[N,d] -> D:[N,N]  (Euclidean L2 distance, not squared)."""
    sq = (x * x).sum(dim=1, keepdim=True)
    dist2 = torch.clamp(sq + sq.t() - 2.0 * (x @ x.t()), min=0.0)
    return torch.sqrt(dist2 + 1e-12)

def _pairwise_l2_sq(x: torch.Tensor) -> torch.Tensor:
    """x:[N,d] -> D2:[N,N]  (squared Euclidean distance)."""
    sq = (x * x).sum(dim=1, keepdim=True)
    return torch.clamp(sq + sq.t() - 2.0 * (x @ x.t()), min=0.0)

# --------- 1) Contrastive (Siamese margin) ---------
def contrastive_siamese(z: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Classic contrastive loss (Siamese with margin) on embeddings z and labels y.
    Pulls positives together, pushes negatives apart up to a margin.
    """
    D = _pairwise_l2(z)
    N = z.size(0)
    eye = torch.eye(N, device=z.device, dtype=torch.bool)
    same = (y[:, None] == y[None, :]) & (~eye)
    diff = (~same) & (~eye)

    pos = D[same]
    neg = D[diff]
    loss_pos = (pos ** 2).mean() if pos.numel() > 0 else z.new_tensor(0.0)
    loss_neg = torch.clamp(margin - neg, min=0.0).pow(2).mean() if neg.numel() > 0 else z.new_tensor(0.0)
    return loss_pos + loss_neg

# --------- 2) Triplet-hard ---------
def triplet_hard(z: torch.Tensor, y: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    Triplet loss with hard mining:
      - For each anchor i, pick the farthest positive and the closest negative.
      - Loss = max(0, margin + d(ap) - d(an)).
    """
    D = _pairwise_l2(z)
    N = z.size(0)
    loss = z.new_tensor(0.0)
    valid = 0
    for i in range(N):
        pos = (y == y[i]).clone(); pos[i] = False
        neg = (y != y[i])
        if pos.sum() == 0 or neg.sum() == 0:  # cannot form a triplet
            continue
        d_ap = D[i, pos].max()  # hardest positive (farthest)
        d_an = D[i, neg].min()  # hardest negative (closest)
        loss = loss + torch.clamp(margin + d_ap - d_an, min=0.0)
        valid += 1
    return loss / max(valid, 1)

# --------- 3) Triplet semi-hard ---------
def triplet_semi_hard(z: torch.Tensor, y: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    Triplet loss with semi-hard negative mining:
      - Choose negatives with d(an) in (d(ap), d(ap)+margin); fallback to closest negative if none.
    """
    D = _pairwise_l2(z)
    N = z.size(0)
    loss = z.new_tensor(0.0)
    valid = 0
    for i in range(N):
        pos = (y == y[i]).clone(); pos[i] = False
        neg = (y != y[i])
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        d_ap = D[i, pos].max()
        d_an_all = D[i, neg]
        candidates = d_an_all[(d_an_all > d_ap) & (d_an_all < d_ap + margin)]
        d_an = candidates.min() if candidates.numel() > 0 else d_an_all.min()
        loss = loss + torch.clamp(margin + d_ap - d_an, min=0.0)
        valid += 1
    return loss / max(valid, 1)

# --------- 4) N-pairs (use -L2^2/τ as similarity) ---------
def npairs(z: torch.Tensor, y: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    N-pairs loss:
      - Build similarity matrix S = -||z_i - z_j||^2 / τ.
      - For each class, pair anchors/positives, treat all others as negatives in a softmax.
    """
    device = z.device
    d2 = _pairwise_l2_sq(z)                    # [N,N]
    S = -d2 / max(1e-12, float(tau))           # similarity

    loss = z.new_tensor(0.0)
    pairs = 0
    for cls in torch.unique(y):
        idx = torch.nonzero(y == cls, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        k = (idx.numel() // 2) * 2
        anchors = idx[:k // 2]
        positives = idx[k // 2:k]
        for a, p in zip(anchors.tolist(), positives.tolist()):
            # Place the positive at class-0 position in the softmax; others are negatives.
            s_ap = S[a, p].view(1, 1)
            s_an = S[a, y != y[a]].view(1, -1)
            logits = torch.cat([s_ap, s_an], dim=1)
            tgt = torch.zeros(1, dtype=torch.long, device=device)
            loss = loss + F.cross_entropy(logits, tgt)
            pairs += 1
    return loss / max(pairs, 1)

# --------- unified entry point ---------
def contrastive_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    method: str = "none",
    margin: float = 0.2,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Unified wrapper for multiple contrastive objectives.

    Args:
        z: [N, D] embeddings.
        y: [N] integer labels.
        method: one of {"none","contrastive/siamese","triplet_hard","triplet_semi","npairs"}.
        margin: margin used by triplet/contrastive where applicable.
        tau: temperature used by n-pairs.
    """
    m = (method or "none").lower()
    if m in ("none", "", "off"):
        return z.new_tensor(0.0)
    if m in ("contrastive", "siamese", "siamese_margin"):
        return contrastive_siamese(z, y, margin=margin)
    if m in ("triplet_hard", "hard"):
        return triplet_hard(z, y, margin=margin)
    if m in ("triplet_semi", "semi", "semi_hard", "triplet_semihard"):
        return triplet_semi_hard(z, y, margin=margin)
    if m in ("npairs", "n-pairs", "n_pairs"):
        return npairs(z, y, tau=tau)
    raise ValueError(f"Unknown contrastive method: {method}")
