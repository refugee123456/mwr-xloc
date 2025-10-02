# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

# ---------- basic utils ----------

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / n

def pairwise_sqdist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [N, D], B: [M, D] -> [N, M]
    A2 = (A * A).sum(dim=1, keepdim=True)          # [N,1]
    B2 = (B * B).sum(dim=1, keepdim=True).t()      # [1,M]
    D2 = A2 + B2 - 2.0 * (A @ B.t())
    return torch.clamp(D2, min=0.0)

def distance_logits(Z: torch.Tensor, P: torch.Tensor, gamma: float = 16.0, scale_by_dim: bool = False) -> torch.Tensor:
    """
    Use negative squared Euclidean distance as logits (the closer, the larger).
    For binary classification returns [N, C].
    """
    if Z.numel() == 0 or P.numel() == 0:
        return Z.new_zeros(Z.size(0), max(1, P.size(0)))
    D2 = pairwise_sqdist(Z, P)  # [N, C]
    g = gamma / float(Z.size(1)) if scale_by_dim else gamma
    return -g * D2

# ---------- recalibration ----------

@torch.no_grad()
def _cosine_sim(X: torch.Tensor) -> torch.Tensor:
    Xn = l2_normalize(X, dim=-1)
    return Xn @ Xn.t()

@torch.no_grad()
def _rbf_kernel(X: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    D2 = pairwise_sqdist(X, X)
    return torch.exp(-gamma * D2)

@torch.no_grad()
def instance_recalibrate(
    Z: torch.Tensor,
    Y: Optional[torch.Tensor],
    tau: float = 0.25,
    sim: str = "rbf",
    rbf_gamma: float = 0.5,
    exclude_self: bool = True
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Dict]:
    """
    Instance-wise recalibration with class-restricted kernel weights:
      z'_i = sum_j softmax(K_ij / tau) * z_j (restricted to same-class neighbors)

    Returns:
      Z_rc:   [N, D] recalibrated embeddings
      w_list: length-N list of per-sample importance (how much each sample is "attended to"
              by same-class peers); can be None
      aux:    dict with debug info
    """
    N, D = int(Z.size(0)), int(Z.size(1))
    if Y is None or Y.numel() == 0:
        return Z.clone(), None, {"note": "no_label"}
    Z_rc = Z.clone()
    imp_list: List[torch.Tensor] = [Z.new_zeros(()) for _ in range(N)]

    uniq = Y.unique(sorted=True)
    for c in uniq:
        idx = (Y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        Xc = Z[idx]  # [Nc, D]
        if Xc.size(0) == 1:
            Z_rc[idx] = Xc
            imp_list[idx.item()] = Z.new_tensor(1.0)
            continue

        if sim.lower() == "cos":
            K = _cosine_sim(Xc)
        else:  # "rbf"
            K = _rbf_kernel(Xc, gamma=rbf_gamma)

        if exclude_self:
            K.fill_diagonal_(float("-inf"))

        # Row-wise softmax with temperature
        W = F.softmax(K / max(1e-6, tau), dim=1)  # [Nc, Nc]
        Xc_rc = W @ Xc                             # [Nc, D]
        Z_rc[idx] = Xc_rc

        # Use “being attended” as importance: column-wise sum of weights for each sample
        imp = W.sum(dim=0)  # [Nc]
        for k, j in enumerate(idx.tolist()):
            imp_list[j] = imp[k].detach()

    aux = {"classes": int(uniq.numel())}
    return Z_rc, imp_list, aux

def compute_prototypes(
    Z: torch.Tensor,
    Y: torch.Tensor,
    w_list: Optional[List[torch.Tensor]] = None
) -> torch.Tensor:
    """Compute class prototypes with optional per-instance weights."""
    protos = []
    uniq = Y.unique(sorted=True)
    for c in uniq:
        idx = (Y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        Xc = Z[idx]  # [Nc,D]
        if w_list is not None:
            w = torch.stack([w_list[i].to(Z.device).view(1) for i in idx.tolist()], 0)  # [Nc,1]
            w = torch.clamp(w, min=1e-6)
            w = w / w.sum()
            pc = (w * Xc).sum(dim=0, keepdim=True)
        else:
            pc = Xc.mean(dim=0, keepdim=True)
        protos.append(pc)
    if len(protos) == 0:
        return Z.new_zeros(0, Z.size(1))
    return torch.cat(protos, dim=0)

# ---------- simple PFA blocks kept for compatibility ----------

def ridge_reprojection(P: torch.Tensor, Q_tilde: torch.Tensor, lam: float = 0.06) -> torch.Tensor:
    """
    Solve W = argmin ||P W - Q_tilde||^2 + lam ||W||^2
      → W = (P^T P + lam I)^(-1) P^T Q_tilde
    """
    D = P.size(1)
    A = P.t() @ P + lam * torch.eye(D, device=P.device, dtype=P.dtype)
    B = P.t() @ Q_tilde
    W = torch.linalg.solve(A, B)
    return W  # [D,D]

def spectral_clip(W: torch.Tensor, smax: float = 3.0) -> torch.Tensor:
    """Clip singular values of W by smax."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    S = torch.clamp(S, max=smax)
    return (U * S) @ Vh

def apply_reprojection(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply linear re-projection X @ W."""
    return X @ W

def vtan_normalize(Zs: torch.Tensor, Zq: torch.Tensor, eps: float = 1e-6):
    """
    Normalize by support set statistics per dimension (align shift/scale).
    Returns standardized (Zs_n, Zq_n) and the (mu, std) used.
    """
    mu = Zs.mean(dim=0, keepdim=True)
    std = Zs.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    Zs_n = (Zs - mu) / std
    Zq_n = (Zq - mu) / std
    return Zs_n, Zq_n, (mu.squeeze(0), std.squeeze(0))
