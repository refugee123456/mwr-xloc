# -*- coding: utf-8 -*-
"""
Utility math helpers used across the project.
- Works with both NumPy arrays and PyTorch tensors.
- Functions:
    * l2_normalize(x)                        -> row-wise normalize to unit length
    * cosine_similarity(x, y)                -> scalar cosine between two 1D vectors
    * euclidean_distance(X, Y)               -> row-wise euclid distances between X[i], Y[i]
    * pairwise_euclidean_distance(X, Y, squared=False) -> [N,M] distance matrix
    * cholesky_solve(A, B)                   -> solve A X = B for SPD A (NumPy/Torch)
"""

from __future__ import annotations
from typing import Tuple
import math

try:
    import numpy as np
except Exception as e:
    np = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore

__all__ = [
    "l2_normalize",
    "cosine_similarity",
    "euclidean_distance",
    "pairwise_euclidean_distance",
    "cholesky_solve",
]

_EPS = 1e-12


def _is_numpy(x) -> bool:
    return (np is not None) and isinstance(x, np.ndarray)


def _is_torch(x) -> bool:
    return (torch is not None) and hasattr(torch, "Tensor") and isinstance(x, torch.Tensor)


# --------------------------------------------------------------------------- #
# Row-wise L2 normalize
# --------------------------------------------------------------------------- #
def l2_normalize(x):
    """
    Row-wise L2 normalize.
    - x: [N,D] (numpy array or torch tensor)
    - return: same type/shape
    """
    if _is_numpy(x):
        nrm = np.sqrt((x * x).sum(axis=1, keepdims=True) + _EPS)
        return x / nrm
    elif _is_torch(x):
        nrm = torch.sqrt((x * x).sum(dim=1, keepdim=True) + _EPS)
        return x / nrm
    else:
        raise TypeError("l2_normalize expects a NumPy array or a torch.Tensor")


# --------------------------------------------------------------------------- #
# Cosine similarity between two vectors
# --------------------------------------------------------------------------- #
def cosine_similarity(x, y) -> float:
    """
    Cosine similarity between two 1-D vectors (returns Python float).
    """
    if _is_numpy(x) and _is_numpy(y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        num = float((x * y).sum())
        den = math.sqrt(float((x * x).sum()) + _EPS) * math.sqrt(float((y * y).sum()) + _EPS)
        return num / den
    elif _is_torch(x) and _is_torch(y):
        x = x.reshape(-1)
        y = y.reshape(-1)
        num = float(torch.dot(x, y).item())
        den = math.sqrt(float(torch.dot(x, x).item()) + _EPS) * math.sqrt(float(torch.dot(y, y).item()) + _EPS)
        return num / den
    else:
        raise TypeError("cosine_similarity expects both inputs to be NumPy arrays or torch.Tensors")


# --------------------------------------------------------------------------- #
# Row-wise euclidean distance: X[i] vs Y[i]
# --------------------------------------------------------------------------- #
def euclidean_distance(X, Y):
    """
    Row-wise Euclidean distance between X[i] and Y[i].
    - X, Y: [N,D]
    - return: [N] distances
    """
    if _is_numpy(X) and _is_numpy(Y):
        diff = (X - Y).astype(np.float64)
        return np.sqrt((diff * diff).sum(axis=1))
    elif _is_torch(X) and _is_torch(Y):
        diff = X - Y
        return torch.sqrt((diff * diff).sum(dim=1) + _EPS)
    else:
        raise TypeError("euclidean_distance expects X, Y are both NumPy arrays or both torch.Tensors")


# --------------------------------------------------------------------------- #
# Pairwise euclidean distance matrix
# --------------------------------------------------------------------------- #
def pairwise_euclidean_distance(X, Y, squared: bool = False):
    """
    Pairwise Euclidean distance between rows of X and Y.
    - X: [N,D], Y: [M,D]
    - return: [N,M] distances (or squared distances if squared=True)
    """
    if _is_numpy(X) and _is_numpy(Y):
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        X2 = (X * X).sum(axis=1, keepdims=True)   # [N,1]
        Y2 = (Y * Y).sum(axis=1, keepdims=True).T # [1,M]
        D2 = X2 + Y2 - 2.0 * (X @ Y.T)            # [N,M]
        # 数值稳定：裁剪到 >= 0
        D2 = np.maximum(D2, 0.0)
        if squared:
            return D2
        return np.sqrt(D2 + _EPS)
    elif _is_torch(X) and _is_torch(Y):
        X2 = (X * X).sum(dim=1, keepdim=True)     # [N,1]
        Y2 = (Y * Y).sum(dim=1, keepdim=True).t() # [1,M]
        D2 = X2 + Y2 - 2.0 * (X @ Y.t())          # [N,M]
        D2 = torch.clamp(D2, min=0.0)
        if squared:
            return D2
        return torch.sqrt(D2 + _EPS)
    else:
        raise TypeError("pairwise_euclidean_distance expects X, Y are both NumPy arrays or both torch.Tensors")


# --------------------------------------------------------------------------- #
# Cholesky solve for SPD systems: A X = B
# --------------------------------------------------------------------------- #
def _np_forward_subst(L: "np.ndarray", B: "np.ndarray") -> "np.ndarray":
    """Solve L Y = B for lower-triangular L (NumPy)."""
    n = L.shape[0]
    Y = np.zeros_like(B)
    for i in range(n):
        s = (L[i, :i] @ Y[:i]).reshape(-1, *B.shape[1:])
        Y[i] = (B[i] - s) / (L[i, i] + _EPS)
    return Y


def _np_backward_subst(U: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
    """Solve U X = Y for upper-triangular U (NumPy)."""
    n = U.shape[0]
    X = np.zeros_like(Y)
    for i in range(n - 1, -1, -1):
        s = (U[i, i + 1:] @ X[i + 1:]).reshape(-1, *Y.shape[1:])
        X[i] = (Y[i] - s) / (U[i, i] + _EPS)
    return X


def cholesky_solve(A, B):
    """
    Solve A X = B for symmetric positive-definite A using Cholesky.
    - A: [n,n]
    - B: [n] or [n,k]
    Returns X with same backend (NumPy/Torch) and shape as B.
    Fallback to generic solver if Cholesky fails.
    """
    # ------- NumPy path -------
    if _is_numpy(A) and _is_numpy(B):
        A = A.astype(np.float64)
        B = B.astype(np.float64)
        try:
            L = np.linalg.cholesky(A)           # A = L L^T
            # Solve L Y = B
            if B.ndim == 1:
                Y = _np_forward_subst(L, B[:, None])  # [n,1]
            else:
                Y = _np_forward_subst(L, B)
            # Solve L^T X = Y
            X = _np_backward_subst(L.T, Y)
            return X.squeeze() if B.ndim == 1 else X
        except Exception:
            # Fallback to np.linalg.solve (robust but slower)
            X = np.linalg.solve(A, B)
            return X
    # ------- Torch path -------
    elif _is_torch(A) and _is_torch(B):
        # Ensure correct dtype/device
        A = A.to(dtype=torch.get_default_dtype())
        B = B.to(dtype=torch.get_default_dtype())
        n = A.size(0)
        try:
            # Prefer torch.linalg API when available
            if hasattr(torch.linalg, "cholesky"):
                L = torch.linalg.cholesky(A)       # [n,n]
            else:
                L = torch.cholesky(A, upper=False)
            # Use triangular_solve
            # Solve L Y = B
            Y = torch.linalg.solve_triangular(L, B, upper=False)
            # Solve L^T X = Y
            X = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True)
            return X
        except Exception:
            # Fallback: generic solver
            if hasattr(torch.linalg, "solve"):
                return torch.linalg.solve(A, B)
            else:
                # final fallback via LU
                return torch.gesv(B, A)[0]  # deprecated in new torch, but keeps compatibility
    else:
        raise TypeError("cholesky_solve expects A, B are both NumPy arrays or both torch.Tensors")
