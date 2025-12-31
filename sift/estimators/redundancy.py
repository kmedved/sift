"""Redundancy scoring: feature-feature association."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def corr_with_selected(
    X: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """Pearson correlation of each candidate with the selected feature."""
    n, p = X.shape

    s_centered = selected - np.mean(selected)
    s_ss = np.sum(s_centered**2)

    if s_ss < 1e-12:
        return np.zeros(p, dtype=np.float64)

    corrs = np.empty(p, dtype=np.float64)
    for j in prange(p):
        x_centered = X[:, j] - np.mean(X[:, j])
        x_ss = np.sum(x_centered**2)
        if x_ss < 1e-12:
            corrs[j] = 0.0
        else:
            corrs[j] = np.sum(x_centered * s_centered) / np.sqrt(x_ss * s_ss)

    return corrs


@njit(cache=True)
def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Full Pearson correlation matrix."""
    n, p = X.shape

    X_centered = np.empty_like(X)
    for j in range(p):
        X_centered[:, j] = X[:, j] - np.mean(X[:, j])

    R = np.empty((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i, p):
            ss_i = np.sum(X_centered[:, i] ** 2)
            ss_j = np.sum(X_centered[:, j] ** 2)
            if ss_i < 1e-12 or ss_j < 1e-12:
                R[i, j] = 0.0
            else:
                R[i, j] = np.sum(X_centered[:, i] * X_centered[:, j]) / np.sqrt(
                    ss_i * ss_j
                )
            R[j, i] = R[i, j]

    np.fill_diagonal(R, 1.0)
    return R
