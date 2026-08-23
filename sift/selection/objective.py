"""Objective path computation for information-theoretic feature selection."""

from __future__ import annotations

import numpy as np


def objective_from_corr_path(
    R_path: np.ndarray,
    r_path: np.ndarray,
    *,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Gaussian MI objective for a fixed feature path.

    Given correlation matrix R_path and target correlations r_path for an
    ordered sequence of features, computes:
        obj[t] = log|Σ_S| - log|Σ_{y,S}| = 2 * I(y; S)

    Uses Schur complement updates for O(k²) total complexity.

    Parameters
    ----------
    R_path : ndarray of shape (k, k)
        Correlation matrix of features in path order.
    r_path : ndarray of shape (k,)
        Correlations between each feature and target y.
    shrink : float
        Shrinkage toward identity (numerical stability).
    eps : float
        Floor for determinant values.

    Returns
    -------
    objective : ndarray of shape (k,)
        Cumulative objective at each step (monotonically non-decreasing).
    """
    r = np.asarray(r_path, dtype=np.float64).ravel()
    k = r.size
    if k == 0:
        return np.empty(0, dtype=np.float64)

    R = np.asarray(R_path, dtype=np.float64)
    if R.shape != (k, k):
        raise ValueError(f"R_path must be shape ({k}, {k}), got {R.shape}")
    if shrink > 0.0:
        R = (1.0 - shrink) * R
        r = (1.0 - shrink) * r
        np.fill_diagonal(R, 1.0)

    obj = np.empty(k, dtype=np.float64)

    # Partial Cholesky recursion in path order: ``d`` holds the residual
    # variance of each path feature given the earlier ones, ``c`` its residual
    # covariance with y, and ``dy`` the residual variance of y. The two Schur
    # complements of step t are s1 = d[t] and s2 = d[t] - c[t]^2 / dy, so
    # log|Sigma_S| - log|Sigma_{y,S}| accumulates log(s1) - log(s2). This is
    # O(k^2) and numerically stable for shrunk (positive definite) inputs.
    L = np.zeros((k, k), dtype=np.float64)
    d = np.ones(k, dtype=np.float64)
    c = r.copy()
    dy = 1.0
    logdet_S = 0.0
    logdet_yS = 0.0

    for t in range(k):
        s1 = max(float(d[t]), eps)
        s2 = max(float(d[t] - c[t] * c[t] / dy), eps)
        logdet_S += np.log(s1)
        logdet_yS += np.log(s2)
        obj[t] = logdet_S - logdet_yS
        if t + 1 >= k:
            break
        sq = np.sqrt(s1)
        ly = c[t] / sq
        dy -= ly * ly
        rest = slice(t + 1, k)
        l_col = (R[rest, t] - L[rest, :t] @ L[t, :t]) / sq
        L[rest, t] = l_col
        d[rest] -= l_col * l_col
        c[rest] -= l_col * ly

    return obj
