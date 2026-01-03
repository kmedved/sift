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

    logdet_S = 0.0
    inv_S = np.array([[1.0]], dtype=np.float64)

    r0 = float(r[0])
    det_yS = max(1.0 - r0 * r0, eps)
    logdet_yS = float(np.log(det_yS))
    inv_yS = (1.0 / det_yS) * np.array([[1.0, -r0], [-r0, 1.0]], dtype=np.float64)

    obj[0] = logdet_S - logdet_yS

    for t in range(1, k):
        b = R[:t, t].reshape(-1, 1)
        v = inv_S @ b
        s1 = max(float(1.0 - (b.T @ v)[0, 0]), eps)

        inv_S_new = np.empty((t + 1, t + 1), dtype=np.float64)
        inv_S_new[:t, :t] = inv_S + (v @ v.T) / s1
        inv_S_new[:t, t] = -v[:, 0] / s1
        inv_S_new[t, :t] = -v[:, 0] / s1
        inv_S_new[t, t] = 1.0 / s1
        inv_S = inv_S_new
        logdet_S += np.log(s1)

        b2 = np.empty((t + 1, 1), dtype=np.float64)
        b2[0, 0] = r[t]
        b2[1:, 0] = b[:, 0]

        v2 = inv_yS @ b2
        s2 = max(float(1.0 - (b2.T @ v2)[0, 0]), eps)

        inv_yS_new = np.empty((t + 2, t + 2), dtype=np.float64)
        inv_yS_new[: t + 1, : t + 1] = inv_yS + (v2 @ v2.T) / s2
        inv_yS_new[: t + 1, t + 1] = -v2[:, 0] / s2
        inv_yS_new[t + 1, : t + 1] = -v2[:, 0] / s2
        inv_yS_new[t + 1, t + 1] = 1.0 / s2
        inv_yS = inv_yS_new
        logdet_yS += np.log(s2)

        obj[t] = logdet_S - logdet_yS

    return obj
