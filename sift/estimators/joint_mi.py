"""Joint mutual information estimators: I(f, s; y)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from scipy.special import digamma


@njit(cache=True)
def _entropy_from_counts(counts: np.ndarray) -> float:
    """Entropy from count array."""
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    ent = 0.0
    for i in range(len(p)):
        if p[i] > 1e-12:
            ent -= p[i] * np.log(p[i])
    return ent


def binned_joint_mi(
    selected: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    y_kind: Literal["discrete", "continuous"] = "continuous",
) -> np.ndarray:
    """
    Binned joint MI: I(f, s; y) for each candidate f.

    Parameters
    ----------
    y_kind : str
        - "discrete": y is categorical (factorize to codes)
        - "continuous": quantile-bin y
    """
    n, p = candidates.shape

    s_binned = _quantile_bin(selected, n_bins)

    if y_kind == "discrete":
        y_binned = _factorize(y)
        n_y_bins = int(y_binned.max()) + 1
    else:
        y_binned = _quantile_bin(y, n_bins)
        n_y_bins = n_bins

    scores = np.empty(p, dtype=np.float64)

    for j in range(p):
        f_binned = _quantile_bin(candidates[:, j], n_bins)

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _entropy_from_array(fs_binned)
        h_y = _entropy_from_array(y_binned)
        h_fsy = _entropy_from_array(fsy_binned)

        scores[j] = max(0.0, h_fs + h_y - h_fsy)

    return scores


@njit(cache=True)
def r2_joint_mi(
    selected: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    R²-based joint MI approximation.

    For predicting y from (f, s), uses analytic R² formula:
    R² = r_ys² + (r_yf - r_ys * r_fs)² / (1 - r_fs²)

    Then: I(f, s; y) ≈ -0.5 * log(1 - R²)
    """
    n, p = candidates.shape

    y_s = (y - np.mean(y)) / (np.std(y) + 1e-12)
    s_s = (selected - np.mean(selected)) / (np.std(selected) + 1e-12)

    r_ys = np.sum(s_s * y_s) / n

    scores = np.empty(p, dtype=np.float64)

    for j in range(p):
        f = candidates[:, j]
        f_s = (f - np.mean(f)) / (np.std(f) + 1e-12)

        r_yf = np.sum(f_s * y_s) / n
        r_fs = np.sum(f_s * s_s) / n

        denom = 1.0 - r_fs * r_fs
        if denom < 1e-8:
            r2 = r_ys * r_ys
        else:
            a = r_yf - r_ys * r_fs
            r2 = r_ys * r_ys + (a * a) / denom

        r2 = min(max(r2, 0.0), 0.99999)
        scores[j] = -0.5 * np.log(1.0 - r2)

    return scores


def ksg_joint_mi(
    selected: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """KSG k-NN estimator for joint MI."""
    n, p = candidates.shape
    scores = np.empty(p, dtype=np.float64)

    y_s = (y - y.mean()) / (y.std() + 1e-10)
    s_s = (selected - selected.mean()) / (selected.std() + 1e-10)

    for j in range(p):
        f = candidates[:, j]
        f_s = (f - f.mean()) / (f.std() + 1e-10)

        X_joint = np.column_stack([f_s, s_s])
        Y_marginal = y_s.reshape(-1, 1)
        XY_full = np.column_stack([f_s, s_s, y_s])

        tree_full = cKDTree(XY_full)
        tree_x = cKDTree(X_joint)
        tree_y = cKDTree(Y_marginal)

        dists, _ = tree_full.query(XY_full, k=k + 1, p=np.inf)
        eps = np.maximum(dists[:, -1] - 1e-15, 0.0)

        n_x = _safe_count_neighbors(tree_x, X_joint, eps, n)
        n_y = _safe_count_neighbors(tree_y, Y_marginal, eps, n)

        n_x = np.maximum(n_x, 0)
        n_y = np.maximum(n_y, 0)

        mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
        scores[j] = max(mi, 0.0)

    return scores


def _quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-based binning."""
    if x.size == 0 or np.std(x) < 1e-12:
        return np.zeros(len(x), dtype=np.int32)
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(x, percentiles)
    bins[0] -= 1e-10
    bins[-1] += 1e-10
    return np.digitize(x, bins[1:-1]).astype(np.int32)


def _factorize(x: np.ndarray) -> np.ndarray:
    """Convert to integer codes."""
    _, codes = np.unique(x, return_inverse=True)
    return codes.astype(np.int32)


def _entropy_from_array(x: np.ndarray) -> float:
    """Entropy from discrete array."""
    _, counts = np.unique(x, return_counts=True)
    return _entropy_from_counts(counts)


def _safe_count_neighbors(tree: cKDTree, points: np.ndarray, radii: np.ndarray, n: int) -> np.ndarray:
    """Count neighbors with fallback for older SciPy."""
    try:
        return np.array(
            [
                tree.query_ball_point(points[i], radii[i], p=np.inf, return_length=True) - 1
                for i in range(n)
            ],
            dtype=np.int64,
        )
    except TypeError:
        return np.array(
            [
                len(tree.query_ball_point(points[i], radii[i], p=np.inf)) - 1
                for i in range(n)
            ],
            dtype=np.int64,
        )
