"""Joint mutual information estimators: I(f, s; y)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit, prange
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


def _weighted_entropy_from_codes(
    codes: np.ndarray,
    w: np.ndarray,
    *,
    n_states: int | None = None,
    w_sum: float | None = None,
    dense_max_states: int = 200_000,
) -> float:
    """Weighted entropy for non-negative integer codes using bincount."""
    codes_i = np.asarray(codes, dtype=np.int64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()

    if w_sum is None:
        w_sum = float(w.sum())
    if w_sum <= 0.0:
        return 0.0

    if n_states is None:
        n_states = int(codes_i.max()) + 1 if codes_i.size else 1

    if n_states <= dense_max_states:
        counts = np.bincount(codes_i, weights=w, minlength=n_states)
    else:
        _, inv = np.unique(codes_i, return_inverse=True)
        counts = np.bincount(inv, weights=w)

    p = counts / w_sum
    mask = p > 1e-12
    return float(-(p[mask] * np.log(p[mask])).sum())


def binned_joint_mi(
    selected: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
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
    w = np.asarray(w, dtype=np.float64).ravel()
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return np.zeros(p, dtype=np.float64)

    s_binned = _quantile_bin(selected, n_bins)

    if y_kind == "discrete":
        y_binned = _factorize(y)
        n_y_bins = int(y_binned.max()) + 1 if y_binned.size else 1
    else:
        y_binned = _quantile_bin(y, n_bins)
        n_y_bins = n_bins

    h_y = _weighted_entropy_from_codes(y_binned, w, n_states=n_y_bins, w_sum=w_sum)

    fs_states = n_bins * n_bins
    fsy_states = fs_states * n_y_bins

    scores = np.empty(p, dtype=np.float64)

    for j in range(p):
        f_binned = _quantile_bin(candidates[:, j], n_bins)

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _weighted_entropy_from_codes(fs_binned, w, n_states=fs_states, w_sum=w_sum)
        h_fsy = _weighted_entropy_from_codes(fsy_binned, w, n_states=fsy_states, w_sum=w_sum)

        scores[j] = max(0.0, h_fs + h_y - h_fsy)

    return scores


@njit(cache=True)
def r2_joint_mi(
    selected: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Weighted R²-based joint MI approximation.

    For predicting y from (f, s), uses analytic R² formula:
    R² = r_ys² + (r_yf - r_ys * r_fs)² / (1 - r_fs²)

    Then: I(f, s; y) ≈ -0.5 * log(1 - R²)

    Parameters
    ----------
    selected : ndarray of shape (n,)
        Previously selected feature values.
    candidates : ndarray of shape (n, p)
        Candidate feature matrix.
    y : ndarray of shape (n,)
        Target values.
    w : ndarray of shape (n,)
        Sample weights (should be normalized to sum=n).
    """
    n, p = candidates.shape

    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    y_mean = 0.0
    for i in range(n):
        y_mean += w[i] * y[i]
    y_mean /= w_sum

    y_var = 0.0
    for i in range(n):
        y_var += w[i] * (y[i] - y_mean) ** 2
    y_var /= w_sum
    y_std = np.sqrt(y_var) if y_var > 1e-12 else 1.0

    y_s = np.empty(n, dtype=np.float64)
    for i in range(n):
        y_s[i] = (y[i] - y_mean) / y_std

    s_mean = 0.0
    for i in range(n):
        s_mean += w[i] * selected[i]
    s_mean /= w_sum

    s_var = 0.0
    for i in range(n):
        s_var += w[i] * (selected[i] - s_mean) ** 2
    s_var /= w_sum
    s_std = np.sqrt(s_var) if s_var > 1e-12 else 1.0

    s_s = np.empty(n, dtype=np.float64)
    for i in range(n):
        s_s[i] = (selected[i] - s_mean) / s_std

    r_ys = 0.0
    for i in range(n):
        r_ys += w[i] * s_s[i] * y_s[i]
    r_ys /= w_sum

    scores = np.empty(p, dtype=np.float64)

    for j in range(p):
        f_mean = 0.0
        for i in range(n):
            f_mean += w[i] * candidates[i, j]
        f_mean /= w_sum

        f_var = 0.0
        for i in range(n):
            f_var += w[i] * (candidates[i, j] - f_mean) ** 2
        f_var /= w_sum
        f_std = np.sqrt(f_var) if f_var > 1e-12 else 1.0

        r_yf = 0.0
        r_fs = 0.0
        for i in range(n):
            f_s_i = (candidates[i, j] - f_mean) / f_std
            r_yf += w[i] * f_s_i * y_s[i]
            r_fs += w[i] * f_s_i * s_s[i]
        r_yf /= w_sum
        r_fs /= w_sum

        denom = 1.0 - r_fs * r_fs
        if denom < 1e-8:
            r2 = r_ys * r_ys
        else:
            a = r_yf - r_ys * r_fs
            r2 = r_ys * r_ys + (a * a) / denom

        r2 = min(max(r2, 0.0), 0.99999)
        scores[j] = -0.5 * np.log(1.0 - r2)

    return scores


@njit(cache=True, parallel=True)
def r2_joint_mi_indexed(
    X_full: np.ndarray,
    cand_idx: np.ndarray,
    selected: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    R²-based joint MI WITHOUT array copying.

    Uses indices to avoid candidate matrix allocations.
    """
    n = X_full.shape[0]
    m = len(cand_idx)

    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    y_mean = 0.0
    for i in range(n):
        y_mean += w[i] * y[i]
    y_mean /= w_sum

    y_var = 0.0
    for i in range(n):
        y_var += w[i] * (y[i] - y_mean) ** 2
    y_var /= w_sum
    y_std = np.sqrt(y_var) if y_var > 1e-12 else 1.0

    y_s = np.empty(n, dtype=np.float64)
    for i in range(n):
        y_s[i] = (y[i] - y_mean) / y_std

    s_mean = 0.0
    for i in range(n):
        s_mean += w[i] * selected[i]
    s_mean /= w_sum

    s_var = 0.0
    for i in range(n):
        s_var += w[i] * (selected[i] - s_mean) ** 2
    s_var /= w_sum
    s_std = np.sqrt(s_var) if s_var > 1e-12 else 1.0

    s_s = np.empty(n, dtype=np.float64)
    for i in range(n):
        s_s[i] = (selected[i] - s_mean) / s_std

    r_ys = 0.0
    for i in range(n):
        r_ys += w[i] * s_s[i] * y_s[i]
    r_ys /= w_sum

    scores = np.empty(m, dtype=np.float64)

    for ci in prange(m):
        j = cand_idx[ci]

        f_mean = 0.0
        for i in range(n):
            f_mean += w[i] * X_full[i, j]
        f_mean /= w_sum

        f_var = 0.0
        for i in range(n):
            f_var += w[i] * (X_full[i, j] - f_mean) ** 2
        f_var /= w_sum
        f_std = np.sqrt(f_var) if f_var > 1e-12 else 1.0

        r_yf = 0.0
        r_fs = 0.0
        for i in range(n):
            f_s_i = (X_full[i, j] - f_mean) / f_std
            r_yf += w[i] * f_s_i * y_s[i]
            r_fs += w[i] * f_s_i * s_s[i]
        r_yf /= w_sum
        r_fs /= w_sum

        denom = 1.0 - r_fs * r_fs
        if denom < 1e-8:
            r2 = r_ys * r_ys
        else:
            a = r_yf - r_ys * r_fs
            r2 = r_ys * r_ys + (a * a) / denom

        r2 = min(max(r2, 0.0), 0.99999)
        scores[ci] = -0.5 * np.log(1.0 - r2)

    return scores


def binned_joint_mi_indexed(
    X_full: np.ndarray,
    cand_idx: np.ndarray,
    selected: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_bins: int = 10,
    y_kind: Literal["discrete", "continuous"] = "continuous",
) -> np.ndarray:
    """Binned joint MI WITHOUT candidate-matrix copying."""
    X_full = np.asarray(X_full)
    if X_full.ndim != 2:
        raise ValueError("X_full must be 2D")

    cand_idx = np.asarray(cand_idx, dtype=np.int64).ravel()
    m = cand_idx.size
    if m == 0:
        return np.empty(0, dtype=np.float64)

    w = np.asarray(w, dtype=np.float64).ravel()
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return np.zeros(m, dtype=np.float64)

    s_binned = _quantile_bin(selected, n_bins)

    if y_kind == "discrete":
        y_binned = _factorize(y)
        n_y_bins = int(y_binned.max()) + 1 if y_binned.size else 1
    else:
        y_binned = _quantile_bin(y, n_bins)
        n_y_bins = n_bins

    h_y = _weighted_entropy_from_codes(y_binned, w, n_states=n_y_bins, w_sum=w_sum)

    fs_states = n_bins * n_bins
    fsy_states = fs_states * n_y_bins

    scores = np.empty(m, dtype=np.float64)

    for ci in range(m):
        j = int(cand_idx[ci])
        f_binned = _quantile_bin(X_full[:, j], n_bins)

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _weighted_entropy_from_codes(fs_binned, w, n_states=fs_states, w_sum=w_sum)
        h_fsy = _weighted_entropy_from_codes(fsy_binned, w, n_states=fsy_states, w_sum=w_sum)

        scores[ci] = max(0.0, h_fs + h_y - h_fsy)

    return scores


def ksg_joint_mi(
    selected: np.ndarray,
    candidates: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """
    KSG k-NN estimator for joint MI.

    Note: This estimator does not support sample weights.
    """
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


def _weighted_entropy_from_array(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted entropy from discrete array."""
    unique_vals = np.unique(x)
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0

    ent = 0.0
    for val in unique_vals:
        mask = x == val
        p = w[mask].sum() / w_sum
        if p > 1e-12:
            ent -= p * np.log(p)
    return ent


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
