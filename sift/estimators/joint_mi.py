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
        y_arr = np.asarray(y)
        if (
            np.issubdtype(y_arr.dtype, np.integer)
            and y_arr.size > 0
            and y_arr.min() >= 0
            and y_arr.max() <= 200_000
        ):
            y_binned = y_arr.astype(np.int64, copy=False)
        else:
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


# Keep this kernel serial for the same CatBoost/OpenMP compatibility reason as
# the relevance kernels.
@njit(cache=True)
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

    for ci in range(m):
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


def _weighted_standardize_2d(
    X: np.ndarray,
    w: np.ndarray,
    w_sum: float,
) -> np.ndarray:
    """Weighted-standardize columns with the same zero-variance convention as R2 JMI."""
    X_arr = np.asarray(X, dtype=np.float64)
    mean = (X_arr * w[:, None]).sum(axis=0) / w_sum
    centered = X_arr - mean
    var = (centered * centered * w[:, None]).sum(axis=0) / w_sum
    std = np.where(var > 1e-12, np.sqrt(var), 1.0)
    return centered / std


def _weighted_standardize_1d(
    x: np.ndarray,
    w: np.ndarray,
    w_sum: float,
) -> np.ndarray:
    """Weighted-standardize one vector with the same zero-variance convention as R2 JMI."""
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    mean = float(np.dot(w, x_arr) / w_sum)
    centered = x_arr - mean
    var = float(np.dot(w, centered * centered) / w_sum)
    std = np.sqrt(var) if var > 1e-12 else 1.0
    return centered / std


def _prepare_r2_joint_mi_state(
    X_full: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Precompute standardized features and feature-target correlations for R2 JMI."""
    w_arr = np.asarray(w, dtype=np.float64).ravel()
    w_sum = float(w_arr.sum())
    if w_sum <= 0.0:
        raise ValueError("w must have positive total weight")

    Z = _weighted_standardize_2d(X_full, w_arr, w_sum)
    y_s = _weighted_standardize_1d(y, w_arr, w_sum)
    r_y = Z.T @ (w_arr * y_s) / w_sum
    return Z, r_y, w_arr, w_sum


def _r2_joint_mi_scores_from_correlations(
    r_ys: float,
    r_yf: np.ndarray,
    r_fs: np.ndarray,
) -> np.ndarray:
    """Vectorized R2 JMI formula from weighted correlations."""
    denom = 1.0 - r_fs * r_fs
    r2 = np.empty_like(r_yf, dtype=np.float64)

    near_singular = denom < 1e-8
    r2[near_singular] = r_ys * r_ys

    stable = ~near_singular
    if np.any(stable):
        a = r_yf[stable] - r_ys * r_fs[stable]
        r2[stable] = r_ys * r_ys + (a * a) / denom[stable]

    r2 = np.clip(r2, 0.0, 0.99999)
    return -0.5 * np.log(1.0 - r2)


def _r2_joint_mi_indexed_from_state(
    Z_full: np.ndarray,
    r_y: np.ndarray,
    cand_idx: np.ndarray,
    selected_idx: int,
    w: np.ndarray,
    w_sum: float,
) -> np.ndarray:
    """R2 JMI for candidate indices using precomputed standardized features."""
    cand_idx = np.asarray(cand_idx, dtype=np.int64).ravel()
    if cand_idx.size == 0:
        return np.empty(0, dtype=np.float64)
    if w_sum <= 0.0:
        return np.zeros(cand_idx.size, dtype=np.float64)

    selected_idx = int(selected_idx)
    r_ys = float(r_y[selected_idx])
    r_yf = r_y[cand_idx]
    weighted_selected = w * Z_full[:, selected_idx]
    r_fs_all = Z_full.T @ weighted_selected / w_sum
    r_fs = r_fs_all[cand_idx]
    return _r2_joint_mi_scores_from_correlations(r_ys, r_yf, r_fs)


def binned_joint_mi_indexed(
    X_full: np.ndarray,
    cand_idx: np.ndarray,
    selected: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_bins: int = 10,
    y_kind: Literal["discrete", "continuous"] = "continuous",
) -> np.ndarray:
    """Binned joint MI with streaming per-candidate binning (low-memory).

    Convenience wrapper; for repeated scoring, prebin once with
    ``quantile_bin_matrix`` + ``binned_joint_mi_indexed_prebinned``.
    """
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
        y_arr = np.asarray(y)
        if (
            np.issubdtype(y_arr.dtype, np.integer)
            and y_arr.size > 0
            and y_arr.min() >= 0
            and y_arr.max() <= 200_000
        ):
            y_binned = y_arr.astype(np.int64, copy=False)
        else:
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


def binned_joint_mi_indexed_prebinned(
    X_binned: np.ndarray,
    cand_idx: np.ndarray,
    s_binned: np.ndarray,
    y_binned: np.ndarray,
    w: np.ndarray,
    n_bins: int,
    n_y_bins: int,
) -> np.ndarray:
    """Binned joint MI with prebinned candidate matrix and target."""
    X_binned = np.asarray(X_binned)
    if X_binned.ndim != 2:
        raise ValueError("X_binned must be 2D")

    cand_idx = np.asarray(cand_idx, dtype=np.int64).ravel()
    m = cand_idx.size
    if m == 0:
        return np.empty(0, dtype=np.float64)

    w = np.asarray(w, dtype=np.float64).ravel()
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return np.zeros(m, dtype=np.float64)

    s_binned = np.asarray(s_binned, dtype=np.int64).ravel()
    y_binned = np.asarray(y_binned, dtype=np.int64).ravel()
    if X_binned.shape[0] != s_binned.size or s_binned.size != y_binned.size:
        raise ValueError("Row mismatch between X_binned, s_binned, y_binned")

    h_y = _weighted_entropy_from_codes(y_binned, w, n_states=n_y_bins, w_sum=w_sum)

    fs_states = n_bins * n_bins
    fsy_states = fs_states * n_y_bins

    scores = np.empty(m, dtype=np.float64)

    for ci in range(m):
        j = int(cand_idx[ci])
        f_binned = X_binned[:, j]

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _weighted_entropy_from_codes(fs_binned, w, n_states=fs_states, w_sum=w_sum)
        h_fsy = _weighted_entropy_from_codes(fsy_binned, w, n_states=fsy_states, w_sum=w_sum)

        scores[ci] = max(0.0, h_fs + h_y - h_fsy)

    return scores


def quantile_bin_matrix(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin each column of X into integer codes."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    out = np.empty((n, p), dtype=np.int32)
    for j in range(p):
        out[:, j] = _quantile_bin(X[:, j], n_bins)
    return out


def quantile_bin_matrix_indexed(
    X_full: np.ndarray,
    cand_idx: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Quantile-bin candidate columns into an (n, m) int32 matrix.

    WARNING: allocates O(n*m) memory; intended for smaller problems or when
    you explicitly want prebinned matrices.
    """
    X_full = np.asarray(X_full)
    if X_full.ndim != 2:
        raise ValueError("X_full must be 2D")
    cand_idx = np.asarray(cand_idx, dtype=np.int64).ravel()
    n = X_full.shape[0]
    m = cand_idx.size
    out = np.empty((n, m), dtype=np.int32)
    for i, j in enumerate(cand_idx):
        out[:, i] = _quantile_bin(X_full[:, int(j)], n_bins)
    return out


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
