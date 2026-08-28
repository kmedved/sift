"""Joint mutual information estimators: I(f, s; y)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from sift._numba import njit_optional_cache


_MAX_EXACT_FLOAT_INTEGER = float(2**53 - 1)


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


def _canonical_binned_weights(
    weights: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return scale-free edge/entropy weights for one binned-MI call.

    Continuous weights are max-normalized.  Frequency-like integer ratios are
    converted to their primitive counts so entropy accumulation is stable and
    agrees with the documented common-factor reduction policy.
    """
    raw = np.asarray(weights, dtype=np.float64).ravel()
    if raw.size != n:
        raise ValueError(f"weights has {raw.size} elements but expected {n}")
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("weights must be finite and non-negative")
    positive = raw > 0.0
    if not np.any(positive):
        return raw, raw, 0.0

    edge_weights = raw / float(np.max(raw[positive]))
    entropy_weights = edge_weights.copy()
    atomic_mass = _frequency_atomic_mass(edge_weights[positive])
    if atomic_mass is not None:
        ratios = edge_weights[positive] / atomic_mass
        nearest = np.rint(ratios)
        counts_are_safe = (
            np.isfinite(ratios).all()
            and np.all(ratios >= 1.0)
            and float(np.max(ratios)) <= _MAX_EXACT_FLOAT_INTEGER
            and np.all(
                np.abs(ratios - nearest)
                <= 1e-7 * np.maximum(1.0, np.abs(ratios))
            )
            and float(nearest.sum()) <= _MAX_EXACT_FLOAT_INTEGER
        )
        if counts_are_safe:
            entropy_weights[positive] = nearest

    return edge_weights, entropy_weights, float(entropy_weights.sum())


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
    edge_w, entropy_w, w_sum = _canonical_binned_weights(w, n)
    if w_sum <= 0.0:
        return np.zeros(p, dtype=np.float64)

    s_binned = _quantile_bin_for_weights(selected, n_bins, edge_w)

    if y_kind == "discrete":
        y_binned = _compact_discrete_target_codes(y)
        n_y_bins = int(y_binned.max()) + 1 if y_binned.size else 1
    else:
        y_binned = _quantile_bin_for_weights(y, n_bins, edge_w)
        n_y_bins = n_bins

    h_y = _weighted_entropy_from_codes(y_binned, entropy_w, n_states=n_y_bins, w_sum=w_sum)

    fs_states = n_bins * n_bins
    fsy_states = fs_states * n_y_bins

    scores = np.empty(p, dtype=np.float64)

    for j in range(p):
        f_binned = _quantile_bin_for_weights(candidates[:, j], n_bins, edge_w)

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _weighted_entropy_from_codes(fs_binned, entropy_w, n_states=fs_states, w_sum=w_sum)
        h_fsy = _weighted_entropy_from_codes(fsy_binned, entropy_w, n_states=fsy_states, w_sum=w_sum)

        scores[j] = max(0.0, h_fs + h_y - h_fsy)

    return scores


@njit_optional_cache(cache=True)
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
    y_std = np.sqrt(y_var) if y_var > 0.0 else 1.0

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
    s_std = np.sqrt(s_var) if s_var > 0.0 else 1.0

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
        f_std = np.sqrt(f_var) if f_var > 0.0 else 1.0

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
@njit_optional_cache(cache=True)
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
    y_std = np.sqrt(y_var) if y_var > 0.0 else 1.0

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
    s_std = np.sqrt(s_var) if s_var > 0.0 else 1.0

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
        f_std = np.sqrt(f_var) if f_var > 0.0 else 1.0

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
    std = np.where(var > 0.0, np.sqrt(var), 1.0)
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
    std = np.sqrt(var) if var > 0.0 else 1.0
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

    edge_w, entropy_w, w_sum = _canonical_binned_weights(w, X_full.shape[0])
    if w_sum <= 0.0:
        return np.zeros(m, dtype=np.float64)

    s_binned = _quantile_bin_for_weights(selected, n_bins, edge_w)

    if y_kind == "discrete":
        y_binned = _compact_discrete_target_codes(y)
        n_y_bins = int(y_binned.max()) + 1 if y_binned.size else 1
    else:
        y_binned = _quantile_bin_for_weights(y, n_bins, edge_w)
        n_y_bins = n_bins

    h_y = _weighted_entropy_from_codes(y_binned, entropy_w, n_states=n_y_bins, w_sum=w_sum)

    fs_states = n_bins * n_bins
    fsy_states = fs_states * n_y_bins

    scores = np.empty(m, dtype=np.float64)

    for ci in range(m):
        j = int(cand_idx[ci])
        f_binned = _quantile_bin_for_weights(X_full[:, j], n_bins, edge_w)

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _weighted_entropy_from_codes(fs_binned, entropy_w, n_states=fs_states, w_sum=w_sum)
        h_fsy = _weighted_entropy_from_codes(fsy_binned, entropy_w, n_states=fsy_states, w_sum=w_sum)

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

    _edge_w, entropy_w, w_sum = _canonical_binned_weights(
        w, X_binned.shape[0]
    )
    if w_sum <= 0.0:
        return np.zeros(m, dtype=np.float64)

    s_binned = np.asarray(s_binned, dtype=np.int64).ravel()
    y_binned = np.asarray(y_binned, dtype=np.int64).ravel()
    if X_binned.shape[0] != s_binned.size or s_binned.size != y_binned.size:
        raise ValueError("Row mismatch between X_binned, s_binned, y_binned")

    h_y = _weighted_entropy_from_codes(y_binned, entropy_w, n_states=n_y_bins, w_sum=w_sum)

    fs_states = n_bins * n_bins
    fsy_states = fs_states * n_y_bins

    scores = np.empty(m, dtype=np.float64)

    for ci in range(m):
        j = int(cand_idx[ci])
        f_binned = X_binned[:, j]

        fs_binned = f_binned * n_bins + s_binned
        fsy_binned = fs_binned * n_y_bins + y_binned

        h_fs = _weighted_entropy_from_codes(fs_binned, entropy_w, n_states=fs_states, w_sum=w_sum)
        h_fsy = _weighted_entropy_from_codes(fsy_binned, entropy_w, n_states=fsy_states, w_sum=w_sum)

        scores[ci] = max(0.0, h_fs + h_y - h_fsy)

    return scores


def quantile_bin_matrix(
    X: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Quantile-bin each column of X into integer codes.

    If ``weights`` is provided, each column's edges are weighted quantiles;
    zero-weight rows are excluded from edge computation.  Omitting weights
    retains the original unweighted path.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    w = _validate_bin_weights(weights, n) if weights is not None else None
    out = np.empty((n, p), dtype=np.int32)
    for j in range(p):
        out[:, j] = _quantile_bin_for_weights(X[:, j], n_bins, w)
    return out


def quantile_bin_matrix_indexed(
    X_full: np.ndarray,
    cand_idx: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None = None,
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
    w = _validate_bin_weights(weights, n) if weights is not None else None
    out = np.empty((n, m), dtype=np.int32)
    for i, j in enumerate(cand_idx):
        out[:, i] = _quantile_bin_for_weights(X_full[:, int(j)], n_bins, w)
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

    y_centered = y - y.mean()
    y_std = y_centered.std()
    y_s = y_centered / y_std if y_std > 0.0 else np.zeros_like(y_centered)
    s_centered = selected - selected.mean()
    s_std = s_centered.std()
    s_s = s_centered / s_std if s_std > 0.0 else np.zeros_like(s_centered)
    Y_marginal = y_s.reshape(-1, 1)
    tree_y = cKDTree(Y_marginal)

    for j in range(p):
        f = candidates[:, j]
        f_centered = f - f.mean()
        f_std = f_centered.std()
        f_s = f_centered / f_std if f_std > 0.0 else np.zeros_like(f_centered)

        X_joint = np.column_stack([f_s, s_s])
        XY_full = np.column_stack([f_s, s_s, y_s])

        tree_full = cKDTree(XY_full)
        tree_x = cKDTree(X_joint)

        dists, _ = tree_full.query(XY_full, k=k + 1, p=np.inf)
        eps = np.nextafter(dists[:, -1], 0.0)

        n_x = _safe_count_neighbors(tree_x, X_joint, eps, n)
        n_y = _safe_count_neighbors(tree_y, Y_marginal, eps, n)

        n_x = np.maximum(n_x, 0)
        n_y = np.maximum(n_y, 0)

        mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
        scores[j] = max(mi, 0.0)

    return scores


def _validate_bin_weights(weights: np.ndarray, n: int) -> np.ndarray:
    """Validate weights used only for weighted quantile edge construction."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n:
        raise ValueError(f"weights has {w.size} elements but expected {n}")
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(w > 0.0):
        raise ValueError("weights must contain at least one positive value")
    return w


def _quantile_bin_for_weights(
    x: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Quantile-bin using weighted edges while retaining the unweighted fast path."""
    if weights is None or np.all(weights == weights[0]):
        return _quantile_bin(x, n_bins)
    return _quantile_bin(x, n_bins, weights)


def _weighted_percentile(
    x: np.ndarray,
    percentiles: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Scale-invariant weighted linear percentiles.

    Frequency-like weights are reduced to a scale-free atomic mass and follow
    NumPy's linear percentile semantics without physically repeating rows.
    General continuous weights use weighted-CDF midpoint interpolation. Both
    paths are invariant to multiplying every weight by the same positive
    constant, and zero-weight observations are ignored entirely.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    w_arr = np.asarray(weights, dtype=np.float64).ravel()
    if x_arr.size != w_arr.size:
        raise ValueError("x and weights must have the same length")
    positive = w_arr > 0.0
    if not np.any(positive):
        raise ValueError("weights must contain at least one positive value")
    x_pos = x_arr[positive]
    w_pos = w_arr[positive]
    order = np.argsort(x_pos, kind="mergesort")
    values = x_pos[order]
    sorted_weights = w_pos[order]
    if values.size == 1 or np.ptp(values) == 0.0:
        return np.full(np.asarray(percentiles).shape, values[0], dtype=np.float64)

    quantiles = np.clip(np.asarray(percentiles, dtype=np.float64) / 100.0, 0.0, 1.0)

    def stabilize_edges(result: np.ndarray) -> np.ndarray:
        """Remove cumulative-sum reversals without changing query order."""
        order = np.argsort(quantiles, kind="mergesort")
        monotone = np.maximum.accumulate(result[order])
        stabilized = np.empty_like(result)
        stabilized[order] = monotone
        return stabilized

    # Normalize before cumulative arithmetic. Besides avoiding overflow, this
    # makes multiplication of every sample weight by a positive constant a
    # numerically stable no-op even when weights span many orders of magnitude.
    normalized_weights = sorted_weights / float(np.max(sorted_weights))
    total = float(normalized_weights.sum())
    atomic_mass = _frequency_atomic_mass(normalized_weights)
    if atomic_mass is None:
        cdf_midpoints = (
            np.cumsum(normalized_weights) - 0.5 * normalized_weights
        ) / total
        result = np.interp(
            quantiles,
            cdf_midpoints,
            values,
            left=values[0],
            right=values[-1],
        )
        # Interpolation is mathematically monotone, but cumulative floating
        # point arithmetic can leave adjacent edges a few ulps out of order.
        # Binning requires a monotone edge vector.
        result = stabilize_edges(result)
        return np.where(
            quantiles <= 0.0,
            values[0],
            np.where(quantiles >= 1.0, values[-1], result),
        )

    # For manageable primitive integer ratios, use integer rank arithmetic.
    # This is exactly NumPy's linear percentile definition on the conceptual
    # repeated rows, without expanding those rows or allowing cumulative
    # floating-point ulps to move a tied observation across an edge.
    ratios = normalized_weights / atomic_mass
    nearest = np.rint(ratios)
    counts_are_safe = (
        np.isfinite(ratios).all()
        and np.all(ratios >= 1.0)
        and float(np.max(ratios)) <= _MAX_EXACT_FLOAT_INTEGER
        and np.all(
            np.abs(ratios - nearest)
            <= 1e-7 * np.maximum(1.0, np.abs(ratios))
        )
    )
    total_count_float = float(nearest.sum()) if counts_are_safe else np.inf
    if counts_are_safe and total_count_float <= _MAX_EXACT_FLOAT_INTEGER:
        counts = nearest.astype(np.int64)
        total_count = int(counts.sum(dtype=np.int64))
        ranks = np.clip(
            quantiles * (total_count - 1),
            0.0,
            float(total_count - 1),
        )
        lower_rank = np.floor(ranks).astype(np.int64)
        upper_rank = np.ceil(ranks).astype(np.int64)
        cumulative_counts = np.cumsum(counts, dtype=np.int64)
        lower_idx = np.searchsorted(cumulative_counts, lower_rank, side="right")
        upper_idx = np.searchsorted(cumulative_counts, upper_rank, side="right")
        result = values[lower_idx] + (ranks - lower_rank) * (
            values[upper_idx] - values[lower_idx]
        )
        result = stabilize_edges(result)
        return np.where(
            quantiles <= 0.0,
            values[0],
            np.where(quantiles >= 1.0, values[-1], result),
        )

    positions = quantiles * max(total - atomic_mass, 0.0)
    cumulative = np.cumsum(normalized_weights)
    # Each observation contributes a constant run followed by one atomic-mass
    # linear transition to the next value. This is the compact equivalent of
    # expanding primitive integer frequency ratios (after removing a common
    # global factor) and passing them to np.percentile(..., method="linear").
    transition = cumulative - atomic_mass
    # Select the latest transition that has started; ``-1`` handles the first
    # run before its transition, which is clipped back to the first value.
    idx = np.searchsorted(transition, positions, side="right") - 1
    idx = np.clip(idx, 0, values.size - 1)
    has_next = idx < values.size - 1
    next_idx = np.minimum(idx + 1, values.size - 1)
    fraction = np.zeros_like(positions)
    fraction[has_next] = np.clip(
        (positions[has_next] - transition[idx[has_next]]) / atomic_mass,
        0.0,
        1.0,
    )
    # A percentile landing exactly on a transition can be represented as a
    # tiny positive/negative interpolation fraction after cumulative sums.
    # Snap those boundary cases to the source value so tied observations are
    # assigned to the same bin as in the equivalent replicated data.
    boundary_tol = 64.0 * np.finfo(np.float64).eps
    fraction = np.where(
        np.abs(fraction) <= boundary_tol,
        0.0,
        np.where(np.abs(fraction - 1.0) <= boundary_tol, 1.0, fraction),
    )
    result = values[idx] + fraction * (values[next_idx] - values[idx])
    result = stabilize_edges(result)
    # Exact endpoints are part of the percentile contract. Explicitly pinning
    # them also avoids a one-ULP cumulative-sum difference choosing the
    # penultimate transition after an otherwise harmless weight rescaling.
    return np.where(
        quantiles <= 0.0,
        values[0],
        np.where(quantiles >= 1.0, values[-1], result),
    )


def _frequency_atomic_mass(weights: np.ndarray) -> float | None:
    """Return a scale-free row quantum for simple rational weight ratios."""
    minimum = float(np.min(weights))
    if not np.isfinite(minimum) or minimum <= 0.0:
        return None
    ratios = np.asarray(weights, dtype=np.float64) / minimum
    if not np.isfinite(ratios).all():
        return None
    for denominator in range(1, 65):
        scaled = ratios * denominator
        nearest = np.rint(scaled)
        tolerance = 1e-7 * np.maximum(1.0, np.abs(scaled))
        if np.all(np.abs(scaled - nearest) <= tolerance):
            return minimum / denominator
    return None


def _quantile_bin(
    x: np.ndarray,
    n_bins: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Quantile-based binning, optionally using weighted edges."""
    x_arr = np.asarray(x)
    if x_arr.size == 0:
        return np.zeros(len(x_arr), dtype=np.int32)
    if weights is None:
        if np.ptp(x_arr) == 0.0:
            return np.zeros(len(x_arr), dtype=np.int32)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x_arr, percentiles)
    else:
        w = _validate_bin_weights(weights, len(x_arr))
        positive = w > 0.0
        if not np.any(positive):
            return np.zeros(len(x_arr), dtype=np.int32)
        x_positive = np.asarray(x_arr[positive], dtype=np.float64)
        if x_positive.size == 1 or np.ptp(x_positive) == 0.0:
            return np.zeros(len(x_arr), dtype=np.int32)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = _weighted_percentile(x_arr, percentiles, w)
    return np.digitize(x_arr, bins[1:-1]).astype(np.int32)


def _factorize(x: np.ndarray) -> np.ndarray:
    """Convert to integer codes."""
    _, codes = np.unique(x, return_inverse=True)
    return codes.astype(np.int32)


def _compact_discrete_target_codes(y: np.ndarray) -> np.ndarray:
    """Return dense non-negative target codes for discrete MI targets."""
    y_arr = np.asarray(y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        return _factorize(y_arr)

    if y_arr.size == 0:
        return y_arr.astype(np.int64, copy=False)
    min_value = int(y_arr.min())
    max_value = int(y_arr.max())
    if min_value >= 0 and max_value < y_arr.size:
        return y_arr.astype(np.int64, copy=False)
    return _factorize(y_arr)


def _safe_count_neighbors(tree: cKDTree, points: np.ndarray, radii: np.ndarray, n: int) -> np.ndarray:
    """Count neighbors within per-point radii (excluding the point itself).

    SciPy's ``query_ball_point`` accepts an array of radii, so a single
    vectorized call replaces the former per-point Python loop.
    """
    counts = tree.query_ball_point(
        points, np.asarray(radii, dtype=np.float64), p=np.inf, return_length=True
    )
    return np.asarray(counts, dtype=np.int64) - 1
