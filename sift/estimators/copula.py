"""Gaussian copula transforms and caching for fast selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit
from scipy.special import ndtri


@dataclass
class FeatureCache:
    """Cached feature data for multi-target selection."""

    Z: np.ndarray
    Rxx: np.ndarray | None
    valid_cols: np.ndarray
    row_idx: np.ndarray
    sample_weight: np.ndarray
    feature_names: list[str] | None = None


def build_cache(
    X,
    sample_weight: np.ndarray | None = None,
    subsample: int | None = 50_000,
    random_state: int = 0,
    compute_Rxx: bool = False,
    min_std: float = 1e-12,
) -> FeatureCache:
    """Build feature cache for multi-target selection."""
    from sift._impute import mean_impute
    from sift._preprocess import ensure_weights, extract_feature_names, to_numpy

    feature_names = extract_feature_names(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Encode categorical columns before using gaussian estimator."
            )
    X_arr = to_numpy(X, dtype=np.float64)
    n, p = X_arr.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]

    w = ensure_weights(sample_weight, n, normalize=True)

    if subsample is not None and n > subsample:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(n, size=subsample, replace=False)
    else:
        row_idx = np.arange(n)

    Xs = X_arr[row_idx]
    ws = w[row_idx]
    Xs = mean_impute(Xs, copy=False)

    stds = np.std(Xs, axis=0)
    valid_mask = stds > min_std
    valid_cols = np.where(valid_mask)[0]
    Xs = Xs[:, valid_mask]
    if Xs.shape[1] == 0:
        raise ValueError("All features were filtered out (constant or invalid). Cannot build cache.")

    Z = weighted_rank_gauss_2d(Xs, ws)

    Rxx = weighted_correlation_matrix(Z, ws) if compute_Rxx else None

    return FeatureCache(
        Z=Z.astype(np.float32),
        Rxx=Rxx.astype(np.float32) if Rxx is not None else None,
        valid_cols=valid_cols,
        row_idx=row_idx,
        sample_weight=ws.astype(np.float32),
        feature_names=feature_names,
    )


def weighted_rank_gauss_1d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted rank-based Gaussian transform."""
    mask = np.isfinite(x)
    m = mask.sum()
    if m <= 1:
        return np.zeros_like(x, dtype=np.float32)

    x_valid = x[mask]
    w_valid = w[mask]

    order = np.argsort(x_valid)
    w_sorted = w_valid[order]

    cumsum = np.cumsum(w_sorted)
    total = cumsum[-1]

    ranks = np.empty_like(cumsum)
    ranks[0] = 0.5 * w_sorted[0]
    ranks[1:] = cumsum[:-1] + 0.5 * w_sorted[1:]

    u = np.clip(ranks / total, 1e-6, 1 - 1e-6)
    z = ndtri(u)

    z_mean = np.dot(w_sorted, z) / total
    z_centered = z - z_mean
    z_var = np.dot(w_sorted, z_centered ** 2) / total
    z_std = np.sqrt(z_var) if z_var > 1e-12 else 1.0
    z_standardized = z_centered / z_std

    inv_order = np.argsort(order)
    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = z_standardized[inv_order].astype(np.float32)
    return out


def weighted_rank_gauss_2d(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    for j in range(p):
        Z[:, j] = weighted_rank_gauss_1d(X[:, j], w)
    return Z


@njit(cache=True)
def weighted_correlation_matrix_numba(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Numba fallback for small matrices or when BLAS is slower."""
    n, p = Z.shape
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    R = np.empty((p, p), dtype=np.float64)
    for j in range(p):
        for k in range(j, p):
            val = 0.0
            for i in range(n):
                val += w[i] * Z[i, j] * Z[i, k]
            val /= w_sum
            val = max(-0.999999, min(0.999999, val))
            R[j, k] = val
            R[k, j] = val
        R[j, j] = 1.0
    return R


def weighted_correlation_matrix_blas(
    Z: np.ndarray,
    w: np.ndarray,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Weighted correlation matrix using chunked BLAS."""
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")

    Z = np.asarray(Z)
    n, p = Z.shape
    w = np.asarray(w, dtype=np.float64).ravel()

    if w.shape[0] != n:
        raise ValueError("w length must match Z rows")
    if not np.isfinite(w).all():
        raise ValueError("Non-finite weights are not allowed")
    if np.any(w < 0):
        raise ValueError("Negative weights are not allowed")

    w_sum = float(w.sum())
    if w_sum <= 0.0:
        raise ValueError("Weights must sum to > 0")

    R = np.zeros((p, p), dtype=np.float64)
    batch_size = max(1, int(batch_size))

    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        Zb = Z[start:stop]
        wb = w[start:stop]
        if Zb.dtype != np.float64:
            Zb = Zb.astype(np.float64, copy=False)
        R += Zb.T @ (Zb * wb[:, None])

    R /= w_sum
    R = 0.5 * (R + R.T)
    np.clip(R, -0.999999, 0.999999, out=R)
    np.fill_diagonal(R, 1.0)

    return R


def weighted_correlation_matrix(
    Z: np.ndarray,
    w: np.ndarray,
    *,
    backend: Literal["auto", "blas", "numba"] = "auto",
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Weighted correlation matrix.

    backend="blas" (default): chunked BLAS, fast for moderate/large p
    backend="numba": njit loop fallback, useful for tiny p or njit-call sites
    """
    if backend == "auto":
        Z0 = np.asarray(Z)
        n, p = Z0.shape
        backend = "numba" if p <= 32 and n <= 50_000 else "blas"
    if backend == "blas":
        return weighted_correlation_matrix_blas(Z, w, batch_size=batch_size)
    if backend == "numba":
        Z64 = np.asarray(Z, dtype=np.float64)
        w64 = np.asarray(w, dtype=np.float64).ravel()
        if Z64.shape[0] != w64.shape[0]:
            raise ValueError("w length must match Z rows")
        if not np.isfinite(w64).all():
            raise ValueError("Non-finite weights are not allowed")
        if np.any(w64 < 0):
            raise ValueError("Negative weights are not allowed")
        if float(w64.sum()) <= 0.0:
            raise ValueError("Weights must sum to > 0")
        return weighted_correlation_matrix_numba(Z64, w64)
    raise ValueError(f"Unknown backend: {backend}")


@njit(cache=True)
def weighted_corr_with_vector(Z: np.ndarray, zy: np.ndarray, w: np.ndarray) -> np.ndarray:
    n, p = Z.shape
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    r = np.empty(p, dtype=np.float32)
    for j in range(p):
        val = 0.0
        for i in range(n):
            val += w[i] * Z[i, j] * zy[i]
        r[j] = val / w_sum
    return np.clip(r, -0.999999, 0.999999)


@njit(cache=True)
def gaussian_mi_from_corr(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Gaussian MI approximation: I(X;Y) = -0.5 * log(1 - r²)."""
    r2 = np.clip(r * r, 0.0, 1.0 - eps)
    return -0.5 * np.log(1.0 - r2)


def greedy_corr_prune(
    candidates: np.ndarray,
    Rxx: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.95,
) -> np.ndarray:
    """Prune candidates with high correlation to higher-scoring features."""
    if len(candidates) == 0:
        return candidates

    order = candidates[np.argsort(-scores[candidates])]
    keep = []
    active = np.ones(len(order), dtype=bool)

    for i, fi in enumerate(order):
        if not active[i]:
            continue
        keep.append(fi)

        for j in range(i + 1, len(order)):
            if active[j]:
                fj = order[j]
                if np.abs(Rxx[fi, fj]) >= threshold:
                    active[j] = False

    return np.array(keep, dtype=np.int64)
