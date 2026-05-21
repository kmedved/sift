"""Gaussian copula transforms and caching for fast selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.special import ndtri

from sift._numba import njit_optional_cache

RankBackend = Literal["serial", "processes"]


@dataclass
class FeatureCache:
    """Cached feature data for multi-target selection."""

    Z: np.ndarray
    Rxx: np.ndarray | None
    valid_cols: np.ndarray
    row_idx: np.ndarray
    sample_weight: np.ndarray
    n_rows_original: int
    feature_names: list[str] | None = None


def build_cache(
    X,
    sample_weight: np.ndarray | None = None,
    subsample: int | None = 50_000,
    random_state: int = 0,
    compute_Rxx: bool = False,
    min_std: float = 1e-12,
    n_jobs: int = 1,
    rank_backend: RankBackend = "serial",
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
    if float(ws.sum()) <= 0.0:
        raise ValueError("Subsample has zero total weight; check sample_weight/subsample.")
    Xs = mean_impute(Xs, copy=False)

    stds = np.std(Xs, axis=0)
    valid_mask = stds > min_std
    valid_cols = np.where(valid_mask)[0]
    Xs = Xs[:, valid_mask]
    if Xs.shape[1] == 0:
        raise ValueError("All features were filtered out (constant or invalid). Cannot build cache.")

    Z = weighted_rank_gauss_2d(Xs, ws, n_jobs=n_jobs, rank_backend=rank_backend)

    Rxx = weighted_correlation_matrix(Z, ws) if compute_Rxx else None

    return FeatureCache(
        Z=Z.astype(np.float32),
        Rxx=Rxx.astype(np.float32) if Rxx is not None else None,
        valid_cols=valid_cols,
        row_idx=row_idx,
        sample_weight=ws.astype(np.float32),
        n_rows_original=n,
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

    order = np.argsort(x_valid, kind="mergesort")
    x_sorted = x_valid[order]
    w_sorted = w_valid[order]

    total = float(w_sorted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(x, dtype=np.float32)

    ranks = np.empty_like(w_sorted, dtype=np.float64)
    cum_weight = 0.0
    start = 0
    while start < m:
        stop = start + 1
        while stop < m and x_sorted[stop] == x_sorted[start]:
            stop += 1
        block_weight = float(w_sorted[start:stop].sum())
        ranks[start:stop] = cum_weight + 0.5 * block_weight
        cum_weight += block_weight
        start = stop

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


def _validate_rank_backend(rank_backend: RankBackend, n_jobs: int) -> RankBackend:
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    if rank_backend not in ("serial", "processes"):
        raise ValueError("rank_backend must be one of 'serial' or 'processes'")
    return rank_backend


def _weighted_rank_gauss_chunk(
    X: np.ndarray,
    w: np.ndarray,
    start: int,
    stop: int,
) -> tuple[int, np.ndarray]:
    chunk = np.empty((X.shape[0], stop - start), dtype=np.float32)
    for offset, j in enumerate(range(start, stop)):
        chunk[:, offset] = weighted_rank_gauss_1d(X[:, j], w)
    return start, chunk


def weighted_rank_gauss_2d(
    X: np.ndarray,
    w: np.ndarray,
    *,
    n_jobs: int = 1,
    rank_backend: RankBackend = "serial",
) -> np.ndarray:
    rank_backend = _validate_rank_backend(rank_backend, n_jobs)
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    n_workers = max(1, min(p, effective_n_jobs(n_jobs))) if p else 1
    if rank_backend == "serial" or n_workers <= 1:
        for j in range(p):
            Z[:, j] = weighted_rank_gauss_1d(X[:, j], w)
        return Z

    bounds = np.linspace(0, p, n_workers + 1, dtype=np.int64)
    chunks = Parallel(n_jobs=n_jobs, prefer="processes", max_nbytes="16M", batch_size=1)(
        delayed(_weighted_rank_gauss_chunk)(X, w, int(bounds[i]), int(bounds[i + 1]))
        for i in range(n_workers)
        if bounds[i] < bounds[i + 1]
    )
    for start, chunk in chunks:
        Z[:, start : start + chunk.shape[1]] = chunk
    return Z


@njit_optional_cache(cache=True)
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


@njit_optional_cache(cache=True)
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


@njit_optional_cache(cache=True)
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
