"""Gaussian copula transforms and caching for fast selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.special import ndtri

from sift._numba import njit_optional_cache

RankBackend = Literal["serial", "threads", "processes"]


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
    feature_names_are_synthetic: bool = False


def build_cache(
    X,
    sample_weight: np.ndarray | None = None,
    subsample: int | None = 50_000,
    random_state: int = 0,
    compute_Rxx: bool = False,
    min_std: float = 0.0,
    n_jobs: int = 1,
    rank_backend: RankBackend = "serial",
) -> FeatureCache:
    """Build feature cache for multi-target selection."""
    from sift._impute import mean_impute
    from sift._preprocess import (
        ensure_weights,
        extract_feature_names,
        reject_datetime_like_features,
        to_numpy,
    )

    feature_names = extract_feature_names(X)
    feature_names_are_synthetic = feature_names is None
    reject_datetime_like_features(X)
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

    if not np.isfinite(min_std) or min_std < 0.0:
        raise ValueError("min_std must be finite and non-negative")

    w = ensure_weights(sample_weight, n, normalize=True)
    positive = np.flatnonzero(w > 0.0)
    if subsample is not None and positive.size > subsample:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(positive, size=subsample, replace=False)
    else:
        row_idx = positive

    Xs = X_arr[row_idx]
    ws = w[row_idx]
    weight_mean = float(ws.mean())
    if not np.isfinite(weight_mean) or weight_mean <= 0.0:
        raise ValueError("Subsample has zero total weight; check sample_weight/subsample.")
    ws = ws / weight_mean
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
        Z=Z.astype(np.float32, copy=False),
        Rxx=Rxx.astype(np.float32) if Rxx is not None else None,
        valid_cols=valid_cols,
        row_idx=row_idx,
        sample_weight=ws.astype(np.float32),
        n_rows_original=n,
        feature_names=feature_names,
        feature_names_are_synthetic=feature_names_are_synthetic,
    )


def _standardized_gauss_scores(ranks: np.ndarray, w_sorted: np.ndarray, total: float) -> np.ndarray:
    """Map weighted mid-ranks to weighted-standardized Gaussian scores."""
    u = ranks / total
    np.clip(u, 1e-6, 1 - 1e-6, out=u)
    z = ndtri(u)
    z_mean = np.dot(w_sorted, z) / total
    z -= z_mean
    z_var = np.dot(w_sorted, z * z) / total
    z_std = np.sqrt(z_var) if z_var > 1e-12 else 1.0
    z /= z_std
    return z


def _uniform_gauss_template(m: int, weight: float = 1.0) -> np.ndarray:
    """Standardized Gaussian scores for ``m`` untied rows with equal weights.

    With constant weights and no ties the weighted mid-rank of the ``i``-th
    sorted value is ``(i + 0.5) * w`` out of ``m * w``, so the transformed
    column is the same vector for every feature; only the sort order differs.
    The arithmetic mirrors the general path exactly so the template is
    bitwise identical to a per-column computation.
    """
    w_sorted = np.full(m, float(weight), dtype=np.float64)
    return _standardized_gauss_scores(_untied_ranks(w_sorted), w_sorted, float(w_sorted.sum()))


def _untied_ranks(w_sorted: np.ndarray) -> np.ndarray:
    """Weighted mid-ranks for untied sorted rows (same arithmetic as tie blocks)."""
    m = w_sorted.shape[0]
    ranks = np.empty(m, dtype=np.float64)
    ranks[0] = 0.0
    if m > 1:
        np.cumsum(w_sorted[:-1], out=ranks[1:])
    ranks += 0.5 * w_sorted
    return ranks


def weighted_rank_gauss_1d(
    x: np.ndarray,
    w: np.ndarray,
    *,
    _template: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted rank-based Gaussian transform.

    Ties receive the same weighted mid-rank. Weights are accumulated in
    float64 regardless of the input dtype. ``_template`` is an internal
    fast-path hook used by :func:`weighted_rank_gauss_2d`: when all weights are
    equal and a column has no ties or missing values, the standardized scores
    of the sorted column equal a precomputed template and only need to be
    scattered back into row order.
    """
    mask = np.isfinite(x)
    m = int(mask.sum())
    if m <= 1:
        return np.zeros_like(x, dtype=np.float32)

    w64 = np.asarray(w, dtype=np.float64)
    all_finite = m == x.shape[0]
    x_valid = x if all_finite else x[mask]
    w_valid = w64 if all_finite else w64[mask]

    # Tie blocks make the result independent of the within-tie order (up to
    # float64 summation order of tied weights, invisible after the float32
    # cast), so the faster default introsort is safe here.
    order = np.argsort(x_valid, kind="quicksort")
    x_sorted = x_valid[order]

    block_start = np.empty(m, dtype=bool)
    block_start[0] = True
    np.not_equal(x_sorted[1:], x_sorted[:-1], out=block_start[1:])
    no_ties = bool(block_start.all())

    if no_ties and all_finite and _template is not None and _template.shape[0] == m:
        out = np.empty(x.shape[0], dtype=np.float32)
        out[order] = _template
        return out

    w_sorted = w_valid[order]
    total = float(w_sorted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(x, dtype=np.float32)

    if no_ties:
        ranks = _untied_ranks(w_sorted)
    else:
        starts = np.flatnonzero(block_start)
        block_weights = np.add.reduceat(w_sorted, starts)
        cum_before = np.empty_like(block_weights, dtype=np.float64)
        cum_before[0] = 0.0
        if block_weights.shape[0] > 1:
            np.cumsum(block_weights[:-1], out=cum_before[1:])
        block_ranks = cum_before + 0.5 * block_weights
        block_lengths = np.diff(np.append(starts, m))
        ranks = np.repeat(block_ranks, block_lengths)

    z = _standardized_gauss_scores(ranks, w_sorted, total)

    if all_finite:
        out = np.empty(x.shape[0], dtype=np.float32)
        out[order] = z
        return out
    scattered = np.empty(m, dtype=np.float32)
    scattered[order] = z
    out = np.zeros(x.shape[0], dtype=np.float32)
    out[mask] = scattered
    return out


def _validate_rank_backend(rank_backend: RankBackend, n_jobs: int) -> RankBackend:
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    if rank_backend not in ("serial", "threads", "processes"):
        raise ValueError("rank_backend must be one of 'serial', 'threads', or 'processes'")
    return rank_backend


def _rank_gauss_template(w: np.ndarray, n: int) -> np.ndarray | None:
    """Return the shared no-tie template when all weights are equal."""
    w_arr = np.asarray(w)
    if n <= 1 or w_arr.shape[0] != n:
        return None
    if not np.all(w_arr == w_arr[0]) or not np.isfinite(w_arr[0]) or w_arr[0] <= 0.0:
        return None
    return _uniform_gauss_template(n, float(w_arr[0])).astype(np.float32)


def _weighted_rank_gauss_chunk(
    X: np.ndarray,
    w: np.ndarray,
    start: int,
    stop: int,
    template: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    chunk = np.empty((X.shape[0], stop - start), dtype=np.float32)
    for offset, j in enumerate(range(start, stop)):
        chunk[:, offset] = weighted_rank_gauss_1d(X[:, j], w, _template=template)
    return start, chunk


def _weighted_rank_gauss_into(
    Z: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    start: int,
    stop: int,
    template: np.ndarray | None,
) -> None:
    for j in range(start, stop):
        Z[:, j] = weighted_rank_gauss_1d(X[:, j], w, _template=template)


def weighted_rank_gauss_2d(
    X: np.ndarray,
    w: np.ndarray,
    *,
    n_jobs: int = 1,
    rank_backend: RankBackend = "serial",
) -> np.ndarray:
    """Column-wise weighted rank-Gaussian transform.

    ``rank_backend="threads"`` parallelizes columns with threads (the sort and
    the Gaussian quantile release the GIL); ``"processes"`` uses joblib process
    workers, which costs a copy of ``X`` per worker.
    """
    rank_backend = _validate_rank_backend(rank_backend, n_jobs)
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    template = _rank_gauss_template(w, n)
    n_workers = max(1, min(p, effective_n_jobs(n_jobs))) if p else 1
    if rank_backend == "serial" or n_workers <= 1:
        _weighted_rank_gauss_into(Z, X, w, 0, p, template)
        return Z

    bounds = np.linspace(0, p, n_workers + 1, dtype=np.int64)
    if rank_backend == "threads":
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_weighted_rank_gauss_into)(Z, X, w, int(bounds[i]), int(bounds[i + 1]), template)
            for i in range(n_workers)
            if bounds[i] < bounds[i + 1]
        )
        return Z

    chunks = Parallel(n_jobs=n_jobs, prefer="processes", max_nbytes="16M", batch_size=1)(
        delayed(_weighted_rank_gauss_chunk)(X, w, int(bounds[i]), int(bounds[i + 1]), template)
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
    sqrt_w = np.sqrt(w)

    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        # Scale rows by sqrt(w) so the weighted Gram is a symmetric rank-k
        # update (Zw' Zw), which BLAS computes with half the flops of Z' (w Z).
        Zw = Z[start:stop] * sqrt_w[start:stop, None]
        if Zw.dtype != np.float64:
            Zw = Zw.astype(np.float64, copy=False)
        # Some CatBoost/OpenMP builds leave stale floating-point status flags
        # that NumPy reports on the next finite BLAS matmul. Suppress only the
        # operation-level warning and validate the accumulated Gram below so
        # genuine overflow/non-finite output still fails explicitly.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            R += Zw.T @ Zw

    if not np.isfinite(R).all():
        raise FloatingPointError("Weighted correlation Gram was non-finite")

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
def _weighted_corr_with_vector_numba(Z: np.ndarray, zy: np.ndarray, w: np.ndarray) -> np.ndarray:
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


def weighted_corr_with_vector_blas(
    Z: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    *,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Uncentered weighted Z'y moments using chunked BLAS."""
    Z_arr = np.asarray(Z)
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 2D")
    n, p = Z_arr.shape
    zy64 = np.asarray(zy, dtype=np.float64).ravel()
    w64 = np.asarray(w, dtype=np.float64).ravel()
    if zy64.shape[0] != n:
        raise ValueError("zy length must match Z rows")
    if w64.shape[0] != n:
        raise ValueError("w length must match Z rows")
    if not np.isfinite(w64).all():
        raise ValueError("Non-finite weights are not allowed")
    if np.any(w64 < 0):
        raise ValueError("Negative weights are not allowed")
    w_sum = float(w64.sum())
    if w_sum <= 0.0:
        raise ValueError("Weights must sum to > 0")

    acc = np.zeros(p, dtype=np.float64)
    wy = w64 * zy64
    batch_size = max(1, int(batch_size))
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        Zb = Z_arr[start:stop]
        if Zb.dtype != np.float64:
            Zb = Zb.astype(np.float64, copy=False)
        acc += Zb.T @ wy[start:stop]

    r = acc / w_sum
    np.clip(r, -0.999999, 0.999999, out=r)
    return r.astype(np.float32)


def weighted_corr_with_vector(
    Z: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    *,
    backend: Literal["auto", "blas", "numba"] = "auto",
    batch_size: int = 50_000,
) -> np.ndarray:
    """
    Uncentered weighted correlation with one standardized vector.

    This computes weighted second moments; callers rely on cache columns and
    ``zy`` being weighted-standardized before calling.
    """
    Z_arr = np.asarray(Z)
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 2D")
    zy_arr = np.asarray(zy).ravel()
    w_arr = np.asarray(w).ravel()
    if zy_arr.shape[0] != Z_arr.shape[0]:
        raise ValueError("zy length must match Z rows")
    if w_arr.shape[0] != Z_arr.shape[0]:
        raise ValueError("w length must match Z rows")
    if backend == "auto":
        n, p = Z_arr.shape
        backend = "blas" if n * p >= 1_000_000 else "numba"
    if backend == "blas":
        return weighted_corr_with_vector_blas(Z_arr, zy_arr, w_arr, batch_size=batch_size)
    if backend == "numba":
        return _weighted_corr_with_vector_numba(Z_arr, zy_arr, w_arr)
    raise ValueError(f"Unknown backend: {backend}")


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

    order = candidates[np.lexsort((candidates, -scores[candidates]))]
    keep = []
    active = np.ones(len(order), dtype=bool)

    for i, fi in enumerate(order):
        if not active[i]:
            continue
        keep.append(fi)
        later = order[i + 1 :]
        active[i + 1 :] &= ~(np.abs(Rxx[fi, later]) >= threshold)

    return np.array(keep, dtype=np.int64)
