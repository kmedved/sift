"""Greedy selection loops and classic incremental selectors."""

from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
from typing import Literal, Optional

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from threadpoolctl import threadpool_limits

from sift._numba import njit_optional_cache
from sift._preprocess import validate_k

FLOOR = 1e-6
MrmrBackend = Literal["auto", "serial", "blas", "processes"]


def _single_threaded_r2_jmi(func):
    """Limit native pools for repeated R2-JMI correlation matvecs."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        estimator = kwargs.get(
            "mi_estimator",
            args[4] if len(args) > 4 else "r2",
        )
        context = threadpool_limits(limits=1) if estimator == "r2" else nullcontext()
        with context:
            return func(*args, **kwargs)

    return wrapped


def resolve_mrmr_backend(mrmr_backend: MrmrBackend, n_jobs: int) -> Literal["serial", "blas", "processes"]:
    """Validate and resolve mRMR backend options."""
    if n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    if mrmr_backend not in ("auto", "serial", "blas", "processes"):
        raise ValueError(
            "mrmr_backend must be one of 'auto', 'serial', 'blas', or 'processes'"
        )
    if mrmr_backend == "auto":
        # BLAS matvec redundancy updates beat the serial Numba loop by 3-10x on
        # realistic sizes and never pay process start-up or pickling costs, so
        # "auto" resolves to "blas" regardless of n_jobs. Pass "processes"
        # explicitly to opt into joblib workers.
        return "blas"
    return mrmr_backend


# =============================================================================
# Classic mRMR (incremental redundancy, O(p) memory)
# =============================================================================

@njit_optional_cache(cache=True)
def _standardize_columns_weighted(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n, p = X.shape
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    # Row-major sweeps keep access contiguous on C-ordered X while preserving
    # the per-column summation order (bitwise identical to a column loop).
    mean = np.zeros(p, dtype=np.float64)
    for i in range(n):
        wi = w[i]
        for j in range(p):
            mean[j] += wi * X[i, j]
    for j in range(p):
        mean[j] /= w_sum

    var = np.zeros(p, dtype=np.float64)
    for i in range(n):
        wi = w[i]
        for j in range(p):
            var[j] += wi * (X[i, j] - mean[j]) ** 2
    std = np.empty(p, dtype=np.float64)
    for j in range(p):
        var[j] /= w_sum
        std[j] = np.sqrt(var[j]) if var[j] > 0.0 else 1.0

    Z = np.empty((n, p), dtype=np.float64)
    for i in range(n):
        for j in range(p):
            Z[i, j] = (X[i, j] - mean[j]) / std[j]
    return Z


@njit_optional_cache(cache=True)
def _weighted_corr_with_last(Z: np.ndarray, last_idx: int, p: int, w: np.ndarray) -> np.ndarray:
    n = Z.shape[0]
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    corrs = np.empty(p, dtype=np.float64)
    for j in range(p):
        val = 0.0
        for i in range(n):
            val += w[i] * Z[i, j] * Z[i, last_idx]
        corrs[j] = np.abs(val / w_sum)
    return corrs


@njit_optional_cache(cache=True)
def mrmr_loop_incremental(
    Z: np.ndarray,
    relevance: np.ndarray,
    k: int,
    use_quotient: bool,
    w: np.ndarray,
) -> np.ndarray:
    """mRMR with weighted correlation for redundancy."""
    n, p = Z.shape
    k = min(k, p)

    selected = np.empty(k, dtype=np.int64)
    is_selected = np.zeros(p, dtype=np.bool_)
    red_sum = np.zeros(p, dtype=np.float64)

    best = 0
    best_val = relevance[0]
    for j in range(1, p):
        if relevance[j] > best_val:
            best_val = relevance[j]
            best = j

    selected[0] = best
    is_selected[best] = True

    for t in range(1, k):
        last = selected[t - 1]
        new_red = _weighted_corr_with_last(Z, last, p, w)

        for j in range(p):
            if not is_selected[j]:
                red_sum[j] += new_red[j]

        best_idx = -1
        best_score = -1e300

        for j in range(p):
            if is_selected[j]:
                continue

            mean_red = red_sum[j] / t
            if use_quotient:
                score = relevance[j] / max(mean_red, FLOOR)
            else:
                score = relevance[j] - mean_red

            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx < 0:
            return selected[:t]

        selected[t] = best_idx
        is_selected[best_idx] = True

    return selected


def _mrmr_loop_blas(
    Z: np.ndarray,
    relevance: np.ndarray,
    k: int,
    use_quotient: bool,
    w: np.ndarray,
) -> np.ndarray:
    """mRMR loop using BLAS matrix-vector redundancy updates."""
    n, p = Z.shape
    k = min(k, p)
    if k <= 0 or p == 0:
        return np.empty(0, dtype=np.int64)

    selected = np.empty(k, dtype=np.int64)
    is_selected = np.zeros(p, dtype=bool)
    red_sum = np.zeros(p, dtype=np.float64)
    w_sum = float(w.sum())

    best = int(np.argmax(relevance))
    selected[0] = best
    is_selected[best] = True
    count = 1

    for t in range(1, k):
        last = int(selected[t - 1])
        weighted_last = w * Z[:, last]
        new_red = np.abs(Z.T @ weighted_last / w_sum)

        mask = ~is_selected
        red_sum[mask] += new_red[mask]
        mean_red = red_sum / t
        if use_quotient:
            score = relevance / np.maximum(mean_red, FLOOR)
        else:
            score = relevance - mean_red
        score[is_selected] = -np.inf

        best_idx = int(np.argmax(score))
        if not np.isfinite(score[best_idx]):
            break

        selected[t] = best_idx
        is_selected[best_idx] = True
        count += 1

    return selected[:count]


def _corr_chunk_process(
    Z: np.ndarray,
    weighted_last: np.ndarray,
    w_sum: float,
    start: int,
    stop: int,
) -> np.ndarray:
    """Worker helper for process-backed correlation chunks."""
    with threadpool_limits(limits=1):
        return np.abs(Z[:, start:stop].T @ weighted_last / w_sum)


def _weighted_corr_with_last_processes(
    Z: np.ndarray,
    weighted_last: np.ndarray,
    w_sum: float,
    bounds: np.ndarray,
    parallel: Parallel,
) -> np.ndarray:
    """Compute one redundancy update across process workers."""
    chunks = parallel(
        delayed(_corr_chunk_process)(Z, weighted_last, w_sum, int(bounds[i]), int(bounds[i + 1]))
        for i in range(len(bounds) - 1)
        if bounds[i] < bounds[i + 1]
    )
    return np.concatenate(chunks)


def _mrmr_loop_processes(
    Z: np.ndarray,
    relevance: np.ndarray,
    k: int,
    use_quotient: bool,
    w: np.ndarray,
    n_jobs: int,
) -> np.ndarray:
    """mRMR loop using process-backed redundancy updates."""
    n, p = Z.shape
    k = min(k, p)
    if k <= 0 or p == 0:
        return np.empty(0, dtype=np.int64)

    selected = np.empty(k, dtype=np.int64)
    is_selected = np.zeros(p, dtype=bool)
    red_sum = np.zeros(p, dtype=np.float64)

    best = int(np.argmax(relevance))
    selected[0] = best
    is_selected[best] = True
    count = 1
    if k <= 1:
        return selected[:count]

    w_sum = float(w.sum())
    n_workers = max(1, min(p, effective_n_jobs(n_jobs)))
    bounds = np.linspace(0, p, n_workers + 1, dtype=np.int64)

    parallel_context = (
        Parallel(
            n_jobs=n_jobs,
            prefer="processes",
            max_nbytes="16M",
            batch_size=1,
        )
        if n_workers > 1
        else nullcontext(None)
    )

    def redundancy_update(last_idx: int, parallel: Parallel | None) -> np.ndarray:
        weighted_last = w * Z[:, last_idx]
        if parallel is None:
            return np.abs(Z.T @ weighted_last / w_sum)
        return _weighted_corr_with_last_processes(
            Z,
            weighted_last,
            w_sum,
            bounds,
            parallel,
        )

    with parallel_context as parallel:
        for t in range(1, k):
            last = int(selected[t - 1])
            new_red = redundancy_update(last, parallel)

            mask = ~is_selected
            red_sum[mask] += new_red[mask]
            mean_red = red_sum / t
            if use_quotient:
                score = relevance / np.maximum(mean_red, FLOOR)
            else:
                score = relevance - mean_red
            score[is_selected] = -np.inf

            best_idx = int(np.argmax(score))
            if not np.isfinite(score[best_idx]):
                break

            selected[t] = best_idx
            is_selected[best_idx] = True
            count += 1

    return selected[:count]


def mrmr_select(
    X: np.ndarray,
    relevance: np.ndarray,
    k: int,
    formula: str = "quotient",
    top_m: Optional[int] = None,
    sample_weight: np.ndarray | None = None,
    n_jobs: int = 1,
    mrmr_backend: MrmrBackend = "auto",
) -> np.ndarray:
    """mRMR feature selection with incremental redundancy."""
    k = validate_k(k, allow_auto=False)
    if top_m is not None and (
        isinstance(top_m, (bool, np.bool_))
        or not isinstance(top_m, (int, np.integer))
        or int(top_m) < 1
    ):
        raise ValueError("top_m must be a positive integer or None")
    if top_m is not None:
        top_m = int(top_m)
    backend = resolve_mrmr_backend(mrmr_backend, n_jobs)
    n, p = X.shape
    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    valid_mask = relevance > 0
    if not valid_mask.any():
        return np.array([], dtype=np.int64)

    valid_idx = np.where(valid_mask)[0]
    X_valid = X if valid_idx.size == p else X[:, valid_idx]
    rel_valid = relevance[valid_idx]

    if top_m is not None and top_m < len(valid_idx):
        top_local = np.argpartition(rel_valid, -top_m)[-top_m:]
        X_sub = X_valid[:, top_local]
        rel_sub = rel_valid[top_local]
        idx_map = valid_idx[top_local]
    else:
        X_sub = X_valid
        rel_sub = rel_valid
        idx_map = valid_idx

    Z = _standardize_columns_weighted(X_sub.astype(np.float64, copy=False), w)
    use_quot = formula == "quotient"

    if backend == "serial":
        sel_local = mrmr_loop_incremental(Z, rel_sub, k, use_quot, w)
    elif backend == "blas":
        sel_local = _mrmr_loop_blas(Z, rel_sub, k, use_quot, w)
    elif backend == "processes":
        sel_local = _mrmr_loop_processes(Z, rel_sub, k, use_quot, w, n_jobs)
    else:  # pragma: no cover - guarded by resolve_mrmr_backend
        raise ValueError(f"Unknown mRMR backend: {backend}")

    return idx_map[sel_local]


# =============================================================================
# Classic JMI/JMIM (incremental scoring)
# =============================================================================

@_single_threaded_r2_jmi
def jmi_select(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    relevance: np.ndarray,
    mi_estimator: Literal["binned", "r2", "ksg"] = "r2",
    aggregation: Literal["sum", "min"] = "sum",
    top_m: Optional[int] = None,
    y_kind: Literal["discrete", "continuous"] = "continuous",
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """JMI/JMIM selection with incremental scoring."""
    from sift.estimators import joint_mi as jmi_est

    k = validate_k(k, allow_auto=False)
    if top_m is not None and (
        isinstance(top_m, (bool, np.bool_))
        or not isinstance(top_m, (int, np.integer))
        or int(top_m) < 1
    ):
        raise ValueError("top_m must be a positive integer or None")
    if top_m is not None:
        top_m = int(top_m)
    if mi_estimator == "ksg" and sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")

    n, p = X.shape
    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    y_arr = y.astype(np.float64, copy=False)
    w_arr = w.astype(np.float64, copy=False)

    valid_mask = relevance > 0
    if not valid_mask.any():
        return np.array([], dtype=np.int64)

    valid_idx = np.where(valid_mask)[0]
    X_valid = X if valid_idx.size == p else X[:, valid_idx]
    rel_valid = relevance[valid_idx]

    if top_m is not None and top_m < len(valid_idx):
        top_local = np.argpartition(rel_valid, -top_m)[-top_m:]
        X_cand = X_valid[:, top_local]
        rel_cand = rel_valid[top_local]
        idx_map = valid_idx[top_local]
    else:
        X_cand = X_valid
        rel_cand = rel_valid
        idx_map = valid_idx

    m = X_cand.shape[1]
    k = min(k, m)

    use_indexed = mi_estimator in ("r2", "binned")
    y_binned = None
    n_y_bins = None
    X_binned = None

    if mi_estimator == "r2":
        Z_cand, r_y, r2_w, r2_w_sum = jmi_est._prepare_r2_joint_mi_state(
            X_cand,
            y_arr,
            w_arr,
        )

        def mi_func_indexed(last_idx, idx):
            return jmi_est._r2_joint_mi_indexed_from_state(
                Z_cand,
                r_y,
                idx,
                last_idx,
                r2_w,
                r2_w_sum,
            )
    elif mi_estimator == "binned":
        X_binned = jmi_est.quantile_bin_matrix(X_cand, n_bins=10)
        if y_kind == "discrete":
            y_vals = np.asarray(y_arr)
            y_binned = jmi_est._factorize(y_vals)
            n_y_bins = int(y_binned.max()) + 1 if y_binned.size else 1
        else:
            y_binned = jmi_est._quantile_bin(y_arr, 10)
            n_y_bins = 10

        def mi_func_indexed(last_idx, idx):
            s_binned = X_binned[:, int(last_idx)]
            return jmi_est.binned_joint_mi_indexed_prebinned(
                X_binned,
                idx,
                s_binned,
                y_binned,
                w_arr,
                n_bins=10,
                n_y_bins=n_y_bins,
            )
    elif mi_estimator == "ksg":
        def mi_func_matrix(s, c):
            return jmi_est.ksg_joint_mi(s, c, y_arr)
        use_indexed = False
    else:
        raise ValueError(f"Unknown mi_estimator: {mi_estimator}")

    if aggregation == "sum":
        scores = np.zeros(m, dtype=np.float64)
    else:
        scores = np.full(m, np.inf, dtype=np.float64)

    is_selected = np.zeros(m, dtype=bool)
    selected = np.empty(k, dtype=np.int64)

    best = int(np.argmax(rel_cand))
    selected[0] = best
    is_selected[best] = True
    count = 1

    for t in range(1, k):
        last = int(selected[t - 1])

        cand_indices = np.where(~is_selected)[0]
        if len(cand_indices) == 0:
            break

        if use_indexed:
            cand_idx64 = cand_indices.astype(np.int64, copy=False)
            mi_values = mi_func_indexed(last, cand_idx64)
        else:
            s_feat = X_cand[:, last]
            candidates = X_cand[:, cand_indices]
            mi_values = mi_func_matrix(s_feat, candidates)

        if aggregation == "sum":
            scores[cand_indices] += mi_values
        else:
            # Scalar min keeps the current score when the new value is NaN.
            not_nan = ~np.isnan(mi_values)
            update_idx = cand_indices[not_nan]
            scores[update_idx] = np.minimum(scores[update_idx], mi_values[not_nan])

        candidate_scores = scores[cand_indices]
        candidate_scores = np.where(
            np.isfinite(candidate_scores),
            candidate_scores,
            rel_cand[cand_indices],
        )
        best_idx = int(cand_indices[int(np.argmax(candidate_scores))])

        if best_idx < 0:
            break

        selected[t] = best_idx
        is_selected[best_idx] = True
        count += 1

    return idx_map[selected[:count]]
