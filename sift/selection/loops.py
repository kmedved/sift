"""Greedy selection loops and classic incremental selectors."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numba import njit

FLOOR = 1e-6


# =============================================================================
# Classic mRMR (incremental redundancy, O(p) memory)
# =============================================================================

@njit(cache=True)
def _standardize_columns_weighted(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n, p = X.shape
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    Z = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        mean = 0.0
        for i in range(n):
            mean += w[i] * X[i, j]
        mean /= w_sum

        var = 0.0
        for i in range(n):
            var += w[i] * (X[i, j] - mean) ** 2
        var /= w_sum
        std = np.sqrt(var) if var > 1e-12 else 1.0

        for i in range(n):
            Z[i, j] = (X[i, j] - mean) / std
    return Z


@njit(cache=True)
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


@njit(cache=True)
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


def mrmr_select(
    X: np.ndarray,
    relevance: np.ndarray,
    k: int,
    formula: str = "quotient",
    top_m: Optional[int] = None,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """mRMR feature selection with incremental redundancy."""
    n, p = X.shape
    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    valid_mask = relevance > 0
    if not valid_mask.any():
        return np.array([], dtype=np.int64)

    valid_idx = np.where(valid_mask)[0]
    X_valid = X[:, valid_idx]
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

    Z = _standardize_columns_weighted(X_sub.astype(np.float64), w)
    use_quot = formula == "quotient"

    sel_local = mrmr_loop_incremental(Z, rel_sub, k, use_quot, w)

    return idx_map[sel_local]


# =============================================================================
# Classic JMI/JMIM (incremental scoring)
# =============================================================================

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

    n, p = X.shape
    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    y_arr = y.astype(np.float64)
    w_arr = w.astype(np.float64)

    valid_mask = relevance > 0
    if not valid_mask.any():
        return np.array([], dtype=np.int64)

    valid_idx = np.where(valid_mask)[0]
    X_valid = X[:, valid_idx]
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

    if mi_estimator == "r2":
        def mi_func_indexed(s, idx):
            return jmi_est.r2_joint_mi_indexed(X_cand, idx, s, y_arr, w_arr)
    elif mi_estimator == "binned":
        def mi_func_indexed(s, idx):
            return jmi_est.binned_joint_mi_indexed(
                X_cand,
                idx,
                s,
                y_arr,
                w_arr,
                n_bins=10,
                y_kind=y_kind,
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
        last = selected[t - 1]
        s_feat = X_cand[:, last]

        cand_indices = np.where(~is_selected)[0]
        if len(cand_indices) == 0:
            break

        if use_indexed:
            cand_idx64 = cand_indices.astype(np.int64, copy=False)
            mi_values = mi_func_indexed(s_feat, cand_idx64)
        else:
            candidates = X_cand[:, cand_indices]
            mi_values = mi_func_matrix(s_feat, candidates)

        for i, idx in enumerate(cand_indices):
            if aggregation == "sum":
                scores[idx] += mi_values[i]
            else:
                scores[idx] = min(scores[idx], mi_values[i])

        best_score = -np.inf
        best_idx = -1
        for idx in cand_indices:
            score = scores[idx] if np.isfinite(scores[idx]) else rel_cand[idx]
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected[t] = best_idx
        is_selected[best_idx] = True
        count += 1

    return idx_map[selected[:count]]
