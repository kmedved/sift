"""Incremental mRMR selection."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

FLOOR = 1e-6


@njit(cache=True)
def _standardize_columns(X: np.ndarray) -> np.ndarray:
    """Standardize columns to zero mean, unit variance (sample std)."""
    n, p = X.shape
    if n < 2:
        return np.zeros_like(X)
    Z = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += X[i, j]
        mean = s / n

        ss = 0.0
        for i in range(n):
            ss += (X[i, j] - mean) ** 2
        std = np.sqrt(ss / (n - 1))

        if std < 1e-12:
            for i in range(n):
                Z[i, j] = 0.0
        else:
            for i in range(n):
                Z[i, j] = (X[i, j] - mean) / std
    return Z


@njit(cache=True)
def _corr_with_last(Z: np.ndarray, last_idx: int, p: int) -> np.ndarray:
    """Compute |correlation| of each column with column last_idx."""
    n = Z.shape[0]
    corrs = np.empty(p, dtype=np.float64)

    for j in range(p):
        dot = 0.0
        for i in range(n):
            dot += Z[i, j] * Z[i, last_idx]
        corrs[j] = np.abs(dot / (n - 1))

    return corrs


@njit(cache=True)
def mrmr_loop_incremental(
    Z: np.ndarray,
    relevance: np.ndarray,
    k: int,
    use_quotient: bool,
) -> np.ndarray:
    """mRMR greedy selection with O(p) memory."""
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

        new_red = _corr_with_last(Z, last, p)
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
) -> np.ndarray:
    """mRMR feature selection with incremental redundancy."""
    n, p = X.shape

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

    Z = _standardize_columns(X_sub.astype(np.float64))
    use_quot = formula == "quotient"

    sel_local = mrmr_loop_incremental(Z, rel_sub, k, use_quot)

    return idx_map[sel_local]
