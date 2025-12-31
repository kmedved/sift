"""Generic greedy selection loops."""

from __future__ import annotations

import numpy as np
from numba import njit

FLOOR = 1e-6


@njit(cache=True)
def mrmr_loop(
    relevance: np.ndarray,
    redundancy_matrix: np.ndarray,
    k: int,
    use_quotient: bool = True,
) -> np.ndarray:
    """
    mRMR greedy loop.

    quotient: score = rel / mean(red)
    difference: score = rel - mean(red)
    """
    p = len(relevance)
    k = min(k, p)

    selected = np.empty(k, dtype=np.int64)
    is_selected = np.zeros(p, dtype=np.bool_)
    red_sum = np.zeros(p, dtype=np.float64)

    best = np.argmax(relevance)
    selected[0] = best
    is_selected[best] = True

    for t in range(1, k):
        last = selected[t - 1]

        for j in range(p):
            if not is_selected[j]:
                red_sum[j] += redundancy_matrix[j, last]

        best_idx = -1
        best_score = -1e300

        for j in range(p):
            if is_selected[j]:
                continue

            mean_red = red_sum[j] / t

            if use_quotient:
                denom = max(mean_red, FLOOR)
                score = relevance[j] / denom
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


@njit(cache=True)
def jmi_loop(
    relevance: np.ndarray,
    joint_mi_matrix: np.ndarray,
    k: int,
    use_min: bool = False,
) -> np.ndarray:
    """
    JMI/JMIM greedy loop.

    JMI: score = Σ I(f, s; y)
    JMIM: score = min I(f, s; y)
    """
    p = len(relevance)
    k = min(k, p)

    selected = np.empty(k, dtype=np.int64)
    is_selected = np.zeros(p, dtype=np.bool_)

    if use_min:
        scores = np.full(p, np.inf, dtype=np.float64)
    else:
        scores = np.zeros(p, dtype=np.float64)

    best = np.argmax(relevance)
    selected[0] = best
    is_selected[best] = True

    for t in range(1, k):
        last = selected[t - 1]

        for j in range(p):
            if is_selected[j]:
                continue
            mi = joint_mi_matrix[j, last]
            if use_min:
                if mi < scores[j]:
                    scores[j] = mi
            else:
                scores[j] += mi

        best_idx = -1
        best_score = -1e300

        for j in range(p):
            if is_selected[j]:
                continue
            if use_min and np.isinf(scores[j]):
                score = relevance[j]
            else:
                score = scores[j]

            if score > best_score:
                best_score = score
                best_idx = j

        if best_idx < 0:
            return selected[:t]

        selected[t] = best_idx
        is_selected[best_idx] = True

    return selected
