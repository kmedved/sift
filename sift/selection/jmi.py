"""Incremental JMI/JMIM selection."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from sift.estimators import joint_mi as jmi_est


def jmi_select(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    relevance: np.ndarray,
    mi_estimator: Literal["binned", "r2", "ksg"] = "r2",
    aggregation: Literal["sum", "min"] = "sum",
    top_m: Optional[int] = None,
    y_kind: Literal["discrete", "continuous"] = "continuous",
) -> np.ndarray:
    """JMI/JMIM selection with incremental scoring."""
    n, p = X.shape

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

    mi_funcs = {
        "binned": lambda s, c: jmi_est.binned_joint_mi(s, c, y, y_kind=y_kind),
        "r2": lambda s, c: jmi_est.r2_joint_mi(s, c, y),
        "ksg": lambda s, c: jmi_est.ksg_joint_mi(s, c, y),
    }
    mi_func = mi_funcs[mi_estimator]

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

        candidates = X_cand[:, cand_indices]
        mi_values = mi_func(s_feat, candidates)

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
