from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .gcmi import (
    FeatureCache,
    build_cache,
    _corr_matrix,
    _corr_with_vector,
    _rank_gauss_1d,
    greedy_corr_prune,
)

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    import warnings

    warnings.warn(
        "Numba not installed; jmi_fast/jmim_fast will be slow",
        RuntimeWarning,
    )


@njit(cache=True)
def _fast_jmi_core(
    r_y: np.ndarray,
    R: np.ndarray,
    k: int,
    method: int,
) -> np.ndarray:
    m = r_y.shape[0]
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64)
    if k > m:
        k = m

    selected = np.empty(k, dtype=np.int64)
    used = np.zeros(m, dtype=np.bool_)

    r_y2 = r_y * r_y
    s = int(np.argmax(r_y2))
    selected[0] = s
    used[s] = True

    if method == 0:
        scores = np.zeros(m, dtype=np.float64)
    else:
        scores = np.full(m, np.inf, dtype=np.float64)

    count = 1
    for i in range(1, k):
        r_ys = r_y[s]
        for j in range(m):
            if used[j]:
                continue
            r_yj = r_y[j]
            r_js = R[j, s]
            denom = 1.0 - r_js * r_js
            if denom < 1e-8:
                r2 = r_ys * r_ys
            else:
                diff = r_yj - r_ys * r_js
                r2 = r_ys * r_ys + (diff * diff) / denom

            if r2 < 0.0:
                r2 = 0.0
            elif r2 > 0.999999:
                r2 = 0.999999

            mi = -0.5 * np.log(1.0 - r2)
            if method == 0:
                scores[j] += mi
            else:
                if mi < scores[j]:
                    scores[j] = mi

        best = -1
        best_score = -1.0e30
        for j in range(m):
            if used[j]:
                continue
            if scores[j] > best_score:
                best_score = scores[j]
                best = j

        if best < 0:
            break

        s = best
        selected[i] = s
        used[s] = True
        count += 1

    return selected[:count]


def _fast_jmi_core_numpy(
    r_y: np.ndarray,
    R: np.ndarray,
    k: int,
    method: int,
) -> np.ndarray:
    m = r_y.shape[0]
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, m)

    selected: List[int] = []
    used = np.zeros(m, dtype=bool)
    scores = np.zeros(m, dtype=np.float64) if method == 0 else np.full(m, np.inf, dtype=np.float64)

    s = int(np.argmax(r_y * r_y))
    selected.append(s)
    used[s] = True

    for _ in range(1, k):
        r_ys = r_y[s]
        r_js = R[:, s]
        denom = np.maximum(1.0 - r_js * r_js, 1e-8)
        diff = r_y - r_ys * r_js
        r2 = np.clip(r_ys * r_ys + (diff * diff) / denom, 0.0, 0.999999)
        mi = -0.5 * np.log(1.0 - r2)

        if method == 0:
            scores += mi
        else:
            scores = np.minimum(scores, mi)

        scores_masked = np.where(used, -np.inf, scores)
        s = int(np.argmax(scores_masked))
        if scores_masked[s] == -np.inf:
            break
        selected.append(s)
        used[s] = True

    return np.array(selected, dtype=np.int64)


def _jmi_select(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    K: int,
    top_m: Optional[int],
    corr_prune: float,
    subsample: int,
    random_state: int,
    method: int,
    cache: Optional[FeatureCache] = None,
) -> Union[List[str], List[int]]:
    if K <= 0:
        return []

    if cache is None:
        cache = build_cache(
            X,
            subsample=subsample,
            mode="copula",
            random_state=random_state,
            compute_Rxx=False,
        )

    if isinstance(y, pd.Series):
        y_values = y.to_numpy()
    else:
        y_values = np.asarray(y)

    ys = y_values[cache.row_idx]
    zy = _rank_gauss_1d(ys)
    r_y = _corr_with_vector(cache.Z, zy)

    if top_m is None:
        top_m = max(5 * K, 250)
    top_m = min(top_m, r_y.size)
    if top_m <= 0:
        return []

    if top_m == r_y.size:
        cand = np.arange(r_y.size)
    else:
        cand = np.argpartition(np.abs(r_y), -top_m)[-top_m:]

    Z_cand = cache.Z[:, cand]
    R_cand = _corr_matrix(Z_cand)
    del Z_cand  # free memory

    keep = greedy_corr_prune(
        np.arange(len(cand)),
        R_cand,
        score=np.abs(r_y[cand]),
        corr_threshold=corr_prune,
    )
    cand = cand[keep]
    if cand.size == 0:
        return []

    R_cand = R_cand[np.ix_(keep, keep)]
    r_y_cand = r_y[cand]

    k = min(K, len(cand))
    r_y64 = r_y_cand.astype(np.float64)
    R64 = R_cand.astype(np.float64)
    if HAS_NUMBA:
        sel_local = _fast_jmi_core(r_y64, R64, k, method)
    else:
        sel_local = _fast_jmi_core_numpy(r_y64, R64, k, method)

    selected = cache.valid_cols[cand[sel_local]]
    if cache.feature_names is not None:
        return [cache.feature_names[i] for i in selected]
    return selected.tolist()


def jmi_fast(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    K: int,
    top_m: Optional[int] = None,
    corr_prune: float = 0.95,
    subsample: int = 50_000,
    random_state: int = 0,
    cache: Optional[FeatureCache] = None,
) -> Union[List[str], List[int]]:
    return _jmi_select(
        X,
        y,
        K,
        top_m=top_m,
        corr_prune=corr_prune,
        subsample=subsample,
        random_state=random_state,
        method=0,
        cache=cache,
    )


def jmim_fast(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    K: int,
    top_m: Optional[int] = None,
    corr_prune: float = 0.95,
    subsample: int = 50_000,
    random_state: int = 0,
    cache: Optional[FeatureCache] = None,
) -> Union[List[str], List[int]]:
    return _jmi_select(
        X,
        y,
        K,
        top_m=top_m,
        corr_prune=corr_prune,
        subsample=subsample,
        random_state=random_state,
        method=1,
        cache=cache,
    )
