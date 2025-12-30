from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import warnings

from sift._optional import HAS_NUMBA, njit
from sift.mi.copula import (
    FeatureCache,
    _corr_matrix,
    _corr_with_vector,
    _gaussian_mi_from_corr,
    _rank_gauss_1d,
    _standardize_1d,
    build_cache,
    greedy_corr_prune,
)

_NUMBA_WARNING_SHOWN = False


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
    global _NUMBA_WARNING_SHOWN
    if not HAS_NUMBA and not _NUMBA_WARNING_SHOWN:
        warnings.warn(
            "Numba not installed; jmi_fast/jmim_fast will be slow",
            RuntimeWarning,
        )
        _NUMBA_WARNING_SHOWN = True
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


# =============================================================================
# Numba-accelerated mRMR implementations
# =============================================================================

@njit(cache=True)
def _mrmr_fcd_core(rel: np.ndarray, red: np.ndarray, k: int) -> np.ndarray:
    """
    mRMR with Forward-selection using Difference criterion.
    score(f) = relevance(f) - mean(redundancy with selected)
    """
    m = rel.shape[0]
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64)
    if k > m:
        k = m

    selected = np.empty(k, dtype=np.int64)
    used = np.zeros(m, dtype=np.bool_)
    red_sum = np.zeros(m, dtype=np.float64)

    best = 0
    best_val = rel[0]
    for j in range(1, m):
        if rel[j] > best_val:
            best_val = rel[j]
            best = j
    selected[0] = best
    used[best] = True

    for t in range(1, k):
        last = selected[t - 1]
        for j in range(m):
            red_sum[j] += red[j, last]

        best = -1
        best_score = -1e300
        for j in range(m):
            if used[j]:
                continue
            score = rel[j] - red_sum[j] / t
            if score > best_score:
                best_score = score
                best = j
        selected[t] = best
        used[best] = True

    return selected


@njit(cache=True)
def _mrmr_fcq_core(
    rel: np.ndarray, red: np.ndarray, k: int, eps: float = 1e-12
) -> np.ndarray:
    """
    mRMR with Forward-selection using Quotient criterion.
    score(f) = relevance(f) / mean(redundancy with selected)
    """
    m = rel.shape[0]
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64)
    if k > m:
        k = m

    selected = np.empty(k, dtype=np.int64)
    used = np.zeros(m, dtype=np.bool_)
    red_sum = np.zeros(m, dtype=np.float64)

    best = 0
    best_val = rel[0]
    for j in range(1, m):
        if rel[j] > best_val:
            best_val = rel[j]
            best = j
    selected[0] = best
    used[best] = True

    for t in range(1, k):
        last = selected[t - 1]
        for j in range(m):
            red_sum[j] += red[j, last]

        best = -1
        best_score = -1e300
        for j in range(m):
            if used[j]:
                continue
            denom = red_sum[j] / t + eps
            score = rel[j] / denom
            if score > best_score:
                best_score = score
                best = j
        selected[t] = best
        used[best] = True

    return selected


# =============================================================================
# CEFS+ with log-det Schur complement updates
# =============================================================================

def _cefsplus_logdet_select(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: Optional[np.ndarray] = None,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    CEFS+-style greedy selection using log-det updates and MI proxy scoring.

    This implements a log-det MI proxy using efficient matrix updates:
    - Lf = log det(R_S∪{f}) measures feature-only copula entropy
    - Lc = log det(R_{y,S,f}) measures feature+target copula entropy
    - Score by maximizing (Lf - Lc), which is proportional to the Gaussian MI proxy
      I(y; S∪{f}) ≈ 0.5 * (Lf - Lc) up to an additive constant.

    Uses Schur complement for O(s²) updates instead of O(s³) matrix inversions.

    Parameters
    ----------
    R : (m, m) feature-feature correlation on transformed scale
    r : (m,) feature-target correlation on transformed scale
    k : number of features to select
    tie_break_rel : relevance scores for tie-breaking (default: MI from r)
    shrink : shrinkage toward diagonal for numerical stability
    eps : numerical floor

    Returns
    -------
    Selected feature indices
    """
    R = np.asarray(R, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).ravel()
    m = R.shape[0]
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, m)

    if shrink > 0:
        R = (1.0 - shrink) * R
        np.fill_diagonal(R, 1.0)
        r = (1.0 - shrink) * r

    if tie_break_rel is None:
        tie_break_rel = _gaussian_mi_from_corr(r, eps=eps)
    else:
        tie_break_rel = np.asarray(tie_break_rel, dtype=np.float64).ravel()

    j0 = int(np.argmax(tie_break_rel))
    selected = [j0]
    remaining = np.ones(m, dtype=bool)
    remaining[j0] = False

    inv_S = np.array([[1.0]], dtype=np.float64)
    logdet_S = 0.0

    r0 = float(r[j0])
    det_yS = max(1.0 - r0 * r0, eps)
    inv_yS = (1.0 / det_yS) * np.array(
        [[1.0, -r0], [-r0, 1.0]], dtype=np.float64
    )
    logdet_yS = float(np.log(det_yS))

    while len(selected) < k and remaining.any():
        S = np.asarray(selected, dtype=int)
        rem = np.where(remaining)[0]
        s = len(selected)

        B = R[np.ix_(S, rem)]

        tmp = inv_S @ B
        t1 = np.sum(B * tmp, axis=0)
        s1 = np.maximum(1.0 - t1, eps)
        Lf = logdet_S + np.log(s1)

        B2 = np.vstack((r[rem][None, :], B))
        tmp2 = inv_yS @ B2
        t2 = np.sum(B2 * tmp2, axis=0)
        s2 = np.maximum(1.0 - t2, eps)
        Lc = logdet_yS + np.log(s2)

        # Greedy Gaussian MI proxy: maximize (Lf - Lc)
        score = Lf - Lc
        best = score.max()
        ties = np.where(np.isclose(score, best, rtol=1e-12, atol=1e-12))[0]
        best_pos = (
            int(ties[0])
            if ties.size == 1
            else int(ties[np.argmax(tie_break_rel[rem][ties])])
        )

        j = int(rem[best_pos])

        b = B[:, best_pos].reshape(-1, 1)
        v = inv_S @ b
        s1_best = float(s1[best_pos])

        inv_S_new = np.empty((s + 1, s + 1), dtype=np.float64)
        inv_S_new[:s, :s] = inv_S + (v @ v.T) / s1_best
        inv_S_new[:s, s] = (-v[:, 0]) / s1_best
        inv_S_new[s, :s] = inv_S_new[:s, s]
        inv_S_new[s, s] = 1.0 / s1_best
        inv_S = inv_S_new
        logdet_S += float(np.log(s1_best))

        b2 = B2[:, best_pos].reshape(-1, 1)
        v2 = inv_yS @ b2
        s2_best = float(s2[best_pos])

        inv_yS_new = np.empty((s + 2, s + 2), dtype=np.float64)
        inv_yS_new[:-1, :-1] = inv_yS + (v2 @ v2.T) / s2_best
        inv_yS_new[:-1, -1] = (-v2[:, 0]) / s2_best
        inv_yS_new[-1, :-1] = inv_yS_new[:-1, -1]
        inv_yS_new[-1, -1] = 1.0 / s2_best
        inv_yS = inv_yS_new
        logdet_yS += float(np.log(s2_best))

        selected.append(j)
        remaining[j] = False

    return np.asarray(selected, dtype=np.int64)


def select_features_cached(
    cache: FeatureCache,
    y: Union[np.ndarray, pd.Series],
    k: int = 50,
    top_m: int = 250,
    corr_prune_threshold: float = 0.95,
    method: Literal["cefsplus", "mrmr_fcd", "mrmr_fcq"] = "cefsplus",
    return_names: bool = True,
) -> Union[np.ndarray, List[str]]:
    """
    Select features using cached X data.

    Parameters
    ----------
    cache : FeatureCache from build_cache()
    y : target variable
    k : number of features to select
    top_m : pre-filter to top_m by relevance before selection (speed vs accuracy)
    corr_prune_threshold : drop features with |corr| >= this after relevance filter
    method : selection algorithm
        - 'cefsplus': CEFS+-style with log-det MI proxy (handles interactions)
        - 'mrmr_fcd': mRMR difference criterion (fast, stable)
        - 'mrmr_fcq': mRMR quotient criterion
    return_names : if True and cache has feature_names, return names instead of indices

    Returns
    -------
    Selected feature indices (or names if return_names=True and available)
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y).ravel()
    ys = y[cache.row_idx]

    if cache.mode == "zscore":
        zy = _standardize_1d(ys)
    else:
        zy = _rank_gauss_1d(ys)

    r = _corr_with_vector(cache.Z, zy)
    rel = _gaussian_mi_from_corr(r)

    p_valid = r.shape[0]
    if p_valid == 0 or k <= 0:
        return [] if (return_names and cache.feature_names is not None) else np.array([], dtype=np.int64)
    top_m = int(min(top_m, p_valid))
    k = int(min(k, p_valid))

    if top_m <= 0:
        raise ValueError(
            "top_m must be > 0; using all features can allocate O(p^2) memory."
        )
    else:
        top_m = min(top_m, p_valid)
        cand = np.argpartition(np.abs(r), -top_m)[-top_m:]

    if cache.Rxx is None:
        # Avoid O(p^2) full matrix: compute correlations only on candidate subset
        Z_cand = cache.Z[:, cand]
        R_cand = _corr_matrix(Z_cand)
        keep_idx = greedy_corr_prune(
            np.arange(cand.size),
            R_cand,
            score=np.abs(r[cand]),
            corr_threshold=corr_prune_threshold,
        )
        if keep_idx.size == 0:
            return [] if (return_names and cache.feature_names is not None) else np.array([], dtype=np.int64)
        cand = cand[keep_idx]
        R_sub = R_cand[np.ix_(keep_idx, keep_idx)]
    else:
        cand = greedy_corr_prune(
            cand, cache.Rxx, score=np.abs(r), corr_threshold=corr_prune_threshold
        )
        if cand.size == 0:
            return [] if (return_names and cache.feature_names is not None) else np.array([], dtype=np.int64)
        R_sub = cache.Rxx[np.ix_(cand, cand)]
    r_sub = r[cand]
    rel_sub = rel[cand]

    if method == "cefsplus":
        sel_local = _cefsplus_logdet_select(
            R_sub, r_sub, k=min(k, cand.size), tie_break_rel=rel_sub
        )
    elif method == "mrmr_fcd":
        red_mi = _gaussian_mi_from_corr(R_sub)
        np.fill_diagonal(red_mi, 0.0)
        sel_local = _mrmr_fcd_core(
            rel_sub.astype(np.float64),
            red_mi.astype(np.float64),
            k=min(k, cand.size),
        )
    elif method == "mrmr_fcq":
        red_mi = _gaussian_mi_from_corr(R_sub)
        np.fill_diagonal(red_mi, 0.0)
        sel_local = _mrmr_fcq_core(
            rel_sub.astype(np.float64),
            red_mi.astype(np.float64),
            k=min(k, cand.size),
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'cefsplus', 'mrmr_fcd', or 'mrmr_fcq'."
        )

    selected_valid = cand[sel_local]
    selected_original = cache.valid_cols[selected_valid]

    if return_names and cache.feature_names is not None:
        return [cache.feature_names[i] for i in selected_original]

    return selected_original


# =============================================================================
# Convenience function matching mrmr_regression API
# =============================================================================

def cefsplus_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    K: int,
    mode: Literal["zscore", "copula"] = "copula",
    top_m: Optional[int] = None,
    corr_prune_threshold: float = 0.95,
    subsample: Optional[int] = 50_000,
    method: Literal["cefsplus", "mrmr_fcd", "mrmr_fcq"] = "cefsplus",
    random_state: int = 0,
    verbose: bool = True,
    cache: Optional[FeatureCache] = None,
    impute: Optional[Literal["mean", "median"]] = "mean",
    compute_Rxx: bool = False,
) -> Union[List[str], np.ndarray]:
    """
    CEFS+/mRMR feature selection with Gaussian-copula MI estimation.

    This is a fast, scalable implementation suitable for:
    - Hundreds to thousands of features
    - Millions of rows (via subsampling)
    - Multi-target selection (via caching)

    Parameters
    ----------
    X : feature DataFrame or array
    y : target variable
    K : number of features to select
    mode : transform mode
        - 'copula': rank-Gaussian transform (default, robust)
        - 'zscore': simple standardization (faster, assumes linearity)
    top_m : pre-filter to top_m features by relevance (default: 5*K)
    corr_prune_threshold : drop features correlated > this (default: 0.95)
    subsample : max rows to use (default: 50000, None = all)
    method : selection algorithm
        - 'cefsplus': CEFS+-style with log-det MI proxy (default)
        - 'mrmr_fcd': mRMR difference criterion
        - 'mrmr_fcq': mRMR quotient criterion
    random_state : random seed
    verbose : ignored
    cache : optional FeatureCache to reuse precomputed X data
    impute : imputation strategy when building cache (ignored if cache provided)

    Returns
    -------
    List of selected feature names (or indices if array input)
    """
    if top_m is None:
        top_m = max(5 * K, 250)

    if cache is None:
        cache = build_cache(
            X,
            subsample=subsample,
            mode=mode,
            random_state=random_state,
            impute=impute,
            compute_Rxx=compute_Rxx,
        )

    return select_features_cached(
        cache,
        y,
        k=K,
        top_m=top_m,
        corr_prune_threshold=corr_prune_threshold,
        method=method,
        return_names=True,
    )
