"""
CEFS+ and fast mRMR implementations using Gaussian-copula MI approximation.

Key innovations:
- Gaussian-copula transform (rank → normal scores) makes MI estimation O(1) per pair
- Log-det / Schur complement updates for efficient greedy multivariate selection
- Rank/dominance-count selection matching the CEFS+ paper's spirit
- Subsampling support for million-row datasets
- X caching for efficient multi-target selection (e.g., CV folds)

Methods:
- cefsplus: CEFS+-style with rank-count selection on log-det quantities
- mrmr_fcd: mRMR with difference criterion (relevance - redundancy)
- mrmr_fcq: mRMR with quotient criterion (relevance / redundancy)
"""

from dataclasses import dataclass
import importlib.util
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata

_numba_spec = importlib.util.find_spec("numba")
if _numba_spec is not None:
    from numba import njit

    HAS_NUMBA = True
else:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap


# =============================================================================
# Transforms and MI utilities
# =============================================================================

def _standardize_2d(X: np.ndarray) -> np.ndarray:
    """Z-score standardization for 2D array."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    return Z.astype(np.float32, copy=False)


def _standardize_1d(y: np.ndarray) -> np.ndarray:
    """Z-score standardization for 1D array."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y - y.mean()
    sd = y.std()
    if sd > 0:
        y = y / sd
    return y.astype(np.float32, copy=False)


def _rank_gauss_1d(x: np.ndarray) -> np.ndarray:
    """
    Rank-based Gaussian (normal scores) transform.

    Maps data to N(0,1) via: rank → uniform → inverse normal CDF
    This is the copula transform that makes correlation ≈ copula dependence.

    Uses scipy.stats.rankdata with method='average' for proper tie handling,
    which is important for zero-inflated or highly discretized data.
    """
    x = np.asarray(x)
    n = x.shape[0]
    ranks = rankdata(x, method="average")
    u = ranks / (n + 1)
    z = ndtri(u)
    z -= z.mean()
    sd = z.std()
    if sd > 0:
        z /= sd
    return z.astype(np.float32, copy=False)


def _rank_gauss_2d(X: np.ndarray) -> np.ndarray:
    """Apply rank-Gaussian transform to each column."""
    X = np.asarray(X)
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    for j in range(p):
        Z[:, j] = _rank_gauss_1d(X[:, j])
    return Z


def _gaussian_mi_from_corr(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Mutual information (nats) for Gaussian variables from correlation.

    For bivariate Gaussian: I(X; Y) = -0.5 * log(1 - ρ²)
    This is exact for Gaussian, approximate for copula-transformed data.
    """
    r = np.asarray(r, dtype=np.float64)
    r2 = np.clip(r * r, 0.0, 1.0 - eps)
    return (-0.5 * np.log1p(-r2)).astype(np.float64)


def _corr_matrix(Z: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from standardized data."""
    Z = np.asarray(Z, dtype=np.float32)
    n = Z.shape[0]
    R = (Z.T @ Z) / max(n - 1, 1)
    R = np.clip(R, -0.999999, 0.999999)
    np.fill_diagonal(R, 1.0)
    return R.astype(np.float32, copy=False)


def _corr_with_vector(Z: np.ndarray, zy: np.ndarray) -> np.ndarray:
    """Compute correlation of each column of Z with vector zy."""
    Z = np.asarray(Z, dtype=np.float32)
    zy = np.asarray(zy, dtype=np.float32).ravel()
    n = Z.shape[0]
    r = (Z.T @ zy) / max(n - 1, 1)
    return np.clip(r, -0.999999, 0.999999).astype(np.float32, copy=False)


# =============================================================================
# Candidate pruning
# =============================================================================

def greedy_corr_prune(
    cand: np.ndarray,
    Rxx: np.ndarray,
    score: np.ndarray,
    corr_threshold: float = 0.95,
) -> np.ndarray:
    """
    Greedy correlation-based pruning.

    Sort candidates by score descending, keep each feature and drop all
    remaining features with |corr| >= threshold.

    Parameters
    ----------
    cand : array of candidate indices
    Rxx : correlation matrix
    score : relevance scores for ranking
    corr_threshold : drop features with |corr| >= this

    Returns
    -------
    Pruned candidate indices
    """
    cand = np.asarray(cand, dtype=np.int64)
    if cand.size == 0:
        return cand

    order = cand[np.argsort(score[cand])[::-1]]
    keep = []
    active = np.ones(order.shape[0], dtype=bool)

    for i in range(order.shape[0]):
        if not active[i]:
            continue
        fi = order[i]
        keep.append(fi)
        c = np.abs(Rxx[fi, order])
        active &= c < corr_threshold

    return np.asarray(keep, dtype=np.int64)


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
    CEFS+-style greedy selection using log-det updates and rank-count scoring.

    This implements the paper's "rank technique" using efficient matrix updates:
    - Lf = log det(R_S∪{f}) measures feature-only copula entropy
    - Lc = log det(R_{y,S,f}) measures feature+target copula entropy
    - Score by rank(Lf) - rank(Lc), tie-break with relevance

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

        m_rem = rem.shape[0]

        order_f = np.argsort(Lf, kind="mergesort")
        gdc_f = np.empty(m_rem, dtype=np.int32)
        gdc_f[order_f] = np.arange(m_rem, dtype=np.int32)

        order_c = np.argsort(Lc, kind="mergesort")
        gdc_c = np.empty(m_rem, dtype=np.int32)
        gdc_c[order_c] = np.arange(m_rem, dtype=np.int32)

        diff = gdc_f - gdc_c
        best_diff = diff.max()
        ties = np.where(diff == best_diff)[0]

        if ties.size == 1:
            best_pos = int(ties[0])
        else:
            best_pos = int(ties[np.argmax(tie_break_rel[rem][ties])])

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


# =============================================================================
# X Cache for efficient multi-target selection
# =============================================================================

@dataclass
class FeatureCache:
    """
    Cached feature data for efficient multi-target feature selection.

    Build once per dataset/fold, then call select_features() for each target.

    Attributes
    ----------
    Z : transformed and standardized features (n_sub, p_valid)
    Rxx : feature-feature correlation matrix (p_valid, p_valid)
    valid_cols : original column indices of kept features
    row_idx : indices of subsampled rows
    mode : transform mode ('zscore' or 'copula')
    feature_names : original feature names (if DataFrame input)
    """

    Z: np.ndarray
    Rxx: np.ndarray
    valid_cols: np.ndarray
    row_idx: np.ndarray
    mode: str
    feature_names: Optional[List[str]] = None


def build_cache(
    X: Union[np.ndarray, pd.DataFrame],
    subsample: Optional[int] = 50_000,
    mode: Literal["zscore", "copula"] = "copula",
    random_state: int = 0,
    min_std: float = 1e-12,
    impute: Optional[Literal["mean", "median"]] = "mean",
) -> FeatureCache:
    """
    Build feature cache for efficient multi-target selection.

    Parameters
    ----------
    X : feature matrix (n_samples, n_features)
    subsample : max rows to use (None = all). Use 50k for speed on large data.
    mode : transform mode
        - 'zscore': simple standardization (fast, assumes linearity)
        - 'copula': rank-Gaussian transform (robust to monotonic nonlinearity)
    random_state : random seed for subsampling
    min_std : drop features with std < this (near-constants)
    impute : impute missing values on the subsample
        - 'mean': column mean (default)
        - 'median': column median
        - None: leave NaNs (may propagate)

    Returns
    -------
    FeatureCache for use with select_features()
    """
    feature_names = None
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values

    X = np.asarray(X)
    n, _p = X.shape
    rng = np.random.default_rng(random_state)

    if subsample is not None and n > subsample:
        row_idx = rng.choice(n, size=subsample, replace=False)
    else:
        row_idx = np.arange(n)

    Xs = np.asarray(X[row_idx], dtype=np.float64)

    if impute is not None:
        if impute == "mean":
            col_stat = np.nanmean(Xs, axis=0)
        elif impute == "median":
            col_stat = np.nanmedian(Xs, axis=0)
        else:
            raise ValueError("impute must be 'mean', 'median', or None")
        col_stat = np.where(np.isnan(col_stat), 0.0, col_stat)
        inds = np.where(np.isnan(Xs))
        Xs[inds] = col_stat[inds[1]]

    sd = Xs.std(axis=0)
    valid = sd > min_std
    valid_cols = np.where(valid)[0]
    Xs = Xs[:, valid]

    if mode == "zscore":
        Z = _standardize_2d(Xs)
    elif mode == "copula":
        Z = _rank_gauss_2d(Xs)
    else:
        raise ValueError("mode must be 'zscore' or 'copula'")

    Rxx = _corr_matrix(Z)

    return FeatureCache(
        Z=Z,
        Rxx=Rxx,
        valid_cols=valid_cols,
        row_idx=row_idx,
        mode=mode,
        feature_names=feature_names,
    )


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
        - 'cefsplus': CEFS+-style with rank-count on log-det (handles interactions)
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
    top_m = int(min(top_m, p_valid))
    k = int(min(k, p_valid))

    cand = np.argpartition(np.abs(r), -top_m)[-top_m:]

    cand = greedy_corr_prune(
        cand, cache.Rxx, score=np.abs(r), corr_threshold=corr_prune_threshold
    )

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
    show_progress: bool = True,
    cache: Optional[FeatureCache] = None,
    impute: Optional[Literal["mean", "median"]] = "mean",
) -> List[str]:
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
        - 'cefsplus': CEFS+-style with rank-count (default)
        - 'mrmr_fcd': mRMR difference criterion
        - 'mrmr_fcq': mRMR quotient criterion
    random_state : random seed
    show_progress : ignored (for API compatibility)
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
