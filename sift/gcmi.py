"""
Gaussian-copula MI utilities, transforms, and feature cache helpers.

This module is the shared "engine" for copula-Gaussian correlation + MI
approximations used by CEFS+/mRMR variants and any future cache-based selectors.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata


# =============================================================================
# Transforms and MI utilities
# =============================================================================

def _standardize_2d(X: np.ndarray) -> np.ndarray:
    """Z-score standardization for 2D array."""
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    np.nan_to_num(Z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return Z.astype(np.float32, copy=False)


def _standardize_1d(y: np.ndarray) -> np.ndarray:
    """Z-score standardization for 1D array."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = np.where(np.isfinite(y), y, np.nan)
    y = y - np.nanmean(y)
    sd = np.nanstd(y)
    if sd > 0:
        y = y / sd
    np.nan_to_num(y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32, copy=False)


def _rank_gauss_1d(x: np.ndarray) -> np.ndarray:
    """
    Rank-based Gaussian (normal scores) transform.

    Maps data to N(0,1) via: rank → uniform → inverse normal CDF.
    Uses rankdata(method='average') for tie handling.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mask = np.isfinite(x)
    m = int(mask.sum())
    if m == 0:
        return np.zeros_like(x, dtype=np.float32)

    ranks = rankdata(x[mask], method="average")
    u = ranks / (m + 1.0)
    z = ndtri(u)
    z -= z.mean()
    sd = z.std()
    if sd > 0:
        z /= sd

    out = np.zeros_like(x, dtype=np.float64)
    out[mask] = z
    return out.astype(np.float32, copy=False)


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
    Exact for Gaussian; used as an approximation in copula space.
    """
    r = np.asarray(r, dtype=np.float64)
    r2 = np.clip(r * r, 0.0, 1.0 - eps)
    return (-0.5 * np.log1p(-r2)).astype(np.float64)


def _corr_matrix(Z: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from standardized data (float32)."""
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
    """
    cand = np.asarray(cand, dtype=np.int64)
    if cand.size == 0:
        return cand

    order = cand[np.argsort(score[cand])[::-1]]
    keep: List[int] = []
    active = np.ones(order.shape[0], dtype=bool)

    for i in range(order.shape[0]):
        if not active[i]:
            continue
        fi = int(order[i])
        keep.append(fi)
        c = np.abs(Rxx[fi, order])
        active &= c < corr_threshold

    return np.asarray(keep, dtype=np.int64)


# =============================================================================
# X Cache for efficient multi-target selection
# =============================================================================

@dataclass
class FeatureCache:
    """
    Cached feature data for efficient multi-target selection.

    Notes:
    - If compute_Rxx=False, Rxx is None and downstream code should compute
      correlations on demand for candidate subsets to avoid O(p^2) memory.
    """

    Z: np.ndarray
    Rxx: Optional[np.ndarray]
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
    compute_Rxx: bool = False,
) -> FeatureCache:
    """
    Build feature cache for efficient multi-target selection.

    Parameters
    ----------
    compute_Rxx:
        If True, compute full Rxx (O(p^2)).
        If False, store only Z and compute correlations on demand for candidate
        subsets (usually the better default for anything beyond small p).
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
    Xs = np.where(np.isfinite(Xs), Xs, np.nan)

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

    sd = np.nanstd(Xs, axis=0)
    valid = sd > min_std
    valid_cols = np.where(valid)[0]
    Xs = Xs[:, valid]

    if mode == "zscore":
        Z = _standardize_2d(Xs)
    elif mode == "copula":
        Z = _rank_gauss_2d(Xs)
    else:
        raise ValueError("mode must be 'zscore' or 'copula'")

    Rxx = _corr_matrix(Z) if compute_Rxx else None

    return FeatureCache(
        Z=Z,
        Rxx=Rxx,
        valid_cols=valid_cols,
        row_idx=row_idx,
        mode=mode,
        feature_names=feature_names,
    )
