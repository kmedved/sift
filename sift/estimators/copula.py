"""Gaussian copula transforms and caching for fast selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numba import njit
from scipy.special import ndtri
from scipy.stats import rankdata


@dataclass
class FeatureCache:
    """Cached feature data for multi-target selection."""

    Z: np.ndarray
    Rxx: Optional[np.ndarray]
    valid_cols: np.ndarray
    row_idx: np.ndarray
    feature_names: Optional[List[str]] = None


def build_cache(
    X,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    compute_Rxx: bool = False,
    min_std: float = 1e-12,
) -> FeatureCache:
    """Build feature cache for multi-target selection."""
    from sift._preprocess import extract_feature_names, to_numpy

    feature_names = extract_feature_names(X)
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

    rng = np.random.default_rng(random_state)
    if subsample is not None and n > subsample:
        row_idx = rng.choice(n, size=subsample, replace=False)
    else:
        row_idx = np.arange(n)

    Xs = X_arr[row_idx]

    Xs = np.where(np.isfinite(Xs), Xs, np.nan)
    col_means = np.nanmean(Xs, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(Xs)
    Xs[nan_mask] = col_means[np.where(nan_mask)[1]]

    stds = np.std(Xs, axis=0)
    valid_mask = stds > min_std
    valid_cols = np.where(valid_mask)[0]
    Xs = Xs[:, valid_mask]
    if Xs.shape[1] == 0:
        raise ValueError("All features were filtered out (constant or invalid). Cannot build cache.")

    Z = rank_gauss_2d(Xs)

    Rxx = correlation_matrix_fast(Z) if compute_Rxx else None

    return FeatureCache(
        Z=Z.astype(np.float32),
        Rxx=Rxx.astype(np.float32) if Rxx is not None else None,
        valid_cols=valid_cols,
        row_idx=row_idx,
        feature_names=feature_names,
    )


def rank_gauss_1d(x: np.ndarray) -> np.ndarray:
    """Rank-based Gaussian transform for 1D array."""
    mask = np.isfinite(x)
    m = mask.sum()
    if m <= 1:
        return np.zeros_like(x, dtype=np.float32)

    ranks = rankdata(x[mask], method="average")
    u = ranks / (m + 1.0)
    z = ndtri(u).astype(np.float64)
    z -= z.mean()
    std = z.std(ddof=1)
    if std < 1e-12:
        z[:] = 0.0
    else:
        z /= std

    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = z.astype(np.float32)
    return out


def rank_gauss_2d(X: np.ndarray) -> np.ndarray:
    """Apply rank-Gaussian transform to each column."""
    n, p = X.shape
    Z = np.empty((n, p), dtype=np.float32)
    for j in range(p):
        Z[:, j] = rank_gauss_1d(X[:, j])
    return Z


@njit(cache=True)
def correlation_matrix_fast(Z: np.ndarray) -> np.ndarray:
    """Correlation matrix from standardized data."""
    n, p = Z.shape
    R = (Z.T @ Z) / max(n - 1, 1)
    for i in range(p):
        for j in range(p):
            if R[i, j] > 0.999999:
                R[i, j] = 0.999999
            elif R[i, j] < -0.999999:
                R[i, j] = -0.999999
        R[i, i] = 1.0
    return R


@njit(cache=True)
def corr_with_vector(Z: np.ndarray, zy: np.ndarray) -> np.ndarray:
    """Correlation of each column of Z with vector zy."""
    n, p = Z.shape
    r = np.empty(p, dtype=np.float32)
    for j in range(p):
        r[j] = np.sum(Z[:, j] * zy) / max(n - 1, 1)
    return np.clip(r, -0.999999, 0.999999)


@njit(cache=True)
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

    order = candidates[np.argsort(-scores[candidates])]
    keep = []
    active = np.ones(len(order), dtype=bool)

    for i, fi in enumerate(order):
        if not active[i]:
            continue
        keep.append(fi)

        for j in range(i + 1, len(order)):
            if active[j]:
                fj = order[j]
                if np.abs(Rxx[fi, fj]) >= threshold:
                    active[j] = False

    return np.array(keep, dtype=np.int64)
