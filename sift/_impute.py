"""Centralized imputation utilities."""

from __future__ import annotations

import numpy as np


def mean_impute(X: np.ndarray, *, copy: bool = True) -> np.ndarray:
    """Replace non-finite values with column means.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix, may contain NaN/inf.
    copy : bool
        If True, operate on a copy. If False, modify in place when possible.
        Non-floating inputs are always cast to float64, which allocates.

    Returns
    -------
    X_imputed : ndarray of shape (n, p)
        Feature matrix with non-finite values replaced by column means.
        Columns that are entirely non-finite are filled with 0.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("mean_impute expects 2D array")

    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64, copy=True)
    elif copy:
        X = X.copy()

    mask = ~np.isfinite(X)
    if not mask.any():
        return X

    X[mask] = np.nan
    finite = np.isfinite(X)
    counts = finite.sum(axis=0)
    sums = np.where(finite, X, 0.0).sum(axis=0, dtype=np.float64)
    col_means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    ).astype(X.dtype, copy=False)

    row_idx, col_idx = np.where(mask)
    X[row_idx, col_idx] = col_means[col_idx]
    return X
