"""Relevance scoring: feature-target association."""

from __future__ import annotations

import numpy as np
from numba import njit, prange


def _ensure_weights(w: np.ndarray | None, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n, dtype=np.float64)
    return np.asarray(w, dtype=np.float64)


@njit(cache=True, parallel=True)
def f_regression(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted F-statistic for regression.

    For unweighted, caller passes np.ones(n).
    """
    n, p = X.shape
    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    y_mean = 0.0
    for i in range(n):
        y_mean += w[i] * y[i]
    y_mean /= w_sum

    y_ss = 0.0
    for i in range(n):
        y_ss += w[i] * (y[i] - y_mean) ** 2

    scores = np.empty(p, dtype=np.float64)
    for j in prange(p):
        x_mean = 0.0
        for i in range(n):
            x_mean += w[i] * X[i, j]
        x_mean /= w_sum

        x_ss = 0.0
        xy_cov = 0.0
        for i in range(n):
            xc = X[i, j] - x_mean
            yc = y[i] - y_mean
            x_ss += w[i] * xc * xc
            xy_cov += w[i] * xc * yc

        if x_ss < 1e-12 or y_ss < 1e-12:
            scores[j] = 0.0
        else:
            r = xy_cov / np.sqrt(x_ss * y_ss)
            r2 = min(r * r, 0.99999)
            scores[j] = r2 / (1.0 - r2) * (w_sum - 2)

    return scores


@njit(cache=True, parallel=True)
def f_classif(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted F-statistic for classification (weighted ANOVA)."""
    n, p = X.shape
    n_classes = int(y.max()) + 1

    class_weights = np.zeros(n_classes, dtype=np.float64)
    for i in range(n):
        class_weights[int(y[i])] += w[i]

    w_sum = 0.0
    for i in range(n):
        w_sum += w[i]

    scores = np.empty(p, dtype=np.float64)

    for j in prange(p):
        x_mean = 0.0
        for i in range(n):
            x_mean += w[i] * X[i, j]
        x_mean /= w_sum

        class_sums = np.zeros(n_classes, dtype=np.float64)
        class_sq_sums = np.zeros(n_classes, dtype=np.float64)

        for i in range(n):
            val = X[i, j]
            c_idx = int(y[i])
            class_sums[c_idx] += w[i] * val
            class_sq_sums[c_idx] += w[i] * val * val

        ss_between = 0.0
        ss_within = 0.0

        for c_idx in range(n_classes):
            w_c = class_weights[c_idx]
            if w_c < 1e-12:
                continue
            mean_c = class_sums[c_idx] / w_c
            ss_between += w_c * (mean_c - x_mean) ** 2
            ss_within += class_sq_sums[c_idx] - w_c * mean_c * mean_c

        df_between = n_classes - 1
        df_within = w_sum - n_classes

        if df_within <= 0 or df_between <= 0 or ss_within < 1e-12:
            scores[j] = 0.0
        else:
            scores[j] = (ss_between / df_between) / (ss_within / df_within)

    return scores


def ks_classif(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Kolmogorov-Smirnov statistic for classification."""
    from scipy.stats import ks_2samp

    classes = np.unique(y)
    n, p = X.shape
    scores = np.zeros(p, dtype=np.float64)

    for j in range(p):
        x = X[:, j]
        ks_sum = 0.0
        count = 0
        for c in classes:
            mask = y == c
            if mask.sum() < 2:
                continue
            stat, _ = ks_2samp(x[mask], x[~mask])
            ks_sum += stat
            count += 1
        scores[j] = ks_sum / max(count, 1)

    return scores


def rf_regression(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    max_depth: int = 5,
) -> np.ndarray:
    """Random forest importance for regression."""
    from sklearn.ensemble import RandomForestRegressor

    X_filled = np.nan_to_num(X, nan=np.nanmin(X) - 1)
    rf = RandomForestRegressor(max_depth=max_depth, n_estimators=100, random_state=0)
    rf.fit(X_filled, y, sample_weight=w)
    return rf.feature_importances_


def rf_classif(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    max_depth: int = 5,
) -> np.ndarray:
    """Random forest importance for classification."""
    from sklearn.ensemble import RandomForestClassifier

    X_filled = np.nan_to_num(X, nan=np.nanmin(X) - 1)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=0)
    rf.fit(X_filled, y, sample_weight=w)
    return rf.feature_importances_
