"""Relevance scoring: feature-target association."""

from __future__ import annotations

import numpy as np

from sift._numba import njit_optional_cache

CLASSIFICATION_BLOCK_SIZE = 256


# Keep these kernels serial: CatBoost/OpenMP can crash when Numba's parallel
# runtime is initialized after CatBoost has already been imported.
@njit_optional_cache(cache=True)
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

    # Row-major traversal: X is C-contiguous (n, p), so accumulating all
    # columns while sweeping rows keeps memory access contiguous. The
    # per-column summation order is unchanged (row 0..n-1), so results are
    # bitwise identical to the column-by-column form.
    x_mean = np.zeros(p, dtype=np.float64)
    for i in range(n):
        wi = w[i]
        for j in range(p):
            x_mean[j] += wi * X[i, j]
    for j in range(p):
        x_mean[j] /= w_sum

    x_ss = np.zeros(p, dtype=np.float64)
    xy_cov = np.zeros(p, dtype=np.float64)
    for i in range(n):
        wi = w[i]
        yc = y[i] - y_mean
        for j in range(p):
            xc = X[i, j] - x_mean[j]
            x_ss[j] += wi * xc * xc
            xy_cov[j] += wi * xc * yc

    scores = np.empty(p, dtype=np.float64)
    for j in range(p):
        if x_ss[j] <= 0.0 or y_ss <= 0.0:
            scores[j] = 0.0
        else:
            r = xy_cov[j] / np.sqrt(x_ss[j] * y_ss)
            r2 = min(r * r, 0.99999)
            scores[j] = r2 / (1.0 - r2) * (w_sum - 2)

    return scores


@njit_optional_cache(cache=True)
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

    df_between = n_classes - 1
    df_within = w_sum - n_classes

    # Sweep rows within bounded column blocks. Per-column accumulation remains
    # in row order (and therefore bitwise-equivalent to the scalar loop), while
    # peak scratch memory is O(n_classes * block_size), not O(n_classes * p).
    for start in range(0, p, CLASSIFICATION_BLOCK_SIZE):
        stop = min(start + CLASSIFICATION_BLOCK_SIZE, p)
        width = stop - start
        x_mean = np.zeros(width, dtype=np.float64)
        class_sums = np.zeros((n_classes, width), dtype=np.float64)
        class_sq_sums = np.zeros((n_classes, width), dtype=np.float64)
        for i in range(n):
            wi = w[i]
            c_idx = int(y[i])
            for local_j in range(width):
                val = X[i, start + local_j]
                x_mean[local_j] += wi * val
                class_sums[c_idx, local_j] += wi * val
                class_sq_sums[c_idx, local_j] += wi * val * val

        for local_j in range(width):
            x_mean_j = x_mean[local_j] / w_sum
            ss_between = 0.0
            ss_within = 0.0
            for c_idx in range(n_classes):
                w_c = class_weights[c_idx]
                if w_c < 1e-12:
                    continue
                mean_c = class_sums[c_idx, local_j] / w_c
                ss_between += w_c * (mean_c - x_mean_j) ** 2
                ss_within += (
                    class_sq_sums[c_idx, local_j] - w_c * mean_c * mean_c
                )

            j = start + local_j
            if df_within <= 0 or df_between <= 0:
                scores[j] = 0.0
            elif ss_within <= 0.0:
                scores[j] = (
                    1.0 / np.finfo(np.float64).eps
                    if ss_between > 0.0
                    else 0.0
                )
            else:
                scores[j] = (ss_between / df_between) / (ss_within / df_within)

    return scores


def _weighted_ks_2samp(x1: np.ndarray, w1: np.ndarray, x2: np.ndarray, w2: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)

    m1 = np.isfinite(x1) & np.isfinite(w1) & (w1 > 0)
    m2 = np.isfinite(x2) & np.isfinite(w2) & (w2 > 0)
    x1, w1 = x1[m1], w1[m1]
    x2, w2 = x2[m2], w2[m2]
    if x1.size == 0 or x2.size == 0:
        return 0.0

    o1 = np.argsort(x1, kind="mergesort")
    o2 = np.argsort(x2, kind="mergesort")
    x1, w1 = x1[o1], w1[o1]
    x2, w2 = x2[o2], w2[o2]

    c1 = np.cumsum(w1)
    c2 = np.cumsum(w2)
    c1 /= c1[-1]
    c2 /= c2[-1]

    xs = np.unique(np.concatenate([x1, x2]))
    i1 = np.searchsorted(x1, xs, side="right") - 1
    i2 = np.searchsorted(x2, xs, side="right") - 1

    F1 = np.where(i1 >= 0, c1[i1], 0.0)
    F2 = np.where(i2 >= 0, c2[i2], 0.0)
    return float(np.max(np.abs(F1 - F2)))


def ks_classif(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Kolmogorov-Smirnov statistic for classification."""
    from scipy.stats import ks_2samp

    classes = np.unique(y)
    n, p = X.shape
    scores = np.zeros(p, dtype=np.float64)

    w_arr = None
    if w is not None:
        w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
        if w_arr.shape[0] != n:
            raise ValueError(f"w has {w_arr.shape[0]} elements but X has {n} rows")
        if not np.isfinite(w_arr).all():
            raise ValueError("w must be finite.")
        if np.any(w_arr < 0):
            raise ValueError("w must be non-negative.")

    for j in range(p):
        x = X[:, j]
        finite_mask = np.isfinite(x)
        ks_sum = 0.0
        count = 0
        for c in classes:
            mask = (y == c) & finite_mask
            other_mask = (y != c) & finite_mask
            if w_arr is None:
                if mask.sum() < 2 or other_mask.sum() < 2:
                    continue
                stat, _ = ks_2samp(x[mask], x[other_mask])
            else:
                mask_w = mask & (w_arr > 0)
                other_mask_w = other_mask & (w_arr > 0)
                if mask_w.sum() < 2 or other_mask_w.sum() < 2:
                    continue
                stat = _weighted_ks_2samp(
                    x[mask_w],
                    w_arr[mask_w],
                    x[other_mask_w],
                    w_arr[other_mask_w],
                )
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
    from sift._impute import mean_impute
    from sklearn.ensemble import RandomForestRegressor

    X_arr = np.asarray(X)
    if X_arr.dtype == np.float32:
        X_arr = X_arr.astype(np.float32, copy=False)
    elif X_arr.dtype != np.float64:
        X_arr = X_arr.astype(np.float64, copy=False)

    X_filled = mean_impute(X_arr, copy=True)
    rf = RandomForestRegressor(max_depth=max_depth, n_estimators=100, random_state=0)
    rf.fit(X_filled, y, sample_weight=w)
    importances = np.asarray(rf.feature_importances_, dtype=np.float64).reshape(-1)
    if importances.size != X_filled.shape[1]:
        raise RuntimeError(
            f"RandomForest returned {importances.size} importances for {X_filled.shape[1]} features"
        )
    return importances


def rf_classif(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
    max_depth: int = 5,
) -> np.ndarray:
    """Random forest importance for classification."""
    from sift._impute import mean_impute
    from sklearn.ensemble import RandomForestClassifier

    X_arr = np.asarray(X)
    if X_arr.dtype == np.float32:
        X_arr = X_arr.astype(np.float32, copy=False)
    elif X_arr.dtype != np.float64:
        X_arr = X_arr.astype(np.float64, copy=False)

    X_filled = mean_impute(X_arr, copy=True)
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=100, random_state=0)
    rf.fit(X_filled, y, sample_weight=w)
    importances = np.asarray(rf.feature_importances_, dtype=np.float64).reshape(-1)
    if importances.size != X_filled.shape[1]:
        raise RuntimeError(
            f"RandomForest returned {importances.size} importances for {X_filled.shape[1]} features"
        )
    return importances
