"""Fast mutual information estimators for JMI/JMIM.

Three approaches, from fastest to most accurate:
1. regression_mi: R²-based approximation (fastest, good for linear relationships)
2. binned_mi: Discretization-based (fast, nonparametric)
3. ksg_mi: KSG estimator with scipy KDTree (accurate, moderate speed)
"""

import numpy as np
from scipy.special import digamma
from joblib import Parallel, delayed
from multiprocessing import cpu_count


# =============================================================================
# Option 1: Regression-based MI approximation (FASTEST)
# =============================================================================
# For Gaussian variables: I(X; Y) = -0.5 * log(1 - corr²)
# For joint MI: use analytic R² formula from correlation triplets

def regression_joint_mi(target_column, features, X, y, n_jobs=-1, block_size=256):
    """Fast joint MI using vectorized analytic R² from correlations.

    For predicting y from (f, s), the R² can be computed analytically:
    R² = (r_yf² + r_ys² - 2·r_yf·r_ys·r_fs) / (1 - r_fs²)

    Then: I(f, s; y) ≈ -0.5 * log(1 - R²)

    This is ~10x faster than OLS loop since it's fully vectorized.
    """
    import pandas as pd

    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = (y.values if hasattr(y, 'values') else y).ravel()
    col_idx = {c: i for i, c in enumerate(X.columns)}
    target_idx = col_idx[target_column]
    feat_idx = np.array([col_idx[f] for f in features])

    n = len(y_arr)

    # Standardize (z-score)
    def zscore(v):
        v = np.asarray(v, dtype=np.float64)
        mu = np.nanmean(v, axis=0)
        sd = np.nanstd(v, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (v - mu) / sd

    y_s = zscore(y_arr)
    s_s = zscore(X_arr[:, target_idx])  # selected feature (target_column)

    # Compute correlations (vectorized)
    # r_ys: corr(selected, y) -> scalar
    r_ys = np.dot(s_s, y_s) / n

    mi_vals = np.empty(len(features), dtype=np.float64)

    for start in range(0, len(feat_idx), block_size):
        block = feat_idx[start:start + block_size]
        F_s = zscore(X_arr[:, block])    # candidate features (n, block_size)

        # r_yf: corr(candidates, y) -> (block_size,)
        r_yf = (F_s.T @ y_s) / n

        # r_fs: corr(candidates, selected) -> (block_size,)
        r_fs = (F_s.T @ s_s) / n

        # Analytic R² for y ~ f + s
        # R² = (r_yf² + r_ys² - 2·r_yf·r_ys·r_fs) / (1 - r_fs²)
        denom = 1.0 - r_fs**2
        denom = np.where(denom < 1e-6, 1e-6, denom)  # avoid division by zero (collinearity)

        num = r_yf**2 + r_ys**2 - 2 * r_yf * r_ys * r_fs
        r2 = num / denom
        r2 = np.clip(r2, 0.0, 0.99999)

        # Convert R² to MI scale: I(f,s; y) ≈ -0.5 * log(1 - R²)
        mi_vals[start:start + len(block)] = -0.5 * np.log(1.0 - r2)

    return pd.Series(mi_vals, index=features)


# =============================================================================
# Option 2: Binned MI (FAST, nonparametric)
# =============================================================================

def _binned_mi_single(x1, x2, y, n_bins=10):
    """Compute I(x1, x2; y) by discretizing into bins.

    Faster than k-NN, handles nonlinear relationships.
    """
    mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    if mask.sum() < 20:
        return 0.0

    x1, x2, y = x1[mask], x2[mask], y[mask]
    n = len(y)

    # Discretize each variable into bins
    def discretize(arr):
        # Use quantile-based binning for robustness
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(arr, percentiles)
        bins[0] -= 1e-10
        bins[-1] += 1e-10
        return np.digitize(arr, bins[1:-1])

    x1_d = discretize(x1)
    x2_d = discretize(x2)
    y_d = discretize(y)

    # Joint variable for (x1, x2)
    joint_x = x1_d * n_bins + x2_d

    # Compute MI using histogram counts
    # I(X; Y) = H(X) + H(Y) - H(X, Y)
    def entropy(arr):
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log(probs + 1e-10))

    # Create joint (x1, x2, y) variable
    joint_all = joint_x * n_bins + y_d

    h_xy = entropy(joint_x)
    h_y = entropy(y_d)
    h_xy_y = entropy(joint_all)

    mi = h_xy + h_y - h_xy_y
    return max(mi, 0.0)


def binned_joint_mi(target_column, features, X, y, n_bins=10, n_jobs=-1):
    """Fast joint MI using binned/discretized estimation."""
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y
    col_idx = {c: i for i, c in enumerate(X.columns)}
    target_idx = col_idx[target_column]

    def compute(f):
        return _binned_mi_single(
            X_arr[:, col_idx[f]],
            X_arr[:, target_idx],
            y_arr,
            n_bins=n_bins
        )

    n_jobs = min(cpu_count(), len(features)) if n_jobs == -1 else min(cpu_count(), n_jobs)

    if n_jobs == 1 or len(features) <= 2:
        results = {f: compute(f) for f in features}
    else:
        results_list = Parallel(n_jobs=n_jobs)(delayed(compute)(f) for f in features)
        results = dict(zip(features, results_list))

    import pandas as pd
    return pd.Series(results)


# =============================================================================
# Option 3: Fast KSG estimator using scipy KDTree
# =============================================================================

def _ksg_mi_joint(x1, x2, y, k=3):
    """KSG estimator for I(x1, x2; y) using scipy KDTree.

    Uses the Kraskov-Stögbauer-Grassberger algorithm:
    I(X; Y) = ψ(k) + ψ(N) - <ψ(n_x + 1) + ψ(n_y + 1)>

    Where n_x, n_y are neighbor counts in marginal spaces within
    the k-th neighbor distance in joint space.
    """
    from scipy.spatial import cKDTree

    mask = ~(np.isnan(x1) | np.isnan(x2) | np.isnan(y))
    if mask.sum() < k + 5:
        return 0.0

    x1, x2, y = x1[mask], x2[mask], y[mask]
    n = len(y)

    # Standardize to unit variance
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-10)
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-10)
    y = (y - y.mean()) / (y.std() + 1e-10)

    # Joint space: (x1, x2) and marginal: y
    X_joint = np.column_stack([x1, x2])
    Y_marginal = y.reshape(-1, 1)
    XY_full = np.column_stack([x1, x2, y])

    # Build KD-trees
    tree_full = cKDTree(XY_full)
    tree_x = cKDTree(X_joint)
    tree_y = cKDTree(Y_marginal)

    # Find k+1 nearest neighbors (includes self) and get k-th distance
    # Using Chebyshev (max) distance for proper KSG algorithm
    dists, _ = tree_full.query(XY_full, k=k+1, p=np.inf)
    eps = dists[:, -1] - 1e-15  # k-th neighbor distance (excluding self)
    eps = np.maximum(eps, 0.0)

    # Count neighbors within eps in marginal spaces
    # Subtract 1 because query includes the point itself
    n_x = np.array([
        tree_x.query_ball_point(X_joint[i], eps[i], p=np.inf, return_length=True) - 1
        for i in range(n)
    ])
    n_y = np.array([
        tree_y.query_ball_point(Y_marginal[i], eps[i], p=np.inf, return_length=True) - 1
        for i in range(n)
    ])

    # Handle edge cases
    n_x = np.maximum(n_x, 0)
    n_y = np.maximum(n_y, 0)

    # KSG estimator
    mi = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))

    return max(mi, 0.0)


def ksg_joint_mi(target_column, features, X, y, k=3, n_jobs=-1):
    """KSG joint MI estimation."""
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else y
    col_idx = {c: i for i, c in enumerate(X.columns)}
    target_idx = col_idx[target_column]

    def compute(f):
        return _ksg_mi_joint(
            X_arr[:, col_idx[f]],
            X_arr[:, target_idx],
            y_arr,
            k=k
        )

    n_jobs = min(cpu_count(), len(features)) if n_jobs == -1 else min(cpu_count(), n_jobs)

    if n_jobs == 1 or len(features) <= 2:
        results = {f: compute(f) for f in features}
    else:
        results_list = Parallel(n_jobs=n_jobs)(delayed(compute)(f) for f in features)
        results = dict(zip(features, results_list))

    import pandas as pd
    return pd.Series(results)


# =============================================================================
# Convenience: auto-select best available method
# =============================================================================

def auto_joint_mi(target_column, features, X, y, method='auto', n_jobs=-1, **kwargs):
    """Automatically select best MI estimator.

    Parameters
    ----------
    method: str
        'auto': Use regression (fastest)
        'regression': R²-based approximation (fastest, assumes linearity)
        'binned': Discretization-based (fast, nonparametric)
        'ksg': KSG k-NN estimator (slower, most accurate)
    """
    if method == 'auto':
        # Default to regression for speed
        method = 'regression'

    if method == 'regression':
        return regression_joint_mi(target_column, features, X, y, n_jobs=n_jobs)
    if method == 'binned':
        n_bins = kwargs.get('n_bins', 10)
        return binned_joint_mi(target_column, features, X, y, n_bins=n_bins, n_jobs=n_jobs)
    if method == 'ksg':
        k = kwargs.get('k', 3)
        return ksg_joint_mi(target_column, features, X, y, k=k, n_jobs=n_jobs)
    raise ValueError(f"Unknown method: {method}. Use 'regression', 'binned', or 'ksg'.")
