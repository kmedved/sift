import functools
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd

from sift._optional import HAS_CATEGORY_ENCODERS

from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def parallel_df(func, df, series, n_jobs, prefer="threads"):
    # Handle empty DataFrame
    if len(df.columns) == 0:
        return pd.Series(dtype=float)

    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    n_jobs = max(1, n_jobs)  # Ensure at least 1 job
    n_jobs = min(n_jobs, len(df.columns))
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def _f_classif(X, y):
    y_notna = ~pd.isna(y)

    def _f_classif_series(x, y):
        mask = (~x.isna()) & y_notna
        if mask.sum() == 0:
            return 0
        if pd.Series(y[mask]).nunique(dropna=True) < 2:
            return 0
        return sklearn_f_classif(x[mask].to_frame(), y[mask])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y):
    y_notna = ~pd.isna(y)

    def _f_regression_series(x, y):
        mask = (~x.isna()) & y_notna
        if mask.sum() == 0:
            return 0
        return sklearn_f_regression(x[mask].to_frame(), y[mask])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def f_classif(X, y, n_jobs=-1, parallel_prefer="threads"):
    return parallel_df(_f_classif, X, y, n_jobs=n_jobs, prefer=parallel_prefer)


def f_regression(X, y, n_jobs=-1, parallel_prefer="threads"):
    return parallel_df(_f_regression, X, y, n_jobs=n_jobs, prefer=parallel_prefer)


def _ks_classif(X, y):
    y_notna = ~pd.isna(y)

    def _ks_classif_series(x, y):
        mask = (~x.isna()) & y_notna
        if mask.sum() == 0:
            return 0
        if pd.Series(y[mask]).nunique(dropna=True) < 2:
            return 0
        x = x[mask]
        y = y[mask]
        return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()

    return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)


def ks_classif(X, y, n_jobs=-1, parallel_prefer="threads"):
    return parallel_df(_ks_classif, X, y, n_jobs=n_jobs, prefer=parallel_prefer)


def random_forest_classif(X, y):
    fill_base = X.min(numeric_only=True).min()
    if not np.isfinite(fill_base):
        fill_base = 0.0
    X_filled = X.fillna(fill_base - 1.0)
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X_filled, y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    fill_base = X.min(numeric_only=True).min()
    if not np.isfinite(fill_base):
        fill_base = 0.0
    X_filled = X.fillna(fill_base - 1.0)
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X_filled, y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X, n_jobs=-1, parallel_prefer="threads"):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)

    return parallel_df(
        _correlation,
        X.loc[:, features],
        X.loc[:, target_column],
        n_jobs=n_jobs,
        prefer=parallel_prefer,
    )


def binned_joint_mi_classif(
    target_column,
    features,
    X,
    y,
    n_bins=10,
    n_jobs=-1,
    parallel_prefer="threads",
):
    """Compute I(f, target_column; y) for classification using binning.

    This correctly computes joint MI between the pair (f, target_column) and
    discrete target y using histogram-based estimation.

    Parameters
    ----------
    target_column: str
        The already-selected feature to pair with each candidate.
    features: list of str
        Candidate features to evaluate.
    X: pandas.DataFrame
        Feature matrix.
    y: pandas.Series
        Target variable (categorical/discrete).
    n_bins: int
        Number of bins for continuous features.
    n_jobs: int
        Number of parallel jobs.

    Returns
    -------
    pandas.Series
        Joint MI score for each candidate feature.
    """
    X_arr = X.values if hasattr(X, 'values') else X
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    col_idx = {c: i for i, c in enumerate(X.columns)}
    target_idx = col_idx[target_column]

    def _compute_joint_mi(f):
        x1 = X_arr[:, col_idx[f]]
        x2 = X_arr[:, target_idx]

        # Handle missing values
        mask = np.isfinite(x1) & np.isfinite(x2) & pd.notna(y_arr)
        if mask.sum() < 20:
            return 0.0

        x1, x2, y_vals = x1[mask], x2[mask], y_arr[mask]
        n = len(y_vals)

        # Discretize continuous features by quantiles
        def qbin(a):
            a = np.asarray(a, dtype=np.float64)
            # Handle constant features
            if np.nanstd(a) < 1e-10:
                return np.zeros(len(a), dtype=np.int32)
            qs = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(a, qs)
            bins[0] -= 1e-12
            bins[-1] += 1e-12
            return np.digitize(a, bins[1:-1]).astype(np.int32)

        x1d = qbin(x1)
        x2d = qbin(x2)

        # Joint bin id for (x1, x2)
        xd = x1d * n_bins + x2d

        # Factorize y to 0..C-1
        y_codes, _ = pd.factorize(y_vals, sort=False)
        n_classes = y_codes.max() + 1

        # Joint id for (x, y)
        xy = xd.astype(np.int64) * n_classes + y_codes.astype(np.int64)

        # Compute entropies
        def entropy_from_counts(arr):
            _, counts = np.unique(arr, return_counts=True)
            p = counts / n
            return -np.sum(p * np.log(p + 1e-12))

        h_x = entropy_from_counts(xd)
        h_y = entropy_from_counts(y_codes)
        h_xy = entropy_from_counts(xy)

        # MI = H(X) + H(Y) - H(X,Y)
        mi = h_x + h_y - h_xy
        return max(mi, 0.0)

    n_jobs = min(cpu_count(), len(features)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    n_jobs = max(1, n_jobs)

    if n_jobs == 1 or len(features) <= 2:
        results = {f: _compute_joint_mi(f) for f in features}
    else:
        results_list = Parallel(n_jobs=n_jobs, prefer=parallel_prefer)(
            delayed(_compute_joint_mi)(f) for f in features
        )
        results = dict(zip(features, results_list))

    return pd.Series(results)


def encode_df(X, y, cat_features, cat_encoding):
    if not HAS_CATEGORY_ENCODERS:
        raise ImportError(
            "category_encoders is required for categorical encoding. "
            "Install it with: pip install category_encoders\n"
            "Or set cat_features=None to disable categorical encoding."
        )
    import category_encoders as ce
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X
