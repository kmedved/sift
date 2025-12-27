import functools
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd

# Lazy import - category_encoders is optional
try:
    import category_encoders as ce
    HAS_CATEGORY_ENCODERS = True
except Exception:
    ce = None
    HAS_CATEGORY_ENCODERS = False

from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .main import mrmr_base, jmi_base


def _subsample_xy(X, y, subsample, random_state, warn_subsample):
    if subsample is None:
        return X, y
    n = len(X)
    if n <= subsample:
        return X, y
    if warn_subsample:
        import warnings
        warnings.warn(
            f"Subsampling from {n} to {subsample} rows. Set subsample=None to use all data.",
            RuntimeWarning,
            stacklevel=3,
        )
    rng = np.random.default_rng(random_state)
    row_idx = rng.choice(n, size=subsample, replace=False)
    X_sub = X.iloc[row_idx] if hasattr(X, "iloc") else X[row_idx]
    y_sub = y.iloc[row_idx] if hasattr(y, "iloc") else y[row_idx]
    return X_sub, y_sub


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
    def _f_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y):
    def _f_regression_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def f_classif(X, y, n_jobs=-1, parallel_prefer="threads"):
    return parallel_df(_f_classif, X, y, n_jobs=n_jobs, prefer=parallel_prefer)


def f_regression(X, y, n_jobs=-1, parallel_prefer="threads"):
    return parallel_df(_f_regression, X, y, n_jobs=n_jobs, prefer=parallel_prefer)


def _ks_classif(X, y):
    def _ks_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        x = x[x_not_na]
        y = y[x_not_na]
        return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()

    return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)


def ks_classif(X, y, n_jobs=-1, parallel_prefer="threads"):
    return parallel_df(_ks_classif, X, y, n_jobs=n_jobs, prefer=parallel_prefer)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
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
        mask = ~(pd.isna(x1) | pd.isna(x2) | pd.isna(y_arr))
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
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X


def mrmr_classif(
        X, y, K,
        method='mrmr',
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True,
        subsample=50_000, random_state=0,
        warn_subsample=False,
        parallel_prefer="threads",
):
    """MRMR/JMI/JMIM feature selection for a classification task
    
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    method: str (optional, default='mrmr')
        Feature selection method. Options:
        - 'mrmr': Minimum Redundancy Maximum Relevance (default)
            score(f) = I(f; y) / mean_{s in S} |corr(f, s)|
        - 'jmi': Joint Mutual Information
            score(f) = sum_{s in S} I(f, s; y)
        - 'jmim': JMI Maximization (more conservative)
            score(f) = min_{s in S} I(f, s; y)
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "ks" (kolmogorov-smirnov), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method (only used when method='mrmr').
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score (only used when method='mrmr').
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy/joint_mi) is returned.
    n_jobs: int (optional, default=-1)
        Maximum number of workers to use.
        If -1, use as many workers as min(cpu count, number of features).
    subsample: int or None (optional, default=50000)
        Maximum number of rows to use for selection. If None, use all rows.
    random_state: int (optional, default=0)
        Random seed for row subsampling.
    warn_subsample: bool (optional, default=False)
        If True, warn when subsampling is applied.
    parallel_prefer: str (optional, default="threads")
        Joblib backend preference ("threads" or "processes").
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
        
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    X, y = _subsample_xy(
        X,
        y,
        subsample=subsample,
        random_state=random_state,
        warn_subsample=warn_subsample,
    )

    if cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_features) > 0:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    # Setup relevance function
    if relevance == "f":
        relevance_func = functools.partial(
            f_classif, n_jobs=n_jobs, parallel_prefer=parallel_prefer
        )
    elif relevance == "ks":
        relevance_func = functools.partial(
            ks_classif, n_jobs=n_jobs, parallel_prefer=parallel_prefer
        )
    elif relevance == "rf":
        relevance_func = random_forest_classif
    else:
        relevance_func = relevance

    relevance_args = {'X': X, 'y': y}

    if method == 'mrmr':
        # Original MRMR
        redundancy_func = functools.partial(
            correlation, n_jobs=n_jobs, parallel_prefer=parallel_prefer
        ) if redundancy == 'c' else redundancy
        denominator_func = np.mean if denominator == 'mean' else (
            np.max if denominator == 'max' else denominator)
        redundancy_args = {'X': X}

        return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                         relevance_args=relevance_args, redundancy_args=redundancy_args,
                         denominator_func=denominator_func, only_same_domain=only_same_domain,
                         return_scores=return_scores, show_progress=show_progress)
    
    elif method in ('jmi', 'jmim'):
        # JMI or JMIM - use binned joint MI for classification
        joint_mi_func = functools.partial(
            binned_joint_mi_classif,
            n_jobs=n_jobs,
            parallel_prefer=parallel_prefer,
        )
        joint_mi_args = {'X': X, 'y': y}

        return jmi_base(K=K, relevance_func=relevance_func, joint_mi_func=joint_mi_func,
                        relevance_args=relevance_args, joint_mi_args=joint_mi_args,
                        method=method, only_same_domain=only_same_domain,
                        return_scores=return_scores, show_progress=show_progress)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'mrmr', 'jmi', 'jmim'.")


def mrmr_regression(
        X, y, K,
        method='mrmr',
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True,
        mi_method='regression',
        subsample=50_000, random_state=0,
        warn_subsample=False,
        parallel_prefer="threads",
):
    """MRMR/JMI/JMIM feature selection for a regression task
    
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (continuous) target variable.
    K: int
        Number of features to select.
    method: str (optional, default='mrmr')
        Feature selection method. Options:
        - 'mrmr': Minimum Redundancy Maximum Relevance (default)
            score(f) = I(f; y) / mean_{s in S} |corr(f, s)|
        - 'jmi': Joint Mutual Information
            score(f) = sum_{s in S} I(f, s; y)
        - 'jmim': JMI Maximization (more conservative)
            score(f) = min_{s in S} I(f, s; y)
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method (only used when method='mrmr').
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score (only used when method='mrmr').
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy/joint_mi) is returned.
    n_jobs: int (optional, default=-1)
        Maximum number of workers to use.
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    mi_method: str (optional, default='regression')
        Method for computing joint mutual information (only used for JMI/JMIM). Options:
        - 'regression': R²-based approximation (fastest, assumes ~linear relationships)
        - 'binned': Discretization-based (fast, nonparametric)
        - 'ksg': KSG k-NN estimator (slower, most accurate)
    subsample: int or None (optional, default=50000)
        Maximum number of rows to use for selection. If None, use all rows.
    random_state: int (optional, default=0)
        Random seed for row subsampling.
    warn_subsample: bool (optional, default=False)
        If True, warn when subsampling is applied.
    parallel_prefer: str (optional, default="threads")
        Joblib backend preference ("threads" or "processes").
        
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    X, y = _subsample_xy(
        X,
        y,
        subsample=subsample,
        random_state=random_state,
        warn_subsample=warn_subsample,
    )

    if cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_features) > 0:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    # Setup relevance function
    relevance_func = functools.partial(
        f_regression, n_jobs=n_jobs, parallel_prefer=parallel_prefer
    ) if relevance == 'f' else (
        random_forest_regression if relevance == 'rf' else relevance)

    relevance_args = {'X': X, 'y': y}

    if method == 'mrmr':
        # Original MRMR
        redundancy_func = functools.partial(
            correlation, n_jobs=n_jobs, parallel_prefer=parallel_prefer
        ) if redundancy == 'c' else redundancy
        denominator_func = np.mean if denominator == 'mean' else (
            np.max if denominator == 'max' else denominator)
        redundancy_args = {'X': X}

        return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                         relevance_args=relevance_args, redundancy_args=redundancy_args,
                         denominator_func=denominator_func, only_same_domain=only_same_domain,
                         return_scores=return_scores, show_progress=show_progress)
    
    elif method in ('jmi', 'jmim'):
        # JMI or JMIM - select MI estimation method
        from .fast_mi import regression_joint_mi, binned_joint_mi, ksg_joint_mi
        if mi_method == 'regression':
            joint_mi_func = functools.partial(regression_joint_mi, n_jobs=n_jobs)
        elif mi_method == 'binned':
            joint_mi_func = functools.partial(binned_joint_mi, n_jobs=n_jobs)
        elif mi_method == 'ksg':
            joint_mi_func = functools.partial(ksg_joint_mi, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unknown mi_method: {mi_method}. Use 'regression', 'binned', or 'ksg'.")
        
        joint_mi_args = {'X': X, 'y': y}

        return jmi_base(K=K, relevance_func=relevance_func, joint_mi_func=joint_mi_func,
                        relevance_args=relevance_args, joint_mi_args=joint_mi_args,
                        method=method, only_same_domain=only_same_domain,
                        return_scores=return_scores, show_progress=show_progress)
    
    elif method in ('cefsplus', 'mrmr_fcd', 'mrmr_fcq'):
        # Fast Gaussian-copula methods
        from .cefsplus import cefsplus_regression
        mode = 'copula' if mi_method in ('copula', 'regression') else 'zscore'
        return cefsplus_regression(
            X,
            y,
            K=K,
            mode=mode,
            method=method,
            subsample=None,
            # X, y already subsampled above; avoid second subsample inside cefsplus_regression.
            random_state=random_state,
            show_progress=show_progress,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'mrmr', 'jmi', 'jmim', 'cefsplus', 'mrmr_fcd', 'mrmr_fcq'.")


# Convenience aliases
def jmi_regression(X, y, K, mi_method='regression', **kwargs):
    """Convenience wrapper for JMI regression.
    
    Parameters
    ----------
    mi_method: str (default='regression')
        MI estimation method: 'regression' (fast), 'binned', 'ksg'
    """
    return mrmr_regression(X, y, K, method='jmi', mi_method=mi_method, **kwargs)


def jmim_regression(X, y, K, mi_method='regression', **kwargs):
    """Convenience wrapper for JMIM regression.
    
    Parameters
    ----------
    mi_method: str (default='regression')
        MI estimation method: 'regression' (fast), 'binned', 'ksg'
    """
    return mrmr_regression(X, y, K, method='jmim', mi_method=mi_method, **kwargs)


def cefsplus_select(X, y, K, **kwargs):
    """Convenience wrapper for CEFS+ selection.
    
    Fast Gaussian-copula implementation with log-det updates.
    Handles feature interactions better than mRMR.
    """
    return mrmr_regression(X, y, K, method='cefsplus', **kwargs)


def jmi_classif(X, y, K, **kwargs):
    """Convenience wrapper for JMI classification."""
    return mrmr_classif(X, y, K, method='jmi', **kwargs)


def jmim_classif(X, y, K, **kwargs):
    """Convenience wrapper for JMIM classification."""
    return mrmr_classif(X, y, K, method='jmim', **kwargs)


__all__ = [
    "mrmr_classif",
    "mrmr_regression",
    "jmi_classif",
    "jmi_regression",
    "jmim_classif",
    "jmim_regression",
    "cefsplus_select",
    "f_classif",
    "f_regression",
    "correlation",
]
