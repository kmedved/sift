import functools
import numpy as np
import pandas as pd

from sift.core.algorithms import mrmr_base, jmi_base
from sift.backends.pandas import (
    parallel_df,
    f_classif,
    f_regression,
    ks_classif,
    random_forest_classif,
    random_forest_regression,
    correlation,
    encode_df,
    binned_joint_mi_classif,
)
from sift.core.sampling import subsample_xy
from sift.mi.estimators import regression_joint_mi, binned_joint_mi, ksg_joint_mi
from sift.mi.fast_selectors import cefsplus_regression as _cefsplus_regression


def mrmr_classif(
        X, y, K,
        method='mrmr',
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, verbose=True,
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
    verbose: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    X, y = subsample_xy(
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
                         return_scores=return_scores, verbose=verbose)

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
                        return_scores=return_scores, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'mrmr', 'jmi', 'jmim'.")


def mrmr_regression(
        X, y, K,
        method='mrmr',
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, verbose=True,
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
    verbose: bool (optional, default=True)
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
    X, y = subsample_xy(
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
                         return_scores=return_scores, verbose=verbose)

    elif method in ('jmi', 'jmim'):
        # JMI or JMIM - select MI estimation method
        if mi_method == 'regression':
            joint_mi_func = regression_joint_mi
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
                        return_scores=return_scores, verbose=verbose)

    elif method in ('cefsplus', 'mrmr_fcd', 'mrmr_fcq'):
        # Fast Gaussian-copula methods
        if return_scores:
            raise ValueError(f"Method '{method}' does not support return_scores=True.")
        mode = 'copula' if mi_method in ('copula', 'regression') else 'zscore'
        return _cefsplus_regression(
            X,
            y,
            K=K,
            mode=mode,
            method=method,
            subsample=None,
            # X, y already subsampled above; avoid second subsample inside cefsplus_regression.
            random_state=random_state,
            verbose=verbose,
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
    """Convenience wrapper for CEFS+ selection."""
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
