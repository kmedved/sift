"""User-facing API for feature selection."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import (
    CatEncoding,
    EstimatorJMI,
    EstimatorMRMR,
    Formula,
    RelevanceMethod,
    Task,
    check_regression_only,
    encode_categoricals,
    extract_feature_names,
    resolve_jmi_estimator,
    subsample_xy,
    to_numpy,
    validate_inputs,
)
from sift.estimators import relevance as rel_est
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.cefsplus import select_cached
from sift.selection.jmi import jmi_select
from sift.selection.mrmr import mrmr_select


def select_mrmr(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: Task,
    relevance: RelevanceMethod = "f",
    estimator: EstimatorMRMR = "classic",
    formula: Formula = "quotient",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
) -> List[str]:
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target variable.
    k : int
        Number of features to select.
    task : {"regression", "classification"}
        Task type.
    relevance : {"f", "ks", "rf"}
        Relevance scoring (only for estimator="classic").
    estimator : {"classic", "gaussian"}
        - "classic": F-stat relevance, Pearson correlation redundancy
        - "gaussian": Gaussian MI proxy (fast, regression only)
    formula : {"quotient", "difference"}
        - "quotient": rel / mean(red)
        - "difference": rel - mean(red)
    top_m : int, optional
        Prefilter to top_m features by relevance. Default: max(5*k, 250).

    Returns
    -------
    List[str]
        Selected feature names.
    """
    if estimator == "gaussian":
        check_regression_only(task, estimator)
        return _mrmr_gaussian(
            X,
            y,
            k,
            formula,
            top_m,
            cat_features,
            cat_encoding,
            subsample,
            random_state,
            verbose,
        )

    return _mrmr_classic(
        X,
        y,
        k,
        task,
        relevance,
        formula,
        top_m,
        cat_features,
        cat_encoding,
        subsample,
        random_state,
        verbose,
    )


def _mrmr_classic(
    X,
    y,
    k,
    task,
    relevance_method,
    formula,
    top_m,
    cat_features,
    cat_encoding,
    subsample,
    random_state,
    verbose,
):
    """Classic mRMR implementation."""
    feature_names = extract_feature_names(X)

    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")

    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)

    X_arr, y_arr, feature_names = validate_inputs(X, y, task)
    X_arr, y_arr = subsample_xy(X_arr, y_arr, subsample, random_state)

    if task == "regression":
        rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
    else:
        rel_funcs = {
            "f": rel_est.f_classif,
            "ks": rel_est.ks_classif,
            "rf": rel_est.rf_classif,
        }

    if relevance_method not in rel_funcs:
        raise ValueError(
            f"relevance='{relevance_method}' not valid for task='{task}'. "
            f"Valid options: {sorted(rel_funcs.keys())}"
        )

    rel = rel_funcs[relevance_method](X_arr, y_arr)

    if top_m is None:
        top_m = max(5 * k, 250)

    if verbose:
        print(f"mRMR classic: selecting {k} features from {X_arr.shape[1]} (top_m={top_m})")

    selected_idx = mrmr_select(X_arr, rel, k, formula=formula, top_m=top_m)

    return [feature_names[i] for i in selected_idx]


def _mrmr_gaussian(
    X,
    y,
    k,
    formula,
    top_m,
    cat_features,
    cat_encoding,
    subsample,
    random_state,
    verbose,
):
    """Gaussian mRMR via cached selection."""
    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")

    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)
    if verbose:
        print(f"mRMR gaussian: selecting {k} features (top_m={top_m})")
    cache = build_cache(X, subsample=subsample, random_state=random_state)
    method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
    return select_cached(cache, y, k, method=method, top_m=top_m)


def select_jmi(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: Task,
    estimator: EstimatorJMI = "auto",
    relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
) -> List[str]:
    """
    Joint Mutual Information feature selection.

    score(f) = Σ_{s ∈ S} I(f, s; y)
    """
    estimator = resolve_jmi_estimator(estimator, task)

    check_regression_only(task, estimator)

    if estimator == "gaussian":
        if isinstance(X, pd.DataFrame) and cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
            raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
        if cat_features and cat_encoding != "none":
            X = encode_categoricals(X, y, cat_features, cat_encoding)
        if verbose:
            print(f"JMI gaussian: selecting {k} features (top_m={top_m})")
        cache = build_cache(X, subsample=subsample, random_state=random_state)
        return select_cached(cache, y, k, method="jmi", top_m=top_m)

    return _jmi_classic(
        X,
        y,
        k,
        task,
        estimator,
        relevance,
        False,
        top_m,
        cat_features,
        cat_encoding,
        subsample,
        random_state,
        verbose,
    )


def select_jmim(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: Task,
    estimator: EstimatorJMI = "auto",
    relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
) -> List[str]:
    """
    JMI Maximization — conservative variant.

    score(f) = min_{s ∈ S} I(f, s; y)
    """
    estimator = resolve_jmi_estimator(estimator, task)

    check_regression_only(task, estimator)

    if estimator == "gaussian":
        if isinstance(X, pd.DataFrame) and cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
            raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
        if cat_features and cat_encoding != "none":
            X = encode_categoricals(X, y, cat_features, cat_encoding)
        if verbose:
            print(f"JMIM gaussian: selecting {k} features (top_m={top_m})")
        cache = build_cache(X, subsample=subsample, random_state=random_state)
        return select_cached(cache, y, k, method="jmim", top_m=top_m)

    return _jmi_classic(
        X,
        y,
        k,
        task,
        estimator,
        relevance,
        True,
        top_m,
        cat_features,
        cat_encoding,
        subsample,
        random_state,
        verbose,
    )


def _jmi_classic(
    X,
    y,
    k,
    task,
    mi_estimator,
    relevance_method,
    use_min,
    top_m,
    cat_features,
    cat_encoding,
    subsample,
    random_state,
    verbose,
):
    """Classic JMI/JMIM implementation."""
    feature_names = extract_feature_names(X)

    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")

    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)

    X_arr, y_arr, feature_names = validate_inputs(X, y, task)
    X_arr, y_arr = subsample_xy(X_arr, y_arr, subsample, random_state)

    if task == "regression":
        rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
    else:
        rel_funcs = {
            "f": rel_est.f_classif,
            "ks": rel_est.ks_classif,
            "rf": rel_est.rf_classif,
        }

    if relevance_method not in rel_funcs:
        raise ValueError(
            f"relevance='{relevance_method}' not valid for task='{task}'. "
            f"Valid options: {sorted(rel_funcs.keys())}"
        )

    rel = rel_funcs[relevance_method](X_arr, y_arr)

    y_kind = "discrete" if task == "classification" else "continuous"
    aggregation = "min" if use_min else "sum"

    if top_m is None:
        top_m = max(5 * k, 250)

    if verbose:
        method = "JMIM" if use_min else "JMI"
        print(f"{method} classic: selecting {k} features from {X_arr.shape[1]} (top_m={top_m})")

    selected_idx = jmi_select(
        X_arr,
        y_arr,
        k,
        rel,
        mi_estimator=mi_estimator,
        aggregation=aggregation,
        top_m=top_m,
        y_kind=y_kind,
    )

    return [feature_names[i] for i in selected_idx]


def select_cefsplus(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    top_m: Optional[int] = None,
    corr_prune: float = 0.95,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
) -> List[str]:
    """
    CEFS+ feature selection using log-det Gaussian MI proxy.

    REGRESSION ONLY.
    """
    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)
    y_arr = to_numpy(y, dtype=np.float32).ravel()
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    if len(y_arr) != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {len(y_arr)}")
    if not np.isfinite(y_arr).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")
    if verbose:
        print(f"CEFS+: selecting {k} features (top_m={top_m}, corr_prune={corr_prune})")
    cache = build_cache(X, subsample=subsample, random_state=random_state)
    return select_cached(
        cache,
        y,
        k,
        method="cefsplus",
        top_m=top_m,
        corr_prune=corr_prune,
    )


__all__ = [
    "FeatureCache",
    "build_cache",
    "select_cached",
    "select_cefsplus",
    "select_jmi",
    "select_jmim",
    "select_mrmr",
]
