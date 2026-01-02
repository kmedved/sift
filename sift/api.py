"""User-facing API for feature selection."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

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
    resolve_jmi_estimator,
    validate_inputs,
)
from sift.estimators import relevance as rel_est
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.auto_k import AutoKConfig, select_k_auto, select_k_elbow
from sift.selection.cefsplus import select_cached
from sift.selection.loops import jmi_select, mrmr_select


def _default_top_m(top_m: Optional[int], k: int) -> int:
    tm = max(5 * k, 250) if top_m is None else int(top_m)
    # Ensure we can still return k features when a user passes top_m < k.
    return max(tm, int(k))


def _prepare_xy_classic(
    X,
    y,
    *,
    task: Task,
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    subsample: Optional[int],
    random_state: int,
    sample_weight: Optional[np.ndarray],
):
    """
    Shared preparation for 'classic' selectors:
    - infer cat_features for DataFrames
    - optional categorical encoding
    - validate_inputs + optional subsample
    Returns: (X_arr, y_arr, w, feature_names, row_idx)
    """
    if cat_features and cat_encoding != "none" and not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")

    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    if cat_features and cat_encoding != "none":
        X = encode_categoricals(X, y, cat_features, cat_encoding)

    X_arr, y_arr, feature_names = validate_inputs(X, y, task)
    w = rel_est._ensure_weights(sample_weight, X_arr.shape[0])

    n_rows = X_arr.shape[0]
    if subsample is not None and n_rows > subsample:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(n_rows, size=subsample, replace=False)
    else:
        row_idx = np.arange(n_rows)

    X_arr = X_arr[row_idx]
    y_arr = y_arr[row_idx]
    w = w[row_idx]
    return X_arr, y_arr, w, feature_names, row_idx


def select_mrmr(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Task,
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
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
    k : int or "auto"
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
    if k == "auto":
        if auto_k_config is None:
            if time is not None:
                auto_k_config = AutoKConfig(strategy="time_holdout")
            elif groups is not None:
                auto_k_config = AutoKConfig(strategy="group_cv")
            else:
                raise ValueError("k='auto' requires either time or groups parameter")

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if estimator == "gaussian":
            check_regression_only(task, estimator)
            if verbose:
                print(
                    f"mRMR gaussian auto-k: building path to {max_k} features "
                    f"(top_m={top_m_eff}, strategy={auto_k_config.strategy})"
                )
            X_enc = X
            if isinstance(X, pd.DataFrame) and cat_features is None:
                cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            if cat_features and cat_encoding != "none":
                if not isinstance(X, pd.DataFrame):
                    raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
                X_enc = encode_categoricals(X, y, cat_features, cat_encoding)
            if cache is None:
                cache = build_cache(
                    X_enc,
                    sample_weight=sample_weight,
                    subsample=subsample,
                    random_state=random_state,
                )
            method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
            use_elbow = auto_k_config.k_method == "elbow"
            if use_elbow:
                path, objective = select_cached(
                    cache,
                    y,
                    max_k,
                    method=method,
                    top_m=top_m_eff,
                    return_objective=True,
                )
                elbow_k, _ = select_k_elbow(
                    objective,
                    min_k=auto_k_config.min_k,
                    max_k=len(path),
                    min_rel_gain=auto_k_config.elbow_min_rel_gain,
                    patience=auto_k_config.elbow_patience,
                )
                if verbose:
                    print(f"  Elbow selected k={elbow_k}")
                return path[:elbow_k]
            path = select_cached(cache, y, max_k, method=method, top_m=top_m_eff, return_objective=False)
            X_df = X_enc if isinstance(X_enc, pd.DataFrame) else pd.DataFrame(X_enc, columns=cache.feature_names)
            y_arr = np.asarray(y).ravel()
            if subsample and len(X) > subsample:
                eval_X = X_df.iloc[cache.row_idx]
                eval_y = y_arr[cache.row_idx]
                eval_groups = groups[cache.row_idx] if groups is not None else None
                eval_time = time[cache.row_idx] if time is not None else None
            else:
                eval_X, eval_y = X_df, y_arr
                eval_groups, eval_time = groups, time
            if auto_k_config.strategy == "time_holdout" and time is None:
                raise ValueError("auto-k evaluate requires time when strategy='time_holdout'")
            if auto_k_config.strategy == "group_cv" and groups is None:
                raise ValueError("auto-k evaluate requires groups when strategy='group_cv'")

            best_k, selected, _ = select_k_auto(
                eval_X,
                eval_y,
                path,
                auto_k_config,
                groups=eval_groups,
                time=eval_time,
            )
            if verbose:
                print(f"  CV/holdout selected k={best_k}")
            return selected

        X_arr, y_arr, w, feature_names, row_idx = _prepare_xy_classic(
            X,
            y,
            task=task,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            subsample=subsample,
            random_state=random_state,
            sample_weight=sample_weight,
        )

        if task == "regression":
            rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
        else:
            rel_funcs = {
                "f": rel_est.f_classif,
                "ks": rel_est.ks_classif,
                "rf": rel_est.rf_classif,
            }

        if relevance not in rel_funcs:
            raise ValueError(
                f"relevance='{relevance}' not valid for task='{task}'. "
                f"Valid options: {sorted(rel_funcs.keys())}"
            )

        rel = rel_funcs[relevance](X_arr, y_arr, w)

        if verbose:
            print(
                f"mRMR classic auto-k: building path to {max_k} features "
                f"(top_m={top_m_eff}, strategy={auto_k_config.strategy})"
            )

        path_idx = mrmr_select(X_arr, rel, max_k, formula=formula, top_m=top_m_eff, sample_weight=w)
        path = [feature_names[i] for i in path_idx]
        X_df = pd.DataFrame(X_arr, columns=feature_names)

        eval_groups = groups[row_idx] if groups is not None else None
        eval_time = time[row_idx] if time is not None else None

        best_k, selected, _ = select_k_auto(
            X_df,
            y_arr,
            path,
            auto_k_config,
            groups=eval_groups,
            time=eval_time,
        )
        if verbose:
            print(f"  CV/holdout selected k={best_k}")
        return selected

    if estimator == "gaussian":
        check_regression_only(task, estimator)
        if cache is not None:
            method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
            return select_cached(cache, y, k, method=method, top_m=top_m)
        return _mrmr_gaussian(
            X,
            y,
            k,
            formula,
            top_m,
            cat_features,
            cat_encoding,
            sample_weight,
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
        sample_weight,
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
    sample_weight,
    subsample,
    random_state,
    verbose,
):
    """Classic mRMR implementation."""
    X_arr, y_arr, w, feature_names, _ = _prepare_xy_classic(
        X,
        y,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        subsample=subsample,
        random_state=random_state,
        sample_weight=sample_weight,
    )

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

    rel = rel_funcs[relevance_method](X_arr, y_arr, w)

    top_m = _default_top_m(top_m, k)

    if verbose:
        print(f"mRMR classic: selecting {k} features from {X_arr.shape[1]} (top_m={top_m})")

    selected_idx = mrmr_select(X_arr, rel, k, formula=formula, top_m=top_m, sample_weight=w)

    return [feature_names[i] for i in selected_idx]


def _mrmr_gaussian(
    X,
    y,
    k,
    formula,
    top_m,
    cat_features,
    cat_encoding,
    sample_weight,
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
    cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)
    method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
    return select_cached(cache, y, k, method=method, top_m=top_m)


def select_jmi(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Task,
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
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

    if k == "auto":
        if auto_k_config is None:
            if time is not None:
                auto_k_config = AutoKConfig(strategy="time_holdout")
            elif groups is not None:
                auto_k_config = AutoKConfig(strategy="group_cv")
            else:
                raise ValueError("k='auto' requires either time or groups parameter")

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if estimator == "gaussian":
            if verbose:
                print(
                    f"JMI gaussian auto-k: building path to {max_k} features "
                    f"(top_m={top_m_eff}, strategy={auto_k_config.strategy})"
                )
            X_enc = X
            if isinstance(X, pd.DataFrame) and cat_features is None:
                cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            if cat_features and cat_encoding != "none":
                if not isinstance(X, pd.DataFrame):
                    raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
                X_enc = encode_categoricals(X, y, cat_features, cat_encoding)
            if cache is None:
                cache = build_cache(
                    X_enc,
                    sample_weight=sample_weight,
                    subsample=subsample,
                    random_state=random_state,
                )
            use_elbow = auto_k_config.k_method == "elbow"
            if use_elbow:
                path, objective = select_cached(
                    cache,
                    y,
                    max_k,
                    method="jmi",
                    top_m=top_m_eff,
                    return_objective=True,
                )
                elbow_k, _ = select_k_elbow(
                    objective,
                    min_k=auto_k_config.min_k,
                    max_k=len(path),
                    min_rel_gain=auto_k_config.elbow_min_rel_gain,
                    patience=auto_k_config.elbow_patience,
                )
                if verbose:
                    print(f"  Elbow selected k={elbow_k}")
                return path[:elbow_k]
            path = select_cached(cache, y, max_k, method="jmi", top_m=top_m_eff, return_objective=False)
            X_df = X_enc if isinstance(X_enc, pd.DataFrame) else pd.DataFrame(X_enc, columns=cache.feature_names)
            y_arr = np.asarray(y).ravel()
            if subsample and len(X) > subsample:
                eval_X = X_df.iloc[cache.row_idx]
                eval_y = y_arr[cache.row_idx]
                eval_groups = groups[cache.row_idx] if groups is not None else None
                eval_time = time[cache.row_idx] if time is not None else None
            else:
                eval_X, eval_y = X_df, y_arr
                eval_groups, eval_time = groups, time
            if auto_k_config.strategy == "time_holdout" and time is None:
                raise ValueError("auto-k evaluate requires time when strategy='time_holdout'")
            if auto_k_config.strategy == "group_cv" and groups is None:
                raise ValueError("auto-k evaluate requires groups when strategy='group_cv'")

            best_k, selected, _ = select_k_auto(
                eval_X,
                eval_y,
                path,
                auto_k_config,
                groups=eval_groups,
                time=eval_time,
            )
            if verbose:
                print(f"  CV/holdout selected k={best_k}")
            return selected

        X_arr, y_arr, w, feature_names, row_idx = _prepare_xy_classic(
            X,
            y,
            task=task,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            subsample=subsample,
            random_state=random_state,
            sample_weight=sample_weight,
        )

        if task == "regression":
            rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
        else:
            rel_funcs = {
                "f": rel_est.f_classif,
                "ks": rel_est.ks_classif,
                "rf": rel_est.rf_classif,
            }

        if relevance not in rel_funcs:
            raise ValueError(
                f"relevance='{relevance}' not valid for task='{task}'. "
                f"Valid options: {sorted(rel_funcs.keys())}"
            )

        rel = rel_funcs[relevance](X_arr, y_arr, w)

        y_kind = "discrete" if task == "classification" else "continuous"

        if verbose:
            print(
                f"JMI classic auto-k: building path to {max_k} features "
                f"(top_m={top_m_eff}, strategy={auto_k_config.strategy})"
            )

        path_idx = jmi_select(
            X_arr,
            y_arr,
            max_k,
            rel,
            mi_estimator=estimator,
            aggregation="sum",
            top_m=top_m_eff,
            y_kind=y_kind,
            sample_weight=w,
        )
        path = [feature_names[i] for i in path_idx]
        X_df = pd.DataFrame(X_arr, columns=feature_names)

        eval_groups = groups[row_idx] if groups is not None else None
        eval_time = time[row_idx] if time is not None else None

        best_k, selected, _ = select_k_auto(
            X_df,
            y_arr,
            path,
            auto_k_config,
            groups=eval_groups,
            time=eval_time,
        )
        if verbose:
            print(f"  CV/holdout selected k={best_k}")
        return selected

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
        if cache is None:
            cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)
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
        sample_weight,
        subsample,
        random_state,
        verbose,
    )


def select_jmim(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Task,
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
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

    if k == "auto":
        if auto_k_config is None:
            if time is not None:
                auto_k_config = AutoKConfig(strategy="time_holdout")
            elif groups is not None:
                auto_k_config = AutoKConfig(strategy="group_cv")
            else:
                raise ValueError("k='auto' requires either time or groups parameter")

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if estimator == "gaussian":
            if verbose:
                print(
                    f"JMIM gaussian auto-k: building path to {max_k} features "
                    f"(top_m={top_m_eff}, strategy={auto_k_config.strategy})"
                )
            X_enc = X
            if isinstance(X, pd.DataFrame) and cat_features is None:
                cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
            if cat_features and cat_encoding != "none":
                if not isinstance(X, pd.DataFrame):
                    raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
                X_enc = encode_categoricals(X, y, cat_features, cat_encoding)
            if cache is None:
                cache = build_cache(
                    X_enc,
                    sample_weight=sample_weight,
                    subsample=subsample,
                    random_state=random_state,
                )
            use_elbow = auto_k_config.k_method == "elbow"
            if use_elbow:
                path, objective = select_cached(
                    cache,
                    y,
                    max_k,
                    method="jmim",
                    top_m=top_m_eff,
                    return_objective=True,
                )
                elbow_k, _ = select_k_elbow(
                    objective,
                    min_k=auto_k_config.min_k,
                    max_k=len(path),
                    min_rel_gain=auto_k_config.elbow_min_rel_gain,
                    patience=auto_k_config.elbow_patience,
                )
                if verbose:
                    print(f"  Elbow selected k={elbow_k}")
                return path[:elbow_k]
            path = select_cached(cache, y, max_k, method="jmim", top_m=top_m_eff, return_objective=False)
            X_df = X_enc if isinstance(X_enc, pd.DataFrame) else pd.DataFrame(X_enc, columns=cache.feature_names)
            y_arr = np.asarray(y).ravel()
            if subsample and len(X) > subsample:
                eval_X = X_df.iloc[cache.row_idx]
                eval_y = y_arr[cache.row_idx]
                eval_groups = groups[cache.row_idx] if groups is not None else None
                eval_time = time[cache.row_idx] if time is not None else None
            else:
                eval_X, eval_y = X_df, y_arr
                eval_groups, eval_time = groups, time
            if auto_k_config.strategy == "time_holdout" and time is None:
                raise ValueError("auto-k evaluate requires time when strategy='time_holdout'")
            if auto_k_config.strategy == "group_cv" and groups is None:
                raise ValueError("auto-k evaluate requires groups when strategy='group_cv'")

            best_k, selected, _ = select_k_auto(
                eval_X,
                eval_y,
                path,
                auto_k_config,
                groups=eval_groups,
                time=eval_time,
            )
            if verbose:
                print(f"  CV/holdout selected k={best_k}")
            return selected

        X_arr, y_arr, w, feature_names, row_idx = _prepare_xy_classic(
            X,
            y,
            task=task,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            subsample=subsample,
            random_state=random_state,
            sample_weight=sample_weight,
        )

        if task == "regression":
            rel_funcs = {"f": rel_est.f_regression, "rf": rel_est.rf_regression}
        else:
            rel_funcs = {
                "f": rel_est.f_classif,
                "ks": rel_est.ks_classif,
                "rf": rel_est.rf_classif,
            }

        if relevance not in rel_funcs:
            raise ValueError(
                f"relevance='{relevance}' not valid for task='{task}'. "
                f"Valid options: {sorted(rel_funcs.keys())}"
            )

        rel = rel_funcs[relevance](X_arr, y_arr, w)

        y_kind = "discrete" if task == "classification" else "continuous"

        if verbose:
            print(
                f"JMIM classic auto-k: building path to {max_k} features "
                f"(top_m={top_m_eff}, strategy={auto_k_config.strategy})"
            )

        path_idx = jmi_select(
            X_arr,
            y_arr,
            max_k,
            rel,
            mi_estimator=estimator,
            aggregation="min",
            top_m=top_m_eff,
            y_kind=y_kind,
            sample_weight=w,
        )
        path = [feature_names[i] for i in path_idx]
        X_df = pd.DataFrame(X_arr, columns=feature_names)

        eval_groups = groups[row_idx] if groups is not None else None
        eval_time = time[row_idx] if time is not None else None

        best_k, selected, _ = select_k_auto(
            X_df,
            y_arr,
            path,
            auto_k_config,
            groups=eval_groups,
            time=eval_time,
        )
        if verbose:
            print(f"  CV/holdout selected k={best_k}")
        return selected

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
        if cache is None:
            cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)
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
        sample_weight,
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
    sample_weight,
    subsample,
    random_state,
    verbose,
):
    """Classic JMI/JMIM implementation."""
    X_arr, y_arr, w, feature_names, _ = _prepare_xy_classic(
        X,
        y,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        subsample=subsample,
        random_state=random_state,
        sample_weight=sample_weight,
    )

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

    rel = rel_funcs[relevance_method](X_arr, y_arr, w)

    y_kind = "discrete" if task == "classification" else "continuous"
    aggregation = "min" if use_min else "sum"

    top_m = _default_top_m(top_m, k)

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
        sample_weight=w,
    )

    return [feature_names[i] for i in selected_idx]


def select_cefsplus(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]] = 75,
    *,
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
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
    from sift._preprocess import to_numpy

    y_arr = to_numpy(y, dtype=np.float32).ravel()
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    if len(y_arr) != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {len(y_arr)}")
    if not np.isfinite(y_arr).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")
    if k == "auto":
        if auto_k_config is None:
            if time is not None:
                auto_k_config = AutoKConfig(strategy="time_holdout")
            elif groups is not None:
                auto_k_config = AutoKConfig(strategy="group_cv")
            else:
                raise ValueError("k='auto' requires either time or groups parameter")

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        use_elbow = auto_k_config.k_method == "elbow"
        if verbose:
            mode = "elbow" if use_elbow else f"evaluate/{auto_k_config.strategy}"
            print(
                f"CEFS+ auto-k ({mode}): building path to {max_k} features "
                f"(top_m={top_m_eff}, corr_prune={corr_prune})"
            )

        if cache is None:
            cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)

        if use_elbow:
            path, objective = select_cached(
                cache,
                y,
                max_k,
                method="cefsplus",
                top_m=top_m_eff,
                corr_prune=corr_prune,
                return_objective=True,
            )
            elbow_k, _ = select_k_elbow(
                objective,
                min_k=auto_k_config.min_k,
                max_k=len(path),
                min_rel_gain=auto_k_config.elbow_min_rel_gain,
                patience=auto_k_config.elbow_patience,
            )
            if verbose:
                print(f"  Elbow selected k={elbow_k}")
            return path[:elbow_k]

        path = select_cached(
            cache,
            y,
            max_k,
            method="cefsplus",
            top_m=top_m_eff,
            corr_prune=corr_prune,
            return_objective=False,
        )

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=cache.feature_names)
        y_arr = np.asarray(y).ravel()

        if subsample and len(X) > subsample:
            eval_X = X_df.iloc[cache.row_idx]
            eval_y = y_arr[cache.row_idx]
            eval_groups = groups[cache.row_idx] if groups is not None else None
            eval_time = time[cache.row_idx] if time is not None else None
        else:
            eval_X, eval_y = X_df, y_arr
            eval_groups, eval_time = groups, time

        if auto_k_config.strategy == "time_holdout" and time is None:
            raise ValueError("auto-k evaluate requires time when strategy='time_holdout'")
        if auto_k_config.strategy == "group_cv" and groups is None:
            raise ValueError("auto-k evaluate requires groups when strategy='group_cv'")

        best_k, selected, _ = select_k_auto(
            eval_X,
            eval_y,
            path,
            auto_k_config,
            groups=eval_groups,
            time=eval_time,
        )

        if verbose:
            print(f"  CV/holdout selected k={best_k}")

        return selected

    k_int = int(k)
    top_m_eff = _default_top_m(top_m, k_int)
    if verbose:
        print(f"CEFS+: selecting {k_int} features (top_m={top_m_eff}, corr_prune={corr_prune})")
    if cache is None:
        cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)
    return select_cached(
        cache,
        y,
        k_int,
        method="cefsplus",
        top_m=top_m_eff,
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
