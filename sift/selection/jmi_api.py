"""JMI-family function-style selector APIs."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import (
    CatEncoding,
    EstimatorJMI,
    RelevanceMethod,
    Task,
    check_regression_only,
    resolve_jmi_estimator,
    validate_k,
)
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.auto_k import AutoKConfig
from sift.selection.cefsplus import select_cached
from sift.selection.filter_api_common import (
    _auto_k_classic,
    _auto_k_gaussian,
    _build_selector_metadata,
    _compute_relevance,
    _default_top_m,
    _encode_categoricals_for_selector,
    _prepare_eval_data,
    _prepare_xy_classic,
    _reject_unsupported_auto_k_method,
    _resolve_auto_k_config,
    _resolve_cat_features,
    _safe_name_indices,
    _to_filter_result,
    _validate_groups_time,
)
from sift.selection.loops import jmi_select
from sift.selection.result import FilterSelectionResult


_Aggregation = Literal["sum", "min"]
_SelectorName = Literal["jmi", "jmim"]


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
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> List[str] | FilterSelectionResult:
    """Joint Mutual Information feature selection."""
    return _select_jmi_family(
        X,
        y,
        k,
        task=task,
        selector_name="jmi",
        aggregation="sum",
        display_name="JMI",
        cache=cache,
        groups=groups,
        time=time,
        auto_k_config=auto_k_config,
        sample_weight=sample_weight,
        estimator=estimator,
        relevance=relevance,
        top_m=top_m,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        subsample=subsample,
        random_state=random_state,
        verbose=verbose,
        return_result=return_result,
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
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> List[str] | FilterSelectionResult:
    """JMI Maximization, using the conservative minimum-pair aggregation."""
    return _select_jmi_family(
        X,
        y,
        k,
        task=task,
        selector_name="jmim",
        aggregation="min",
        display_name="JMIM",
        cache=cache,
        groups=groups,
        time=time,
        auto_k_config=auto_k_config,
        sample_weight=sample_weight,
        estimator=estimator,
        relevance=relevance,
        top_m=top_m,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        subsample=subsample,
        random_state=random_state,
        verbose=verbose,
        return_result=return_result,
    )


def _select_jmi_family(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Task,
    selector_name: _SelectorName,
    aggregation: _Aggregation,
    display_name: str,
    cache: Optional[FeatureCache],
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    auto_k_config: Optional[AutoKConfig],
    sample_weight: np.ndarray | None,
    estimator: EstimatorJMI,
    relevance: RelevanceMethod,
    top_m: Optional[int],
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    allow_full_data_target_encoding: bool,
    subsample: Optional[int],
    random_state: int,
    verbose: bool,
    return_result: bool,
) -> List[str] | FilterSelectionResult:
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    n_features_input = int(np.asarray(X).shape[1])
    groups, time = _validate_groups_time(groups, time, n_rows)
    estimator = resolve_jmi_estimator(estimator, task)
    k = validate_k(k)

    check_regression_only(task, estimator)
    if estimator == "ksg" and sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")

    if k == "auto":
        return _select_jmi_auto(
            X,
            y,
            task=task,
            selector_name=selector_name,
            aggregation=aggregation,
            display_name=display_name,
            cache=cache,
            groups=groups,
            time=time,
            auto_k_config=auto_k_config,
            sample_weight=sample_weight,
            estimator=estimator,
            relevance=relevance,
            top_m=top_m,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            allow_full_data_target_encoding=allow_full_data_target_encoding,
            subsample=subsample,
            random_state=random_state,
            verbose=verbose,
            return_result=return_result,
            n_features_input=n_features_input,
        )

    if estimator == "gaussian":
        cat_features = _resolve_cat_features(X, cat_features)
        X_encoded = _encode_categoricals_for_selector(
            X,
            y,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding=allow_full_data_target_encoding,
        )
        if verbose:
            print(f"{display_name} gaussian: selecting {k} features (top_m={top_m})")
        if cache is None:
            cache = build_cache(
                X_encoded,
                sample_weight=sample_weight,
                subsample=subsample,
                random_state=random_state,
            )
        if return_result:
            selected_features, selected_indices = select_cached(
                cache,
                y,
                k,
                method=selector_name,
                top_m=top_m,
                return_indices=True,
            )
            feature_names = (
                cache.feature_names
                if cache.feature_names is not None
                else [f"x{i}" for i in range(n_features_input)]
            )
            return _jmi_result(
                selector_name,
                selected_features,
                selected_indices,
                k_requested=int(k),
                top_m=_default_top_m(top_m, int(k)),
                n_features=len(feature_names),
                auto_k=False,
                task=task,
                estimator=estimator,
                relevance=relevance,
                return_result=return_result,
            )
        return select_cached(cache, y, k, method=selector_name, top_m=top_m)

    if return_result:
        selected_features, selected_indices = _jmi_classic(
            X,
            y,
            k,
            task,
            estimator,
            relevance,
            aggregation,
            display_name,
            top_m,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding,
            sample_weight,
            subsample,
            random_state,
            verbose,
            return_indices=True,
        )
        return _jmi_result(
            selector_name,
            selected_features,
            selected_indices,
            k_requested=int(k),
            top_m=_default_top_m(top_m, int(k)),
            n_features=n_features_input,
            auto_k=False,
            task=task,
            estimator=estimator,
            relevance=relevance,
            return_result=return_result,
        )

    return _jmi_classic(
        X,
        y,
        k,
        task,
        estimator,
        relevance,
        aggregation,
        display_name,
        top_m,
        cat_features,
        cat_encoding,
        allow_full_data_target_encoding,
        sample_weight,
        subsample,
        random_state,
        verbose,
    )


def _select_jmi_auto(
    X,
    y,
    *,
    task: Task,
    selector_name: _SelectorName,
    aggregation: _Aggregation,
    display_name: str,
    cache: Optional[FeatureCache],
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    auto_k_config: Optional[AutoKConfig],
    sample_weight: np.ndarray | None,
    estimator: EstimatorJMI,
    relevance: RelevanceMethod,
    top_m: Optional[int],
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    allow_full_data_target_encoding: bool,
    subsample: Optional[int],
    random_state: int,
    verbose: bool,
    return_result: bool,
    n_features_input: int,
):
    auto_k_config = _resolve_auto_k_config(auto_k_config, time, groups)
    _reject_unsupported_auto_k_method(
        selector_name=display_name,
        estimator=estimator,
        auto_k_config=auto_k_config,
    )
    cat_features = _resolve_cat_features(X, cat_features)
    max_k = auto_k_config.max_k
    top_m_eff = _default_top_m(top_m, max_k)

    if estimator == "gaussian":
        if verbose:
            print(
                f"{display_name} gaussian auto-k: building path to {max_k} features "
                f"(top_m={top_m_eff})"
            )
        if cache is None:
            X_enc = _encode_categoricals_for_selector(
                X,
                y,
                cat_features,
                cat_encoding,
                allow_full_data_target_encoding=allow_full_data_target_encoding,
            )
            cache = build_cache(
                X_enc,
                sample_weight=sample_weight,
                subsample=subsample,
                random_state=random_state,
            )
        eval_X, eval_y, eval_groups, eval_time, eval_weight = _prepare_eval_data(
            X, y, cache, groups, time, sample_weight
        )
        selected_features, selected_indices = _auto_k_gaussian_result(
            cache=cache,
            y=y,
            method=selector_name,
            max_k=max_k,
            top_m=top_m_eff,
            auto_k_config=auto_k_config,
            eval_X=eval_X,
            eval_y=eval_y,
            groups=eval_groups,
            time=eval_time,
            sample_weight=eval_weight,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            verbose=verbose,
            return_result=return_result,
        )
        if return_result:
            if selected_indices is None:
                selected_indices = _safe_name_indices(
                    cache.feature_names
                    if cache is not None and cache.feature_names is not None
                    else [f"x{i}" for i in range(n_features_input)],
                    selected_features,
                )
            return _jmi_result(
                selector_name,
                selected_features,
                selected_indices,
                k_requested="auto",
                top_m=top_m_eff,
                n_features=n_features_input,
                auto_k=True,
                task=task,
                estimator=estimator,
                relevance=relevance,
                auto_k_config=auto_k_config,
                return_result=return_result,
            )
        return selected_features

    X_arr, y_arr, w, feature_names, row_idx = _prepare_xy_classic(
        X,
        y,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        subsample=subsample,
        random_state=random_state,
        sample_weight=sample_weight,
    )
    rel = _compute_relevance(X_arr, y_arr, w, task, relevance)
    y_kind = "discrete" if task == "classification" else "continuous"

    if verbose:
        print(
            f"{display_name} classic auto-k: building path to {max_k} features "
            f"(top_m={top_m_eff})"
        )

    path_idx = jmi_select(
        X_arr,
        y_arr,
        max_k,
        rel,
        mi_estimator=estimator,
        aggregation=aggregation,
        top_m=top_m_eff,
        y_kind=y_kind,
        sample_weight=None if estimator == "ksg" else w,
    )
    X_eval = pd.DataFrame(X, columns=feature_names) if not isinstance(X, pd.DataFrame) else X
    X_eval = X_eval.iloc[row_idx]
    eval_groups = groups[row_idx] if groups is not None else None
    eval_time = time[row_idx] if time is not None else None

    if return_result:
        selected_features, selected_indices = _auto_k_classic(
            y_arr=y_arr,
            eval_X=X_eval,
            feature_names=feature_names,
            path_idx=path_idx,
            auto_k_config=auto_k_config,
            eval_groups=eval_groups,
            eval_time=eval_time,
            sample_weight=w,
            task=task,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            verbose=verbose,
            return_indices=True,
        )
        return _jmi_result(
            selector_name,
            selected_features,
            selected_indices,
            k_requested="auto",
            top_m=top_m_eff,
            n_features=len(feature_names),
            auto_k=True,
            task=task,
            estimator=estimator,
            relevance=relevance,
            auto_k_config=auto_k_config,
            return_result=return_result,
        )

    selected_features = _auto_k_classic(
        y_arr=y_arr,
        eval_X=X_eval,
        feature_names=feature_names,
        path_idx=path_idx,
        auto_k_config=auto_k_config,
        eval_groups=eval_groups,
        eval_time=eval_time,
        sample_weight=w,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        verbose=verbose,
    )
    return selected_features


def _auto_k_gaussian_result(*, return_result: bool, **kwargs):
    if return_result:
        return _auto_k_gaussian(return_indices=True, **kwargs)
    return _auto_k_gaussian(**kwargs), None


def _jmi_classic(
    X,
    y,
    k,
    task,
    mi_estimator,
    relevance_method,
    aggregation: _Aggregation,
    display_name: str,
    top_m,
    cat_features,
    cat_encoding,
    allow_full_data_target_encoding,
    sample_weight,
    subsample,
    random_state,
    verbose,
    return_indices: bool = False,
):
    """Classic JMI/JMIM implementation."""
    if mi_estimator == "ksg" and sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")

    X_arr, y_arr, w, feature_names, _ = _prepare_xy_classic(
        X,
        y,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        subsample=subsample,
        random_state=random_state,
        sample_weight=sample_weight,
    )
    rel = _compute_relevance(X_arr, y_arr, w, task, relevance_method)
    y_kind = "discrete" if task == "classification" else "continuous"
    top_m = _default_top_m(top_m, k)

    if verbose:
        print(
            f"{display_name} classic: selecting {k} features from "
            f"{X_arr.shape[1]} (top_m={top_m})"
        )

    selected_idx = jmi_select(
        X_arr,
        y_arr,
        k,
        rel,
        mi_estimator=mi_estimator,
        aggregation=aggregation,
        top_m=top_m,
        y_kind=y_kind,
        sample_weight=None if mi_estimator == "ksg" else w,
    )
    selected_features = [feature_names[i] for i in selected_idx]
    if return_indices:
        return selected_features, selected_idx.astype(int).tolist()
    return selected_features


def _jmi_result(
    selector_name: _SelectorName,
    selected_features: List[str],
    selected_indices: Optional[List[int]],
    *,
    k_requested: int | Literal["auto"],
    top_m: Optional[int],
    n_features: int,
    auto_k: bool,
    task: Task,
    estimator: EstimatorJMI,
    relevance: RelevanceMethod,
    return_result: bool,
    auto_k_config: Optional[AutoKConfig] = None,
) -> List[str] | FilterSelectionResult:
    extra = {
        "task": task,
        "estimator": estimator,
        "relevance": relevance,
    }
    if auto_k_config is not None:
        extra.update(
            {
                "auto_k_mode": auto_k_config.auto_k_mode,
                "k_method": auto_k_config.k_method,
                "auto_k_strategy": auto_k_config.strategy,
            }
        )
    return _to_filter_result(
        selected_features,
        selected_indices,
        _build_selector_metadata(
            selector_name,
            k=len(selected_features),
            k_requested=k_requested,
            top_m=top_m,
            n_features=n_features,
            auto_k=auto_k,
            extra=extra,
        ),
        return_result,
    )
