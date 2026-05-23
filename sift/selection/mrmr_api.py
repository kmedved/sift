"""mRMR function-style selector API."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sift._preprocess import CatEncoding, EstimatorMRMR, Formula, RelevanceMethod, Task, check_regression_only, validate_k
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.cefsplus import select_cached
from sift.selection.auto_k import AutoKConfig
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
from sift.selection.loops import MrmrBackend, mrmr_select, resolve_mrmr_backend
from sift.selection.result import FilterSelectionResult


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
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    n_jobs: int = 1,
    mrmr_backend: MrmrBackend = "auto",
    verbose: bool = True,
    return_result: bool = False,
) -> List[str] | FilterSelectionResult:
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target variable.
    k : int or "auto"
        Maximum number of features to select. Fixed-k selectors may return fewer
        than k when fewer valid candidates remain.
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
    n_jobs : int, default=1
        Number of worker processes for mrmr_backend="processes". n_jobs=0 is invalid.
    mrmr_backend : {"auto", "serial", "blas", "processes"}, default="auto"
        Classic mRMR redundancy backend. "auto" keeps the existing serial path when
        n_jobs=1 and uses process workers when n_jobs!=1. Gaussian mRMR uses this
        setting only for cache/rank construction when a cache is not supplied.

    Returns
    -------
    List[str] or FilterSelectionResult
        Selected feature names, or result object when ``return_result=True``.
    """
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    n_features_input = int(np.asarray(X).shape[1])
    feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(n_features_input)]
    groups, time = _validate_groups_time(groups, time, n_rows)
    k = validate_k(k)
    mrmr_backend_resolved = resolve_mrmr_backend(mrmr_backend, n_jobs)
    rank_backend = "processes" if mrmr_backend_resolved == "processes" else "serial"

    if k == "auto":
        auto_k_config = _resolve_auto_k_config(auto_k_config, time, groups)
        _reject_unsupported_auto_k_method(
            selector_name="MRMR",
            estimator=estimator,
            auto_k_config=auto_k_config,
        )
        cat_features = _resolve_cat_features(X, cat_features)

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if estimator == "gaussian":
            check_regression_only(task, estimator)
            if verbose:
                print(
                    f"mRMR gaussian auto-k: building path to {max_k} features "
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
                    n_jobs=n_jobs,
                    rank_backend=rank_backend,
                )
            method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
            eval_X, eval_y, eval_groups, eval_time, eval_weight = _prepare_eval_data(
                X, y, cache, groups, time, sample_weight
            )

            if return_result:
                selected_features, selected_indices = _auto_k_gaussian(
                    cache=cache,
                    y=y,
                    method=method,
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
                    return_indices=True,
                )
            else:
                selected_features = _auto_k_gaussian(
                    cache=cache,
                    y=y,
                    method=method,
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
                )
                selected_indices = None

            if return_result:
                if selected_indices is None:
                    selected_indices = _safe_name_indices(
                        cache.feature_names
                        if cache is not None and cache.feature_names is not None
                        else [f"x{i}" for i in range(n_features_input)],
                        selected_features,
                    )
                return _to_filter_result(
                    selected_features,
                    selected_indices,
                    _build_selector_metadata(
                        "mrmr",
                        k=len(selected_features),
                        k_requested="auto",
                        top_m=top_m_eff,
                        n_features=n_features_input,
                        auto_k=True,
                        extra={
                            "task": task,
                            "estimator": estimator,
                            "formula": formula,
                            "relevance": relevance,
                            "auto_k_mode": auto_k_config.auto_k_mode,
                            "k_method": auto_k_config.k_method,
                            "auto_k_strategy": auto_k_config.strategy,
                        },
                    ),
                    return_result,
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

        if verbose:
            print(
                f"mRMR classic auto-k: building path to {max_k} features (top_m={top_m_eff})"
            )

        path_idx = mrmr_select(
            X_arr,
            rel,
            max_k,
            formula=formula,
            top_m=top_m_eff,
            sample_weight=w,
            n_jobs=n_jobs,
            mrmr_backend=mrmr_backend_resolved,
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
        else:
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
            selected_indices = None

        if return_result:
            if selected_indices is None:
                selected_indices = _safe_name_indices(feature_names, selected_features)
            return _to_filter_result(
                selected_features,
                selected_indices,
                _build_selector_metadata(
                    "mrmr",
                    k=len(selected_features),
                    k_requested="auto",
                    top_m=top_m_eff,
                    n_features=len(feature_names),
                    auto_k=True,
                    extra={
                        "task": task,
                        "estimator": estimator,
                        "formula": formula,
                        "relevance": relevance,
                        "auto_k_mode": auto_k_config.auto_k_mode,
                        "k_method": auto_k_config.k_method,
                        "auto_k_strategy": auto_k_config.strategy,
                    },
                ),
                return_result,
            )

        return selected_features

    if estimator == "gaussian":
        check_regression_only(task, estimator)
        method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
        if return_result:
            if cache is not None:
                selected_features, selected_indices = select_cached(
                    cache,
                    y,
                    k,
                    method=method,
                    top_m=top_m,
                    return_indices=True,
                )
            else:
                selected_features, selected_indices = _mrmr_gaussian(
                    X,
                    y,
                    k,
                    formula,
                    top_m,
                    cat_features,
                    cat_encoding,
                    allow_full_data_target_encoding,
                    sample_weight,
                    subsample,
                    random_state,
                    n_jobs,
                    rank_backend,
                    verbose,
                    return_indices=True,
                )

            if cache is not None:
                feature_names = (
                    cache.feature_names
                    if cache.feature_names is not None
                    else feature_names
                )
            elif not isinstance(X, pd.DataFrame):
                feature_names = [f"x{i}" for i in range(n_features_input)]

            return _to_filter_result(
                selected_features,
                selected_indices,
                _build_selector_metadata(
                    "mrmr",
                    k=len(selected_features),
                    k_requested=int(k),
                    top_m=_default_top_m(top_m, int(k)),
                    n_features=len(feature_names),
                    auto_k=False,
                    extra={
                        "task": task,
                        "estimator": estimator,
                        "formula": formula,
                        "relevance": relevance,
                    },
                ),
                return_result,
            )

        if cache is not None:
            return select_cached(cache, y, k, method=method, top_m=top_m)
        return _mrmr_gaussian(
            X,
            y,
            k,
            formula,
            top_m,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding,
            sample_weight,
            subsample,
            random_state,
            n_jobs,
            rank_backend,
            verbose,
        )

    if return_result:
        selected_features, selected_indices = _mrmr_classic(
            X,
            y,
            k,
            task,
            relevance,
            formula,
            top_m,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding,
            sample_weight,
            subsample,
            random_state,
            n_jobs,
            mrmr_backend_resolved,
            verbose,
            return_indices=True,
        )
        return _to_filter_result(
            selected_features,
            selected_indices,
            _build_selector_metadata(
                "mrmr",
                k=len(selected_features),
                k_requested=int(k),
                top_m=_default_top_m(top_m, int(k)),
                n_features=n_features_input,
                auto_k=False,
                extra={
                    "task": task,
                    "estimator": estimator,
                    "formula": formula,
                    "relevance": relevance,
                },
            ),
            return_result,
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
        allow_full_data_target_encoding,
        sample_weight,
        subsample,
        random_state,
        n_jobs,
        mrmr_backend_resolved,
        verbose,
        return_indices=False,
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
    allow_full_data_target_encoding,
    sample_weight,
    subsample,
    random_state,
    n_jobs,
    mrmr_backend,
    verbose,
    return_indices: bool = False,
) -> List[str] | Tuple[List[str], List[int]]:
    """Classic mRMR implementation."""
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

    top_m = _default_top_m(top_m, k)

    if verbose:
        print(f"mRMR classic: selecting {k} features from {X_arr.shape[1]} (top_m={top_m})")

    selected_idx = mrmr_select(
        X_arr,
        rel,
        k,
        formula=formula,
        top_m=top_m,
        sample_weight=w,
        n_jobs=n_jobs,
        mrmr_backend=mrmr_backend,
    )
    selected_features = [feature_names[i] for i in selected_idx]
    if return_indices:
        return selected_features, selected_idx.astype(int).tolist()
    return selected_features


def _mrmr_gaussian(
    X,
    y,
    k,
    formula,
    top_m,
    cat_features,
    cat_encoding,
    allow_full_data_target_encoding,
    sample_weight,
    subsample,
    random_state,
    n_jobs,
    rank_backend,
    verbose,
    return_indices: bool = False,
) -> List[str] | Tuple[List[str], List[int]]:
    """Gaussian mRMR via cached selection."""
    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    X = _encode_categoricals_for_selector(
        X,
        y,
        cat_features,
        cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
    )
    if verbose:
        print(f"mRMR gaussian: selecting {k} features (top_m={top_m})")
    cache = build_cache(
        X,
        sample_weight=sample_weight,
        subsample=subsample,
        random_state=random_state,
        n_jobs=n_jobs,
        rank_backend=rank_backend,
    )
    method = "mrmr_quot" if formula == "quotient" else "mrmr_diff"
    if return_indices:
        selected_features, selected_indices = select_cached(
            cache,
            y,
            k,
            method=method,
            top_m=top_m,
            return_indices=True,
        )
        return selected_features, selected_indices

    return select_cached(cache, y, k, method=method, top_m=top_m)
