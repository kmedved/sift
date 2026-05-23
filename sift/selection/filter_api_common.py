"""Shared implementation helpers for function-style filter selectors."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

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
    ensure_weights,
    resolve_jmi_estimator,
    subsample_xy,
    validate_k,
    validate_inputs,
)
from sift.estimators import relevance as rel_est
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.auto_k import (
    AutoKConfig,
    resolve_auto_k_config,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
)
from sift.selection.cefsplus import select_cached
from sift.selection.api_helpers import (
    auto_k_summary as _auto_k_summary,
    build_selector_metadata as _build_selector_metadata,
    safe_name_indices as _safe_name_indices,
    to_filter_result as _to_filter_result,
    validate_groups_time as _validate_groups_time,
)
from sift.selection.loops import MrmrBackend, jmi_select, mrmr_select, resolve_mrmr_backend
from sift.selection.result import FilterSelectionResult


def _resolve_auto_k_config(
    auto_k_config: Optional[AutoKConfig],
    time: Optional[np.ndarray],
    groups: Optional[np.ndarray],
) -> AutoKConfig:
    """Resolve auto-k config, inferring strategy from available data."""
    return resolve_auto_k_config(auto_k_config, time, groups)


def _reject_unsupported_auto_k_method(
    *,
    selector_name: str,
    estimator: str,
    auto_k_config: AutoKConfig,
) -> None:
    """Fail before preprocessing for auto-k methods unsupported by this route."""
    if estimator == "gaussian":
        if auto_k_config.k_method == "penalized_objective":
            raise ValueError(
                "k_method='penalized_objective' is supported only for CEFS+ paths. "
                f"Use k_method='evaluate' for Gaussian {selector_name}."
            )
        return

    if auto_k_config.k_method != "evaluate":
        raise ValueError(
            "classic MRMR/JMI/JMIM auto-k supports only k_method='evaluate'. "
            "Use CEFS+ for k_method='elbow' or k_method='penalized_objective'."
        )


def _prepare_eval_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    cache: FeatureCache,
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray],
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Prepare evaluation data, respecting subsample indices from cache."""
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=cache.feature_names)
    y_arr = np.asarray(y).ravel()
    if len(y_arr) != len(X_df):
        raise ValueError(f"X has {len(X_df)} rows but y has {len(y_arr)}")

    use_cache_rows = cache.row_idx is not None and len(cache.row_idx) < len(X_df)
    if sample_weight is not None:
        w_arr = ensure_weights(sample_weight, len(X_df), normalize=True)
        eval_weight = w_arr[cache.row_idx] if use_cache_rows else w_arr
    else:
        eval_weight = np.asarray(cache.sample_weight, dtype=np.float64)

    if use_cache_rows:
        eval_X = X_df.iloc[cache.row_idx]
        eval_y = y_arr[cache.row_idx]
        eval_groups = groups[cache.row_idx] if groups is not None else None
        eval_time = time[cache.row_idx] if time is not None else None
    else:
        eval_X, eval_y = X_df, y_arr
        eval_groups, eval_time = groups, time

    return eval_X, eval_y, eval_groups, eval_time, eval_weight


def _resolve_cat_features(
    X: Union[pd.DataFrame, np.ndarray],
    cat_features: Optional[List[str]],
) -> Optional[List[str]]:
    if cat_features is None and isinstance(X, pd.DataFrame):
        return X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return cat_features


_SUPERVISED_CAT_ENCODINGS = frozenset({"target", "loo", "james_stein", "loo_logit"})


def _encode_categoricals_for_selector(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    *,
    allow_full_data_target_encoding: bool,
) -> Union[pd.DataFrame, np.ndarray]:
    """Encode categorical columns after enforcing explicit supervised opt-in."""
    if not cat_features or cat_encoding == "none":
        return X
    if not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
    present_cat_features = [col for col in cat_features if col in X.columns]
    if (
        present_cat_features
        and cat_encoding in _SUPERVISED_CAT_ENCODINGS
        and not allow_full_data_target_encoding
    ):
        raise ValueError(
            f"cat_encoding='{cat_encoding}' fits a supervised categorical encoder "
            "on the full dataset in function-style selectors. Pass "
            "allow_full_data_target_encoding=True to opt into this leakage-prone "
            "behavior, or set cat_encoding='none' and pre-encode categoricals in a "
            "leakage-safe pipeline."
        )
    return encode_categoricals(X, y, cat_features, cat_encoding)


def _auto_k_gaussian(
    *,
    cache: FeatureCache,
    y: np.ndarray,
    method: str,
    max_k: int,
    top_m: int,
    auto_k_config: AutoKConfig,
    eval_X: pd.DataFrame,
    eval_y: np.ndarray,
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray],
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    corr_prune: float | None | Literal["auto"] = "auto",
    verbose: bool = True,
    return_indices: bool = False,
    return_details: bool = False,
) -> List[str] | Tuple[List[str], List[int]]:
    """Shared auto-k logic for gaussian estimators."""
    want_indices = return_indices or return_details

    def _finish(selected, selected_idx, path, diag, summary):
        if return_details:
            return selected, selected_idx, diag, summary
        if return_indices:
            return selected, selected_idx
        return selected

    if auto_k_config.k_method == "penalized_objective":
        if method != "cefsplus":
            raise ValueError(
                "k_method='penalized_objective' is supported only for CEFS+ paths. "
                "Use k_method='evaluate' for Gaussian MRMR/JMI/JMIM."
            )
        if want_indices:
            path, path_indices, objective = select_cached(
                cache,
                y,
                max_k,
                method=method,
                top_m=top_m,
                corr_prune=corr_prune,
                return_objective=True,
                return_indices=True,
            )
            path_indices = list(path_indices)
        else:
            path, objective = select_cached(
                cache,
                y,
                max_k,
                method=method,
                top_m=top_m,
                corr_prune=corr_prune,
                return_objective=True,
            )
            path_indices = []
        best_k, auto_diag = select_k_penalized_objective(
            objective,
            auto_k_config,
            objective_scale="n_eff",
            n_samples=len(cache.sample_weight),
            sample_weight=cache.sample_weight,
            min_k=auto_k_config.min_k,
            max_k=len(path),
        )
        selected_count = min(best_k, len(path))
        if verbose:
            print(f"  Penalized objective selected k={selected_count}")
        selected = path[:selected_count]
        summary = _auto_k_summary(
            auto_k_config,
            selected_k=selected_count,
            path_length=len(path),
            effective_max_k=min(int(auto_k_config.max_k), len(path)),
            diagnostics=auto_diag,
            extra={
                "objective_penalty": auto_k_config.objective_penalty,
                "objective_scale": "gaussian_2mi",
                "proxy_only_objective": True,
            },
        )
        return _finish(selected, path_indices[:selected_count], path, auto_diag, summary)

    if auto_k_config.k_method == "elbow":
        if want_indices:
            path, path_indices, objective = select_cached(
                cache,
                y,
                max_k,
                method=method,
                top_m=top_m,
                corr_prune=corr_prune,
                return_objective=True,
                return_indices=True,
            )
            path_indices = list(path_indices)
        else:
            path, objective = select_cached(
                cache,
                y,
                max_k,
                method=method,
                top_m=top_m,
                corr_prune=corr_prune,
                return_objective=True,
            )
            path_indices = []
        elbow_k, auto_diag = select_k_elbow(
            objective,
            min_k=auto_k_config.min_k,
            max_k=len(path),
            min_rel_gain=auto_k_config.elbow_min_rel_gain,
            patience=auto_k_config.elbow_patience,
        )
        selected_count = min(elbow_k, len(path))
        if verbose:
            print(f"  Elbow selected k={selected_count}")
        selected = path[:selected_count]
        summary = _auto_k_summary(
            auto_k_config,
            selected_k=selected_count,
            path_length=len(path),
            effective_max_k=min(int(auto_k_config.max_k), len(path)),
            diagnostics=auto_diag,
            extra={"proxy_only_objective": True},
        )
        return _finish(selected, path_indices[:selected_count], path, auto_diag, summary)

    if auto_k_config.strategy == "time_holdout" and time is None:
        raise ValueError("auto-k evaluate with strategy='time_holdout' requires time parameter")
    if auto_k_config.strategy == "group_cv" and groups is None:
        raise ValueError("auto-k evaluate with strategy='group_cv' requires groups parameter")

    if want_indices:
        path, path_indices = select_cached(
            cache,
            y,
            max_k,
            method=method,
            top_m=top_m,
            corr_prune=corr_prune,
            return_indices=True,
        )
        path_indices = list(path_indices)
    else:
        path = select_cached(
            cache,
            y,
            max_k,
            method=method,
            top_m=top_m,
            corr_prune=corr_prune,
        )
        path_indices = []
    best_k, selected, auto_diag = select_k_auto(
        eval_X,
        eval_y,
        path,
        auto_k_config,
        groups=groups,
        time=time,
        task="regression",
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        sample_weight=sample_weight,
    )
    if verbose:
        print(f"  CV/holdout selected k={best_k}")
    summary = _auto_k_summary(
        auto_k_config,
        selected_k=len(selected),
        path_length=len(path),
        effective_max_k=min(int(auto_k_config.max_k), len(path)),
        diagnostics=auto_diag,
        extra={"proxy_only_objective": False},
    )
    return _finish(selected, path_indices[:len(selected)], path, auto_diag, summary)


def _auto_k_classic(
    *,
    y_arr: np.ndarray,
    eval_X: pd.DataFrame,
    feature_names: List[str],
    path_idx: np.ndarray,
    auto_k_config: AutoKConfig,
    eval_groups: Optional[np.ndarray],
    eval_time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray],
    task: Task,
    cat_features: Optional[List[str]],
    cat_encoding: CatEncoding,
    verbose: bool = True,
    return_indices: bool = False,
) -> List[str] | Tuple[List[str], List[int]]:
    """Shared auto-k evaluation for classic estimators."""
    if auto_k_config.k_method != "evaluate":
        raise ValueError(
            "classic MRMR/JMI/JMIM auto-k supports only k_method='evaluate'. "
            "Use CEFS+ for k_method='elbow' or k_method='penalized_objective'."
        )

    path = [feature_names[i] for i in path_idx]

    if auto_k_config.strategy == "time_holdout" and eval_time is None:
        raise ValueError("auto-k evaluate with strategy='time_holdout' requires time parameter")
    if auto_k_config.strategy == "group_cv" and eval_groups is None:
        raise ValueError("auto-k evaluate with strategy='group_cv' requires groups parameter")

    best_k, selected, _ = select_k_auto(
        eval_X,
        y_arr,
        path,
        auto_k_config,
        groups=eval_groups,
        time=eval_time,
        task=task,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        sample_weight=sample_weight,
    )
    if verbose:
        print(f"  CV/holdout selected k={best_k}")
    if return_indices:
        selected_indices = [int(i) for i in path_idx[: len(selected)]]
        return selected, selected_indices
    return selected


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
    allow_full_data_target_encoding: bool,
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
    if isinstance(X, pd.DataFrame) and cat_features is None:
        cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    X = _encode_categoricals_for_selector(
        X,
        y,
        cat_features,
        cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
    )

    X_arr, y_arr, feature_names = validate_inputs(X, y, task)
    X_arr, y_arr, w, row_idx = subsample_xy(
        X_arr,
        y_arr,
        subsample,
        random_state,
        sample_weight=sample_weight,
        return_idx=True,
    )
    return X_arr, y_arr, w, feature_names, row_idx


def _compute_relevance(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    w: np.ndarray,
    task: Task,
    relevance: RelevanceMethod,
) -> np.ndarray:
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

    return rel_funcs[relevance](X_arr, y_arr, w)
