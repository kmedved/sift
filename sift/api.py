"""User-facing API for feature selection."""

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
)
from sift.selection.cefsplus import select_cached
from sift.selection.loops import MrmrBackend, jmi_select, mrmr_select, resolve_mrmr_backend
from sift.selection.result import FilterSelectionResult


def _resolve_auto_k_config(
    auto_k_config: Optional[AutoKConfig],
    time: Optional[np.ndarray],
    groups: Optional[np.ndarray],
) -> AutoKConfig:
    """Resolve auto-k config, inferring strategy from available data."""
    return resolve_auto_k_config(auto_k_config, time, groups)


def _validate_groups_time(
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    n_rows: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate and coerce groups/time arrays."""
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != n_rows:
            raise ValueError(f"groups has {len(groups)} elements but X has {n_rows} rows")
    if time is not None:
        time = np.asarray(time).reshape(-1)
        if len(time) != n_rows:
            raise ValueError(f"time has {len(time)} elements but X has {n_rows} rows")
    return groups, time


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


def _to_filter_result(
    selected_features: List[str],
    selected_indices: Optional[List[int]],
    selector_metadata: dict,
    return_result: bool,
) -> List[str] | FilterSelectionResult:
    """Return a result object only when requested, otherwise list output."""
    if return_result:
        return FilterSelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            selector_metadata=selector_metadata,
        )
    return selected_features


def _build_selector_metadata(
    selector: str,
    *,
    k: int | str,
    k_requested: int | str,
    top_m: Optional[int],
    n_features: int,
    auto_k: bool,
    extra: Optional[dict] = None,
) -> dict:
    """Build a concise metadata payload for filter-result return mode."""
    metadata = {
        "selector": selector,
        "k_requested": k_requested,
        "k": k,
        "top_m": top_m,
        "n_features": int(n_features),
        "auto_k": auto_k,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _safe_name_indices(
    feature_names: List[str],
    selected_features: List,  # names or indices from selectors
) -> Optional[List[int]]:
    """Map selected names to indices when the mapping is unambiguous."""
    if len(feature_names) != len(set(feature_names)):
        return None

    if not selected_features:
        return []

    if all(isinstance(v, (int, np.integer)) for v in selected_features):
        selected_indices = [int(v) for v in selected_features]
        if not all(0 <= i < len(feature_names) for i in selected_indices):
            return None
        return selected_indices

    index_map = {feature: idx for idx, feature in enumerate(feature_names)}
    selected_indices = []
    for name in selected_features:
        idx = index_map.get(name)
        if idx is None:
            return None
        selected_indices.append(idx)
    return selected_indices


_SUPERVISED_CAT_ENCODINGS = frozenset({"target", "loo", "james_stein"})


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
) -> List[str] | Tuple[List[str], List[int]]:
    """Shared auto-k logic for gaussian estimators."""
    if auto_k_config.k_method == "elbow":
        if return_indices:
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
        elbow_k, _ = select_k_elbow(
            objective,
            min_k=auto_k_config.min_k,
            max_k=len(path),
            min_rel_gain=auto_k_config.elbow_min_rel_gain,
            patience=auto_k_config.elbow_patience,
        )
        if verbose:
            print(f"  Elbow selected k={elbow_k}")
        selected = path[:elbow_k]
        if return_indices:
            return selected, path_indices[:elbow_k]
        return selected

    if auto_k_config.strategy == "time_holdout" and time is None:
        raise ValueError("auto-k evaluate with strategy='time_holdout' requires time parameter")
    if auto_k_config.strategy == "group_cv" and groups is None:
        raise ValueError("auto-k evaluate with strategy='group_cv' requires groups parameter")

    if return_indices:
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
    best_k, selected, _ = select_k_auto(
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
    if return_indices:
        return selected, path_indices[:len(selected)]
    return selected


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
    if auto_k_config.k_method == "elbow":
        raise ValueError(
            "k_method='elbow' is not supported for classic MRMR/JMI/JMIM "
            "auto-k paths. Use k_method='evaluate' with time/groups, or use a "
            "Gaussian/cache-backed path that exposes an objective path."
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
    """
    Joint Mutual Information feature selection.

    score(f) = Σ_{s ∈ S} I(f, s; y)

    Fixed k is a maximum. The selector may return fewer than k features when
    fewer valid candidates remain after filtering.
    """
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    n_features_input = int(np.asarray(X).shape[1])
    groups, time = _validate_groups_time(groups, time, n_rows)
    estimator = resolve_jmi_estimator(estimator, task)
    k = validate_k(k)

    check_regression_only(task, estimator)
    if estimator == "ksg" and sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")

    if k == "auto":
        auto_k_config = _resolve_auto_k_config(auto_k_config, time, groups)
        cat_features = _resolve_cat_features(X, cat_features)

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if estimator == "gaussian":
            if verbose:
                print(
                    f"JMI gaussian auto-k: building path to {max_k} features (top_m={top_m_eff})"
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

            if return_result:
                selected_features, selected_indices = _auto_k_gaussian(
                    cache=cache,
                    y=y,
                    method="jmi",
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
                    method="jmi",
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
                        "jmi",
                        k=len(selected_features),
                        k_requested="auto",
                        top_m=top_m_eff,
                        n_features=n_features_input,
                        auto_k=True,
                        extra={
                            "task": task,
                            "estimator": estimator,
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
                f"JMI classic auto-k: building path to {max_k} features (top_m={top_m_eff})"
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
                    "jmi",
                    k=len(selected_features),
                    k_requested="auto",
                    top_m=top_m_eff,
                    n_features=len(feature_names),
                    auto_k=True,
                    extra={
                        "task": task,
                        "estimator": estimator,
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
        if isinstance(X, pd.DataFrame) and cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        X = _encode_categoricals_for_selector(
            X,
            y,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding=allow_full_data_target_encoding,
        )
        if verbose:
            print(f"JMI gaussian: selecting {k} features (top_m={top_m})")
        if cache is None:
            cache = build_cache(
                X,
                sample_weight=sample_weight,
                subsample=subsample,
                random_state=random_state,
            )
        if return_result:
            selected_features, selected_indices = select_cached(
                cache,
                y,
                k,
                method="jmi",
                top_m=top_m,
                return_indices=True,
            )
        else:
            selected_features = select_cached(cache, y, k, method="jmi", top_m=top_m)
            selected_indices = None

        if return_result:
            feature_names = (
                cache.feature_names
                if cache.feature_names is not None
                else [f"x{i}" for i in range(n_features_input)]
            )
            return _to_filter_result(
                selected_features,
                selected_indices,
                _build_selector_metadata(
                    "jmi",
                    k=len(selected_features),
                    k_requested=int(k),
                    top_m=_default_top_m(top_m, int(k)),
                    n_features=len(feature_names),
                    auto_k=False,
                    extra={
                        "task": task,
                        "estimator": estimator,
                        "relevance": relevance,
                    },
                ),
                return_result,
            )

        return selected_features

    if return_result:
        selected_features, selected_indices = _jmi_classic(
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
            allow_full_data_target_encoding,
            sample_weight,
            subsample,
            random_state,
            verbose,
            return_indices=True,
        )
        return _to_filter_result(
            selected_features,
            selected_indices,
            _build_selector_metadata(
                "jmi",
                k=len(selected_features),
                k_requested=int(k),
                top_m=_default_top_m(top_m, int(k)),
                n_features=n_features_input,
                auto_k=False,
                extra={
                    "task": task,
                    "estimator": estimator,
                    "relevance": relevance,
                },
            ),
            return_result,
        )

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
        allow_full_data_target_encoding,
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
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> List[str] | FilterSelectionResult:
    """
    JMI Maximization — conservative variant.

    score(f) = min_{s ∈ S} I(f, s; y)

    Fixed k is a maximum. The selector may return fewer than k features when
    fewer valid candidates remain after filtering.
    """
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    n_features_input = int(np.asarray(X).shape[1])
    groups, time = _validate_groups_time(groups, time, n_rows)
    estimator = resolve_jmi_estimator(estimator, task)
    k = validate_k(k)

    check_regression_only(task, estimator)
    if estimator == "ksg" and sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")

    if k == "auto":
        auto_k_config = _resolve_auto_k_config(auto_k_config, time, groups)
        cat_features = _resolve_cat_features(X, cat_features)

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if estimator == "gaussian":
            if verbose:
                print(
                    f"JMIM gaussian auto-k: building path to {max_k} features (top_m={top_m_eff})"
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

            if return_result:
                selected_features, selected_indices = _auto_k_gaussian(
                    cache=cache,
                    y=y,
                    method="jmim",
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
                    method="jmim",
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
                        "jmim",
                        k=len(selected_features),
                        k_requested="auto",
                        top_m=top_m_eff,
                        n_features=n_features_input,
                        auto_k=True,
                        extra={
                            "task": task,
                            "estimator": estimator,
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
                f"JMIM classic auto-k: building path to {max_k} features (top_m={top_m_eff})"
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
                    "jmim",
                    k=len(selected_features),
                    k_requested="auto",
                    top_m=top_m_eff,
                    n_features=len(feature_names),
                    auto_k=True,
                    extra={
                        "task": task,
                        "estimator": estimator,
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
        if isinstance(X, pd.DataFrame) and cat_features is None:
            cat_features = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        X = _encode_categoricals_for_selector(
            X,
            y,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding=allow_full_data_target_encoding,
        )
        if verbose:
            print(f"JMIM gaussian: selecting {k} features (top_m={top_m})")
        if cache is None:
            cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)
        if return_result:
            selected_features, selected_indices = select_cached(
                cache,
                y,
                k,
                method="jmim",
                top_m=top_m,
                return_indices=True,
            )
        else:
            selected_features = select_cached(cache, y, k, method="jmim", top_m=top_m)
            selected_indices = None

        if return_result:
            feature_names = (
                cache.feature_names
                if cache.feature_names is not None
                else [f"x{i}" for i in range(n_features_input)]
            )
            return _to_filter_result(
                selected_features,
                selected_indices,
                _build_selector_metadata(
                    "jmim",
                    k=len(selected_features),
                    k_requested=int(k),
                    top_m=_default_top_m(top_m, int(k)),
                    n_features=len(feature_names),
                    auto_k=False,
                    extra={
                        "task": task,
                        "estimator": estimator,
                        "relevance": relevance,
                    },
                ),
                return_result,
            )

        return selected_features

    if return_result:
        selected_features, selected_indices = _jmi_classic(
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
            allow_full_data_target_encoding,
            sample_weight,
            subsample,
            random_state,
            verbose,
            return_indices=True,
        )
        return _to_filter_result(
            selected_features,
            selected_indices,
            _build_selector_metadata(
                "jmim",
                k=len(selected_features),
                k_requested=int(k),
                top_m=_default_top_m(top_m, int(k)),
                n_features=n_features_input,
                auto_k=False,
                extra={
                    "task": task,
                    "estimator": estimator,
                    "relevance": relevance,
                },
            ),
            return_result,
        )

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
        allow_full_data_target_encoding,
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
        sample_weight=None if mi_estimator == "ksg" else w,
    )

    selected_features = [feature_names[i] for i in selected_idx]
    if return_indices:
        return selected_features, selected_idx.astype(int).tolist()
    return selected_features


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
    corr_prune: float | None = 0.95,
    cat_features: Optional[List[str]] = None,
    cat_encoding: CatEncoding = "loo",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> List[str] | FilterSelectionResult:
    """
    CEFS+ feature selection using log-det Gaussian MI proxy.

    REGRESSION ONLY.

    Fixed k is a maximum. The selector may return fewer than k features when
    fewer valid or unpruned candidates remain.
    """
    n_rows = X.shape[0] if hasattr(X, "shape") else len(X)
    groups, time = _validate_groups_time(groups, time, n_rows)
    k = validate_k(k)
    X_raw = X
    n_features_input = int(np.asarray(X_raw).shape[1])
    cat_features = _resolve_cat_features(X, cat_features)
    from sift._preprocess import to_numpy

    y_arr = to_numpy(y, dtype=np.float32).ravel()
    n_rows = X_raw.shape[0] if hasattr(X_raw, "shape") else len(X_raw)
    if len(y_arr) != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {len(y_arr)}")
    if not np.isfinite(y_arr).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")
    if k == "auto":
        auto_k_config = _resolve_auto_k_config(auto_k_config, time, groups)

        max_k = auto_k_config.max_k
        top_m_eff = _default_top_m(top_m, max_k)

        if cache is None:
            X = _encode_categoricals_for_selector(
                X_raw,
                y,
                cat_features,
                cat_encoding,
                allow_full_data_target_encoding=allow_full_data_target_encoding,
            )
            cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)

        if verbose:
            mode = "elbow" if auto_k_config.k_method == "elbow" else f"evaluate/{auto_k_config.strategy}"
            print(
                f"CEFS+ auto-k ({mode}): building path to {max_k} features "
                f"(top_m={top_m_eff}, corr_prune={corr_prune})"
            )

        eval_X, eval_y, eval_groups, eval_time, eval_weight = _prepare_eval_data(
            X_raw, y, cache, groups, time, sample_weight
        )

        if return_result:
            selected_features, selected_indices = _auto_k_gaussian(
                cache=cache,
                y=y,
                method="cefsplus",
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
                corr_prune=corr_prune,
                verbose=verbose,
                return_indices=True,
            )
        else:
            selected_features = _auto_k_gaussian(
                cache=cache,
                y=y,
                method="cefsplus",
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
                corr_prune=corr_prune,
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
                    "cefsplus",
                    k=len(selected_features),
                    k_requested="auto",
                    top_m=top_m_eff,
                    n_features=n_features_input,
                    auto_k=True,
                    extra={
                        "auto_k_mode": auto_k_config.auto_k_mode,
                        "k_method": auto_k_config.k_method,
                        "auto_k_strategy": auto_k_config.strategy,
                    },
                ),
                return_result,
            )

        return selected_features

    k_int = int(k)
    top_m_eff = _default_top_m(top_m, k_int)
    if verbose:
        print(f"CEFS+: selecting {k_int} features (top_m={top_m_eff}, corr_prune={corr_prune})")
    if cache is None:
        X = _encode_categoricals_for_selector(
            X_raw,
            y,
            cat_features,
            cat_encoding,
            allow_full_data_target_encoding=allow_full_data_target_encoding,
        )
        cache = build_cache(X, sample_weight=sample_weight, subsample=subsample, random_state=random_state)
    if return_result:
        selected_features, selected_indices = select_cached(
            cache,
            y,
            k_int,
            method="cefsplus",
            top_m=top_m_eff,
            corr_prune=corr_prune,
            return_indices=True,
        )
        return _to_filter_result(
            selected_features,
            selected_indices,
            _build_selector_metadata(
                "cefsplus",
                k=len(selected_features),
                k_requested=k_int,
                top_m=top_m_eff,
                n_features=n_features_input,
                auto_k=False,
            ),
            return_result,
        )

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
