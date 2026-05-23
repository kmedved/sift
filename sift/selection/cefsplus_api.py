"""CEFS+ function-style selector API."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import CatEncoding, validate_k
from sift.estimators.copula import FeatureCache, build_cache
from sift.selection.auto_k import AutoKConfig
from sift.selection.cefsplus import select_cached
from sift.selection.filter_api_common import (
    _auto_k_gaussian,
    _build_selector_metadata,
    _default_top_m,
    _encode_categoricals_for_selector,
    _prepare_eval_data,
    _resolve_auto_k_config,
    _resolve_cat_features,
    _safe_name_indices,
    _to_filter_result,
    _validate_groups_time,
)
from sift.selection.result import FilterSelectionResult


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
            if auto_k_config.k_method == "elbow":
                mode = "elbow"
            elif auto_k_config.k_method == "penalized_objective":
                mode = f"penalized_objective/{auto_k_config.objective_penalty}"
            else:
                mode = f"evaluate/{auto_k_config.strategy}/{auto_k_config.selection_rule}"
            print(
                f"CEFS+ auto-k ({mode}): building path to {max_k} features "
                f"(top_m={top_m_eff}, corr_prune={corr_prune})"
            )

        eval_X, eval_y, eval_groups, eval_time, eval_weight = _prepare_eval_data(
            X_raw, y, cache, groups, time, sample_weight
        )

        if return_result:
            selected_features, selected_indices, auto_diag, auto_summary = _auto_k_gaussian(
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
                return_details=True,
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
            auto_diag = None
            auto_summary = None

        if return_result:
            if selected_indices is None:
                selected_indices = _safe_name_indices(
                    cache.feature_names
                    if cache is not None and cache.feature_names is not None
                    else [f"x{i}" for i in range(n_features_input)],
                    selected_features,
                )
            return FilterSelectionResult(
                selected_features=selected_features,
                selected_indices=selected_indices,
                selector_metadata=_build_selector_metadata(
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
                        "selection_rule": auto_k_config.selection_rule,
                        "objective_penalty": auto_k_config.objective_penalty
                        if auto_k_config.k_method == "penalized_objective"
                        else None,
                    },
                ),
                diagnostics_={
                    "auto_k": auto_summary,
                    "auto_k_diagnostics": auto_diag,
                },
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
