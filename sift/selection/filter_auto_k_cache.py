"""Cache, evaluation, and classic auto-k route helpers."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from sift._preprocess import ensure_weights
from sift.selection import auto_k as auto_k_module
from sift.selection.auto_k import AutoKConfig
from sift.selection.filter_auto_k_common import (
    _effective_max_k,
    _print_selected_k,
    _require_eval_split_context,
    auto_k_summary,
)

EvalData = tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]


def prepare_filter_eval_data(
    X, y: np.ndarray, cache, groups: Optional[np.ndarray], time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray], feature_names: Optional[list[str]] = None,
) -> EvalData:
    columns = cache.feature_names if cache.feature_names is not None else feature_names
    if isinstance(X, pd.DataFrame):
        if _cache_uses_synthetic_feature_names(cache):
            _require_positional_cache_dataframe_alignment(cache, X)
        X_df = X
    else:
        X_df = pd.DataFrame(X, columns=columns)
    y_arr = np.asarray(y).ravel()
    if len(y_arr) != len(X_df):
        raise ValueError(f"X has {len(X_df)} rows but y has {len(y_arr)}")
    if getattr(cache, "n_rows_original", len(X_df)) != len(X_df):
        raise ValueError(
            f"cache was built with {cache.n_rows_original} rows but X has {len(X_df)} rows"
        )

    use_cache_rows = cache.row_idx is not None and len(cache.row_idx) < len(X_df)
    if sample_weight is not None:
        w_arr = ensure_weights(sample_weight, len(X_df), normalize=True)
        eval_weight = w_arr[cache.row_idx] if use_cache_rows else w_arr
    else:
        eval_weight = np.asarray(cache.sample_weight, dtype=np.float64)

    if not use_cache_rows:
        return X_df, y_arr, groups, time, eval_weight
    return (
        X_df.iloc[cache.row_idx],
        y_arr[cache.row_idx],
        groups[cache.row_idx] if groups is not None else None,
        time[cache.row_idx] if time is not None else None,
        eval_weight,
    )


class CachedFilterPath(tuple):
    """Path triple plus additional-block raw prefix widths.

    Unpacks as ``(names, indices, objective)`` so existing callers stay valid.
    ``prefix_widths[t-1]`` is the raw discovery length after ``t`` additional
    blocks (or columns when there are no blocks).
    """

    def __new__(cls, names, indices, objective, prefix_widths):
        inst = tuple.__new__(cls, (names, indices, objective))
        inst.prefix_widths = tuple(int(width) for width in prefix_widths)
        return inst


def _cached_filter_path(
    cache, y, k: int, *, method: str, top_m: int, corr_prune,
    want_indices: bool, return_objective: bool, callback=None,
    include=None, exclude=None, candidates=None, feature_blocks=None,
) -> CachedFilterPath:
    from sift.selection.blocks import discovery_prefix_widths, resolve_feature_blocks
    from sift.selection.cefsplus import _select_cached_impl
    from sift.selection.conditioning import named_feature_space

    result = _select_cached_impl(
        cache,
        y,
        k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        return_indices=want_indices,
        return_objective=return_objective,
        warn_noise_floor=False,
        callback=callback,
        include=include,
        exclude=exclude,
        candidates=candidates,
        feature_blocks=feature_blocks,
        compose_include=False,
    )
    if return_objective and want_indices:
        path, indices, objective = result
        indices = list(indices)
    elif return_objective:
        path, objective = result
        indices = []
    elif want_indices:
        path, indices = result
        indices = list(indices)
        objective = None
    else:
        path, indices, objective = result, [], None
    cache_names = list(cache.feature_names) if cache.feature_names is not None else [
        f"x{i}"
        for i in range(int(np.max(cache.valid_cols)) + 1 if len(cache.valid_cols) else 0)
    ]
    named = named_feature_space(
        cache.feature_names,
        synthetic=bool(getattr(cache, "feature_names_are_synthetic", False))
        or cache.feature_names is None,
    )
    blocks = resolve_feature_blocks(
        feature_blocks, feature_names=cache_names, named=named
    )
    widths = discovery_prefix_widths(indices, blocks)
    if not widths and path:
        widths = tuple(range(1, len(path) + 1))
    if (
        objective is not None
        and widths
        and len(np.asarray(objective).ravel()) == len(path)
        and len(widths) != len(path)
    ):
        objective = np.asarray(objective, dtype=np.float64).ravel()[
            np.asarray(widths, dtype=np.int64) - 1
        ]
    path = _PathList(path, widths)
    return CachedFilterPath(path, indices, objective, widths)


class _PathList(list):
    """Selected-name list carrying additional-block prefix widths."""

    def __init__(self, names, prefix_widths):
        super().__init__(names)
        self.prefix_widths = tuple(int(width) for width in prefix_widths)


def _cache_uses_synthetic_feature_names(cache) -> bool:
    return bool(getattr(cache, "feature_names_are_synthetic", False))


def _require_positional_cache_dataframe_alignment(cache, X: pd.DataFrame) -> None:
    cache_names = list(cache.feature_names or [])
    if list(X.columns) == cache_names:
        return
    raise ValueError(
        "Gaussian auto-k with a cache built from unnamed/positional features "
        "requires X to use the cache's synthetic column names in the same order. "
        "Build the cache from the named DataFrame, pass ndarray input, or rename "
        "X columns to match the positional cache names."
    )


def select_filter_classic_auto_k(
    *, y_arr: np.ndarray, eval_X: pd.DataFrame, feature_names: list[str], path_idx: np.ndarray,
    auto_k_config: AutoKConfig, eval_groups: Optional[np.ndarray], eval_time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray],
    task: Literal["regression", "classification"],
    cat_features: Optional[list[str]], cat_encoding: str,
    within: str | None = None,
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    verbose: bool = True,
    return_indices: bool = False,
    return_diagnostics: bool = False,
    base_features: list | None = None,
    feature_blocks=None,
) -> list[str] | tuple:
    from sift.selection.blocks import discovery_prefix_widths, resolve_feature_blocks

    path = [feature_names[i] for i in path_idx]
    named = bool(getattr(eval_X, "columns", None) is not None)
    blocks = resolve_feature_blocks(
        feature_blocks, feature_names=feature_names, named=named
    )
    prefix_sizes = discovery_prefix_widths(path_idx, blocks) if blocks is not None else None
    _require_eval_split_context(auto_k_config, eval_groups, eval_time)
    best_k, selected, auto_diag = auto_k_module.select_k_auto(
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
        target_cv_n_splits=target_cv_n_splits,
        target_cv_smoothing=target_cv_smoothing,
        target_prior=target_prior,
        warmup_policy=warmup_policy,
        base_features=base_features,
        within=within,
        prefix_sizes=prefix_sizes,
    )
    _print_selected_k("CV/holdout", best_k, verbose)
    path_steps = len(prefix_sizes) if prefix_sizes is not None else len(path)
    result: tuple = (selected,)
    if return_indices:
        raw = len(selected)
        result += ([int(i) for i in path_idx[:raw]],)
    if return_diagnostics:
        summary = auto_k_summary(
            auto_k_config,
            selected_k=int(best_k),
            path_length=path_steps,
            effective_max_k=_effective_max_k(auto_k_config, path_steps),
            diagnostics=auto_diag,
        )
        result += (auto_diag, summary)
    if len(result) == 1:
        return selected
    return result
