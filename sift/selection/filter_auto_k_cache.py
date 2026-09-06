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


def remap_onehot_prefix_evaluate(
    *,
    path_names,
    eval_X: pd.DataFrame,
    onehot_raw_X=None,
    onehot_encoder=None,
    onehot_cat_features=None,
    onehot_max_levels=32,
    onehot_include_names=None,
    row_idx=None,
    encoded_prefix_sizes=None,
) -> dict | None:
    """Score prefix-evaluate on the raw frame with fold-local one-hot vocab."""
    if onehot_raw_X is None or onehot_encoder is None:
        return None
    if not isinstance(onehot_raw_X, pd.DataFrame):
        raise TypeError("onehot_raw_X must be a pandas DataFrame")
    raw = onehot_raw_X
    if row_idx is not None:
        idx = np.asarray(row_idx)
        if idx.size and idx.size < len(raw):
            raw = raw.iloc[idx]
    if len(raw) != len(eval_X):
        raise ValueError(
            "onehot raw evaluation frame is not aligned with the encoded eval_X"
        )
    include = list(onehot_include_names or ())
    include_set = set(include)
    encoded_names = list(path_names)
    collapsed = onehot_encoder.collapse_to_raw(encoded_names)
    path = [name for name in collapsed if name not in include_set]
    prefix_sizes = None
    if encoded_prefix_sizes:
        raw_sizes: list[int] = []
        for width in encoded_prefix_sizes:
            w = int(width)
            if w <= 0:
                continue
            raw_prefix = onehot_encoder.collapse_to_raw(encoded_names[:w])
            raw_sizes.append(len([name for name in raw_prefix if name not in include_set]))
        if raw_sizes and raw_sizes != list(range(1, len(raw_sizes) + 1)):
            prefix_sizes = tuple(raw_sizes)
    cats = list(onehot_cat_features or getattr(onehot_encoder, "cols", ()) or ())
    return {
        "eval_X": raw,
        "path": path,
        "cat_features": cats,
        "cat_encoding": "onehot",
        "onehot_max_levels": int(
            onehot_max_levels
            if onehot_max_levels is not None
            else getattr(onehot_encoder, "max_levels", 32)
        ),
        "base_features": include,
        "prefix_sizes": prefix_sizes,
    }


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
    from sift.selection.cefsplus_multi import as_regression_targets

    y_arr, _n_y = as_regression_targets(y, int(len(X_df)))
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
    onehot_raw_X=None,
    onehot_encoder=None,
    onehot_cat_features=None,
    onehot_max_levels: int = 32,
    onehot_include_names=None,
    onehot_row_idx=None,
) -> list[str] | tuple:
    from sift.selection.blocks import discovery_prefix_widths, resolve_feature_blocks

    path = [feature_names[i] for i in path_idx]
    named = bool(getattr(eval_X, "columns", None) is not None)
    blocks = resolve_feature_blocks(
        feature_blocks, feature_names=feature_names, named=named
    )
    prefix_sizes = discovery_prefix_widths(path_idx, blocks) if blocks is not None else None
    remapped = remap_onehot_prefix_evaluate(
        path_names=path,
        eval_X=eval_X,
        onehot_raw_X=onehot_raw_X,
        onehot_encoder=onehot_encoder,
        onehot_cat_features=onehot_cat_features,
        onehot_max_levels=onehot_max_levels,
        onehot_include_names=onehot_include_names,
        row_idx=onehot_row_idx,
        encoded_prefix_sizes=prefix_sizes,
    )
    eval_frame = eval_X
    eval_path = path
    eval_cat_features = cat_features
    eval_cat_encoding = cat_encoding
    eval_base = base_features
    eval_prefix = prefix_sizes
    eval_max_levels = onehot_max_levels
    if remapped is not None:
        eval_frame = remapped["eval_X"]
        eval_path = remapped["path"]
        eval_cat_features = remapped["cat_features"]
        eval_cat_encoding = remapped["cat_encoding"]
        eval_base = remapped["base_features"]
        eval_prefix = remapped["prefix_sizes"]
        eval_max_levels = remapped["onehot_max_levels"]
    _require_eval_split_context(auto_k_config, eval_groups, eval_time)
    best_k, selected, auto_diag = auto_k_module.select_k_auto(
        eval_frame,
        y_arr,
        eval_path,
        auto_k_config,
        groups=eval_groups,
        time=eval_time,
        task=task,
        cat_features=eval_cat_features,
        cat_encoding=eval_cat_encoding,
        onehot_max_levels=eval_max_levels,
        sample_weight=sample_weight,
        target_cv_n_splits=target_cv_n_splits,
        target_cv_smoothing=target_cv_smoothing,
        target_prior=target_prior,
        warmup_policy=warmup_policy,
        base_features=eval_base,
        within=within,
        prefix_sizes=eval_prefix,
    )
    if remapped is not None and onehot_encoder is not None:
        selected = onehot_encoder.expand_selected(selected)
        encoded_index = {name: i for i, name in enumerate(onehot_encoder.output_names_)}
        selected_idx = [
            int(encoded_index[name]) for name in selected if name in encoded_index
        ]
    else:
        selected_idx = [int(i) for i in path_idx[: len(selected)]]
    _print_selected_k("CV/holdout", best_k, verbose)
    path_steps = len(eval_prefix) if eval_prefix is not None else len(eval_path)
    result: tuple = (selected,)
    if return_indices:
        result += (selected_idx,)
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
