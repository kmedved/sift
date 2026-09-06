"""Payload builders for function-style filter selector APIs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sift._logging import logger
from sift._preprocess import (
    CatEncoding,
    RelevanceMethod,
    Task,
    TargetCVEncoder,
    OneHotBlockEncoder,
    validate_onehot_max_levels,
    check_regression_only,
    encode_categoricals,
    ensure_weights,
    subsample_xy,
    to_numpy,
    validate_inputs,
    validate_target,
)
from sift.estimators import relevance as rel_est
from sift.estimators.classic_cache import is_classic_cache
from sift.estimators.copula import (
    FeatureCache,
    build_cache,
    gaussian_mi_from_corr,
    weighted_corr_with_vector,
    weighted_rank_gauss_1d,
    weighted_rank_gauss_2d,
)
from sift.selection import auto_k as auto_k_module
from sift.selection.cefsplus import select_cached
from sift.selection.cefsplus_binary import make_diagnostics
from sift.selection.cefsplus_binary_common import (
    BinaryOptions,
    BinaryPathRun,
    BinaryProblem,
    BinarySelection,
    binary_selection_prefix,
    build_binary_logloss_path,
    prepare_binary_problem,
    validate_binary_options,
)
from sift.selection.filter_auto_k import (
    _AUTOK_FIELD_DEFAULTS,
    _strip_router_only_fields,
    AUTO_K_CURVE_KEY,
    auto_k_mode_label,
    build_auto_k_curve_payload,
    prepare_filter_eval_data,
    select_binary_changepoint,
    select_binary_elbow,
    select_binary_evaluate,
    select_binary_penalized,
    select_binary_posterior,
    select_gaussian_auto_path,
    select_filter_classic_auto_k,
    select_gaussian_changepoint_path,
    select_gaussian_chi2_path,
    select_gaussian_consensus_path,
    select_gaussian_cv_path,
    select_gaussian_elbow_path,
    select_gaussian_evaluate_path,
    select_gaussian_forward_stop_path,
    select_gaussian_knockoff_path,
    select_gaussian_penalized_path,
    select_gaussian_perm_gap_path,
    select_gaussian_posterior_path,
    select_gaussian_stability_path,
    select_gaussian_xfit_objective_path,
)
from sift.selection.conditioning import (
    compose_selected,
    conditioning_record,
    require_supported_auto_k,
)
from sift.selection.loops import jmi_select, mrmr_select
from sift.selection.panel import build_candidate_panel
from sift.selection.proxies import proxy_frame_from_panel, reject_unavailable_proxy_positions
from sift.selection.within import (
    as_float_feature_matrix,
    fit_transform_within,
    group_level_design,
    restore_feature_matrix,
)

if TYPE_CHECKING:
    from sift.selection.filter_api import FilterContext


@dataclass(frozen=True)
class ClassicPrepared:
    X_arr: np.ndarray
    y_arr: np.ndarray
    w: np.ndarray
    mi_w: np.ndarray
    feature_names: list[str]
    row_idx: np.ndarray
    target_cv_metadata: dict | None = None
    eval_sample_weight: np.ndarray | None = None
    X_pre_within: np.ndarray | None = None
    y_pre_within: np.ndarray | None = None
    groups_sub: np.ndarray | None = None


@dataclass(frozen=True)
class SelectionPayload:
    selected_features: list[str] | None = None
    selected_indices: list[int] | None = None
    top_m: int | None = None
    n_features: int | None = None
    metadata_extra: dict | None = None
    ranking: pd.DataFrame | None = None
    diagnostics: dict | None = None
    proxy_correlations: pd.DataFrame | None = None


ClassicPath = Callable[["FilterContext", ClassicPrepared, int, int], np.ndarray]
GaussianMethod = Callable[["FilterContext"], str]
GaussianRunner = Callable[..., tuple[list[str], list[int], pd.DataFrame, dict]]


def make_fixed_classic(path_func: ClassicPath) -> Callable[["FilterContext"], SelectionPayload]:
    def fixed_classic(ctx: "FilterContext") -> SelectionPayload:
        prep = _prepare_xy_classic(ctx)
        k = int(ctx.k)
        top_m = _default_top_m(_kw(ctx, "top_m"), k)
        if _kw(ctx, "verbose"):
            logger.info(
                f"{ctx.spec.display_name} classic: selecting {k} features from "
                f"{prep.X_arr.shape[1]} (top_m={top_m})"
            )
        selected_idx = path_func(ctx, prep, k, top_m)
        selected_idx, selected = _compose_classic_selection(ctx, prep, selected_idx)
        ranking = None
        diagnostics = None
        if ctx.request.return_result:
            relevance = _compute_relevance(
                prep.X_arr,
                prep.y_arr,
                prep.w,
                ctx.request.task,
                _kw(ctx, "relevance"),
            )
            ranking = _path_ranking(
                prep.feature_names,
                selected_idx,
                relevance,
                ctx.spec.selector,
                within_relevance=_within_relevance(ctx, relevance),
                between_relevance=_between_relevance_classic(ctx, prep),
                block_ids=_block_id_column(ctx, len(prep.feature_names)),
            )
            diagnostics = {
                "path_relevance": relevance[np.asarray(selected_idx, dtype=np.int64)].astype(float).tolist(),
            }
            cond = _conditioning_diag(ctx, selected_idx)
            if cond is not None:
                diagnostics["conditioning"] = cond
        return SelectionPayload(
            selected_features=selected,
            selected_indices=np.asarray(selected_idx, dtype=np.int64).astype(int).tolist()
            if ctx.request.return_result
            else None,
            top_m=top_m,
            n_features=ctx.n_features_input,
            ranking=ranking,
            diagnostics=diagnostics,
            metadata_extra=_combine_metadata_extra(
                prep.target_cv_metadata,
                _row_run_extra(ctx, prep.row_idx),
                _classic_cache_run_extra(ctx),
            ),
        )

    return fixed_classic


def make_auto_classic(path_func: ClassicPath) -> Callable[["FilterContext"], SelectionPayload]:
    def auto_classic(ctx: "FilterContext") -> SelectionPayload:
        assert ctx.auto_k_config is not None
        prep = _prepare_xy_classic(ctx)
        max_k = int(ctx.auto_k_config.max_k)
        top_m = _default_top_m(_kw(ctx, "top_m"), max_k)
        if _kw(ctx, "verbose"):
            logger.info(
                f"{ctx.spec.display_name} classic auto-k: building path to {max_k} "
                f"features (top_m={top_m})"
            )
        path_idx = path_func(ctx, prep, max_k, top_m)
        X_eval = (
            ctx.request.X
            if isinstance(ctx.request.X, pd.DataFrame)
            else pd.DataFrame(ctx.request.X, columns=prep.feature_names)
        )
        X_eval = X_eval.iloc[prep.row_idx]
        want_result = bool(ctx.request.return_result)
        outcome = select_filter_classic_auto_k(
            y_arr=(
                prep.y_pre_within if prep.y_pre_within is not None else prep.y_arr
            ),
            eval_X=X_eval,
            feature_names=prep.feature_names,
            path_idx=path_idx,
            auto_k_config=ctx.auto_k_config,
            eval_groups=ctx.groups[prep.row_idx] if ctx.groups is not None else None,
            eval_time=ctx.time[prep.row_idx] if ctx.time is not None else None,
            sample_weight=prep.eval_sample_weight,
            within=ctx.within,
            task=ctx.request.task,
            cat_features=_kw(ctx, "cat_features"),
            cat_encoding=_kw(ctx, "cat_encoding"),
            target_cv_n_splits=_kw(ctx, "target_cv_n_splits", 5),
            target_cv_smoothing=_kw(ctx, "target_cv_smoothing", "auto"),
            target_prior=_kw(ctx, "target_prior"),
            warmup_policy=_kw(ctx, "warmup_policy", "zero_weight"),
            verbose=_kw(ctx, "verbose"),
            return_indices=True,
            return_diagnostics=want_result,
            base_features=_include_names(ctx),
            feature_blocks=_kw(ctx, "feature_blocks"),
            onehot_raw_X=getattr(ctx, "onehot_source_X", None),
            onehot_encoder=getattr(ctx, "onehot_encoder", None),
            onehot_cat_features=list(getattr(ctx, "onehot_cat_features", None) or ()),
            onehot_max_levels=_kw(ctx, "onehot_max_levels", 32),
            onehot_include_names=_raw_onehot_include_names(ctx),
            onehot_row_idx=prep.row_idx,
        )
        ranking = None
        diagnostics = None
        if not want_result:
            selected, selected_indices = outcome
        else:
            selected, selected_indices, auto_diag, auto_summary = outcome
        composed_idx, selected = _compose_classic_selection(
            ctx, prep, np.asarray(selected_indices, dtype=np.int64)
        )
        selected_indices = composed_idx.tolist()
        if want_result:
            relevance = _compute_relevance(
                prep.X_arr,
                prep.y_arr,
                prep.w,
                ctx.request.task,
                _kw(ctx, "relevance"),
            )
            ranking = _path_ranking(
                prep.feature_names,
                composed_idx,
                relevance,
                ctx.spec.selector,
                within_relevance=_within_relevance(ctx, relevance),
                between_relevance=_between_relevance_classic(ctx, prep),
            )
            diagnostics = {
                "path_relevance": relevance[composed_idx].astype(float).tolist(),
                "auto_k": auto_summary,
                "auto_k_diagnostics": auto_diag,
                AUTO_K_CURVE_KEY: build_auto_k_curve_payload(
                    k_method=ctx.auto_k_config.k_method,
                    diagnostics=auto_diag,
                    summary=auto_summary,
                ),
            }
            cond = _conditioning_diag(ctx, composed_idx)
            if cond is not None:
                diagnostics["conditioning"] = cond
        return SelectionPayload(
            selected_features=selected,
            selected_indices=selected_indices,
            top_m=top_m,
            n_features=len(prep.feature_names),
            ranking=ranking,
            diagnostics=diagnostics,
            metadata_extra=_combine_metadata_extra(
                prep.target_cv_metadata,
                _row_run_extra(ctx, prep.row_idx),
                _classic_cache_run_extra(ctx),
            ),
        )

    return auto_classic


def make_fixed_gaussian(method_func: GaussianMethod) -> Callable[["FilterContext"], SelectionPayload]:
    def fixed_gaussian(ctx: "FilterContext") -> SelectionPayload:
        cache, _, _, target_cv_metadata, y_sel, X_pre = _cache_for_gaussian(ctx)
        method = method_func(ctx)
        k = int(ctx.k)
        top_m = _default_top_m(_kw(ctx, "top_m"), k)
        if _kw(ctx, "verbose"):
            logger.info(f"{ctx.spec.display_name}: selecting {k} features (top_m={top_m})")
        if ctx.request.return_result:
            selected, selected_indices, objective = select_cached(
                cache,
                y_sel,
                k,
                method=method,
                top_m=top_m,
                corr_prune=_kw(ctx, "corr_prune", "auto"),
                return_indices=True,
                return_objective=True,
                callback=ctx.request.callback,
                include=_kw(ctx, "include"),
                exclude=_kw(ctx, "exclude"),
                candidates=_kw(ctx, "candidates"),
                feature_blocks=_kw(ctx, "feature_blocks"),
            )
        else:
            selected, selected_indices = select_cached(
                cache,
                y_sel,
                k,
                method=method,
                top_m=top_m,
                corr_prune=_kw(ctx, "corr_prune", "auto"),
                return_indices=True,
                callback=ctx.request.callback,
                include=_kw(ctx, "include"),
                exclude=_kw(ctx, "exclude"),
                candidates=_kw(ctx, "candidates"),
                feature_blocks=_kw(ctx, "feature_blocks"),
            )
            objective = None
        selected_features, selected_indices, n_features = _gaussian_payload_selection(
            ctx,
            cache,
            selected,
            selected_indices,
        )
        proxy_correlations = _gaussian_proxy_correlations(
            ctx,
            cache=cache,
            method=method,
            k=k,
            top_m=top_m,
            selected_indices=selected_indices,
            y=y_sel,
        )
        ranking = None
        diagnostics = None
        if ctx.request.return_result:
            objective_arr = np.asarray(objective, dtype=np.float64)
            diagnostics = {
                "objective_path": objective_arr.astype(float).tolist(),
                "objective_gain": np.diff(
                    np.concatenate(([0.0], objective_arr))
                ).astype(float).tolist(),
            }
            if selected_indices is not None:
                relevance = _gaussian_relevance_for_input(ctx, cache, y=y_sel)
                ranking = _path_ranking(
                    ctx.feature_names,
                    selected_indices,
                    relevance,
                    ctx.spec.selector,
                    within_relevance=_within_relevance(ctx, relevance),
                    between_relevance=_between_relevance_gaussian(
                        ctx, cache, X_pre
                    ),
                    block_ids=_block_id_column(ctx, len(ctx.feature_names)),
                )
            cond = _conditioning_diag(ctx, selected_indices or [])
            if cond is not None:
                diagnostics = dict(diagnostics or {})
                diagnostics["conditioning"] = cond
        return SelectionPayload(
            selected_features=selected_features,
            selected_indices=selected_indices,
            top_m=top_m,
            n_features=n_features,
            ranking=ranking,
            diagnostics=diagnostics,
            proxy_correlations=proxy_correlations,
            metadata_extra=_combine_metadata_extra(
                target_cv_metadata,
                _cache_run_extra(cache, prebuilt=ctx.request.cache is not None),
                _multi_target_run_extra(cache, y_sel),
            ),
        )

    return fixed_gaussian


def make_auto_gaussian(
    method_func: GaussianMethod,
    runner: GaussianRunner,
    *,
    include_diagnostics: bool = False,
    include_objective_penalty: bool = False,
) -> Callable[["FilterContext"], SelectionPayload]:
    def auto_gaussian(ctx: "FilterContext") -> SelectionPayload:
        assert ctx.auto_k_config is not None
        if ctx.conditioning is not None and getattr(ctx.conditioning, "active", False):
            require_supported_auto_k(ctx.auto_k_config.k_method)
        cache, cat_features, effective_weight, target_cv_metadata, y_sel, X_pre = (
            _cache_for_gaussian(ctx)
        )
        top_m = _default_top_m(_kw(ctx, "top_m"), int(ctx.auto_k_config.max_k))
        if _kw(ctx, "verbose"):
            logger.info(
                f"{ctx.spec.display_name} auto-k ({auto_k_mode_label(ctx.auto_k_config)}): "
                f"building path to {ctx.auto_k_config.max_k} features (top_m={top_m})"
            )
        eval_y_source = ctx.request.y if ctx.within is not None else y_sel
        eval_X, eval_y, eval_groups, eval_time, eval_weight = prepare_filter_eval_data(
            ctx.request.X,
            eval_y_source,
            cache,
            ctx.groups,
            ctx.time,
            effective_weight,
            feature_names=ctx.feature_names,
        )
        method = method_func(ctx)
        selected, selected_indices, auto_diag, auto_summary = runner(
            cache=cache,
            y=y_sel,
            method=method,
            max_k=int(ctx.auto_k_config.max_k),
            top_m=top_m,
            auto_k_config=ctx.auto_k_config,
            eval_X=eval_X,
            eval_y=eval_y,
            groups=eval_groups,
            time=eval_time,
            sample_weight=eval_weight,
            source_groups=ctx.groups,
            source_time=ctx.time,
            cat_features=cat_features,
            cat_encoding=_kw(ctx, "cat_encoding"),
            corr_prune=_kw(ctx, "corr_prune", "auto"),
            feature_names=ctx.feature_names,
            verbose=_kw(ctx, "verbose"),
            callback=ctx.request.callback,
            target_cv_n_splits=_kw(ctx, "target_cv_n_splits", 5),
            target_cv_smoothing=_kw(ctx, "target_cv_smoothing", "auto"),
            target_prior=_kw(ctx, "target_prior"),
            warmup_policy=_kw(ctx, "warmup_policy", "zero_weight"),
            include=_kw(ctx, "include"),
            exclude=_kw(ctx, "exclude"),
            candidates=_kw(ctx, "candidates"),
            include_names=_include_names(ctx),
            feature_blocks=_kw(ctx, "feature_blocks"),
            within=ctx.within,
            within_X=X_pre,
            within_y=ctx.request.y,
            onehot_raw_X=getattr(ctx, "onehot_source_X", None),
            onehot_max_levels=_kw(ctx, "onehot_max_levels", 32),
            onehot_cat_features=list(getattr(ctx, "onehot_cat_features", None) or ()),
            onehot_raw_blocks=getattr(ctx, "raw_feature_blocks", None),
            onehot_encoder=getattr(ctx, "onehot_encoder", None),
            onehot_include_names=_raw_onehot_include_names(ctx),
        )
        selected, selected_indices = _compose_gaussian_auto_selection(
            ctx, selected, selected_indices
        )
        selected_features, selected_indices, n_features = _gaussian_payload_selection(
            ctx,
            cache,
            selected,
            selected_indices,
        )
        proxy_correlations = _gaussian_proxy_correlations(
            ctx,
            cache=cache,
            method=method,
            k=int(ctx.auto_k_config.max_k),
            top_m=top_m,
            selected_indices=selected_indices,
            y=y_sel,
        )
        ranking = None
        diagnostics = None
        if ctx.request.return_result:
            diagnostics = {}
            if include_diagnostics:
                diagnostics.update(
                    {"auto_k": auto_summary, "auto_k_diagnostics": auto_diag}
                )
            diagnostics[AUTO_K_CURVE_KEY] = build_auto_k_curve_payload(
                k_method=ctx.auto_k_config.k_method,
                diagnostics=auto_diag,
                summary=auto_summary,
            )
            if selected_indices is not None:
                relevance = _gaussian_relevance_for_input(ctx, cache, y=y_sel)
                ranking = _path_ranking(
                    ctx.feature_names,
                    selected_indices,
                    relevance,
                    ctx.spec.selector,
                    within_relevance=_within_relevance(ctx, relevance),
                    between_relevance=_between_relevance_gaussian(
                        ctx, cache, X_pre
                    ),
                    block_ids=_block_id_column(ctx, len(ctx.feature_names)),
                )
            cond = _conditioning_diag(ctx, selected_indices or [])
            if cond is not None:
                diagnostics["conditioning"] = cond
        metadata_extra = {}
        if target_cv_metadata:
            metadata_extra.update(target_cv_metadata)
        if include_objective_penalty:
            metadata_extra["objective_penalty"] = ctx.auto_k_config.objective_penalty
        metadata_extra.update(
            _cache_run_extra(cache, prebuilt=ctx.request.cache is not None)
        )
        metadata_extra.update(_multi_target_run_extra(cache, y_sel))
        return SelectionPayload(
            selected_features=selected_features,
            selected_indices=selected_indices,
            top_m=top_m,
            n_features=n_features,
            metadata_extra=metadata_extra,
            ranking=ranking,
            diagnostics=diagnostics,
            proxy_correlations=proxy_correlations,
        )

    return auto_gaussian


def _gaussian_proxy_correlations(
    ctx: "FilterContext",
    *,
    cache: FeatureCache,
    method: str,
    k: int,
    top_m: int,
    selected_indices: list[int] | None,
    y=None,
) -> pd.DataFrame | None:
    if not ctx.request.store_proxies:
        return None
    if selected_indices is None:
        raise ValueError(
            "store_proxies=True requires unambiguous raw selected-feature positions"
        )
    from sift.selection.blocks import map_blocks_to_valid
    from sift.selection.filter_auto_k_common import _conditioning_valid_sets

    protect, pool = _conditioning_valid_sets(
        cache,
        {
            "include": _kw(ctx, "include"),
            "exclude": _kw(ctx, "exclude"),
            "candidates": _kw(ctx, "candidates"),
        },
    )
    block_members = None
    if getattr(ctx, "feature_blocks", None) is not None:
        _orig_ids, block_members = map_blocks_to_valid(
            ctx.feature_blocks, cache.valid_cols
        )
    reject_unavailable_proxy_positions(
        selected_indices,
        available_original=cache.valid_cols,
        feature_names=cache.feature_names,
    )
    panel = build_candidate_panel(
        cache,
        ctx.request.y if y is None else y,
        k,
        top_m=top_m,
        corr_prune=_kw(ctx, "corr_prune", "auto"),
        method=method,
        protect_valid=protect,
        pool_valid=pool,
        block_members=block_members,
    )
    return proxy_frame_from_panel(
        panel.R,
        candidate_indices=panel.original,
        selected_indices=selected_indices,
    )


def binary_fixed_payload(ctx: "FilterContext") -> SelectionPayload:
    problem, options, run = _build_binary_run(ctx)
    selection = binary_selection_prefix(run.path, len(run.path.selected_features))
    return _binary_payload_from_selection(ctx, problem, options, run, selection)


def binary_auto_evaluate_payload(ctx: "FilterContext") -> SelectionPayload:
    return _binary_auto_payload(ctx, select_binary_evaluate)


def binary_auto_elbow_payload(ctx: "FilterContext") -> SelectionPayload:
    return _binary_auto_payload(ctx, select_binary_elbow)


def binary_auto_penalized_payload(ctx: "FilterContext") -> SelectionPayload:
    return _binary_auto_payload(ctx, select_binary_penalized)


def binary_auto_posterior_payload(ctx: "FilterContext") -> SelectionPayload:
    return _binary_auto_payload(ctx, select_binary_posterior)


def binary_auto_changepoint_payload(ctx: "FilterContext") -> SelectionPayload:
    return _binary_auto_payload(ctx, select_binary_changepoint)


def _reject_binary_auto_dense_options(config) -> None:
    """Binary CEFS+ has no dense-regime diagnostic; reject the opt-in explicitly.

    Silently ignoring ``auto_dense_check`` would let a user believe the
    EBIC-vs-CV cross-check ran when it cannot; keep the contract honest until a
    binary dense diagnostic exists.
    """
    defaults = _AUTOK_FIELD_DEFAULTS
    fields = (
        "auto_dense_check",
        "auto_dense_min_k",
        "auto_dense_min_frac",
        "auto_dense_disagreement_ratio",
    )
    changed = [
        name for name in fields if getattr(config, name) != getattr(defaults, name)
    ]
    if changed:
        raise ValueError(
            "auto_dense_* options are not supported for binary log-loss CEFS+ "
            f"automatic routing (no log-loss dense-regime diagnostic exists): {changed}. "
            "Remove them, use loss='brier' for Gaussian-proxy binary selection, "
            "or run the Gaussian select_cefsplus router for the dense check."
        )


def binary_auto_auto_payload(ctx: "FilterContext") -> SelectionPayload:
    assert ctx.auto_k_config is not None
    auto_k_module.validate_auto_k_config(ctx.auto_k_config)
    _reject_binary_auto_dense_options(ctx.auto_k_config)
    routed = _strip_router_only_fields(
        replace(
            ctx.auto_k_config,
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=0,
        )
    )
    with auto_k_module._suppress_auto_k_unused_field_warnings():
        payload = _binary_auto_payload(
            replace(ctx, auto_k_config=routed),
            select_binary_penalized,
        )
    if payload.diagnostics and "auto_k" in payload.diagnostics:
        summary = dict(payload.diagnostics["auto_k"])
        summary["method"] = "auto"
        summary["routed_method"] = "penalized_objective"
        summary["auto_routing"] = {
            "chosen": "penalized_objective",
            "objective_penalty": "ebic",
            "reason": "binary_cefsplus_measured_default",
            "facts": {"selector_method": "cefsplus_binary"},
        }
        payload.diagnostics["auto_k"] = summary
    return payload


def make_mrmr_classic_path() -> ClassicPath:
    return _mrmr_classic_path


def make_jmi_classic_path(*, aggregation: str, pass_sample_weight: bool) -> ClassicPath:
    def jmi_classic_path(
        ctx: "FilterContext",
        prep: ClassicPrepared,
        k: int,
        top_m: int,
    ) -> np.ndarray:
        rel = _compute_relevance(prep.X_arr, prep.y_arr, prep.w, ctx.request.task, _kw(ctx, "relevance"))
        return jmi_select(
            prep.X_arr,
            prep.y_arr,
            k,
            rel,
            mi_estimator=ctx.estimator,
            aggregation=aggregation,
            top_m=top_m,
            y_kind="discrete" if ctx.request.task == "classification" else "continuous",
            sample_weight=(
                prep.mi_w if pass_sample_weight and ctx.estimator == "binned" else prep.w
            )
            if pass_sample_weight
            else None,
            callback=ctx.request.callback,
            include_idx=_include_indices(ctx),
            candidate_idx=_candidate_indices(ctx),
            blocks=_block_member_arrays(ctx),
        )

    return jmi_classic_path


def mrmr_gaussian_method(ctx: "FilterContext") -> str:
    return "mrmr_quot" if _kw(ctx, "formula") == "quotient" else "mrmr_diff"


def selector_gaussian_method(selector: str) -> GaussianMethod:
    def gaussian_method(_ctx: "FilterContext") -> str:
        return selector

    return gaussian_method


def _gaussian_payload_selection(
    ctx: "FilterContext",
    cache: FeatureCache,
    selected,
    selected_indices,
) -> tuple[list[str], list[int] | None, int]:
    input_feature_names = list(ctx.feature_names)
    selected_features = list(selected)
    cache_is_synthetic = _cache_uses_synthetic_feature_names(cache)
    if not cache_is_synthetic and len(set(input_feature_names)) == len(input_feature_names):
        input_index = {name: idx for idx, name in enumerate(input_feature_names)}
    else:
        input_index = {}
    if input_index and all(name in input_index for name in selected_features):
        return (
            selected_features,
            [int(input_index[name]) for name in selected_features],
            len(input_feature_names),
        )

    feature_names = _gaussian_feature_names(ctx, cache)
    indices = [int(i) for i in selected_indices] if selected_indices is not None else None
    if cache_is_synthetic and indices is not None:
        return [feature_names[i] for i in indices], indices, len(feature_names)
    if indices is not None and list(feature_names) == input_feature_names:
        return [feature_names[i] for i in indices], indices, len(feature_names)
    return selected_features, None, len(input_feature_names)


def _gaussian_feature_names(ctx: "FilterContext", cache: FeatureCache) -> list[str]:
    if cache.feature_names is not None:
        return list(cache.feature_names)
    return [f"x{i}" for i in range(ctx.n_features_input)]


def _cache_uses_synthetic_feature_names(cache: FeatureCache) -> bool:
    return cache.feature_names is None or bool(
        getattr(cache, "feature_names_are_synthetic", False)
    )


def validate_standard(ctx: "FilterContext") -> None:
    check_regression_only(ctx.request.task, ctx.estimator)
    _reject_multi_target_unless_cefsplus(ctx)
    cache = ctx.request.cache
    if cache is not None and ctx.estimator == "gaussian" and is_classic_cache(cache):
        raise ValueError(
            "ClassicFeatureCache cannot be used with estimator='gaussian'; "
            "build a FeatureCache with sift.build_cache"
        )
    if cache is not None and ctx.estimator != "gaussian" and not is_classic_cache(cache):
        raise ValueError("cache is supported only with estimator='gaussian'")
    if ctx.estimator == "gaussian":
        # Gaussian/cache paths bypass validate_inputs, so check the regression
        # target here; otherwise non-finite y rows would silently be treated
        # as neutral (zero) ranks by the copula transform.
        validate_cefsplus(ctx)


def validate_ksg_no_weight(ctx: "FilterContext") -> None:
    validate_standard(ctx)
    if ctx.request.sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")
    cache = ctx.request.cache
    if is_classic_cache(cache) and bool(cache.weights_supplied):
        raise ValueError(
            "estimator='ksg' does not support sample_weight; this "
            "ClassicFeatureCache was built with sample_weight"
        )


def _reject_multi_target_unless_cefsplus(ctx: "FilterContext") -> None:
    arr = np.asarray(ctx.request.y)
    if arr.ndim >= 2 and int(arr.shape[1]) > 1 and ctx.spec.selector != "cefsplus":
        raise ValueError(
            "2-D y is only supported for select_cefsplus / CEFSPlusSelector "
            "and select_cached(method='cefsplus'); "
            f"got selector={ctx.spec.selector!r}"
        )


def validate_cefsplus(ctx: "FilterContext") -> None:
    if ctx.request.cache is not None and ctx.request.sample_weight is not None:
        raise ValueError(
            "sample_weight is already fixed by the supplied cache; "
            "pass weights to build_cache instead"
        )
    from sift.selection.cefsplus_multi import (
        as_regression_targets,
        reject_unsupported_multi_target_context,
    )

    y_arr, n_y = as_regression_targets(ctx.request.y, int(ctx.n_rows))
    if not np.isfinite(np.asarray(y_arr, dtype=np.float64)).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")
    if n_y >= 2:
        k_method = None
        if ctx.k == "auto" and ctx.auto_k_config is not None:
            k_method = ctx.auto_k_config.k_method
        reject_unsupported_multi_target_context(
            n_targets=n_y,
            selector=ctx.spec.selector,
            method="cefsplus" if ctx.spec.selector == "cefsplus" else ctx.spec.selector,
            within=ctx.within,
            cat_encoding=_kw(ctx, "cat_encoding", "none"),
            k_method=k_method,
        )
    if ctx.spec.selector == "cefsplus" and n_y == 1:
        # select_cefsplus has no task parameter, so unlike the task-aware
        # selectors nothing else flags a labels-shaped target here.
        _warn_if_multiclass_labels_as_regression_target(np.asarray(y_arr).reshape(-1))


def _warn_if_multiclass_labels_as_regression_target(y_arr: np.ndarray) -> None:
    """Warn when a continuous-target selector receives labels-shaped y."""
    if y_arr.size == 0 or not np.all(y_arr == np.rint(y_arr)):
        return
    n_unique = int(np.unique(y_arr).size)
    if 3 <= n_unique <= 20:
        warnings.warn(
            "select_cefsplus treats y as a continuous regression target, but y "
            f"contains only {n_unique} distinct integer-valued levels and looks "
            "like multiclass labels. Use a task='classification' selector (or "
            "one-vs-rest targets with select_cefsplus_binary) for categorical "
            "targets; ignore this warning if y is a genuine numeric target.",
            UserWarning,
            stacklevel=5,
        )


def standard_extra(aggregation: str | None = None) -> Callable[["FilterContext"], dict]:
    def metadata_extra(ctx: "FilterContext") -> dict:
        extra = {"task": ctx.request.task, "estimator": ctx.estimator}
        for name in ("formula", "relevance"):
            if name in ctx.selector_kwargs:
                extra[name] = _kw(ctx, name)
        if aggregation is not None:
            extra["aggregation"] = aggregation
        return extra

    return metadata_extra


def no_extra(_ctx: "FilterContext") -> dict:
    return {}


def target_cv_metadata_from_encoder(encoding_cv: dict | None) -> dict | None:
    """Normalize a fitted encoder's ``encoding_cv_`` into result metadata.

    The fitted encoder is the only authority on the fold kind and the effective
    split count, so metadata is never reconstructed from the request.  ``None``
    means no categorical encoding ran and nothing is attached.
    """
    if not encoding_cv:
        return None
    return {
        "cat_encoding": "target_cv",
        "encoding_cv": {
            "kind": encoding_cv["kind"],
            "n_splits": int(encoding_cv["n_splits"]),
        },
    }


def _binary_auto_payload(
    ctx: "FilterContext",
    handler: Callable[..., BinarySelection],
) -> SelectionPayload:
    assert ctx.auto_k_config is not None
    problem, options, run = _build_binary_run(ctx)
    handler_kwargs = {}
    if ctx.auto_k_config.k_method == "evaluate":
        handler_kwargs.update(
            target_cv_n_splits=_kw(ctx, "target_cv_n_splits", 5),
            target_cv_smoothing=_kw(ctx, "target_cv_smoothing", "auto"),
            target_prior=_kw(ctx, "target_prior"),
            warmup_policy=_kw(ctx, "warmup_policy", "zero_weight"),
            onehot_raw_X=getattr(ctx, "onehot_source_X", None),
            onehot_encoder=getattr(ctx, "onehot_encoder", None),
            onehot_cat_features=list(getattr(ctx, "onehot_cat_features", None) or ()),
            onehot_max_levels=_kw(ctx, "onehot_max_levels", 32),
            onehot_include_names=_raw_onehot_include_names(ctx),
        )
    selection = handler(
        ctx.request.X,
        problem,
        run,
        options,
        auto_k_config=ctx.auto_k_config,
        cat_encoding=_kw(ctx, "cat_encoding"),
        verbose=_kw(ctx, "verbose"),
        **handler_kwargs,
    )
    return _binary_payload_from_selection(ctx, problem, options, run, selection)


def _build_binary_run(ctx: "FilterContext") -> tuple[BinaryProblem, BinaryOptions, BinaryPathRun]:
    options = _binary_logloss_options(ctx)
    problem = prepare_binary_problem(
        ctx.request.X,
        ctx.request.y,
        groups=ctx.groups,
        time=ctx.time,
        sample_weight=ctx.request.sample_weight,
        class_weight=_kw(ctx, "class_weight"),
    )
    run = build_binary_logloss_path(
        ctx.request.X,
        problem,
        options,
        auto_k_config=ctx.auto_k_config,
        cat_features=_kw(ctx, "cat_features"),
        cat_encoding=_kw(ctx, "cat_encoding"),
        allow_full_data_target_encoding=_kw(ctx, "allow_full_data_target_encoding"),
        random_state=_kw(ctx, "random_state"),
        verbose=_kw(ctx, "verbose"),
        target_cv_n_splits=_kw(ctx, "target_cv_n_splits", 5),
        target_cv_smoothing=_kw(ctx, "target_cv_smoothing", "auto"),
        target_prior=_kw(ctx, "target_prior"),
        warmup_policy=_kw(ctx, "warmup_policy", "zero_weight"),
        callback=ctx.request.callback,
        include_idx=_include_indices(ctx),
        candidate_idx=_candidate_indices(ctx),
        feature_blocks=getattr(ctx, "feature_blocks", None),
    )
    return problem, options, run


def _binary_logloss_options(ctx: "FilterContext") -> BinaryOptions:
    options = validate_binary_options(
        ctx.k,
        loss=_kw(ctx, "loss"),
        top_m=_kw(ctx, "top_m"),
        corr_prune=_kw(ctx, "corr_prune"),
        subsample=_kw(ctx, "subsample"),
        ridge=_kw(ctx, "ridge"),
        refit_every=_kw(ctx, "refit_every"),
        cat_encoding=_kw(ctx, "cat_encoding"),
        loo_smoothing=_kw(ctx, "loo_smoothing"),
        loo_clip_min=_kw(ctx, "loo_clip_min"),
        loo_clip_max=_kw(ctx, "loo_clip_max"),
        sample_weight=ctx.request.sample_weight,
        class_weight=_kw(ctx, "class_weight"),
    )
    if options.loss != "logloss":
        raise ValueError("CEFS+ binary spec dispatch supports only loss='logloss'")
    return options


def _binary_payload_from_selection(
    ctx: "FilterContext", problem: BinaryProblem, options: BinaryOptions,
    run: BinaryPathRun, selection: BinarySelection,
) -> SelectionPayload:
    include_idx = _include_indices(ctx)
    if include_idx.size:
        names, indices = compose_selected(
            run.feature_names,
            include_idx,
            selection.selected_original,
        )
        selection = BinarySelection(
            selected_features=names,
            selected_original=indices,
            selected_scores=selection.selected_scores,
            auto_diag=selection.auto_diag,
            auto_objective=selection.auto_objective,
            auto_summary=selection.auto_summary,
        )
    if not ctx.request.return_result:
        return SelectionPayload(
            selected_features=selection.selected_features,
            top_m=run.top_m_eff,
            n_features=problem.n_features_input,
        )

    diagnostics = make_diagnostics(run.path)
    cond = _conditioning_diag(ctx, selection.selected_original)
    if cond is not None:
        diagnostics["conditioning"] = cond
    diagnostics.update(
        {
            "subsample_row_idx": None
            if len(run.row_idx) == problem.n_rows
            else run.row_idx.astype(int).tolist(),
            "cat_features_requested": list(run.cat_features or []),
            "cat_features_used": [
                col for col in (run.cat_features or []) if col in run.feature_names
            ],
        }
    )
    if options.k_value == "auto":
        assert ctx.auto_k_config is not None
        diagnostics["auto_k"] = selection.auto_summary
        diagnostics["auto_k_diagnostics"] = selection.auto_diag
        diagnostics[AUTO_K_CURVE_KEY] = build_auto_k_curve_payload(
            k_method=ctx.auto_k_config.k_method,
            diagnostics=selection.auto_diag,
            summary=selection.auto_summary,
        )
        if selection.auto_objective is not None:
            diagnostics["auto_k_objective"] = selection.auto_objective.tolist()

    ranking = _path_ranking(
        run.feature_names,
        selection.selected_original,
        run.path.univariate_scores,
        "cefsplus_binary",
        block_ids=_block_id_column(ctx, len(run.feature_names)),
    )
    path_score_by_index: dict[int, float] = {}
    widths = run.prefix_widths or getattr(run.path, "prefix_widths", ()) or ()
    discovery = list(run.path.selected_original)
    if widths and selection.selected_scores:
        start = 0
        n_steps = min(len(selection.selected_scores), len(widths))
        for t in range(n_steps):
            end = int(widths[t])
            score = float(selection.selected_scores[t])
            for orig in discovery[start:end]:
                path_score_by_index[int(orig)] = score
            start = end
    else:
        path_score_by_index = dict(
            zip(discovery, selection.selected_scores)
        )
    ranking.insert(
        ranking.columns.get_loc("selector"),
        "score",
        [path_score_by_index.get(int(i), np.nan) for i in ranking["selected_index"]],
    )
    metadata_extra = {
        "loss": "logloss",
        "weighted": problem.weighted,
        "class_weight": _kw(ctx, "class_weight"),
        "class_weight_scope": "pre_subsample"
        if _kw(ctx, "class_weight") is not None
        else None,
        "target_mapping": problem.target_mapping,
        "ridge": options.ridge,
        "refit_every": options.refit_every,
        "corr_prune": options.corr_prune,
        "subsample": options.subsample,
        "random_state": _kw(ctx, "random_state"),
        "cat_encoding": _kw(ctx, "cat_encoding"),
        "loo_smoothing": options.loo_smoothing,
        "loo_clip_min": options.loo_clip_min,
        "loo_clip_max": options.loo_clip_max,
    }
    encoding_metadata = target_cv_metadata_from_encoder(run.encoding_cv)
    if encoding_metadata is not None:
        metadata_extra.update(encoding_metadata)
    if options.k_value == "auto" and ctx.auto_k_config is not None:
        metadata_extra.update(_binary_auto_metadata(ctx.auto_k_config))
    metadata_extra.update(_row_run_extra(ctx, run.row_idx))
    return SelectionPayload(
        selected_features=selection.selected_features,
        selected_indices=selection.selected_original,
        top_m=run.top_m_eff,
        n_features=problem.n_features_input,
        metadata_extra=metadata_extra,
        ranking=ranking,
        diagnostics=diagnostics,
    )


def _binary_auto_metadata(auto_k_config) -> dict:
    is_penalized = auto_k_config.k_method == "penalized_objective"
    return {
        "objective_penalty": auto_k_config.objective_penalty if is_penalized else None,
        "binary_objective_mode": auto_k_config.binary_objective_mode
        if is_penalized
        else None,
    }


def _combine_metadata_extra(*parts) -> dict | None:
    extra: dict = {}
    for part in parts:
        if part:
            extra.update(part)
    return extra or None


def _multi_target_run_extra(cache: FeatureCache, y) -> dict:
    arr = np.asarray(y)
    if arr.ndim < 2 or int(arr.shape[1]) < 2:
        return {}
    from sift.selection.cefsplus_multi import copula_target_condition, result_target_metadata

    n_targets, cond = copula_target_condition(cache, y)
    if n_targets < 2:
        return {}
    return result_target_metadata(n_targets, target_condition=cond)


def _cache_run_extra(cache: FeatureCache, *, prebuilt: bool) -> dict:
    n_used = int(np.asarray(cache.row_idx).reshape(-1).size)
    extra = {
        "n_rows_original": int(cache.n_rows_original),
        "n_rows_used": n_used,
    }
    if prebuilt:
        extra["n_rows_cached"] = n_used
        extra["cache_backed"] = True
        extra["feature_names_are_synthetic"] = bool(
            getattr(cache, "feature_names_are_synthetic", False)
        )
    return extra


def _row_run_extra(ctx: "FilterContext", row_idx) -> dict:
    return {
        "n_rows_original": int(ctx.n_rows),
        "n_rows_used": int(np.asarray(row_idx).reshape(-1).size),
    }


def _classic_cache_run_extra(ctx: "FilterContext") -> dict:
    cache = ctx.request.cache
    if not is_classic_cache(cache):
        return {}
    extra = {
        "cache_kind": "classic",
        "cache_backed": True,
        "n_rows_cached": int(np.asarray(cache.row_idx).reshape(-1).size),
        "feature_names_are_synthetic": bool(cache.feature_names_are_synthetic),
        "weights_supplied": bool(cache.weights_supplied),
        "subsample": cache.subsample,
    }
    if cache.subsample_applied:
        extra["random_state"] = int(cache.random_state)
    return extra


def _cache_for_gaussian(
    ctx: "FilterContext",
) -> tuple[
    FeatureCache,
    list[str] | None,
    np.ndarray | None,
    dict | None,
    np.ndarray | object,
    object,
]:
    cat_features = _resolve_cat_features(ctx.request.X, _kw(ctx, "cat_features"))
    y_sel = ctx.request.y
    X_pre = ctx.request.X
    if ctx.request.cache is not None:
        if (
            _kw(ctx, "cat_encoding", "none") in {"target_cv", "onehot"}
            and cat_features
        ):
            raise ValueError(
                f"cat_encoding={_kw(ctx, 'cat_encoding')!r} cannot be combined "
                "with a prebuilt Gaussian cache because the cache has no "
                "encoding provenance"
            )
        return (
            ctx.request.cache,
            cat_features,
            ctx.request.sample_weight,
            None,
            y_sel,
            X_pre,
        )
    X_encoded = _encode_categoricals_for_selector(
        ctx.request.X,
        ctx.request.y,
        cat_features,
        _kw(ctx, "cat_encoding"),
        allow_full_data_target_encoding=_kw(ctx, "allow_full_data_target_encoding"),
        sample_weight=ctx.request.sample_weight,
        task=ctx.request.task,
        groups=ctx.groups,
        time=ctx.time,
        target_cv_n_splits=_kw(ctx, "target_cv_n_splits", 5),
        target_cv_smoothing=_kw(ctx, "target_cv_smoothing", "auto"),
        target_prior=_kw(ctx, "target_prior"),
        warmup_policy=_kw(ctx, "warmup_policy", "zero_weight"),
        onehot_max_levels=_kw(ctx, "onehot_max_levels", 32),
    )
    X_encoded, effective_weight, target_cv_metadata = X_encoded
    X_pre = X_encoded
    if ctx.within is not None:
        X_arr, template = as_float_feature_matrix(X_encoded)
        y_arr = to_numpy(ctx.request.y, dtype=np.float64).ravel()
        n_rows = X_arr.shape[0]
        weights = (
            np.ones(n_rows, dtype=np.float64)
            if effective_weight is None
            else ensure_weights(effective_weight, n_rows, normalize=False)
        )
        X_arr, y_arr, _fitted = fit_transform_within(
            ctx.within,
            X_arr,
            y_arr,
            ctx.groups,
            ctx.time,
            weights,
        )
        X_encoded = restore_feature_matrix(template, X_arr)
        y_sel = y_arr
        positive = weights > 0.0
        if np.any(positive) and not np.any(np.ptp(X_arr[positive], axis=0) > 0.0):
            raise ValueError(
                "within demeaning removed all feature variation; "
                "no within-entity signal remains"
            )
    return (
        build_cache(
            X_encoded,
            sample_weight=effective_weight,
            subsample=_kw(ctx, "subsample", 50_000),
            random_state=_kw(ctx, "random_state", 0),
            n_jobs=ctx.n_jobs,
            rank_backend=ctx.rank_backend,
        ),
        cat_features,
        effective_weight,
        target_cv_metadata,
        y_sel,
        X_pre,
    )


def _mrmr_classic_path(ctx: "FilterContext", prep: ClassicPrepared, k: int, top_m: int) -> np.ndarray:
    rel = _compute_relevance(prep.X_arr, prep.y_arr, prep.w, ctx.request.task, _kw(ctx, "relevance"))
    return mrmr_select(
        prep.X_arr,
        rel,
        k,
        formula=_kw(ctx, "formula"),
        top_m=top_m,
        sample_weight=prep.w,
        n_jobs=ctx.n_jobs,
        mrmr_backend=ctx.mrmr_backend,
        callback=ctx.request.callback,
        include_idx=_include_indices(ctx),
        candidate_idx=_candidate_indices(ctx),
        blocks=_block_member_arrays(ctx),
    )


def _resolve_cat_features(
    X: Union[pd.DataFrame, np.ndarray], cat_features: Optional[list[str]],
) -> Optional[list[str]]:
    if cat_features is None and isinstance(X, pd.DataFrame):
        return X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return cat_features

_SUPERVISED_CAT_ENCODINGS = frozenset({"target", "loo", "james_stein", "loo_logit"})


def _encode_categoricals_for_selector(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
    cat_features: Optional[list[str]], cat_encoding: CatEncoding, *,
    allow_full_data_target_encoding: bool,
    sample_weight=None,
    task: Task = "regression",
    groups=None,
    time=None,
    target_cv_n_splits: int = 5,
    target_cv_smoothing: str | float = "auto",
    target_prior: float | None = None,
    warmup_policy: str = "zero_weight",
    onehot_max_levels: int = 32,
) -> tuple[Union[pd.DataFrame, np.ndarray], np.ndarray | None, dict | None]:
    if not cat_features or cat_encoding == "none":
        return X, sample_weight, None
    if not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
    present_cat_features = [col for col in cat_features if col in X.columns]
    if not present_cat_features:
        return X, sample_weight, None
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
    if cat_encoding == "onehot":
        encoder = OneHotBlockEncoder(
            present_cat_features,
            max_levels=validate_onehot_max_levels(onehot_max_levels),
        )
        return encoder.fit_transform(X, sample_weight=sample_weight), sample_weight, None
    if cat_encoding == "target_cv":
        encoder = TargetCVEncoder(
            present_cat_features,
            target_type="binary" if task == "classification" else "continuous",
            smooth=target_cv_smoothing,
            cv=target_cv_n_splits,
            target_prior=target_prior,
            warmup_policy=warmup_policy,
        )
        encoded = encoder.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
        )
        return (
            encoded,
            encoder.effective_sample_weight_,
            target_cv_metadata_from_encoder(encoder.encoding_cv_),
        )
    return encode_categoricals(
        X,
        y,
        cat_features,
        cat_encoding,
        sample_weight=sample_weight,
        target_type="binary" if task == "classification" else "continuous",
    ), sample_weight, None


def _default_top_m(top_m: Optional[int], k: int) -> int:
    return max(max(5 * k, 250) if top_m is None else int(top_m), int(k))


def _prepare_xy_classic(ctx: "FilterContext") -> ClassicPrepared:
    cache = ctx.request.cache
    if cache is not None:
        if not is_classic_cache(cache):
            raise ValueError("cache is supported only with estimator='gaussian'")
        encoding = _kw(ctx, "cat_encoding", "none")
        if encoding not in (None, "none"):
            raise ValueError(
                f"cat_encoding={encoding!r} cannot be combined with a prebuilt "
                "classic cache because the cache has no encoding provenance"
            )
        y_arr = validate_target(
            ctx.request.y,
            ctx.request.task,
            int(cache.n_rows_original),
        )
        row_idx = np.asarray(cache.row_idx, dtype=np.int64)
        eval_sample_weight = (
            None
            if not cache.weights_supplied
            else np.asarray(cache.sample_weight, dtype=np.float64)
        )
        return ClassicPrepared(
            np.asarray(cache.X, dtype=np.float64),
            y_arr[row_idx],
            np.asarray(cache.sample_weight, dtype=np.float64),
            np.asarray(cache.mi_w, dtype=np.float64),
            list(cache.feature_names),
            row_idx,
            None,
            eval_sample_weight,
            None,
            None,
            None,
        )
    cat_features = _kw(ctx, "cat_features")
    if isinstance(ctx.request.X, pd.DataFrame) and cat_features is None:
        cat_features = ctx.request.X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
    X_encoded = _encode_categoricals_for_selector(
        ctx.request.X,
        ctx.request.y,
        cat_features,
        _kw(ctx, "cat_encoding"),
        allow_full_data_target_encoding=_kw(ctx, "allow_full_data_target_encoding"),
        sample_weight=ctx.request.sample_weight,
        task=ctx.request.task,
        groups=ctx.groups,
        time=ctx.time,
        target_cv_n_splits=_kw(ctx, "target_cv_n_splits", 5),
        target_cv_smoothing=_kw(ctx, "target_cv_smoothing", "auto"),
        target_prior=_kw(ctx, "target_prior"),
        warmup_policy=_kw(ctx, "warmup_policy", "zero_weight"),
        onehot_max_levels=_kw(ctx, "onehot_max_levels", 32),
    )
    X_encoded, effective_weight, target_cv_metadata = X_encoded
    X_arr, y_arr, feature_names = validate_inputs(
        X_encoded,
        ctx.request.y,
        ctx.request.task,
    )
    X_arr, y_arr, w, row_idx = subsample_xy(
        X_arr,
        y_arr,
        _kw(ctx, "subsample"),
        _kw(ctx, "random_state"),
        sample_weight=effective_weight,
        return_idx=True,
    )
    if effective_weight is None:
        mi_w = np.ones(row_idx.size, dtype=np.float64)
    else:
        raw_weight = ensure_weights(
            effective_weight,
            ctx.n_rows,
            normalize=False,
        )
        mi_w = raw_weight[row_idx]
    eval_sample_weight = (
        None
        if effective_weight is None
        else np.asarray(w, dtype=np.float64)
    )
    X_pre_within = None
    y_pre_within = None
    groups_sub = None
    if ctx.within is not None:
        X_pre_within = np.array(X_arr, dtype=np.float64, copy=True)
        y_pre_within = np.array(y_arr, dtype=np.float64, copy=True)
        groups_sub = ctx.groups[row_idx]
        time_sub = ctx.time[row_idx] if ctx.time is not None else None
        X_arr, y_arr, _fitted = fit_transform_within(
            ctx.within,
            X_arr,
            y_arr,
            groups_sub,
            time_sub,
            w,
        )
        positive = np.asarray(w, dtype=np.float64) > 0.0
        if np.any(positive) and not np.any(np.ptp(X_arr[positive], axis=0) > 0.0):
            raise ValueError(
                "within demeaning removed all feature variation; "
                "no within-entity signal remains"
            )
    return ClassicPrepared(
        X_arr,
        y_arr,
        w,
        mi_w,
        feature_names,
        row_idx,
        target_cv_metadata,
        eval_sample_weight,
        X_pre_within,
        y_pre_within,
        groups_sub,
    )


def _compute_relevance(
    X_arr: np.ndarray, y_arr: np.ndarray, w: np.ndarray, task: Task, relevance: RelevanceMethod,
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


def _gaussian_relevance_for_input(
    ctx: "FilterContext",
    cache: FeatureCache,
    *,
    y=None,
) -> np.ndarray:
    from sift.estimators.copula import weighted_correlation_matrix, weighted_rank_gauss_2d
    from sift.selection.cefsplus_multi import (
        as_regression_targets,
        multiple_correlation,
        shrunk_target_covariance,
    )

    y_arr, n_y = as_regression_targets(
        ctx.request.y if y is None else y, int(cache.n_rows_original)
    )
    y_cache = y_arr[np.asarray(cache.row_idx, dtype=np.int64)]
    weights = np.asarray(cache.sample_weight, dtype=np.float64)
    if n_y == 1:
        zy = weighted_rank_gauss_1d(np.asarray(y_cache).reshape(-1), weights)
        rel_valid = gaussian_mi_from_corr(
            weighted_corr_with_vector(cache.Z, zy, weights)
        )
    else:
        zy_mat = weighted_rank_gauss_2d(np.asarray(y_cache, dtype=np.float64), weights)
        Ryy = np.asarray(
            weighted_correlation_matrix(zy_mat, weights, backend="blas"),
            dtype=np.float64,
        )
        C_all = np.column_stack(
            [
                np.asarray(
                    weighted_corr_with_vector(cache.Z, zy_mat[:, j], weights),
                    dtype=np.float64,
                )
                for j in range(n_y)
            ]
        )
        rel_valid = gaussian_mi_from_corr(
            multiple_correlation(
                C_all, shrunk_target_covariance(Ryy), shrink=1e-6, eps=1e-12
            )
        )
    relevance = np.zeros(ctx.n_features_input, dtype=np.float64)
    valid_cols = np.asarray(cache.valid_cols, dtype=np.int64)
    rel_valid_arr = np.asarray(rel_valid, dtype=np.float64)
    cache_names = list(cache.feature_names or [])
    input_names = list(ctx.feature_names)
    if (
        len(cache_names) > 0
        and len(set(cache_names)) == len(cache_names)
        and len(set(input_names)) == len(input_names)
        and set(cache_names) == set(input_names)
    ):
        input_index = {name: idx for idx, name in enumerate(input_names)}
        for valid_pos, cache_original in enumerate(valid_cols):
            if 0 <= int(cache_original) < len(cache_names):
                relevance[input_index[cache_names[int(cache_original)]]] = rel_valid_arr[
                    valid_pos
                ]
    else:
        in_bounds = (valid_cols >= 0) & (valid_cols < relevance.size)
        relevance[valid_cols[in_bounds]] = rel_valid_arr[in_bounds]
    return relevance


def _within_relevance(ctx: "FilterContext", relevance: np.ndarray) -> np.ndarray | None:
    if ctx.within is None:
        return None
    return np.asarray(relevance, dtype=np.float64)


def _between_relevance_classic(ctx: "FilterContext", prep: ClassicPrepared) -> np.ndarray | None:
    if ctx.within is None or prep.X_pre_within is None or prep.y_pre_within is None:
        return None
    X_g, y_g, w_g = group_level_design(
        prep.X_pre_within,
        prep.y_pre_within,
        prep.groups_sub if prep.groups_sub is not None else ctx.groups[prep.row_idx],
        prep.w,
    )
    return _compute_relevance(
        X_g,
        y_g,
        w_g,
        ctx.request.task,
        _kw(ctx, "relevance"),
    )


def _between_relevance_gaussian(
    ctx: "FilterContext",
    cache: FeatureCache,
    X_pre,
) -> np.ndarray | None:
    if ctx.within is None:
        return None
    X_arr, _template = as_float_feature_matrix(X_pre)
    y_arr = to_numpy(ctx.request.y, dtype=np.float64).ravel()
    row_idx = np.asarray(cache.row_idx, dtype=np.int64)
    weights = np.asarray(cache.sample_weight, dtype=np.float64)
    X_g, y_g, w_g = group_level_design(
        X_arr[row_idx],
        y_arr[row_idx],
        ctx.groups[row_idx],
        weights,
    )
    zy = weighted_rank_gauss_1d(y_g, w_g)
    Z_g = weighted_rank_gauss_2d(X_g, w_g, n_jobs=1, rank_backend="serial")
    rel_valid = gaussian_mi_from_corr(weighted_corr_with_vector(Z_g, zy, w_g))
    # X_g still has the original feature width; map through cache.valid_cols
    # only when the pre-within matrix is the full input width.
    relevance = np.zeros(ctx.n_features_input, dtype=np.float64)
    rel_arr = np.asarray(rel_valid, dtype=np.float64)
    if rel_arr.shape[0] == ctx.n_features_input:
        relevance = rel_arr
        return relevance
    valid_cols = np.asarray(cache.valid_cols, dtype=np.int64)
    in_bounds = (valid_cols >= 0) & (valid_cols < relevance.size)
    if rel_arr.shape[0] == valid_cols.shape[0]:
        relevance[valid_cols[in_bounds]] = rel_arr[in_bounds]
        return relevance
    n = min(rel_arr.shape[0], relevance.size)
    relevance[:n] = rel_arr[:n]
    return relevance


def _path_ranking(
    feature_names: list[str],
    selected_indices,
    relevance: np.ndarray,
    selector: str,
    *,
    within_relevance: np.ndarray | None = None,
    between_relevance: np.ndarray | None = None,
    block_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    relevance_arr = np.asarray(relevance, dtype=np.float64).reshape(-1)
    n_features = len(feature_names)
    if relevance_arr.shape[0] != n_features:
        raise RuntimeError("relevance length must match feature names")
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if selected.size and (
        np.any(selected < 0)
        or np.any(selected >= n_features)
        or np.unique(selected).size != selected.size
    ):
        raise RuntimeError("selected indices must be unique and in bounds")
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected] = True
    remaining = np.flatnonzero(~selected_mask)
    finite_relevance = np.where(np.isfinite(relevance_arr), relevance_arr, -np.inf)
    remaining = remaining[
        np.lexsort((remaining, -finite_relevance[remaining]))
    ]
    order = np.concatenate([selected, remaining])
    data = {
        "feature": [feature_names[int(i)] for i in order],
        "rank": np.arange(1, n_features + 1, dtype=np.int64),
        "selected": selected_mask[order],
        "selected_index": order,
        "relevance": relevance_arr[order],
    }
    if within_relevance is not None:
        within_arr = np.asarray(within_relevance, dtype=np.float64).reshape(-1)
        if within_arr.shape[0] != n_features:
            raise RuntimeError("within_relevance length must match feature names")
        data["within_relevance"] = within_arr[order]
    if between_relevance is not None:
        between_arr = np.asarray(between_relevance, dtype=np.float64).reshape(-1)
        if between_arr.shape[0] != n_features:
            raise RuntimeError("between_relevance length must match feature names")
        data["between_relevance"] = between_arr[order]
    data["selector"] = selector
    if block_ids is not None:
        if len(list(block_ids)) != n_features:
            raise RuntimeError("block_ids length must match feature names")
        data["block_id"] = [block_ids[int(i)] for i in order]
    return pd.DataFrame(data)


def _kw(ctx: "FilterContext", name: str, default=None):
    return ctx.selector_kwargs.get(name, default)


def _block_member_arrays(ctx: "FilterContext") -> list[np.ndarray] | None:
    blocks = getattr(ctx, "feature_blocks", None)
    if blocks is None:
        return None
    return [np.asarray(members, dtype=np.int64) for members in blocks.members]


def _block_id_column(ctx: "FilterContext", n_features: int) -> list[Any] | None:
    blocks = getattr(ctx, "feature_blocks", None)
    if blocks is None:
        return None
    return [blocks.block_ids[blocks.column_to_block[i]] for i in range(n_features)]


def _raw_onehot_include_names(ctx: "FilterContext") -> list:
    cond = getattr(ctx, "raw_conditioning", None)
    names = getattr(ctx, "raw_feature_names", None)
    if cond is None or names is None or not getattr(cond, "include", None):
        return []
    return [names[int(i)] for i in cond.include]


def _include_names(ctx: "FilterContext") -> list[str] | None:
    include_idx = _include_indices(ctx)
    if not include_idx.size:
        return None
    return [ctx.feature_names[int(i)] for i in include_idx]


def _include_indices(ctx: "FilterContext") -> np.ndarray:
    cond = getattr(ctx, "conditioning", None)
    if cond is None or not cond.include:
        return np.empty(0, dtype=np.int64)
    return np.asarray(cond.include, dtype=np.int64)


def _candidate_indices(ctx: "FilterContext") -> np.ndarray | None:
    cond = getattr(ctx, "conditioning", None)
    if cond is None:
        return None
    if cond.candidates is None and not cond.exclude:
        return None
    return np.asarray(cond.discovery, dtype=np.int64)


def _compose_classic_selection(ctx, prep, selected_idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    include_idx = _include_indices(ctx)
    if not include_idx.size:
        selected = [prep.feature_names[i] for i in selected_idx]
        return np.asarray(selected_idx, dtype=np.int64), selected
    names, indices = compose_selected(prep.feature_names, include_idx, selected_idx)
    return np.asarray(indices, dtype=np.int64), names


def _compose_gaussian_auto_selection(ctx, selected, selected_indices):
    include_idx = _include_indices(ctx)
    if not include_idx.size:
        return selected, selected_indices
    names, indices = compose_selected(
        ctx.feature_names,
        include_idx,
        selected_indices if selected_indices is not None else [],
    )
    return names, indices


def _conditioning_diag(ctx, selected_idx) -> dict | None:
    cond = getattr(ctx, "conditioning", None)
    if cond is None or not getattr(cond, "active", False):
        return None
    discovered = np.asarray(selected_idx, dtype=np.int64)
    if cond.include:
        include_set = set(int(i) for i in cond.include)
        discovered = np.array([int(i) for i in discovered if int(i) not in include_set], dtype=np.int64)
    return conditioning_record(
        cond,
        feature_names=ctx.feature_names,
        discovered_idx=discovered.tolist(),
    )

GAUSSIAN_EVALUATE = select_gaussian_evaluate_path
GAUSSIAN_AUTO = select_gaussian_auto_path
GAUSSIAN_ELBOW = select_gaussian_elbow_path
GAUSSIAN_PENALIZED = select_gaussian_penalized_path
GAUSSIAN_POSTERIOR = select_gaussian_posterior_path
GAUSSIAN_CHI2 = select_gaussian_chi2_path
GAUSSIAN_FORWARD_STOP = select_gaussian_forward_stop_path
GAUSSIAN_CHANGEPOINT = select_gaussian_changepoint_path
GAUSSIAN_PERM_GAP = select_gaussian_perm_gap_path
GAUSSIAN_XFIT_OBJECTIVE = select_gaussian_xfit_objective_path
GAUSSIAN_CV = select_gaussian_cv_path
GAUSSIAN_KNOCKOFF = select_gaussian_knockoff_path
GAUSSIAN_STABILITY = select_gaussian_stability_path
GAUSSIAN_CONSENSUS = select_gaussian_consensus_path
