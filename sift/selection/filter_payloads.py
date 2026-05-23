"""Payload builders for function-style filter selector APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import (
    CatEncoding,
    RelevanceMethod,
    Task,
    check_regression_only,
    encode_categoricals,
    subsample_xy,
    to_numpy,
    validate_inputs,
)
from sift.estimators import relevance as rel_est
from sift.estimators.copula import FeatureCache, build_cache
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
    auto_k_mode_label,
    prepare_filter_eval_data,
    select_binary_elbow,
    select_binary_evaluate,
    select_binary_penalized,
    select_filter_classic_auto_k,
    select_gaussian_elbow_path,
    select_gaussian_evaluate_path,
    select_gaussian_penalized_path,
)
from sift.selection.loops import jmi_select, mrmr_select

if TYPE_CHECKING:
    from sift.selection.filter_api import FilterContext


@dataclass(frozen=True)
class ClassicPrepared:
    X_arr: np.ndarray
    y_arr: np.ndarray
    w: np.ndarray
    feature_names: list[str]
    row_idx: np.ndarray


@dataclass(frozen=True)
class SelectionPayload:
    selected_features: list[str] | None = None
    selected_indices: list[int] | None = None
    top_m: int | None = None
    n_features: int | None = None
    metadata_extra: dict | None = None
    ranking: pd.DataFrame | None = None
    diagnostics: dict | None = None


ClassicPath = Callable[["FilterContext", ClassicPrepared, int, int], np.ndarray]
GaussianMethod = Callable[["FilterContext"], str]
GaussianRunner = Callable[..., tuple[list[str], list[int], pd.DataFrame, dict]]


def make_fixed_classic(path_func: ClassicPath) -> Callable[["FilterContext"], SelectionPayload]:
    def fixed_classic(ctx: "FilterContext") -> SelectionPayload:
        prep = _prepare_xy_classic(ctx)
        k = int(ctx.k)
        top_m = _default_top_m(_kw(ctx, "top_m"), k)
        if _kw(ctx, "verbose"):
            print(
                f"{ctx.spec.display_name} classic: selecting {k} features from "
                f"{prep.X_arr.shape[1]} (top_m={top_m})"
            )
        selected_idx = path_func(ctx, prep, k, top_m)
        selected = [prep.feature_names[i] for i in selected_idx]
        return SelectionPayload(
            selected_features=selected,
            selected_indices=selected_idx.astype(int).tolist()
            if ctx.request.return_result
            else None,
            top_m=top_m,
            n_features=ctx.n_features_input,
        )

    return fixed_classic


def make_auto_classic(path_func: ClassicPath) -> Callable[["FilterContext"], SelectionPayload]:
    def auto_classic(ctx: "FilterContext") -> SelectionPayload:
        assert ctx.auto_k_config is not None
        prep = _prepare_xy_classic(ctx)
        max_k = int(ctx.auto_k_config.max_k)
        top_m = _default_top_m(_kw(ctx, "top_m"), max_k)
        if _kw(ctx, "verbose"):
            print(
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
        selected, selected_indices = select_filter_classic_auto_k(
            y_arr=prep.y_arr,
            eval_X=X_eval,
            feature_names=prep.feature_names,
            path_idx=path_idx,
            auto_k_config=ctx.auto_k_config,
            eval_groups=ctx.groups[prep.row_idx] if ctx.groups is not None else None,
            eval_time=ctx.time[prep.row_idx] if ctx.time is not None else None,
            sample_weight=prep.w,
            task=ctx.request.task,
            cat_features=_kw(ctx, "cat_features"),
            cat_encoding=_kw(ctx, "cat_encoding"),
            verbose=_kw(ctx, "verbose"),
            return_indices=True,
        )
        return SelectionPayload(
            selected_features=selected,
            selected_indices=selected_indices,
            top_m=top_m,
            n_features=len(prep.feature_names),
        )

    return auto_classic


def make_fixed_gaussian(method_func: GaussianMethod) -> Callable[["FilterContext"], SelectionPayload]:
    def fixed_gaussian(ctx: "FilterContext") -> SelectionPayload:
        cache, _ = _cache_for_gaussian(ctx)
        method = method_func(ctx)
        k = int(ctx.k)
        top_m = _default_top_m(_kw(ctx, "top_m"), k)
        if _kw(ctx, "verbose"):
            print(f"{ctx.spec.display_name}: selecting {k} features (top_m={top_m})")
        if ctx.request.return_result:
            selected, selected_indices = select_cached(
                cache,
                ctx.request.y,
                k,
                method=method,
                top_m=top_m,
                corr_prune=_kw(ctx, "corr_prune", "auto"),
                return_indices=True,
            )
        else:
            selected = select_cached(
                cache,
                ctx.request.y,
                k,
                method=method,
                top_m=top_m,
                corr_prune=_kw(ctx, "corr_prune", "auto"),
            )
            selected_indices = None
        feature_names = cache.feature_names if cache.feature_names is not None else ctx.feature_names
        return SelectionPayload(
            selected_features=list(selected),
            selected_indices=list(selected_indices) if selected_indices is not None else None,
            top_m=top_m,
            n_features=len(feature_names),
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
        cache, cat_features = _cache_for_gaussian(ctx)
        top_m = _default_top_m(_kw(ctx, "top_m"), int(ctx.auto_k_config.max_k))
        if _kw(ctx, "verbose"):
            print(
                f"{ctx.spec.display_name} auto-k ({auto_k_mode_label(ctx.auto_k_config)}): "
                f"building path to {ctx.auto_k_config.max_k} features (top_m={top_m})"
            )
        eval_X, eval_y, eval_groups, eval_time, eval_weight = prepare_filter_eval_data(
            ctx.request.X,
            ctx.request.y,
            cache,
            ctx.groups,
            ctx.time,
            ctx.request.sample_weight,
        )
        selected, selected_indices, auto_diag, auto_summary = runner(
            cache=cache,
            y=ctx.request.y,
            method=method_func(ctx),
            max_k=int(ctx.auto_k_config.max_k),
            top_m=top_m,
            auto_k_config=ctx.auto_k_config,
            eval_X=eval_X,
            eval_y=eval_y,
            groups=eval_groups,
            time=eval_time,
            sample_weight=eval_weight,
            cat_features=cat_features,
            cat_encoding=_kw(ctx, "cat_encoding"),
            corr_prune=_kw(ctx, "corr_prune", "auto"),
            verbose=_kw(ctx, "verbose"),
        )
        diagnostics = None
        if include_diagnostics and ctx.request.return_result:
            diagnostics = {"auto_k": auto_summary, "auto_k_diagnostics": auto_diag}
        metadata_extra = {}
        if include_objective_penalty:
            metadata_extra["objective_penalty"] = ctx.auto_k_config.objective_penalty
        return SelectionPayload(
            selected_features=list(selected),
            selected_indices=list(selected_indices),
            top_m=top_m,
            n_features=ctx.n_features_input,
            metadata_extra=metadata_extra,
            diagnostics=diagnostics,
        )

    return auto_gaussian


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
            sample_weight=prep.w if pass_sample_weight else None,
        )

    return jmi_classic_path


def mrmr_gaussian_method(ctx: "FilterContext") -> str:
    return "mrmr_quot" if _kw(ctx, "formula") == "quotient" else "mrmr_diff"


def selector_gaussian_method(selector: str) -> GaussianMethod:
    def gaussian_method(_ctx: "FilterContext") -> str:
        return selector

    return gaussian_method


def validate_standard(ctx: "FilterContext") -> None:
    check_regression_only(ctx.request.task, ctx.estimator)


def validate_ksg_no_weight(ctx: "FilterContext") -> None:
    validate_standard(ctx)
    if ctx.request.sample_weight is not None:
        raise ValueError("estimator='ksg' does not support sample_weight")


def validate_cefsplus(ctx: "FilterContext") -> None:
    y_arr = to_numpy(ctx.request.y, dtype=np.float32).ravel()
    if len(y_arr) != ctx.n_rows:
        raise ValueError(f"X has {ctx.n_rows} rows but y has {len(y_arr)}")
    if not np.isfinite(y_arr).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")


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


def _binary_auto_payload(
    ctx: "FilterContext",
    handler: Callable[..., BinarySelection],
) -> SelectionPayload:
    assert ctx.auto_k_config is not None
    problem, options, run = _build_binary_run(ctx)
    selection = handler(
        ctx.request.X,
        problem,
        run,
        options,
        auto_k_config=ctx.auto_k_config,
        cat_encoding=_kw(ctx, "cat_encoding"),
        verbose=_kw(ctx, "verbose"),
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
    if not ctx.request.return_result:
        return SelectionPayload(
            selected_features=selection.selected_features,
            top_m=run.top_m_eff,
            n_features=problem.n_features_input,
        )

    diagnostics = make_diagnostics(run.path)
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
        diagnostics["auto_k"] = selection.auto_summary
        diagnostics["auto_k_diagnostics"] = selection.auto_diag
        if selection.auto_objective is not None:
            diagnostics["auto_k_objective"] = selection.auto_objective.tolist()

    ranking = pd.DataFrame(
        {
            "feature": selection.selected_features,
            "rank": np.arange(1, len(selection.selected_features) + 1, dtype=np.int64),
            "selected": np.ones(len(selection.selected_features), dtype=bool),
            "selected_index": selection.selected_original,
            "score": selection.selected_scores,
            "selector": "cefsplus_binary",
        }
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
    if options.k_value == "auto" and ctx.auto_k_config is not None:
        metadata_extra.update(_binary_auto_metadata(ctx.auto_k_config))
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


def _cache_for_gaussian(ctx: "FilterContext") -> tuple[FeatureCache, list[str] | None]:
    cat_features = _resolve_cat_features(ctx.request.X, _kw(ctx, "cat_features"))
    if ctx.request.cache is not None:
        return ctx.request.cache, cat_features
    X_encoded = _encode_categoricals_for_selector(
        ctx.request.X,
        ctx.request.y,
        cat_features,
        _kw(ctx, "cat_encoding"),
        allow_full_data_target_encoding=_kw(ctx, "allow_full_data_target_encoding"),
    )
    return (
        build_cache(
            X_encoded,
            sample_weight=ctx.request.sample_weight,
            subsample=_kw(ctx, "subsample"),
            random_state=_kw(ctx, "random_state"),
            n_jobs=ctx.n_jobs,
            rank_backend=ctx.rank_backend,
        ),
        cat_features,
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
) -> Union[pd.DataFrame, np.ndarray]:
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


def _default_top_m(top_m: Optional[int], k: int) -> int:
    return max(max(5 * k, 250) if top_m is None else int(top_m), int(k))


def _prepare_xy_classic(ctx: "FilterContext") -> ClassicPrepared:
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
    )
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
        sample_weight=ctx.request.sample_weight,
        return_idx=True,
    )
    return ClassicPrepared(X_arr, y_arr, w, feature_names, row_idx)


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


def _kw(ctx: "FilterContext", name: str, default=None):
    return ctx.selector_kwargs.get(name, default)

GAUSSIAN_EVALUATE = select_gaussian_evaluate_path
GAUSSIAN_ELBOW = select_gaussian_elbow_path
GAUSSIAN_PENALIZED = select_gaussian_penalized_path
