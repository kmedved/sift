"""Spec-driven function-style filter selector APIs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import (
    CatEncoding,
    EstimatorJMI,
    EstimatorMRMR,
    Formula,
    RelevanceMethod,
    Task,
    resolve_jmi_estimator,
    validate_k,
)
from sift.estimators.copula import FeatureCache
from sift.selection.auto_k import AutoKConfig, resolve_auto_k_config
from sift.selection.cefsplus_binary_common import (
    prepare_binary_problem,
    validate_binary_options,
)
from sift.selection.filter_payloads import (
    GAUSSIAN_ELBOW,
    GAUSSIAN_EVALUATE,
    GAUSSIAN_PENALIZED,
    SelectionPayload,
    binary_auto_elbow_payload,
    binary_auto_evaluate_payload,
    binary_auto_penalized_payload,
    binary_fixed_payload,
    make_auto_classic,
    make_auto_gaussian,
    make_fixed_classic,
    make_fixed_gaussian,
    make_jmi_classic_path,
    make_mrmr_classic_path,
    mrmr_gaussian_method,
    no_extra,
    selector_gaussian_method,
    standard_extra,
    validate_cefsplus,
    validate_ksg_no_weight,
    validate_standard,
)
from sift.selection.loops import MrmrBackend, resolve_mrmr_backend
from sift.selection.result import FilterSelectionResult, build_selector_metadata

XInput = Union[pd.DataFrame, np.ndarray]
YInput = Union[pd.Series, np.ndarray]
KInput = Union[int, Literal["auto"]]


@dataclass(frozen=True)
class FilterRequest:
    X: XInput
    y: YInput
    k: KInput
    task: Task
    cache: Optional[FeatureCache] = None
    groups: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    auto_k_config: Optional[AutoKConfig] = None
    sample_weight: np.ndarray | None = None
    return_result: bool = False
    selector_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class FilterSpec:
    selector: str
    display_name: str
    estimator: str
    fixed_handler: Callable[["FilterContext"], SelectionPayload]
    auto_k_handlers: dict[str, Callable[["FilterContext"], SelectionPayload]]
    metadata_extra: Callable[["FilterContext"], dict[str, Any]]
    validate: Callable[["FilterContext"], None] = lambda _ctx: None


@dataclass(frozen=True)
class FilterContext:
    spec: FilterSpec
    request: FilterRequest
    selector_kwargs: dict[str, Any]
    k: int | Literal["auto"]
    groups: np.ndarray | None
    time: np.ndarray | None
    auto_k_config: AutoKConfig | None
    n_rows: int
    n_features_input: int
    feature_names: list[str]
    estimator: str
    n_jobs: int
    mrmr_backend: str
    rank_backend: str


_COMMON_REQUEST_LOCAL_NAMES = frozenset(
    {
        "X",
        "y",
        "k",
        "task",
        "cache",
        "groups",
        "time",
        "auto_k_config",
        "sample_weight",
        "return_result",
    }
)

MRMR_SELECTOR_KWARGS = (
    "relevance",
    "estimator",
    "formula",
    "top_m",
    "cat_features",
    "cat_encoding",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "n_jobs",
    "mrmr_backend",
    "verbose",
)
JMI_SELECTOR_KWARGS = (
    "estimator",
    "relevance",
    "top_m",
    "cat_features",
    "cat_encoding",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "verbose",
)
CEFSPLUS_SELECTOR_KWARGS = (
    "top_m",
    "corr_prune",
    "cat_features",
    "cat_encoding",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "verbose",
)
CEFSPLUS_BINARY_SELECTOR_KWARGS = (
    "loss",
    "top_m",
    "corr_prune",
    "class_weight",
    "ridge",
    "refit_every",
    "cat_features",
    "cat_encoding",
    "loo_smoothing",
    "loo_clip_min",
    "loo_clip_max",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "verbose",
)


def _request_from_public_locals(
    values: dict[str, Any],
    *,
    task: Task,
    selector_names: tuple[str, ...],
) -> FilterRequest:
    return FilterRequest(
        values["X"],
        values["y"],
        values["k"],
        task,
        cache=values.get("cache"),
        groups=values.get("groups"),
        time=values.get("time"),
        auto_k_config=values.get("auto_k_config"),
        sample_weight=values.get("sample_weight"),
        return_result=bool(values.get("return_result", False)),
        selector_kwargs={name: values[name] for name in selector_names},
    )


def select_mrmr(
    X: XInput, y: YInput, k: KInput, *, task: Task,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    relevance: RelevanceMethod = "f", estimator: EstimatorMRMR = "classic",
    formula: Formula = "quotient", top_m: Optional[int] = None,
    cat_features: Optional[list[str]] = None, cat_encoding: CatEncoding = "loo",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000, random_state: int = 0, n_jobs: int = 1,
    mrmr_backend: MrmrBackend = "auto",
    verbose: bool = True, return_result: bool = False,
) -> list[str] | FilterSelectionResult:
    """Minimum Redundancy Maximum Relevance feature selection."""
    request = _request_from_public_locals(
        locals(),
        task=task,
        selector_names=MRMR_SELECTOR_KWARGS,
    )
    return _select_filter(_mrmr_spec(request), request)


def select_jmi(
    X: XInput, y: YInput, k: KInput, *, task: Task,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    estimator: EstimatorJMI = "auto", relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None, cat_features: Optional[list[str]] = None,
    cat_encoding: CatEncoding = "loo",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000, random_state: int = 0,
    verbose: bool = True, return_result: bool = False,
) -> list[str] | FilterSelectionResult:
    """Joint Mutual Information feature selection."""
    request = _request_from_public_locals(
        locals(),
        task=task,
        selector_names=JMI_SELECTOR_KWARGS,
    )
    return _select_filter(_jmi_spec(request, JMI_CLASSIC_SPECS, JMI_GAUSSIAN_SPEC), request)


def select_jmim(
    X: XInput, y: YInput, k: KInput, *, task: Task,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    estimator: EstimatorJMI = "auto", relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None, cat_features: Optional[list[str]] = None,
    cat_encoding: CatEncoding = "loo",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000, random_state: int = 0,
    verbose: bool = True, return_result: bool = False,
) -> list[str] | FilterSelectionResult:
    """JMI Maximization, using the conservative minimum-pair aggregation."""
    request = _request_from_public_locals(
        locals(),
        task=task,
        selector_names=JMI_SELECTOR_KWARGS,
    )
    return _select_filter(
        _jmi_spec(request, JMIM_CLASSIC_SPECS, JMIM_GAUSSIAN_SPEC),
        request,
    )


def select_cefsplus(
    X: XInput, y: YInput, k: KInput = 75, *,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    top_m: Optional[int] = None, corr_prune: float | None = 0.95,
    cat_features: Optional[list[str]] = None, cat_encoding: CatEncoding = "loo",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = 50_000, random_state: int = 0,
    verbose: bool = True, return_result: bool = False,
) -> list[str] | FilterSelectionResult:
    """CEFS+ feature selection using log-det Gaussian MI proxy."""
    request = _request_from_public_locals(
        locals(),
        task="regression",
        selector_names=CEFSPLUS_SELECTOR_KWARGS,
    )
    return _select_filter(CEFSPLUS_SPEC, request)


def select_cefsplus_binary(
    X: XInput, y: YInput, k: KInput, *,
    loss: str = "logloss", top_m: Optional[int] = None, corr_prune: float | None = 0.95,
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    class_weight=None,
    ridge: float = 1e-4, refit_every: int = 1,
    cat_features: Optional[list[str]] = None, cat_encoding: str = "loo_logit",
    loo_smoothing: float = 20.0, loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = None, random_state: int = 0,
    verbose: bool = True, return_result: bool = False,
) -> list[str] | FilterSelectionResult:
    """Binary CEFS+ using a greedy conditional Bernoulli deviance proxy."""
    request = _request_from_public_locals(
        locals(),
        task="classification",
        selector_names=CEFSPLUS_BINARY_SELECTOR_KWARGS,
    )
    if str((request.selector_kwargs or {}).get("loss")).lower() == "brier":
        return _select_brier_delegate(request)
    return _select_filter(CEFSPLUS_BINARY_SPEC, request)


def _select_filter(
    spec: FilterSpec,
    request: FilterRequest,
) -> list[str] | FilterSelectionResult:
    ctx = _build_context(spec, request)
    if ctx.k == "auto":
        ctx = replace(
            ctx,
            auto_k_config=resolve_auto_k_config(request.auto_k_config, ctx.time, ctx.groups),
        )
        assert ctx.auto_k_config is not None
        handler = spec.auto_k_handlers.get(ctx.auto_k_config.k_method)
        if handler is None:
            raise ValueError(
                f"{spec.display_name} does not support k_method="
                f"{ctx.auto_k_config.k_method!r}"
            )
        _require_auto_k_eval_context(ctx)
    else:
        handler = spec.fixed_handler

    spec.validate(ctx)
    return _format_payload(ctx, handler(ctx))


def _build_context(spec: FilterSpec, request: FilterRequest) -> FilterContext:
    x_shape = request.X.shape if hasattr(request.X, "shape") else np.asarray(request.X).shape
    if len(x_shape) != 2:
        raise ValueError("X must be a 2D feature matrix")
    n_rows, n_features = int(x_shape[0]), int(x_shape[1])
    groups, time = _validate_groups_time(request.groups, request.time, n_rows)
    selector_kwargs = dict(request.selector_kwargs or {})
    n_jobs = int(selector_kwargs.get("n_jobs", 1))
    mrmr_backend = resolve_mrmr_backend(
        selector_kwargs.get("mrmr_backend", "auto"),
        n_jobs,
    )
    return FilterContext(
        spec=spec,
        request=request,
        selector_kwargs=selector_kwargs,
        k=validate_k(request.k),
        groups=groups,
        time=time,
        auto_k_config=request.auto_k_config,
        n_rows=n_rows,
        n_features_input=n_features,
        feature_names=list(request.X.columns)
        if isinstance(request.X, pd.DataFrame)
        else [f"x{i}" for i in range(n_features)],
        estimator=spec.estimator,
        n_jobs=n_jobs,
        mrmr_backend=mrmr_backend,
        rank_backend="processes" if mrmr_backend == "processes" else "serial",
    )


def _format_payload(
    ctx: FilterContext,
    payload: SelectionPayload,
) -> list[str] | FilterSelectionResult:
    assert payload.selected_features is not None
    if not ctx.request.return_result:
        return payload.selected_features

    extra = ctx.spec.metadata_extra(ctx)
    if ctx.k == "auto":
        assert ctx.auto_k_config is not None
        extra.update(
            {
                "auto_k_mode": ctx.auto_k_config.auto_k_mode,
                "k_method": ctx.auto_k_config.k_method,
                "auto_k_strategy": ctx.auto_k_config.strategy,
                "selection_rule": ctx.auto_k_config.selection_rule,
            }
        )
    if payload.metadata_extra:
        extra.update(payload.metadata_extra)

    metadata = build_selector_metadata(
        ctx.spec.selector,
        k=len(payload.selected_features),
        k_requested="auto" if ctx.k == "auto" else int(ctx.k),
        top_m=payload.top_m,
        n_features=payload.n_features or ctx.n_features_input,
        auto_k=ctx.k == "auto",
        extra=extra,
    )
    return FilterSelectionResult(
        selected_features=payload.selected_features,
        selected_indices=payload.selected_indices,
        selector_metadata=metadata,
        ranking_=payload.ranking,
        diagnostics_=payload.diagnostics,
    )


def _require_auto_k_eval_context(ctx: FilterContext) -> None:
    config = ctx.auto_k_config
    if ctx.k != "auto" or config is None:
        return
    _require_evaluate_context(config, ctx.groups, ctx.time)


def _require_evaluate_context(
    config: AutoKConfig,
    groups: np.ndarray | None,
    time: np.ndarray | None,
) -> None:
    if config.k_method != "evaluate":
        return
    if config.strategy == "time_holdout" and time is None:
        raise ValueError("auto-k evaluate with strategy='time_holdout' requires time parameter")
    if config.strategy == "group_cv" and groups is None:
        raise ValueError("auto-k evaluate with strategy='group_cv' requires groups parameter")


def _select_brier_delegate(request: FilterRequest) -> list[str] | FilterSelectionResult:
    x_shape = request.X.shape if hasattr(request.X, "shape") else np.asarray(request.X).shape
    if len(x_shape) != 2:
        raise ValueError("X must be a 2D feature matrix")
    groups, time = _validate_groups_time(request.groups, request.time, int(x_shape[0]))
    kw = (request.selector_kwargs or {}).get
    k_value = validate_k(request.k)
    if k_value == "auto":
        auto_k_config = resolve_auto_k_config(request.auto_k_config, time, groups)
        _require_evaluate_context(auto_k_config, groups, time)
    options = validate_binary_options(
        request.k,
        loss=kw("loss"),
        top_m=kw("top_m"),
        corr_prune=kw("corr_prune"),
        subsample=kw("subsample"),
        ridge=kw("ridge"),
        refit_every=kw("refit_every"),
        cat_encoding=kw("cat_encoding"),
        loo_smoothing=kw("loo_smoothing"),
        loo_clip_min=kw("loo_clip_min"),
        loo_clip_max=kw("loo_clip_max"),
        sample_weight=request.sample_weight,
        class_weight=kw("class_weight"),
    )
    problem = prepare_binary_problem(
        request.X,
        request.y,
        groups=groups,
        time=time,
        sample_weight=request.sample_weight,
        class_weight=kw("class_weight"),
    )
    cat_encoding_eff = "loo" if kw("cat_encoding") == "loo_logit" else kw("cat_encoding")
    result = select_cefsplus(
        request.X,
        problem.y01.astype(float),
        k=options.k_value,
        groups=problem.groups,
        time=problem.time,
        auto_k_config=request.auto_k_config,
        sample_weight=problem.weights if problem.weighted else None,
        top_m=options.top_m,
        corr_prune=options.corr_prune,
        cat_features=kw("cat_features"),
        cat_encoding=cat_encoding_eff,
        allow_full_data_target_encoding=kw("allow_full_data_target_encoding"),
        subsample=options.subsample,
        random_state=kw("random_state"),
        verbose=kw("verbose"),
        return_result=request.return_result,
    )
    if not request.return_result:
        return result

    assert isinstance(result, FilterSelectionResult)
    metadata = dict(result.selector_metadata)
    metadata.update(
        {
            "selector": "cefsplus_binary",
            "loss": "brier",
            "delegate_selector": "cefsplus",
            "weighted": problem.weighted,
            "class_weight": kw("class_weight"),
            "class_weight_scope": "pre_subsample"
            if kw("class_weight") is not None
            else None,
            "target_mapping": problem.target_mapping,
            "cat_encoding": cat_encoding_eff,
        }
    )
    return FilterSelectionResult(
        selected_features=result.selected_features,
        selected_indices=result.selected_indices,
        selector_metadata=metadata,
        ranking_=result.ranking_,
        diagnostics_=result.diagnostics_,
    )


def _validate_groups_time(
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    n_rows: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != n_rows:
            raise ValueError(f"groups has {len(groups)} elements but X has {n_rows} rows")
    if time is not None:
        time = np.asarray(time).reshape(-1)
        if len(time) != n_rows:
            raise ValueError(f"time has {len(time)} elements but X has {n_rows} rows")
    return groups, time


def _mrmr_spec(request: FilterRequest) -> FilterSpec:
    estimator = str((request.selector_kwargs or {}).get("estimator", "classic"))
    if estimator == "classic":
        return MRMR_CLASSIC_SPEC
    if estimator == "gaussian":
        return MRMR_GAUSSIAN_SPEC
    raise ValueError("estimator must be one of 'classic' or 'gaussian'")


def _jmi_spec(
    request: FilterRequest,
    classic_specs: dict[str, FilterSpec],
    gaussian_spec: FilterSpec,
) -> FilterSpec:
    estimator = resolve_jmi_estimator(
        str((request.selector_kwargs or {}).get("estimator", "auto")),
        request.task,
    )
    if estimator == "gaussian":
        return gaussian_spec
    if estimator in classic_specs:
        return classic_specs[estimator]
    raise ValueError("estimator must be one of 'auto', 'binned', 'r2', 'ksg', or 'gaussian'")


def _classic_spec(
    selector: str,
    display_name: str,
    estimator: str,
    aggregation: str | None,
    path_func,
    validate,
) -> FilterSpec:
    return FilterSpec(
        selector=selector,
        display_name=display_name,
        estimator=estimator,
        fixed_handler=make_fixed_classic(path_func),
        auto_k_handlers={"evaluate": make_auto_classic(path_func)},
        metadata_extra=standard_extra(aggregation),
        validate=validate,
    )


def _gaussian_spec(
    selector: str,
    display_name: str,
    method_func,
    *,
    cefsplus: bool = False,
) -> FilterSpec:
    auto_handlers = {
        "evaluate": make_auto_gaussian(
            method_func,
            GAUSSIAN_EVALUATE,
            include_diagnostics=cefsplus,
        ),
        "elbow": make_auto_gaussian(
            method_func,
            GAUSSIAN_ELBOW,
            include_diagnostics=cefsplus,
        ),
    }
    if cefsplus:
        auto_handlers["penalized_objective"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_PENALIZED,
            include_diagnostics=True,
            include_objective_penalty=True,
        )
    return FilterSpec(
        selector=selector,
        display_name=display_name,
        estimator="gaussian",
        fixed_handler=make_fixed_gaussian(method_func),
        auto_k_handlers=auto_handlers,
        metadata_extra=no_extra if cefsplus else standard_extra(),
        validate=validate_cefsplus if cefsplus else validate_standard,
    )


_MRMR_CLASSIC_PATH = make_mrmr_classic_path()
MRMR_CLASSIC_SPEC = _classic_spec(
    "mrmr",
    "mRMR",
    "classic",
    None,
    _MRMR_CLASSIC_PATH,
    validate_standard,
)
MRMR_GAUSSIAN_SPEC = _gaussian_spec(
    "mrmr",
    "mRMR",
    mrmr_gaussian_method,
)

JMI_CLASSIC_SPECS = {
    estimator: _classic_spec(
        "jmi",
        "JMI",
        estimator,
        "sum",
        make_jmi_classic_path(
            aggregation="sum",
            pass_sample_weight=estimator != "ksg",
        ),
        validate_ksg_no_weight if estimator == "ksg" else validate_standard,
    )
    for estimator in ("r2", "binned", "ksg")
}
JMI_GAUSSIAN_SPEC = _gaussian_spec(
    "jmi",
    "JMI",
    selector_gaussian_method("jmi"),
)

JMIM_CLASSIC_SPECS = {
    estimator: _classic_spec(
        "jmim",
        "JMIM",
        estimator,
        "min",
        make_jmi_classic_path(
            aggregation="min",
            pass_sample_weight=estimator != "ksg",
        ),
        validate_ksg_no_weight if estimator == "ksg" else validate_standard,
    )
    for estimator in ("r2", "binned", "ksg")
}
JMIM_GAUSSIAN_SPEC = _gaussian_spec(
    "jmim",
    "JMIM",
    selector_gaussian_method("jmim"),
)

CEFSPLUS_SPEC = _gaussian_spec(
    "cefsplus",
    "CEFS+",
    selector_gaussian_method("cefsplus"),
    cefsplus=True,
)

CEFSPLUS_BINARY_SPEC = FilterSpec(
    selector="cefsplus_binary",
    display_name="CEFS+ binary",
    estimator="binary",
    fixed_handler=binary_fixed_payload,
    auto_k_handlers={
        "evaluate": binary_auto_evaluate_payload,
        "elbow": binary_auto_elbow_payload,
        "penalized_objective": binary_auto_penalized_payload,
    },
    metadata_extra=no_extra,
)


__all__ = [
    "select_mrmr",
    "select_jmi",
    "select_jmim",
    "select_cefsplus",
    "select_cefsplus_binary",
]
