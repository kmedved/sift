"""Public binary CEFS+ selector orchestration."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import subsample_xy, validate_inputs
from sift.selection.api_helpers import (
    auto_k_summary as _auto_k_summary,
    build_selector_metadata as _build_selector_metadata,
)
from sift.selection.auto_k import (
    AutoKConfig,
    resolve_auto_k_config,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
)
from sift.selection.cefsplus_binary import (
    make_diagnostics,
    select_binary_logistic_path,
)
from sift.selection.cefsplus_binary_common import (
    BinaryOptions,
    BinaryPathRun,
    BinaryProblem,
    BinarySelection,
    binary_refit_loglik_gains,
    check_binary_effective_weights,
    encode_categoricals_for_binary_selector,
    prepare_binary_problem,
    resolve_cat_features,
    validate_binary_options,
)
from sift.selection.result import FilterSelectionResult


def select_cefsplus_binary(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    loss: str = "logloss",
    top_m: Optional[int] = None,
    corr_prune: float | None = 0.95,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    class_weight=None,
    ridge: float = 1e-4,
    refit_every: int = 1,
    cat_features: Optional[List[str]] = None,
    cat_encoding: str = "loo_logit",
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = None,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> List[str] | FilterSelectionResult:
    """Binary CEFS+ using a greedy conditional Bernoulli deviance proxy."""
    options = validate_binary_options(
        k,
        loss=loss,
        top_m=top_m,
        corr_prune=corr_prune,
        subsample=subsample,
        ridge=ridge,
        refit_every=refit_every,
        cat_encoding=cat_encoding,
        loo_smoothing=loo_smoothing,
        loo_clip_min=loo_clip_min,
        loo_clip_max=loo_clip_max,
        sample_weight=sample_weight,
        class_weight=class_weight,
    )
    problem = prepare_binary_problem(
        X,
        y,
        groups=groups,
        time=time,
        sample_weight=sample_weight,
        class_weight=class_weight,
    )
    auto_k = options.k_value == "auto"
    if auto_k:
        auto_k_config = resolve_auto_k_config(auto_k_config, problem.time, problem.groups)

    if options.loss == "brier":
        return _select_binary_brier(
            X,
            problem,
            options,
            auto_k_config=auto_k_config,
            cat_features=cat_features,
            cat_encoding=cat_encoding,
            allow_full_data_target_encoding=allow_full_data_target_encoding,
            random_state=random_state,
            verbose=verbose,
            return_result=return_result,
            class_weight=class_weight,
        )

    run = _build_binary_logloss_path(
        X,
        problem,
        options,
        auto_k_config=auto_k_config,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        random_state=random_state,
        verbose=verbose,
    )
    selection = _select_binary_logloss_features(
        X,
        problem,
        run,
        options,
        auto_k_config=auto_k_config,
        cat_encoding=cat_encoding,
        verbose=verbose,
    )

    if not return_result:
        return selection.selected_features

    return _build_binary_result(
        problem,
        run,
        selection,
        options,
        auto_k_config=auto_k_config,
        cat_encoding=cat_encoding,
        class_weight=class_weight,
        random_state=random_state,
    )


def _select_binary_brier(
    X,
    problem: BinaryProblem,
    options: BinaryOptions,
    *,
    auto_k_config: AutoKConfig | None,
    cat_features: Optional[List[str]],
    cat_encoding: str,
    allow_full_data_target_encoding: bool,
    random_state: int,
    verbose: bool,
    return_result: bool,
    class_weight,
) -> List[str] | FilterSelectionResult:
    from sift.api import select_cefsplus

    cat_encoding_eff = "loo" if cat_encoding == "loo_logit" else cat_encoding
    result = select_cefsplus(
        X,
        problem.y01.astype(float),
        k=options.k_value,
        groups=problem.groups,
        time=problem.time,
        auto_k_config=auto_k_config,
        sample_weight=problem.weights if problem.weighted else None,
        top_m=options.top_m,
        corr_prune=options.corr_prune,
        cat_features=cat_features,
        cat_encoding=cat_encoding_eff,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        subsample=options.subsample,
        random_state=random_state,
        verbose=verbose,
        return_result=return_result,
    )
    if not return_result:
        return result
    assert isinstance(result, FilterSelectionResult)
    metadata = dict(result.selector_metadata)
    metadata.update(
        {
            "selector": "cefsplus_binary",
            "loss": "brier",
            "delegate_selector": "cefsplus",
            "weighted": problem.weighted,
            "class_weight": class_weight,
            "class_weight_scope": "pre_subsample" if class_weight is not None else None,
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


def _build_binary_logloss_path(
    X,
    problem: BinaryProblem,
    options: BinaryOptions,
    *,
    auto_k_config: AutoKConfig | None,
    cat_features: Optional[List[str]],
    cat_encoding: str,
    allow_full_data_target_encoding: bool,
    random_state: int,
    verbose: bool,
) -> BinaryPathRun:
    path_k = int(auto_k_config.max_k) if options.k_value == "auto" else int(options.k_value)
    cat_features = resolve_cat_features(X, cat_features)
    X_encoded = encode_categoricals_for_binary_selector(
        X,
        problem.y01,
        cat_features,
        cat_encoding,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        loo_smoothing=options.loo_smoothing,
        loo_clip_min=options.loo_clip_min,
        loo_clip_max=options.loo_clip_max,
        sample_weight=problem.weights,
    )
    X_arr, _, feature_names = validate_inputs(X_encoded, problem.y01, "regression")
    X_sub, y_sub, w_sub, row_idx = subsample_xy(
        X_arr,
        problem.y01,
        options.subsample,
        random_state,
        sample_weight=problem.weights,
        return_idx=True,
    )
    check_binary_effective_weights(y_sub, w_sub)

    top_m_eff = None if options.top_m is None else max(options.top_m, path_k)
    if verbose:
        _print_binary_path_message(problem, options, auto_k_config, path_k, top_m_eff)

    path = select_binary_logistic_path(
        X_sub.astype(np.float64, copy=False),
        y_sub.astype(np.float64, copy=False),
        w_sub.astype(np.float64, copy=False),
        feature_names,
        k=path_k,
        top_m=top_m_eff,
        corr_prune=options.corr_prune,
        ridge=options.ridge,
        refit_every=options.refit_every,
    )
    return BinaryPathRun(
        path=path,
        feature_names=feature_names,
        X_sub=X_sub,
        y_sub=y_sub,
        w_sub=w_sub,
        row_idx=row_idx,
        top_m_eff=top_m_eff,
        cat_features=cat_features,
    )


def _print_binary_path_message(
    problem: BinaryProblem,
    options: BinaryOptions,
    auto_k_config: AutoKConfig | None,
    path_k: int,
    top_m_eff: int | None,
) -> None:
    weighted_label = "weighted " if problem.weighted else ""
    if options.k_value != "auto":
        print(
            f"CEFS+ binary {weighted_label}logloss: selecting {path_k} features "
            f"(top_m={top_m_eff}, corr_prune={options.corr_prune})"
        )
        return
    assert auto_k_config is not None
    if auto_k_config.k_method == "elbow":
        mode = "elbow"
    elif auto_k_config.k_method == "penalized_objective":
        mode = (
            f"penalized_objective/{auto_k_config.objective_penalty}/"
            f"{auto_k_config.binary_objective_mode}"
        )
    else:
        mode = f"evaluate/{auto_k_config.strategy}/{auto_k_config.selection_rule}"
    print(
        f"CEFS+ binary {weighted_label}logloss auto-k ({mode}): "
        f"building path to {path_k} features "
        f"(top_m={top_m_eff}, corr_prune={options.corr_prune})"
    )


def _select_binary_logloss_features(
    X,
    problem: BinaryProblem,
    run: BinaryPathRun,
    options: BinaryOptions,
    *,
    auto_k_config: AutoKConfig | None,
    cat_encoding: str,
    verbose: bool,
) -> BinarySelection:
    if options.k_value != "auto":
        return _binary_selection_prefix(run.path, len(run.path.selected_features))
    assert auto_k_config is not None

    if auto_k_config.k_method == "elbow":
        return _select_binary_elbow(run, auto_k_config, verbose)
    if auto_k_config.k_method == "penalized_objective":
        return _select_binary_penalized(run, problem, options, auto_k_config, verbose)
    return _select_binary_evaluate(
        X,
        problem,
        run,
        options,
        auto_k_config,
        cat_encoding=cat_encoding,
        verbose=verbose,
    )


def _select_binary_elbow(
    run: BinaryPathRun,
    auto_k_config: AutoKConfig,
    verbose: bool,
) -> BinarySelection:
    auto_objective = np.cumsum(np.asarray(run.path.path_scores, dtype=np.float64))
    best_k, auto_diag = select_k_elbow(
        auto_objective,
        min_k=auto_k_config.min_k,
        max_k=len(run.path.selected_features),
        min_rel_gain=auto_k_config.elbow_min_rel_gain,
        patience=auto_k_config.elbow_patience,
    )
    selected_count = min(best_k, len(run.path.selected_features))
    if verbose:
        print(f"  Elbow selected k={selected_count}")
    auto_summary = _auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(run.path.selected_features),
        effective_max_k=min(int(auto_k_config.max_k), len(run.path.selected_features)),
        diagnostics=auto_diag,
        extra={
            "proxy_only_objective": True,
            "objective_scale": "binary_score_test_gain",
            "score_test_objective_approximation": True,
        },
    )
    return _binary_selection_prefix(
        run.path,
        selected_count,
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=auto_summary,
    )


def _select_binary_penalized(
    run: BinaryPathRun,
    problem: BinaryProblem,
    options: BinaryOptions,
    auto_k_config: AutoKConfig,
    verbose: bool,
) -> BinarySelection:
    if auto_k_config.binary_objective_mode == "score_test":
        auto_objective = np.cumsum(np.asarray(run.path.path_scores, dtype=np.float64))
        binary_refit_failures = 0
        score_test_ic_approximation = True
    else:
        auto_objective, binary_refit_failures = binary_refit_loglik_gains(
            run.X_sub.astype(np.float64, copy=False),
            run.y_sub.astype(np.float64, copy=False),
            run.w_sub.astype(np.float64, copy=False),
            run.path.selected_original,
            ridge=options.ridge,
        )
        score_test_ic_approximation = False

    best_k, auto_diag = select_k_penalized_objective(
        auto_objective,
        auto_k_config,
        objective_scale=2.0,
        n_samples=len(run.y_sub),
        sample_weight=run.w_sub,
        min_k=auto_k_config.min_k,
        max_k=len(run.path.selected_features),
    )
    selected_count = min(best_k, len(run.path.selected_features))
    ic_likelihood_type = (
        "weighted_pseudo_likelihood" if problem.weighted else "bernoulli_log_likelihood"
    )
    if auto_diag is not None and not auto_diag.empty:
        auto_diag["binary_objective_mode"] = auto_k_config.binary_objective_mode
        auto_diag["binary_objective_fit"] = (
            "score_test_approximation"
            if score_test_ic_approximation
            else "ridge_fit_unpenalized_loglik_score"
        )
        auto_diag["score_test_ic_approximation"] = score_test_ic_approximation
        auto_diag["ic_likelihood_type"] = ic_likelihood_type
        auto_diag["binary_refit_failures"] = binary_refit_failures
        auto_diag["refit_every_warning"] = bool(
            score_test_ic_approximation and options.refit_every > 1
        )
    if verbose:
        print(f"  Penalized objective selected k={selected_count}")

    auto_summary = _auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(run.path.selected_features),
        effective_max_k=min(int(auto_k_config.max_k), len(run.path.selected_features)),
        diagnostics=auto_diag,
        extra={
            "proxy_only_objective": True,
            "objective_penalty": auto_k_config.objective_penalty,
            "objective_scale": "binary_loglik_gain",
            "binary_objective_mode": auto_k_config.binary_objective_mode,
            "binary_objective_fit": (
                "score_test_approximation"
                if score_test_ic_approximation
                else "ridge_fit_unpenalized_loglik_score"
            ),
            "score_test_ic_approximation": score_test_ic_approximation,
            "ic_likelihood_type": ic_likelihood_type,
            "binary_refit_failures": binary_refit_failures,
            "warnings": [
                "refit_every > 1 makes cumulative score-test gains more approximate"
            ]
            if score_test_ic_approximation and options.refit_every > 1
            else [],
        },
    )
    return _binary_selection_prefix(
        run.path,
        selected_count,
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=auto_summary,
    )


def _select_binary_evaluate(
    X,
    problem: BinaryProblem,
    run: BinaryPathRun,
    options: BinaryOptions,
    auto_k_config: AutoKConfig,
    *,
    cat_encoding: str,
    verbose: bool,
) -> BinarySelection:
    eval_X = (
        X
        if isinstance(X, pd.DataFrame)
        else pd.DataFrame(np.asarray(X), columns=run.feature_names)
    )
    if len(run.row_idx) < problem.n_rows:
        eval_X = eval_X.iloc[run.row_idx]
        eval_y = problem.y01[run.row_idx]
        eval_groups = problem.groups[run.row_idx] if problem.groups is not None else None
        eval_time = problem.time[run.row_idx] if problem.time is not None else None
    else:
        eval_y = problem.y01
        eval_groups = problem.groups
        eval_time = problem.time

    best_k, selected_features, auto_diag = select_k_auto(
        eval_X,
        eval_y,
        run.path.selected_features,
        auto_k_config,
        groups=eval_groups,
        time=eval_time,
        task="classification",
        cat_features=run.cat_features,
        cat_encoding=cat_encoding,
        sample_weight=run.w_sub,
        loo_smoothing=options.loo_smoothing,
        loo_clip_min=options.loo_clip_min,
        loo_clip_max=options.loo_clip_max,
    )
    selected_count = len(selected_features)
    if verbose:
        print(f"  CV/holdout selected k={best_k}")
    auto_summary = _auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(run.path.selected_features),
        effective_max_k=min(int(auto_k_config.max_k), len(run.path.selected_features)),
        diagnostics=auto_diag,
        extra={"proxy_only_objective": False},
    )
    return _binary_selection_prefix(
        run.path,
        selected_count,
        selected_features=selected_features,
        auto_diag=auto_diag,
        auto_summary=auto_summary,
    )


def _binary_selection_prefix(
    path,
    selected_count: int,
    *,
    selected_features: list[str] | None = None,
    auto_diag: pd.DataFrame | None = None,
    auto_objective: np.ndarray | None = None,
    auto_summary: dict | None = None,
) -> BinarySelection:
    if selected_features is None:
        selected_features = path.selected_features[:selected_count]
    return BinarySelection(
        selected_features=selected_features,
        selected_original=path.selected_original[:selected_count],
        selected_scores=path.path_scores[:selected_count],
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=auto_summary,
    )


def _build_binary_result(
    problem: BinaryProblem,
    run: BinaryPathRun,
    selection: BinarySelection,
    options: BinaryOptions,
    *,
    auto_k_config: AutoKConfig | None,
    cat_encoding: str,
    class_weight,
    random_state: int,
) -> FilterSelectionResult:
    auto_k = options.k_value == "auto"
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
    if auto_k:
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
    extra_metadata = {
        "loss": "logloss",
        "weighted": problem.weighted,
        "class_weight": class_weight,
        "class_weight_scope": "pre_subsample" if class_weight is not None else None,
        "target_mapping": problem.target_mapping,
        "ridge": options.ridge,
        "refit_every": options.refit_every,
        "corr_prune": options.corr_prune,
        "subsample": options.subsample,
        "random_state": random_state,
        "cat_encoding": cat_encoding,
        "loo_smoothing": options.loo_smoothing,
        "loo_clip_min": options.loo_clip_min,
        "loo_clip_max": options.loo_clip_max,
    }
    if auto_k:
        assert auto_k_config is not None
        extra_metadata.update(
            {
                "auto_k_mode": auto_k_config.auto_k_mode,
                "k_method": auto_k_config.k_method,
                "auto_k_strategy": auto_k_config.strategy,
                "selection_rule": auto_k_config.selection_rule,
                "objective_penalty": auto_k_config.objective_penalty
                if auto_k_config.k_method == "penalized_objective"
                else None,
                "binary_objective_mode": auto_k_config.binary_objective_mode
                if auto_k_config.k_method == "penalized_objective"
                else None,
            }
        )

    metadata = _build_selector_metadata(
        "cefsplus_binary",
        k=len(selection.selected_features),
        k_requested="auto" if auto_k else int(options.k_value),
        top_m=run.top_m_eff,
        n_features=problem.n_features_input,
        auto_k=auto_k,
        extra=extra_metadata,
    )
    return FilterSelectionResult(
        selected_features=selection.selected_features,
        selected_indices=selection.selected_original,
        selector_metadata=metadata,
        ranking_=ranking,
        diagnostics_=diagnostics,
    )
