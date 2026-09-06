"""Binary auto-k route helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from sift.selection import auto_k as auto_k_module
from sift.selection.auto_k import AutoKConfig
from sift.selection.auto_k_stop import select_k_changepoint
from sift.selection.cefsplus_binary_common import (
    BinaryOptions,
    BinaryPathRun,
    BinaryProblem,
    BinarySelection,
    binary_refit_loglik_gains,
    binary_selection_prefix,
)
from sift.selection.filter_auto_k_cache import remap_onehot_prefix_evaluate
from sift.selection.filter_auto_k_common import (
    _effective_max_k,
    _objective_n_eff,
    _print_selected_k,
    _require_eval_split_context,
    _select_elbow_count,
    _select_penalized_count,
    _select_posterior_count,
    _zero_capable_effective_min_k,
    auto_k_summary,
)


def select_binary_elbow(
    _X, _problem: BinaryProblem, run: BinaryPathRun, _options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
) -> BinarySelection:
    del cat_encoding
    auto_objective = np.cumsum(np.asarray(run.path.path_scores, dtype=np.float64))
    path_length = (
        len(run.prefix_widths)
        if run.prefix_widths
        else len(run.path.selected_features)
    )
    selected_count, auto_diag = _select_elbow_count(
        auto_objective,
        auto_k_config,
        path_length,
    )
    _print_selected_k("Elbow", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=_effective_max_k(auto_k_config, path_length),
        diagnostics=auto_diag,
        extra={
            "proxy_only_objective": True,
            "objective_scale": "binary_score_test_gain",
            "score_test_objective_approximation": True,
        },
    )
    return binary_selection_prefix(
        run.path,
        selected_count,
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=summary,
    )


def select_binary_penalized(
    _X, problem: BinaryProblem, run: BinaryPathRun, options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
) -> BinarySelection:
    del cat_encoding
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
            include_original=run.include_original,
            prefix_widths=run.prefix_widths,
        )
        score_test_ic_approximation = False

    path_length = (
        len(run.prefix_widths)
        if run.prefix_widths
        else len(run.path.selected_features)
    )
    n_candidates = problem.n_features_input
    df_path = None
    ic_dimension = "k"
    if run.prefix_widths:
        from sift.selection.cefsplus_binary import binary_logistic_prefix_df

        ic_dimension = "df"
        n_candidates = max(int(run.n_discovery_candidates or path_length), path_length, 1)
        df_path = binary_logistic_prefix_df(
            run.X_sub.astype(np.float64, copy=False),
            run.w_sub.astype(np.float64, copy=False),
            run.path.selected_original,
            run.prefix_widths,
            include_indices=run.include_original,
        )
    selected_count, auto_diag = _select_penalized_count(
        auto_objective,
        auto_k_config,
        objective_scale=2.0,
        n_samples=len(run.y_sub),
        sample_weight=run.w_sub,
        n_candidates=n_candidates,
        path_length=path_length,
        df_path=df_path,
        ic_dimension=ic_dimension,
    )
    ic_likelihood_type = (
        "weighted_pseudo_likelihood" if problem.weighted else "bernoulli_log_likelihood"
    )
    objective_fit = "score_test_approximation" if score_test_ic_approximation else "ridge_fit_unpenalized_loglik_score"
    if auto_diag is not None and not auto_diag.empty:
        auto_diag["binary_objective_mode"] = auto_k_config.binary_objective_mode
        auto_diag["binary_objective_fit"] = objective_fit
        auto_diag["score_test_ic_approximation"] = score_test_ic_approximation
        auto_diag["ic_likelihood_type"] = ic_likelihood_type
        auto_diag["binary_refit_failures"] = binary_refit_failures
        auto_diag["refit_every_warning"] = bool(
            score_test_ic_approximation and options.refit_every > 1
        )
    _print_selected_k("Penalized objective", selected_count, verbose)

    warnings = []
    if score_test_ic_approximation and options.refit_every > 1:
        warnings.append("refit_every > 1 makes cumulative score-test gains more approximate")
    effective_max_k = _effective_max_k(auto_k_config, path_length)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra={
            "proxy_only_objective": True,
            "objective_penalty": auto_k_config.objective_penalty,
            "objective_scale": "binary_loglik_gain",
            "binary_objective_mode": auto_k_config.binary_objective_mode,
            "binary_objective_fit": objective_fit,
            "score_test_ic_approximation": score_test_ic_approximation,
            "ic_likelihood_type": ic_likelihood_type,
            "binary_refit_failures": binary_refit_failures,
            "warnings": warnings,
        },
    )
    return binary_selection_prefix(
        run.path,
        selected_count,
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=summary,
    )


def select_binary_posterior(
    _X, problem: BinaryProblem, run: BinaryPathRun, options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
) -> BinarySelection:
    del cat_encoding
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
            include_original=run.include_original,
        )
        score_test_ic_approximation = False

    path_length = len(run.path.selected_features)
    selected_count, auto_diag = _select_posterior_count(
        auto_objective,
        auto_k_config,
        objective_scale=2.0,
        n_samples=len(run.y_sub),
        sample_weight=run.w_sub,
        n_candidates=problem.n_features_input,
        path_length=path_length,
    )
    if auto_diag is not None and not auto_diag.empty:
        auto_diag["binary_objective_mode"] = auto_k_config.binary_objective_mode
        auto_diag["score_test_ic_approximation"] = score_test_ic_approximation
        auto_diag["binary_refit_failures"] = binary_refit_failures
    _print_selected_k("K posterior", selected_count, verbose)
    extra = {
        "proxy_only_objective": True,
        "objective_scale": "binary_loglik_gain",
        "binary_objective_mode": auto_k_config.binary_objective_mode,
        "score_test_ic_approximation": score_test_ic_approximation,
        "binary_refit_failures": binary_refit_failures,
    }
    if auto_diag is not None and not auto_diag.empty:
        for column in ("posterior_level", "hpd_lo", "hpd_hi", "p_zero", "entropy", "ebic_gamma"):
            extra[column] = auto_diag[column].iloc[0]
    effective_max_k = _effective_max_k(auto_k_config, path_length)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra=extra,
    )
    return binary_selection_prefix(
        run.path,
        selected_count,
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=summary,
    )


def select_binary_changepoint(
    _X, _problem: BinaryProblem, run: BinaryPathRun, _options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
) -> BinarySelection:
    del cat_encoding
    auto_objective = np.cumsum(np.asarray(run.path.path_scores, dtype=np.float64))
    path_length = len(run.path.selected_features)
    n_eff, n_eff_source = _objective_n_eff(auto_k_config, run.w_sub, len(run.y_sub))
    selected_count, auto_diag = select_k_changepoint(
        auto_objective,
        auto_k_config,
        objective_scale=2.0,
        n_eff=n_eff,
        p_candidates=len(run.feature_names),
    )
    selected_count = min(selected_count, path_length)
    _print_selected_k("Changepoint", selected_count, verbose)
    extra = {
        "proxy_only_objective": True,
        "objective_scale": "binary_score_test_gain",
        "score_test_objective_approximation": True,
        "n_eff": n_eff,
        "n_eff_source": n_eff_source,
        "floor_z": auto_k_config.floor_z,
    }
    if auto_diag is not None and not auto_diag.empty:
        extra["floor_not_reached"] = bool(auto_diag["floor_not_reached"].iloc[0])
    effective_max_k = int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra=extra,
    )
    return binary_selection_prefix(
        run.path,
        selected_count,
        auto_diag=auto_diag,
        auto_objective=auto_objective,
        auto_summary=summary,
    )


def select_binary_evaluate(
    X, problem: BinaryProblem, run: BinaryPathRun, options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    onehot_raw_X=None,
    onehot_encoder=None,
    onehot_cat_features=None,
    onehot_max_levels: int = 32,
    onehot_include_names=None,
) -> BinarySelection:
    eval_X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=run.feature_names)
    eval_X = eval_X.iloc[run.row_idx]
    eval_y = problem.y01[run.row_idx]
    eval_groups = problem.groups[run.row_idx] if problem.groups is not None else None
    eval_time = problem.time[run.row_idx] if problem.time is not None else None

    _require_eval_split_context(auto_k_config, eval_groups, eval_time)
    remapped = remap_onehot_prefix_evaluate(
        path_names=list(run.path.selected_features),
        eval_X=eval_X,
        onehot_raw_X=onehot_raw_X,
        onehot_encoder=onehot_encoder,
        onehot_cat_features=onehot_cat_features,
        onehot_max_levels=onehot_max_levels,
        onehot_include_names=onehot_include_names,
        row_idx=run.row_idx,
        encoded_prefix_sizes=run.prefix_widths,
    )
    eval_frame = eval_X
    eval_path = run.path.selected_features
    eval_cat_features = run.cat_features
    eval_cat_encoding = cat_encoding
    eval_base = [run.feature_names[int(i)] for i in run.include_original]
    eval_prefix = run.prefix_widths
    eval_max_levels = onehot_max_levels
    if remapped is not None:
        eval_frame = remapped["eval_X"]
        eval_path = remapped["path"]
        eval_cat_features = remapped["cat_features"]
        eval_cat_encoding = remapped["cat_encoding"]
        eval_base = remapped["base_features"]
        eval_prefix = remapped["prefix_sizes"]
        eval_max_levels = remapped["onehot_max_levels"]
    best_k, selected_features, auto_diag = auto_k_module.select_k_auto(
        eval_frame,
        eval_y,
        eval_path,
        auto_k_config,
        groups=eval_groups,
        time=eval_time,
        task="classification",
        base_features=eval_base,
        cat_features=eval_cat_features,
        cat_encoding=eval_cat_encoding,
        onehot_max_levels=eval_max_levels,
        sample_weight=(
            run.w_sub
            if problem.weighted
            or (
                cat_encoding == "target_cv"
                and problem.time is not None
                and target_prior is None
            )
            else None
        ),
        loo_smoothing=options.loo_smoothing,
        loo_clip_min=options.loo_clip_min,
        loo_clip_max=options.loo_clip_max,
        target_cv_n_splits=target_cv_n_splits,
        target_cv_smoothing=target_cv_smoothing,
        target_prior=target_prior,
        warmup_policy=warmup_policy,
        prefix_sizes=eval_prefix,
    )
    selected_count = int(best_k)
    if remapped is not None:
        selected_features = None
    _print_selected_k("CV/holdout", best_k, verbose)
    path_length = (
        len(eval_prefix)
        if eval_prefix
        else len(eval_path)
    )
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=_effective_max_k(auto_k_config, path_length),
        diagnostics=auto_diag,
        extra={"proxy_only_objective": False},
    )
    return binary_selection_prefix(
        run.path,
        selected_count,
        selected_features=selected_features,
        auto_diag=auto_diag,
        auto_summary=summary,
    )
