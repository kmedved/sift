"""Filter-selector orchestration around automatic k selection."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from sift._preprocess import ensure_weights
from sift.selection import auto_k as auto_k_module
from sift.selection.auto_k import AutoKConfig
from sift.selection.cefsplus_binary_common import (
    BinaryOptions,
    BinaryPathRun,
    BinaryProblem,
    BinarySelection,
    binary_refit_loglik_gains,
    binary_selection_prefix,
)

EvalData = tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
GaussianAutoKResult = tuple[list[str], list[int], pd.DataFrame, dict]


def auto_k_summary(
    config: AutoKConfig, *, selected_k: int, path_length: int, effective_max_k: int,
    effective_min_k: Optional[int] = None, diagnostics: Optional[pd.DataFrame] = None,
    extra: Optional[dict] = None,
) -> dict:
    if effective_min_k is None:
        effective_min_k = max(1, min(int(config.min_k), int(effective_max_k)))
    summary = {
        "method": config.k_method,
        "selection_rule": config.selection_rule,
        "selected_k": int(selected_k),
        "min_k": int(config.min_k),
        "max_k": int(config.max_k),
        "effective_min_k": int(effective_min_k),
        "effective_max_k": int(effective_max_k),
        "path_length": int(path_length),
        "selected_at_min_k": bool(selected_k == int(effective_min_k)),
        "selected_at_effective_max_k": bool(selected_k == effective_max_k),
        "selected_at_config_max_k": bool(selected_k == int(config.max_k)),
        "path_exhausted_before_max_k": bool(effective_max_k < int(config.max_k)),
    }
    if diagnostics is not None and not diagnostics.empty:
        for column, cast in (
            ("best_k", int),
            ("best_score", float),
            ("one_se_unavailable", bool),
            ("objective_nonmonotone_steps", int),
        ):
            if column in diagnostics:
                summary[column] = cast(diagnostics[column].iloc[0])
        if "selection_rule_effective" in diagnostics:
            summary["selection_rule_effective"] = diagnostics["selection_rule_effective"].iloc[0]
    if extra:
        summary.update(extra)
    return summary


def prepare_filter_eval_data(
    X, y: np.ndarray, cache, groups: Optional[np.ndarray], time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray],
) -> EvalData:
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

    if not use_cache_rows:
        return X_df, y_arr, groups, time, eval_weight
    return (
        X_df.iloc[cache.row_idx],
        y_arr[cache.row_idx],
        groups[cache.row_idx] if groups is not None else None,
        time[cache.row_idx] if time is not None else None,
        eval_weight,
    )


def auto_k_mode_label(config: AutoKConfig) -> str:
    labels = {
        "elbow": "elbow",
        "penalized_objective": f"penalized_objective/{config.objective_penalty}",
        "evaluate": f"evaluate/{config.strategy}/{config.selection_rule}",
    }
    return labels[config.k_method]


def _effective_max_k(config: AutoKConfig, path_length: int) -> int:
    return min(int(config.max_k), int(path_length))


def _require_eval_split_context(
    config: AutoKConfig,
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
) -> None:
    if config.strategy == "time_holdout" and time is None:
        raise ValueError("auto-k evaluate with strategy='time_holdout' requires time parameter")
    if config.strategy == "group_cv" and groups is None:
        raise ValueError("auto-k evaluate with strategy='group_cv' requires groups parameter")


def _print_selected_k(label: str, selected_count: int, verbose: bool) -> None:
    if verbose:
        print(f"  {label} selected k={selected_count}")


def _select_elbow_count(
    objective: np.ndarray,
    config: AutoKConfig,
    path_length: int,
) -> tuple[int, pd.DataFrame]:
    best_k, diagnostics = auto_k_module.select_k_elbow(
        objective,
        min_k=config.min_k,
        max_k=path_length,
        min_rel_gain=config.elbow_min_rel_gain,
        patience=config.elbow_patience,
    )
    return min(best_k, path_length), diagnostics


def _select_penalized_count(
    objective: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale,
    n_samples: int,
    sample_weight: Optional[np.ndarray],
    path_length: int,
) -> tuple[int, pd.DataFrame]:
    best_k, diagnostics = auto_k_module.select_k_penalized_objective(
        objective,
        config,
        objective_scale=objective_scale,
        n_samples=n_samples,
        sample_weight=sample_weight,
        min_k=config.min_k,
        max_k=path_length,
    )
    return min(best_k, path_length), diagnostics


def select_gaussian_evaluate_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int,
    auto_k_config: AutoKConfig, eval_X: pd.DataFrame, eval_y: np.ndarray,
    groups: Optional[np.ndarray], time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray], cat_features: Optional[list[str]], cat_encoding: str,
    corr_prune: float | None | Literal["auto"] = "auto",
    verbose: bool = True,
) -> GaussianAutoKResult:
    _require_eval_split_context(auto_k_config, groups, time)

    path, path_indices, _ = _cached_filter_path(
        cache,
        y,
        max_k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        want_indices=True,
        return_objective=False,
    )
    best_k, selected, auto_diag = auto_k_module.select_k_auto(
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
    _print_selected_k("CV/holdout", best_k, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=len(selected),
        path_length=len(path),
        effective_max_k=_effective_max_k(auto_k_config, len(path)),
        diagnostics=auto_diag,
        extra={"proxy_only_objective": False},
    )
    return selected, path_indices[: len(selected)], auto_diag, summary


def select_gaussian_elbow_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    path, path_indices, objective = _cached_filter_path(
        cache,
        y,
        max_k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        want_indices=True,
        return_objective=True,
    )
    selected_count, auto_diag = _select_elbow_count(objective, auto_k_config, len(path))
    _print_selected_k("Elbow", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=_effective_max_k(auto_k_config, len(path)),
        diagnostics=auto_diag,
        extra={"proxy_only_objective": True},
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_penalized_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    path, path_indices, objective = _cached_filter_path(
        cache,
        y,
        max_k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        want_indices=True,
        return_objective=True,
    )
    selected_count, auto_diag = _select_penalized_count(
        objective,
        auto_k_config,
        objective_scale="n_eff",
        n_samples=len(cache.sample_weight),
        sample_weight=cache.sample_weight,
        path_length=len(path),
    )
    _print_selected_k("Penalized objective", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=_effective_max_k(auto_k_config, len(path)),
        diagnostics=auto_diag,
        extra={
            "objective_penalty": auto_k_config.objective_penalty,
            "objective_scale": "gaussian_2mi",
            "proxy_only_objective": True,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def _cached_filter_path(
    cache, y, k: int, *, method: str, top_m: int, corr_prune,
    want_indices: bool, return_objective: bool,
) -> tuple[list[str], list[int], np.ndarray | None]:
    from sift.selection.cefsplus import select_cached

    result = select_cached(
        cache,
        y,
        k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        return_indices=want_indices,
        return_objective=return_objective,
    )
    if return_objective and want_indices:
        path, indices, objective = result
        return path, list(indices), objective
    if return_objective:
        path, objective = result
        return path, [], objective
    if want_indices:
        path, indices = result
        return path, list(indices), None
    return result, [], None


def select_filter_classic_auto_k(
    *, y_arr: np.ndarray, eval_X: pd.DataFrame, feature_names: list[str], path_idx: np.ndarray,
    auto_k_config: AutoKConfig, eval_groups: Optional[np.ndarray], eval_time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray],
    task: Literal["regression", "classification"],
    cat_features: Optional[list[str]], cat_encoding: str,
    verbose: bool = True,
    return_indices: bool = False,
) -> list[str] | tuple[list[str], list[int]]:
    path = [feature_names[i] for i in path_idx]
    _require_eval_split_context(auto_k_config, eval_groups, eval_time)
    best_k, selected, _ = auto_k_module.select_k_auto(
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
    _print_selected_k("CV/holdout", best_k, verbose)
    if return_indices:
        return selected, [int(i) for i in path_idx[: len(selected)]]
    return selected


def select_binary_elbow(
    _X, _problem: BinaryProblem, run: BinaryPathRun, _options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
) -> BinarySelection:
    del cat_encoding
    auto_objective = np.cumsum(np.asarray(run.path.path_scores, dtype=np.float64))
    path_length = len(run.path.selected_features)
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
        )
        score_test_ic_approximation = False

    path_length = len(run.path.selected_features)
    selected_count, auto_diag = _select_penalized_count(
        auto_objective,
        auto_k_config,
        objective_scale=2.0,
        n_samples=len(run.y_sub),
        sample_weight=run.w_sub,
        path_length=path_length,
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
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=_effective_max_k(auto_k_config, path_length),
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


def select_binary_evaluate(
    X, problem: BinaryProblem, run: BinaryPathRun, options: BinaryOptions, *,
    auto_k_config: AutoKConfig, cat_encoding: str, verbose: bool,
) -> BinarySelection:
    eval_X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=run.feature_names)
    if len(run.row_idx) < problem.n_rows:
        eval_X = eval_X.iloc[run.row_idx]
        eval_y = problem.y01[run.row_idx]
        eval_groups = problem.groups[run.row_idx] if problem.groups is not None else None
        eval_time = problem.time[run.row_idx] if problem.time is not None else None
    else:
        eval_y = problem.y01
        eval_groups = problem.groups
        eval_time = problem.time

    _require_eval_split_context(auto_k_config, eval_groups, eval_time)
    best_k, selected_features, auto_diag = auto_k_module.select_k_auto(
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
    _print_selected_k("CV/holdout", best_k, verbose)
    path_length = len(run.path.selected_features)
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
