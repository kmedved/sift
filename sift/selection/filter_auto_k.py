"""Filter-selector orchestration around automatic k selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Optional
import time as time_module
import warnings

import numpy as np
import pandas as pd

from sift._preprocess import ensure_weights
from sift.selection import auto_k as auto_k_module
from sift.selection.auto_k import AutoKConfig
from sift.selection.auto_k_knockoff import select_k_knockoff_path
from sift.selection.auto_k_resample import (
    bootstrap_paths,
    null_objective_paths,
    select_k_perm_gap,
    select_k_stability,
)
from sift.selection.auto_k_stop import (
    select_k_changepoint,
    select_k_chi2_stop,
    select_k_forward_stop,
)
from sift.selection.auto_k_xfit import (
    gaussian_cv_curves,
    select_k_gaussian_cv,
    select_k_xfit_objective,
    xfit_objective_curves,
)
from sift.selection.cefsplus_binary_common import (
    BinaryOptions,
    BinaryPathRun,
    BinaryProblem,
    BinarySelection,
    binary_refit_loglik_gains,
    binary_selection_prefix,
)
from sift.selection.panel import build_candidate_panel

EvalData = tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
GaussianAutoKResult = tuple[list[str], list[int], pd.DataFrame, dict]


def auto_k_summary(
    config: AutoKConfig, *, selected_k: int, path_length: int, effective_max_k: int,
    effective_min_k: Optional[int] = None, diagnostics: Optional[pd.DataFrame] = None,
    extra: Optional[dict] = None,
) -> dict:
    if effective_min_k is None:
        effective_min_k = max(1, min(int(config.min_k), int(effective_max_k)))
    configured_max_k = int(config.max_k)
    path_length = int(path_length)
    effective_max_k = int(effective_max_k)
    summary = {
        "method": config.k_method,
        "selection_rule": config.selection_rule,
        "selected_k": int(selected_k),
        "min_k": int(config.min_k),
        "max_k": configured_max_k,
        "effective_min_k": int(effective_min_k),
        "effective_max_k": effective_max_k,
        "path_length": path_length,
        "selected_at_min_k": bool(selected_k == int(effective_min_k)),
        "selected_at_effective_max_k": bool(selected_k == effective_max_k),
        "selected_at_config_max_k": bool(selected_k == configured_max_k),
        "selected_at_path_end": bool(selected_k == path_length),
        "path_exhausted_before_max_k": bool(path_length < configured_max_k),
        "evaluation_limited_before_path_end": bool(
            effective_max_k < min(path_length, configured_max_k)
        ),
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


def _zero_capable_effective_min_k(
    config: AutoKConfig,
    *,
    selected_k: int,
    effective_max_k: int,
) -> int:
    if int(config.min_k) <= 0 or int(selected_k) == 0:
        return 0
    return max(1, min(int(config.min_k), int(effective_max_k)))


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


def auto_k_mode_label(config: AutoKConfig) -> str:
    labels = {
        "auto": "auto",
        "chi2_stop": f"chi2_stop/alpha={config.alpha:g}",
        "forward_stop": f"forward_stop/alpha={config.alpha:g}",
        "perm_gap": f"perm_gap/{config.perm_null}/{config.gap_rule}",
        "knockoff_path": f"knockoff_path/q={config.knockoff_q:g}/{config.knockoff_return}",
        "xfit_objective": f"xfit_objective/{config.strategy}/{config.selection_rule}",
        "gaussian_cv": f"gaussian_cv/{config.strategy}/{config.selection_rule}",
        "k_posterior": f"k_posterior/{config.posterior_pick}",
        "stability": f"stability/{config.stability_rule}",
        "changepoint": "changepoint",
        "consensus": "consensus",
        "elbow": "elbow",
        "penalized_objective": f"penalized_objective/{config.objective_penalty}",
        "evaluate": f"evaluate/{config.strategy}/{config.selection_rule}",
    }
    return labels[config.k_method]


def _auto_route_facts(cache, *, method: str, groups, time) -> dict:
    w = np.asarray(cache.sample_weight, dtype=np.float64).reshape(-1)
    weight_sum = float(np.sum(w))
    sum_sq = float(np.sum(w * w))
    n_eff_kish = float(weight_sum * weight_sum / sum_sq) if sum_sq > 0.0 else float("nan")
    n_rows = int(w.size)
    p_valid = int(len(cache.valid_cols))
    return {
        "selector_method": method,
        "n_rows": n_rows,
        "p_valid": p_valid,
        "n_eff_kish": n_eff_kish,
        "n_eff_over_p": float(n_eff_kish / p_valid) if p_valid > 0 else float("inf"),
        "weight_skew_ratio": float(n_eff_kish / n_rows) if n_rows > 0 else float("nan"),
        "has_groups": groups is not None,
        "has_time": time is not None,
    }


_AUTOK_FIELD_DEFAULTS = AutoKConfig()


def _strip_router_only_fields(config: AutoKConfig) -> AutoKConfig:
    """Reset fields consumed by the router itself before dispatching a routed method.

    The dense-check knobs are read from the caller's original ``auto`` config;
    leaving them on the routed copy makes the routed method's unused-field
    validation warn about options the router already honored.
    """
    return replace(
        config,
        auto_dense_check=_AUTOK_FIELD_DEFAULTS.auto_dense_check,
        auto_dense_min_k=_AUTOK_FIELD_DEFAULTS.auto_dense_min_k,
        auto_dense_min_frac=_AUTOK_FIELD_DEFAULTS.auto_dense_min_frac,
        auto_dense_disagreement_ratio=_AUTOK_FIELD_DEFAULTS.auto_dense_disagreement_ratio,
    )


def _auto_route_config(config: AutoKConfig, facts: dict) -> tuple[AutoKConfig, str]:
    if facts["selector_method"] != "cefsplus":
        strategy = config.strategy
        if strategy == "time_holdout" and not facts["has_time"]:
            strategy = "kfold"
        if strategy == "group_cv" and not facts["has_groups"]:
            strategy = "kfold"
        # A time holdout is a single split, so the one-SE rule has no split
        # spread to work with and would immediately fall back to "best" with a
        # warning. Route that case to "best" explicitly.
        selection_rule = "best" if strategy == "time_holdout" else "one_se"
        routed = replace(
            config,
            k_method="gaussian_cv",
            strategy=strategy,
            selection_rule=selection_rule,
            min_k=max(1, int(config.min_k)),
        )
        reason = "non_cefsplus_gaussian_selector"
    elif facts["p_valid"] > facts["n_eff_kish"]:
        routed = replace(
            config,
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=0,
        )
        reason = "p_valid_exceeds_kish_n_eff"
    elif facts["weight_skew_ratio"] < 0.8:
        routed = replace(
            config,
            k_method="perm_gap",
            min_k=0,
            perm_null="auto",
        )
        reason = "heavy_weight_skew"
    else:
        routed = replace(
            config,
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=0,
        )
        reason = "measured_default_ebic"
    return _strip_router_only_fields(routed), reason


def _run_gaussian_routed_path(routed_config: AutoKConfig, **kwargs) -> GaussianAutoKResult:
    method = routed_config.k_method
    kwargs = dict(kwargs)
    kwargs["auto_k_config"] = routed_config
    if method == "gaussian_cv":
        return select_gaussian_cv_path(**kwargs)
    if method == "perm_gap":
        return select_gaussian_perm_gap_path(**kwargs)
    if method == "penalized_objective":
        return select_gaussian_penalized_path(**kwargs)
    if method == "chi2_stop":
        return select_gaussian_chi2_path(**kwargs)
    fallback_config = replace(
        routed_config,
        k_method="penalized_objective",
        objective_penalty="ebic",
        min_k=0,
    )
    kwargs["auto_k_config"] = fallback_config
    return select_gaussian_penalized_path(**kwargs)


def _auto_dense_check_requested(config: AutoKConfig, summary: dict, route: dict) -> tuple[bool, str]:
    if not bool(config.auto_dense_check):
        return False, "disabled"
    if route.get("chosen") != "penalized_objective" or route.get("objective_penalty") != "ebic":
        return False, "route_not_ebic"
    selected_k = int(summary.get("selected_k", 0))
    effective_max_k = int(summary.get("effective_max_k", config.max_k))
    if effective_max_k <= 0:
        return False, "empty_path"
    large_by_count = selected_k >= int(config.auto_dense_min_k)
    large_by_fraction = selected_k >= float(config.auto_dense_min_frac) * effective_max_k
    if not (large_by_count or large_by_fraction):
        return False, "selected_k_not_large"
    return True, "large_ebic_pick"


def _run_auto_dense_check(
    *,
    config: AutoKConfig,
    summary: dict,
    route: dict,
    cache,
    y: np.ndarray,
    method: str,
    groups,
    time,
    top_m: int,
    corr_prune,
) -> None:
    should_run, reason = _auto_dense_check_requested(config, summary, route)
    if not bool(config.auto_dense_check):
        return

    selected_k = int(summary.get("selected_k", 0))
    effective_max_k = int(summary.get("effective_max_k", config.max_k))
    route["dense_check"] = {
        "enabled": True,
        "reason": reason,
        "ran": False,
        "ebic_k": selected_k,
        "effective_max_k": effective_max_k,
        "method": "gaussian_cv/best",
        "min_k": int(config.auto_dense_min_k),
        "min_frac": float(config.auto_dense_min_frac),
        "disagreement_ratio": float(config.auto_dense_disagreement_ratio),
    }
    if not should_run:
        return

    strategy = config.strategy
    if strategy == "time_holdout" and time is None:
        strategy = "kfold"
    if strategy == "group_cv" and groups is None:
        strategy = "kfold"
    check_config = replace(
        config,
        k_method="gaussian_cv",
        strategy=strategy,
        selection_rule="best",
        min_k=max(1, min(int(config.min_k), max(1, effective_max_k))),
        max_k=max(1, effective_max_k),
        auto_dense_check=False,
        auto_dense_min_k=100,
        auto_dense_min_frac=0.25,
        auto_dense_disagreement_ratio=2.0,
    )
    try:
        curves = gaussian_cv_curves(
            cache,
            y,
            config=check_config,
            groups=groups,
            time=time,
            top_m=top_m,
            corr_prune=corr_prune,
            method=method,
        )
        cv_k, _diag = select_k_gaussian_cv(curves, check_config)
    except Exception as exc:  # pragma: no cover - rare defensive diagnostics path
        route["dense_check"].update(
            {
                "reason": "gaussian_cv_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        warnings.warn(
            "Auto-K dense check could not run gaussian_cv/best; selected k is unchanged. "
            f"Reason: {type(exc).__name__}: {exc}",
            UserWarning,
            stacklevel=3,
        )
        return

    cv_k = int(min(max(cv_k, 0), effective_max_k))
    denom = max(1, min(selected_k, cv_k))
    ratio = float(max(selected_k, cv_k) / denom)
    disagrees = ratio > float(config.auto_dense_disagreement_ratio)
    route["dense_check"].update(
        {
            "ran": True,
            "reason": "checked",
            "gaussian_cv_best_k": cv_k,
            "strategy": strategy,
            "ratio": ratio,
            "warned": disagrees,
        }
    )
    if disagrees:
        warnings.warn(
            "Auto-K dense-signal diagnostic: EBIC counts detectable features "
            f"(k={selected_k}); for downstream sizing consider gaussian_cv/best "
            f"or a prefix-risk curve (k≈{cv_k}).",
            UserWarning,
            stacklevel=3,
        )


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
    if path_length <= 0:
        return 0, pd.DataFrame()
    best_k, diagnostics = auto_k_module.select_k_elbow(
        objective,
        min_k=min(int(config.min_k), int(path_length)),
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
    n_candidates: int | None,
    path_length: int,
) -> tuple[int, pd.DataFrame]:
    best_k, diagnostics = auto_k_module.select_k_penalized_objective(
        objective,
        config,
        objective_scale=objective_scale,
        n_samples=n_samples,
        sample_weight=sample_weight,
        n_candidates=n_candidates,
        min_k=config.min_k,
        max_k=path_length,
    )
    return min(best_k, path_length), diagnostics


def _select_posterior_count(
    objective: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale,
    n_samples: int,
    sample_weight: Optional[np.ndarray],
    n_candidates: int,
    path_length: int,
) -> tuple[int, pd.DataFrame]:
    best_k, diagnostics = auto_k_module.select_k_posterior(
        objective,
        config,
        objective_scale=objective_scale,
        n_samples=n_samples,
        sample_weight=sample_weight,
        n_candidates=n_candidates,
        min_k=config.min_k,
        max_k=path_length,
    )
    return min(best_k, path_length), diagnostics


def _objective_n_eff(config: AutoKConfig, sample_weight, n_samples: int) -> tuple[float, str]:
    _w, _weight_sum, _kish, n_eff, n_eff_source = auto_k_module._objective_weight_diagnostics(
        sample_weight,
        n_samples,
        config,
    )
    return float(n_eff), str(n_eff_source)


def _gain_test_candidate_inputs(
    cache,
    y,
    k: int,
    top_m: int,
    corr_prune,
    method: str,
    config: AutoKConfig,
) -> tuple[int, np.ndarray | None]:
    if config.m_mode == "all":
        return len(cache.valid_cols), None
    panel = build_candidate_panel(
        cache,
        y,
        k,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
    )
    if config.m_mode == "panel":
        return len(panel.cand), None
    return len(cache.valid_cols), np.linalg.eigvalsh(panel.R) if panel.R.size else None


def select_gaussian_evaluate_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int,
    auto_k_config: AutoKConfig, eval_X: pd.DataFrame, eval_y: np.ndarray,
    groups: Optional[np.ndarray], time: Optional[np.ndarray],
    sample_weight: Optional[np.ndarray], cat_features: Optional[list[str]], cat_encoding: str,
    corr_prune: float | None | Literal["auto"] = "auto",
    feature_names: Optional[list[str]] = None,
    verbose: bool = True,
    **_unused,
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
    eval_feature_index = {name: idx for idx, name in enumerate(eval_X.columns)}
    selected_indices = [int(eval_feature_index[name]) for name in selected]
    return selected, selected_indices, auto_diag, summary


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
        n_candidates=len(cache.valid_cols),
        path_length=len(path),
    )
    _print_selected_k("Penalized objective", selected_count, verbose)
    effective_max_k = _effective_max_k(auto_k_config, len(path))
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra={
            "objective_penalty": auto_k_config.objective_penalty,
            "objective_scale": "gaussian_2mi",
            "proxy_only_objective": True,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_posterior_path(
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
    selected_count, auto_diag = _select_posterior_count(
        objective,
        auto_k_config,
        objective_scale="n_eff",
        n_samples=len(cache.sample_weight),
        sample_weight=cache.sample_weight,
        n_candidates=len(cache.valid_cols),
        path_length=len(path),
    )
    _print_selected_k("K posterior", selected_count, verbose)
    extra = {
        "objective_scale": "gaussian_2mi",
        "proxy_only_objective": True,
    }
    if auto_diag is not None and not auto_diag.empty:
        for column in ("posterior_level", "hpd_lo", "hpd_hi", "p_zero", "entropy", "ebic_gamma"):
            extra[column] = auto_diag[column].iloc[0]
    effective_max_k = _effective_max_k(auto_k_config, len(path))
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra=extra,
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_chi2_path(
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
    n_eff, n_eff_source = _objective_n_eff(auto_k_config, cache.sample_weight, len(cache.sample_weight))
    p_candidates, panel_eigs = _gain_test_candidate_inputs(
        cache,
        y,
        max_k,
        top_m,
        corr_prune,
        method,
        auto_k_config,
    )
    selected_count, auto_diag = select_k_chi2_stop(
        objective,
        auto_k_config,
        n_eff=n_eff,
        p_candidates=p_candidates,
        panel_eigs=panel_eigs,
    )
    selected_count = min(selected_count, len(path))
    _print_selected_k("Chi2 stop", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0,
        effective_min_k=0,
        diagnostics=auto_diag,
        extra={
            "alpha": auto_k_config.alpha,
            "m_mode": auto_k_config.m_mode,
            "n_eff": n_eff,
            "n_eff_source": n_eff_source,
            "proxy_only_objective": True,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_forward_stop_path(
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
    n_eff, n_eff_source = _objective_n_eff(auto_k_config, cache.sample_weight, len(cache.sample_weight))
    p_candidates, panel_eigs = _gain_test_candidate_inputs(
        cache,
        y,
        max_k,
        top_m,
        corr_prune,
        method,
        auto_k_config,
    )
    selected_count, auto_diag = select_k_forward_stop(
        objective,
        auto_k_config,
        n_eff=n_eff,
        p_candidates=p_candidates,
        panel_eigs=panel_eigs,
    )
    selected_count = min(selected_count, len(path))
    _print_selected_k("ForwardStop", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0,
        effective_min_k=0,
        diagnostics=auto_diag,
        extra={
            "alpha": auto_k_config.alpha,
            "m_mode": auto_k_config.m_mode,
            "n_eff": n_eff,
            "n_eff_source": n_eff_source,
            "proxy_only_objective": True,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_changepoint_path(
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
    n_eff, n_eff_source = _objective_n_eff(auto_k_config, cache.sample_weight, len(cache.sample_weight))
    selected_count, auto_diag = select_k_changepoint(
        objective,
        auto_k_config,
        objective_scale=n_eff,
        n_eff=n_eff,
        p_candidates=len(cache.valid_cols),
    )
    selected_count = min(selected_count, len(path))
    _print_selected_k("Changepoint", selected_count, verbose)
    extra = {
        "objective_scale": "gaussian_n_eff_gain",
        "n_eff": n_eff,
        "n_eff_source": n_eff_source,
        "floor_z": auto_k_config.floor_z,
        "proxy_only_objective": True,
    }
    if auto_diag is not None and not auto_diag.empty:
        extra["floor_not_reached"] = bool(auto_diag["floor_not_reached"].iloc[0])
    effective_max_k = int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra=extra,
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_perm_gap_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    source_groups: Optional[np.ndarray] = None,
    source_time: Optional[np.ndarray] = None,
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    del groups, time
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
    nulls = null_objective_paths(
        cache,
        y,
        B=int(auto_k_config.perm_B),
        max_k=len(path),
        null=auto_k_config.perm_null,
        time=source_time,
        groups=source_groups,
        top_m=top_m,
        corr_prune=corr_prune,
        random_state=int(auto_k_config.random_state),
    )
    selected_count, auto_diag = select_k_perm_gap(objective, nulls, auto_k_config)
    selected_count = min(selected_count, len(path))
    _print_selected_k("Permutation gap", selected_count, verbose)
    effective_max_k = int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra={
            "perm_B": int(auto_k_config.perm_B),
            "perm_null": auto_k_config.perm_null,
            "gap_rule": auto_k_config.gap_rule,
            "proxy_only_objective": True,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_xfit_objective_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    path, path_indices, _objective = _cached_filter_path(
        cache,
        y,
        max_k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        want_indices=True,
        return_objective=False,
    )
    curves = xfit_objective_curves(
        cache,
        y,
        config=auto_k_config,
        groups=groups,
        time=time,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
    )
    selected_count, auto_diag = select_k_xfit_objective(curves, auto_k_config)
    selected_count = min(selected_count, len(path))
    _print_selected_k("Cross-fit objective", selected_count, verbose)
    effective_max_k = int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0
    stopped_by = None
    if auto_diag is not None:
        stopped_by = auto_diag.attrs.get("stopped_by")
        if stopped_by is None and not auto_diag.empty and "stopped_by" in auto_diag:
            stopped_by = auto_diag["stopped_by"].iloc[0]
    extra = {
        "xfit_mode": auto_k_config.xfit_mode,
        "xfit_folds": int(auto_diag["n_splits"].max()) if auto_diag is not None and not auto_diag.empty else 0,
        "debias": True,
        "proxy_only_objective": True,
    }
    if stopped_by:
        extra["stopped_by"] = stopped_by
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra=extra,
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_cv_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    path, path_indices, _objective = _cached_filter_path(
        cache,
        y,
        max_k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        want_indices=True,
        return_objective=False,
    )
    curves = gaussian_cv_curves(
        cache,
        y,
        config=auto_k_config,
        groups=groups,
        time=time,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
    )
    selected_count, auto_diag = select_k_gaussian_cv(curves, auto_k_config)
    selected_count = min(selected_count, len(path))
    stopped_by = None
    if auto_diag is not None:
        stopped_by = auto_diag.attrs.get("stopped_by")
    if stopped_by is None:
        stopped_by = curves.attrs.get("stopped_by")
    _print_selected_k("Gaussian CV", selected_count, verbose)
    effective_max_k = (
        int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else len(path)
    )
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        diagnostics=auto_diag,
        extra={
            "xfit_mode": auto_k_config.xfit_mode,
            "xfit_folds": int(auto_diag["n_splits"].max()) if auto_diag is not None and not auto_diag.empty else 0,
            "proxy": "gaussian_linear_copula",
            "xfit_ridge": float(auto_k_config.xfit_ridge),
            "proxy_only_objective": False,
            "stopped_by": stopped_by,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_knockoff_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    del method
    selected_valid, selected_count, auto_diag = select_k_knockoff_path(
        cache,
        y,
        auto_k_config,
        top_m=top_m,
    )
    selected_original = np.asarray(cache.valid_cols, dtype=np.int64)[selected_valid]
    if cache.feature_names is not None:
        names = list(cache.feature_names)
    else:
        n_names = int(np.max(cache.valid_cols)) + 1 if len(cache.valid_cols) else 0
        names = [f"x{i}" for i in range(n_names)]
    if auto_k_config.knockoff_return == "prefix":
        path, path_indices, _objective = _cached_filter_path(
            cache,
            y,
            max_k,
            method="cefsplus",
            top_m=top_m,
            corr_prune=corr_prune,
            want_indices=True,
            return_objective=False,
        )
        selected_count = min(selected_count, len(path))
        selected = path[:selected_count]
        selected_indices = path_indices[:selected_count]
        path_length = len(path)
        fdr_control = "none"
        approximate_fdr_control = False
    else:
        selected = [names[int(i)] for i in selected_original]
        selected_indices = selected_original.astype(int).tolist()
        path_length = selected_count
        fdr_control = (
            "approximate_plugin"
            if int(auto_k_config.knockoff_draws) == 1
            else "none"
        )
        approximate_fdr_control = int(auto_k_config.knockoff_draws) == 1
    _print_selected_k("Knockoff path", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=path_length,
        effective_max_k=path_length,
        effective_min_k=0,
        diagnostics=auto_diag,
        extra={
            "knockoff_q": float(auto_k_config.knockoff_q),
            "knockoff_draws": int(auto_k_config.knockoff_draws),
            "knockoff_s_method": auto_k_config.knockoff_s_method,
            "knockoff_return": auto_k_config.knockoff_return,
            "fdr_control": fdr_control,
            "approximate_fdr_control": approximate_fdr_control,
            "per_draw_fdr_control": "approximate_plugin",
            "q_scope": "per_draw",
            "aggregation": (
                "single_draw"
                if int(auto_k_config.knockoff_draws) == 1
                else "selection_frequency"
            ),
            "aggregation_threshold": (
                None if int(auto_k_config.knockoff_draws) == 1 else 0.5
            ),
            "aggregation_fdr_control": (
                "not_applicable"
                if int(auto_k_config.knockoff_draws) == 1
                else "none"
            ),
            "aggregation_preserves_per_draw_fdr": (
                int(auto_k_config.knockoff_draws) == 1
            ),
            "count_only": auto_k_config.knockoff_return == "prefix",
            "corr_prune_disabled": True,
        },
    )
    return selected, selected_indices, auto_diag, summary


def select_gaussian_stability_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    path, path_indices, _objective = _cached_filter_path(
        cache,
        y,
        max_k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        want_indices=True,
        return_objective=False,
    )
    boot = bootstrap_paths(
        cache,
        y,
        B=int(auto_k_config.boot_B),
        max_k=len(path),
        boot_mode=auto_k_config.boot_mode,
        top_m=top_m,
        corr_prune=corr_prune,
        random_state=int(auto_k_config.random_state),
        method=method,
    )
    selected_count, auto_diag = select_k_stability(boot, len(cache.valid_cols), auto_k_config)
    selected_count = min(selected_count, len(path))
    freq = np.zeros(len(cache.valid_cols), dtype=np.float64)
    for boot_path in boot:
        prefix = np.asarray(boot_path[:selected_count], dtype=np.int64)
        prefix = prefix[(prefix >= 0) & (prefix < len(freq))]
        if prefix.size:
            freq[prefix] += 1.0
    if boot:
        freq /= float(len(boot))
    freq_order = np.lexsort((np.arange(len(freq), dtype=np.int64), -freq))
    freq_order = freq_order[freq[freq_order] > 0.0][:100]
    top_freq = {
        int(cache.valid_cols[int(i)]): float(freq[int(i)])
        for i in freq_order
    }
    _print_selected_k("Stability", selected_count, verbose)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=int(auto_diag["k"].max()) if auto_diag is not None and not auto_diag.empty else 0,
        ),
        diagnostics=auto_diag,
        extra={
            "boot_B": int(auto_k_config.boot_B),
            "boot_mode": auto_k_config.boot_mode,
            "stability_rule": auto_k_config.stability_rule,
            "stopped_by": auto_diag.attrs.get("stopped_by")
            if auto_diag is not None
            else None,
            "max_phi": auto_diag.attrs.get("max_phi")
            if auto_diag is not None
            else None,
            "pi_at_k_hat": top_freq,
            "proxy_only_objective": True,
        },
    )
    return path[:selected_count], path_indices[:selected_count], auto_diag, summary


def select_gaussian_auto_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    source_groups: Optional[np.ndarray] = None,
    source_time: Optional[np.ndarray] = None,
    verbose: bool = True,
    **kwargs,
) -> GaussianAutoKResult:
    auto_k_module.validate_auto_k_config(auto_k_config)
    if auto_k_config.k_method != "auto":
        raise ValueError("select_gaussian_auto_path requires AutoKConfig(k_method='auto')")
    facts = _auto_route_facts(cache, method=method, groups=groups, time=time)
    routed_config, reason = _auto_route_config(auto_k_config, facts)
    route = {
        "chosen": routed_config.k_method,
        "reason": reason,
        "facts": facts,
        "objective_penalty": routed_config.objective_penalty
        if routed_config.k_method == "penalized_objective"
        else None,
    }
    runner_kwargs = {
        "cache": cache,
        "y": y,
        "method": method,
        "max_k": max_k,
        "top_m": top_m,
        "auto_k_config": routed_config,
        "corr_prune": corr_prune,
        "groups": groups,
        "time": time,
        "source_groups": source_groups,
        "source_time": source_time,
        "verbose": verbose,
        **kwargs,
    }
    selected, selected_indices, auto_diag, summary = _run_gaussian_routed_path(
        routed_config,
        **runner_kwargs,
    )
    stopped_by = summary.get("stopped_by")
    degenerate_stop = stopped_by in {"degenerate_folds", "degenerate"}
    empty_terminal_stop = not selected and stopped_by == "max_k"
    if (
        (degenerate_stop or empty_terminal_stop)
        and routed_config.k_method != "penalized_objective"
    ):
        fallback_config = replace(
            auto_k_config,
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=0,
        )
        route["primary"] = route["chosen"]
        route["chosen"] = "penalized_objective"
        route["objective_penalty"] = "ebic"
        route["fallback"] = {
            "chosen": "penalized_objective",
            "objective_penalty": "ebic",
            "reason": f"primary stopped_by={stopped_by}",
        }
        selected, selected_indices, auto_diag, summary = _run_gaussian_routed_path(
            fallback_config,
            **{**runner_kwargs, "auto_k_config": fallback_config},
        )

    summary = dict(summary)
    summary["method"] = "auto"
    summary["routed_method"] = route["chosen"]
    saturated = bool(summary.get("selected_at_effective_max_k", False))
    configured_max_k = int(summary.get("max_k", routed_config.max_k))
    effective_max_k = int(summary.get("effective_max_k", configured_max_k))
    path_length = int(summary.get("path_length", len(selected)))
    selected_k = int(summary.get("selected_k", len(selected)))
    path_exhausted = bool(path_length < configured_max_k)
    evaluation_limited = bool(
        effective_max_k < min(path_length, configured_max_k)
    )
    summary["path_exhausted_before_max_k"] = path_exhausted
    summary["evaluation_limited_before_path_end"] = evaluation_limited
    summary["selected_at_path_end"] = bool(selected_k == path_length)
    route["saturated"] = saturated
    if saturated:
        if evaluation_limited:
            route["saturation_reason"] = "evaluation_curve_limited"
            message = (
                "Auto-K router selected the effective max_k because the evaluation "
                "curve ended before the available candidate path; the result is "
                "censored at a fold/statistical limit. Increasing max_k alone "
                "cannot extend this curve; inspect fold sample sizes and evaluation "
                "diagnostics."
            )
        elif path_exhausted and selected_k >= path_length:
            route["saturation_reason"] = "candidate_path_exhausted"
            message = (
                "Auto-K router selected the effective max_k because the candidate "
                "path was exhausted before configured max_k; the result is censored "
                "at the available path boundary. Increasing max_k alone cannot "
                "extend this path; inspect valid candidates and corr_prune/top_m "
                "settings."
            )
        else:
            route["saturation_reason"] = "configured_max_k"
            message = (
                "Auto-K router selected the effective max_k; the configured max_k "
                "was reached, so the result is censored and should be interpreted "
                "as at least that many features. Increase max_k or inspect the "
                "objective/risk curve before treating this as an interior "
                "automatic-k optimum."
            )
        warnings.warn(message, UserWarning, stacklevel=2)
    _run_auto_dense_check(
        config=auto_k_config,
        summary=summary,
        route=route,
        cache=cache,
        y=y,
        method=method,
        groups=groups,
        time=time,
        top_m=top_m,
        corr_prune=corr_prune,
    )
    summary["auto_routing"] = route
    return selected, selected_indices, auto_diag, summary


def _consensus_method_seed(random_state: int, method: str) -> int:
    """Derive deterministic, method-distinct consensus RNG streams."""
    entropy = [int(random_state) % (2**32), *[ord(char) for char in method.lower()]]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def _consensus_method_k(
    name: str,
    *,
    cache,
    y,
    method: str,
    objective: np.ndarray,
    config: AutoKConfig,
    top_m: int,
    corr_prune,
    groups,
    time,
    source_groups,
    source_time,
    path_length: int,
) -> tuple[int | None, str]:
    lower = name.lower()
    base = replace(
        config,
        min_k=min(int(config.min_k), int(path_length)),
        max_k=path_length,
    )
    if lower in {"ebic", "ric"}:
        cfg = replace(
            base,
            k_method="penalized_objective",
            objective_penalty=lower,
            min_k=0,
        )
        k_hat, _diag = _select_penalized_count(
            objective,
            cfg,
            objective_scale="n_eff",
            n_samples=len(cache.sample_weight),
            sample_weight=cache.sample_weight,
            n_candidates=len(cache.valid_cols),
            path_length=path_length,
        )
        return k_hat, ""
    if lower in {"posterior", "k_posterior"}:
        cfg = replace(base, k_method="k_posterior", min_k=0)
        k_hat, _diag = _select_posterior_count(
            objective,
            cfg,
            objective_scale="n_eff",
            n_samples=len(cache.sample_weight),
            sample_weight=cache.sample_weight,
            n_candidates=len(cache.valid_cols),
            path_length=path_length,
        )
        return k_hat, ""
    if lower == "chi2_stop":
        cfg = replace(base, k_method="chi2_stop", min_k=0)
        n_eff, _source = _objective_n_eff(cfg, cache.sample_weight, len(cache.sample_weight))
        p_candidates, panel_eigs = _gain_test_candidate_inputs(
            cache,
            y,
            path_length,
            top_m,
            corr_prune,
            method,
            cfg,
        )
        k_hat, _diag = select_k_chi2_stop(
            objective,
            cfg,
            n_eff=n_eff,
            p_candidates=p_candidates,
            panel_eigs=panel_eigs,
        )
        return min(k_hat, path_length), ""
    if lower == "forward_stop":
        cfg = replace(base, k_method="forward_stop", min_k=0)
        n_eff, _source = _objective_n_eff(cfg, cache.sample_weight, len(cache.sample_weight))
        p_candidates, panel_eigs = _gain_test_candidate_inputs(
            cache,
            y,
            path_length,
            top_m,
            corr_prune,
            method,
            cfg,
        )
        k_hat, _diag = select_k_forward_stop(
            objective,
            cfg,
            n_eff=n_eff,
            p_candidates=p_candidates,
            panel_eigs=panel_eigs,
        )
        return min(k_hat, path_length), ""
    if lower == "changepoint":
        cfg = replace(base, k_method="changepoint")
        n_eff, _source = _objective_n_eff(cfg, cache.sample_weight, len(cache.sample_weight))
        k_hat, _diag = select_k_changepoint(
            objective,
            cfg,
            objective_scale=n_eff,
            n_eff=n_eff,
            p_candidates=len(cache.valid_cols),
        )
        return min(k_hat, path_length), ""
    if lower == "perm_gap":
        cfg = replace(
            base,
            k_method="perm_gap",
            random_state=_consensus_method_seed(int(config.random_state), lower),
        )
        nulls = null_objective_paths(
            cache,
            y,
            B=int(cfg.perm_B),
            max_k=path_length,
            null=cfg.perm_null,
            time=source_time,
            groups=source_groups,
            top_m=top_m,
            corr_prune=corr_prune,
            random_state=int(cfg.random_state),
        )
        k_hat, _diag = select_k_perm_gap(objective, nulls, cfg)
        return min(k_hat, path_length), ""
    if lower in {"gaussian_cv", "xfit_objective"}:
        strategy = config.strategy
        if strategy == "time_holdout" and time is None:
            strategy = "kfold"
        if strategy == "group_cv" and groups is None:
            strategy = "kfold"
        cfg = replace(
            base,
            k_method=lower,
            strategy=strategy,
            random_state=_consensus_method_seed(int(config.random_state), lower),
        )
        if lower == "gaussian_cv":
            curves = gaussian_cv_curves(
                cache,
                y,
                config=cfg,
                groups=groups,
                time=time,
                top_m=top_m,
                corr_prune=corr_prune,
                method=method,
            )
            k_hat, _diag = select_k_gaussian_cv(curves, cfg)
        else:
            curves = xfit_objective_curves(
                cache,
                y,
                config=cfg,
                groups=groups,
                time=time,
                top_m=top_m,
                corr_prune=corr_prune,
                method=method,
            )
            k_hat, _diag = select_k_xfit_objective(curves, cfg)
        return min(k_hat, path_length), f"strategy={strategy}"
    if lower == "stability":
        cfg = replace(
            base,
            k_method="stability",
            random_state=_consensus_method_seed(int(config.random_state), lower),
        )
        boot = bootstrap_paths(
            cache,
            y,
            B=int(cfg.boot_B),
            max_k=path_length,
            boot_mode=cfg.boot_mode,
            top_m=top_m,
            corr_prune=corr_prune,
            random_state=int(cfg.random_state),
        )
        k_hat, _diag = select_k_stability(boot, len(cache.valid_cols), cfg)
        return min(k_hat, path_length), ""
    return None, "unknown_method"


def select_gaussian_consensus_path(
    *, cache, y: np.ndarray, method: str, max_k: int, top_m: int, auto_k_config: AutoKConfig,
    corr_prune: float | None | Literal["auto"] = "auto",
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    source_groups: Optional[np.ndarray] = None,
    source_time: Optional[np.ndarray] = None,
    verbose: bool = True,
    **_unused,
) -> GaussianAutoKResult:
    auto_k_module.validate_auto_k_config(auto_k_config)
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
    rows = []
    for name in auto_k_config.consensus_methods:
        start = time_module.perf_counter()
        note = ""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"AutoKConfig\..*does not use it\.",
                category=UserWarning,
            )
            try:
                k_hat, note = _consensus_method_k(
                    name,
                    cache=cache,
                    y=y,
                    method=method,
                    objective=objective,
                    config=auto_k_config,
                    top_m=top_m,
                    corr_prune=corr_prune,
                    groups=groups,
                    time=time,
                    source_groups=source_groups,
                    source_time=source_time,
                    path_length=len(path),
                )
            except Exception as exc:
                raise ValueError(f"consensus submethod {name!r} failed: {exc}") from exc
        if k_hat is None:
            raise ValueError(f"consensus submethod {name!r} did not return a k value: {note}")
        rows.append(
            {
                "method": name,
                "k_hat": int(k_hat),
                "runtime_s": float(time_module.perf_counter() - start),
                "note": note,
                "participated": True,
            }
        )
    auto_diag = pd.DataFrame(rows)
    participated = auto_diag[auto_diag["participated"]]
    if participated.empty:
        notes = ", ".join(
            f"{row.method}: {row.note or 'no result'}"
            for row in auto_diag.itertuples(index=False)
        )
        raise ValueError(
            "consensus auto-k did not get a result from any configured method; "
            f"methods={tuple(auto_k_config.consensus_methods)!r}; notes={notes}"
        )
    values = sorted(participated["k_hat"].astype(int).tolist())
    selected_count = int(values[(len(values) - 1) // 2])
    selected_count = min(selected_count, len(path))
    min_k = max(1, int(np.nanmin(participated["k_hat"].to_numpy(dtype=float))))
    spread = float(np.nanmax(participated["k_hat"].to_numpy(dtype=float)) / min_k)
    if np.isfinite(spread) and spread > 2.0:
        warnings.warn(
            "consensus auto-k methods disagree by more than 2x; k is ill-determined.",
            UserWarning,
            stacklevel=2,
        )
    auto_diag["selected"] = auto_diag["k_hat"] == selected_count
    _print_selected_k("Consensus", selected_count, verbose)
    effective_max_k = len(path)
    summary = auto_k_summary(
        auto_k_config,
        selected_k=selected_count,
        path_length=len(path),
        effective_max_k=effective_max_k,
        effective_min_k=_zero_capable_effective_min_k(
            auto_k_config,
            selected_k=selected_count,
            effective_max_k=effective_max_k,
        ),
        diagnostics=auto_diag,
        extra={
            "consensus_spread": spread,
            "consensus_n_methods": int(participated.shape[0]),
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
        warn_noise_floor=False,
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
        n_candidates=problem.n_features_input,
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
) -> BinarySelection:
    eval_X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=run.feature_names)
    eval_X = eval_X.iloc[run.row_idx]
    eval_y = problem.y01[run.row_idx]
    eval_groups = problem.groups[run.row_idx] if problem.groups is not None else None
    eval_time = problem.time[run.row_idx] if problem.time is not None else None

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
