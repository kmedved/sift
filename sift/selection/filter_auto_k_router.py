"""Auto-k router configuration helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sift.selection.auto_k import AutoKConfig


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
