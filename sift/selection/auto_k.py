"""Automatic k selection for filter methods."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from sklearn.model_selection import GroupKFold

from sift._preprocess import (
    LeaveOneOutLogitEncoder,
    ensure_weights,
    suppress_category_encoder_pandas_warnings,
)
from sift.selection.auto_k_core import (
    build_k_grid,
    build_score_curve_diagnostics,
    evaluate_numeric_prefixes,
    resolve_metric,
    split_weights,
    time_holdout_split,
)

if TYPE_CHECKING:
    from sift.estimators.copula import FeatureCache


@dataclass
class AutoKConfig:
    """Configuration for automatic k selection.

    ``auto_k_mode="prefix_only"`` is the current public behavior: build one
    supervised feature path, then evaluate prefixes of that fixed path. It is
    fast, but it is not an unbiased estimate of a nested selector procedure.
    ``auto_k_mode="nested"`` is implemented by sklearn-style selector classes,
    where each validation split fits its own train-only selector path. The
    function-style selectors still reject nested mode and keep this helper on
    the prefix-only contract.
    """

    k_method: Literal[
        "evaluate",
        "elbow",
        "penalized_objective",
        "chi2_stop",
        "forward_stop",
        "perm_gap",
        "knockoff_path",
        "xfit_objective",
        "gaussian_cv",
        "k_posterior",
        "stability",
        "changepoint",
        "consensus",
        "auto",
    ] = "evaluate"
    strategy: Literal["time_holdout", "group_cv", "kfold"] = "time_holdout"
    metric: Literal["rmse", "mae", "logloss", "error", "auto"] = "auto"
    max_k: int = 100
    min_k: int = 5
    val_frac: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    elbow_min_rel_gain: float = 0.02
    elbow_patience: int = 3
    auto_k_mode: Literal["prefix_only", "nested"] = "prefix_only"
    selection_rule: Literal["best", "one_se", "plateau", "tolerance"] = "best"
    one_se_multiplier: float = 1.0
    score_abs_tol: float | None = None
    score_rel_tol: float | None = None
    plateau_prefer: Literal["smallest", "center", "best", "largest"] = "smallest"
    plateau_min_points: int = 2
    objective_penalty: Literal["bic", "mdl", "aic", "hqc", "custom", "ebic", "ric"] = "bic"
    objective_penalty_weight: float | None = None
    objective_n_eff: float | None = None
    binary_objective_mode: Literal["refit", "score_test"] = "refit"
    n_eff_mode: Literal["auto", "kish", "weight_sum"] | float = "auto"
    alpha: float = 0.05
    m_mode: Literal["all", "panel", "li_ji"] = "all"
    stop_patience: int = 2
    perm_B: int = 20
    perm_null: Literal["auto", "permute", "circular_shift", "within_group"] = "auto"
    gap_rule: Literal["tibshirani", "argmax", "gain_envelope"] = "tibshirani"
    knockoff_q: float = 0.2
    knockoff_draws: int = 1
    knockoff_s_method: Literal["equi", "mvr", "me"] = "equi"
    knockoff_return: Literal["set", "prefix"] = "set"
    xfit_folds: int = 5
    xfit_mode: Literal["shared_z", "exact"] = "shared_z"
    xfit_ridge: float = 1e-3
    ebic_gamma: Literal["auto"] | float = "auto"
    posterior_level: float = 0.9
    posterior_pick: Literal["map", "smallest_in_hpd"] = "map"
    boot_B: int = 30
    boot_mode: Literal["bayes", "half"] = "bayes"
    stability_rule: Literal["max_one_se", "pi_threshold"] = "max_one_se"
    stability_pi: float = 0.6
    floor_z: float = 2.5
    floor_window: float | int = 0.2
    consensus_methods: tuple[str, ...] = ("ebic", "chi2_stop", "perm_gap", "gaussian_cv")
    auto_dense_check: bool = False
    auto_dense_min_k: int = 100
    auto_dense_min_frac: float = 0.25
    auto_dense_disagreement_ratio: float = 2.0


_VALID_K_METHODS = frozenset(
    {
        "evaluate",
        "elbow",
        "penalized_objective",
        "chi2_stop",
        "forward_stop",
        "perm_gap",
        "knockoff_path",
        "xfit_objective",
        "gaussian_cv",
        "k_posterior",
        "stability",
        "changepoint",
        "consensus",
        "auto",
    }
)
_VALID_STRATEGIES = frozenset({"time_holdout", "group_cv", "kfold"})
_VALID_SELECTION_RULES = frozenset({"best", "one_se", "plateau", "tolerance"})
_VALID_PLATEAU_PREFERS = frozenset({"smallest", "center", "best", "largest"})
_VALID_OBJECTIVE_PENALTIES = frozenset({"bic", "mdl", "aic", "hqc", "custom", "ebic", "ric"})
_VALID_BINARY_OBJECTIVE_MODES = frozenset({"refit", "score_test"})
_POSITIVE_INT_FIELDS = (
    "max_k",
    "n_splits",
    "elbow_patience",
    "plateau_min_points",
    "stop_patience",
    "perm_B",
    "knockoff_draws",
    "xfit_folds",
    "boot_B",
)
_NONNEGATIVE_INT_FIELDS = ("min_k",)
_VALID_N_EFF_MODES = frozenset({"auto", "kish", "weight_sum"})
_VALID_M_MODES = frozenset({"all", "panel", "li_ji"})
_VALID_PERM_NULLS = frozenset({"auto", "permute", "circular_shift", "within_group"})
_VALID_GAP_RULES = frozenset({"tibshirani", "argmax", "gain_envelope"})
_VALID_KNOCKOFF_S_METHODS = frozenset({"equi", "mvr", "me"})
_VALID_KNOCKOFF_RETURNS = frozenset({"set", "prefix"})
_VALID_XFIT_MODES = frozenset({"shared_z", "exact"})
_VALID_POSTERIOR_PICKS = frozenset({"map", "smallest_in_hpd"})
_VALID_BOOT_MODES = frozenset({"bayes", "half"})
_VALID_STABILITY_RULES = frozenset({"max_one_se", "pi_threshold"})
_VALID_CONSENSUS_METHODS = frozenset(
    {
        "ebic",
        "ric",
        "posterior",
        "k_posterior",
        "chi2_stop",
        "forward_stop",
        "changepoint",
        "perm_gap",
        "gaussian_cv",
        "xfit_objective",
        "stability",
    }
)
_DEFAULT_AUTOK_CONFIG = None
_REAL_TYPES = (int, float, np.integer, np.floating)


def _is_real_number(value) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, _REAL_TYPES)


def validate_auto_k_config(config: AutoKConfig) -> None:
    """Validate runtime values on an AutoKConfig instance."""
    if config.k_method not in _VALID_K_METHODS:
        raise ValueError(
            "AutoKConfig.k_method must be one of "
            f"{sorted(_VALID_K_METHODS)}; got {config.k_method!r}"
        )

    if config.strategy not in _VALID_STRATEGIES:
        raise ValueError(
            "AutoKConfig.strategy must be one of "
            f"{sorted(_VALID_STRATEGIES)}; got {config.strategy!r}"
        )
    if config.k_method == "evaluate" and config.strategy == "kfold":
        raise ValueError(
            "AutoKConfig.strategy='kfold' is only supported by gaussian_cv and "
            "xfit_objective; use time_holdout or group_cv for k_method='evaluate'"
        )

    for name in _POSITIVE_INT_FIELDS:
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"AutoKConfig.{name} must be a positive integer")
    for name in _NONNEGATIVE_INT_FIELDS:
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"AutoKConfig.{name} must be a non-negative integer")

    if int(config.min_k) > int(config.max_k):
        raise ValueError("AutoKConfig.min_k must be <= AutoKConfig.max_k")
    if not isinstance(config.auto_dense_check, (bool, np.bool_)):
        raise ValueError("AutoKConfig.auto_dense_check must be boolean")
    if (
        isinstance(config.auto_dense_min_k, (bool, np.bool_))
        or not isinstance(config.auto_dense_min_k, (int, np.integer))
        or int(config.auto_dense_min_k) < 0
    ):
        raise ValueError("AutoKConfig.auto_dense_min_k must be a non-negative integer")

    if (
        not _is_real_number(config.val_frac)
        or not np.isfinite(config.val_frac)
        or not 0.0 < float(config.val_frac) < 1.0
    ):
        raise ValueError("AutoKConfig.val_frac must be finite and between 0 and 1")

    if (
        not _is_real_number(config.elbow_min_rel_gain)
        or not np.isfinite(config.elbow_min_rel_gain)
        or float(config.elbow_min_rel_gain) < 0.0
    ):
        raise ValueError("AutoKConfig.elbow_min_rel_gain must be finite and non-negative")

    if config.selection_rule not in _VALID_SELECTION_RULES:
        raise ValueError(
            "AutoKConfig.selection_rule must be one of "
            f"{sorted(_VALID_SELECTION_RULES)}; got {config.selection_rule!r}"
        )
    if config.plateau_prefer not in _VALID_PLATEAU_PREFERS:
        raise ValueError(
            "AutoKConfig.plateau_prefer must be one of "
            f"{sorted(_VALID_PLATEAU_PREFERS)}; got {config.plateau_prefer!r}"
        )
    if (
        not _is_real_number(config.one_se_multiplier)
        or not np.isfinite(config.one_se_multiplier)
        or float(config.one_se_multiplier) <= 0.0
    ):
        raise ValueError("AutoKConfig.one_se_multiplier must be positive and finite")
    for name in ("score_abs_tol", "score_rel_tol"):
        value = getattr(config, name)
        if value is not None and (
            not _is_real_number(value) or not np.isfinite(value) or float(value) < 0.0
        ):
            raise ValueError(f"AutoKConfig.{name} must be None or finite and non-negative")
    if (
        config.k_method == "evaluate"
        and config.selection_rule in {"plateau", "tolerance"}
        and config.score_abs_tol is None
        and config.score_rel_tol is None
    ):
        raise ValueError(
            "selection_rule='plateau' or 'tolerance' requires score_abs_tol or score_rel_tol"
        )

    if config.n_eff_mode not in _VALID_N_EFF_MODES and (
        not _is_real_number(config.n_eff_mode)
        or not np.isfinite(config.n_eff_mode)
        or float(config.n_eff_mode) <= 1.0
    ):
        raise ValueError(
            "AutoKConfig.n_eff_mode must be 'auto', 'kish', 'weight_sum', or a finite float > 1"
        )

    if config.m_mode not in _VALID_M_MODES:
        raise ValueError(
            "AutoKConfig.m_mode must be one of "
            f"{sorted(_VALID_M_MODES)}; got {config.m_mode!r}"
        )
    if config.perm_null not in _VALID_PERM_NULLS:
        raise ValueError(
            "AutoKConfig.perm_null must be one of "
            f"{sorted(_VALID_PERM_NULLS)}; got {config.perm_null!r}"
        )
    if config.gap_rule not in _VALID_GAP_RULES:
        raise ValueError(
            "AutoKConfig.gap_rule must be one of "
            f"{sorted(_VALID_GAP_RULES)}; got {config.gap_rule!r}"
        )
    if config.knockoff_s_method not in _VALID_KNOCKOFF_S_METHODS:
        raise ValueError(
            "AutoKConfig.knockoff_s_method must be one of "
            f"{sorted(_VALID_KNOCKOFF_S_METHODS)}; got {config.knockoff_s_method!r}"
        )
    if config.knockoff_return not in _VALID_KNOCKOFF_RETURNS:
        raise ValueError(
            "AutoKConfig.knockoff_return must be one of "
            f"{sorted(_VALID_KNOCKOFF_RETURNS)}; got {config.knockoff_return!r}"
        )
    if config.xfit_mode not in _VALID_XFIT_MODES:
        raise ValueError(
            "AutoKConfig.xfit_mode must be one of "
            f"{sorted(_VALID_XFIT_MODES)}; got {config.xfit_mode!r}"
        )
    if config.posterior_pick not in _VALID_POSTERIOR_PICKS:
        raise ValueError(
            "AutoKConfig.posterior_pick must be one of "
            f"{sorted(_VALID_POSTERIOR_PICKS)}; got {config.posterior_pick!r}"
        )
    if config.boot_mode not in _VALID_BOOT_MODES:
        raise ValueError(
            "AutoKConfig.boot_mode must be one of "
            f"{sorted(_VALID_BOOT_MODES)}; got {config.boot_mode!r}"
        )
    if config.stability_rule not in _VALID_STABILITY_RULES:
        raise ValueError(
            "AutoKConfig.stability_rule must be one of "
            f"{sorted(_VALID_STABILITY_RULES)}; got {config.stability_rule!r}"
        )

    for name in ("alpha", "knockoff_q", "posterior_level"):
        value = getattr(config, name)
        if (
            not _is_real_number(value)
            or not np.isfinite(value)
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(f"AutoKConfig.{name} must be finite and between 0 and 1")
    if (
        not _is_real_number(config.stability_pi)
        or not np.isfinite(config.stability_pi)
        or not 0.5 < float(config.stability_pi) <= 1.0
    ):
        raise ValueError("AutoKConfig.stability_pi must be finite and in (0.5, 1]")
    if (
        not _is_real_number(config.xfit_ridge)
        or not np.isfinite(config.xfit_ridge)
        or float(config.xfit_ridge) < 0.0
    ):
        raise ValueError("AutoKConfig.xfit_ridge must be finite and non-negative")
    if (
        not _is_real_number(config.floor_z)
        or not np.isfinite(config.floor_z)
        or float(config.floor_z) <= 0.0
    ):
        raise ValueError("AutoKConfig.floor_z must be positive and finite")
    if not _is_real_number(config.floor_window) or not np.isfinite(config.floor_window):
        raise ValueError("AutoKConfig.floor_window must be finite")
    if isinstance(config.floor_window, (int, np.integer)):
        if int(config.floor_window) < 5:
            raise ValueError("AutoKConfig.floor_window as an integer must be >= 5")
    elif not 0.0 < float(config.floor_window) <= 0.5:
        raise ValueError("AutoKConfig.floor_window as a fraction must be in (0, 0.5]")
    if config.ebic_gamma != "auto" and (
        not _is_real_number(config.ebic_gamma)
        or not np.isfinite(config.ebic_gamma)
        or not 0.0 <= float(config.ebic_gamma) <= 1.0
    ):
        raise ValueError("AutoKConfig.ebic_gamma must be 'auto' or finite in [0, 1]")
    if not isinstance(config.consensus_methods, tuple) or not config.consensus_methods:
        raise ValueError("AutoKConfig.consensus_methods must be a non-empty tuple")
    if not all(isinstance(method, str) and method for method in config.consensus_methods):
        raise ValueError("AutoKConfig.consensus_methods must contain non-empty strings")
    unknown_consensus = [
        method
        for method in config.consensus_methods
        if method.lower() not in _VALID_CONSENSUS_METHODS
    ]
    if unknown_consensus:
        raise ValueError(
            "AutoKConfig.consensus_methods contains unsupported method(s): "
            f"{unknown_consensus}; supported methods are {sorted(_VALID_CONSENSUS_METHODS)}"
        )

    if config.objective_penalty not in _VALID_OBJECTIVE_PENALTIES:
        raise ValueError(
            "AutoKConfig.objective_penalty must be one of "
            f"{sorted(_VALID_OBJECTIVE_PENALTIES)}; got {config.objective_penalty!r}"
        )
    if config.objective_penalty == "custom":
        if config.objective_penalty_weight is None:
            raise ValueError(
                "AutoKConfig.objective_penalty_weight is required when "
                "objective_penalty='custom'"
            )
        if (
            not _is_real_number(config.objective_penalty_weight)
            or not np.isfinite(config.objective_penalty_weight)
            or float(config.objective_penalty_weight) < 0.0
        ):
            raise ValueError(
                "AutoKConfig.objective_penalty_weight must be finite and non-negative"
            )
    elif config.objective_penalty_weight is not None:
        raise ValueError(
            "AutoKConfig.objective_penalty_weight is only valid when "
            "objective_penalty='custom'"
        )

    if config.objective_n_eff is not None and (
        not _is_real_number(config.objective_n_eff)
        or not np.isfinite(config.objective_n_eff)
        or float(config.objective_n_eff) <= 1.0
    ):
        raise ValueError("AutoKConfig.objective_n_eff must be None or finite and > 1")
    if (
        not _is_real_number(config.auto_dense_min_frac)
        or not np.isfinite(config.auto_dense_min_frac)
        or not 0.0 <= float(config.auto_dense_min_frac) <= 1.0
    ):
        raise ValueError("AutoKConfig.auto_dense_min_frac must be finite and between 0 and 1")
    if (
        not _is_real_number(config.auto_dense_disagreement_ratio)
        or not np.isfinite(config.auto_dense_disagreement_ratio)
        or float(config.auto_dense_disagreement_ratio) <= 1.0
    ):
        raise ValueError("AutoKConfig.auto_dense_disagreement_ratio must be finite and > 1")
    if config.objective_penalty == "hqc" and (
        config.objective_n_eff is not None and float(config.objective_n_eff) <= np.e
    ):
        raise ValueError("AutoKConfig.objective_n_eff must be > e for HQC")

    if config.binary_objective_mode not in _VALID_BINARY_OBJECTIVE_MODES:
        raise ValueError(
            "AutoKConfig.binary_objective_mode must be one of "
            f"{sorted(_VALID_BINARY_OBJECTIVE_MODES)}; got {config.binary_objective_mode!r}"
        )

    _warn_unused_method_fields(config)


def _warn_unused_method_fields(config: AutoKConfig) -> None:
    if config.k_method == "auto":
        return
    global _DEFAULT_AUTOK_CONFIG
    if _DEFAULT_AUTOK_CONFIG is None:
        _DEFAULT_AUTOK_CONFIG = AutoKConfig()
    defaults = _DEFAULT_AUTOK_CONFIG
    used_by = {
        "alpha": {"chi2_stop", "forward_stop", "perm_gap"},
        "m_mode": {"chi2_stop", "forward_stop"},
        "stop_patience": {"chi2_stop", "changepoint", "perm_gap"},
        "perm_B": {"perm_gap"},
        "perm_null": {"perm_gap"},
        "gap_rule": {"perm_gap"},
        "knockoff_q": {"knockoff_path"},
        "knockoff_draws": {"knockoff_path"},
        "knockoff_s_method": {"knockoff_path"},
        "knockoff_return": {"knockoff_path"},
        "xfit_folds": {"xfit_objective", "gaussian_cv"},
        "xfit_mode": {"xfit_objective", "gaussian_cv"},
        "xfit_ridge": {"gaussian_cv"},
        "ebic_gamma": {"penalized_objective", "k_posterior"},
        "posterior_level": {"k_posterior"},
        "posterior_pick": {"k_posterior"},
        "boot_B": {"stability"},
        "boot_mode": {"stability"},
        "stability_rule": {"stability"},
        "stability_pi": {"stability"},
        "floor_z": {"changepoint"},
        "floor_window": {"changepoint"},
        "consensus_methods": {"consensus"},
        "auto_dense_check": {"auto"},
        "auto_dense_min_k": {"auto"},
        "auto_dense_min_frac": {"auto"},
        "auto_dense_disagreement_ratio": {"auto"},
    }
    consensus_methods = None
    if config.k_method == "consensus":
        consensus_aliases = {
            "ebic": "penalized_objective",
            "ric": "penalized_objective",
            "posterior": "k_posterior",
        }
        consensus_methods = {
            consensus_aliases.get(method.lower(), method.lower())
            for method in config.consensus_methods
        }
        consensus_methods.add("consensus")
    for field_name, methods in used_by.items():
        if config.k_method in methods:
            continue
        if consensus_methods is not None and bool(consensus_methods & methods):
            continue
        if getattr(config, field_name) != getattr(defaults, field_name):
            warnings.warn(
                f"AutoKConfig.{field_name} is set but k_method={config.k_method!r} "
                "does not use it.",
                UserWarning,
                stacklevel=3,
            )


def _ensure_supported_auto_k_mode(
    config: AutoKConfig,
    *,
    allow_nested: bool = False,
) -> None:
    """Validate path-selection semantics for the current implementation."""
    validate_auto_k_config(config)
    if config.auto_k_mode == "prefix_only":
        return
    if config.auto_k_mode == "nested":
        if allow_nested:
            return
        raise NotImplementedError(
            "AutoKConfig(auto_k_mode='nested') is not implemented yet. "
            "Use auto_k_mode='prefix_only' for the current behavior: build one "
            "supervised feature path on the rows available to the selector, "
            "then evaluate prefixes. This is fast but is not an unbiased "
            "estimate of the full nested selector-plus-k-selection procedure."
        )
    raise ValueError(
        "auto_k_mode must be 'prefix_only' or 'nested'; "
        f"got {config.auto_k_mode!r}"
    )


def with_effective_k_bounds(config: AutoKConfig, *, min_k: int, max_k: int) -> AutoKConfig:
    """Return a config copy with k bounds clamped to an actual feature path."""
    return replace(config, min_k=int(min_k), max_k=int(max_k))


def resolve_auto_k_config(
    auto_k_config: Optional[AutoKConfig],
    time: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    *,
    allow_nested: bool = False,
) -> AutoKConfig:
    """Resolve auto-k config, inferring strategy from supplied split context."""
    if auto_k_config is not None:
        _ensure_supported_auto_k_mode(auto_k_config, allow_nested=allow_nested)
        return auto_k_config
    if time is not None:
        config = AutoKConfig(strategy="time_holdout")
        _ensure_supported_auto_k_mode(config, allow_nested=allow_nested)
        return config
    if groups is not None:
        config = AutoKConfig(strategy="group_cv")
        _ensure_supported_auto_k_mode(config, allow_nested=allow_nested)
        return config
    raise ValueError(
        "k='auto' requires time, groups, or auto_k_config with an explicit "
        "AutoKConfig for a non-evaluate k_method such as 'elbow', "
        "'penalized_objective', 'gaussian_cv', or 'perm_gap'"
    )


def _score_curve_tolerance(best_score: float, config: AutoKConfig) -> float:
    tol = 0.0
    if config.score_abs_tol is not None:
        tol = max(tol, float(config.score_abs_tol))
    if config.score_rel_tol is not None:
        tol = max(tol, abs(best_score) * float(config.score_rel_tol))
    return tol


def _choose_best_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    del best_row, best_score, config, lower_is_better
    diag["within_tolerance"] = diag["k"] == best_k
    return best_k, "best", False


def _choose_one_se_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    best_se = float(best_row.get("score_se", np.nan))
    if not np.isfinite(best_se):
        warnings.warn(
            "selection_rule='one_se' requires at least two finite split scores; "
            "falling back to selection_rule='best'.",
            UserWarning,
            stacklevel=3,
        )
        diag["within_tolerance"] = diag["k"] == best_k
        return best_k, "best", True

    tol = float(config.one_se_multiplier) * best_se
    if lower_is_better:
        diag["within_tolerance"] = diag["score_mean"] <= best_score + tol
    else:
        diag["within_tolerance"] = diag["score_mean"] >= best_score - tol
    eligible = diag[diag["within_tolerance"] & np.isfinite(diag["score_mean"])]
    selected_k = int(eligible.sort_values("k", kind="mergesort").iloc[0]["k"])
    return selected_k, "one_se", False


def _mark_tolerance(
    diag: pd.DataFrame,
    best_score: float,
    config: AutoKConfig,
    *,
    lower_is_better: bool,
) -> None:
    tol = _score_curve_tolerance(best_score, config)
    if lower_is_better:
        diag["within_tolerance"] = diag["score_mean"] <= best_score + tol
    else:
        diag["within_tolerance"] = diag["score_mean"] >= best_score - tol
    diag.loc[~np.isfinite(diag["score_mean"]), "within_tolerance"] = False


def _choose_tolerance_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    del best_row, best_k
    _mark_tolerance(diag, best_score, config, lower_is_better=lower_is_better)
    eligible = diag[diag["within_tolerance"]]
    selected_k = int(eligible.sort_values("k", kind="mergesort").iloc[0]["k"])
    return selected_k, "tolerance", False


def _selected_plateau_ks(diag: pd.DataFrame, best_k: int) -> list[int]:
    eligible_mask = diag["within_tolerance"].to_numpy(dtype=bool)
    best_positions = np.flatnonzero(diag["k"].to_numpy(dtype=int) == best_k)
    if not best_positions.size:
        return [best_k]
    pos = int(best_positions[0])
    start = pos
    while start > 0 and eligible_mask[start - 1]:
        start -= 1
    end = pos
    while end + 1 < len(eligible_mask) and eligible_mask[end + 1]:
        end += 1
    diag.iloc[start : end + 1, diag.columns.get_loc("in_selected_plateau")] = True
    return diag.iloc[start : end + 1]["k"].astype(int).tolist()


def _choose_plateau_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    del best_row
    _mark_tolerance(diag, best_score, config, lower_is_better=lower_is_better)
    plateau_ks = _selected_plateau_ks(diag, best_k)
    if len(plateau_ks) < int(config.plateau_min_points):
        selected_k = best_k
    elif config.plateau_prefer == "smallest":
        selected_k = int(plateau_ks[0])
    elif config.plateau_prefer == "largest":
        selected_k = int(plateau_ks[-1])
    elif config.plateau_prefer == "center":
        selected_k = int(plateau_ks[len(plateau_ks) // 2])
    else:
        selected_k = best_k
    return selected_k, "plateau", False


_RULE_SELECTORS = {
    "best": _choose_best_rule,
    "one_se": _choose_one_se_rule,
    "tolerance": _choose_tolerance_rule,
    "plateau": _choose_plateau_rule,
}


def choose_k_from_score_curve(
    diagnostics: pd.DataFrame,
    config: AutoKConfig,
    *,
    lower_is_better: bool = True,
) -> Tuple[int, pd.DataFrame]:
    """Choose k from an evaluated score curve according to AutoKConfig."""
    validate_auto_k_config(config)
    diag = diagnostics.copy()
    if "k" not in diag.columns:
        raise ValueError("score-curve diagnostics must include a 'k' column")
    diag["k"] = diag["k"].astype(int)
    diag = diag[
        (diag["k"] >= int(config.min_k)) & (diag["k"] <= int(config.max_k))
    ].copy()
    diag = diag.sort_values("k", kind="mergesort").reset_index(drop=True)
    if diag.empty:
        return 0, diag
    if "score_mean" not in diag.columns:
        diag["score_mean"] = diag["score"]
    diag["score"] = diag["score_mean"]

    finite = diag[np.isfinite(diag["score_mean"])].copy()
    fallback_k = int(diag["k"].max())
    if finite.empty:
        diag["best_k"] = fallback_k
        diag["best_score"] = np.inf if lower_is_better else -np.inf
        diag["within_tolerance"] = False
        diag["in_selected_plateau"] = False
        diag["selected"] = diag["k"] == fallback_k
        diag["selection_rule"] = config.selection_rule
        diag["selection_rule_effective"] = config.selection_rule
        diag["one_se_unavailable"] = config.selection_rule == "one_se"
        return fallback_k, diag

    ascending = [lower_is_better, True]
    best_rows = finite.sort_values(["score_mean", "k"], ascending=ascending, kind="mergesort")
    best_row = best_rows.iloc[0]
    best_k = int(best_row["k"])
    best_score = float(best_row["score_mean"])
    rule = config.selection_rule
    effective_rule = rule
    one_se_unavailable = False

    diag["best_k"] = best_k
    diag["best_score"] = best_score
    diag["within_tolerance"] = False
    diag["in_selected_plateau"] = False
    diag["selection_rule"] = rule

    selector = _RULE_SELECTORS.get(rule)
    if selector is None:
        raise ValueError(f"Unknown selection_rule: {rule!r}")
    selected_k, effective_rule, one_se_unavailable = selector(
        diag,
        best_row,
        best_k,
        best_score,
        config,
        lower_is_better=lower_is_better,
    )

    diag["selection_rule_effective"] = effective_rule
    diag["one_se_unavailable"] = one_se_unavailable
    diag["selected"] = diag["k"] == selected_k
    return int(selected_k), diag


def _evaluate_prefix_split(
    *,
    X_path_df: pd.DataFrame,
    valid_features: List[str],
    y_arr: np.ndarray,
    w_arr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task: Literal["regression", "classification"],
    metric: str,
    k_grid: list[int],
    cat_features: Optional[List[str]],
    cat_encoding: Literal["none", "target", "loo", "james_stein", "loo_logit"],
    loo_smoothing: float,
    loo_clip_min: float,
    loo_clip_max: float,
) -> dict:
    """Evaluate all k values for one train/validation split."""
    Xtr_df = X_path_df.iloc[train_idx]
    Xva_df = X_path_df.iloc[val_idx]
    ytr = y_arr[train_idx]
    yva = y_arr[val_idx]
    wtr = split_weights(w_arr, train_idx, "train")
    wva = split_weights(w_arr, val_idx, "validation")

    if cat_features is None:
        fold_cat = (
            Xtr_df.select_dtypes(include=["object", "category", "string"])
            .columns.intersection(valid_features)
            .tolist()
        )
    else:
        fold_cat = [col for col in cat_features if col in Xtr_df.columns]

    if cat_encoding == "loo_logit" and fold_cat:
        if task != "classification":
            raise ValueError("cat_encoding='loo_logit' requires task='classification'")
        enc = LeaveOneOutLogitEncoder(
            cols=fold_cat,
            smoothing=loo_smoothing,
            clip_min=loo_clip_min,
            clip_max=loo_clip_max,
        )
        Xtr_df = enc.fit_transform(Xtr_df, ytr, sample_weight=wtr)
        Xva_df = enc.transform(Xva_df)
    elif cat_encoding != "none" and fold_cat:
        if importlib.util.find_spec("category_encoders") is None:
            raise ImportError(
                "cat_encoding requires category_encoders. Install with: pip install category_encoders"
            )
        import category_encoders as ce

        enc_map = {
            "loo": ce.LeaveOneOutEncoder,
            "target": ce.TargetEncoder,
            "james_stein": ce.JamesSteinEncoder,
        }
        Encoder = enc_map[cat_encoding]
        try:
            enc = Encoder(
                cols=fold_cat,
                handle_missing="return_nan",
                handle_unknown="value",
            )
        except TypeError:
            enc = Encoder(cols=fold_cat, handle_missing="return_nan")
        with suppress_category_encoder_pandas_warnings():
            Xtr_df = enc.fit_transform(Xtr_df, ytr)
            Xva_df = enc.transform(Xva_df)

    return evaluate_numeric_prefixes(
        Xtr_df,
        Xva_df,
        ytr,
        yva,
        wtr,
        wva,
        task=task,
        metric=metric,
        k_grid=k_grid,
        ridge_alpha_strategy="full_path",
    )


def select_k_auto(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_path: List[str],
    config: AutoKConfig,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    task: Literal["regression", "classification"] = "regression",
    cat_encoding: Literal["none", "target", "loo", "james_stein", "loo_logit"] = "none",
    cat_features: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
) -> Tuple[int, List[str], pd.DataFrame]:
    """Select optimal k by evaluating prefixes of feature_path."""
    _ensure_supported_auto_k_mode(config)
    if config.k_method != "evaluate":
        raise ValueError(
            "select_k_auto supports only AutoKConfig(k_method='evaluate'). "
            "Use select_k_elbow(...) or a selector path that explicitly supports "
            "objective-path auto-k."
        )

    if not feature_path:
        return 0, [], pd.DataFrame()
    if isinstance(X, pd.DataFrame) and not X.columns.is_unique:
        duplicates = pd.Index(X.columns[X.columns.duplicated()]).unique().astype(str).tolist()
        sample = duplicates[:5]
        suffix = "..." if len(duplicates) > 5 else ""
        raise ValueError(
            "select_k_auto requires unique DataFrame column labels because "
            "feature_path entries are name-based. "
            f"Duplicate labels: {sample}{suffix}"
        )

    y_arr = np.asarray(y).ravel()
    w_arr = ensure_weights(sample_weight, len(y_arr), normalize=True)
    max_k = min(config.max_k, len(feature_path))
    min_k = max(1, min(config.min_k, max_k))

    valid_features = [f for f in feature_path if f in X.columns]
    if not valid_features:
        return 0, [], pd.DataFrame()

    max_k = min(max_k, len(valid_features))
    min_k = max(1, min(config.min_k, max_k))
    valid_features = valid_features[:max_k]
    k_grid = build_k_grid(min_k, max_k)

    X_path_df = X[valid_features]

    metric = resolve_metric(config.metric, task)
    eval_kwargs = {
        "X_path_df": X_path_df,
        "valid_features": valid_features,
        "y_arr": y_arr,
        "w_arr": w_arr,
        "task": task,
        "metric": metric,
        "k_grid": k_grid,
        "cat_features": cat_features,
        "cat_encoding": cat_encoding,
        "loo_smoothing": loo_smoothing,
        "loo_clip_min": loo_clip_min,
        "loo_clip_max": loo_clip_max,
    }

    if config.strategy == "time_holdout":
        if time is None:
            raise ValueError("time_holdout strategy requires time parameter")

        train_idx, val_idx = time_holdout_split(time, config.val_frac)
        scores = _evaluate_prefix_split(
            train_idx=train_idx,
            val_idx=val_idx,
            **eval_kwargs,
        )
        split_scores = {k: [score] for k, score in scores.items()}
        diag = build_score_curve_diagnostics(k_grid, split_scores)

    elif config.strategy == "group_cv":
        if groups is None:
            raise ValueError("group_cv strategy requires groups parameter")

        n_unique = len(np.unique(groups))
        n_splits = min(config.n_splits, n_unique)
        if n_splits < 2:
            raise ValueError(f"group_cv requires at least 2 groups, got {n_unique}")

        gkf = GroupKFold(n_splits=n_splits)

        all_scores = {k: [] for k in k_grid}
        for train_idx, val_idx in gkf.split(X_path_df, y_arr, groups):
            fold_scores = _evaluate_prefix_split(
                train_idx=train_idx,
                val_idx=val_idx,
                **eval_kwargs,
            )
            for k, score in fold_scores.items():
                all_scores[k].append(score)

        diag = build_score_curve_diagnostics(k_grid, all_scores)

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    if diag.empty:
        return max_k, valid_features[:max_k], diag

    curve_config = with_effective_k_bounds(config, min_k=min_k, max_k=max_k)
    best_k, diag = choose_k_from_score_curve(diag, curve_config, lower_is_better=True)

    return best_k, valid_features[:best_k], diag


def select_k_elbow(
    objective_path: np.ndarray,
    min_k: int = 5,
    max_k: int = 100,
    min_rel_gain: float = 0.02,
    patience: int = 3,
) -> Tuple[int, pd.DataFrame]:
    """Select k via elbow detection on an objective path."""
    obj = np.asarray(objective_path).ravel()
    max_k = min(max_k, len(obj))

    if max_k <= 0:
        return 0, pd.DataFrame()

    delta = np.zeros_like(obj)
    delta[0] = obj[0]
    delta[1:] = obj[1:] - obj[:-1]

    rel_gain = np.zeros_like(obj)
    rel_gain[0] = np.inf
    denom = np.maximum(np.abs(obj[:-1]), 1.0)
    rel_gain[1:] = delta[1:] / denom

    best_k = max_k
    run = 0

    for k in range(max(min_k, 2), max_k + 1):
        if rel_gain[k - 1] < min_rel_gain:
            run += 1
            if run >= patience:
                best_k = k - patience + 1
                break
        else:
            run = 0

    diag = pd.DataFrame(
        {
            "k": np.arange(1, max_k + 1),
            "objective": obj[:max_k],
            "delta": delta[:max_k],
            "rel_gain": rel_gain[:max_k],
        }
    )

    return best_k, diag


def _resolve_n_eff_mode(config: AutoKConfig) -> str | float:
    mode = config.n_eff_mode
    if mode == "auto":
        v2_methods = {
            "chi2_stop",
            "forward_stop",
            "perm_gap",
            "knockoff_path",
            "xfit_objective",
            "gaussian_cv",
            "k_posterior",
            "stability",
            "changepoint",
            "consensus",
            "auto",
        }
        if config.k_method in v2_methods or config.objective_penalty in {"ebic", "ric"}:
            return "kish"
        return "weight_sum"
    return mode


def _penalty_weight(config: AutoKConfig, n_eff: float) -> float:
    if config.objective_penalty in {"bic", "mdl", "ebic"}:
        return float(np.log(n_eff))
    if config.objective_penalty == "aic":
        return 2.0
    if config.objective_penalty == "hqc":
        if n_eff <= np.e:
            raise ValueError("n_eff must be > e for objective_penalty='hqc'")
        return float(2.0 * np.log(np.log(n_eff)))
    if config.objective_penalty == "custom":
        assert config.objective_penalty_weight is not None
        return float(config.objective_penalty_weight)
    if config.objective_penalty == "ric":
        return 0.0
    raise ValueError(f"Unknown objective_penalty: {config.objective_penalty!r}")


def _log_comb(n: int, k: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k, dtype=np.float64)
    out = gammaln(float(n) + 1.0) - gammaln(k_arr + 1.0) - gammaln(float(n) - k_arr + 1.0)
    out[(k_arr < 0) | (k_arr > n)] = np.inf
    return out


def _resolve_ebic_gamma(config: AutoKConfig, *, n_eff: float, n_candidates: int) -> float:
    if config.ebic_gamma == "auto":
        if n_candidates <= 1:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - np.log(n_eff) / (2.0 * np.log(n_candidates)))))
    return float(config.ebic_gamma)


def _penalty_array(
    config: AutoKConfig,
    ks: np.ndarray,
    *,
    n_eff: float,
    n_candidates: int | None,
) -> tuple[np.ndarray, float, float | None, int | None]:
    penalty_kind = config.objective_penalty
    if penalty_kind in {"ebic", "ric"}:
        if n_candidates is None:
            raise ValueError("n_candidates is required for EBIC/RIC objective penalties")
        n_candidates_int = int(n_candidates)
        if n_candidates_int < 1:
            raise ValueError("n_candidates must be a positive integer")
        if np.max(ks, initial=0) > n_candidates_int:
            raise ValueError("n_candidates must be >= the largest evaluated k")
    else:
        n_candidates_int = None

    if penalty_kind == "ebic":
        gamma = _resolve_ebic_gamma(config, n_eff=n_eff, n_candidates=n_candidates_int)
        penalty = ks.astype(np.float64) * np.log(n_eff) + 2.0 * gamma * _log_comb(n_candidates_int, ks)
        return penalty, float(np.log(n_eff)), gamma, n_candidates_int
    if penalty_kind == "ric":
        gamma = None
        penalty = 2.0 * ks.astype(np.float64) * np.log(float(n_candidates_int))
        return penalty, 2.0 * float(np.log(float(n_candidates_int))), gamma, n_candidates_int

    penalty_weight = _penalty_weight(config, n_eff)
    return penalty_weight * ks.astype(np.float64), penalty_weight, None, n_candidates_int


def _objective_weight_diagnostics(
    sample_weight: Optional[np.ndarray],
    n_samples: int,
    config: AutoKConfig,
) -> tuple[np.ndarray, float, float, float, str]:
    w = ensure_weights(sample_weight, n_samples, normalize=True)
    weight_sum = float(np.sum(w))
    sum_sq = float(np.sum(w * w))
    kish_n_eff = float(weight_sum * weight_sum / sum_sq) if sum_sq > 0.0 else float("nan")
    if config.objective_n_eff is not None:
        n_eff = float(config.objective_n_eff)
        n_eff_source = "objective_n_eff"
    else:
        mode = _resolve_n_eff_mode(config)
        if mode == "kish":
            n_eff = kish_n_eff
            n_eff_source = "kish"
        elif mode == "weight_sum":
            n_eff = weight_sum
            n_eff_source = "selector_weight_sum"
        else:
            n_eff = float(mode)
            n_eff_source = "n_eff_mode"
    if n_eff <= 1.0 or not np.isfinite(n_eff):
        raise ValueError("objective effective sample size must be finite and > 1")
    if config.objective_penalty == "hqc" and n_eff <= np.e:
        raise ValueError("n_eff must be > e for objective_penalty='hqc'")
    return w, weight_sum, kish_n_eff, n_eff, n_eff_source


def select_k_penalized_objective(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale: float | Literal["n_eff"],
    n_samples: int,
    sample_weight: Optional[np.ndarray] = None,
    n_candidates: int | None = None,
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
    df_path: Optional[np.ndarray] = None,
) -> Tuple[int, pd.DataFrame]:
    """Select k by maximizing a penalized CEFS+ proxy objective path."""
    validate_auto_k_config(config)
    if config.k_method != "penalized_objective":
        raise ValueError(
            "select_k_penalized_objective requires "
            "AutoKConfig(k_method='penalized_objective')"
        )

    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)
    path_length = int(len(obj))
    effective_max_k = min(int(max_k if max_k is not None else config.max_k), path_length)
    if effective_max_k <= 0:
        return 0, pd.DataFrame()
    min_k_raw = int(min_k if min_k is not None else config.min_k)
    min_k_eff = max(0, min(min_k_raw, effective_max_k))

    _, weight_sum, kish_n_eff, n_eff, n_eff_source = _objective_weight_diagnostics(
        sample_weight,
        int(n_samples),
        config,
    )
    if objective_scale == "n_eff":
        scale_value = n_eff
        scale_label = "n_eff"
    else:
        scale_value = float(objective_scale)
        scale_label = str(float(objective_scale))
    if not np.isfinite(scale_value):
        raise ValueError("objective_scale must be finite")

    k_start = 0 if min_k_eff == 0 else 1
    ks = np.arange(k_start, effective_max_k + 1, dtype=np.int64)
    if df_path is None:
        df = ks.astype(np.float64)
    else:
        df_arr = np.asarray(df_path, dtype=np.float64).reshape(-1)
        if len(df_arr) < effective_max_k:
            raise ValueError("df_path must be at least as long as the effective objective path")
        if k_start == 0:
            df = np.concatenate(([0.0], df_arr[:effective_max_k]))
        else:
            df = df_arr[:effective_max_k]
    penalty, penalty_weight, ebic_gamma, n_candidates_used = _penalty_array(
        config,
        ks,
        n_eff=n_eff,
        n_candidates=n_candidates,
    )
    if config.objective_penalty not in {"ebic", "ric"}:
        penalty = penalty_weight * df
    objective_used = obj[ks - 1].astype(np.float64, copy=True)
    objective_used[ks == 0] = 0.0
    penalized_score = scale_value * objective_used - penalty
    n_finite_objective = int(np.sum(np.isfinite(objective_used)))
    n_finite_penalized_score = int(np.sum(np.isfinite(penalized_score)))
    valid = (ks >= min_k_eff) & np.isfinite(penalized_score)
    all_penalized_scores_invalid = not bool(valid.any())
    if valid.any():
        order = np.lexsort((ks[valid], -penalized_score[valid]))
        best_pos = np.flatnonzero(valid)[int(order[0])]
        best_k = int(ks[best_pos])
    else:
        warnings.warn(
            "All candidate penalized objective scores are non-finite; "
            "falling back to the effective minimum k.",
            UserWarning,
            stacklevel=2,
        )
        best_k = int(min_k_eff)

    full_objective = np.concatenate(([0.0], obj[:effective_max_k]))
    full_delta = np.diff(full_objective)
    delta_map = dict(zip(np.arange(1, effective_max_k + 1, dtype=np.int64), full_delta))
    delta = np.array([0.0 if k == 0 else delta_map[int(k)] for k in ks], dtype=np.float64)
    objective_nonmonotone_steps = int(np.sum(full_delta[1:] < -1e-12))
    path_exhausted_before_max_k = bool(effective_max_k < int(config.max_k))
    selected_at_effective_max_k = bool(best_k == effective_max_k)
    selected_at_config_max_k = bool(best_k == int(config.max_k))
    selected_at_min_k = bool(best_k == min_k_eff)

    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": objective_used,
            "delta_objective": delta,
            "df": df,
            "penalty_weight": penalty_weight,
            "penalty": penalty,
            "penalty_kind": config.objective_penalty,
            "ebic_gamma": ebic_gamma,
            "n_candidates": n_candidates_used,
            "penalized_score": penalized_score,
            "selected": ks == best_k,
            "n_eff": n_eff,
            "n_eff_source": n_eff_source,
            "weight_sum": weight_sum,
            "kish_n_eff": kish_n_eff,
            "objective_scale": scale_value,
            "objective_scale_source": scale_label,
            "objective_nonmonotone_steps": objective_nonmonotone_steps,
            "n_finite_objective": n_finite_objective,
            "n_finite_penalized_score": n_finite_penalized_score,
            "all_penalized_scores_invalid": all_penalized_scores_invalid,
            "effective_min_k": min_k_eff,
            "effective_max_k": effective_max_k,
            "path_length": path_length,
            "selected_at_effective_max_k": selected_at_effective_max_k,
            "selected_at_config_max_k": selected_at_config_max_k,
            "path_exhausted_before_max_k": path_exhausted_before_max_k,
            "selected_at_min_k": selected_at_min_k,
        }
    )
    return best_k, diag


def select_k_posterior(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale: float | Literal["n_eff"],
    n_samples: int,
    n_candidates: int,
    sample_weight: Optional[np.ndarray] = None,
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
) -> Tuple[int, pd.DataFrame]:
    """Select k from a pseudo-posterior over prefixes on one greedy path.

    HPD intervals are computed over selectable k values. If ``min_k > 0``, the
    zero-feature posterior mass is still reported as ``p_zero`` but is excluded
    from MAP/HPD selection.
    """
    validate_auto_k_config(config)
    if config.k_method != "k_posterior":
        raise ValueError("select_k_posterior requires AutoKConfig(k_method='k_posterior')")

    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)
    path_length = int(len(obj))
    effective_max_k = min(int(max_k if max_k is not None else config.max_k), path_length)
    if effective_max_k <= 0:
        return 0, pd.DataFrame()
    min_k_raw = int(min_k if min_k is not None else config.min_k)
    min_k_eff = max(0, min(min_k_raw, effective_max_k))

    _, weight_sum, kish_n_eff, n_eff, n_eff_source = _objective_weight_diagnostics(
        sample_weight,
        int(n_samples),
        config,
    )
    if objective_scale == "n_eff":
        scale_value = n_eff
        scale_label = "n_eff"
    else:
        scale_value = float(objective_scale)
        scale_label = str(float(objective_scale))
    if not np.isfinite(scale_value):
        raise ValueError("objective_scale must be finite")

    if min_k_eff == 0:
        ks = np.arange(0, effective_max_k + 1, dtype=np.int64)
    else:
        ks = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.arange(min_k_eff, effective_max_k + 1, dtype=np.int64),
            )
        )
    if int(n_candidates) < 1 or int(n_candidates) < int(np.max(ks, initial=0)):
        raise ValueError("n_candidates must be a positive integer >= the largest evaluated k")
    objective_used = obj[ks - 1].astype(np.float64, copy=True)
    objective_used[ks == 0] = 0.0
    gamma = _resolve_ebic_gamma(config, n_eff=n_eff, n_candidates=int(n_candidates))
    log_comb = _log_comb(int(n_candidates), ks)
    log_post = 0.5 * (scale_value * objective_used - ks.astype(np.float64) * np.log(n_eff))
    log_post -= gamma * log_comb
    finite = np.isfinite(log_post)
    if not bool(finite.any()):
        warnings.warn(
            "All posterior log-weights are non-finite; falling back to effective minimum k.",
            UserWarning,
            stacklevel=2,
        )
        best_k = int(min_k_eff)
        post = np.zeros_like(log_post)
        in_hpd = np.zeros_like(finite, dtype=bool)
    else:
        log_norm = float(logsumexp(log_post[finite]))
        post = np.zeros_like(log_post, dtype=np.float64)
        post[finite] = np.exp(log_post[finite] - log_norm)
        selectable = finite.copy()
        if min_k_eff > 0:
            selectable &= ks >= min_k_eff
        if not bool(selectable.any()):
            warnings.warn(
                "No selectable posterior log-weights are finite; falling back to effective minimum k.",
                UserWarning,
                stacklevel=2,
            )
            best_k = int(min_k_eff)
            in_hpd = np.zeros_like(finite, dtype=bool)
        else:
            selectable_pos = np.flatnonzero(selectable)
            selectable_log_norm = float(logsumexp(log_post[selectable]))
            selectable_post = np.exp(log_post[selectable_pos] - selectable_log_norm)
            map_pos = int(np.lexsort((ks[selectable_pos], -selectable_post))[0])
            map_k = int(ks[selectable_pos][map_pos])
            order = np.argsort(-selectable_post, kind="mergesort")
            cumsum = np.cumsum(selectable_post[order])
            cutoff = int(np.searchsorted(cumsum, float(config.posterior_level), side="left"))
            cutoff = min(cutoff, len(order) - 1)
            hpd_positions = selectable_pos[order[: cutoff + 1]]
            in_hpd = np.zeros_like(finite, dtype=bool)
            in_hpd[hpd_positions] = True
            if config.posterior_pick == "smallest_in_hpd":
                best_k = int(np.min(ks[in_hpd]))
            else:
                best_k = map_k

    hpd_ks = ks[in_hpd]
    hpd_lo = int(np.min(hpd_ks)) if hpd_ks.size else int(min_k_eff)
    hpd_hi = int(np.max(hpd_ks)) if hpd_ks.size else int(min_k_eff)
    p_zero = float(post[ks == 0][0]) if np.any(ks == 0) else 0.0
    entropy = float(-np.sum(post[post > 0.0] * np.log(post[post > 0.0])))
    delta = np.zeros_like(objective_used)
    nonzero = ks > 0
    delta[nonzero] = np.diff(np.concatenate(([0.0], obj[:effective_max_k])))[ks[nonzero] - 1]

    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": objective_used,
            "delta_objective": delta,
            "log_post": log_post,
            "post": post,
            "in_hpd": in_hpd,
            "selected": ks == best_k,
            "n_eff": n_eff,
            "n_eff_source": n_eff_source,
            "weight_sum": weight_sum,
            "kish_n_eff": kish_n_eff,
            "objective_scale": scale_value,
            "objective_scale_source": scale_label,
            "ebic_gamma": gamma,
            "n_candidates": int(n_candidates),
            "posterior_level": float(config.posterior_level),
            "hpd_lo": hpd_lo,
            "hpd_hi": hpd_hi,
            "p_zero": p_zero,
            "entropy": entropy,
            "effective_min_k": min_k_eff,
            "effective_max_k": effective_max_k,
            "path_length": path_length,
        }
    )
    return int(best_k), diag


def compute_objective_for_path(
    cache: "FeatureCache",
    y: np.ndarray,
    feature_path: List[str],
    *,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute objective path for an arbitrary ordered feature_path.

    Objective at step t:
        obj[t] = log|Σ_S| - log|Σ_{y,S}|
               = 2 * I(y; S)   (Gaussian MI proxy)
    """
    from sift.estimators.copula import (
        weighted_corr_with_vector,
        weighted_correlation_matrix,
        weighted_rank_gauss_1d,
    )
    from sift.selection.objective import objective_from_corr_path

    if not feature_path:
        return np.empty(0, dtype=np.float64)

    valid_cols = np.asarray(cache.valid_cols)
    orig_to_valid = {int(orig): int(pos) for pos, orig in enumerate(valid_cols)}

    name_to_orig = {}
    if cache.feature_names:
        name_to_orig = {name: i for i, name in enumerate(cache.feature_names)}

    path_valid_pos = []
    for f in feature_path:
        if isinstance(f, str):
            orig_idx = name_to_orig.get(f, None)
            if orig_idx is None:
                continue
        else:
            orig_idx = int(f)

        vpos = orig_to_valid.get(int(orig_idx), None)
        if vpos is None:
            continue
        path_valid_pos.append(vpos)

    if not path_valid_pos:
        return np.empty(0, dtype=np.float64)

    path_valid_pos = np.asarray(path_valid_pos, dtype=np.int64)

    y_arr = np.asarray(y).ravel()
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but cache was built from "
            f"{cache.n_rows_original} rows"
        )
    ys = y_arr[np.asarray(cache.row_idx)]
    zy = weighted_rank_gauss_1d(ys, cache.sample_weight)
    r_y_full = weighted_corr_with_vector(cache.Z, zy, cache.sample_weight).astype(np.float64)

    r_path = r_y_full[path_valid_pos].copy()
    np.clip(r_path, -0.999999, 0.999999, out=r_path)

    if cache.Rxx is not None:
        R_full = np.asarray(cache.Rxx, dtype=np.float64)
        R_path = np.ascontiguousarray(R_full[np.ix_(path_valid_pos, path_valid_pos)], dtype=np.float64)
    else:
        Z_path = np.ascontiguousarray(cache.Z[:, path_valid_pos], dtype=np.float64)
        R_path = weighted_correlation_matrix(
            Z_path,
            np.asarray(cache.sample_weight, dtype=np.float64),
            backend="blas",
        )

    return objective_from_corr_path(R_path, r_path, shrink=shrink, eps=eps)
