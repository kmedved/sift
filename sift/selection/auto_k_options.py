"""Immutable option-group views for ``sift.AutoKConfig``.

These types are module-scoped ergonomic helpers.  ``AutoKConfig`` remains the
sole storage format, with its original flat dataclass fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AutoKObjectiveOptions:
    objective_penalty: Literal[
        "bic", "mdl", "aic", "hqc", "custom", "ebic", "ric"
    ] = "bic"
    objective_penalty_weight: float | None = None
    objective_n_eff: float | None = None
    n_eff_mode: Literal["auto", "kish", "weight_sum"] | float = "auto"
    ebic_gamma: Literal["auto"] | float = "auto"
    binary_objective_mode: Literal["refit", "score_test"] = "refit"


@dataclass(frozen=True)
class AutoKTestOptions:
    alpha: float = 0.05
    m_mode: Literal["all", "panel", "li_ji"] = "all"
    stop_patience: int = 2


@dataclass(frozen=True)
class AutoKPermutationOptions:
    perm_B: int = 20
    perm_null: Literal[
        "auto", "permute", "circular_shift", "within_group"
    ] = "auto"
    gap_rule: Literal["tibshirani", "argmax", "gain_envelope"] = "tibshirani"


@dataclass(frozen=True)
class AutoKKnockoffOptions:
    knockoff_q: float = 0.2
    knockoff_draws: int = 1
    knockoff_s_method: Literal["equi", "mvr", "me"] = "equi"
    knockoff_return: Literal["set", "prefix"] = "set"


@dataclass(frozen=True)
class AutoKCVOptions:
    strategy: Literal["time_holdout", "group_cv", "kfold"] = "time_holdout"
    metric: Any = "auto"
    val_frac: float = 0.2
    n_splits: int = 5
    xfit_folds: int = 5
    xfit_mode: Literal["shared_z", "exact"] = "shared_z"
    xfit_ridge: float = 1e-3
    selection_rule: Literal["best", "one_se", "plateau", "tolerance"] = "best"
    one_se_multiplier: float = 1.0
    score_abs_tol: float | None = None
    score_rel_tol: float | None = None
    plateau_prefer: Literal["smallest", "center", "best", "largest"] = "smallest"
    plateau_min_points: int = 2


@dataclass(frozen=True)
class AutoKStabilityOptions:
    boot_B: int = 30
    boot_mode: Literal["bayes", "half"] = "bayes"
    stability_rule: Literal["max_one_se", "pi_threshold"] = "max_one_se"
    stability_pi: float = 0.6


@dataclass(frozen=True)
class AutoKExperimentalOptions:
    auto_k_mode: Literal["prefix_only", "nested"] = "prefix_only"
    elbow_min_rel_gain: float = 0.02
    elbow_patience: int = 3
    posterior_level: float = 0.9
    posterior_pick: Literal["map", "smallest_in_hpd"] = "map"
    floor_z: float = 2.5
    floor_window: float | int = 0.2
    consensus_methods: tuple[str, ...] = (
        "ebic",
        "chi2_stop",
        "perm_gap",
        "gaussian_cv",
    )
    auto_dense_check: bool = False
    auto_dense_min_k: int = 100
    auto_dense_min_frac: float = 0.25
    auto_dense_disagreement_ratio: float = 2.0


AUTO_K_OPTION_GROUP_TYPES = {
    "objective": AutoKObjectiveOptions,
    "test": AutoKTestOptions,
    "perm": AutoKPermutationOptions,
    "knockoff": AutoKKnockoffOptions,
    "cv": AutoKCVOptions,
    "stability": AutoKStabilityOptions,
    "experimental": AutoKExperimentalOptions,
}


__all__ = [
    "AutoKCVOptions",
    "AutoKExperimentalOptions",
    "AutoKKnockoffOptions",
    "AutoKObjectiveOptions",
    "AutoKPermutationOptions",
    "AutoKStabilityOptions",
    "AutoKTestOptions",
]
