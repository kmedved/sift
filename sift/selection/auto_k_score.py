"""Score-curve selection-rule helpers for automatic k selection."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from sift.selection.auto_k import AutoKConfig


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
