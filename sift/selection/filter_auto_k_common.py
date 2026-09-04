"""Shared auto-k count and guard primitives."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sift._logging import logger
from sift.selection import auto_k as auto_k_module
from sift.selection.auto_k import AutoKConfig
from sift.selection.panel import build_candidate_panel


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
        logger.info(f"  {label} selected k={selected_count}")


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
