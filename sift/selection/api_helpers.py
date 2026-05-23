"""Shared helpers for function-style selector APIs."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from sift.selection.auto_k import AutoKConfig
from sift.selection.result import FilterSelectionResult


def validate_groups_time(
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    n_rows: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate and coerce optional group/time arrays."""
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != n_rows:
            raise ValueError(f"groups has {len(groups)} elements but X has {n_rows} rows")
    if time is not None:
        time = np.asarray(time).reshape(-1)
        if len(time) != n_rows:
            raise ValueError(f"time has {len(time)} elements but X has {n_rows} rows")
    return groups, time


def to_filter_result(
    selected_features: List[str],
    selected_indices: Optional[List[int]],
    selector_metadata: dict,
    return_result: bool,
) -> List[str] | FilterSelectionResult:
    """Return a result object only when requested, otherwise list output."""
    if return_result:
        return FilterSelectionResult(
            selected_features=selected_features,
            selected_indices=selected_indices,
            selector_metadata=selector_metadata,
        )
    return selected_features


def build_selector_metadata(
    selector: str,
    *,
    k: int | str,
    k_requested: int | str,
    top_m: Optional[int],
    n_features: int,
    auto_k: bool,
    extra: Optional[dict] = None,
) -> dict:
    """Build a concise metadata payload for filter-result return mode."""
    metadata = {
        "selector": selector,
        "k_requested": k_requested,
        "k": k,
        "top_m": top_m,
        "n_features": int(n_features),
        "auto_k": auto_k,
    }
    if extra:
        metadata.update(extra)
    return metadata


def auto_k_summary(
    config: AutoKConfig,
    *,
    selected_k: int,
    path_length: int,
    effective_max_k: int,
    effective_min_k: Optional[int] = None,
    diagnostics: Optional[pd.DataFrame] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build compact auto-k summary diagnostics shared by result objects."""
    if effective_min_k is None:
        effective_min_k = max(1, min(int(config.min_k), int(effective_max_k)))
    selected_at_effective_max = bool(selected_k == effective_max_k)
    selected_at_config_max = bool(selected_k == int(config.max_k))
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
        "selected_at_effective_max_k": selected_at_effective_max,
        "selected_at_config_max_k": selected_at_config_max,
        "path_exhausted_before_max_k": bool(effective_max_k < int(config.max_k)),
    }
    if diagnostics is not None and not diagnostics.empty:
        if "best_k" in diagnostics:
            summary["best_k"] = int(diagnostics["best_k"].iloc[0])
        if "best_score" in diagnostics:
            summary["best_score"] = float(diagnostics["best_score"].iloc[0])
        if "selection_rule_effective" in diagnostics:
            summary["selection_rule_effective"] = diagnostics[
                "selection_rule_effective"
            ].iloc[0]
        if "one_se_unavailable" in diagnostics:
            summary["one_se_unavailable"] = bool(diagnostics["one_se_unavailable"].iloc[0])
        if "objective_nonmonotone_steps" in diagnostics:
            summary["objective_nonmonotone_steps"] = int(
                diagnostics["objective_nonmonotone_steps"].iloc[0]
            )
    if extra:
        summary.update(extra)
    return summary


def safe_name_indices(
    feature_names: List[str],
    selected_features: List,
) -> Optional[List[int]]:
    """Map selected names to indices when the mapping is unambiguous."""
    if len(feature_names) != len(set(feature_names)):
        return None

    if not selected_features:
        return []

    if all(isinstance(v, (int, np.integer)) for v in selected_features):
        selected_indices = [int(v) for v in selected_features]
        if not all(0 <= i < len(feature_names) for i in selected_indices):
            return None
        return selected_indices

    index_map = {feature: idx for idx, feature in enumerate(feature_names)}
    selected_indices = []
    for name in selected_features:
        idx = index_map.get(name)
        if idx is None:
            return None
        selected_indices.append(idx)
    return selected_indices
