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


def _slice_auto_prefix(path, path_indices, selected_steps: int, prefix_widths) -> tuple[list, list]:
    """Slice a raw cached path by additional-block (or column) steps."""
    from sift.selection.blocks import slice_prefix_by_steps

    widths = tuple(prefix_widths) if prefix_widths is not None else tuple(
        range(1, len(path) + 1)
    )
    return (
        slice_prefix_by_steps(path, selected_steps, widths),
        slice_prefix_by_steps(path_indices, selected_steps, widths),
    )


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
    df_path: np.ndarray | None = None,
    ic_dimension: str = "k",
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
        df_path=df_path,
        ic_dimension=ic_dimension,
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


def _conditioning_valid_sets(cache, unused: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    from sift.selection.conditioning import (
        map_original_to_valid,
        named_feature_space,
        omitted_conditioning,
        resolve_conditioning,
    )

    unused = unused or {}
    if omitted_conditioning(unused.get("include"), unused.get("exclude"), unused.get("candidates")):
        return None, None
    cache_names = list(cache.feature_names) if cache.feature_names is not None else [
        f"x{i}"
        for i in range(int(np.max(cache.valid_cols)) + 1 if len(cache.valid_cols) else 0)
    ]
    named = named_feature_space(
        cache.feature_names,
        synthetic=bool(getattr(cache, "feature_names_are_synthetic", False))
        or cache.feature_names is None,
    )
    resolved = resolve_conditioning(
        unused.get("include"),
        unused.get("exclude"),
        unused.get("candidates"),
        feature_names=cache_names,
        named=named,
        k=1,
    )
    if resolved is None:
        return None, None
    protect = map_original_to_valid(
        resolved.include,
        cache.valid_cols,
        feature_names=cache_names,
        label="include",
    )
    pool = map_original_to_valid(
        resolved.discovery,
        cache.valid_cols,
        feature_names=cache_names,
        label="candidates",
        missing="drop",
    )
    return protect, pool


def _discovery_n_candidates(cache, unused: dict | None = None) -> int:
    _protect, pool = _conditioning_valid_sets(cache, unused)
    if pool is None:
        return len(cache.valid_cols)
    return int(pool.size)


def _gain_test_candidate_inputs(
    cache,
    y,
    k: int,
    top_m: int,
    corr_prune,
    method: str,
    config: AutoKConfig,
    unused: dict | None = None,
) -> tuple[int, np.ndarray | None]:
    protect, pool = _conditioning_valid_sets(cache, unused)
    if config.m_mode == "all":
        return _discovery_n_candidates(cache, unused), None
    panel = build_candidate_panel(
        cache,
        y,
        k,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
        protect_valid=protect,
        pool_valid=pool,
    )
    n_protect = 0 if protect is None else int(np.asarray(protect).size)
    if n_protect and panel.cand.size >= n_protect:
        n_panel_disc = int(panel.cand.size) - n_protect
    else:
        n_panel_disc = int(panel.cand.size)
    if config.m_mode == "panel":
        return n_panel_disc, None
    if n_protect:
        eigs = _conditioned_discovery_correlation_eigs(panel.R, n_protect)
    else:
        eigs = np.linalg.eigvalsh(panel.R) if panel.R.size else None
    return _discovery_n_candidates(cache, unused), eigs


def _conditioned_discovery_correlation_eigs(
    R: np.ndarray,
    n_protect: int,
    *,
    shrink: float = 1e-6,
) -> np.ndarray | None:
    """Eigenvalues of discovery correlation after partialling out include.

    Uses the same off-diagonal shrink as the conditioned CEFS+ Gaussian
    objective, then the Schur complement of the include block, renormalized
    to a correlation. ``n_protect=0`` is handled by the caller so omitted
    include stays on ``eigvalsh(panel.R)``.
    """
    R = np.asarray(R, dtype=np.float64)
    n_protect = int(n_protect)
    if R.size == 0 or n_protect <= 0 or R.shape[0] <= n_protect:
        return None
    scale = 1.0 - float(shrink)
    G = scale * R
    np.fill_diagonal(G, 1.0)
    rss = np.ascontiguousarray(G[:n_protect, :n_protect])
    rsd = np.ascontiguousarray(G[:n_protect, n_protect:])
    rdd = np.ascontiguousarray(G[n_protect:, n_protect:])
    schur = rdd - rsd.T @ np.linalg.solve(rss, rsd)
    schur = 0.5 * (schur + schur.T)
    diag = np.clip(np.diag(schur), 0.0, np.inf)
    scale_d = np.sqrt(diag)
    inv = np.divide(1.0, scale_d, out=np.zeros_like(scale_d), where=scale_d > 1e-12)
    corr = schur * inv[:, None] * inv[None, :]
    np.fill_diagonal(corr, np.where(scale_d > 1e-12, 1.0, 0.0))
    return np.linalg.eigvalsh(corr)
