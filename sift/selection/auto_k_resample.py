"""Resampling-based automatic-k selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from sift._permute import build_group_info, permute_array, resolve_permutation_method
from sift.estimators.copula import weighted_rank_gauss_1d
from sift.selection.auto_k import AutoKConfig, validate_auto_k_config
from sift.selection.cefsplus import (
    _gaussian_jmi_select,
    _gaussian_mrmr_select,
    cefsplus_loop_with_objective,
)
from sift.selection.panel import build_candidate_panel, local_corr_panel

_STABILITY_PHI_FLOOR = 0.5


def _resolve_null(null: str, *, groups, time) -> str:
    if null == "permute":
        return "global"
    if null == "auto":
        return resolve_permutation_method("auto", groups=groups, time=time)
    if null in {"within_group", "circular_shift"}:
        return null
    raise ValueError("perm_null must be 'auto', 'permute', 'circular_shift', or 'within_group'")


def _run_panel_path(panel, max_k: int, *, method: str = "cefsplus") -> tuple[np.ndarray, np.ndarray]:
    k_actual = min(int(max_k), len(panel.cand))
    if k_actual <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if method == "cefsplus":
        local_path, objective = cefsplus_loop_with_objective(panel.R, panel.r, k_actual, panel.rel)
    elif method in {"mrmr_quot", "mrmr_diff"}:
        local_path = _gaussian_mrmr_select(
            panel.R,
            panel.rel,
            k_actual,
            use_quotient=method == "mrmr_quot",
        )
        objective = np.empty(0, dtype=np.float64)
    elif method in {"jmi", "jmim"}:
        local_path = _gaussian_jmi_select(
            panel.R,
            panel.r,
            panel.rel,
            k_actual,
            use_min=method == "jmim",
        )
        objective = np.empty(0, dtype=np.float64)
    else:
        raise ValueError(f"Unknown Gaussian selector method: {method!r}")
    return panel.cand[local_path].astype(np.int64), np.asarray(objective, dtype=np.float64)


def null_objective_paths(
    cache,
    y,
    *,
    B: int,
    max_k: int,
    null: str,
    time=None,
    groups=None,
    top_m: int,
    corr_prune,
    random_state: int,
) -> np.ndarray:
    """Build permutation-null CEFS+ objective paths, extended flat to max_k."""
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError("y length must match the cache's original row count")
    groups_arr = None if groups is None else np.asarray(groups).reshape(-1)
    time_arr = None if time is None else np.asarray(time).reshape(-1)
    if groups_arr is not None and groups_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("groups length must match y")
    if time_arr is not None and time_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("time length must match y")

    method = _resolve_null(null, groups=groups_arr, time=time_arr)
    if method == "within_group" and groups_arr is None:
        raise ValueError("perm_null='within_group' requires groups")
    if method == "circular_shift" and time_arr is None:
        raise ValueError("perm_null='circular_shift' requires time")
    group_info = None
    if method != "global":
        group_info = build_group_info(groups_arr, time_arr, n_samples=y_arr.shape[0])

    seeds = np.random.SeedSequence(random_state).spawn(int(B))
    out = np.zeros((int(B), int(max_k)), dtype=np.float64)
    for b, child in enumerate(seeds):
        rng = np.random.default_rng(child)
        y_b_full = permute_array(
            y_arr,
            method=method,
            group_info=group_info,
            block_size="auto",
            rng=rng,
        )
        y_b_cache = y_b_full[np.asarray(cache.row_idx, dtype=np.int64)]
        zy_b = weighted_rank_gauss_1d(y_b_cache, cache.sample_weight)
        panel = build_candidate_panel(
            cache,
            None,
            int(max_k),
            top_m=top_m,
            corr_prune=corr_prune,
            method="cefsplus",
            zy=zy_b,
        )
        _path, objective = _run_panel_path(panel, int(max_k), method="cefsplus")
        if objective.size:
            out[b, : objective.size] = objective
            if objective.size < max_k:
                out[b, objective.size :] = objective[-1]
    return out


def select_k_perm_gap(
    objective_path: np.ndarray,
    null_paths: np.ndarray,
    config: AutoKConfig,
) -> tuple[int, pd.DataFrame]:
    """Select k by comparing the real objective curve to permutation null paths."""
    validate_auto_k_config(config)
    if config.k_method != "perm_gap":
        raise ValueError("select_k_perm_gap requires AutoKConfig(k_method='perm_gap')")
    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    nulls = np.asarray(null_paths, dtype=np.float64)
    if obj.size == 0 or nulls.size == 0:
        return 0, pd.DataFrame()
    L = min(obj.size, nulls.shape[1], int(config.max_k))
    obj = obj[:L]
    nulls = nulls[:, :L]
    nulls_full = np.concatenate([np.zeros((nulls.shape[0], 1)), nulls], axis=1)
    obj_full = np.concatenate(([0.0], obj))
    ks = np.arange(0, L + 1, dtype=np.int64)
    null_mean = np.mean(nulls_full, axis=0)
    null_sd = (
        np.std(nulls_full, axis=0, ddof=1)
        if nulls_full.shape[0] >= 2
        else np.zeros(L + 1)
    )
    gap = obj_full - null_mean
    gap_se = null_sd * np.sqrt(1.0 + 1.0 / max(1, nulls.shape[0]))
    floor = max(0, min(int(config.min_k), L))

    if config.gap_rule == "argmax":
        valid = ks >= floor
        selected_k = int(ks[valid][np.argmax(gap[valid])]) if np.any(valid) else 0
    elif config.gap_rule == "gain_envelope":
        step_ks = np.arange(1, L + 1, dtype=np.int64)
        real_gain = np.diff(obj_full)
        null_gain = np.diff(nulls_full, axis=1)
        z = float(norm.ppf(1.0 - float(config.alpha)))
        null_gain_sd = (
            np.std(null_gain, axis=0, ddof=1)
            if null_gain.shape[0] >= 2
            else np.zeros(null_gain.shape[1], dtype=np.float64)
        )
        envelope = np.mean(null_gain, axis=0) + z * null_gain_sd
        bad = real_gain <= envelope
        run = 0
        selected_k = L
        for pos, is_bad in enumerate(bad):
            run = run + 1 if is_bad else 0
            if run >= int(config.stop_patience):
                candidate = int(step_ks[pos - int(config.stop_patience) + 1] - 1)
                if candidate >= floor:
                    selected_k = candidate
                    break
    else:
        valid = ks >= floor
        selected_k = int(ks[valid][np.argmax(gap[valid])]) if np.any(valid) else 0
        for i in range(floor, L):
            if gap[i] >= gap[i + 1] - gap_se[i + 1]:
                selected_k = int(ks[i])
                break

    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": obj_full,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "gap": gap,
            "gap_se": gap_se,
            "selected": ks == selected_k,
            "perm_B": int(nulls.shape[0]),
            "gap_rule": config.gap_rule,
        }
    )
    return selected_k, diag


def bootstrap_paths(
    cache,
    y,
    *,
    B: int,
    max_k: int,
    boot_mode: str,
    top_m: int,
    corr_prune,
    random_state: int,
    method: str = "cefsplus",
) -> list[np.ndarray]:
    """Return bootstrap CEFS+ paths in cache-valid feature coordinates."""
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    y_cache = y_arr[np.asarray(cache.row_idx, dtype=np.int64)]
    seeds = np.random.SeedSequence(random_state).spawn(int(B))
    paths: list[np.ndarray] = []
    for child in seeds:
        rng = np.random.default_rng(child)
        base_w = np.asarray(cache.sample_weight, dtype=np.float64)
        if boot_mode == "half":
            mask = np.zeros(base_w.shape[0], dtype=np.float64)
            chosen = rng.choice(base_w.shape[0], size=max(1, base_w.shape[0] // 2), replace=False)
            mask[chosen] = 1.0
            w_b = base_w * mask
        elif boot_mode == "bayes":
            w_b = base_w * rng.exponential(scale=1.0, size=base_w.shape[0])
        else:
            raise ValueError("boot_mode must be 'bayes' or 'half'")
        if float(np.sum(w_b)) <= 0.0:
            paths.append(np.empty(0, dtype=np.int64))
            continue
        zy_b = weighted_rank_gauss_1d(y_cache, w_b)
        panel = local_corr_panel(
            cache.Z,
            zy_b,
            w_b,
            top_m=top_m,
            corr_prune=corr_prune,
            method=method,
            local_standardize=True,
        )
        path, _objective = _run_panel_path(panel, int(max_k), method=method)
        paths.append(path)
    return paths


def _stability_phi_from_counts(counts: np.ndarray, *, B: int, k: int, p: int) -> float:
    denom = (k / p) * (1.0 - k / p) if 0 < k < p else 0.0
    if denom <= 0.0:
        return 1.0 if k >= p else np.nan
    pi = np.asarray(counts, dtype=np.float64) / max(1, int(B))
    instability = np.mean((B / max(1, B - 1)) * pi * (1.0 - pi))
    return float(1.0 - instability / denom)


def _stability_phi_jackknife_se(indicators: np.ndarray, counts: np.ndarray, *, k: int, p: int) -> float:
    B = int(indicators.shape[0])
    if B < 2:
        return float("nan")
    loo = np.empty(B, dtype=np.float64)
    for b in range(B):
        loo[b] = _stability_phi_from_counts(
            counts - indicators[b],
            B=B - 1,
            k=k,
            p=p,
        )
    finite = loo[np.isfinite(loo)]
    if finite.size < 2:
        return float("nan")
    center = float(np.mean(finite))
    return float(np.sqrt((finite.size - 1) / finite.size * np.sum((finite - center) ** 2)))


def select_k_stability(
    paths: list[np.ndarray],
    p_valid: int,
    config: AutoKConfig,
) -> tuple[int, pd.DataFrame]:
    """Select k from chance-corrected bootstrap path stability."""
    validate_auto_k_config(config)
    if config.k_method != "stability":
        raise ValueError("select_k_stability requires AutoKConfig(k_method='stability')")
    if not paths:
        return 0, pd.DataFrame()
    max_len = min(int(config.max_k), min((len(path) for path in paths), default=0))
    if max_len <= 0:
        return 0, pd.DataFrame()
    effective_min = max(1, min(int(config.min_k), max_len))
    B = len(paths)
    p = int(p_valid)
    rows = []
    indicators = np.zeros((B, p), dtype=bool)
    counts = np.zeros(p, dtype=np.float64)
    set_sizes = np.zeros(B, dtype=np.int64)
    intersections = np.zeros((B, B), dtype=np.int64)
    normalized_paths = [np.asarray(path, dtype=np.int64) for path in paths]
    upper_i, upper_j = np.triu_indices(B, k=1)
    for k in range(1, max_len + 1):
        for b, path in enumerate(normalized_paths):
            feature = int(path[k - 1])
            if feature < 0 or feature >= p or indicators[b, feature]:
                continue
            peers = np.flatnonzero(indicators[:, feature])
            if peers.size:
                intersections[b, peers] += 1
                intersections[peers, b] += 1
            indicators[b, feature] = True
            counts[feature] += 1.0
            set_sizes[b] += 1
        phi = _stability_phi_from_counts(counts, B=B, k=k, p=p)
        phi_se = _stability_phi_jackknife_se(indicators, counts, k=k, p=p)
        union_sizes = (
            set_sizes[upper_i]
            + set_sizes[upper_j]
            - intersections[upper_i, upper_j]
        )
        jaccards = np.ones(union_sizes.size, dtype=np.float64)
        nonempty = union_sizes > 0
        jaccards[nonempty] = (
            intersections[upper_i[nonempty], upper_j[nonempty]]
            / union_sizes[nonempty]
        )
        rows.append(
            {
                "k": k,
                "phi": float(phi),
                "phi_se": phi_se,
                "mean_jaccard": float(np.mean(jaccards)) if jaccards.size else 1.0,
            }
        )
    diag = pd.DataFrame(rows)
    finite_all = diag[np.isfinite(diag["phi"])]
    max_phi = float(finite_all["phi"].max()) if not finite_all.empty else float("nan")
    stopped_by = str(config.stability_rule)
    if config.stability_rule == "pi_threshold":
        raw_selected = int(np.sum(counts / max(1, B) >= float(config.stability_pi)))
        threshold_floor = 0 if int(config.min_k) <= 0 else effective_min
        finite = finite_all
        if finite.empty:
            selected_k = threshold_floor
            stopped_by = "degenerate"
        else:
            best = finite.sort_values(["phi", "k"], ascending=[False, False], kind="mergesort").iloc[0]
            best_phi = float(best["phi"])
            if best_phi < _STABILITY_PHI_FLOOR:
                selected_k = threshold_floor
                stopped_by = "stability_floor"
            else:
                tol = float(best.get("phi_se", np.nan))
                tol = 0.0 if not np.isfinite(tol) else tol
                plateau = finite[finite["phi"] >= best_phi - tol]
                plateau_cap = int(plateau["k"].max()) if not plateau.empty else max_len
                selected_k = min(
                    max_len,
                    max(threshold_floor, min(raw_selected, plateau_cap)),
                )
    else:
        finite = diag[np.isfinite(diag["phi"]) & (diag["k"] >= effective_min)]
        if finite.empty:
            selected_k = effective_min
            stopped_by = "degenerate"
        else:
            best = finite.sort_values(["phi", "k"], ascending=[False, False], kind="mergesort").iloc[0]
            if float(best["phi"]) < _STABILITY_PHI_FLOOR and int(config.min_k) <= 0:
                selected_k = 0
                stopped_by = "stability_floor"
            elif float(best["phi"]) < _STABILITY_PHI_FLOOR:
                selected_k = effective_min
                stopped_by = "stability_floor"
            else:
                tol = float(best.get("phi_se", np.nan))
                tol = 0.0 if not np.isfinite(tol) else tol
                eligible = finite[finite["phi"] >= float(best["phi"]) - tol]
                selected_k = int(eligible.sort_values("k", ascending=False, kind="mergesort").iloc[0]["k"])
    diag["selected"] = diag["k"] == selected_k
    diag["boot_B"] = B
    diag["boot_mode"] = config.boot_mode
    diag["max_phi"] = max_phi
    diag["stability_floor_threshold"] = _STABILITY_PHI_FLOOR
    diag["stopped_by"] = stopped_by
    diag.attrs["max_phi"] = max_phi
    diag.attrs["stability_floor_threshold"] = _STABILITY_PHI_FLOOR
    diag.attrs["stopped_by"] = stopped_by
    return selected_k, diag
