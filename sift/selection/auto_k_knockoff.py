"""Knockoff-interleaved automatic-k selection for CEFS+."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sift.estimators.copula import (
    gaussian_mi_from_corr,
    weighted_corr_with_vector,
    weighted_correlation_matrix,
    weighted_rank_gauss_1d,
)
from sift.estimators.knockoffs import (
    fit_gaussian_knockoffs,
    gaussian_knockoff_mean,
    sample_gaussian_knockoffs,
)
from sift.selection.auto_k import AutoKConfig, validate_auto_k_config
from sift.selection.knockoff_filter import (
    _build_active_rxx,
    _reject_duplicate_feature_names,
    _weighted_variance,
)


def _prepare_knockoff_draw_state(cache, config: AutoKConfig):
    """Fit one cache-level knockoff model for all auto-k draws."""
    _reject_duplicate_feature_names(cache)
    w = np.asarray(cache.sample_weight, dtype=np.float64).reshape(-1)
    if not np.isfinite(w).all() or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("cache.sample_weight must be finite, non-negative, and sum to > 0")
    Z = np.asarray(cache.Z)
    active = _weighted_variance(Z, w) > 1e-12
    if not bool(active.any()):
        raise ValueError("No active non-constant features remain for knockoffs")
    R_active = _build_active_rxx(cache, active, verbose=False)
    model = fit_gaussian_knockoffs(
        R_active,
        s_method=config.knockoff_s_method,
        min_eig=1e-3,
    )
    Z_active = (
        np.asarray(Z, dtype=np.float32)
        if bool(active.all())
        else np.ascontiguousarray(Z[:, active], dtype=np.float32)
    )
    mean = (
        gaussian_knockoff_mean(Z_active, model)
        if int(config.knockoff_draws) > 1
        else None
    )
    return Z, Z_active, active, model, mean


def _pair_aware_cefsplus_entries(
    G: np.ndarray,
    r: np.ndarray,
    *,
    path_depth: int,
    tie_tol: float = 1e-12,
    shrink: float = 1e-6,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair-aware CEFS+ entry sequence on originals plus knockoffs."""
    G_arr = np.asarray(G, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
    if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
        raise ValueError("G must be square")
    if G_arr.shape[0] != r_arr.shape[0] or G_arr.shape[0] % 2:
        raise ValueError("G/r dimensions must describe original-knockoff pairs")

    n_aug = len(r_arr)
    n_pairs = n_aug // 2
    if n_aug == 0 or path_depth <= 0 or np.all(np.abs(r_arr) <= tie_tol):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    Gs = (1.0 - shrink) * G_arr.copy()
    np.fill_diagonal(Gs, 1.0)
    rs = (1.0 - shrink) * r_arr
    tie_break = np.asarray(gaussian_mi_from_corr(rs), dtype=np.float64)

    remaining = np.ones(n_aug, dtype=bool)
    selected = np.empty(0, dtype=np.int64)
    entries: list[int] = []
    gains: list[float] = []
    inv_S = np.empty((0, 0), dtype=np.float64)
    inv_yS = np.array([[1.0]], dtype=np.float64)
    logdet_S = 0.0
    logdet_yS = 0.0

    while len(entries) < int(path_depth) and bool(remaining.any()):
        rem = np.flatnonzero(remaining)
        s = selected.size
        if s == 0:
            s1 = np.ones(rem.size, dtype=np.float64)
            lf = np.zeros(rem.size, dtype=np.float64)
            B = np.empty((0, rem.size), dtype=np.float64)
        else:
            B = Gs[np.ix_(selected, rem)]
            tmp = inv_S @ B
            s1 = np.maximum(1.0 - np.einsum("ij,ij->j", B, tmp), eps)
            lf = logdet_S + np.log(s1)

        B2 = np.vstack([rs[rem], B])
        tmp2 = inv_yS @ B2
        s2 = np.maximum(1.0 - np.einsum("ij,ij->j", B2, tmp2), eps)
        scores = lf - (logdet_yS + np.log(s2))
        best_score = float(np.max(scores))
        if not np.isfinite(best_score):
            break

        tied = rem[np.abs(scores - best_score) <= tie_tol]
        neutralized = False
        pair_ids = tied % n_pairs
        sides = tied >= n_pairs
        for pair_id in np.unique(pair_ids):
            pair_sides = sides[pair_ids == pair_id]
            if pair_sides.size > 1 and bool(np.any(pair_sides)) and bool(np.any(~pair_sides)):
                remaining[int(pair_id)] = False
                remaining[int(pair_id) + n_pairs] = False
                neutralized = True
        if neutralized:
            continue

        best_tie_break = float(np.max(tie_break[tied]))
        tied = tied[np.abs(tie_break[tied] - best_tie_break) <= tie_tol]
        pair_ids = tied % n_pairs
        sides = tied >= n_pairs
        neutralized = False
        for pair_id in np.unique(pair_ids):
            pair_sides = sides[pair_ids == pair_id]
            if pair_sides.size > 1 and bool(np.any(pair_sides)) and bool(np.any(~pair_sides)):
                remaining[int(pair_id)] = False
                remaining[int(pair_id) + n_pairs] = False
                neutralized = True
        if neutralized:
            continue

        j = int(tied[np.argmin(tied % n_pairs)])
        rem_pos = int(np.where(rem == j)[0][0])
        s1_best = float(s1[rem_pos])
        s2_best = float(s2[rem_pos])

        if s == 0:
            inv_S = np.array([[1.0 / s1_best]], dtype=np.float64)
        else:
            b = B[:, rem_pos].reshape(-1, 1)
            v = inv_S @ b
            inv_S_new = np.empty((s + 1, s + 1), dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                inv_S_new[:s, :s] = inv_S + (v @ v.T) / s1_best
                inv_S_new[:s, s] = -v[:, 0] / s1_best
                inv_S_new[s, :s] = -v[:, 0] / s1_best
                inv_S_new[s, s] = 1.0 / s1_best
            if not np.isfinite(inv_S_new).all():
                break
            inv_S = inv_S_new
        logdet_S += float(np.log(s1_best))

        b2 = B2[:, rem_pos].reshape(-1, 1)
        v2 = inv_yS @ b2
        inv_yS_new = np.empty((s + 2, s + 2), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            inv_yS_new[: s + 1, : s + 1] = inv_yS + (v2 @ v2.T) / s2_best
            inv_yS_new[: s + 1, s + 1] = -v2[:, 0] / s2_best
            inv_yS_new[s + 1, : s + 1] = -v2[:, 0] / s2_best
            inv_yS_new[s + 1, s + 1] = 1.0 / s2_best
        if not np.isfinite(inv_yS_new).all():
            break
        inv_yS = inv_yS_new
        logdet_yS += float(np.log(s2_best))

        entries.append(j)
        gains.append(max(float(np.log(s1_best) - np.log(s2_best)), 0.0))
        selected = np.append(selected, j)
        remaining[j] = False

    return np.asarray(entries, dtype=np.int64), np.asarray(gains, dtype=np.float64)


def _knockoff_prefix_table(
    entries: np.ndarray,
    gains: np.ndarray,
    kept: np.ndarray,
    valid_cols: np.ndarray,
    *,
    q: float,
    max_k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build first-entry pair table and SeqStep+ selection from an entry path."""
    kept_arr = np.asarray(kept, dtype=np.int64)
    n_pairs = int(kept_arr.size)
    valid_arr = np.asarray(valid_cols, dtype=np.int64)
    first_seen: dict[int, tuple[int, int, float]] = {}
    for step, entry in enumerate(np.asarray(entries, dtype=np.int64), start=1):
        pair_id = int(entry % n_pairs)
        if pair_id in first_seen:
            continue
        label = 1 if entry < n_pairs else -1
        gain = float(np.asarray(gains, dtype=np.float64)[step - 1])
        first_seen[pair_id] = (step, label, gain)

    rows = []
    pos = 0
    neg = 0
    selected_pair_order: list[int] = []
    ordered = sorted(first_seen.items(), key=lambda item: (item[1][0], item[0]))
    fdp_values = []
    for rank, (pair_id, (entry_step, label, gain)) in enumerate(ordered, start=1):
        if label > 0:
            pos += 1
        else:
            neg += 1
        fdp_hat = (1.0 + neg) / max(1, pos)
        fdp_values.append(fdp_hat)
        kept_pos = int(kept_arr[pair_id])
        rows.append(
            {
                "rank": rank,
                "pair_position": pair_id,
                "feature_index_valid": kept_pos,
                "selected_index": int(valid_arr[kept_pos]),
                "label": label,
                "entry_step": entry_step,
                "entry_gain": gain,
                "fdp_hat": fdp_hat,
            }
        )

    if rows:
        eligible = [i for i, fdp_hat in enumerate(fdp_values) if fdp_hat <= float(q)]
        cutoff = max(eligible) if eligible else -1
        for i, row in enumerate(rows):
            is_selected = bool(i <= cutoff and int(row["label"]) > 0)
            row["selected"] = is_selected
            if is_selected:
                selected_pair_order.append(int(row["pair_position"]))
    selected_pair_order = selected_pair_order[: int(max_k)]
    selected_pair_set = set(selected_pair_order)
    for row in rows:
        row["selected"] = bool(row.get("selected", False) and int(row["pair_position"]) in selected_pair_set)
    selected_valid = (
        kept_arr[np.asarray(selected_pair_order, dtype=np.int64)]
        if selected_pair_order
        else np.empty(0, dtype=np.int64)
    )
    diag = pd.DataFrame(rows)
    if diag.empty:
        diag = pd.DataFrame(
            columns=[
                "rank",
                "pair_position",
                "feature_index_valid",
                "selected_index",
                "label",
                "entry_step",
                "entry_gain",
                "fdp_hat",
                "selected",
            ]
        )
    return selected_valid.astype(np.int64), diag


def _draw_knockoff_path(
    cache,
    y,
    *,
    config: AutoKConfig,
    top_m: int,
    random_state: int,
    draw_state=None,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError("y length must match the cache's original row count")
    y_cache = y_arr[np.asarray(cache.row_idx, dtype=np.int64)]
    w = np.asarray(cache.sample_weight, dtype=np.float64).reshape(-1)
    zy = weighted_rank_gauss_1d(y_cache, w)

    if draw_state is None:
        draw_state = _prepare_knockoff_draw_state(cache, config)
    Z, Z_active, active, model, mean = draw_state
    Zt_active = sample_gaussian_knockoffs(
        Z_active,
        model,
        np.random.default_rng(int(random_state)),
        mean=mean,
    )
    Zt = np.zeros_like(Z, dtype=np.float32)
    Zt[:, active] = Zt_active
    r = weighted_corr_with_vector(Z, zy, w)
    rt = weighted_corr_with_vector(Zt, zy, w)
    p = Z.shape[1]
    m = min(max(1, int(top_m)), p)
    pair_score = np.maximum(np.abs(r), np.abs(rt))
    kept = np.lexsort((np.arange(p, dtype=np.int64), -pair_score))[:m].astype(np.int64)

    Z_aug = np.ascontiguousarray(
        np.column_stack([Z[:, kept], Zt[:, kept]]), dtype=np.float32
    )
    G = weighted_correlation_matrix(Z_aug, w, backend="blas")
    r_aug = np.concatenate([r[kept], rt[kept]])
    entries, gains = _pair_aware_cefsplus_entries(
        G,
        r_aug,
        path_depth=min(2 * m, max(2 * int(config.max_k), int(config.max_k) + 20)),
    )

    selected_valid, diag = _knockoff_prefix_table(
        entries,
        gains,
        kept,
        cache.valid_cols,
        q=float(config.knockoff_q),
        max_k=int(config.max_k),
    )
    diag["q"] = float(config.knockoff_q)
    diag["s_method"] = config.knockoff_s_method
    diag["screen_pairs"] = int(m)
    diag["corr_prune_disabled"] = True
    return selected_valid.astype(np.int64), int(selected_valid.size), diag


def select_k_knockoff_path(
    cache,
    y,
    config: AutoKConfig,
    *,
    top_m: int,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    """Select originals from a pair-aware knockoff-interleaved CEFS+ path."""
    validate_auto_k_config(config)
    if config.k_method != "knockoff_path":
        raise ValueError("select_k_knockoff_path requires AutoKConfig(k_method='knockoff_path')")

    draw_state = _prepare_knockoff_draw_state(cache, config)
    seeds = np.random.SeedSequence(int(config.random_state)).spawn(int(config.knockoff_draws))
    selected_sets: list[np.ndarray] = []
    frames = []
    for draw_idx, child in enumerate(seeds):
        seed = int(np.random.default_rng(child).integers(0, np.iinfo(np.int32).max))
        selected, _k, diag = _draw_knockoff_path(
            cache,
            y,
            config=config,
            top_m=top_m,
            random_state=seed,
            draw_state=draw_state,
        )
        selected_sets.append(selected)
        diag = diag.copy()
        diag.insert(0, "draw", draw_idx)
        frames.append(diag)

    diag_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    multi_draw = int(config.knockoff_draws) > 1
    diag_all.attrs.update(
        {
            "q_scope": "per_draw",
            "per_draw_fdr_control": "approximate_plugin",
            "aggregation": "selection_frequency" if multi_draw else "single_draw",
            "aggregation_threshold": 0.5 if multi_draw else None,
            "aggregation_fdr_control": "none" if multi_draw else "not_applicable",
            "aggregation_preserves_per_draw_fdr": not multi_draw,
        }
    )
    if not selected_sets:
        return np.empty(0, dtype=np.int64), 0, diag_all
    if not multi_draw:
        selected_final = selected_sets[0]
    else:
        counts = np.zeros(cache.Z.shape[1], dtype=np.float64)
        for selected in selected_sets:
            counts[np.asarray(selected, dtype=np.int64)] += 1.0
        freq = counts / float(config.knockoff_draws)
        eligible = np.flatnonzero(freq >= 0.5).astype(np.int64)
        if eligible.size:
            order = np.lexsort((eligible, -freq[eligible]))
            selected_final = eligible[order[: int(config.max_k)]].astype(np.int64)
        else:
            selected_final = np.empty(0, dtype=np.int64)
        if not diag_all.empty:
            diag_all["selection_frequency"] = diag_all["feature_index_valid"].map(
                {int(i): float(freq[int(i)]) for i in np.flatnonzero(freq > 0.0)}
            ).fillna(0.0)
            selected_final_set = set(selected_final.astype(int).tolist())
            diag_all["selected_final"] = diag_all["feature_index_valid"].map(
                lambda value: int(value) in selected_final_set
            )

    return selected_final.astype(np.int64), int(selected_final.size), diag_all


__all__ = ["select_k_knockoff_path"]
