"""CEFS+ selection using log-det Schur complement updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from sift._numba import njit_optional_cache
from sift._progress import ProgressCallback, report_progress
from sift.estimators.copula import (
    FeatureCache,
    gaussian_mi_from_corr,
)
from sift.selection.objective import objective_from_corr_path
from sift.selection.panel import build_candidate_panel
from sift.selection.proxies import (
    proxy_frame_from_panel,
    reject_unavailable_proxy_positions,
)
from sift.selection.result import _PROXY_CORRELATIONS_ATTR
from sift.selection.blocks import (
    map_blocks_to_valid,
    require_atomic_conditioning,
    resolve_feature_blocks,
)
from sift.selection.conditioning import (
    compose_selected,
    conditioning_record,
    map_original_to_valid,
    named_feature_space,
    resolve_conditioning,
)
from sift.selection.knockoff_filter import (
    _reject_duplicate_feature_names,
    _validate_prebuilt_cache_structure,
)

CorrPrune = float | None | Literal["auto"]

if TYPE_CHECKING:
    from sift.selection.view import SelectionView


def _gaussian_mrmr_select(
    R: np.ndarray,
    rel: np.ndarray,
    k: int,
    use_quotient: bool,
    floor: float = 1e-6,
    callback: ProgressCallback | None = None,
    preselected: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    m = len(rel)
    is_sel = np.zeros(m, dtype=bool)
    if eligible is None:
        eligible_mask = np.ones(m, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible, dtype=bool)
    if preselected is not None and len(preselected):
        is_sel[np.asarray(preselected, dtype=np.int64)] = True
    eligible_mask = eligible_mask & ~is_sel
    k = min(k, int(np.sum(eligible_mask)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    selected = np.empty(k, dtype=np.int64)
    red_sum = np.zeros(m, dtype=np.float64)
    n_pre = int(np.sum(is_sel))
    if n_pre:
        for j_pre in np.flatnonzero(is_sel):
            red = gaussian_mi_from_corr(R[int(j_pre)])
            mask = ~is_sel
            red_sum[mask] += red[mask]
        t0 = n_pre
        mean_red = red_sum / t0
        if use_quotient:
            score = rel / np.maximum(mean_red, floor)
        else:
            score = rel - mean_red
        score[~eligible_mask] = -np.inf
        j0 = int(np.argmax(score))
        if not np.isfinite(score[j0]):
            return np.empty(0, dtype=np.int64)
    else:
        scores0 = np.where(eligible_mask, rel, -np.inf)
        j0 = int(np.argmax(scores0))
        t0 = 0

    selected[0] = j0
    is_sel[j0] = True
    eligible_mask[j0] = False
    count = 1
    if callback is not None:
        report_progress(
            callback,
            count,
            k,
            stage="path",
            selector="mrmr_quot" if use_quotient else "mrmr_diff",
        )

    for t in range(1, k):
        last = selected[t - 1]
        red = gaussian_mi_from_corr(R[last])
        mask = ~is_sel
        red_sum[mask] += red[mask]

        mean_red = red_sum / (t + t0)
        if use_quotient:
            score = rel / np.maximum(mean_red, floor)
        else:
            score = rel - mean_red

        score[~eligible_mask] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break

        selected[t] = j
        is_sel[j] = True
        eligible_mask[j] = False
        count += 1
        if callback is not None:
            report_progress(
                callback,
                count,
                k,
                stage="path",
                selector="mrmr_quot" if use_quotient else "mrmr_diff",
            )

    return selected[:count]


def _gaussian_jmi_select(
    R: np.ndarray,
    r_y: np.ndarray,
    rel: np.ndarray,
    k: int,
    use_min: bool,
    callback: ProgressCallback | None = None,
    preselected: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    m = len(r_y)
    is_sel = np.zeros(m, dtype=bool)
    if eligible is None:
        eligible_mask = np.ones(m, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible, dtype=bool)
    if preselected is not None and len(preselected):
        is_sel[np.asarray(preselected, dtype=np.int64)] = True
    eligible_mask = eligible_mask & ~is_sel
    k = min(k, int(np.sum(eligible_mask)))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    selected = np.empty(k, dtype=np.int64)
    scores = np.full(m, np.inf, dtype=np.float64) if use_min else np.zeros(m, dtype=np.float64)

    r2 = np.empty(m, dtype=np.float64)
    frac = np.empty(m, dtype=np.float64)
    eps = 1e-8

    def _accumulate_from(last: int) -> None:
        r_ys = float(r_y[last])
        r_fs = R[last]
        denom = 1.0 - r_fs * r_fs
        a = r_y - r_ys * r_fs
        r2.fill(r_ys * r_ys)
        frac.fill(0.0)
        np.divide(a * a, denom, out=frac, where=denom >= eps)
        np.add(r2, frac, out=r2)
        np.clip(r2, 0.0, 0.99999, out=r2)
        mi = -0.5 * np.log(1.0 - r2)
        mask = eligible_mask
        if use_min:
            scores[mask] = np.minimum(scores[mask], mi[mask])
        else:
            scores[mask] += mi[mask]

    if np.any(is_sel):
        for last in np.flatnonzero(is_sel):
            _accumulate_from(int(last))
        pick_scores = np.where(eligible_mask, scores, -np.inf)
        pick_scores = np.where(np.isfinite(pick_scores), pick_scores, rel)
        j0 = int(np.argmax(np.where(eligible_mask, pick_scores, -np.inf)))
    else:
        j0 = int(np.argmax(np.where(eligible_mask, rel, -np.inf)))
    selected[0] = j0
    is_sel[j0] = True
    eligible_mask[j0] = False
    count = 1
    if callback is not None:
        report_progress(
            callback,
            count,
            k,
            stage="path",
            selector="jmim" if use_min else "jmi",
        )

    for t in range(1, k):
        _accumulate_from(int(selected[t - 1]))
        pick_scores = np.where(eligible_mask, scores, -np.inf)
        j = int(np.argmax(pick_scores))
        if not np.isfinite(pick_scores[j]):
            break

        selected[t] = j
        is_sel[j] = True
        eligible_mask[j] = False
        count += 1
        if callback is not None:
            report_progress(
                callback,
                count,
                k,
                stage="path",
                selector="jmim" if use_min else "jmi",
            )

    return selected[:count]


def _chol_logdet(matrix: np.ndarray, *, shrink: float, eps: float) -> float:
    """Log-determinant with the CEFS+ diagonal floor and optional ridge."""
    a0 = np.array(matrix, dtype=np.float64, copy=True)
    a0 = 0.5 * (a0 + a0.T)
    n = a0.shape[0]
    for i in range(n):
        a0[i, i] = max(float(a0[i, i]), eps)
    ridge = 0.0
    last_err: Exception | None = None
    for _attempt in range(8):
        a = a0 if ridge == 0.0 else a0 + ridge * np.eye(n, dtype=np.float64)
        try:
            chol = np.linalg.cholesky(a)
            diag = np.diag(chol)
            if np.any(diag <= 0.0):
                raise np.linalg.LinAlgError("non-positive Cholesky diagonal")
            return float(2.0 * np.sum(np.log(diag)))
        except np.linalg.LinAlgError as err:
            last_err = err
            ridge = float(shrink) if ridge == 0.0 else max(10.0 * ridge, float(shrink))
    if last_err is not None:
        raise last_err
    raise np.linalg.LinAlgError("Cholesky log-det failed")


def _block_residual_cov(
    R: np.ndarray,
    L: np.ndarray,
    d: np.ndarray,
    members: np.ndarray,
    t: int,
    scale: float,
    eps: float,
) -> np.ndarray:
    b = int(members.size)
    g = np.empty((b, b), dtype=np.float64)
    for a in range(b):
        i = int(members[a])
        g[a, a] = max(float(d[i]), eps)
        for c in range(a + 1, b):
            j = int(members[c])
            acc = float(R[i, j]) * scale
            for s in range(t):
                acc -= float(L[i, s]) * float(L[j, s])
            g[a, c] = acc
            g[c, a] = acc
    return 0.5 * (g + g.T)


def _cefsplus_apply_column(
    j: int,
    *,
    R: np.ndarray,
    L: np.ndarray,
    Ly: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
    remaining: np.ndarray,
    t: int,
    scale: float,
    eps: float,
    dy: float,
) -> float:
    """Apply one CEFS+ Cholesky column and return the new residual y variance."""
    if t == 0:
        s1_best = 1.0
        s2_best = max(1.0 - float(c[j]) * float(c[j]), eps)
    else:
        s1_best = max(float(d[j]), eps)
        s2_best = max(float(d[j]) - float(c[j]) * float(c[j]) / dy, eps)
    sq = np.sqrt(s1_best)
    ly = float(c[j]) / sq
    Ly[t] = ly
    dy = dy - ly * ly
    m = len(c)
    for i in range(m):
        if not remaining[i] or i == j:
            continue
        acc = float(R[i, j]) * scale
        for a in range(t):
            acc -= float(L[i, a]) * float(L[j, a])
        lij = acc / sq
        L[i, t] = lij
        d[i] -= lij * lij
        c[i] -= lij * ly
    remaining[j] = False
    return dy


def _cefsplus_block_gain(
    members: np.ndarray,
    *,
    R: np.ndarray,
    L: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
    t: int,
    dy: float,
    scale: float,
    shrink: float,
    eps: float,
) -> float:
    """Joint log-det gain of a remaining block given the current selected set."""
    g = _block_residual_cov(R, L, d, members, t, scale, eps)
    c_block = np.asarray(c[members], dtype=np.float64)
    dy_eff = max(float(dy), eps)
    g_y = g - np.outer(c_block, c_block) / dy_eff
    return _chol_logdet(g, shrink=shrink, eps=eps) - _chol_logdet(
        g_y, shrink=shrink, eps=eps
    )


def cefsplus_block_loop(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    block_members: Sequence[np.ndarray],
    *,
    forced_blocks: Sequence[int] = (),
    eligible_blocks: Sequence[int] | None = None,
    want_objective: bool = False,
    shrink: float = 1e-6,
    eps: float = 1e-12,
    callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Greedy CEFS+ over blocks using joint residual log-det gain.

    A block's score is ``log|Σ_{B|S}| - log|Σ_{B|S,y}|``, not the gain of a
    representative column. Singleton blocks recover the column CEFS+ step.
    Selected blocks expand in member order (original local index order).
    """
    m = len(r)
    n_blocks = len(block_members)
    if k <= 0 or m == 0 or n_blocks == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            [],
        )
    scale = 1.0 - shrink
    remaining = np.ones(m, dtype=np.bool_)
    remaining_blocks = np.ones(n_blocks, dtype=np.bool_)
    if eligible_blocks is not None:
        remaining_blocks[:] = False
        for idx in eligible_blocks:
            remaining_blocks[int(idx)] = True
    forced = [int(i) for i in forced_blocks]
    for idx in forced:
        remaining_blocks[idx] = False
    n_discover = min(int(k), int(np.sum(remaining_blocks)))
    L = np.zeros((m, m), dtype=np.float64)
    Ly = np.zeros(m, dtype=np.float64)
    d = np.ones(m, dtype=np.float64)
    c = scale * np.asarray(r, dtype=np.float64)
    dy = 1.0
    t = 0
    selected_cols: list[int] = []
    selected_block_ids: list[int] = []
    objective = np.empty(n_discover if want_objective else 0, dtype=np.float64)

    def _commit_block(block_idx: int) -> None:
        nonlocal dy, t
        members = np.asarray(block_members[block_idx], dtype=np.int64)
        for j in members:
            if not remaining[int(j)]:
                continue
            dy = _cefsplus_apply_column(
                int(j),
                R=R,
                L=L,
                Ly=Ly,
                d=d,
                c=c,
                remaining=remaining,
                t=t,
                scale=scale,
                eps=eps,
                dy=dy,
            )
            selected_cols.append(int(j))
            t += 1
        remaining_blocks[block_idx] = False

    for block_idx in forced:
        _commit_block(block_idx)

    count = 0
    while count < n_discover:
        best_idx = -1
        best_gain = -np.inf
        best_rel = -np.inf
        for bidx in range(n_blocks):
            if not remaining_blocks[bidx]:
                continue
            members = np.asarray(block_members[bidx], dtype=np.int64)
            live = members[remaining[members]]
            if live.size == 0:
                remaining_blocks[bidx] = False
                continue
            gain = _cefsplus_block_gain(
                live,
                R=R,
                L=L,
                d=d,
                c=c,
                t=t,
                dy=dy,
                scale=scale,
                shrink=shrink,
                eps=eps,
            )
            rel = float(np.max(tie_break_rel[live]))
            better = False
            if best_idx < 0 or gain > best_gain + 1e-12:
                better = True
            elif abs(gain - best_gain) <= 1e-12:
                if rel > best_rel + 1e-15 or (
                    abs(rel - best_rel) <= 1e-15 and bidx < best_idx
                ):
                    better = True
            if better:
                best_idx = bidx
                best_gain = gain
                best_rel = rel
        if best_idx < 0:
            break
        _commit_block(best_idx)
        selected_block_ids.append(best_idx)
        if want_objective:
            objective[count] = best_gain if count == 0 else objective[count - 1] + best_gain
        count += 1
        if callback is not None:
            report_progress(
                callback,
                count,
                n_discover,
                stage="path",
                selector="cefsplus",
            )

    return (
        np.asarray(selected_cols, dtype=np.int64),
        objective[:count],
        selected_block_ids,
    )


def _gaussian_mrmr_select_blocks(
    R: np.ndarray,
    rel: np.ndarray,
    k: int,
    use_quotient: bool,
    block_members: Sequence[np.ndarray],
    *,
    forced_blocks: Sequence[int] = (),
    eligible_blocks: Sequence[int] | None = None,
    floor: float = 1e-6,
    callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[int]]:
    m = len(rel)
    n_blocks = len(block_members)
    is_sel = np.zeros(m, dtype=bool)
    remaining_blocks = np.ones(n_blocks, dtype=bool)
    if eligible_blocks is not None:
        remaining_blocks[:] = False
        for idx in eligible_blocks:
            remaining_blocks[int(idx)] = True
    for idx in forced_blocks:
        remaining_blocks[int(idx)] = False
        for j in np.asarray(block_members[int(idx)], dtype=np.int64):
            is_sel[int(j)] = True
    n_discover = min(int(k), int(np.sum(remaining_blocks)))
    if n_discover <= 0:
        return np.empty(0, dtype=np.int64), []
    red_sum = np.zeros(m, dtype=np.float64)
    n_pre = int(np.sum(is_sel))
    if n_pre:
        for j_pre in np.flatnonzero(is_sel):
            red = gaussian_mi_from_corr(R[int(j_pre)])
            mask = ~is_sel
            red_sum[mask] += red[mask]
    selected_cols: list[int] = []
    selected_block_ids: list[int] = []
    t0 = n_pre
    for step in range(n_discover):
        best_idx = -1
        best_score = -np.inf
        best_rel = -np.inf
        mean_red = red_sum / max(t0, 1) if t0 else None
        for bidx in range(n_blocks):
            if not remaining_blocks[bidx]:
                continue
            members = np.asarray(block_members[bidx], dtype=np.int64)
            live = members[~is_sel[members]]
            if live.size == 0:
                remaining_blocks[bidx] = False
                continue
            if t0 == 0:
                scores = rel[live]
            elif use_quotient:
                scores = rel[live] / np.maximum(mean_red[live], floor)
            else:
                scores = rel[live] - mean_red[live]
            score = float(np.max(scores))
            rel_b = float(np.max(rel[live]))
            if best_idx < 0 or score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (rel_b > best_rel + 1e-15 or (abs(rel_b - best_rel) <= 1e-15 and bidx < best_idx))
            ):
                best_idx = bidx
                best_score = score
                best_rel = rel_b
        if best_idx < 0 or not np.isfinite(best_score):
            break
        members = np.asarray(block_members[best_idx], dtype=np.int64)
        live = members[~is_sel[members]]
        for j in live:
            selected_cols.append(int(j))
            is_sel[int(j)] = True
            red = gaussian_mi_from_corr(R[int(j)])
            mask = ~is_sel
            red_sum[mask] += red[mask]
            t0 += 1
        remaining_blocks[best_idx] = False
        selected_block_ids.append(best_idx)
        if callback is not None:
            report_progress(
                callback,
                step + 1,
                n_discover,
                stage="path",
                selector="mrmr_quot" if use_quotient else "mrmr_diff",
            )
    return np.asarray(selected_cols, dtype=np.int64), selected_block_ids


def _gaussian_jmi_select_blocks(
    R: np.ndarray,
    r_y: np.ndarray,
    rel: np.ndarray,
    k: int,
    use_min: bool,
    block_members: Sequence[np.ndarray],
    *,
    forced_blocks: Sequence[int] = (),
    eligible_blocks: Sequence[int] | None = None,
    callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[int]]:
    m = len(r_y)
    n_blocks = len(block_members)
    is_sel = np.zeros(m, dtype=bool)
    remaining_blocks = np.ones(n_blocks, dtype=bool)
    if eligible_blocks is not None:
        remaining_blocks[:] = False
        for idx in eligible_blocks:
            remaining_blocks[int(idx)] = True
    scores = np.full(m, np.inf, dtype=np.float64) if use_min else np.zeros(m, dtype=np.float64)
    r2 = np.empty(m, dtype=np.float64)
    frac = np.empty(m, dtype=np.float64)
    eps = 1e-8
    eligible_mask = np.ones(m, dtype=bool)

    def _accumulate_from(last: int) -> None:
        r_ys = float(r_y[last])
        r_fs = R[last]
        denom = 1.0 - r_fs * r_fs
        a = r_y - r_ys * r_fs
        r2.fill(r_ys * r_ys)
        frac.fill(0.0)
        np.divide(a * a, denom, out=frac, where=denom >= eps)
        np.add(r2, frac, out=r2)
        np.clip(r2, 0.0, 0.99999, out=r2)
        mi = -0.5 * np.log(1.0 - r2)
        mask = eligible_mask
        if use_min:
            scores[mask] = np.minimum(scores[mask], mi[mask])
        else:
            scores[mask] += mi[mask]

    for idx in forced_blocks:
        remaining_blocks[int(idx)] = False
        for j in np.asarray(block_members[int(idx)], dtype=np.int64):
            is_sel[int(j)] = True
            eligible_mask[int(j)] = False
            _accumulate_from(int(j))
    n_discover = min(int(k), int(np.sum(remaining_blocks)))
    if n_discover <= 0:
        return np.empty(0, dtype=np.int64), []
    selected_cols: list[int] = []
    selected_block_ids: list[int] = []
    have_selected = bool(np.any(is_sel))
    for step in range(n_discover):
        best_idx = -1
        best_score = -np.inf
        best_rel = -np.inf
        for bidx in range(n_blocks):
            if not remaining_blocks[bidx]:
                continue
            members = np.asarray(block_members[bidx], dtype=np.int64)
            live = members[~is_sel[members]]
            if live.size == 0:
                remaining_blocks[bidx] = False
                continue
            if not have_selected:
                col_scores = rel[live]
            else:
                col_scores = np.where(np.isfinite(scores[live]), scores[live], rel[live])
            score = float(np.max(col_scores))
            rel_b = float(np.max(rel[live]))
            if best_idx < 0 or score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (rel_b > best_rel + 1e-15 or (abs(rel_b - best_rel) <= 1e-15 and bidx < best_idx))
            ):
                best_idx = bidx
                best_score = score
                best_rel = rel_b
        if best_idx < 0 or not np.isfinite(best_score):
            break
        members = np.asarray(block_members[best_idx], dtype=np.int64)
        live = members[~is_sel[members]]
        for j in live:
            selected_cols.append(int(j))
            is_sel[int(j)] = True
            eligible_mask[int(j)] = False
            _accumulate_from(int(j))
            have_selected = True
        remaining_blocks[best_idx] = False
        selected_block_ids.append(best_idx)
        if callback is not None:
            report_progress(
                callback,
                step + 1,
                n_discover,
                stage="path",
                selector="jmim" if use_min else "jmi",
            )
    return np.asarray(selected_cols, dtype=np.int64), selected_block_ids


@njit_optional_cache(cache=True)
def _cefsplus_loop_core(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    want_objective: bool,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy CEFS+ path via a partial Cholesky (residual) recursion.

    Selecting feature ``j`` given the current set ``S`` maximizes
    ``log|Sigma_S+j| - log|Sigma_{y,S+j}|``. With ``d_j`` the residual
    variance of ``x_j`` given ``S``, ``c_j`` the residual covariance of
    ``(x_j, y)`` given ``S`` and ``d_y`` the residual variance of ``y`` given
    ``S``, the two Schur complements are ``s1_j = d_j`` and
    ``s2_j = d_j - c_j^2 / d_y``. Maintaining ``d``, ``c`` and the partial
    Cholesky rows costs O(m * t) per step instead of the O(m * t^2) inverse
    updates, and uses no BLAS calls, so it cannot thrash against the caller's
    BLAS thread pool.
    """
    m = len(r)
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    k = min(k, m)

    scale = 1.0 - shrink
    selected = np.empty(k, dtype=np.int64)
    objective = np.empty(k if want_objective else 0, dtype=np.float64)
    remaining = np.ones(m, dtype=np.bool_)

    # Partial Cholesky rows for the selected columns, residual variances of
    # every candidate, and residual covariances with the (shrunk) target.
    L = np.zeros((m, k), dtype=np.float64)
    Ly = np.zeros(k, dtype=np.float64)
    d = np.ones(m, dtype=np.float64)
    c = np.empty(m, dtype=np.float64)
    for j in range(m):
        c[j] = scale * r[j]
    dy = 1.0

    logdet_S = 0.0
    logdet_yS = 0.0
    score = np.empty(m, dtype=np.float64)
    s1 = np.empty(m, dtype=np.float64)
    s2 = np.empty(m, dtype=np.float64)

    count = 0
    while count < k:
        t = count
        if t == 0:
            j = 0
            best_rel = tie_break_rel[0]
            for jj in range(1, m):
                if tie_break_rel[jj] > best_rel:
                    best_rel = tie_break_rel[jj]
                    j = jj
            s1_best = 1.0
            s2_best = max(1.0 - c[j] * c[j], eps)
        else:
            best_pos = -1
            best_score = -np.inf
            for jj in range(m):
                if not remaining[jj]:
                    continue
                s1_j = max(d[jj], eps)
                s2_j = max(d[jj] - c[jj] * c[jj] / dy, eps)
                s1[jj] = s1_j
                s2[jj] = s2_j
                sc = np.log(s1_j) - np.log(s2_j)
                score[jj] = sc
                if best_pos < 0 or sc > best_score:
                    best_score = sc
                    best_pos = jj
            if best_pos < 0:
                break
            j = best_pos
            best_rel = tie_break_rel[j]
            for jj in range(m):
                if remaining[jj] and np.abs(score[jj] - best_score) < 1e-12:
                    if tie_break_rel[jj] > best_rel:
                        best_rel = tie_break_rel[jj]
                        j = jj
            s1_best = s1[j]
            s2_best = s2[j]

        # Update the residual state for the remaining candidates.
        sq = np.sqrt(s1_best)
        ly = c[j] / sq
        Ly[t] = ly
        dy -= ly * ly
        for i in range(m):
            if not remaining[i] or i == j:
                continue
            acc = R[i, j] * scale
            for a in range(t):
                acc -= L[i, a] * L[j, a]
            lij = acc / sq
            L[i, t] = lij
            d[i] -= lij * lij
            c[i] -= lij * ly

        logdet_S += np.log(s1_best)
        logdet_yS += np.log(s2_best)
        selected[count] = j
        if want_objective:
            objective[count] = logdet_S - logdet_yS
        remaining[j] = False
        count += 1

    return selected[:count], objective[:count]


@njit_optional_cache(cache=True)
def _cefsplus_loop_core_conditioned(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    want_objective: bool,
    forced: np.ndarray,
    eligible: np.ndarray,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """CEFS+ path with a pre-conditioned partial-Cholesky state.

    ``forced`` columns are applied in order to ``L``, ``d``, ``c`` and ``dy``
    and are not returned as discoveries. Greedy steps then run among
    ``eligible`` remaining columns for up to ``k`` additional features.
    """
    m = len(r)
    n_forced = len(forced)
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    k = min(k, int(np.sum(eligible)))
    if k <= 0 and n_forced == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    scale = 1.0 - shrink
    selected = np.empty(k, dtype=np.int64)
    objective = np.empty(k if want_objective else 0, dtype=np.float64)
    remaining = eligible.copy()
    for i in range(n_forced):
        remaining[forced[i]] = True

    total = n_forced + k
    L = np.zeros((m, total), dtype=np.float64)
    Ly = np.zeros(total, dtype=np.float64)
    d = np.ones(m, dtype=np.float64)
    c = np.empty(m, dtype=np.float64)
    for j in range(m):
        c[j] = scale * r[j]
    dy = 1.0

    logdet_S = 0.0
    logdet_yS = 0.0
    score = np.empty(m, dtype=np.float64)
    s1 = np.empty(m, dtype=np.float64)
    s2 = np.empty(m, dtype=np.float64)

    t = 0
    for f in range(n_forced):
        j = int(forced[f])
        if t == 0:
            s1_best = 1.0
            s2_best = max(1.0 - c[j] * c[j], eps)
        else:
            s1_best = max(d[j], eps)
            s2_best = max(d[j] - c[j] * c[j] / dy, eps)
        sq = np.sqrt(s1_best)
        ly = c[j] / sq
        Ly[t] = ly
        dy -= ly * ly
        for i in range(m):
            if not remaining[i] or i == j:
                continue
            acc = R[i, j] * scale
            for a in range(t):
                acc -= L[i, a] * L[j, a]
            lij = acc / sq
            L[i, t] = lij
            d[i] -= lij * lij
            c[i] -= lij * ly
        logdet_S += np.log(s1_best)
        logdet_yS += np.log(s2_best)
        remaining[j] = False
        t += 1

    baseline = logdet_S - logdet_yS
    count = 0
    while count < k:
        if t == 0:
            j = -1
            best_rel = -np.inf
            for jj in range(m):
                if not remaining[jj]:
                    continue
                if j < 0 or tie_break_rel[jj] > best_rel:
                    best_rel = tie_break_rel[jj]
                    j = jj
            if j < 0:
                break
            s1_best = 1.0
            s2_best = max(1.0 - c[j] * c[j], eps)
        else:
            best_pos = -1
            best_score = -np.inf
            for jj in range(m):
                if not remaining[jj]:
                    continue
                s1_j = max(d[jj], eps)
                s2_j = max(d[jj] - c[jj] * c[jj] / dy, eps)
                s1[jj] = s1_j
                s2[jj] = s2_j
                sc = np.log(s1_j) - np.log(s2_j)
                score[jj] = sc
                if best_pos < 0 or sc > best_score:
                    best_score = sc
                    best_pos = jj
            if best_pos < 0:
                break
            j = best_pos
            best_rel = tie_break_rel[j]
            for jj in range(m):
                if remaining[jj] and np.abs(score[jj] - best_score) < 1e-12:
                    if tie_break_rel[jj] > best_rel:
                        best_rel = tie_break_rel[jj]
                        j = jj
            s1_best = s1[j]
            s2_best = s2[j]

        sq = np.sqrt(s1_best)
        ly = c[j] / sq
        Ly[t] = ly
        dy -= ly * ly
        for i in range(m):
            if not remaining[i] or i == j:
                continue
            acc = R[i, j] * scale
            for a in range(t):
                acc -= L[i, a] * L[j, a]
            lij = acc / sq
            L[i, t] = lij
            d[i] -= lij * lij
            c[i] -= lij * ly

        logdet_S += np.log(s1_best)
        logdet_yS += np.log(s2_best)
        selected[count] = j
        if want_objective:
            objective[count] = (logdet_S - logdet_yS) - baseline
        remaining[j] = False
        count += 1
        t += 1

    return selected[:count], objective[:count]


@njit_optional_cache(cache=True)
def _cefsplus_callback_step(
    R: np.ndarray,
    tie_break_rel: np.ndarray,
    scale: float,
    eps: float,
    t: int,
    L: np.ndarray,
    Ly: np.ndarray,
    d: np.ndarray,
    c: np.ndarray,
    remaining: np.ndarray,
    score: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    dy: float,
) -> tuple[int, float, float, float]:
    """Advance one CEFS+ step while keeping the callback in Python space."""
    m = len(c)
    if t == 0:
        j = 0
        best_rel = tie_break_rel[0]
        for jj in range(1, m):
            if tie_break_rel[jj] > best_rel:
                best_rel = tie_break_rel[jj]
                j = jj
        s1_best = 1.0
        s2_best = max(1.0 - c[j] * c[j], eps)
    else:
        best_pos = -1
        best_score = -np.inf
        for jj in range(m):
            if not remaining[jj]:
                continue
            s1_j = max(d[jj], eps)
            s2_j = max(d[jj] - c[jj] * c[jj] / dy, eps)
            s1[jj] = s1_j
            s2[jj] = s2_j
            sc = np.log(s1_j) - np.log(s2_j)
            score[jj] = sc
            if best_pos < 0 or sc > best_score:
                best_score = sc
                best_pos = jj
        if best_pos < 0:
            return -1, dy, 0.0, 0.0
        j = best_pos
        best_rel = tie_break_rel[j]
        for jj in range(m):
            if remaining[jj] and np.abs(score[jj] - best_score) < 1e-12:
                if tie_break_rel[jj] > best_rel:
                    best_rel = tie_break_rel[jj]
                    j = jj
        s1_best = s1[j]
        s2_best = s2[j]

    sq = np.sqrt(s1_best)
    ly = c[j] / sq
    Ly[t] = ly
    dy -= ly * ly
    for i in range(m):
        if not remaining[i] or i == j:
            continue
        acc = R[i, j] * scale
        for a in range(t):
            acc -= L[i, a] * L[j, a]
        lij = acc / sq
        L[i, t] = lij
        d[i] -= lij * lij
        c[i] -= lij * ly

    remaining[j] = False
    return j, dy, np.log(s1_best), np.log(s2_best)


def _cefsplus_loop_with_callback(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    callback: ProgressCallback,
    *,
    want_objective: bool,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the compiled CEFS+ state machine one step at a time for callbacks."""
    m = len(r)
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    k = min(k, m)

    scale = 1.0 - shrink
    selected = np.empty(k, dtype=np.int64)
    objective = np.empty(k if want_objective else 0, dtype=np.float64)
    remaining = np.ones(m, dtype=np.bool_)
    L = np.zeros((m, k), dtype=np.float64)
    Ly = np.zeros(k, dtype=np.float64)
    d = np.ones(m, dtype=np.float64)
    c = scale * np.asarray(r, dtype=np.float64)
    score = np.empty(m, dtype=np.float64)
    s1 = np.empty(m, dtype=np.float64)
    s2 = np.empty(m, dtype=np.float64)
    dy = 1.0
    logdet_S = 0.0
    logdet_yS = 0.0
    count = 0

    while count < k:
        j, dy, log_s1, log_s2 = _cefsplus_callback_step(
            R,
            tie_break_rel,
            scale,
            eps,
            count,
            L,
            Ly,
            d,
            c,
            remaining,
            score,
            s1,
            s2,
            dy,
        )
        if j < 0:
            break
        selected[count] = j
        logdet_S += log_s1
        logdet_yS += log_s2
        if want_objective:
            objective[count] = logdet_S - logdet_yS
        count += 1
        report_progress(
            callback,
            count,
            k,
            stage="path",
            selector="cefsplus",
        )

    return selected[:count], objective[:count]


@njit_optional_cache(cache=True)
def cefsplus_loop(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    CEFS+ greedy selection via log-det updates.

    Maximizes Gaussian MI proxy using efficient Schur complement updates.
    """
    selected, _objective = _cefsplus_loop_core(
        R,
        r,
        k,
        tie_break_rel,
        False,
        shrink,
        eps,
    )
    return selected


@njit_optional_cache(cache=True)
def cefsplus_loop_with_objective(
    R: np.ndarray,
    r: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CEFS+ selection returning both selected indices AND objective path.

    The objective at step t is: log|Σ_S| - log|Σ_{y,S}| = 2 * I(y; S)
    """
    return _cefsplus_loop_core(
        R,
        r,
        k,
        tie_break_rel,
        True,
        shrink,
        eps,
    )


def _conditional_discovery_objective(
    R: np.ndarray,
    r: np.ndarray,
    forced_local: np.ndarray,
    sel_local: np.ndarray,
) -> np.ndarray:
    """Cumulative Gaussian-MI of discoveries given an include prefix."""
    selected = np.asarray(sel_local, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        return np.empty(0, dtype=np.float64)
    forced = np.asarray(forced_local, dtype=np.int64).reshape(-1)
    if forced.size == 0:
        return objective_from_corr_path(
            np.asarray(R)[np.ix_(selected, selected)],
            np.asarray(r)[selected],
        )
    path = np.concatenate([forced, selected])
    full = objective_from_corr_path(
        np.asarray(R)[np.ix_(path, path)],
        np.asarray(r)[path],
    )
    baseline = float(full[forced.size - 1])
    return np.asarray(full[forced.size:], dtype=np.float64) - baseline


def gaussian_noise_floor_mi(p_valid: int, n_eff: float) -> float:
    """Gaussian-MI relevance expected from the strongest of ``p_valid`` null features.

    Under independence a rank-Gaussian correlation is roughly N(0, 1/n_eff), so
    the largest of ``p_valid`` null correlations is about ``sqrt(2 log p / n)``.
    """
    p_eff = max(int(p_valid), 2)
    n = float(n_eff)
    if not np.isfinite(n) or n <= 2.0:
        return float("inf")
    r2 = min(2.0 * np.log(p_eff) / n, 1.0 - 1e-12)
    return float(-0.5 * np.log(1.0 - r2))


def _warn_gaussian_mrmr_noise_floor(panel, sel_local: np.ndarray, method: str) -> None:
    """Warn when Gaussian mRMR admits features whose relevance is at noise level.

    Both mRMR formulas compare an MI relevance against an MI redundancy on the
    same scale, so once the informative features are mutually redundant a
    pure-noise column (tiny relevance, tiny redundancy) can outscore them.
    JMI/JMIM/CEFS+ do not share this failure mode.
    """
    if sel_local.size == 0:
        return
    floor = gaussian_noise_floor_mi(panel.p_valid, panel.n_eff_kish)
    if not np.isfinite(floor):
        return
    below = int(np.sum(np.asarray(panel.rel)[sel_local] < floor))
    if below == 0:
        return
    warnings.warn(
        f"Gaussian mRMR ({method}) selected {below} of {sel_local.size} features whose "
        f"marginal Gaussian-MI relevance is below the noise floor "
        f"({floor:.2e} for {panel.p_valid} candidates and n_eff={panel.n_eff_kish:.0f}). "
        "The mRMR quotient/difference scores let low-relevance, low-redundancy "
        "columns beat mutually redundant informative ones. Consider a smaller "
        "top_m, or the Gaussian JMI/JMIM/CEFS+ paths, which condition on the "
        "selected set instead of penalizing pairwise redundancy.",
        UserWarning,
        stacklevel=3,
    )


def select_cached(
    cache: FeatureCache,
    y,
    k: int,
    method: Literal["cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"] = "cefsplus",
    top_m: Optional[int] = None,
    corr_prune: CorrPrune = "auto",
    return_objective: bool = False,
    return_indices: bool = False,
    warn_noise_floor: bool = True,
    callback: ProgressCallback | None = None,
    return_result: bool = False,
    store_proxies: bool = False,
    include=None,
    exclude=None,
    candidates=None,
    feature_blocks=None,
) -> List[str] | Tuple[List[str], np.ndarray] | Tuple[List[str], List[int]] | Tuple[
    List[str], List[int], np.ndarray
] | "SelectionView":
    """Select features using pre-built cache.

    Runs one greedy Gaussian-copula selection against a target using a
    ``sift.estimators.copula.FeatureCache`` that already holds the
    rank-Gaussian transform of ``X``. Build the cache once with
    ``sift.build_cache``, then call this per 1-D ``y``, or pass a 2-D
    ``y`` of shape ``(n, q)`` with ``method="cefsplus"`` for joint
    multi-output CEFS+. Reusing one cache across separate 1-D calls is not
    joint multi-output support. Only the target-dependent work -- the
    marginal correlations, the candidate panel, and the greedy path -- is
    repeated.  By default it runs CEFS+, screens to ``max(5 * k, 250)``
    candidates, applies no correlation pruning, and returns a plain
    ``list`` of selected feature names.

    ``corr_prune="auto"`` resolves to no pruning for every method. Pass a float to
    opt into marginal-correlation pruning when duplicate suppression is more
    important than retaining possible suppressor pairs. ``warn_noise_floor`` controls
    the Gaussian-mRMR warning about noise-level picks; auto-k path builders
    disable it because they cut the path afterwards.

    Parameters
    ----------
    cache : FeatureCache
        Cache built by ``sift.build_cache`` from the feature matrix.  Its
        structural contract is revalidated on every call, and duplicate
        non-synthetic ``feature_names`` are rejected.
    y : array-like of shape (n_rows_original,) or (n_rows_original, n_targets)
        Numeric target aligned to the matrix the cache was built from, before
        any subsampling: the cache indexes it with ``cache.row_idx``.  Must be
        finite; classification labels have to be encoded numerically first.
        A single column ``(n, 1)`` follows the 1-D path. ``q>=2`` is joint
        CEFS+ and requires ``method="cefsplus"``. Collinear targets whose
        copula correlation has condition number above ``1e6`` are rejected.
    k : int
        Upper bound on the number of features to select.  Must be a positive
        integer -- ``k="auto"`` is not supported here.  Fewer than ``k``
        features come back when the candidate panel is smaller or the greedy
        score stops being finite.
    method : {"cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"}, default "cefsplus"
        Greedy criterion.  ``"cefsplus"`` maximizes the log-determinant
        conditional-information gain; ``"jmi"``/``"jmim"`` aggregate pairwise
        joint information by sum and by minimum; ``"mrmr_quot"``/
        ``"mrmr_diff"`` divide and subtract mean redundancy from relevance.
    top_m : int or None, default None
        Candidate screen: only the ``top_m`` features with the largest absolute
        copula correlation with ``y`` enter the greedy loop.  ``None`` means
        ``max(5 * k, 250)``.  The effective value is raised to at least ``k``
        and clipped to the number of valid cache columns.
    corr_prune : {"auto"}, float or None, default "auto"
        Redundancy prefilter applied to the screened panel.  ``"auto"`` and
        ``None`` both mean no pruning, which keeps suppressor pairs eligible.
        A float in ``(0, 1]`` greedily drops any candidate whose absolute
        correlation with a better-scoring survivor reaches the threshold.
    return_objective : bool, default False
        Also return the cumulative Gaussian-MI objective along the selected
        path.  Cannot be combined with ``return_result=True``.
    return_indices : bool, default False
        Also return the selected positions in the original feature matrix.
        Cannot be combined with ``return_result=True``.
    warn_noise_floor : bool, default True
        Emit the Gaussian-mRMR noise-floor ``UserWarning`` when
        ``method`` is ``"mrmr_quot"`` or ``"mrmr_diff"``.  Auto-k path builders
        pass ``False`` because they truncate the path afterwards and the check
        would fire on features they are about to drop.
    callback : callable or None, default None
        Progress hook ``callback(step, total, info)`` invoked once per
        completed greedy step, with a one-based ``step``.  Exceptions raised
        inside it propagate.  Supplements, never replaces, ``verbose`` logging.
    return_result : bool, default False
        Return a normalized ``sift.SelectionView`` instead of the legacy
        list/tuple forms.  Requires ``cache.feature_names``; rejects
        ``return_objective`` and ``return_indices``, whose information the view
        already carries.
    store_proxies : bool, default False
        Retain the selection-time candidate-by-selected copula correlation
        block on the view so ``view.proxies()`` and ``view.proxies_at()`` can
        report near-duplicate stand-ins for a selected feature.  Requires
        ``return_result=True``.  The block never contains ``X`` or the cache.
        Selected blocks that expand to cache-dropped constant members still
        appear in the selection, but proxy retention then raises rather than
        inventing correlations.
    include : sequence of names or positions, optional
        Conditioning set. The greedy state is initialized from these features
        before step 1. They are not discoveries; ``k`` counts additional
        features. Included names are prepended to the returned list in
        caller order.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    feature_blocks : mapping, {"auto"} or None, default None
        Atomic column groups. A dict maps block labels to member names or
        positions; unlisted columns stay singletons. ``"auto"`` groups
        columns sharing the one-hot prefix ``{block}__{level}`` (double
        underscore) when at least two columns share that prefix; ordinary
        single underscores are not split. ``k`` counts additional blocks
        and selected blocks expand to every raw member column. ``k="auto"``
        is not supported on ``select_cached``; use the public selectors.
        Singleton blocks recover the column selector.

    Returns
    -------
    list or tuple or SelectionView
        With every flag at its default, a ``list`` of selected feature names
        in selection order (a ``list`` of original integer positions when
        ``cache.feature_names`` is ``None``).  The legacy tuple forms are
        ``(features, objective)`` for ``return_objective=True``,
        ``(features, indices)`` for ``return_indices=True``, and
        ``(features, indices, objective)`` for both, where ``indices`` is a
        ``list[int]`` of original column positions and ``objective`` is a
        float64 array of shape ``(n_discoveries,)``.  Without conditioning
        that equals the number of returned features; with ``include`` it is
        the number of additional discoveries, each value being cumulative
        Gaussian MI relative to the include-only baseline.  With
        ``return_result=True``
        a ``sift.SelectionView`` whose ``raw_table`` covers every cached
        input feature and whose ``diagnostics`` carry ``objective`` and
        ``candidate_indices``.

    Raises
    ------
    ValueError
        If ``return_result`` or ``store_proxies`` is not a boolean; if
        ``store_proxies=True`` without ``return_result=True``; if
        ``return_result=True`` is combined with ``return_objective`` or
        ``return_indices``; if ``k`` is not a positive integer or is
        ``"auto"``; if the cache fails its structural or provenance checks or
        carries duplicate feature names; if ``y`` is non-finite or its length
        differs from ``cache.n_rows_original``; if ``corr_prune`` is outside
        ``(0, 1]``; if ``method`` is unknown; if a 2-D ``y`` is passed with a
        method other than ``"cefsplus"`` or with collinear targets; or if
        ``return_result=True`` is requested for a cache without
        ``feature_names``.

    Warns
    -----
    UserWarning
        For ``method="mrmr_quot"`` or ``"mrmr_diff"`` with
        ``warn_noise_floor=True``, when selected features have marginal
        Gaussian-MI relevance below the noise floor expected from the
        strongest of ``p_valid`` null columns.  Both mRMR formulas compare
        relevance against redundancy on one scale, so a low-relevance,
        low-redundancy noise column can outscore mutually redundant
        informative ones; JMI, JMIM and CEFS+ do not share that failure mode.

    See Also
    --------
    sift.build_cache : Build the cache this function consumes.
    sift.FeatureCache : The cache contract and field meanings.
    sift.select_cefsplus : Same objective, starting from ``X`` and ``y``.
    sift.select_fdr : Error-controlled discovery from the same cache.

    Notes
    -----
    CEFS+ maximizes ``log|Sigma_S| - log|Sigma_{y,S}|``, which equals
    ``2 * I(y; S)`` under the fitted Gaussian copula, and is what the returned
    ``objective`` path reports at each step; it is non-decreasing by
    construction.  The greedy step uses a partial Cholesky (residual)
    recursion costing ``O(m * t)`` at step ``t``, so ``O(m * k**2)`` overall
    for ``m`` screened candidates, and calls no BLAS, so it cannot thrash a
    caller's thread pool.  The JMI/JMIM and mRMR paths update one row of the
    candidate correlation matrix per step and cost ``O(m * k)``.  Everything
    downstream of the cache is target-dependent only, which is what makes
    cache reuse across 1-D targets cheap; the cached rows, weights and copula
    transform are fixed and this function never re-derives them. Joint
    multi-output CEFS+ uses the same cache and a rank-1 residual update of
    ``Σ_{Y|S}``.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import build_cache, select_cached
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 8))
    >>> y = X[:, 2] - 2.0 * X[:, 5] + 0.1 * rng.normal(size=300)
    >>> cache = build_cache(X, compute_Rxx=True, subsample=None)
    >>> select_cached(cache, y, k=2)
    ['x5', 'x2']
    >>> names, objective = select_cached(cache, y, k=2, return_objective=True)
    >>> names, bool(objective[1] >= objective[0])
    (['x5', 'x2'], True)
    >>> view = select_cached(cache, y, k=2, return_result=True)
    >>> view.k, view.indices
    (2, [5, 2])
    """
    return _select_cached_impl(
        cache,
        y,
        k,
        method=method,
        top_m=top_m,
        corr_prune=corr_prune,
        return_objective=return_objective,
        return_indices=return_indices,
        warn_noise_floor=warn_noise_floor,
        callback=callback,
        return_result=return_result,
        store_proxies=store_proxies,
        include=include,
        exclude=exclude,
        candidates=candidates,
        feature_blocks=feature_blocks,
        compose_include=True,
    )


def _select_cached_impl(
    cache: FeatureCache,
    y,
    k: int,
    method: Literal["cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"] = "cefsplus",
    top_m: Optional[int] = None,
    corr_prune: CorrPrune = "auto",
    return_objective: bool = False,
    return_indices: bool = False,
    warn_noise_floor: bool = True,
    callback: ProgressCallback | None = None,
    return_result: bool = False,
    store_proxies: bool = False,
    include=None,
    exclude=None,
    candidates=None,
    feature_blocks=None,
    *,
    compose_include: bool = True,
):
    from sift._preprocess import to_numpy, validate_k

    if not isinstance(return_result, (bool, np.bool_)):
        raise ValueError("return_result must be a boolean")
    if not isinstance(store_proxies, (bool, np.bool_)):
        raise ValueError("store_proxies must be a boolean")
    if store_proxies and not return_result:
        raise ValueError("store_proxies=True requires return_result=True")
    if return_result and (return_objective or return_indices):
        raise ValueError(
            "return_result=True cannot be combined with return_objective or "
            "return_indices; the normalized result already carries both"
        )
    k = validate_k(k, allow_auto=False)
    if not isinstance(compose_include, (bool, np.bool_)):
        raise ValueError("compose_include must be a boolean")
    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)
    from sift.selection.cefsplus_multi import as_regression_targets

    y_probe = np.asarray(y)
    n_y_rows = 1 if y_probe.ndim == 0 else int(y_probe.shape[0])
    if n_y_rows != int(cache.n_rows_original):
        raise ValueError(
            f"y has {n_y_rows} rows but cache was built from "
            f"{int(cache.n_rows_original)} rows"
        )
    y_arr, n_y = as_regression_targets(y, int(cache.n_rows_original))
    if n_y == 1 and not np.isfinite(np.asarray(y_arr, dtype=np.float64)).all():
        raise ValueError("y contains non-finite values")
    if n_y >= 2 and method != "cefsplus":
        raise ValueError(
            "2-D y is only supported for method='cefsplus'; "
            f"got method={method!r}"
        )
    cache_names = list(cache.feature_names) if cache.feature_names is not None else [
        f"x{i}" for i in range(int(np.max(cache.valid_cols)) + 1 if len(cache.valid_cols) else 0)
    ]
    named = named_feature_space(
        cache.feature_names,
        synthetic=bool(getattr(cache, "feature_names_are_synthetic", False))
        or cache.feature_names is None,
    )
    resolved = resolve_conditioning(
        include,
        exclude,
        candidates,
        feature_names=cache_names,
        named=named,
        k=k,
    )
    blocks = resolve_feature_blocks(
        feature_blocks,
        feature_names=cache_names,
        named=named,
    )
    require_atomic_conditioning(
        resolved,
        blocks,
        feature_names=cache_names,
    )
    protect_valid = None
    pool_valid = None
    forced_local = np.empty(0, dtype=np.int64)
    block_members_valid = None
    orig_block_ids: list[int] = []
    if resolved is not None:
        protect_valid = map_original_to_valid(
            resolved.include,
            cache.valid_cols,
            feature_names=cache_names,
            label="include",
        )
        pool_valid = map_original_to_valid(
            resolved.discovery,
            cache.valid_cols,
            feature_names=cache_names,
            label="candidates",
            missing="drop",
        )
        if resolved.candidates is not None and pool_valid.size == 0:
            raise ValueError(
                "candidates contains no valid cache columns eligible for discovery"
            )
    if blocks is not None:
        orig_block_ids, block_members_valid = map_blocks_to_valid(
            blocks, cache.valid_cols
        )
        if not block_members_valid:
            raise ValueError(
                "feature_blocks contains no valid cache columns eligible for selection"
            )
    panel = build_candidate_panel(
        cache,
        y_arr,
        k,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
        protect_valid=protect_valid,
        pool_valid=pool_valid,
        block_members=block_members_valid,
    )
    R_cand = panel.R
    r_cand = panel.r
    rel_cand = panel.rel
    n_forced = 0 if protect_valid is None else int(protect_valid.size)
    eligible_local = np.ones(len(panel.cand), dtype=bool)
    if n_forced:
        forced_local = np.arange(n_forced, dtype=np.int64)
        eligible_local[:n_forced] = False
    k_actual = min(k, int(np.sum(eligible_local)))
    use_forced = n_forced > 0
    selected_block_labels: list = []
    use_joint_blocks = False
    panel_blocks: list[np.ndarray] = []
    panel_orig_blocks: list[int] = []
    multi = panel.C is not None and int(panel.n_targets) >= 2
    if blocks is not None and block_members_valid is not None:
        valid_to_panel = {int(v): i for i, v in enumerate(panel.cand)}
        for orig_b, valid_members in zip(orig_block_ids, block_members_valid):
            local = [
                valid_to_panel[int(v)]
                for v in valid_members
                if int(v) in valid_to_panel
            ]
            if local:
                panel_blocks.append(np.asarray(local, dtype=np.int64))
                panel_orig_blocks.append(int(orig_b))
        use_joint_blocks = any(len(group) > 1 for group in panel_blocks)
        include_orig = set(int(i) for i in (resolved.include if resolved is not None else ()))
        forced_blocks = [
            bidx
            for bidx, orig_b in enumerate(panel_orig_blocks)
            if any(int(col) in include_orig for col in blocks.members[orig_b])
        ]
        eligible_blocks = [
            bidx for bidx in range(len(panel_blocks)) if bidx not in set(forced_blocks)
        ]
        k_actual = min(k, len(eligible_blocks))

    objective = None
    target_condition = panel.target_condition
    if use_joint_blocks:
        if method == "cefsplus" and multi:
            from sift.selection.cefsplus_multi import cefsplus_block_loop_multi

            sel_local, objective, picked, target_condition = cefsplus_block_loop_multi(
                R_cand,
                panel.C,
                panel.Ryy,
                k_actual,
                rel_cand,
                panel_blocks,
                forced_blocks=forced_blocks,
                eligible_blocks=eligible_blocks,
                want_objective=return_objective or return_result,
                callback=callback,
            )
        elif method == "cefsplus":
            sel_local, objective, picked = cefsplus_block_loop(
                R_cand,
                r_cand,
                k_actual,
                rel_cand,
                panel_blocks,
                forced_blocks=forced_blocks,
                eligible_blocks=eligible_blocks,
                want_objective=return_objective or return_result,
                callback=callback,
            )
        elif method in ("mrmr_quot", "mrmr_diff"):
            sel_local, picked = _gaussian_mrmr_select_blocks(
                R_cand,
                rel_cand,
                k_actual,
                use_quotient=method == "mrmr_quot",
                block_members=panel_blocks,
                forced_blocks=forced_blocks,
                eligible_blocks=eligible_blocks,
                callback=callback,
            )
            if warn_noise_floor:
                _warn_gaussian_mrmr_noise_floor(panel, sel_local, method)
        elif method in ("jmi", "jmim"):
            sel_local, picked = _gaussian_jmi_select_blocks(
                R_cand,
                r_cand,
                rel_cand,
                k_actual,
                use_min=method == "jmim",
                block_members=panel_blocks,
                forced_blocks=forced_blocks,
                eligible_blocks=eligible_blocks,
                callback=callback,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        discovered_block_ids = [panel_orig_blocks[int(i)] for i in picked]
        selected_block_labels = [blocks.block_ids[i] for i in discovered_block_ids]
        discovered_original = np.asarray(
            blocks.expand(discovered_block_ids), dtype=np.int64
        )
    elif method == "cefsplus" and multi:
        from sift.selection.cefsplus_multi import cefsplus_multi_loop

        sel_local, objective, target_condition = cefsplus_multi_loop(
            R_cand,
            panel.C,
            panel.Ryy,
            k_actual,
            rel_cand,
            forced=forced_local if use_forced else None,
            eligible=eligible_local if use_forced else None,
            want_objective=return_objective or return_result,
            callback=callback,
        )
    elif method == "cefsplus":
        if use_forced:
            want_objective = bool(return_objective or return_result)
            sel_local, cond_objective = _cefsplus_loop_core_conditioned(
                R_cand,
                r_cand,
                k_actual,
                rel_cand,
                want_objective,
                forced_local,
                eligible_local,
            )
            if want_objective:
                objective = cond_objective
            if callback is not None:
                for step in range(1, len(sel_local) + 1):
                    report_progress(
                        callback,
                        step,
                        k_actual,
                        stage="path",
                        selector="cefsplus",
                    )
        elif callback is not None:
            sel_local, callback_objective = _cefsplus_loop_with_callback(
                R_cand,
                r_cand,
                k_actual,
                rel_cand,
                callback,
                want_objective=return_objective or return_result,
            )
            if return_objective or return_result:
                objective = callback_objective
        elif return_objective or return_result:
            sel_local, objective = cefsplus_loop_with_objective(R_cand, r_cand, k_actual, rel_cand)
        else:
            sel_local = cefsplus_loop(R_cand, r_cand, k_actual, rel_cand)
    elif method in ("mrmr_quot", "mrmr_diff"):
        sel_local = _gaussian_mrmr_select(
            R_cand,
            rel_cand,
            k_actual,
            use_quotient=method == "mrmr_quot",
            callback=callback,
            preselected=forced_local if use_forced else None,
            eligible=eligible_local if use_forced else None,
        )
        if warn_noise_floor:
            _warn_gaussian_mrmr_noise_floor(panel, sel_local, method)
    elif method in ("jmi", "jmim"):
        sel_local = _gaussian_jmi_select(
            R_cand,
            r_cand,
            rel_cand,
            k_actual,
            use_min=method == "jmim",
            callback=callback,
            preselected=forced_local if use_forced else None,
            eligible=eligible_local if use_forced else None,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if not use_joint_blocks:
        discovered_original = panel.original[sel_local].astype(np.int64, copy=False)
        if blocks is not None and discovered_original.size:
            discovered_block_ids = []
            seen_b: set[int] = set()
            for col in discovered_original.tolist():
                bidx = blocks.column_to_block[int(col)]
                if bidx not in seen_b:
                    seen_b.add(bidx)
                    discovered_block_ids.append(bidx)
            discovered_original = np.asarray(
                blocks.expand(discovered_block_ids), dtype=np.int64
            )
            selected_block_labels = [blocks.block_ids[i] for i in discovered_block_ids]
    if compose_include and resolved is not None and resolved.include:
        _composed_names, composed_idx = compose_selected(
            cache_names,
            resolved.include,
            discovered_original.tolist(),
        )
        selected_original = np.asarray(composed_idx, dtype=np.int64)
    else:
        selected_original = discovered_original
    if cache.feature_names is not None:
        out = [cache.feature_names[i] for i in selected_original]
    else:
        out = selected_original.tolist()

    if return_result:
        if objective is None:
            if multi:
                objective = np.empty(0, dtype=np.float64)
            else:
                objective = _conditional_discovery_objective(
                    R_cand, r_cand, forced_local, sel_local
                )
        if cache.feature_names is None:
            raise ValueError(
                "return_result=True requires cache.feature_names so the normalized "
                "view can prove the complete input feature identity"
            )
        feature_names = list(cache.feature_names)
        selected_indices = selected_original.astype(np.int64).tolist()
        selected_rank = {
            position: rank
            for rank, position in enumerate(selected_indices, start=1)
        }
        rank = pd.array(
            [selected_rank.get(position, pd.NA) for position in range(len(feature_names))],
            dtype="Int64",
        )
        relevance = np.full(len(feature_names), np.nan, dtype=np.float64)
        relevance[np.asarray(panel.original, dtype=np.int64)] = np.asarray(
            panel.rel,
            dtype=np.float64,
        )
        ranking_data = {
            "feature": feature_names,
            "rank": rank,
            "selected": [position in selected_rank for position in range(len(feature_names))],
            "selected_index": pd.array(
                range(len(feature_names)),
                dtype="Int64",
            ),
            "relevance": relevance,
            "selector": f"cached_{method}",
        }
        if blocks is not None:
            ranking_data["block_id"] = [
                blocks.block_ids[blocks.column_to_block[position]]
                for position in range(len(feature_names))
            ]
        ranking = pd.DataFrame(ranking_data)
        from sift.selection.blocks import block_result_metadata
        from sift.selection.result import FilterSelectionResult, build_selector_metadata
        from sift.selection.view import as_result

        cond_record = conditioning_record(
            resolved,
            feature_names=feature_names,
            discovered_idx=discovered_original.tolist(),
        )
        extra = {
            "cache_backed": True,
            "method": method,
            "corr_prune": corr_prune,
            "n_rows_original": int(cache.n_rows_original),
            "n_rows_cached": int(len(cache.row_idx)),
            "feature_names_are_synthetic": bool(cache.feature_names_are_synthetic),
        }
        if int(panel.n_targets) >= 2:
            from sift.selection.cefsplus_multi import result_target_metadata

            extra.update(
                result_target_metadata(
                    int(panel.n_targets),
                    target_condition=panel.target_condition,
                )
            )
        diagnostics = {
            "objective": np.asarray(objective, dtype=np.float64).copy(),
            "candidate_indices": np.asarray(
                panel.original,
                dtype=np.int64,
            ).copy(),
        }
        if blocks is not None:
            include_idx = resolved.include if resolved is not None else ()
            extra.update(
                block_result_metadata(
                    blocks,
                    selected_indices,
                    include_idx,
                    n_columns_selected=len(selected_indices),
                )
            )
            include_block_ids = []
            if include_idx:
                include_block_ids = list(
                    dict.fromkeys(
                        blocks.column_to_block[int(i)] for i in include_idx
                    )
                )
            diagnostics["selected_blocks"] = [
                blocks.block_ids[i] for i in include_block_ids
            ] + list(selected_block_labels)
            diagnostics["block_path"] = list(selected_block_labels)
        if cond_record is not None:
            extra["conditioning"] = cond_record
            diagnostics["conditioning"] = cond_record
        result = FilterSelectionResult(
            selected_features=list(out),
            selected_indices=selected_indices,
            selector_metadata=build_selector_metadata(
                f"cached_{method}",
                k=len(out),
                k_requested=k,
                top_m=top_m,
                n_features=len(feature_names),
                auto_k=False,
                extra=extra,
            ),
            ranking_=ranking,
            diagnostics_=diagnostics,
        )
        if store_proxies:
            reject_unavailable_proxy_positions(
                selected_indices,
                available_original=cache.valid_cols,
                feature_names=feature_names,
            )
            proxy_correlations = proxy_frame_from_panel(
                panel.R,
                candidate_indices=panel.original,
                selected_indices=selected_indices,
            )
            object.__setattr__(
                result,
                _PROXY_CORRELATIONS_ATTR,
                proxy_correlations,
            )
        return as_result(result, input_features=feature_names)

    if return_objective:
        if objective is None:
            objective = _conditional_discovery_objective(
                R_cand, r_cand, forced_local, sel_local
            )
        if return_indices:
            return out, selected_original.tolist(), objective
        return out, objective

    if return_indices:
        return out, selected_original.tolist()

    return out
