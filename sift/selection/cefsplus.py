"""CEFS+ selection using log-det Schur complement updates."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
from numba import njit

from sift.estimators.copula import (
    FeatureCache,
    build_cache,
    corr_with_vector,
    correlation_matrix_fast,
    gaussian_mi_from_corr,
    greedy_corr_prune,
    rank_gauss_1d,
)


def _gaussian_mrmr_select(
    R: np.ndarray,
    rel: np.ndarray,
    k: int,
    use_quotient: bool,
    floor: float = 1e-6,
) -> np.ndarray:
    m = len(rel)
    k = min(k, m)
    selected = np.empty(k, dtype=np.int64)
    is_sel = np.zeros(m, dtype=bool)
    red_sum = np.zeros(m, dtype=np.float64)

    j0 = int(np.argmax(rel))
    selected[0] = j0
    is_sel[j0] = True
    count = 1

    for t in range(1, k):
        last = selected[t - 1]
        red = gaussian_mi_from_corr(R[last])
        mask = ~is_sel
        red_sum[mask] += red[mask]

        mean_red = red_sum / t
        if use_quotient:
            score = rel / np.maximum(mean_red, floor)
        else:
            score = rel - mean_red

        score[is_sel] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break

        selected[t] = j
        is_sel[j] = True
        count += 1

    return selected[:count]


def _gaussian_jmi_select(
    R: np.ndarray,
    r_y: np.ndarray,
    rel: np.ndarray,
    k: int,
    use_min: bool,
) -> np.ndarray:
    m = len(r_y)
    k = min(k, m)
    selected = np.empty(k, dtype=np.int64)
    is_sel = np.zeros(m, dtype=bool)
    scores = np.full(m, np.inf, dtype=np.float64) if use_min else np.zeros(m, dtype=np.float64)

    j0 = int(np.argmax(rel))
    selected[0] = j0
    is_sel[j0] = True
    count = 1

    # Scratch buffers to avoid per-iteration allocations.
    r2 = np.empty(m, dtype=np.float64)
    frac = np.empty(m, dtype=np.float64)
    eps = 1e-8

    for t in range(1, k):
        last = selected[t - 1]
        r_ys = float(r_y[last])

        # Use row access (contiguous) rather than column access (strided).
        r_fs = R[last]
        denom = 1.0 - r_fs * r_fs
        a = r_y - r_ys * r_fs
        # Match the original scalar fallback exactly, but without np.where() eager
        # evaluation (which can emit divide-by-zero warnings):
        #   if denom < eps: r2 = r_ys^2
        #   else:          r2 = r_ys^2 + a^2 / denom
        r2.fill(r_ys * r_ys)
        frac.fill(0.0)
        np.divide(a * a, denom, out=frac, where=denom >= eps)
        r2 += frac
        np.clip(r2, 0.0, 0.99999, out=r2)
        mi = -0.5 * np.log(1.0 - r2)

        mask = ~is_sel
        if use_min:
            scores[mask] = np.minimum(scores[mask], mi[mask])
        else:
            scores[mask] += mi[mask]

        scores[is_sel] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]):
            break

        selected[t] = j
        is_sel[j] = True
        count += 1

    return selected[:count]


@njit(cache=True)
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
    m = len(r)
    if k <= 0 or m == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, m)

    R_shrunk = (1.0 - shrink) * R.copy()
    for i in range(m):
        R_shrunk[i, i] = 1.0
    r_shrunk = (1.0 - shrink) * r.copy()

    selected = np.empty(k, dtype=np.int64)
    remaining = np.ones(m, dtype=np.bool_)

    j0 = 0
    best_rel = tie_break_rel[0]
    for j in range(1, m):
        if tie_break_rel[j] > best_rel:
            best_rel = tie_break_rel[j]
            j0 = j

    selected[0] = j0
    remaining[j0] = False
    count = 1

    inv_S = np.array([[1.0]], dtype=np.float64)
    logdet_S = 0.0

    r0 = r_shrunk[j0]
    det_yS = max(1.0 - r0 * r0, eps)
    inv_yS = (1.0 / det_yS) * np.array([[1.0, -r0], [-r0, 1.0]], dtype=np.float64)
    logdet_yS = np.log(det_yS)

    while count < k:
        n_rem = 0
        for j in range(m):
            if remaining[j]:
                n_rem += 1
        if n_rem == 0:
            break

        rem = np.empty(n_rem, dtype=np.int64)
        idx = 0
        for j in range(m):
            if remaining[j]:
                rem[idx] = j
                idx += 1

        s = count

        B = np.empty((s, n_rem), dtype=np.float64)
        for si in range(s):
            for ri in range(n_rem):
                B[si, ri] = R_shrunk[selected[si], rem[ri]]

        tmp = inv_S @ B
        t1 = np.zeros(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            for si in range(s):
                t1[ri] += B[si, ri] * tmp[si, ri]

        s1 = np.empty(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            s1[ri] = max(1.0 - t1[ri], eps)

        lf = np.empty(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            lf[ri] = logdet_S + np.log(s1[ri])

        B2 = np.empty((s + 1, n_rem), dtype=np.float64)
        for ri in range(n_rem):
            B2[0, ri] = r_shrunk[rem[ri]]
        for si in range(s):
            for ri in range(n_rem):
                B2[si + 1, ri] = B[si, ri]

        tmp2 = inv_yS @ B2
        t2 = np.zeros(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            for si in range(s + 1):
                t2[ri] += B2[si, ri] * tmp2[si, ri]

        s2 = np.empty(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            s2[ri] = max(1.0 - t2[ri], eps)

        lc = np.empty(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            lc[ri] = logdet_yS + np.log(s2[ri])

        score = np.empty(n_rem, dtype=np.float64)
        for ri in range(n_rem):
            score[ri] = lf[ri] - lc[ri]

        best_pos = 0
        best_score = score[0]
        for ri in range(1, n_rem):
            if score[ri] > best_score:
                best_score = score[ri]
                best_pos = ri

        best_rel = tie_break_rel[rem[best_pos]]
        for ri in range(n_rem):
            if np.abs(score[ri] - best_score) < 1e-12:
                if tie_break_rel[rem[ri]] > best_rel:
                    best_rel = tie_break_rel[rem[ri]]
                    best_pos = ri

        j = rem[best_pos]

        b = B[:, best_pos].copy().reshape(-1, 1)
        v = inv_S @ b
        s1_best = s1[best_pos]

        inv_S_new = np.empty((s + 1, s + 1), dtype=np.float64)
        for i in range(s):
            for jj in range(s):
                inv_S_new[i, jj] = inv_S[i, jj] + v[i, 0] * v[jj, 0] / s1_best
        for i in range(s):
            inv_S_new[i, s] = -v[i, 0] / s1_best
            inv_S_new[s, i] = -v[i, 0] / s1_best
        inv_S_new[s, s] = 1.0 / s1_best
        inv_S = inv_S_new
        logdet_S += np.log(s1_best)

        b2 = B2[:, best_pos].copy().reshape(-1, 1)
        v2 = inv_yS @ b2
        s2_best = s2[best_pos]

        inv_yS_new = np.empty((s + 2, s + 2), dtype=np.float64)
        for i in range(s + 1):
            for jj in range(s + 1):
                inv_yS_new[i, jj] = inv_yS[i, jj] + v2[i, 0] * v2[jj, 0] / s2_best
        for i in range(s + 1):
            inv_yS_new[i, s + 1] = -v2[i, 0] / s2_best
            inv_yS_new[s + 1, i] = -v2[i, 0] / s2_best
        inv_yS_new[s + 1, s + 1] = 1.0 / s2_best
        inv_yS = inv_yS_new
        logdet_yS += np.log(s2_best)

        selected[count] = j
        remaining[j] = False
        count += 1

    return selected[:count]


def select_cached(
    cache: FeatureCache,
    y,
    k: int,
    method: Literal["cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"] = "cefsplus",
    top_m: Optional[int] = None,
    corr_prune: float = 0.95,
) -> List[str]:
    """Select features using pre-built cache."""
    from sift._preprocess import to_numpy

    y_arr = to_numpy(y, dtype=np.float32).ravel()
    ys = y_arr[cache.row_idx]
    zy = rank_gauss_1d(ys)

    r = corr_with_vector(cache.Z, zy)
    rel = gaussian_mi_from_corr(r)

    p_valid = len(r)
    if top_m is None:
        top_m = max(5 * k, 250)
    top_m = max(int(top_m), int(k))
    top_m = min(top_m, p_valid)

    if top_m < p_valid:
        cand = np.argpartition(np.abs(r), -top_m)[-top_m:]
    else:
        cand = np.arange(p_valid)

    Z_cand = np.ascontiguousarray(cache.Z[:, cand], dtype=np.float64)
    R_cand = correlation_matrix_fast(Z_cand)

    keep = greedy_corr_prune(np.arange(len(cand)), R_cand, np.abs(r[cand]), corr_prune)
    cand = cand[keep]
    R_cand = np.ascontiguousarray(R_cand[np.ix_(keep, keep)])
    r_cand = r[cand].astype(np.float64)
    rel_cand = rel[cand].astype(np.float64)

    k_actual = min(k, len(cand))

    if method == "cefsplus":
        sel_local = cefsplus_loop(R_cand, r_cand, k_actual, rel_cand)
    elif method in ("mrmr_quot", "mrmr_diff"):
        sel_local = _gaussian_mrmr_select(
            R_cand,
            rel_cand,
            k_actual,
            use_quotient=method == "mrmr_quot",
        )
    elif method in ("jmi", "jmim"):
        sel_local = _gaussian_jmi_select(
            R_cand,
            r_cand,
            rel_cand,
            k_actual,
            use_min=method == "jmim",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    selected_valid = cand[sel_local]
    selected_original = cache.valid_cols[selected_valid]

    if cache.feature_names is not None:
        return [cache.feature_names[i] for i in selected_original]
    return selected_original.tolist()
