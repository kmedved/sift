"""CEFS+ selection using log-det Schur complement updates."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple
import warnings

import numpy as np

from sift._numba import njit_optional_cache
from sift.estimators.copula import (
    FeatureCache,
    gaussian_mi_from_corr,
)
from sift.selection.objective import objective_from_corr_path
from sift.selection.panel import build_candidate_panel
from sift.selection.knockoff_filter import (
    _reject_duplicate_feature_names,
    _validate_prebuilt_cache_structure,
)

CorrPrune = float | None | Literal["auto"]


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
) -> List[str] | Tuple[List[str], np.ndarray] | Tuple[List[str], List[int]] | Tuple[
    List[str], List[int], np.ndarray
]:
    """Select features using pre-built cache.

    corr_prune="auto" resolves to no pruning for every method. Pass a float to
    opt into marginal-correlation pruning when duplicate suppression is more
    important than retaining possible suppressor pairs. ``warn_noise_floor`` controls
    the Gaussian-mRMR warning about noise-level picks; auto-k path builders
    disable it because they cut the path afterwards.
    """
    from sift._preprocess import to_numpy, validate_k

    k = validate_k(k, allow_auto=False)
    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)
    y_arr = to_numpy(y, dtype=np.float64).ravel()
    if not np.isfinite(y_arr).all():
        raise ValueError("y contains non-finite values")
    panel = build_candidate_panel(
        cache,
        y_arr,
        k,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
    )
    R_cand = panel.R
    r_cand = panel.r
    rel_cand = panel.rel

    k_actual = min(k, len(panel.cand))

    objective = None
    if method == "cefsplus":
        if return_objective:
            sel_local, objective = cefsplus_loop_with_objective(R_cand, r_cand, k_actual, rel_cand)
        else:
            sel_local = cefsplus_loop(R_cand, r_cand, k_actual, rel_cand)
    elif method in ("mrmr_quot", "mrmr_diff"):
        sel_local = _gaussian_mrmr_select(
            R_cand,
            rel_cand,
            k_actual,
            use_quotient=method == "mrmr_quot",
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
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    selected_original = panel.original[sel_local]

    if cache.feature_names is not None:
        out = [cache.feature_names[i] for i in selected_original]
    else:
        out = selected_original.tolist()

    if return_objective:
        if objective is None:
            R_path = R_cand[np.ix_(sel_local, sel_local)]
            r_path = r_cand[sel_local]
            objective = objective_from_corr_path(R_path, r_path)
        if return_indices:
            return out, selected_original.tolist(), objective
        return out, objective

    if return_indices:
        return out, selected_original.tolist()

    return out
