"""CEFS+ selection using log-det Schur complement updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple
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
from sift.selection.proxies import proxy_frame_from_panel
from sift.selection.result import _PROXY_CORRELATIONS_ATTR
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
    if callback is not None:
        report_progress(
            callback,
            count,
            k,
            stage="path",
            selector="jmim" if use_min else "jmi",
        )

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
        if callback is not None:
            report_progress(
                callback,
                count,
                k,
                stage="path",
                selector="jmim" if use_min else "jmi",
            )

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
) -> List[str] | Tuple[List[str], np.ndarray] | Tuple[List[str], List[int]] | Tuple[
    List[str], List[int], np.ndarray
] | "SelectionView":
    """Select features using pre-built cache.

    Runs one greedy Gaussian-copula selection against a target using a
    ``sift.estimators.copula.FeatureCache`` that already holds the
    rank-Gaussian transform.  This is the entry point for multi-target work:
    build the cache once with ``sift.build_cache``, then call this per
    ``y``.  Only the target-dependent work -- the marginal correlations, the
    candidate panel, and the greedy path -- is repeated.  By default it runs
    CEFS+, screens to ``max(5 * k, 250)`` candidates, applies no correlation
    pruning, and returns a plain ``list`` of selected feature names.

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
    y : array-like of shape (n_rows_original,)
        Numeric target aligned to the matrix the cache was built from, before
        any subsampling: the cache indexes it with ``cache.row_idx``.  Must be
        finite; classification labels have to be encoded numerically first.
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
        float64 array of shape ``(n_selected,)``.  With ``return_result=True``
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
        ``(0, 1]``; if ``method`` is unknown; or if ``return_result=True`` is
        requested for a cache without ``feature_names``.

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
    cache reuse across targets cheap; the cached rows, weights and copula
    transform are fixed and this function never re-derives them.

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
        if callback is not None:
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
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    selected_original = panel.original[sel_local]

    if cache.feature_names is not None:
        out = [cache.feature_names[i] for i in selected_original]
    else:
        out = selected_original.tolist()

    if return_result:
        if objective is None:
            R_path = R_cand[np.ix_(sel_local, sel_local)]
            r_path = r_cand[sel_local]
            objective = objective_from_corr_path(R_path, r_path)
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
        ranking = pd.DataFrame(
            {
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
        )
        from sift.selection.result import FilterSelectionResult, build_selector_metadata
        from sift.selection.view import as_result

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
                extra={
                    "cache_backed": True,
                    "method": method,
                    "corr_prune": corr_prune,
                    "n_rows_original": int(cache.n_rows_original),
                    "n_rows_cached": int(len(cache.row_idx)),
                },
            ),
            ranking_=ranking,
            diagnostics_={
                "objective": np.asarray(objective, dtype=np.float64).copy(),
                "candidate_indices": np.asarray(
                    panel.original,
                    dtype=np.int64,
                ).copy(),
            },
        )
        if store_proxies:
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
            R_path = R_cand[np.ix_(sel_local, sel_local)]
            r_path = r_cand[sel_local]
            objective = objective_from_corr_path(R_path, r_path)
        if return_indices:
            return out, selected_original.tolist(), objective
        return out, objective

    if return_indices:
        return out, selected_original.tolist()

    return out
