"""Joint multi-target CEFS+ on the Gaussian copula.

The scalar CEFS+ step maximizes ``log|Σ_{S∪j}| - log|Σ_{y,S∪j}|``, which
equals ``-log(1 - c_j² / (d_j d_y))`` and is twice the Gaussian MI. For a
``q``-dimensional copula target ``Y`` the same identity is

    ``-log|Σ_{Y|S∪j}| + log|Σ_{Y|S}| = -log(1 - c_jᵀ Σ_{Y|S}^{-1} c_j / d_j)``

with ``d_j = var(x_j|S)`` and ``c_j = cov(Y, x_j|S)``. The public F5 formula
is half of that quantity (true MI); greedy ordering is unchanged by the
factor of one half, and the stored path objective keeps the historical
``2 I(Y; S)`` convention so one-column ``y`` matches the 1-D loop.

Residual state is a rank-1 update: after selecting ``j``,

    ``u = c_j / √d_j``,  ``Σ_{Y|S} ← Σ_{Y|S} - u uᵀ``,

and remaining feature residuals are downdated exactly as in scalar CEFS+.

``Σ_Y`` uses the same off-diagonal shrink as the feature Gram for the path.
The collinearity guard is applied to the *unshrunk* target correlation:
the copula Gram clips ``|r|`` at ``0.999999``, so duplicate targets have
``cond(Ryy)≈2e6``. ``TARGET_CONDITION_CAP=1e6`` therefore rejects copies
while still allowing ordinary dependence. Targets above the cap are
rejected (drop or combine them). During the
path, ``Σ_{Y|S}`` is factored with the same Cholesky-plus-ridge guard as
the feature-side log-det.

Information-criterion degrees of freedom: a shared design of ``k`` copula
features and ``q`` responses is multivariate linear regression with ``k q``
mean parameters. Residual-covariance parameters do not depend on ``k`` and
cancel in pairwise model comparison, so they are not part of the k-penalty.
The search is over feature subsets of size ``k``, so EBIC still uses
``log C(p, k)`` once (not ``q`` independent searches). The likelihood
dimension is therefore ``df = q·k``, passed through ``ic_dimension='df'``.
Summing ``q`` independent-target EBICs on the *same* ``S`` would charge
``q k log n`` for the mean parameters *and* multiply the combinatorial
term by ``q``; the extra multiplicity is wrong for a joint path. Using
``df = k`` under-penalizes the likelihood. This rule is used for the
penalized-objective / measured-auto CEFS+ route after a shared-signal
synthetic check; it is not assumed as ``q*k`` in the search index, and
it is not a default for methods that do not take a degrees-of-freedom path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.linalg import solve_triangular

from sift._progress import ProgressCallback, report_progress
from sift.selection.cefsplus import _block_residual_cov, _chol_logdet


TARGET_CONDITION_CAP = 1e6
_SHRINK = 1e-6
_EPS = 1e-12
SUPPORTED_MULTI_TARGET_AUTO_K = frozenset(
    {"auto", "evaluate", "elbow", "penalized_objective"}
)
SUPPORTED_MULTI_TARGET_CAT_ENCODING = frozenset(
    {"none", "onehot", "ordinal", "frequency"}
)
_MULTI_TARGET_AUTO_K_HELP = (
    "evaluate, elbow, penalized_objective, or k='auto' when the measured "
    "router selects EBIC"
)


def as_regression_targets(y, n_rows: int) -> tuple[np.ndarray, int]:
    """Return ``(array, q)`` with ``q == 1`` as a 1-D vector.

    A single column ``(n, 1)`` is treated as the scalar path so one-column
    targets stay bit-identical to 1-D ``y``.
    """
    arr = np.asarray(y, dtype=np.float64)
    if isinstance(y, np.ndarray) and arr is y:
        arr = np.array(arr, dtype=np.float64, copy=True)
    if arr.ndim == 0:
        raise ValueError("y must be a one-dimensional or two-dimensional array")
    if arr.ndim == 1:
        if int(arr.shape[0]) != int(n_rows):
            raise ValueError(f"X has {n_rows} rows but y has {int(arr.shape[0])}")
        return arr, 1
    if arr.ndim != 2:
        raise ValueError("y must be a one-dimensional or two-dimensional array")
    if int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"X has {n_rows} rows but y has {int(arr.shape[0])}")
    if int(arr.shape[1]) < 1:
        raise ValueError("y must contain at least one target column")
    if int(arr.shape[1]) == 1:
        return arr.reshape(-1), 1
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite values in y are not allowed for regression.")
    return np.ascontiguousarray(arr, dtype=np.float64), int(arr.shape[1])


def as_selector_targets(
    y, n_rows: int, *, task: str = "regression"
) -> tuple[np.ndarray, int]:
    """Normalize ``y`` for filter/auto-k scoring without breaking 1-D labels.

    Regression uses ``as_regression_targets``. Classification keeps the
    historical 1-D label dtypes (including strings); a genuine multi-output
    classification target is rejected rather than flattened.
    """
    if task != "classification":
        return as_regression_targets(y, n_rows)
    arr = np.asarray(y)
    if arr.ndim == 2 and int(arr.shape[1]) > 1:
        raise ValueError(
            "2-D y is not supported for classification; joint multi-target "
            "CEFS+ is regression-only. Pass a 1-D label vector."
        )
    if arr.ndim == 2 and int(arr.shape[1]) == 1:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError("y must be a one-dimensional or two-dimensional array")
    if int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"X has {n_rows} rows but y has {int(arr.shape[0])}")
    return arr, 1


def multivariate_ic_df(k_path: np.ndarray | Sequence[int], n_targets: int) -> np.ndarray:
    """Likelihood degrees of freedom ``q·k`` for a shared multi-output path.

    ``k_path`` is the search-step / usable-design dimension already used in
    the scalar path (column count, or copula rank of a block prefix). The
    combinatorial EBIC term stays ``log C(p, k)``; only this likelihood
    dimension is scaled by ``q``.
    """
    k = np.asarray(k_path, dtype=np.float64).reshape(-1)
    q = int(n_targets)
    if q < 1:
        raise ValueError("n_targets must be >= 1")
    return q * k


def copula_target_condition(cache, y) -> tuple[int, float | None]:
    """Return ``(q, cond(Ryy))`` for a cache-aligned target; ``cond`` is None if ``q==1``."""
    from sift.estimators.copula import weighted_correlation_matrix, weighted_rank_gauss_2d

    y_mat, n_targets = as_regression_targets(y, int(cache.n_rows_original))
    if n_targets < 2:
        return n_targets, None
    ys = np.asarray(y_mat, dtype=np.float64)[np.asarray(cache.row_idx)]
    zy = weighted_rank_gauss_2d(ys, cache.sample_weight)
    reject_degenerate_multi_targets(zy, cache.sample_weight)
    Ryy = np.asarray(
        weighted_correlation_matrix(zy, cache.sample_weight, backend="blas"),
        dtype=np.float64,
    )
    cond = reject_ill_conditioned_targets(Ryy)
    return n_targets, cond


def result_target_metadata(
    n_targets: int,
    *,
    target_condition: float | None = None,
) -> dict:
    """Selector-metadata extras for 1-D and joint multi-target results."""
    extra = {"n_targets": int(n_targets)}
    if int(n_targets) >= 2:
        extra["target_condition_cap"] = float(TARGET_CONDITION_CAP)
        extra["target_condition_number"] = target_condition
        extra["ic_df_rule"] = "q_k"
    return extra


def reject_unsupported_multi_target_context(
    *,
    n_targets: int,
    selector: str = "cefsplus",
    method: str = "cefsplus",
    within=None,
    cat_encoding: str | None = "none",
    k_method: str | None = None,
    routed: bool = False,
) -> None:
    """Reject 2-D ``y`` combinations that are not joint CEFS+."""
    if int(n_targets) < 2:
        return
    if selector not in {"cefsplus", "cached_cefsplus"}:
        raise ValueError(
            "2-D y is only supported for select_cefsplus / CEFSPlusSelector "
            f"and select_cached(method='cefsplus'); got selector={selector!r}"
        )
    if method != "cefsplus":
        raise ValueError(
            "2-D y is only supported for method='cefsplus'; "
            f"got method={method!r}"
        )
    if within is not None:
        raise ValueError(
            "2-D y is not supported with within demeaning; drop within= or "
            "reduce to a 1-D target"
        )
    encoding = "none" if cat_encoding is None else str(cat_encoding)
    if encoding not in SUPPORTED_MULTI_TARGET_CAT_ENCODING:
        raise ValueError(
            "2-D y is not supported with supervised cat_encoding="
            f"{encoding!r}; use 'none', 'onehot', 'ordinal', or "
            "'frequency', or encode first"
        )
    if k_method is None:
        return
    allowed = (
        {"evaluate", "elbow", "penalized_objective"}
        if routed
        else SUPPORTED_MULTI_TARGET_AUTO_K
    )
    if k_method not in allowed:
        raise ValueError(
            "2-D y is not supported with auto-k method "
            f"{k_method!r}; use {_MULTI_TARGET_AUTO_K_HELP}"
        )


def reject_degenerate_multi_targets(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float = _EPS,
) -> None:
    """Reject constant or singular multi-target columns on positive-weight rows.

    Copula correlation forces a unit diagonal, so a zero-variance column
    would otherwise report ``cond(Ryy)=1`` and inflate ``q·k``. Scalar 1-D
    targets are not checked here.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[1]) < 2:
        return
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != arr.shape[0]:
        raise ValueError("target rows must match sample_weight length")
    positive = w > 0.0
    if not np.any(positive):
        raise ValueError(
            "Multi-target CEFS+ requires positive-weight rows; "
            "drop or combine degenerate targets before joint selection."
        )
    cols = arr[positive]
    w_eff = w[positive]
    w_sum = float(w_eff.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError(
            "Multi-target CEFS+ requires positive-weight rows; "
            "drop or combine degenerate targets before joint selection."
        )
    means = (w_eff @ cols) / w_sum
    centered = cols - means
    var = (w_eff @ (centered * centered)) / w_sum
    if (not np.isfinite(var).all()) or np.any(var <= float(eps)):
        raise ValueError(
            "Multi-target CEFS+ requires every target column to vary on the "
            "effective retained positive-weight rows; a constant or singular "
            "column was found. Drop or combine degenerate targets before "
            "joint selection."
        )


def reject_ill_conditioned_targets(sigma: np.ndarray, *, cap: float = TARGET_CONDITION_CAP) -> float:
    """Reject collinear copula targets above a documented condition cap.

    Applied to the unshrunk target correlation. The feature-side shrink is
    still used when factoring ``Σ_{Y|S}`` along the path.
    """
    matrix = np.asarray(sigma, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("target covariance must be square")
    cond = float(np.linalg.cond(matrix))
    if not np.isfinite(cond) or cond > float(cap):
        raise ValueError(
            "Multi-target CEFS+ requires a well-conditioned target correlation; "
            f"cond(Ryy)={cond:.3g} exceeds {float(cap):.3g}. Drop or combine "
            "collinear targets before joint selection."
        )
    return cond


def shrunk_target_covariance(Ryy: np.ndarray, *, shrink: float = _SHRINK) -> np.ndarray:
    """Apply the feature-side off-diagonal shrink, keeping unit diagonal."""
    scale = 1.0 - float(shrink)
    sigma = scale * np.asarray(Ryy, dtype=np.float64)
    np.fill_diagonal(sigma, 1.0)
    sigma = 0.5 * (sigma + sigma.T)
    return sigma


def _factor_spd(sigma: np.ndarray, *, shrink: float, eps: float) -> np.ndarray:
    """Cholesky of ``sigma`` with the CEFS+ ridge guard; lower-triangular."""
    q = int(sigma.shape[0])
    a0 = np.array(sigma, dtype=np.float64, copy=True)
    a0 = 0.5 * (a0 + a0.T)
    for i in range(q):
        a0[i, i] = max(float(a0[i, i]), eps)
    ridge = 0.0
    last_err: Exception | None = None
    for _attempt in range(8):
        a = a0 if ridge == 0.0 else a0 + ridge * np.eye(q, dtype=np.float64)
        try:
            return np.linalg.cholesky(a)
        except np.linalg.LinAlgError as err:
            last_err = err
            ridge = float(shrink) if ridge == 0.0 else max(10.0 * ridge, float(shrink))
    if last_err is not None:
        raise last_err
    raise np.linalg.LinAlgError("target covariance factorization failed")


class _GuardedTargetFactor:
    """Reuse one guarded Cholesky of residual ``Σ_{Y|S}`` across candidates."""

    __slots__ = ("shrink", "eps", "chol")

    def __init__(self, sigma: np.ndarray, *, shrink: float, eps: float):
        self.shrink = float(shrink)
        self.eps = float(eps)
        self.chol = _factor_spd(sigma, shrink=self.shrink, eps=self.eps)

    def replace(self, sigma: np.ndarray) -> None:
        self.chol = _factor_spd(sigma, shrink=self.shrink, eps=self.eps)

    def quadratic(self, c: np.ndarray) -> float:
        c_arr = np.asarray(c, dtype=np.float64).reshape(-1)
        z = solve_triangular(self.chol, c_arr, lower=True, check_finite=False)
        return float(z @ z)

    def inverse_middle(self, c_block: np.ndarray) -> np.ndarray:
        c_arr = np.asarray(c_block, dtype=np.float64)
        z = solve_triangular(self.chol, c_arr.T, lower=True, check_finite=False)
        return z.T @ z


def _spd_quad(sigma: np.ndarray, c: np.ndarray, *, shrink: float, eps: float) -> float:
    """``cᵀ Σ^{-1} c`` with the CEFS+ Cholesky-plus-ridge guard."""
    return _GuardedTargetFactor(sigma, shrink=shrink, eps=eps).quadratic(c)


def _spd_inverse_middle(
    sigma: np.ndarray,
    c_block: np.ndarray,
    *,
    shrink: float,
    eps: float,
) -> np.ndarray:
    """``C Σ^{-1} Cᵀ`` for a block of residual target covariances."""
    return _GuardedTargetFactor(sigma, shrink=shrink, eps=eps).inverse_middle(c_block)


def multiple_correlation(C: np.ndarray, sigma: np.ndarray, *, shrink: float, eps: float) -> np.ndarray:
    """Scalar screening scores ``√(cᵀ Σ^{-1} c)`` clipped to a valid correlation."""
    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.size == 0:
        return np.empty(0, dtype=np.float64)
    q = int(sigma.shape[0])
    a0 = np.array(sigma, dtype=np.float64, copy=True)
    a0 = 0.5 * (a0 + a0.T)
    for i in range(q):
        a0[i, i] = max(float(a0[i, i]), eps)
    ridge = 0.0
    last_err: Exception | None = None
    for _attempt in range(8):
        a = a0 if ridge == 0.0 else a0 + ridge * np.eye(q, dtype=np.float64)
        try:
            chol = np.linalg.cholesky(a)
            z = np.linalg.solve(chol, C_arr.T)
            quad = np.sum(z * z, axis=0)
            return np.sqrt(np.clip(quad, 0.0, 1.0 - eps)).astype(np.float64, copy=False)
        except np.linalg.LinAlgError as err:
            last_err = err
            ridge = float(shrink) if ridge == 0.0 else max(10.0 * ridge, float(shrink))
    if last_err is not None:
        raise last_err
    raise np.linalg.LinAlgError("target covariance factorization failed")


def _apply_column(
    j: int,
    *,
    R: np.ndarray,
    L: np.ndarray,
    d: np.ndarray,
    C_res: np.ndarray,
    remaining: np.ndarray,
    t: int,
    scale: float,
    eps: float,
    s1: float,
) -> np.ndarray:
    """Rank-1 residual update; returns the target-side vector ``u``."""
    sq = np.sqrt(max(float(s1), eps))
    u = C_res[j] / sq
    m = int(C_res.shape[0])
    for i in range(m):
        if not remaining[i] or i == j:
            continue
        acc = float(R[i, j]) * scale
        for a in range(t):
            acc -= float(L[i, a]) * float(L[j, a])
        lij = acc / sq
        L[i, t] = lij
        d[i] -= lij * lij
        C_res[i] -= lij * u
    remaining[j] = False
    return u


def cefsplus_multi_loop(
    R: np.ndarray,
    C: np.ndarray,
    Ryy: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    *,
    forced: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    want_objective: bool = False,
    shrink: float = _SHRINK,
    eps: float = _EPS,
    callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Greedy joint CEFS+; returns local indices, objective, and ``cond(Ryy)``."""
    C_arr = np.asarray(C, dtype=np.float64)
    m, q = C_arr.shape
    if k <= 0 or m == 0 or q < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            float("nan"),
        )
    k = min(int(k), m)
    scale = 1.0 - float(shrink)
    cond = reject_ill_conditioned_targets(np.asarray(Ryy, dtype=np.float64))
    sigma = shrunk_target_covariance(Ryy, shrink=shrink)
    remaining = np.ones(m, dtype=bool)
    if eligible is not None:
        remaining = np.asarray(eligible, dtype=bool).copy()
    forced_arr = (
        np.empty(0, dtype=np.int64)
        if forced is None
        else np.asarray(forced, dtype=np.int64)
    )
    for j in forced_arr:
        remaining[int(j)] = True
    L = np.zeros((m, m), dtype=np.float64)
    d = np.ones(m, dtype=np.float64)
    C_res = scale * C_arr
    t = 0
    selected: list[int] = []
    objective_acc = 0.0
    obj_out: list[float] = []
    factor = _GuardedTargetFactor(sigma, shrink=shrink, eps=eps)

    def _commit(j: int) -> float:
        nonlocal t, sigma, objective_acc, factor
        quad = factor.quadratic(C_res[j])
        s1 = max(float(d[j]), eps)
        s2 = max(s1 - quad, eps)
        u = _apply_column(
            j,
            R=R,
            L=L,
            d=d,
            C_res=C_res,
            remaining=remaining,
            t=t,
            scale=scale,
            eps=eps,
            s1=s1,
        )
        sigma = sigma - np.outer(u, u)
        sigma = 0.5 * (sigma + sigma.T)
        factor.replace(sigma)
        t += 1
        selected.append(int(j))
        gain = float(np.log(s1) - np.log(s2))
        objective_acc += gain
        return gain

    n_forced_committed = 0
    for j in forced_arr:
        if remaining[int(j)]:
            _commit(int(j))
            n_forced_committed += 1
    objective_acc = 0.0

    n_discover = min(k, int(np.sum(remaining)))
    count = 0
    while count < n_discover:
        live = np.flatnonzero(remaining)
        if live.size == 0:
            break
        if t == 0:
            j = int(live[int(np.argmax(tie_break_rel[live]))])
        else:
            best_j = -1
            best_score = -np.inf
            best_rel = -np.inf
            for jj in live.tolist():
                quad = factor.quadratic(C_res[jj])
                s1 = max(float(d[jj]), eps)
                s2 = max(s1 - quad, eps)
                sc = float(np.log(s1) - np.log(s2))
                rel = float(tie_break_rel[jj])
                better = best_j < 0 or sc > best_score + 1e-12
                if not better and abs(sc - best_score) <= 1e-12:
                    if rel > best_rel + 1e-15 or (
                        abs(rel - best_rel) <= 1e-15 and jj < best_j
                    ):
                        better = True
                if better:
                    best_j = int(jj)
                    best_score = sc
                    best_rel = rel
            if best_j < 0:
                break
            j = best_j
        _commit(j)
        if want_objective:
            obj_out.append(objective_acc)
        count += 1
        if callback is not None:
            report_progress(
                callback,
                count,
                n_discover,
                stage="path",
                selector="cefsplus",
            )

    discovered = np.asarray(selected[n_forced_committed:], dtype=np.int64)
    return discovered, np.asarray(obj_out, dtype=np.float64), cond


def cefsplus_block_loop_multi(
    R: np.ndarray,
    C: np.ndarray,
    Ryy: np.ndarray,
    k: int,
    tie_break_rel: np.ndarray,
    block_members: Sequence[np.ndarray],
    *,
    forced_blocks: Sequence[int] = (),
    eligible_blocks: Sequence[int] | None = None,
    want_objective: bool = False,
    shrink: float = _SHRINK,
    eps: float = _EPS,
    callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    """Joint residual log-det CEFS+ over atomic blocks for ``q`` targets."""
    C_arr = np.asarray(C, dtype=np.float64)
    m, q = C_arr.shape
    n_blocks = len(block_members)
    if k <= 0 or m == 0 or n_blocks == 0 or q < 2:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            [],
            float("nan"),
        )
    scale = 1.0 - float(shrink)
    cond = reject_ill_conditioned_targets(np.asarray(Ryy, dtype=np.float64))
    sigma = shrunk_target_covariance(Ryy, shrink=shrink)
    remaining = np.ones(m, dtype=bool)
    remaining_blocks = np.ones(n_blocks, dtype=bool)
    if eligible_blocks is not None:
        remaining_blocks[:] = False
        for idx in eligible_blocks:
            remaining_blocks[int(idx)] = True
    forced = [int(i) for i in forced_blocks]
    for idx in forced:
        remaining_blocks[idx] = False
    L = np.zeros((m, m), dtype=np.float64)
    d = np.ones(m, dtype=np.float64)
    C_res = scale * C_arr
    t = 0
    selected_cols: list[int] = []
    selected_block_ids: list[int] = []
    objective_acc = 0.0
    obj_out: list[float] = []
    factor = _GuardedTargetFactor(sigma, shrink=shrink, eps=eps)

    def _commit_block(block_idx: int) -> None:
        nonlocal t, sigma, factor
        members = np.asarray(block_members[block_idx], dtype=np.int64)
        for j in members:
            jj = int(j)
            if not remaining[jj]:
                continue
            s1 = max(float(d[jj]), eps)
            u = _apply_column(
                jj,
                R=R,
                L=L,
                d=d,
                C_res=C_res,
                remaining=remaining,
                t=t,
                scale=scale,
                eps=eps,
                s1=s1,
            )
            sigma = sigma - np.outer(u, u)
            sigma = 0.5 * (sigma + sigma.T)
            selected_cols.append(jj)
            t += 1
        factor.replace(sigma)
        remaining_blocks[block_idx] = False

    for block_idx in forced:
        _commit_block(block_idx)
    objective_acc = 0.0

    n_discover = min(int(k), int(np.sum(remaining_blocks)))
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
            g = _block_residual_cov(R, L, d, live, t, scale, eps)
            c_block = np.asarray(C_res[live], dtype=np.float64)
            mid = factor.inverse_middle(c_block)
            g_y = g - mid
            gain = _chol_logdet(g, shrink=shrink, eps=eps) - _chol_logdet(
                g_y, shrink=shrink, eps=eps
            )
            rel = float(np.max(tie_break_rel[live]))
            better = best_idx < 0 or gain > best_gain + 1e-12
            if not better and abs(gain - best_gain) <= 1e-12:
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
        objective_acc += float(best_gain)
        if want_objective:
            obj_out.append(objective_acc)
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
        np.asarray(obj_out, dtype=np.float64),
        selected_block_ids,
        cond,
    )


def joint_logdet_oracle(
    Rxx: np.ndarray,
    C: np.ndarray,
    Ryy: np.ndarray,
    selected: Sequence[int],
    *,
    shrink: float = _SHRINK,
    eps: float = _EPS,
) -> float:
    """``-log|Σ_{Y|S}| + log|Σ_Y|`` from the copula Gram of a fixed set ``S``."""
    idx = np.asarray(list(selected), dtype=np.int64)
    scale = 1.0 - float(shrink)
    sigma_y = shrunk_target_covariance(Ryy, shrink=shrink)
    if idx.size == 0:
        return 0.0
    g = scale * np.asarray(Rxx, dtype=np.float64)[np.ix_(idx, idx)]
    np.fill_diagonal(g, 1.0)
    c = scale * np.asarray(C, dtype=np.float64)[idx]
    a0 = 0.5 * (g + g.T)
    k = int(a0.shape[0])
    for i in range(k):
        a0[i, i] = max(float(a0[i, i]), eps)
    ridge = 0.0
    last_err: Exception | None = None
    z = None
    for _attempt in range(8):
        a = a0 if ridge == 0.0 else a0 + ridge * np.eye(k, dtype=np.float64)
        try:
            chol = np.linalg.cholesky(a)
            z = np.linalg.solve(chol, c)
            break
        except np.linalg.LinAlgError as err:
            last_err = err
            ridge = float(shrink) if ridge == 0.0 else max(10.0 * ridge, float(shrink))
            z = None
    if z is None:
        if last_err is not None:
            raise last_err
        raise np.linalg.LinAlgError("feature covariance factorization failed")
    sigma_yg = sigma_y - z.T @ z
    return _chol_logdet(sigma_y, shrink=shrink, eps=eps) - _chol_logdet(
        sigma_yg, shrink=shrink, eps=eps
    )
