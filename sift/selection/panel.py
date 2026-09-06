"""Candidate-panel construction for cache-backed Gaussian selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from sift.estimators.copula import (
    gaussian_mi_from_corr,
    greedy_corr_prune,
    weighted_corr_with_vector,
    weighted_correlation_matrix,
    weighted_rank_gauss_1d,
)
from sift.selection.blocks import prune_blocks_by_corr, screen_block_indices
from sift.selection.objective import objective_from_corr_path


GaussianMethod = Literal["cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"]
CorrPrune = float | None | Literal["auto"]


@dataclass(frozen=True)
class CandidatePanel:
    """Shared candidate state after cache screening and optional pruning."""

    cand: np.ndarray
    original: np.ndarray
    R: np.ndarray
    r: np.ndarray
    rel: np.ndarray
    p_valid: int
    n_eff_kish: float
    n_eff_sum: float
    names: list[str] | None
    C: np.ndarray | None = None
    Ryy: np.ndarray | None = None
    n_targets: int = 1
    target_condition: float | None = None


def resolve_corr_prune(method: GaussianMethod, corr_prune: CorrPrune) -> float | None:
    """Resolve the public corr_prune option for a cache-backed method."""
    if corr_prune == "auto":
        return None
    if corr_prune is None:
        return None
    if isinstance(corr_prune, (bool, np.bool_)):
        raise ValueError("corr_prune must be 'auto', None, or a finite float in (0, 1]")
    threshold = float(corr_prune)
    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError("corr_prune must be 'auto', None, or a finite float in (0, 1]")
    return threshold


def effective_sample_sizes(w: np.ndarray) -> tuple[float, float]:
    """Return (Kish n_eff, weight sum) for non-negative cache weights."""
    w_arr = np.asarray(w, dtype=np.float64).ravel()
    weight_sum = float(np.sum(w_arr))
    sum_sq = float(np.sum(w_arr * w_arr))
    kish = float(weight_sum * weight_sum / sum_sq) if sum_sq > 0.0 else float("nan")
    return kish, weight_sum


def local_standardize(
    Z: np.ndarray,
    w: np.ndarray,
    *,
    columns: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Weighted-center and scale rows/columns under local weights.

    Zero-variance columns are returned as zeros so downstream correlation
    formation stays finite and those columns carry neutral relevance.
    """
    Z_arr = np.asarray(Z)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    if columns is not None:
        Z_arr = Z_arr[:, np.asarray(columns, dtype=np.int64)]
    Z_arr = np.asarray(Z_arr, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64).ravel()
    if Z_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("w length must match Z rows")
    if not np.isfinite(Z_arr).all():
        raise ValueError("Z must contain only finite values")
    if not np.isfinite(w_arr).all() or np.any(w_arr < 0.0):
        raise ValueError("w must contain finite non-negative weights")
    w_sum = float(w_arr.sum())
    if w_sum <= 0.0:
        raise ValueError("Weights must sum to > 0")

    mean = (w_arr @ Z_arr) / w_sum
    centered = Z_arr - mean
    var = (w_arr @ (centered * centered)) / w_sum
    scale = np.sqrt(np.maximum(var, 0.0))
    out = np.zeros_like(centered, dtype=np.float64)
    good = scale > eps
    if np.any(good):
        out[:, good] = centered[:, good] / scale[good]
    return out


def _candidate_order(
    r: np.ndarray,
    *,
    top_m: int,
    pool: np.ndarray | None = None,
) -> np.ndarray:
    p_valid = int(len(r))
    if pool is None:
        eligible = np.arange(p_valid, dtype=np.int64)
    else:
        eligible = np.asarray(pool, dtype=np.int64)
        if eligible.size:
            if np.any(eligible < 0) or np.any(eligible >= p_valid):
                raise ValueError("candidate pool positions must be in-range valid columns")
    n_eligible = int(eligible.size)
    top_m_eff = min(max(int(top_m), 0), n_eligible)
    if top_m_eff <= 0:
        return np.empty(0, dtype=np.int64)
    if top_m_eff < n_eligible:
        pick = np.argpartition(np.abs(r[eligible]), -top_m_eff)[-top_m_eff:]
        return eligible[pick].astype(np.int64)
    return eligible.astype(np.int64, copy=False)


def _block_candidate_order(
    r: np.ndarray,
    block_members: list[np.ndarray],
    *,
    top_m: int,
    pool: np.ndarray | None = None,
    protect: np.ndarray | None = None,
) -> np.ndarray:
    """Screen discovery blocks by max |r|; included blocks are added later.

    ``top_m`` is a discovery-block budget. Protected/included members are
    omitted here so they cannot consume that budget, matching F1 column
    screening. Whole blocks are kept or dropped together.
    """
    p_valid = int(len(r))
    pool_set = None if pool is None else set(int(i) for i in np.asarray(pool, dtype=np.int64))
    protect_set = set(int(i) for i in np.asarray(protect, dtype=np.int64)) if protect is not None else set()
    discovery_blocks: list[np.ndarray] = []
    for members in block_members:
        arr = np.asarray(members, dtype=np.int64)
        if arr.size == 0:
            continue
        if np.any(arr < 0) or np.any(arr >= p_valid):
            raise ValueError("block member positions must be in-range valid columns")
        if any(int(i) in protect_set for i in arr):
            continue
        if pool_set is not None:
            arr = np.asarray([i for i in arr if int(i) in pool_set], dtype=np.int64)
            if arr.size == 0:
                continue
        discovery_blocks.append(arr)
    if not discovery_blocks:
        return np.empty(0, dtype=np.int64)
    scores = np.abs(np.asarray(r, dtype=np.float64))
    chosen = screen_block_indices(
        discovery_blocks,
        scores,
        top_m=int(top_m),
        protect=(),
    )
    cols: list[int] = []
    seen: set[int] = set()
    for bidx in chosen:
        for col in discovery_blocks[int(bidx)]:
            key = int(col)
            if key not in seen:
                cols.append(key)
                seen.add(key)
    return np.asarray(cols, dtype=np.int64)


def _panel_from_corr(
    R_all: np.ndarray | None,
    Z: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    *,
    top_m: int,
    corr_prune: CorrPrune,
    method: GaussianMethod,
    original: np.ndarray | None,
    names_all: list[str] | None,
    protect: np.ndarray | None = None,
    pool: np.ndarray | None = None,
    block_members: list[np.ndarray] | None = None,
    C_all: np.ndarray | None = None,
    Ryy: np.ndarray | None = None,
    n_targets: int = 1,
    target_condition: float | None = None,
) -> CandidatePanel:
    p_valid = int(len(r))
    protect_arr = (
        np.empty(0, dtype=np.int64)
        if protect is None
        else np.asarray(protect, dtype=np.int64)
    )
    corr_prune_eff = resolve_corr_prune(method, corr_prune)
    use_block_screen = block_members is not None and any(
        len(np.asarray(members)) > 1 for members in block_members
    )
    if use_block_screen:
        cand = _block_candidate_order(
            r,
            block_members,
            top_m=top_m,
            pool=pool,
            protect=protect_arr,
        )
    else:
        cand = _candidate_order(r, top_m=top_m, pool=pool)

    if cand.size == 0:
        R_cand = np.empty((0, 0), dtype=np.float64)
    elif R_all is not None:
        R_full = np.asarray(R_all, dtype=np.float64)
        R_cand = np.ascontiguousarray(R_full[np.ix_(cand, cand)], dtype=np.float64)
    else:
        Z_cand = np.ascontiguousarray(np.asarray(Z, dtype=np.float64)[:, cand])
        R_cand = weighted_correlation_matrix(
            Z_cand,
            np.asarray(w, dtype=np.float64),
            backend="blas",
        )

    if corr_prune_eff is not None and cand.size:
        if use_block_screen:
            cand_set = {int(i): pos for pos, i in enumerate(cand)}
            local_blocks: list[np.ndarray] = []
            protect_local: list[int] = []
            protect_set = set(int(i) for i in protect_arr)
            for members in block_members:
                local = [cand_set[int(i)] for i in members if int(i) in cand_set]
                if not local:
                    continue
                bidx = len(local_blocks)
                arr = np.asarray(local, dtype=np.int64)
                local_blocks.append(arr)
                if any(int(members[j]) in protect_set for j in range(len(members)) if int(members[j]) in cand_set):
                    protect_local.append(bidx)
            keep_blocks = prune_blocks_by_corr(
                local_blocks,
                R_cand,
                np.abs(r[cand]),
                corr_prune_eff,
                protect=protect_local,
            )
            keep_cols: list[int] = []
            seen_local: set[int] = set()
            for bidx in keep_blocks:
                for col in local_blocks[int(bidx)]:
                    key = int(col)
                    if key not in seen_local:
                        keep_cols.append(key)
                        seen_local.add(key)
            keep = np.asarray(keep_cols, dtype=np.int64)
        else:
            keep = greedy_corr_prune(
                np.arange(len(cand), dtype=np.int64),
                R_cand,
                np.abs(r[cand]),
                corr_prune_eff,
            )
        cand = cand[keep]
        R_cand = np.ascontiguousarray(R_cand[np.ix_(keep, keep)], dtype=np.float64)

    if protect_arr.size:
        protect_unique = []
        seen_protect = set()
        for idx in protect_arr:
            key = int(idx)
            if key in seen_protect:
                continue
            if key < 0 or key >= p_valid:
                raise ValueError("protected feature positions must be in-range valid columns")
            protect_unique.append(key)
            seen_protect.add(key)
        discovery = [int(i) for i in cand if int(i) not in seen_protect]
        ordered = np.asarray(protect_unique + discovery, dtype=np.int64)
        if R_all is not None:
            R_full = np.asarray(R_all, dtype=np.float64)
            R_cand = np.ascontiguousarray(R_full[np.ix_(ordered, ordered)], dtype=np.float64)
        elif ordered.size:
            Z_cand = np.ascontiguousarray(np.asarray(Z, dtype=np.float64)[:, ordered])
            R_cand = weighted_correlation_matrix(
                Z_cand,
                np.asarray(w, dtype=np.float64),
                backend="blas",
            )
        else:
            R_cand = np.empty((0, 0), dtype=np.float64)
        cand = ordered

    original_arr = cand if original is None else np.asarray(original, dtype=np.int64)[cand]
    rel = gaussian_mi_from_corr(r)
    kish, weight_sum = effective_sample_sizes(w)
    names = None
    if names_all is not None:
        names = [names_all[int(i)] for i in original_arr]

    C_cand = None
    if C_all is not None and cand.size:
        C_cand = np.ascontiguousarray(
            np.asarray(C_all, dtype=np.float64)[cand],
            dtype=np.float64,
        )
    elif C_all is not None:
        C_cand = np.empty((0, int(np.asarray(C_all).shape[1])), dtype=np.float64)
    return CandidatePanel(
        cand=np.asarray(cand, dtype=np.int64),
        original=np.asarray(original_arr, dtype=np.int64),
        R=np.ascontiguousarray(R_cand, dtype=np.float64),
        r=np.asarray(r[cand], dtype=np.float64),
        rel=np.asarray(rel[cand], dtype=np.float64),
        p_valid=p_valid,
        n_eff_kish=kish,
        n_eff_sum=weight_sum,
        names=names,
        C=C_cand,
        Ryy=None if Ryy is None else np.ascontiguousarray(Ryy, dtype=np.float64),
        n_targets=int(n_targets),
        target_condition=target_condition,
    )


def build_candidate_panel(
    cache,
    y,
    k: int,
    *,
    top_m: int | None = None,
    corr_prune: CorrPrune = "auto",
    method: GaussianMethod = "cefsplus",
    zy: np.ndarray | None = None,
    protect_valid: np.ndarray | None = None,
    pool_valid: np.ndarray | None = None,
    block_members: list[np.ndarray] | None = None,
) -> CandidatePanel:
    """Build the screened/pruned candidate panel used by cache selectors."""
    from sift.estimators.copula import weighted_correlation_matrix, weighted_rank_gauss_2d
    from sift.selection.cefsplus_multi import (
        as_regression_targets,
        multiple_correlation,
        reject_degenerate_multi_targets,
        reject_ill_conditioned_targets,
        shrunk_target_covariance,
    )

    C_all = None
    Ryy = None
    n_targets = 1
    target_condition = None
    if zy is None:
        y_mat, n_targets = as_regression_targets(y, int(cache.n_rows_original))
        ys = y_mat[np.asarray(cache.row_idx)]
        if n_targets == 1:
            zy_arr = weighted_rank_gauss_1d(np.asarray(ys).reshape(-1), cache.sample_weight)
            r = weighted_corr_with_vector(cache.Z, zy_arr, cache.sample_weight)
        else:
            zy_mat = weighted_rank_gauss_2d(np.asarray(ys, dtype=np.float64), cache.sample_weight)
            reject_degenerate_multi_targets(zy_mat, cache.sample_weight)
            Ryy = np.asarray(
                weighted_correlation_matrix(zy_mat, cache.sample_weight, backend="blas"),
                dtype=np.float64,
            )
            target_condition = reject_ill_conditioned_targets(
                np.asarray(Ryy, dtype=np.float64)
            )
            sigma = shrunk_target_covariance(Ryy)
            C_all = np.column_stack(
                [
                    np.asarray(
                        weighted_corr_with_vector(cache.Z, zy_mat[:, j], cache.sample_weight),
                        dtype=np.float64,
                    )
                    for j in range(n_targets)
                ]
            )
            r = multiple_correlation(C_all, sigma, shrink=1e-6, eps=1e-12)
    else:
        zy_arr = np.asarray(zy, dtype=np.float64)
        if zy_arr.ndim == 1:
            if zy_arr.shape[0] != cache.Z.shape[0]:
                raise ValueError("zy length must match cache rows")
            r = weighted_corr_with_vector(cache.Z, zy_arr, cache.sample_weight)
        elif zy_arr.ndim == 2 and zy_arr.shape[1] == 1:
            zy_arr = zy_arr.reshape(-1)
            if zy_arr.shape[0] != cache.Z.shape[0]:
                raise ValueError("zy length must match cache rows")
            r = weighted_corr_with_vector(cache.Z, zy_arr, cache.sample_weight)
        elif zy_arr.ndim == 2:
            if zy_arr.shape[0] != cache.Z.shape[0]:
                raise ValueError("zy length must match cache rows")
            n_targets = int(zy_arr.shape[1])
            reject_degenerate_multi_targets(zy_arr, cache.sample_weight)
            Ryy = np.asarray(
                weighted_correlation_matrix(zy_arr, cache.sample_weight, backend="blas"),
                dtype=np.float64,
            )
            target_condition = reject_ill_conditioned_targets(
                np.asarray(Ryy, dtype=np.float64)
            )
            sigma = shrunk_target_covariance(Ryy)
            C_all = np.column_stack(
                [
                    np.asarray(
                        weighted_corr_with_vector(cache.Z, zy_arr[:, j], cache.sample_weight),
                        dtype=np.float64,
                    )
                    for j in range(n_targets)
                ]
            )
            r = multiple_correlation(C_all, sigma, shrink=1e-6, eps=1e-12)
        else:
            raise ValueError("zy must be one- or two-dimensional")
    p_valid = int(len(r))
    if top_m is None:
        top_m = max(5 * int(k), 250)
    top_m_eff = max(int(top_m), int(k))
    n_units = len(block_members) if block_members is not None else p_valid
    top_m_eff = min(top_m_eff, n_units)
    names_all = list(cache.feature_names) if cache.feature_names is not None else None

    return _panel_from_corr(
        cache.Rxx,
        cache.Z,
        np.asarray(r, dtype=np.float64),
        cache.sample_weight,
        top_m=top_m_eff,
        corr_prune=corr_prune,
        method=method,
        original=np.asarray(cache.valid_cols, dtype=np.int64),
        names_all=names_all,
        protect=protect_valid,
        pool=pool_valid,
        block_members=block_members,
        C_all=C_all,
        Ryy=Ryy,
        n_targets=n_targets,
        target_condition=target_condition,
    )


def local_corr_panel(
    Z: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    *,
    top_m: int,
    corr_prune: CorrPrune,
    method: GaussianMethod,
    Rxx: np.ndarray | None = None,
    local_standardize: bool = True,
    block_members: list[np.ndarray] | None = None,
    protect: np.ndarray | None = None,
) -> CandidatePanel:
    """Build a candidate panel from fold/bootstrap-local correlations."""
    w_arr = np.asarray(w, dtype=np.float64).ravel()
    Z_arr = np.asarray(Z)
    zy_arr = np.asarray(zy, dtype=np.float64)
    if zy_arr.ndim == 2 and int(zy_arr.shape[1]) > 1:
        raise ValueError(
            "local fold correlations do not support 2-D y; joint multi-target "
            "CEFS+ is not available for gaussian_cv/xfit_objective"
        )
    zy_arr = zy_arr.reshape(-1)
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 2D")
    if Z_arr.shape[0] != zy_arr.shape[0] or Z_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("Z, zy, and w must have matching row counts")

    if local_standardize:
        if not np.isfinite(Z_arr).all() or not np.isfinite(zy_arr).all():
            raise ValueError("Z and zy must contain only finite values")
        if not np.isfinite(w_arr).all() or np.any(w_arr < 0.0):
            raise ValueError("w must contain finite non-negative weights")
        w_sum = float(w_arr.sum())
        if w_sum <= 0.0:
            raise ValueError("Weights must sum to > 0")

        z_mean = (w_arr @ Z_arr) / w_sum
        y_mean = float(w_arr @ zy_arr / w_sum)
        y_centered = zy_arr - y_mean
        y_scale = float(
            np.sqrt(max(float(w_arr @ (y_centered * y_centered) / w_sum), 0.0))
        )
        z_var = np.zeros(Z_arr.shape[1], dtype=np.float64)
        covariance = np.zeros(Z_arr.shape[1], dtype=np.float64)
        weighted_y = w_arr * y_centered
        # Center bounded column blocks so large-offset inputs retain the
        # stable two-pass variance of local_standardize without allocating a
        # full float64 centered copy of the entire panel.
        for start in range(0, Z_arr.shape[1], 256):
            stop = min(start + 256, Z_arr.shape[1])
            centered_block = (
                np.asarray(Z_arr[:, start:stop], dtype=np.float64)
                - z_mean[None, start:stop]
            )
            z_var[start:stop] = (
                np.einsum(
                    "i,ij,ij->j",
                    w_arr,
                    centered_block,
                    centered_block,
                    optimize=True,
                )
                / w_sum
            )
            covariance[start:stop] = centered_block.T @ weighted_y / w_sum
        z_scale = np.sqrt(np.maximum(z_var, 0.0))
        r = np.zeros(Z_arr.shape[1], dtype=np.float64)
        good = z_scale > 1e-12
        if y_scale > 1e-12 and np.any(good):
            r[good] = covariance[good] / (z_scale[good] * y_scale)
            np.clip(r, -0.999999, 0.999999, out=r)
            # The prior explicit path returned float32 correlations from
            # weighted_corr_with_vector; retain its candidate/tie behavior.
            r = r.astype(np.float32).astype(np.float64)

        if block_members is not None or protect is not None:
            Z_std = globals()["local_standardize"](Z_arr, w_arr)
            return _panel_from_corr(
                None,
                Z_std,
                np.asarray(r, dtype=np.float64),
                w_arr,
                top_m=top_m,
                corr_prune=corr_prune,
                method=method,
                original=None,
                names_all=None,
                block_members=block_members,
                protect=protect,
            )
        cand = _candidate_order(r, top_m=top_m)
        if cand.size:
            Z_cand = globals()["local_standardize"](Z_arr, w_arr, columns=cand)
            R_cand = weighted_correlation_matrix(Z_cand, w_arr, backend="blas")
        else:
            R_cand = np.empty((0, 0), dtype=np.float64)
        corr_prune_eff = resolve_corr_prune(method, corr_prune)
        if corr_prune_eff is not None and cand.size:
            keep = greedy_corr_prune(
                np.arange(cand.size, dtype=np.int64),
                R_cand,
                np.abs(r[cand]),
                corr_prune_eff,
            )
            cand = cand[keep]
            R_cand = np.ascontiguousarray(
                R_cand[np.ix_(keep, keep)], dtype=np.float64
            )
        rel = gaussian_mi_from_corr(r)
        kish, weight_sum = effective_sample_sizes(w_arr)
        return CandidatePanel(
            cand=np.asarray(cand, dtype=np.int64),
            original=np.asarray(cand, dtype=np.int64),
            R=np.ascontiguousarray(R_cand, dtype=np.float64),
            r=np.asarray(r[cand], dtype=np.float64),
            rel=np.asarray(rel[cand], dtype=np.float64),
            p_valid=int(Z_arr.shape[1]),
            n_eff_kish=kish,
            n_eff_sum=weight_sum,
            names=None,
        )
    else:
        Z_used = Z_arr
        zy_used = zy_arr
        R_all = Rxx
        r = weighted_corr_with_vector(Z_used, zy_used, w_arr)
    return _panel_from_corr(
        R_all,
        Z_used,
        np.asarray(r, dtype=np.float64),
        w_arr,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
        original=None,
        names_all=None,
        block_members=block_members,
        protect=protect,
    )


def score_path_from_corr(
    R_path: np.ndarray,
    r_path: np.ndarray,
    *,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """Evaluate the CEFS+ objective for an ordered correlation path."""
    return objective_from_corr_path(R_path, r_path, shrink=shrink, eps=eps)
