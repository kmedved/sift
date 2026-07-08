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


def resolve_corr_prune(method: GaussianMethod, corr_prune: CorrPrune) -> float | None:
    """Resolve the public corr_prune option for a cache-backed method."""
    if corr_prune == "auto":
        return 0.95 if method == "cefsplus" else None
    if corr_prune is None:
        return None
    if isinstance(corr_prune, (bool, np.bool_)):
        raise ValueError("corr_prune must be 'auto', None, or a positive finite float")
    threshold = float(corr_prune)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("corr_prune must be 'auto', None, or a positive finite float")
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
    Z_arr = np.asarray(Z, dtype=np.float64)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    if columns is not None:
        Z_arr = Z_arr[:, np.asarray(columns, dtype=np.int64)]
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


def _candidate_order(r: np.ndarray, *, top_m: int) -> np.ndarray:
    p_valid = int(len(r))
    top_m_eff = min(max(int(top_m), 0), p_valid)
    if top_m_eff <= 0:
        return np.empty(0, dtype=np.int64)
    if top_m_eff < p_valid:
        return np.argpartition(np.abs(r), -top_m_eff)[-top_m_eff:].astype(np.int64)
    return np.arange(p_valid, dtype=np.int64)


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
) -> CandidatePanel:
    p_valid = int(len(r))
    cand = _candidate_order(r, top_m=top_m)
    corr_prune_eff = resolve_corr_prune(method, corr_prune)

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
        keep = greedy_corr_prune(
            np.arange(len(cand), dtype=np.int64),
            R_cand,
            np.abs(r[cand]),
            corr_prune_eff,
        )
        cand = cand[keep]
        R_cand = np.ascontiguousarray(R_cand[np.ix_(keep, keep)], dtype=np.float64)

    original_arr = cand if original is None else np.asarray(original, dtype=np.int64)[cand]
    rel = gaussian_mi_from_corr(r)
    kish, weight_sum = effective_sample_sizes(w)
    names = None
    if names_all is not None:
        names = [names_all[int(i)] for i in original_arr]

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
) -> CandidatePanel:
    """Build the screened/pruned candidate panel used by cache selectors."""
    if zy is None:
        y_arr = np.asarray(y).ravel()
        if y_arr.shape[0] != cache.n_rows_original:
            raise ValueError(
                f"y has {y_arr.shape[0]} rows but cache was built from "
                f"{cache.n_rows_original} rows"
            )
        ys = y_arr[np.asarray(cache.row_idx)]
        zy_arr = weighted_rank_gauss_1d(ys, cache.sample_weight)
    else:
        zy_arr = np.asarray(zy, dtype=np.float64).ravel()
        if zy_arr.shape[0] != cache.Z.shape[0]:
            raise ValueError("zy length must match cache rows")

    r = weighted_corr_with_vector(cache.Z, zy_arr, cache.sample_weight)
    p_valid = int(len(r))
    if top_m is None:
        top_m = max(5 * int(k), 250)
    top_m_eff = max(int(top_m), int(k))
    top_m_eff = min(top_m_eff, p_valid)
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
) -> CandidatePanel:
    """Build a candidate panel from fold/bootstrap-local correlations."""
    w_arr = np.asarray(w, dtype=np.float64).ravel()
    Z_arr = np.asarray(Z, dtype=np.float64)
    zy_arr = np.asarray(zy, dtype=np.float64).ravel()
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 2D")
    if Z_arr.shape[0] != zy_arr.shape[0] or Z_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("Z, zy, and w must have matching row counts")

    if local_standardize:
        Z_used = globals()["local_standardize"](Z_arr, w_arr)
        zy_used = globals()["local_standardize"](zy_arr, w_arr).ravel()
        R_all = None
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
