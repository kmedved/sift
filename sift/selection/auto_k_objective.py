"""Penalty and objective helpers for automatic k selection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.special import gammaln

from sift._preprocess import ensure_weights

if TYPE_CHECKING:
    from sift.selection.auto_k import AutoKConfig


def _resolve_n_eff_mode(config: AutoKConfig) -> str | float:
    mode = config.n_eff_mode
    if mode == "auto":
        v2_methods = {
            "chi2_stop",
            "forward_stop",
            "perm_gap",
            "knockoff_path",
            "xfit_objective",
            "gaussian_cv",
            "k_posterior",
            "stability",
            "changepoint",
            "consensus",
            "auto",
        }
        if config.k_method in v2_methods or config.objective_penalty in {"ebic", "ric"}:
            return "kish"
        return "weight_sum"
    return mode


def _penalty_weight(config: AutoKConfig, n_eff: float) -> float:
    if config.objective_penalty in {"bic", "mdl", "ebic"}:
        return float(np.log(n_eff))
    if config.objective_penalty == "aic":
        return 2.0
    if config.objective_penalty == "hqc":
        if n_eff <= np.e:
            raise ValueError("n_eff must be > e for objective_penalty='hqc'")
        return float(2.0 * np.log(np.log(n_eff)))
    if config.objective_penalty == "custom":
        assert config.objective_penalty_weight is not None
        return float(config.objective_penalty_weight)
    if config.objective_penalty == "ric":
        return 0.0
    raise ValueError(f"Unknown objective_penalty: {config.objective_penalty!r}")


def _log_comb(n: int, k: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k, dtype=np.float64)
    out = gammaln(float(n) + 1.0) - gammaln(k_arr + 1.0) - gammaln(float(n) - k_arr + 1.0)
    out[(k_arr < 0) | (k_arr > n)] = np.inf
    return out


def _resolve_ebic_gamma(config: AutoKConfig, *, n_eff: float, n_candidates: int) -> float:
    if config.ebic_gamma == "auto":
        if n_candidates <= 1:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - np.log(n_eff) / (2.0 * np.log(n_candidates)))))
    return float(config.ebic_gamma)


def _penalty_array(
    config: AutoKConfig,
    ks: np.ndarray,
    *,
    n_eff: float,
    n_candidates: int | None,
    dimension: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float | None, int | None]:
    penalty_kind = config.objective_penalty
    if penalty_kind in {"ebic", "ric"}:
        if n_candidates is None:
            raise ValueError("n_candidates is required for EBIC/RIC objective penalties")
        n_candidates_int = int(n_candidates)
        if n_candidates_int < 1:
            raise ValueError("n_candidates must be a positive integer")
        if np.max(ks, initial=0) > n_candidates_int:
            raise ValueError("n_candidates must be >= the largest evaluated k")
    else:
        n_candidates_int = None

    dim = ks.astype(np.float64) if dimension is None else np.asarray(dimension, dtype=np.float64)
    if penalty_kind == "ebic":
        gamma = _resolve_ebic_gamma(config, n_eff=n_eff, n_candidates=n_candidates_int)
        penalty = dim * np.log(n_eff) + 2.0 * gamma * _log_comb(n_candidates_int, ks)
        return penalty, float(np.log(n_eff)), gamma, n_candidates_int
    if penalty_kind == "ric":
        gamma = None
        penalty = 2.0 * dim * np.log(float(n_candidates_int))
        return penalty, 2.0 * float(np.log(float(n_candidates_int))), gamma, n_candidates_int

    penalty_weight = _penalty_weight(config, n_eff)
    return penalty_weight * ks.astype(np.float64), penalty_weight, None, n_candidates_int


def _objective_weight_diagnostics(
    sample_weight: Optional[np.ndarray],
    n_samples: int,
    config: AutoKConfig,
) -> tuple[np.ndarray, float, float, float, str]:
    w = ensure_weights(sample_weight, n_samples, normalize=True)
    weight_sum = float(np.sum(w))
    sum_sq = float(np.sum(w * w))
    kish_n_eff = float(weight_sum * weight_sum / sum_sq) if sum_sq > 0.0 else float("nan")
    if config.objective_n_eff is not None:
        n_eff = float(config.objective_n_eff)
        n_eff_source = "objective_n_eff"
    else:
        mode = _resolve_n_eff_mode(config)
        if mode == "kish":
            n_eff = kish_n_eff
            n_eff_source = "kish"
        elif mode == "weight_sum":
            n_eff = weight_sum
            n_eff_source = "selector_weight_sum"
        else:
            n_eff = float(mode)
            n_eff_source = "n_eff_mode"
    if n_eff <= 1.0 or not np.isfinite(n_eff):
        raise ValueError("objective effective sample size must be finite and > 1")
    if config.objective_penalty == "hqc" and n_eff <= np.e:
        raise ValueError("n_eff must be > e for objective_penalty='hqc'")
    return w, weight_sum, kish_n_eff, n_eff, n_eff_source
