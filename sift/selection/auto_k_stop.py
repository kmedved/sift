"""Path-only automatic-k stopping rules for CEFS+ objective curves."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, f

from sift.selection.auto_k import AutoKConfig, validate_auto_k_config


def _objective_gains(objective_path: np.ndarray) -> np.ndarray:
    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)
    if obj.size == 0:
        return np.empty(0, dtype=np.float64)
    gains = np.empty_like(obj)
    gains[0] = obj[0]
    if obj.size > 1:
        gains[1:] = np.diff(obj)
    np.maximum(gains, 0.0, out=gains)
    return gains


def _sidak_max_pvalue(p_single: np.ndarray, m_eff: np.ndarray) -> np.ndarray:
    p1 = np.clip(np.asarray(p_single, dtype=np.float64), 0.0, 1.0)
    m = np.maximum(np.asarray(m_eff, dtype=np.float64), 1.0)
    m = np.broadcast_to(m, p1.shape).astype(np.float64, copy=False)
    out = np.empty_like(p1, dtype=np.float64)
    small = p1 < 1e-12
    certain = p1 >= 1.0
    mid = ~(small | certain)
    out[certain] = 1.0
    out[small] = np.minimum(1.0, m[small] * p1[small])
    out[mid] = -np.expm1(m[mid] * np.log1p(-p1[mid]))
    return np.clip(out, 0.0, 1.0)


def _li_ji_effective_tests(panel_eigs: np.ndarray | None) -> float | None:
    if panel_eigs is None:
        return None
    eigs = np.asarray(panel_eigs, dtype=np.float64).reshape(-1)
    if eigs.size == 0 or not np.isfinite(eigs).all():
        return None
    parts = np.where(eigs >= 1.0, 1.0 + (eigs - np.floor(eigs)), eigs)
    return float(max(1.0, np.sum(parts)))


def _gain_test_frame(
    objective_path: np.ndarray,
    *,
    n_eff: float,
    p_candidates: int,
    m_mode: str = "all",
    panel_eigs: np.ndarray | None = None,
) -> pd.DataFrame:
    gains = _objective_gains(objective_path)
    if gains.size == 0:
        return pd.DataFrame()
    if not np.isfinite(n_eff) or n_eff <= 2.0:
        raise ValueError("n_eff must be finite and > 2 for path gain tests")
    p = int(p_candidates)
    if p < 1:
        raise ValueError("p_candidates must be positive")

    stat_max_k = min(gains.size, max(0, int(np.floor(n_eff)) - 2))
    if stat_max_k <= 0:
        return pd.DataFrame()
    ks = np.arange(1, stat_max_k + 1, dtype=np.int64)
    gains = gains[:stat_max_k]
    nu = n_eff - ks.astype(np.float64) - 1.0
    valid_nu = nu > 0.0

    if m_mode == "panel":
        m_eff = np.maximum(1.0, p - ks + 1.0)
    elif m_mode == "li_ji":
        m_base = _li_ji_effective_tests(panel_eigs)
        if m_base is None:
            m_eff = np.maximum(1.0, p - ks + 1.0)
        else:
            panel_width = max(1, len(panel_eigs))
            m_eff = np.maximum(1.0, m_base * (p - ks + 1.0) / panel_width)
    else:
        m_eff = np.maximum(1.0, p - ks + 1.0)

    F_stat = np.full_like(gains, np.nan, dtype=np.float64)
    p_single = np.ones_like(gains, dtype=np.float64)
    F_stat[valid_nu] = nu[valid_nu] * np.expm1(gains[valid_nu])
    p_single[valid_nu] = f.sf(F_stat[valid_nu], 1.0, nu[valid_nu])
    p_max = _sidak_max_pvalue(p_single, m_eff)
    objective = np.asarray(objective_path, dtype=np.float64).reshape(-1)[:stat_max_k]
    return pd.DataFrame(
        {
            "k": ks,
            "objective": objective,
            "gain": gains,
            "F_stat": F_stat,
            "nu": nu,
            "m_eff": m_eff,
            "p_single": p_single,
            "p_max": p_max,
            "stat_max_k": stat_max_k,
        }
    )


def path_gain_pvalues(
    objective_path: np.ndarray,
    *,
    n_eff: float,
    p_candidates: int,
    m_mode: str = "all",
    panel_eigs: np.ndarray | None = None,
) -> np.ndarray:
    """Return Sidak-corrected max-gain p-values for each evaluated path step."""
    frame = _gain_test_frame(
        objective_path,
        n_eff=n_eff,
        p_candidates=p_candidates,
        m_mode=m_mode,
        panel_eigs=panel_eigs,
    )
    return frame["p_max"].to_numpy(dtype=np.float64) if not frame.empty else np.empty(0)


def select_k_chi2_stop(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    n_eff: float,
    p_candidates: int,
    panel_eigs: np.ndarray | None = None,
) -> tuple[int, pd.DataFrame]:
    """Stop at the first patience-smoothed non-significant max-gain run."""
    validate_auto_k_config(config)
    if config.k_method != "chi2_stop":
        raise ValueError("select_k_chi2_stop requires AutoKConfig(k_method='chi2_stop')")
    objective_eval = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    diag = _gain_test_frame(
        objective_eval,
        n_eff=n_eff,
        p_candidates=p_candidates,
        m_mode=config.m_mode,
        panel_eigs=panel_eigs,
    )
    if diag.empty:
        return 0, diag

    floor = max(0, min(int(config.min_k), int(diag["k"].max())))
    patience = int(config.stop_patience)
    significant = diag["p_max"].to_numpy(dtype=np.float64) <= float(config.alpha)
    bad = ~significant
    selected_k = int(diag["k"].max())
    stopped_by = "max_k"
    run = 0
    for pos, is_bad in enumerate(bad):
        if is_bad:
            run += 1
        else:
            run = 0
        if run >= patience:
            start_pos = pos - patience + 1
            candidate_k = int(diag["k"].iloc[start_pos] - 1)
            selected_k = max(candidate_k, floor)
            stopped_by = "floored" if candidate_k < floor else "test"
            break

    diag = diag.copy()
    diag["significant"] = significant
    diag["selected"] = diag["k"] == selected_k
    diag["alpha"] = float(config.alpha)
    diag["m_mode"] = config.m_mode
    diag["n_eff"] = float(n_eff)
    diag["stopped_by"] = stopped_by
    return selected_k, diag


def select_k_forward_stop(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    n_eff: float,
    p_candidates: int,
    panel_eigs: np.ndarray | None = None,
) -> tuple[int, pd.DataFrame]:
    """Select the largest prefix accepted by the ForwardStop accumulation rule."""
    validate_auto_k_config(config)
    if config.k_method != "forward_stop":
        raise ValueError("select_k_forward_stop requires AutoKConfig(k_method='forward_stop')")
    objective_eval = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    diag = _gain_test_frame(
        objective_eval,
        n_eff=n_eff,
        p_candidates=p_candidates,
        m_mode=config.m_mode,
        panel_eigs=panel_eigs,
    )
    if diag.empty:
        return 0, diag

    eps = np.finfo(np.float64).eps
    pvals = np.clip(diag["p_max"].to_numpy(dtype=np.float64), 0.0, 1.0 - eps)
    Y = -np.log1p(-pvals)
    ks = diag["k"].to_numpy(dtype=np.int64)
    running = np.cumsum(Y) / ks
    floor = max(1, min(int(config.min_k), int(ks.max())))
    eligible = (ks >= floor) & (running <= float(config.alpha))
    selected_k = int(ks[eligible][-1]) if np.any(eligible) else 0

    diag = diag.copy()
    diag["Y"] = Y
    diag["Y_running_mean"] = running
    diag["eligible"] = eligible
    diag["selected"] = diag["k"] == selected_k
    diag["alpha"] = float(config.alpha)
    diag["m_mode"] = config.m_mode
    diag["n_eff"] = float(n_eff)
    diag["stopped_by"] = "forward_stop" if selected_k else "empty"
    return selected_k, diag


def _median_smooth(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width % 2 == 0:
        width += 1
    if width <= 1 or x.size < width:
        return x.copy()
    half = width // 2
    out = x.copy()
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def select_k_changepoint(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale: float,
    n_eff: float,
    p_candidates: int,
) -> tuple[int, pd.DataFrame]:
    """Select k from a noise-floor changepoint on scaled objective gains."""
    validate_auto_k_config(config)
    if config.k_method != "changepoint":
        raise ValueError("select_k_changepoint requires AutoKConfig(k_method='changepoint')")
    objective_eval = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    gains = _objective_gains(objective_eval)
    L = int(gains.size)
    if L == 0:
        return 0, pd.DataFrame()
    effective_max = min(L, max(0, int(np.floor(n_eff)) - 2)) if np.isfinite(n_eff) else L
    effective_max = max(0, effective_max)
    if effective_max <= 0:
        return 0, pd.DataFrame()
    gains = gains[:effective_max]
    ks = np.arange(1, effective_max + 1, dtype=np.int64)
    scale = float(objective_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("objective_scale must be positive and finite")
    x = np.log(scale * gains + 1e-12)

    if effective_max < 3:
        warnings.warn(
            "changepoint requires at least three objective gains; falling back to the method floor.",
            UserWarning,
            stacklevel=2,
        )
        selected_k = max(0, min(int(config.min_k), effective_max))
        floor_not_reached = False
        tail_width = 0
        floor_mu = floor_sigma = analytic = threshold = np.nan
        exceeds = np.zeros(effective_max, dtype=bool)
    else:
        if isinstance(config.floor_window, (int, np.integer)):
            requested = int(config.floor_window)
        else:
            requested = int(np.ceil(float(config.floor_window) * effective_max))
        tail_width = min(effective_max - 1, max(10, requested))
        pre_tail = effective_max - tail_width
        if pre_tail <= 0:
            warnings.warn(
                "changepoint tail window leaves no pre-tail range; falling back to effective max.",
                UserWarning,
                stacklevel=2,
            )
            selected_k = effective_max
            floor_not_reached = False
            floor_mu = floor_sigma = analytic = threshold = np.nan
            exceeds = np.zeros(effective_max, dtype=bool)
        else:
            smooth_width = int(config.stop_patience) if int(config.stop_patience) > 2 else 3
            x_used = _median_smooth(x, smooth_width)
            tail = x_used[-tail_width:]
            floor_mu = float(np.median(tail))
            floor_sigma = float(1.4826 * np.median(np.abs(tail - floor_mu)))
            m_tail = max(1, int(p_candidates) - effective_max + 1)
            analytic = float(np.log(chi2.isf(-np.expm1(np.log(0.5) / m_tail), df=1)))
            threshold = floor_mu + float(config.floor_z) * floor_sigma
            floor_not_reached = bool(floor_mu > analytic + 3.0 * floor_sigma)
            if floor_not_reached:
                warnings.warn(
                    "changepoint noise floor was not reached before max_k; returning effective max.",
                    UserWarning,
                    stacklevel=2,
                )
                selected_k = effective_max
                exceeds = np.zeros(effective_max, dtype=bool)
            else:
                exceeds = x_used > threshold
                pre_exceeds = np.flatnonzero(exceeds[:pre_tail])
                floor = max(0, min(int(config.min_k), effective_max))
                selected_k = int(pre_exceeds[-1] + 1) if pre_exceeds.size else floor

    objective = objective_eval[:effective_max]
    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": objective,
            "gain": gains,
            "log_scaled_gain": x,
            "objective_scale": scale,
            "n_eff": float(n_eff),
            "floor_mu": floor_mu,
            "floor_sigma": floor_sigma,
            "analytic_floor_median": analytic,
            "threshold": threshold,
            "tail_width": tail_width,
            "floor_not_reached": floor_not_reached,
            "exceeds": exceeds,
            "selected": ks == selected_k,
        }
    )
    return int(selected_k), diag
