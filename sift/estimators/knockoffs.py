"""Gaussian Model-X knockoff sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, cholesky, eigh


@dataclass(frozen=True)
class GaussianKnockoffModel:
    """Precomputed sampling operators for one Gaussian knockoff model."""

    s: np.ndarray
    Sigma_g: np.ndarray
    mean_op: np.ndarray
    noise_chol: np.ndarray
    gamma: float
    lambda_min: float


_S_METHODS = ("equi", "mvr", "me")
_PSD_SLACK = 1e-6


def _validate_min_eig(min_eig: float) -> float:
    if isinstance(min_eig, (bool, np.bool_)):
        raise ValueError("min_eig must be a finite float in (0, 1)")
    min_eig_float = float(min_eig)
    if not np.isfinite(min_eig_float) or not 0.0 < min_eig_float < 1.0:
        raise ValueError("min_eig must be a finite float in (0, 1)")
    return min_eig_float


def _validate_sigma(Sigma: np.ndarray) -> np.ndarray:
    Sigma_arr = np.asarray(Sigma, dtype=np.float64)
    if Sigma_arr.ndim != 2 or Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise ValueError("Sigma must be a square 2D array")
    if Sigma_arr.shape[0] == 0:
        raise ValueError("Sigma must contain at least one feature")
    if not np.isfinite(Sigma_arr).all():
        raise ValueError("Sigma must contain only finite values")
    Sigma_arr = 0.5 * (Sigma_arr + Sigma_arr.T)
    return Sigma_arr


def _validate_s_method(s_method: str) -> str:
    key = str(s_method).lower()
    if key not in _S_METHODS:
        valid = "', '".join(_S_METHODS)
        raise ValueError(f"s_method must be one of: '{valid}'")
    return key


def _noise_cholesky(N: np.ndarray) -> np.ndarray:
    N = np.asarray(N, dtype=np.float64)
    scale = float(np.trace(N)) / max(1, N.shape[0])
    jitter = max(scale, 1.0) * 1e-12
    for _ in range(4):
        try:
            return cholesky(N, lower=True, check_finite=False)
        except LinAlgError:
            N = N + np.eye(N.shape[0], dtype=np.float64) * jitter
            jitter *= 10.0

    evals, evecs = eigh(N, check_finite=False)
    evals = np.clip(evals, 0.0, None)
    return evecs @ np.diag(np.sqrt(evals))


def _equi_s(lambda_for_s: float, p: int) -> np.ndarray:
    s_value = min(2.0 * lambda_for_s, 1.0) * (1.0 - _PSD_SLACK)
    return np.full(p, s_value, dtype=np.float64)


def _min_noise_eig(Sigma: np.ndarray, s: np.ndarray) -> float:
    return float(eigh(2.0 * Sigma - np.diag(s), subset_by_index=[0, 0], eigvals_only=True, check_finite=False)[0])


def _diagonal_s_loss(Sigma: np.ndarray, s: np.ndarray, *, objective: str) -> float:
    s_arr = np.asarray(s, dtype=np.float64)
    if s_arr.ndim != 1 or np.any(s_arr <= 0.0) or not np.isfinite(s_arr).all():
        return float(np.inf)
    A = 2.0 * Sigma - np.diag(s_arr)
    try:
        if objective == "mvr":
            cf = cho_factor(A, lower=True, check_finite=False)
            Ainv = cho_solve(cf, np.eye(Sigma.shape[0], dtype=np.float64), check_finite=False)
            return float(np.trace(Ainv) + np.sum(1.0 / s_arr))
        if objective == "me":
            L = cholesky(A, lower=True, check_finite=False)
            return float(-2.0 * np.sum(np.log(np.diag(L))) - np.sum(np.log(s_arr)))
    except LinAlgError:
        return float(np.inf)
    raise ValueError("objective must be 'mvr' or 'me'")


def _mvr_me_coordinate_delta(Ainv: np.ndarray, s: np.ndarray, j: int, *, objective: str) -> float:
    c = float(Ainv[j, j])
    if c <= 0.0 or not np.isfinite(c):
        return 0.0
    if objective == "mvr":
        sqrt_v = float(np.sqrt(Ainv[:, j] @ Ainv[:, j]))
        return float((1.0 - sqrt_v * s[j]) / (sqrt_v + c))
    if objective == "me":
        return float((1.0 - c * s[j]) / (2.0 * c))
    raise ValueError("objective must be 'mvr' or 'me'")


def _solve_mvr_me_s(
    Sigma: np.ndarray,
    *,
    objective: str,
    s_init: np.ndarray,
    sweeps: int = 8,
    rtol: float = 1e-4,
) -> np.ndarray:
    p = Sigma.shape[0]
    s = np.clip(np.asarray(s_init, dtype=np.float64), 1e-8, None)
    if objective not in {"mvr", "me"}:
        raise ValueError("objective must be 'mvr' or 'me'")

    last_valid = s.copy()
    for _ in range(sweeps):
        A = 2.0 * Sigma - np.diag(s)
        try:
            cf = cho_factor(A, lower=True, check_finite=False)
            Ainv = cho_solve(cf, np.eye(p, dtype=np.float64), check_finite=False)
        except LinAlgError:
            return last_valid
        last_valid = s.copy()
        max_rel = 0.0
        for j in range(p):
            c = float(Ainv[j, j])
            if c <= 0.0 or not np.isfinite(c):
                continue
            delta = _mvr_me_coordinate_delta(Ainv, s, j, objective=objective)
            delta = min(float(delta), (1.0 - _PSD_SLACK) / c)
            delta = max(delta, 1e-8 - s[j])
            denom = 1.0 - delta * c
            if abs(delta) < 1e-14 or denom <= 0.0 or not np.isfinite(denom):
                continue
            u = Ainv[:, j].copy()
            Ainv += (delta / denom) * np.outer(u, u)
            s[j] += delta
            max_rel = max(max_rel, abs(delta) / max(float(s[j]), 1e-8))
        if max_rel < rtol:
            break
    return s


def _solve_diagonal_s(Sigma: np.ndarray, *, s_method: str, equi: np.ndarray) -> np.ndarray:
    if s_method == "equi":
        return equi

    s = _solve_mvr_me_s(Sigma, objective=s_method, s_init=equi)
    if _min_noise_eig(Sigma, s) < -1e-8:
        return equi
    if _diagonal_s_loss(Sigma, s, objective=s_method) > _diagonal_s_loss(Sigma, equi, objective=s_method):
        return equi
    return s


def fit_gaussian_knockoffs(
    Sigma: np.ndarray,
    *,
    s_method: str = "equi",
    min_eig: float = 1e-3,
) -> GaussianKnockoffModel:
    """Fit second-order Gaussian knockoff sampling operators."""

    s_method_key = _validate_s_method(s_method)
    min_eig_float = _validate_min_eig(min_eig)
    Sigma_arr = _validate_sigma(Sigma)
    p = Sigma_arr.shape[0]

    lambda_min = float(eigh(Sigma_arr, subset_by_index=[0, 0], eigvals_only=True)[0])
    if lambda_min >= min_eig_float:
        gamma = 0.0
        Sigma_g = Sigma_arr.copy()
        lambda_for_s = lambda_min
    else:
        gamma = float((min_eig_float - lambda_min) / (1.0 - lambda_min))
        Sigma_g = (1.0 - gamma) * Sigma_arr + gamma * np.eye(p, dtype=np.float64)
        lambda_for_s = min_eig_float
        warnings.warn(
            "Gaussian knockoff covariance was shrunk; this is an approximate "
            "plug-in model and exact Model-X FDR is not claimed "
            f"(lambda_min={lambda_min:.6g}, gamma={gamma:.6g}).",
            UserWarning,
            stacklevel=2,
        )

    s = _solve_diagonal_s(Sigma_g, s_method=s_method_key, equi=_equi_s(lambda_for_s, p))
    D = np.diag(s)

    cf = cho_factor(Sigma_g, lower=True, check_finite=False)
    V = cho_solve(cf, D, check_finite=False)
    mean_op = np.eye(p, dtype=np.float64) - V
    N = 2.0 * D - s[:, None] * V
    N = 0.5 * (N + N.T)
    noise_chol = _noise_cholesky(N)

    return GaussianKnockoffModel(
        s=s,
        Sigma_g=Sigma_g,
        mean_op=mean_op,
        noise_chol=noise_chol,
        gamma=gamma,
        lambda_min=lambda_min,
    )


def sample_gaussian_knockoffs(
    Z: np.ndarray,
    model: GaussianKnockoffModel,
    rng: np.random.Generator,
    *,
    mean: np.ndarray | None = None,
) -> np.ndarray:
    """Sample Gaussian knockoffs in row blocks."""

    Z_arr = np.asarray(Z)
    if Z_arr.ndim != 2:
        raise ValueError("Z must be a 2D array")
    n, p = Z_arr.shape
    if model.mean_op.shape != (p, p) or model.noise_chol.shape != (p, p):
        raise ValueError("model dimensions must match Z columns")
    mean_arr = None if mean is None else np.asarray(mean)
    if mean_arr is not None and mean_arr.shape != (n, p):
        raise ValueError("mean must have the same shape as Z")

    out = np.empty((n, p), dtype=np.float32)
    mean_op32 = np.asarray(model.mean_op, dtype=np.float32)
    noise_chol_t32 = np.asarray(model.noise_chol.T, dtype=np.float32)
    # Sequential block draws consume the same stream as one full draw; this is a
    # memory knob, not part of the seeded-output contract.
    block_size = 8192
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        if mean_arr is None:
            Z_block = np.asarray(Z_arr[start:stop], dtype=np.float32)
            mean_block = Z_block @ mean_op32
        else:
            mean_block = np.asarray(mean_arr[start:stop], dtype=np.float32)
        E = rng.standard_normal((stop - start, p), dtype=np.float32)
        out[start:stop] = mean_block + E @ noise_chol_t32
    return out


def gaussian_knockoff_mean(Z: np.ndarray, model: GaussianKnockoffModel) -> np.ndarray:
    """Compute the deterministic conditional mean term for repeated draws."""

    Z_arr = np.asarray(Z)
    if Z_arr.ndim != 2:
        raise ValueError("Z must be a 2D array")
    n, p = Z_arr.shape
    if model.mean_op.shape != (p, p):
        raise ValueError("model dimensions must match Z columns")
    out = np.empty((n, p), dtype=np.float32)
    mean_op32 = np.asarray(model.mean_op, dtype=np.float32)
    block_size = 8192
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        out[start:stop] = np.asarray(Z_arr[start:stop], dtype=np.float32) @ mean_op32
    return out


__all__ = [
    "GaussianKnockoffModel",
    "fit_gaussian_knockoffs",
    "gaussian_knockoff_mean",
    "sample_gaussian_knockoffs",
]
