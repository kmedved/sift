import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

from sift.estimators.knockoffs import (
    _diagonal_s_loss,
    _min_noise_eig,
    _mvr_me_coordinate_delta,
    fit_gaussian_knockoffs,
    gaussian_knockoff_mean,
    sample_gaussian_knockoffs,
)


def _ar1_cov(p: int, rho: float = 0.4) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _noise_cov(model):
    D = np.diag(model.s)
    cf = cho_factor(model.Sigma_g, lower=True, check_finite=False)
    return 2.0 * D - D @ cho_solve(cf, D, check_finite=False)


def _old_heuristic_s_reference(Sigma: np.ndarray, equi: np.ndarray) -> np.ndarray:
    cf = cho_factor(Sigma, lower=True, check_finite=False)
    inv_diag = np.diag(cho_solve(cf, np.eye(Sigma.shape[0]), check_finite=False))
    base = np.maximum(np.minimum(2.0 / np.maximum(inv_diag, 1e-12), 1.0), equi)
    lo = 0.0
    hi = 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        candidate = base * mid * (1.0 - 1e-6)
        if np.linalg.eigvalsh(2.0 * Sigma - np.diag(candidate))[0] >= -1e-10:
            lo = mid
        else:
            hi = mid
    return base * lo * (1.0 - 1e-6)


def _heterogeneous_cov() -> np.ndarray:
    left = _ar1_cov(24, rho=0.9)
    right = np.eye(24)
    return np.block(
        [
            [left, np.zeros((left.shape[0], right.shape[0]))],
            [np.zeros((right.shape[0], left.shape[0])), right],
        ]
    )


@pytest.mark.slow
def test_sample_gaussian_knockoffs_matches_analytic_joint_covariance():
    rng = np.random.default_rng(0)
    Sigma = _ar1_cov(6, rho=0.35)
    model = fit_gaussian_knockoffs(Sigma, min_eig=1e-3)
    Z = rng.multivariate_normal(np.zeros(Sigma.shape[0]), model.Sigma_g, size=25_000)

    Zt = sample_gaussian_knockoffs(Z.astype(np.float32), model, rng)

    D = np.diag(model.s)
    G = np.block(
        [
            [model.Sigma_g, model.Sigma_g - D],
            [model.Sigma_g - D, model.Sigma_g],
        ]
    )
    empirical = np.cov(np.column_stack([Z, Zt]).T, bias=True)
    np.testing.assert_allclose(empirical, G, atol=0.05)


def test_noise_cholesky_orientation_matches_noise_covariance():
    Sigma = np.array(
        [
            [1.0, 0.55, 0.20],
            [0.55, 1.0, 0.35],
            [0.20, 0.35, 1.0],
        ],
        dtype=np.float64,
    )
    model = fit_gaussian_knockoffs(Sigma, min_eig=1e-3)

    np.testing.assert_allclose(
        model.noise_chol @ model.noise_chol.T,
        _noise_cov(model),
        atol=1e-10,
    )


def test_equi_s_uses_smallest_eigenvalue_with_slack():
    Sigma = np.array(
        [
            [1.0, 0.2, 0.0],
            [0.2, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    model = fit_gaussian_knockoffs(Sigma, min_eig=1e-3)

    expected = min(2.0 * 0.8, 1.0) * (1.0 - 1e-6)
    np.testing.assert_allclose(model.s, np.full(3, expected))


@pytest.mark.parametrize("s_method", ["mvr", "me"])
def test_mvr_and_me_s_methods_are_feasible_and_improve_their_objective(s_method):
    Sigma = _ar1_cov(8, rho=0.75)
    equi = fit_gaussian_knockoffs(Sigma, s_method="equi")
    model = fit_gaussian_knockoffs(Sigma, s_method=s_method)

    assert np.all(model.s > 0.0)
    assert _diagonal_s_loss(Sigma, model.s, objective=s_method) <= _diagonal_s_loss(
        Sigma,
        equi.s,
        objective=s_method,
    )
    assert _min_noise_eig(Sigma, model.s) >= -1e-8
    min_noise_eig = float(np.linalg.eigvalsh(_noise_cov(model)).min())
    assert min_noise_eig >= -1e-8
    np.testing.assert_allclose(model.noise_chol @ model.noise_chol.T, _noise_cov(model), atol=1e-8)


@pytest.mark.parametrize("objective", ["mvr", "me"])
@pytest.mark.parametrize("coord", [1, 4])
def test_mvr_me_coordinate_delta_matches_bruteforce_coordinate_minimum(objective, coord):
    Sigma = _ar1_cov(6, rho=0.4)
    s = fit_gaussian_knockoffs(Sigma, s_method="equi").s.copy()
    A = 2.0 * Sigma - np.diag(s)
    cf = cho_factor(A, lower=True, check_finite=False)
    Ainv = cho_solve(cf, np.eye(Sigma.shape[0]), check_finite=False)
    c = float(Ainv[coord, coord])
    lower = 1e-8 - s[coord]
    upper = (1.0 - 1e-6) / c

    closed = _mvr_me_coordinate_delta(Ainv, s, coord, objective=objective)
    closed = min(max(closed, lower), upper)

    def loss_for_delta(delta):
        candidate = s.copy()
        candidate[coord] += delta
        return _diagonal_s_loss(Sigma, candidate, objective=objective)

    brute = minimize_scalar(loss_for_delta, bounds=(lower, upper), method="bounded", options={"xatol": 1e-12})

    assert brute.success
    assert closed == pytest.approx(float(brute.x), abs=1e-4)


@pytest.mark.parametrize("Sigma", [_ar1_cov(40, rho=0.5), _ar1_cov(40, rho=0.7), _heterogeneous_cov()])
def test_mvr_loss_dominates_equi_and_old_scaled_variance_heuristic(Sigma):
    equi = fit_gaussian_knockoffs(Sigma, s_method="equi")
    mvr = fit_gaussian_knockoffs(Sigma, s_method="mvr")
    heuristic_s = _old_heuristic_s_reference(Sigma, equi.s)

    mvr_loss = _diagonal_s_loss(Sigma, mvr.s, objective="mvr")
    assert mvr_loss <= _diagonal_s_loss(Sigma, equi.s, objective="mvr")
    assert mvr_loss <= _diagonal_s_loss(Sigma, heuristic_s, objective="mvr")
    assert _min_noise_eig(Sigma, mvr.s) >= -1e-8


def test_mvr_accepts_lower_mean_s_when_loss_improves():
    Sigma = _ar1_cov(100, rho=0.7)
    equi = fit_gaussian_knockoffs(Sigma, s_method="equi")
    mvr = fit_gaussian_knockoffs(Sigma, s_method="mvr")

    assert float(np.mean(mvr.s)) < float(np.mean(equi.s))
    assert _diagonal_s_loss(Sigma, mvr.s, objective="mvr") < _diagonal_s_loss(Sigma, equi.s, objective="mvr")
    assert not np.allclose(mvr.s, equi.s)


def test_shrinkage_triggers_warning_and_stores_sigma_g():
    Sigma = np.ones((4, 4), dtype=np.float64)

    with pytest.warns(UserWarning, match="approximate plug-in"):
        model = fit_gaussian_knockoffs(Sigma, min_eig=1e-2)

    assert model.gamma > 0.0
    assert model.lambda_min < 1e-8
    assert np.linalg.eigvalsh(model.Sigma_g).min() == pytest.approx(1e-2, abs=1e-8)


def test_sample_gaussian_knockoffs_is_deterministic_for_same_seed():
    rng = np.random.default_rng(4)
    Sigma = _ar1_cov(5, rho=0.2)
    model = fit_gaussian_knockoffs(Sigma)
    Z = rng.normal(size=(200, 5)).astype(np.float32)

    out1 = sample_gaussian_knockoffs(Z, model, np.random.default_rng(123))
    out2 = sample_gaussian_knockoffs(Z, model, np.random.default_rng(123))
    out3 = sample_gaussian_knockoffs(Z, model, np.random.default_rng(124))

    np.testing.assert_array_equal(out1, out2)
    assert not np.array_equal(out1, out3)


def test_sample_knockoffs_pinned_seed_stream_regression():
    # Guards the seeded-output contract: a change to the noise dtype, the draw
    # order, or numpy's Generator bit-stream must fail this test loudly instead
    # of silently changing every seeded knockoff result. The tolerance absorbs
    # BLAS-level float32 differences across platforms; a stream change produces
    # O(1) differences.
    import pandas as pd

    from sift.estimators.copula import build_cache
    from sift.selection.knockoff_filter import sample_knockoffs

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((64, 4)), columns=["a", "b", "c", "d"])
    cache = build_cache(X, compute_Rxx=True)

    Zt = sample_knockoffs(cache, random_state=123)

    expected = np.array(
        [
            [0.14068142, 1.1330442],
            [1.5409, -0.10475793],
            [0.90301967, -0.21040349],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(Zt[:3, :2], expected, atol=1e-4)


def test_sample_gaussian_knockoffs_accepts_precomputed_mean():
    rng = np.random.default_rng(44)
    Sigma = _ar1_cov(5, rho=0.35)
    model = fit_gaussian_knockoffs(Sigma)
    Z = rng.normal(size=(256, 5)).astype(np.float32)
    mean = gaussian_knockoff_mean(Z, model)

    seed = 12345
    direct = sample_gaussian_knockoffs(Z, model, np.random.default_rng(seed))
    reused = sample_gaussian_knockoffs(Z, model, np.random.default_rng(seed), mean=mean)

    np.testing.assert_array_equal(direct, reused)


@pytest.mark.parametrize("bad_min_eig", [0.0, 1.0, True, np.inf])
def test_fit_gaussian_knockoffs_rejects_bad_min_eig(bad_min_eig):
    with pytest.raises(ValueError, match="min_eig"):
        fit_gaussian_knockoffs(np.eye(3), min_eig=bad_min_eig)


def test_fit_gaussian_knockoffs_rejects_unknown_s_method():
    with pytest.raises(ValueError, match="s_method"):
        fit_gaussian_knockoffs(np.eye(3), s_method="sdp")
