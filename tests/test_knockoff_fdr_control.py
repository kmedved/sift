import numpy as np

from sift.selection.knockoff_filter import select_fdr


def _ar1_cov(p: int, rho: float = 0.5) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _reference_design(seed: int, *, n: int = 800, p: int = 40, n_signal: int = 8):
    rng = np.random.default_rng(seed)
    Sigma = _ar1_cov(p, rho=0.5)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    beta = np.zeros(p)
    beta[:n_signal] = np.linspace(1.6, 1.0, n_signal)
    y = X @ beta + rng.normal(scale=1.0, size=n)
    return X, y, set(range(n_signal))


def _weight_sensitive_design(seed: int, *, n: int = 800, p: int = 40, n_signal: int = 8):
    rng = np.random.default_rng(seed)
    Sigma = _ar1_cov(p, rho=0.3)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    beta = np.zeros(p)
    beta[:n_signal] = np.linspace(1.4, 0.9, n_signal)
    half = n // 2
    noise = rng.normal(scale=1.0, size=n)
    y = np.empty(n, dtype=np.float64)
    y[:half] = X[:half] @ beta + noise[:half]
    y[half:] = -(X[half:] @ beta) + noise[half:]
    weights = np.empty(n, dtype=np.float64)
    weights[:half] = 1.0
    weights[half:] = 1e-3
    return X, y, weights, set(range(n_signal))


def _fdp_power(selected, truth: set[int]) -> tuple[float, float]:
    selected_set = set(selected or [])
    false = len(selected_set - truth)
    true = len(selected_set & truth)
    return false / max(1, len(selected_set)), true / len(truth)


def _jaccard(a, b) -> float:
    a_set = set(a or [])
    b_set = set(b or [])
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


def test_default_relevance_reference_calibration_and_power():
    fdps = []
    powers = []

    for seed in range(30):
        X, y, truth = _reference_design(seed)
        result = select_fdr(X, y, q=0.2, statistic="relevance", random_state=seed, verbose=False)
        fdp, power = _fdp_power(result.selected_indices, truth)
        fdps.append(fdp)
        powers.append(power)

    assert float(np.mean(fdps)) <= 0.30
    assert float(np.mean(powers)) >= 0.60


def test_cefsplus_reference_calibration_and_power():
    fdps = []
    powers = []

    for seed in range(15):
        X, y, truth = _reference_design(seed)
        result = select_fdr(X, y, q=0.2, statistic="cefsplus", random_state=seed, verbose=False)
        fdp, power = _fdp_power(result.selected_indices, truth)
        fdps.append(fdp)
        powers.append(power)

    assert float(np.mean(fdps)) <= 0.30
    assert float(np.mean(powers)) >= 0.80


def test_cefsplus_strong_wide_signal_sanity():
    rng = np.random.default_rng(123)
    n = 3_000
    p = 1_000
    X = rng.standard_normal((n, p), dtype=np.float32)
    beta = np.zeros(p, dtype=np.float32)
    beta[:5] = 2.0
    y = X @ beta + rng.standard_normal(n).astype(np.float32)

    result = select_fdr(X, y, q=0.2, statistic="cefsplus", random_state=0, verbose=False)

    assert len(set(result.selected_indices) & set(range(5))) >= 4


def test_mvr_relevance_power_exceeds_equi_on_correlated_design():
    equi_fdps = []
    mvr_fdps = []
    equi_powers = []
    mvr_powers = []
    signal_indices = np.linspace(0, 19, 4, dtype=int)
    truth = set(signal_indices)

    for seed in range(10):
        data_seed = 10_000 + seed
        rng = np.random.default_rng(data_seed)
        Sigma = _ar1_cov(40, rho=0.7)
        X = rng.multivariate_normal(np.zeros(40), Sigma, size=300)
        beta = np.zeros(40)
        beta[signal_indices] = np.linspace(1.2, 0.6, signal_indices.shape[0])
        y = X @ beta + rng.normal(size=300)

        equi = select_fdr(
            X,
            y,
            q=0.2,
            statistic="relevance",
            s_method="equi",
            random_state=seed,
            verbose=False,
        )
        mvr = select_fdr(
            X,
            y,
            q=0.2,
            statistic="relevance",
            s_method="mvr",
            random_state=seed,
            verbose=False,
        )
        fdp, power = _fdp_power(equi.selected_indices, truth)
        equi_fdps.append(fdp)
        equi_powers.append(power)
        fdp, power = _fdp_power(mvr.selected_indices, truth)
        mvr_fdps.append(fdp)
        mvr_powers.append(power)

    assert float(np.mean(equi_fdps)) <= 0.30
    assert float(np.mean(mvr_fdps)) <= 0.30
    assert float(np.mean(mvr_powers)) > float(np.mean(equi_powers))


def test_weighted_reference_path_is_weight_sensitive():
    weighted_fdps = []
    weighted_powers = []
    unweighted_powers = []

    for seed in range(10):
        X, y, weights, truth = _weight_sensitive_design(seed)
        weighted = select_fdr(
            X,
            y,
            q=0.2,
            sample_weight=weights,
            subsample=None,
            random_state=seed,
            verbose=False,
        )
        unweighted = select_fdr(X, y, q=0.2, subsample=None, random_state=seed, verbose=False)
        fdp, power = _fdp_power(weighted.selected_indices, truth)
        weighted_fdps.append(fdp)
        weighted_powers.append(power)
        _, unweighted_power = _fdp_power(unweighted.selected_indices, truth)
        unweighted_powers.append(unweighted_power)
        assert weighted.selector_metadata["weighted_model"] is True
        assert not np.allclose(weighted.W["W"].to_numpy(), unweighted.W["W"].to_numpy())

    assert float(np.mean(weighted_fdps)) <= 0.30
    assert float(np.mean(weighted_powers)) >= 0.80
    assert float(np.mean(unweighted_powers)) <= 0.20


def test_feature_group_global_null_selects_rarely():
    selected_any = 0
    feature_groups = np.repeat(np.arange(10), 4)

    for seed in range(20):
        rng = np.random.default_rng(500 + seed)
        X = rng.multivariate_normal(np.zeros(40), _ar1_cov(40, rho=0.4), size=500)
        y = rng.normal(size=500)
        result = select_fdr(
            X,
            y,
            q=0.2,
            feature_groups=feature_groups,
            random_state=seed,
            verbose=False,
        )
        selected_any += int(bool(result.selected_indices))

    assert selected_any <= 8


def test_global_null_selects_rarely_and_zero_target_selects_nothing():
    selected_any = 0
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        X = rng.multivariate_normal(np.zeros(40), _ar1_cov(40, rho=0.5), size=800)
        y = rng.normal(size=800)
        result = select_fdr(X, y, q=0.2, random_state=seed, verbose=False)
        selected_any += int(bool(result.selected_indices))

    assert selected_any <= 12

    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(np.zeros(40), _ar1_cov(40, rho=0.5), size=200)
    zero = select_fdr(X, np.ones(200), q=0.2, random_state=0, verbose=False)
    assert zero.selected_features == []
    assert np.isinf(zero.threshold)


def test_derandomized_selection_is_more_stable_on_fixed_weak_signal_design():
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(np.zeros(40), _ar1_cov(40, rho=0.6), size=500)
    beta = np.zeros(40)
    beta[:8] = np.linspace(0.4, 0.22, 8)
    y = X @ beta + rng.normal(scale=1.5, size=500)

    single_1 = select_fdr(X, y, q=0.25, offset=0, random_state=1, verbose=False)
    single_2 = select_fdr(X, y, q=0.25, offset=0, random_state=2, verbose=False)
    derand_1 = select_fdr(
        X,
        y,
        q=0.25,
        offset=0,
        n_draws=9,
        eta=0.5,
        random_state=1,
        verbose=False,
    )
    derand_2 = select_fdr(
        X,
        y,
        q=0.25,
        offset=0,
        n_draws=9,
        eta=0.5,
        random_state=2,
        verbose=False,
    )

    assert _jaccard(derand_1.selected_indices, derand_2.selected_indices) >= _jaccard(
        single_1.selected_indices,
        single_2.selected_indices,
    )
