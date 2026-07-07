import numpy as np

from sift import build_cache, select_fdr


def _matrix(n: int, p: int = 4) -> np.ndarray:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.linspace(-1.0, 1.0, n)
    return X


def test_build_cache_weighted_subsample_retains_all_positive_rows_when_under_cap():
    n = 10_000
    X = _matrix(n)
    positive = np.arange(7, 7 + 50 * 101, 101)
    w = np.zeros(n, dtype=np.float64)
    w[positive] = 1.0

    cache = build_cache(X, sample_weight=w, subsample=100, random_state=0)

    np.testing.assert_array_equal(np.sort(cache.row_idx), positive)
    assert cache.sample_weight.shape[0] == positive.shape[0]


def test_build_cache_weighted_subsample_draws_only_positive_rows_when_over_cap():
    n = 10_000
    X = _matrix(n)
    positive = np.arange(0, n, 2)
    w = np.zeros(n, dtype=np.float64)
    w[positive] = 1.0

    cache = build_cache(X, sample_weight=w, subsample=100, random_state=7)

    assert cache.row_idx.shape[0] == 100
    assert np.all(w[cache.row_idx] > 0.0)


def test_build_cache_unweighted_subsample_preserves_seeded_row_choices():
    n = 10_000
    X = _matrix(n)
    expected = np.random.default_rng(11).choice(n, size=100, replace=False)

    cache = build_cache(X, subsample=100, random_state=11)

    np.testing.assert_array_equal(cache.row_idx, expected)


def test_select_fdr_weighted_x_path_handles_sparse_positive_support():
    n = 400
    rng = np.random.default_rng(45)
    X = rng.normal(size=(n, 8))
    positive = np.arange(0, n, 4)
    w = np.zeros(n, dtype=np.float64)
    w[positive] = rng.uniform(0.5, 2.0, size=positive.shape[0])
    y = X[:, 0] + 0.1 * rng.normal(size=n)

    result = select_fdr(
        X,
        y,
        q=0.2,
        sample_weight=w,
        subsample=120,
        random_state=3,
        verbose=False,
    )

    assert result.selector_metadata["weighted_model"] is True
    assert result.selector_metadata["n_rows_used"] == positive.shape[0]
