import numpy as np
from scipy.special import ndtri

from sift.estimators.copula import greedy_corr_prune, weighted_corr_with_vector, weighted_rank_gauss_1d


def _weighted_rank_gauss_1d_reference(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x)
    m = mask.sum()
    if m <= 1:
        return np.zeros_like(x, dtype=np.float32)

    x_valid = x[mask]
    w_valid = w[mask]
    order = np.argsort(x_valid, kind="mergesort")
    x_sorted = x_valid[order]
    w_sorted = w_valid[order]
    total = float(w_sorted.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(x, dtype=np.float32)

    ranks = np.empty_like(w_sorted, dtype=np.float64)
    cum_weight = 0.0
    start = 0
    while start < m:
        stop = start + 1
        while stop < m and x_sorted[stop] == x_sorted[start]:
            stop += 1
        block_weight = float(w_sorted[start:stop].sum())
        ranks[start:stop] = cum_weight + 0.5 * block_weight
        cum_weight += block_weight
        start = stop

    u = np.clip(ranks / total, 1e-6, 1 - 1e-6)
    z = ndtri(u)
    z_mean = np.dot(w_sorted, z) / total
    z_centered = z - z_mean
    z_var = np.dot(w_sorted, z_centered**2) / total
    z_std = np.sqrt(z_var) if z_var > 1e-12 else 1.0
    inv_order = np.argsort(order)
    out = np.zeros_like(x, dtype=np.float32)
    out[mask] = (z_centered / z_std)[inv_order].astype(np.float32)
    return out


def test_weighted_rank_gauss_matches_scalar_reference_for_continuous_and_ties():
    rng = np.random.default_rng(123)
    continuous = rng.normal(size=512)
    tied = rng.choice(np.array([-2.0, -1.0, 0.0, 1.0, 3.0]), size=512)
    tied[::23] = np.nan
    weights = rng.lognormal(mean=0.0, sigma=0.6, size=512)

    for x in (continuous, tied):
        np.testing.assert_array_equal(
            weighted_rank_gauss_1d(x, weights),
            _weighted_rank_gauss_1d_reference(x, weights),
        )


def test_weighted_rank_gauss_determinism_with_ties():
    x = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.float64)
    rng = np.random.default_rng(0)
    w = rng.random(x.shape[0]).astype(np.float64)

    out1 = weighted_rank_gauss_1d(x, w)
    out2 = weighted_rank_gauss_1d(x, w)

    assert out1.shape == x.shape
    assert np.all(np.isfinite(out1))
    np.testing.assert_allclose(out1, out2)


def test_weighted_rank_gauss_equal_values_share_transform():
    x = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.float64)
    w = np.array([1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 5.0], dtype=np.float64)

    out = weighted_rank_gauss_1d(x, w)

    for value in np.unique(x):
        tied = out[x == value]
        np.testing.assert_allclose(tied, tied[0])


def test_weighted_rank_gauss_weighted_ties_are_row_order_invariant():
    x = np.array([3, 1, 1, 2, 2, 2, 1, 3, 4, 4, 4, 2], dtype=np.float64)
    w = np.array([1.0, 0.25, 3.0, 1.5, 0.5, 2.25, 4.0, 2.0, 0.75, 5.0, 1.25, 3.5], dtype=np.float64)
    row_id = np.arange(x.shape[0])
    perm = np.array([10, 6, 3, 1, 7, 5, 0, 11, 2, 8, 4, 9])

    baseline_by_row = weighted_rank_gauss_1d(x, w)
    shuffled = weighted_rank_gauss_1d(x[perm], w[perm])
    shuffled_by_row = np.empty_like(shuffled)
    shuffled_by_row[row_id[perm]] = shuffled

    np.testing.assert_allclose(shuffled_by_row, baseline_by_row)


def test_weighted_rank_gauss_binary_feature_has_two_values():
    x = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float64)
    w = np.linspace(0.5, 2.0, x.shape[0], dtype=np.float64)

    out = weighted_rank_gauss_1d(x, w)

    assert len(np.unique(out[x == 0])) == 1
    assert len(np.unique(out[x == 1])) == 1
    assert len(np.unique(out)) == 2


def test_weighted_corr_with_vector_blas_matches_numba_backend():
    rng = np.random.default_rng(321)
    Z = rng.normal(size=(512, 37)).astype(np.float32)
    zy = rng.normal(size=512).astype(np.float32)
    w = rng.lognormal(mean=0.0, sigma=0.4, size=512).astype(np.float32)

    blas = weighted_corr_with_vector(Z, zy, w, backend="blas", batch_size=127)
    numba = weighted_corr_with_vector(Z, zy, w, backend="numba")

    np.testing.assert_allclose(blas, numba, atol=1e-5)


def test_greedy_corr_prune_ties_keep_lowest_candidate_index():
    candidates = np.arange(50, dtype=np.int64)
    scores = np.ones(50, dtype=np.float64)
    Rxx = np.eye(50, dtype=np.float64)
    Rxx[0, :] = 1.0
    Rxx[:, 0] = 1.0

    keep = greedy_corr_prune(candidates, Rxx, scores, threshold=0.95)

    np.testing.assert_array_equal(keep, np.array([0], dtype=np.int64))
