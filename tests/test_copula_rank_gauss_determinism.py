import numpy as np

from sift.estimators.copula import greedy_corr_prune, weighted_rank_gauss_1d


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


def test_greedy_corr_prune_ties_keep_lowest_candidate_index():
    candidates = np.arange(50, dtype=np.int64)
    scores = np.ones(50, dtype=np.float64)
    Rxx = np.eye(50, dtype=np.float64)
    Rxx[0, :] = 1.0
    Rxx[:, 0] = 1.0

    keep = greedy_corr_prune(candidates, Rxx, scores, threshold=0.95)

    np.testing.assert_array_equal(keep, np.array([0], dtype=np.int64))
