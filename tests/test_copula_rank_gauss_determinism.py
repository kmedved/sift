import numpy as np

from sift.estimators.copula import weighted_rank_gauss_1d


def test_weighted_rank_gauss_determinism_with_ties():
    x = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.float64)
    rng = np.random.default_rng(0)
    w = rng.random(x.shape[0]).astype(np.float64)

    out1 = weighted_rank_gauss_1d(x, w)
    out2 = weighted_rank_gauss_1d(x, w)

    assert out1.shape == x.shape
    assert np.all(np.isfinite(out1))
    np.testing.assert_allclose(out1, out2)
