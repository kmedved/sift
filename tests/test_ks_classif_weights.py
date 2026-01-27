import numpy as np
import pytest

from sift.estimators.relevance import ks_classif


def test_ks_classif_uses_weights():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = np.zeros(n, dtype=int)
    y[:100] = 1
    X = x.reshape(-1, 1)

    w = np.ones(n, dtype=np.float64)
    w[:10] = 100.0
    X[:10, 0] = 5.0

    s_unw = ks_classif(X, y, w=None)[0]
    s_w = ks_classif(X, y, w=w)[0]
    assert s_w > s_unw


@pytest.mark.parametrize(
    ("weights", "error_message"),
    [
        (np.array([1.0, -1.0, 1.0]), "non-negative"),
        (np.array([1.0, np.nan, 1.0]), "finite"),
    ],
)
def test_ks_classif_invalid_weight_values(weights, error_message):
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    with pytest.raises(ValueError, match=error_message):
        ks_classif(X, y, w=weights)


def test_ks_classif_invalid_weight_length():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])
    w = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="rows"):
        ks_classif(X, y, w=w)


def test_ks_classif_handles_zero_weights():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    w = np.array([0.0, 1.0, 0.0, 1.0])
    out = ks_classif(X, y, w=w)
    assert out.shape == (1,)
    assert np.isfinite(out[0])
