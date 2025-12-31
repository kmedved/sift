import numpy as np

from sift.estimators import joint_mi


def test_r2_joint_mi_shape():
    rng = np.random.default_rng(42)
    n, p = 200, 10
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    selected = X[:, 0]
    candidates = X[:, 1:]

    scores = joint_mi.r2_joint_mi(selected, candidates, y)

    assert scores.shape == (p - 1,)


def test_binned_joint_mi_returns_nonnegative():
    rng = np.random.default_rng(123)
    n, p = 100, 6
    X = rng.normal(size=(n, p))
    y = (X[:, 0] > 0).astype(int)

    selected = X[:, 0]
    candidates = X[:, 1:]

    scores = joint_mi.binned_joint_mi(selected, candidates, y, y_kind="discrete")

    assert scores.shape == (p - 1,)
    assert (scores >= 0).all()
