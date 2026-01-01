import numpy as np
from sift import select_jmi, select_jmim


def test_jmi_classic_accepts_weights():
    rng = np.random.default_rng(42)
    n, p = 100, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    # r2 estimator
    selected = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="r2",
        verbose=False,
    )
    assert len(selected) == 3
    assert "x0" in selected  # should pick the most predictive feature

    # binned estimator
    selected_binned = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="binned",
        verbose=False,
    )
    assert len(selected_binned) == 3


def test_jmim_classic_accepts_weights():
    rng = np.random.default_rng(42)
    n, p = 100, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    selected = select_jmim(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="r2",
        verbose=False,
    )
    assert len(selected) == 3


def test_weight_scaling_invariance_jmi():
    """Weights scaled by constant should give same results."""
    rng = np.random.default_rng(123)
    n, p = 80, 6
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2 + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    sel1 = select_jmi(X, y, k=3, task="regression", sample_weight=w, estimator="r2", verbose=False)
    sel2 = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w * 10,
        estimator="r2",
        verbose=False,
    )
    sel3 = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w / w.sum(),
        estimator="r2",
        verbose=False,
    )

    assert sel1 == sel2 == sel3
