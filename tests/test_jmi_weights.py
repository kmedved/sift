import numpy as np
import pytest
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


def test_ksg_jmi_allows_unweighted_selection():
    rng = np.random.default_rng(321)
    n, p = 35, 5
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1

    selected = select_jmi(
        X,
        y,
        k=2,
        task="regression",
        estimator="ksg",
        top_m=4,
        verbose=False,
    )

    assert len(selected) <= 2


@pytest.mark.parametrize("selector", [select_jmi, select_jmim])
def test_ksg_public_selectors_reject_sample_weight(selector):
    rng = np.random.default_rng(654)
    n, p = 30, 5
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.1

    with pytest.raises(ValueError, match="ksg.*sample_weight"):
        selector(
            X,
            y,
            k=2,
            task="regression",
            estimator="ksg",
            sample_weight=np.ones(n),
            verbose=False,
        )


def test_ksg_low_level_rejects_sample_weight():
    from sift.selection.loops import jmi_select

    rng = np.random.default_rng(654)
    n, p = 30, 5
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = X[:, 0] + rng.normal(size=n).astype(np.float32) * 0.1
    relevance = np.linspace(1.0, 0.2, p)

    with pytest.raises(ValueError, match="ksg.*sample_weight"):
        jmi_select(
            X,
            y,
            k=2,
            relevance=relevance,
            mi_estimator="ksg",
            sample_weight=np.ones(n),
        )
