import numpy as np

from sift import select_cefsplus, select_jmi, select_mrmr


def test_filter_selectors_accept_sample_weight():
    rng = np.random.default_rng(0)
    n, p = 80, 6
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 1.5 + rng.normal(size=n) * 0.1
    w = rng.uniform(0.5, 2.0, size=n)

    selected_mrmr = select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        relevance="f",
        estimator="classic",
        random_state=0,
        verbose=False,
    )
    assert len(selected_mrmr) <= 3

    selected_jmi = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        sample_weight=w,
        estimator="r2",
        random_state=0,
        verbose=False,
    )
    assert len(selected_jmi) <= 3

    selected_cefs = select_cefsplus(
        X,
        y,
        k=3,
        sample_weight=w,
        random_state=0,
        verbose=False,
    )
    assert len(selected_cefs) <= 3


def test_sample_weight_length_mismatch_raises():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = X[:, 0] + rng.normal(size=30) * 0.1
    w_bad = np.ones(29)

    try:
        select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            sample_weight=w_bad,
            relevance="f",
            estimator="classic",
            verbose=False,
        )
    except ValueError as exc:
        assert "sample_weight" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched sample_weight length.")
