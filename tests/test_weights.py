import numpy as np
import pytest

from sift import select_cefsplus, select_jmi, select_jmim, select_mrmr


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


def test_filter_selectors_return_up_to_k_when_valid_candidates_are_fewer():
    rng = np.random.default_rng(11)
    n = 80
    signal = rng.normal(size=n)
    X = np.column_stack(
        [
            signal,
            np.ones(n),
            np.ones(n) * 2.0,
            np.ones(n) * -3.0,
        ]
    )
    y = signal + rng.normal(scale=0.05, size=n)

    selected_mrmr = select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        estimator="classic",
        verbose=False,
    )
    selected_jmi = select_jmi(
        X,
        y,
        k=3,
        task="regression",
        estimator="r2",
        verbose=False,
    )
    selected_jmim = select_jmim(
        X,
        y,
        k=3,
        task="regression",
        estimator="r2",
        verbose=False,
    )
    selected_cefs = select_cefsplus(X, y, k=3, verbose=False)

    assert len(selected_mrmr) == 1
    assert len(selected_jmi) == 1
    assert len(selected_jmim) == 1
    assert len(selected_cefs) == 1


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


def test_filter_selectors_reject_invalid_k():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5))
    y = X[:, 0] + rng.normal(size=40) * 0.1

    with pytest.raises(ValueError, match="k must be >= 1"):
        select_mrmr(X, y, k=0, task="regression", estimator="classic", verbose=False)

    with pytest.raises(ValueError, match="k must be >= 1"):
        select_jmi(X, y, k=-1, task="regression", estimator="r2", verbose=False)

    with pytest.raises(ValueError, match="k must be >= 1"):
        select_jmim(X, y, k=0, task="regression", estimator="r2", verbose=False)

    with pytest.raises(ValueError, match="k must be >= 1"):
        select_cefsplus(X, y, k=0, verbose=False)


def test_ksg_jmi_rejects_sample_weight():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] + rng.normal(size=30) * 0.1
    w = np.ones(30)

    with pytest.raises(ValueError, match="ksg.*sample_weight"):
        select_jmi(
            X,
            y,
            k=2,
            task="regression",
            estimator="ksg",
            sample_weight=w,
            verbose=False,
        )
