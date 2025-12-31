import numpy as np
import pandas as pd
import pytest

from sift import select_cefsplus, select_jmi, select_jmim, select_mrmr


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 20), columns=[f"f{i}" for i in range(20)])
    y = X["f0"] + 0.5 * X["f1"] + np.random.randn(500) * 0.1
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 20), columns=[f"f{i}" for i in range(20)])
    y = (X["f0"] + X["f1"] > 0).astype(int)
    return X, y


def test_mrmr_regression_returns_k(regression_data):
    X, y = regression_data
    result = select_mrmr(X, y, k=5, task="regression", verbose=False)
    assert len(result) == 5


def test_mrmr_classif_returns_k(classification_data):
    X, y = classification_data
    result = select_mrmr(X, y, k=5, task="classification", verbose=False)
    assert len(result) == 5


def test_cefsplus_returns_k(regression_data):
    X, y = regression_data
    result = select_cefsplus(X, y, k=5, verbose=False)
    assert len(result) <= 5


def test_jmi_fast_deterministic(regression_data):
    X, y = regression_data
    r1 = select_jmi(X, y, k=5, task="regression", estimator="gaussian", random_state=0)
    r2 = select_jmi(X, y, k=5, task="regression", estimator="gaussian", random_state=0)
    assert r1 == r2


def test_jmim_returns_k(regression_data):
    X, y = regression_data
    result = select_jmim(X, y, k=5, task="regression")
    assert len(result) == 5


def test_classification_string_labels():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(["spam", "ham"] * 50)

    result = select_mrmr(X, y, k=3, task="classification", verbose=False)
    assert len(result) == 3
