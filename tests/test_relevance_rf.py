import numpy as np

from sift.estimators.relevance import rf_classif, rf_regression


def test_rf_relevance_all_nan_inputs():
    rng = np.random.default_rng(0)
    X = np.full((50, 10), np.nan)
    y_reg = rng.normal(size=50)
    y_cls = rng.integers(0, 2, size=50)

    reg_imp = rf_regression(X, y_reg)
    cls_imp = rf_classif(X, y_cls)

    assert reg_imp.shape == (10,)
    assert cls_imp.shape == (10,)


def test_rf_relevance_mixed_nan_inputs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8))
    X[0, :] = np.nan
    X[5:10, 2] = np.nan
    y_reg = rng.normal(size=40)
    y_cls = rng.integers(0, 3, size=40)

    reg_imp = rf_regression(X, y_reg)
    cls_imp = rf_classif(X, y_cls)

    assert reg_imp.shape == (8,)
    assert cls_imp.shape == (8,)
