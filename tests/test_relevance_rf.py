import numpy as np
import pytest

from sift import select_mrmr
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


@pytest.mark.parametrize(
    ("relevance", "target"),
    [
        (rf_regression, "regression"),
        (rf_classif, "classification"),
    ],
)
def test_rf_relevance_preserves_feature_order_at_large_offsets(relevance, target):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(240, 5))
    y = 2.0 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(size=len(X)) * 0.1
    if target == "classification":
        y = (y > np.median(y)).astype(np.int64)

    shifted = X.copy()
    shifted[:, 0] += 1e10
    np.testing.assert_allclose(relevance(X, y), relevance(shifted, y))


@pytest.mark.parametrize(
    ("relevance", "target"),
    [
        (rf_regression, "regression"),
        (rf_classif, "classification"),
    ],
)
def test_rf_relevance_zero_weight_extreme_does_not_set_feature_scale(relevance, target):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(240, 5))
    y = 2.0 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(size=len(X)) * 0.1
    if target == "classification":
        y = (y > np.median(y)).astype(np.int64)

    weights = np.ones(len(X))
    weights[-1] = 0.0
    extreme = X.copy()
    extreme[-1, 0] = 1e300
    np.testing.assert_allclose(
        relevance(X, y, weights), relevance(extreme, y, weights)
    )


@pytest.mark.parametrize(
    "task",
    ["regression", "classification"],
)
def test_public_mrmr_rf_relevance_is_large_offset_invariant(task):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(240, 5))
    y = 1.5 * X[:, 0] + 0.2 * X[:, 1] + rng.normal(size=len(X)) * 0.1
    if task == "classification":
        y = (y > np.median(y)).astype(np.int64)

    shifted = X.copy()
    shifted[:, 0] += 1e10
    kwargs = dict(
        k=2,
        task=task,
        estimator="classic",
        relevance="rf",
        subsample=None,
        verbose=False,
    )
    assert select_mrmr(X, y, **kwargs) == select_mrmr(shifted, y, **kwargs)
