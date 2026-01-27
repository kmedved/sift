import numpy as np
import pytest

pytest.importorskip("catboost")
from catboost import CatBoostRegressor

from sift.boruta import _clone_estimator, _set_n_estimators


def test_clone_estimator_does_not_create_catboost_random_state_alias():
    est0 = CatBoostRegressor(iterations=5, random_seed=0, verbose=False)
    est = _clone_estimator(est0, seed=123)

    params = est.get_params()
    assert "random_seed" in params
    assert params["random_seed"] == 123
    assert "random_state" not in params


def test_set_n_estimators_uses_iterations_for_catboost():
    est = CatBoostRegressor(iterations=5, random_seed=0, verbose=False)
    _set_n_estimators(est, 10)

    params = est.get_params()
    assert params["iterations"] == 10
    assert "n_estimators" not in params


def test_catboost_fit_still_works_after_helpers():
    X = np.random.randn(50, 5)
    y = np.random.randn(50)

    est0 = CatBoostRegressor(iterations=5, random_seed=0, verbose=False)
    est = _clone_estimator(est0, seed=123)
    _set_n_estimators(est, 10)

    est.fit(X, y)
