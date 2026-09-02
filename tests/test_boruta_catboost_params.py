import numpy as np
import pytest

pytest.importorskip("catboost")

pytestmark = pytest.mark.catboost
from catboost import CatBoostRegressor

from sift.boruta import _clone_estimator, _set_n_estimators


def test_clone_estimator_does_not_create_catboost_random_state_alias():
    est0 = CatBoostRegressor(iterations=5, random_seed=0, verbose=False)
    est = _clone_estimator(est0, seed=123)

    params = est.get_params()
    seed_val = params.get("random_seed", params.get("random_state"))
    assert seed_val == 123
    # Ensure we don't have both synonyms set to non-None values
    rs = params.get("random_state")
    rseed = params.get("random_seed")
    assert not (rs is not None and rseed is not None)


def test_set_n_estimators_uses_iterations_for_catboost():
    est = CatBoostRegressor(iterations=5, random_seed=0, verbose=False)
    _set_n_estimators(est, 10)

    params = est.get_params()
    iter_val = params.get("iterations", params.get("n_estimators"))
    assert iter_val == 10
    # Ensure we don't have conflicting synonyms
    iters = params.get("iterations")
    n_est = params.get("n_estimators")
    assert not (iters is not None and n_est is not None)


def test_clone_estimator_respects_existing_catboost_seed_alias():
    est0 = CatBoostRegressor(iterations=5, verbose=False)
    if "random_state" not in est0.get_params():
        pytest.skip("CatBoostRegressor does not expose random_state in this version.")
    est0.set_params(random_state=0)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = rng.normal(size=30)

    est = _clone_estimator(est0, seed=123)
    est.fit(X, y)


def test_catboost_fit_still_works_after_helpers():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    y = rng.normal(size=50)

    est0 = CatBoostRegressor(iterations=5, random_seed=0, verbose=False)
    est = _clone_estimator(est0, seed=123)
    _set_n_estimators(est, 10)

    # Should not raise "only one of the parameters ..." errors
    est.fit(X, y)
