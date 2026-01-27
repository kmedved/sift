import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

import sift.boruta as boruta_mod


def test_boruta_groups_without_time_does_not_call_train_test_split(monkeypatch):
    n = 60
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)
    groups = np.repeat(np.arange(10), 6)

    def boom(*args, **kwargs):
        raise AssertionError("train_test_split should not be called when groups are provided")

    monkeypatch.setattr(boruta_mod, "train_test_split", boom)

    sel = boruta_mod.BorutaSelector(
        task="regression",
        importance="native",
        importance_data="test",
        test_size=0.3,
        max_iter=1,
        verbose=False,
    )

    est = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=0)
    w = np.ones(n, dtype=np.float64)

    imp = sel._compute_importance(
        est,
        X,
        y,
        w_score=w,
        w_fit=w,
        groups=groups,
        time=None,
        seed=0,
    )
    assert imp.shape[0] == X.shape[1]
