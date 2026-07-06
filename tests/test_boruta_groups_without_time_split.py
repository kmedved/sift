import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

import sift.boruta as boruta_mod


def test_boruta_importance_data_test_builds_shadows_after_split(monkeypatch):
    n = 40
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    w = np.ones(n, dtype=np.float64)
    seen_rows = []

    class RecordingEstimator:
        def fit(self, X_fit, y_fit, sample_weight=None):
            self.feature_importances_ = np.ones(X_fit.shape[1], dtype=np.float64)
            return self

    def record_permute_matrix(X_part, **kwargs):
        seen_rows.append(X_part.shape[0])
        return np.zeros_like(X_part)

    monkeypatch.setattr(boruta_mod, "permute_matrix", record_permute_matrix)
    sel = boruta_mod.BorutaSelector(
        task="regression",
        importance="native",
        importance_data="test",
        test_size=0.25,
        max_iter=1,
        verbose=False,
    )

    imp = sel._compute_importance(
        RecordingEstimator(),
        X,
        y,
        w_score=w,
        w_fit=w,
        groups=None,
        time=None,
        seed=0,
        shadow_method="global",
        shadow_mode="columns",
        block_size="auto",
    )

    assert sorted(seen_rows) == [10, 30]
    assert n not in seen_rows
    assert imp.shape[0] == X.shape[1] * 2


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
        shadow_method="within_group",
        shadow_mode="columns",
        block_size="auto",
    )
    assert imp.shape[0] == X.shape[1] * 2


def test_boruta_importance_data_test_rejects_single_group_holdout():
    n = 30
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    y = rng.normal(size=n)
    groups = np.zeros(n, dtype=int)
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

    with pytest.raises(ValueError, match="requires at least 2 groups"):
        sel._compute_importance(
            est,
            X,
            y,
            w_score=w,
            w_fit=w,
            groups=groups,
            time=None,
            seed=0,
            shadow_method="within_group",
            shadow_mode="columns",
            block_size="auto",
        )
