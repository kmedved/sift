import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sift.boruta import BorutaLoopResult, BorutaSelector


def _fake_loop(self, fit_data):
    p = fit_data.X_arr.shape[1]
    status = np.full(p, -1, dtype=np.int8)
    status[: min(2, p)] = 1
    return BorutaLoopResult(
        status=status,
        hits=np.ones(p, dtype=np.int32),
        n_trials=1,
        shadow_thresholds=np.array([0.0]),
        mean_importance=np.arange(p, dtype=np.float64),
    )


def test_boruta_get_feature_names_out_preserves_dataframe_names(monkeypatch):
    X = pd.DataFrame(np.ones((8, 4)), columns=["alpha", "beta", "gamma", "delta"])
    y = np.arange(len(X), dtype=float)
    monkeypatch.setattr(BorutaSelector, "_run_boruta_iterations", _fake_loop)

    selector = BorutaSelector(max_iter=1, verbose=False).fit(X, y)

    np.testing.assert_array_equal(
        selector.get_feature_names_out(), np.asarray(["alpha", "beta"], dtype=object)
    )
    np.testing.assert_array_equal(
        selector.get_feature_names_out(list(X.columns)),
        np.asarray(["alpha", "beta"], dtype=object),
    )
    with pytest.raises(ValueError, match="input_features"):
        selector.get_feature_names_out(["wrong"] * X.shape[1])


def test_boruta_get_feature_names_out_uses_x_names_for_arrays(monkeypatch):
    X = np.ones((8, 4))
    y = np.arange(len(X), dtype=float)
    monkeypatch.setattr(BorutaSelector, "_run_boruta_iterations", _fake_loop)

    selector = BorutaSelector(max_iter=1, verbose=False).fit(X, y)

    np.testing.assert_array_equal(
        selector.get_feature_names_out(), np.asarray(["x0", "x1"], dtype=object)
    )
    with pytest.raises(ValueError, match="same number"):
        selector.get_feature_names_out(["x0"])


def test_boruta_get_feature_names_out_requires_fit():
    with pytest.raises(NotFittedError):
        BorutaSelector(verbose=False).get_feature_names_out()
