"""Tests for explicit feature-path evaluation helper."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from sift.selection.path_eval import evaluate_feature_path



def _toy_regression_data(n: int = 500):
    rng = np.random.default_rng(42)
    x0 = rng.normal(size=n)
    x1 = 0.05 * rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x0 + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    return X, y



def test_evaluate_feature_path_explicit_grid_returns_expected_fields():
    X, y = _toy_regression_data()

    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1", "x2"],
        k_grid=[1, 3],
        estimator=LinearRegression(),
        scoring="rmse",
        val_frac=0.25,
        random_state=1,
    )

    assert result.k == [1, 3]
    assert result.best_k in {1, 3}
    assert result.features == (["x0"] if result.best_k == 1 else ["x0", "x1", "x2"])
    assert set(result.diagnostics.columns) >= {"k", "score", "std", "n_finite", "n_splits", "best_score"}
    assert len(result.scores) == 2



def test_evaluate_feature_path_callable_scoring_and_weight_handling():
    X, y = _toy_regression_data()
    calls: list[int] = []

    def scorer(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
        calls.append(1)
        return float(np.average(np.abs(y_true - y_pred), weights=weight))

    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1"],
        k_grid=[1, 2],
        estimator=LinearRegression(),
        scoring=scorer,
        val_frac=0.2,
        random_state=2,
        sample_weight=np.ones(len(y)),
    )

    assert len(calls) == 2
    assert result.scores[result.best_k] == result.diagnostics.loc[
        result.diagnostics["k"] == result.best_k, "score"
    ].iloc[0]


def test_evaluate_feature_path_ties_prefer_smallest_k_not_grid_order():
    X, y = _toy_regression_data()

    def tied_scorer(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
        return 1.0

    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1", "x2"],
        k_grid=[3, 1, 2],
        estimator=LinearRegression(),
        scoring=tied_scorer,
        val_frac=0.2,
        random_state=2,
    )

    assert result.best_k == 1
    assert result.features == ["x0"]


def test_evaluate_feature_path_default_estimator_accepts_sample_weight():
    X, y = _toy_regression_data()

    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1", "x2"],
        k_grid=[1, 2, 3],
        scoring="rmse",
        val_frac=0.2,
        random_state=2,
        sample_weight=np.linspace(0.5, 2.0, len(y)),
    )

    assert result.best_k in {1, 2, 3}
    assert np.isfinite(list(result.scores.values())).all()
    assert result.diagnostics["n_finite"].min() == 1



def test_evaluate_feature_path_with_splitter_and_estimator_factory():
    X, y = _toy_regression_data(300)
    k_grid = [1, 2, 3]

    class CountingEstimator:
        def fit(self, X_fit, y_fit, sample_weight=None):
            return self

        def predict(self, X_pred):
            return np.zeros(len(X_pred), dtype=np.float64)

    calls: list[int] = []

    def estimator_factory():
        calls.append(1)
        return CountingEstimator()

    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1", "x2"],
        k_grid=k_grid,
        estimator_factory=estimator_factory,
        scoring="mae",
        splitter=splitter,
    )

    assert result.features is not None
    assert len(result.diagnostics) == len(k_grid)
    assert set(result.diagnostics["n_splits"]) == {3}
    assert result.diagnostics["n_finite"].min() == 3
    assert len(calls) == len(k_grid) * 3


def test_evaluate_feature_path_accepts_precomputed_split_iterable():
    X, y = _toy_regression_data(120)
    splits = [
        (np.arange(0, 60), np.arange(60, 90)),
        (np.arange(30, 90), np.arange(90, 120)),
    ]

    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1", "x2"],
        k_grid=[1, 2],
        estimator=LinearRegression(),
        scoring="rmse",
        splitter=splits,
    )

    assert set(result.diagnostics["n_splits"]) == {2}
    assert result.diagnostics["n_finite"].min() == 2


def test_splitter_object_indices_are_validated():
    X, y = _toy_regression_data(40)

    class InvalidSplitter:
        def split(self, X_split, y_split):
            del X_split, y_split
            yield np.array([0, 1, 100]), np.array([2, 3])

    with pytest.raises(ValueError, match="out of range"):
        evaluate_feature_path(
            X,
            y,
            feature_path=["x0"],
            k_grid=[1],
            splitter=InvalidSplitter(),
        )


def test_splitter_without_groups_argument_rejects_groups():
    X, y = _toy_regression_data(40)

    class UngroupedSplitter:
        def split(self, X_split, y_split):
            del X_split, y_split
            yield np.arange(20), np.arange(20, 40)

    with pytest.raises(TypeError, match="does not accept a groups"):
        evaluate_feature_path(
            X,
            y,
            feature_path=["x0"],
            k_grid=[1],
            splitter=UngroupedSplitter(),
            groups=np.repeat([0, 1], 20),
        )


def test_internal_fit_type_error_is_not_retried_without_weights():
    X, y = _toy_regression_data(40)
    calls = []

    class BrokenEstimator:
        def fit(self, X_fit, y_fit, sample_weight=None):
            del X_fit, y_fit, sample_weight
            calls.append(1)
            raise TypeError("internal shape failure")

        def predict(self, X_pred):
            return np.zeros(len(X_pred))

    estimator = BrokenEstimator()
    with pytest.warns(RuntimeWarning, match="internal shape failure"):
        result = evaluate_feature_path(
            X,
            y,
            feature_path=["x0"],
            k_grid=[1],
            estimator=estimator,
        )

    assert len(calls) == 1
    assert np.isinf(result.scores[1])


def test_partial_fold_failure_cannot_win_feature_path_evaluation():
    n = 60
    X = pd.DataFrame(np.arange(n * 3, dtype=float).reshape(n, 3), columns=list("abc"))
    y = np.ones(n)
    splits = [
        (np.arange(10, n), np.arange(0, 10)),
        (np.arange(0, 30), np.arange(30, n)),
    ]

    class PartialFailureEstimator:
        def fit(self, X_fit, y_fit, sample_weight=None):
            del X_fit, y_fit, sample_weight
            return self

        def predict(self, X_pred):
            if X_pred.shape[1] == 2:
                if len(X_pred) > 20:
                    raise RuntimeError("intentional partial-fold failure")
                return np.ones(len(X_pred))
            return np.zeros(len(X_pred))

    with pytest.warns(RuntimeWarning, match="intentional partial-fold failure"):
        result = evaluate_feature_path(
            X,
            y,
            feature_path=list("abc"),
            k_grid=[1, 2, 3],
            estimator=PartialFailureEstimator(),
            scoring="rmse",
            splitter=splits,
        )

    assert result.diagnostics.set_index("k").loc[2, "n_finite"] == 1
    assert np.isinf(result.scores[2])
    assert result.best_k == 1


def test_evaluate_feature_path_ignores_unused_non_numeric_dataframe_columns():
    X, y = _toy_regression_data(160)
    X["label"] = np.where(np.arange(len(X)) % 2 == 0, "home", "away")

    result = evaluate_feature_path(
        X,
        y,
        feature_path=["x0", "x1"],
        k_grid=[1, 2],
        estimator=LinearRegression(),
        scoring="rmse",
        random_state=4,
    )

    assert result.feature_path == ["x0", "x1"]
    assert result.best_k in {1, 2}


@pytest.mark.parametrize("bad_metric", ["rmsex", 123])
def test_evaluate_feature_path_invalid_scoring_metric_raises(bad_metric):
    X, y = _toy_regression_data(120)

    with pytest.raises(ValueError):
        evaluate_feature_path(
            X,
            y,
            feature_path=["x0", "x1"],
            k_grid=[1, 2],
            estimator=LinearRegression(),
            scoring=bad_metric,  # type: ignore[arg-type]
        )
