"""Compare rejects unused overrides and reports the design actually scored."""


import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit

from sift import CEFSPlusSelector, compare


class _ZeroWidthTransform(BaseEstimator):
    def fit(self, X, y):
        self.selected_indices_ = np.array([0])
        return self

    def transform(self, X):
        return np.empty((len(X), 0))


@pytest.mark.parametrize("mode", ["cv", "in_sample_path"])
def test_zero_width_scoring_is_reported_empty_even_with_raw_selection(mode):
    X = np.arange(60.0).reshape(30, 2)
    y = np.arange(30.0)
    result = compare({"zero": _ZeroWidthTransform}, X, y, mode=mode, cv=3)
    assert result.scores["empty"].all()
    assert result.scores["n_encoded_columns"].eq(0).all()
    assert result.scores["k"].eq(1).all()
    assert result.summary["n_empty"].tolist() == [3]
    assert np.isfinite(result.scores["score"]).all()
    if mode == "in_sample_path":
        assert result.prefix_scores["n_encoded_columns"].eq(0).all()


@pytest.mark.parametrize("value", [False, None, "0.2", np.nan, np.inf, -1, 0, 1, 5, 0.7])
def test_invalid_or_unused_val_frac_fails_before_factory_work(value):
    def factory():
        raise AssertionError("invalid val_frac should fail before selector construction")

    with pytest.raises(ValueError, match="val_frac"):
        compare({"s": factory}, np.ones((30, 2)), np.arange(30), val_frac=value)


def test_requested_holdout_size_belongs_to_cv_splitter():
    X = np.arange(60.0).reshape(30, 2)
    result = compare(
        {"zero": _ZeroWidthTransform}, X, np.arange(30),
        cv=ShuffleSplit(n_splits=1, test_size=0.7, random_state=9),
    )
    assert result.folds["n_val"].tolist() == [21]


@pytest.mark.parametrize("mode", ["cv", "in_sample_path"])
def test_compare_allows_labels_without_optional_identity_hash(mode):
    rng = np.random.default_rng(23)
    X = pd.DataFrame(
        rng.normal(size=(60, 4)), columns=[object() for _ in range(4)]
    )
    y = X.iloc[:, 0].to_numpy() + rng.normal(size=60)
    result = compare(
        {"cefs": lambda: CEFSPlusSelector(k=2, verbose=False)},
        X, y, mode=mode, cv=3,
    )
    assert result.diagnostics["raw_columns_hash"] is None
    assert np.isfinite(result.scores["score"]).all()
