"""Compatibility contracts for the public permutation-importance function."""

import warnings

import numpy as np
import pandas as pd
import pytest

import sift


class RegimePredictor:
    """Use a different signal column in each deterministic row regime."""

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return np.where(
                X["regime"].to_numpy() < 0.5,
                X["major"].to_numpy(),
                X["minor"].to_numpy(),
            )
        X_arr = np.asarray(X)
        return np.where(X_arr[:, 2] < 0.5, X_arr[:, 0], X_arr[:, 1])


@pytest.fixture(scope="module")
def importance_contract_data():
    n_major, n_minor = 16, 8
    major = np.concatenate(
        (np.linspace(-3.0, 3.0, n_major), np.linspace(0.2, 0.8, n_minor))
    )
    minor = np.concatenate(
        (np.linspace(-0.5, 0.5, n_major), np.linspace(-4.0, 4.0, n_minor))
    )
    regime = np.concatenate((np.zeros(n_major), np.ones(n_minor)))
    X = pd.DataFrame(
        {
            "major": major,
            "minor": minor,
            "regime": regime,
            "noise": np.sin(np.arange(n_major + n_minor)),
        }
    )
    y = np.where(regime < 0.5, major, minor)
    sample_weight = np.concatenate((np.ones(n_major), np.full(n_minor, 8.0)))
    return X, y, sample_weight


def _importance_without_warnings(X, y, sample_weight, *, explicit):
    options = {"n_jobs": 1, "random_state": 7}
    if explicit:
        options.update(
            {
                "scoring": "neg_mse",
                "higher_is_better": None,
                "n_repeats": 10,
                "permute_method": "global",
                "block_size": "auto",
                "parallel_backend": "threads",
            }
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sift.permutation_importance(
            RegimePredictor(),
            X,
            y,
            sample_weight=sample_weight,
            **options,
        )
    assert [(item.category, str(item.message)) for item in caught] == []
    return result


@pytest.mark.parametrize("input_kind", ("dataframe", "ndarray"))
@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_permutation_importance_default_explicit_weight_and_return_contract(
    importance_contract_data,
    input_kind,
    weighted,
):
    X_frame, y, discriminating_weight = importance_contract_data
    X = X_frame if input_kind == "dataframe" else X_frame.to_numpy()
    sample_weight = discriminating_weight if weighted else None
    X_before = X.copy()

    implicit = _importance_without_warnings(
        X,
        y.copy(),
        None if sample_weight is None else sample_weight.copy(),
        explicit=False,
    )
    explicit = _importance_without_warnings(
        X,
        y.copy(),
        None if sample_weight is None else sample_weight.copy(),
        explicit=True,
    )

    assert type(implicit) is pd.DataFrame
    assert implicit.shape == (4, 4)
    assert list(implicit.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    ]
    all_features = list(X_frame.columns) if input_kind == "dataframe" else list(range(4))
    expected_order = (
        [all_features[1], all_features[2], all_features[0], all_features[3]]
        if weighted
        else all_features
    )
    assert implicit["feature"].tolist() == expected_order
    assert implicit["importance_mean"].is_monotonic_decreasing
    assert implicit["baseline_score"].tolist() == [0.0] * 4
    assert implicit.iloc[-1]["importance_mean"] == 0.0
    assert implicit.iloc[-1]["importance_std"] == 0.0
    assert implicit.iloc[-1]["feature"] == all_features[3]

    pd.testing.assert_frame_equal(implicit, explicit)
    if input_kind == "dataframe":
        pd.testing.assert_frame_equal(X, X_before)
    else:
        np.testing.assert_array_equal(X, X_before)
