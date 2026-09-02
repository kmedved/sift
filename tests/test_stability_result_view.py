"""Contracts for normalized views over fitted stability selectors."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import sift
from sift.selection.view import CURVE_COLUMNS


class _SameReprLabel:
    def __repr__(self):
        return "same"


@pytest.fixture
def stability_data():
    rng = np.random.default_rng(711)
    X = pd.DataFrame(
        rng.normal(size=(60, 4)),
        columns=["signal", "proxy", "noise", "weak"],
    )
    y = 2.5 * X["signal"].to_numpy() + 0.1 * rng.normal(size=len(X))
    return X, y


def _selector(**kwargs):
    options = {
        "n_bootstrap": 5,
        "alpha": 0.15,
        "n_jobs": 1,
        "random_state": 0,
        "verbose": False,
    }
    options.update(kwargs)
    return sift.StabilitySelector(**options)


def test_fitted_dataframe_view_has_complete_authoritative_table(stability_data):
    X, y = stability_data
    selector = _selector().fit(X, y)

    direct = sift.as_result(selector)
    view = selector.result_view_

    assert view.to_dict() == direct.to_dict()
    assert view.features == selector.selected_feature_names_
    assert view.indices == selector.selected_features_.tolist()
    np.testing.assert_array_equal(view.support_, selector.get_support())
    assert view.k == selector.n_features_selected_
    assert view.raw_features == list(X.columns)
    assert view.table["feature"].tolist() == list(X.columns)
    assert view.table["selected_index"].tolist() == list(range(X.shape[1]))
    np.testing.assert_array_equal(
        view.table["selected"].to_numpy(), selector.get_support()
    )
    np.testing.assert_array_equal(
        view.table["selection_frequency"].to_numpy(),
        selector.selection_frequencies_,
    )
    assert view.table["mean_abs_coef"].dtype == np.float32
    np.testing.assert_array_equal(
        view.table["gain"].to_numpy(), selector.mean_abs_coef_
    )
    ranks = view.table.set_index("selected_index")["path_rank"]
    assert [ranks.loc[index] for index in view.indices] == list(
        range(1, view.k + 1)
    )
    assert view.curve.empty
    assert list(view.curve.columns) == list(CURVE_COLUMNS)
    assert view.metadata["adapter"] == "StabilitySelector"
    assert view.metadata["input_kind"] == "dataframe"
    assert view.metadata["table_complete"] is True
    assert view.metadata["raw_namespace"] == "fitted_candidate_features"
    assert view.metadata["transform_available"] is True
    assert view.metadata["inverse_transform_available"] is False
    np.testing.assert_array_equal(view.transform(X), selector.transform(X))
    json.dumps(view.to_dict(), allow_nan=False)


def test_positional_and_explicit_name_transform_contracts(stability_data):
    X, y = stability_data
    unnamed = _selector().fit(X.to_numpy(), y)
    unnamed_view = unnamed.result_view_

    assert unnamed_view.raw_features == [f"x{i}" for i in range(X.shape[1])]
    assert unnamed_view.metadata["input_kind"] == "positional"
    np.testing.assert_array_equal(
        unnamed_view.transform(X.to_numpy()), unnamed.transform(X.to_numpy())
    )
    with pytest.raises(ValueError, match="generated feature names"):
        unnamed_view.transform(X)

    named = _selector().fit(X.to_numpy(), y, feature_names=X.columns)
    named_view = named.result_view_
    assert named_view.metadata["input_kind"] == "positional"
    np.testing.assert_array_equal(named_view.transform(X), named.transform(X))
    with pytest.raises(ValueError, match="exact order"):
        sift.as_result(named, input_features=list(reversed(X.columns)))


def test_dataframe_subset_view_uses_fitted_candidate_namespace(stability_data):
    X, y = stability_data
    subset = ["noise", "signal"]
    selector = _selector().fit(X, y, feature_names=subset)
    view = selector.result_view_

    assert view.raw_features == subset
    assert view.raw_input["n_features"] == len(subset)
    assert view.metadata["raw_namespace"] == "fitted_candidate_features"
    assert view.metadata["input_kind"] == "dataframe"
    np.testing.assert_array_equal(view.transform(X), selector.transform(X))


def test_view_membership_uses_capped_indices_not_threshold_only_info(
    stability_data,
    monkeypatch,
):
    X, y = stability_data

    def fixed_chunks(self, X_scaled, target, weights, split_iter):
        return (
            np.array([4, 3, 4, 1], dtype=np.int32),
            np.array([8.0, 6.0, 4.0, 2.0]),
            4,
        )

    monkeypatch.setattr(
        sift.StabilitySelector,
        "_run_stability_chunks",
        fixed_chunks,
    )
    selector = _selector(threshold=0.2, max_features=2, store_coefs=False).fit(X, y)
    view = selector.result_view_

    assert view.indices == [0, 2]
    assert view.features == ["signal", "noise"]
    assert view.table["selected"].tolist() == [True, False, True, False]
    assert selector.get_feature_info()["selected"].sum() == 4
    assert view.table["path_rank"].tolist() == [1, pd.NA, 2, pd.NA]


def test_zero_selection_and_store_coefs_false_remain_valid(
    stability_data,
    monkeypatch,
):
    X, y = stability_data

    def empty_chunks(self, X_scaled, target, weights, split_iter):
        p = X_scaled.shape[1]
        return np.zeros(p, dtype=np.int32), np.zeros(p), 3

    monkeypatch.setattr(
        sift.StabilitySelector,
        "_run_stability_chunks",
        empty_chunks,
    )
    selector = _selector(store_coefs=False).fit(X, y)
    view = selector.result_view_

    assert view.features == []
    assert view.indices == []
    assert view.k == 0
    assert not view.support_.any()
    assert not view.table["selected"].any()
    assert view.table["path_rank"].isna().all()
    assert view.transform(X).shape == (len(X), 0)
    assert view.metadata["coefs_available"] is False
    assert view.diagnostics["completed_bootstraps"] is None
    json.dumps(view.to_dict(), allow_nan=False)


def test_view_transform_is_frozen_and_preserves_set_output(stability_data):
    X, y = stability_data
    selector = _selector().set_output(transform="pandas").fit(X, y)
    view = selector.result_view_
    expected_features = view.features

    transformed = view.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.columns.tolist() == expected_features

    selector.set_threshold(1.0)
    assert view.features == expected_features
    assert view.transform(X).columns.tolist() == expected_features


def test_view_and_fitted_selector_pickle_without_live_state(stability_data):
    X, y = stability_data
    selector = _selector().fit(X, y)
    view = selector.result_view_

    restored_view = pickle.loads(pickle.dumps(view))
    restored_selector = pickle.loads(pickle.dumps(selector))
    assert restored_view.to_dict() == view.to_dict()
    np.testing.assert_array_equal(restored_view.transform(X), view.transform(X))
    assert restored_selector.result_view_.to_dict() == view.to_dict()

    cloned = clone(selector)
    with pytest.raises(NotFittedError):
        _ = cloned.result_view_


def test_pre_provenance_fitted_state_uses_safe_legacy_fallback(stability_data):
    X, y = stability_data
    positional = _selector().fit(X.to_numpy(), y)
    del positional._fit_feature_names_generated_
    del positional._fit_input_kind_

    positional_view = positional.result_view_
    assert positional_view.metadata["input_kind"] == "unknown"
    synthetic_frame = X.set_axis(positional.feature_names_in_, axis=1)
    np.testing.assert_array_equal(
        positional_view.transform(synthetic_frame),
        positional.transform(X.to_numpy()),
    )

    named = _selector().fit(X, y)
    del named._fit_feature_names_generated_
    del named._fit_input_kind_
    assert named.result_view_.metadata["input_kind"] == "unknown"
    np.testing.assert_array_equal(named.result_view_.transform(X), named.transform(X))

    synthetic_named = _selector().fit(synthetic_frame, y)
    del synthetic_named._fit_feature_names_generated_
    del synthetic_named._fit_input_kind_
    synthetic_view = synthetic_named.result_view_
    assert synthetic_view.metadata["input_kind"] == "unknown"
    np.testing.assert_array_equal(
        synthetic_view.transform(synthetic_frame),
        synthetic_named.transform(synthetic_frame),
    )


@pytest.mark.parametrize(
    "missing_marker",
    ["_fit_feature_names_generated_", "_fit_input_kind_"],
)
def test_partial_provenance_state_falls_back_atomically(
    stability_data,
    missing_marker,
):
    X, y = stability_data
    selector = _selector().fit(X.to_numpy(), y)
    delattr(selector, missing_marker)
    view = selector.result_view_

    assert view.metadata["input_kind"] == "unknown"
    synthetic_frame = X.set_axis(selector.feature_names_in_, axis=1)
    np.testing.assert_array_equal(
        view.transform(synthetic_frame),
        selector.transform(X.to_numpy()),
    )


def test_unfitted_and_failed_refit_have_no_result_view(stability_data):
    X, y = stability_data
    selector = _selector()
    with pytest.raises(NotFittedError):
        _ = selector.result_view_

    selector.fit(X, y)
    with pytest.raises(ValueError, match="Duplicate DataFrame"):
        selector.fit(X.set_axis(["a", "a", "b", "c"], axis=1), y)
    with pytest.raises(NotFittedError):
        _ = selector.result_view_


def test_view_transform_preserves_named_validation(stability_data):
    X, y = stability_data
    view = _selector().fit(X, y).result_view_
    duplicate = X.copy()
    duplicate.columns = ["signal", "signal", "noise", "weak"]

    with pytest.raises(ValueError, match="Duplicate DataFrame"):
        view.transform(duplicate)
    with pytest.raises(ValueError, match="missing selected feature"):
        view.transform(X.drop(columns=view.features[0]))


def test_tuple_feature_labels_remain_single_positional_identities(stability_data):
    X, y = stability_data
    labels = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
    tuple_frame = X.copy()
    tuple_frame.columns = labels
    selector = _selector().fit(tuple_frame, y)
    view = selector.result_view_

    assert view.raw_features == labels
    np.testing.assert_array_equal(view.transform(tuple_frame), selector.transform(tuple_frame))


def test_distinct_hashable_labels_with_same_repr_remain_distinct(stability_data):
    X, y = stability_data
    labels = [_SameReprLabel() for _ in range(X.shape[1])]
    frame = X.copy()
    frame.columns = labels
    selector = _selector().fit(frame, y)
    view = selector.result_view_

    assert all(observed is expected for observed, expected in zip(view.raw_features, labels))
    np.testing.assert_array_equal(view.transform(frame), selector.transform(frame))


@pytest.mark.parametrize("replacement", [1.0, True])
def test_input_identity_rejects_type_distinct_pandas_equivalent_labels(
    stability_data,
    replacement,
):
    X, y = stability_data
    frame = X.copy()
    frame.columns = [1, "b", "c", "d"]
    selector = _selector().fit(frame, y)

    with pytest.raises(ValueError, match="exact order"):
        sift.as_result(selector, input_features=[replacement, "b", "c", "d"])


def test_multiclass_fitted_view_preserves_selector_state():
    rng = np.random.default_rng(719)
    X = pd.DataFrame(
        rng.normal(size=(90, 4)),
        columns=["a", "b", "c", "d"],
    )
    score = X["a"].to_numpy() - 0.7 * X["b"].to_numpy()
    y = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
    selector = _selector(task="classification", alpha=0.5).fit(X, y)
    view = selector.result_view_

    assert view.metadata["task"] == "classification"
    assert view.features == selector.selected_feature_names_
    assert view.indices == selector.selected_features_.tolist()
    np.testing.assert_array_equal(view.transform(X), selector.transform(X))


def test_inconsistent_selection_membership_is_rejected(stability_data):
    X, y = stability_data
    selector = _selector().fit(X, y)
    selector.selection_frequencies_ = np.zeros(X.shape[1], dtype=np.float64)

    with pytest.raises(ValueError, match="inconsistent with its threshold"):
        _ = selector.result_view_


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("threshold", np.nan, "threshold must be a finite"),
        ("max_features", -1, "max_features must be >= 1"),
        ("task", "bogus", "task must be"),
        ("alpha_", np.nan, "alpha_ must be a finite"),
        ("alpha_rule_effective_", "bogus", "alpha_rule_effective_ must be"),
        ("coef_threshold", np.nan, "coef_threshold must be a finite"),
    ],
)
def test_malformed_fitted_configuration_is_rejected(
    stability_data,
    attribute,
    value,
    match,
):
    X, y = stability_data
    selector = _selector().fit(X, y)
    setattr(selector, attribute, value)

    with pytest.raises(ValueError, match=match):
        _ = selector.result_view_


def _differing_order_selector():
    """Fit a selector whose legacy order really differs from original order."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        rng.normal(size=(150, 8)),
        columns=[f"f{i}" for i in range(8)],
    )
    # Signal strength grows with column position, so descending-frequency
    # (legacy) order is the reverse of ascending-position (original) order.
    y = (
        0.35 * X["f1"].to_numpy()
        + 0.9 * X["f4"].to_numpy()
        + 2.5 * X["f7"].to_numpy()
        + rng.normal(size=len(X))
    )
    return X, y


@pytest.mark.parametrize("output_order", ["legacy", "original"])
def test_view_follows_the_fitted_output_order(output_order):
    X, y = _differing_order_selector()
    selector = _selector(
        n_bootstrap=40,
        alpha=0.05,
        threshold=0.5,
        output_order=output_order,
    ).fit(X, y)
    view = selector.result_view_

    expected_names = list(selector.get_feature_names_out())
    expected_indices = [int(index) for index in selector.get_support(indices=True)]

    assert view.metadata["output_order"] == output_order
    assert view.features == expected_names
    assert view.indices == expected_indices

    table = view.table
    ranked = table.loc[table["path_rank"].notna()].sort_values("path_rank")
    assert ranked["feature"].tolist() == expected_names

    # A frozen transform must reproduce the fitted selector's column order.
    np.testing.assert_array_equal(view.transform(X), selector.transform(X))
    np.testing.assert_array_equal(view.support_, selector.get_support())


def test_non_default_output_order_actually_reorders_the_view():
    """Guard the regression: the two orders must not be trivially identical."""
    X, y = _differing_order_selector()
    legacy = _selector(
        n_bootstrap=40, alpha=0.05, threshold=0.5, output_order="legacy"
    ).fit(X, y)
    original = _selector(
        n_bootstrap=40, alpha=0.05, threshold=0.5, output_order="original"
    ).fit(X, y)

    legacy_view = legacy.result_view_
    original_view = original.result_view_

    assert set(legacy_view.features) == set(original_view.features)
    assert legacy_view.features != original_view.features
    assert original_view.indices == sorted(original_view.indices)
    assert legacy_view.indices != sorted(legacy_view.indices)
    # The frozen transformer must not silently fall back to the default order.
    np.testing.assert_array_equal(
        original_view.transform(X), original.transform(X)
    )
    assert not np.array_equal(
        original_view.transform(X), legacy_view.transform(X)
    )
