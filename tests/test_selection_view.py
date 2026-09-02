"""Contracts for additive normalized views over legacy selection results."""

from __future__ import annotations

import builtins
import dataclasses
from dataclasses import fields
import datetime
import json
import pickle

import numpy as np
import pandas as pd
import warnings
import pytest

import sift
from sift.catboost_common import CatBoostSelectionResult
from sift.selection.view import CURVE_COLUMNS, _append_rows_like, _json_safe


@pytest.fixture(scope="module")
def selection_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(912)
    X = pd.DataFrame(
        rng.normal(size=(160, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    y = (
        3.0 * X["f0"].to_numpy()
        - 2.0 * X["f2"].to_numpy()
        + 0.15 * rng.normal(size=len(X))
    )
    return X, y


@pytest.fixture(scope="module")
def real_filter_result(selection_data):
    X, y = selection_data
    return sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        estimator="classic",
        subsample=None,
        n_jobs=1,
        verbose=False,
        return_result=True,
    )


@pytest.fixture(scope="module")
def real_knockoff_result(selection_data):
    X, y = selection_data
    return sift.select_fdr(
        X,
        y,
        q=0.5,
        offset=0,
        statistic="relevance",
        subsample=None,
        random_state=17,
        n_jobs=1,
        verbose=False,
    )


def _assert_five_accessor_lines(view: sift.SelectionView) -> None:
    features = view.features
    indices = view.indices
    k = view.k
    table = view.table
    metadata = view.metadata

    assert isinstance(features, list)
    assert indices is None or isinstance(indices, list)
    assert k == len(features)
    assert isinstance(table, pd.DataFrame)
    assert metadata["schema_version"] == "1"


def _full_filter_result(
    labels,
    *,
    selected_indices=(),
    relevance=None,
) -> sift.FilterSelectionResult:
    labels = list(labels)
    selected_indices = list(selected_indices)
    selected_set = set(selected_indices)
    if relevance is None:
        relevance = np.arange(len(labels), 0, -1, dtype=np.float64)
    ranking = pd.DataFrame(
        {
            "feature": labels,
            "rank": np.arange(1, len(labels) + 1, dtype=np.int64),
            "selected": [index in selected_set for index in range(len(labels))],
            "selected_index": np.arange(len(labels), dtype=np.int64),
            "relevance": relevance,
            "selector": "fixture",
        }
    )
    return sift.FilterSelectionResult(
        selected_features=[labels[index] for index in selected_indices],
        selected_indices=selected_indices,
        selector_metadata={
            "selector": "fixture",
            "k": len(selected_indices),
            "n_features": len(labels),
        },
        ranking_=ranking,
        diagnostics_={"source": "test"},
    )


def _knockoff_result(
    W: pd.DataFrame,
    *,
    selected_features=("a",),
    selected_indices=(0,),
) -> sift.KnockoffSelectionResult:
    return sift.KnockoffSelectionResult(
        selected_features=list(selected_features),
        selected_indices=list(selected_indices),
        selector_metadata={"selector": "knockoff_fdr", "n_features": len(W)},
        W=W,
        threshold=0.5,
        selection_frequency=None,
    )


def test_real_filter_and_knockoff_results_share_five_accessor_lines(
    selection_data,
    real_filter_result,
    real_knockoff_result,
):
    X, _ = selection_data
    filter_view = sift.as_result(real_filter_result)
    knockoff_view = sift.as_result(
        real_knockoff_result,
        input_features=X.columns,
    )

    for view in (filter_view, knockoff_view):
        _assert_five_accessor_lines(view)
        assert view.metadata["table_complete"] is True
        assert view.support_ is not None
        assert view.support_.shape == (X.shape[1],)

    assert filter_view.features == real_filter_result.selected_features
    assert filter_view.indices == real_filter_result.selected_indices
    assert knockoff_view.features == real_knockoff_result.selected_features
    assert knockoff_view.indices == real_knockoff_result.selected_indices


def test_as_result_identity_and_existing_view_rejects_input_features(real_filter_result):
    view = sift.as_result(real_filter_result)

    assert sift.as_result(view) is view
    with pytest.raises(ValueError, match="already a SelectionView"):
        sift.as_result(view, input_features=["unused"])


def test_result_view_methods_match_public_adapter(
    selection_data,
    real_filter_result,
    real_knockoff_result,
):
    X, _ = selection_data
    direct_filter = sift.as_result(real_filter_result)
    method_filter = real_filter_result.result_view()
    direct_knockoff = sift.as_result(real_knockoff_result, input_features=X.columns)
    method_knockoff = real_knockoff_result.result_view(input_features=X.columns)

    assert method_filter.to_dict() == direct_filter.to_dict()
    assert method_knockoff.to_dict() == direct_knockoff.to_dict()


def test_legacy_filter_dataclass_fields_equality_and_pickle_are_unchanged():
    expected_fields = [
        "selected_features",
        "selected_indices",
        "selector_metadata",
        "ranking_",
        "diagnostics_",
    ]
    original = sift.FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "fixture", "n_features": 1},
    )
    equal_value = sift.FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "fixture", "n_features": 1},
    )

    assert [field.name for field in fields(sift.FilterSelectionResult)] == expected_fields
    assert original == equal_value
    assert pickle.loads(pickle.dumps(original)) == original


def test_legacy_knockoff_dataclass_fields_and_pickle_are_unchanged():
    expected_fields = [
        "selected_features",
        "selected_indices",
        "selector_metadata",
        "W",
        "threshold",
        "selection_frequency",
        "diagnostics_",
    ]
    W = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "selected_index": [0, 1],
            "W": [1.0, -0.25],
            "selected": [True, False],
        }
    )
    original = sift.KnockoffSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "knockoff_fdr", "n_features": 2},
        W=W,
        threshold=0.5,
        selection_frequency=None,
        diagnostics_={"thresholds": [0.5]},
    )
    shared_frame = sift.KnockoffSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "knockoff_fdr", "n_features": 2},
        W=W,
        threshold=0.5,
        selection_frequency=None,
        diagnostics_={"thresholds": [0.5]},
    )

    assert [field.name for field in fields(sift.KnockoffSelectionResult)] == expected_fields
    assert original == shared_frame

    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is sift.KnockoffSelectionResult
    assert restored.selected_features == original.selected_features
    assert restored.selected_indices == original.selected_indices
    assert restored.selector_metadata == original.selector_metadata
    pd.testing.assert_frame_equal(restored.W, original.W)
    assert restored.threshold == original.threshold
    assert restored.selection_frequency is original.selection_frequency is None
    assert restored.diagnostics_ == original.diagnostics_


def test_filter_partial_table_retains_width_support_with_and_without_names():
    partial = sift.FilterSelectionResult(
        selected_features=["b"],
        selected_indices=[1],
        selector_metadata={"selector": "fixture", "k": 1, "n_features": 3},
    )

    unnamed = sift.as_result(partial)
    named = sift.as_result(partial, input_features=["a", "b", "c"])

    assert unnamed.metadata["table_complete"] is False
    assert unnamed.raw_features is None
    np.testing.assert_array_equal(unnamed.support_, [False, True, False])
    assert unnamed.table["feature"].tolist() == ["b"]

    assert named.metadata["table_complete"] is False
    assert named.raw_features == ["a", "b", "c"]
    np.testing.assert_array_equal(named.support_, [False, True, False])
    assert named.table["feature"].tolist() == ["b"]


def test_knockoff_input_features_complete_partial_table_and_support():
    W = pd.DataFrame(
        {
            "feature": ["a", "c"],
            "selected_index": [0, 2],
            "W": [2.0, -0.5],
            "selected": [True, False],
        }
    )
    result = sift.KnockoffSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "knockoff_fdr", "n_features": 2},
        W=W,
        threshold=1.0,
        selection_frequency=None,
    )

    unnamed = sift.as_result(result)
    named = sift.as_result(result, input_features=["a", "constant", "c"])

    assert unnamed.metadata["table_complete"] is False
    assert unnamed.support_ is None
    assert unnamed.raw_input["n_features"] is None

    assert named.metadata["table_complete"] is False
    assert named.raw_features == ["a", "constant", "c"]
    np.testing.assert_array_equal(named.support_, [True, False, False])
    assert named.table["selected_index"].tolist() == [0, 2]


def test_table_accessors_return_isolated_deep_copies(real_filter_result):
    view = sift.as_result(real_filter_result)
    table = view.table
    raw_table = view.raw_table
    original_feature = view.table.loc[0, "feature"]

    table.loc[0, "feature"] = "mutated-table"
    raw_table.loc[0, "feature"] = "mutated-raw-table"

    assert view.table.loc[0, "feature"] == original_feature
    assert view.raw_table.loc[0, "feature"] == original_feature


def test_empty_selection_has_zero_k_and_all_false_support():
    result = _full_filter_result(["a", "b", "c"])
    view = sift.as_result(result)

    assert view.features == []
    assert view.indices == []
    assert view.k == 0
    np.testing.assert_array_equal(view.support_, np.zeros(3, dtype=bool))
    assert not view.table["selected"].any()


@pytest.mark.parametrize(
    "result,input_features,match",
    [
        (
            sift.FilterSelectionResult(["a"], [0, 1], {"n_features": 2}),
            None,
            "same length",
        ),
        (
            sift.FilterSelectionResult(["a"], [2], {"n_features": 3}),
            ["a", "b"],
            "outside input_features",
        ),
        (
            sift.FilterSelectionResult(["a"], [1], {"n_features": 2}),
            ["a", "b"],
            "do not match input_features",
        ),
        (
            sift.FilterSelectionResult(["dup"], None, {"n_features": 2}),
            ["dup", "dup"],
            "missing or ambiguous",
        ),
    ],
)
def test_filter_adapter_rejects_invalid_selected_identity(
    result,
    input_features,
    match,
):
    with pytest.raises(ValueError, match=match):
        sift.as_result(result, input_features=input_features)


def test_filter_adapter_rejects_ranking_identity_mismatch():
    result = _full_filter_result(["a", "b"], selected_indices=[0])

    with pytest.raises(ValueError, match="ranking feature identities"):
        sift.as_result(result, input_features=["a", "renamed"])


@pytest.mark.parametrize(
    "selected_index,match",
    [
        ([0, 0], "unique|selected rows do not match features"),
        ([0, 3], "outside input_features"),
    ],
)
def test_filter_adapter_rejects_invalid_full_ranking_positions(selected_index, match):
    result = _full_filter_result(["a", "b"], selected_indices=[0])
    result.ranking_["selected_index"] = selected_index

    with pytest.raises(ValueError, match=match):
        sift.as_result(result, input_features=["a", "b"])


def test_filter_adapter_validates_partial_ranking_names_positionally():
    result = sift.FilterSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "fixture", "n_features": 3},
        ranking_=pd.DataFrame(
            {
                "feature": ["a", "wrong"],
                "selected_index": [0, 2],
                "selected": [True, False],
            }
        ),
    )

    with pytest.raises(ValueError, match="ranking feature identities"):
        sift.as_result(result, input_features=["a", "b", "c"])


def test_filter_adapter_rejects_false_explicit_width_and_boolean_positions():
    result = _full_filter_result(["a", "b"], selected_indices=[0])
    with pytest.raises(ValueError, match="input_features length.*n_features"):
        sift.as_result(result, input_features=["a", "b", "extra"])

    result.ranking_["selected_index"] = pd.Series([0, True], dtype=object)
    with pytest.raises(ValueError, match="selected_index values must be integers"):
        sift.as_result(result, input_features=["a", "b"])


@pytest.mark.parametrize(
    "positions,features,match",
    [
        ([0, 0], ["a", "b"], "unique"),
        ([-1, 1], ["a", "b"], "non-negative"),
        ([0.5, 1], ["a", "b"], "integer"),
        ([0, 2], ["a", "b"], "outside input_features"),
        ([0, 1], ["wrong", "b"], "feature identities"),
    ],
)
def test_knockoff_adapter_rejects_invalid_W_identity(positions, features, match):
    W = pd.DataFrame(
        {
            "feature": features,
            "selected_index": positions,
            "W": [1.0, -0.5],
            "selected": [True, False],
        }
    )
    result = _knockoff_result(W)

    with pytest.raises(ValueError, match=match):
        sift.as_result(result, input_features=["a", "b"])


def test_knockoff_adapter_rejects_selected_mask_disagreement():
    W = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "selected_index": [0, 1],
            "W": [1.0, -0.5],
            "selected": [False, False],
        }
    )
    result = _knockoff_result(W)

    with pytest.raises(
        ValueError,
        match="selected features.*W selected mask|selected rows do not match features",
    ):
        sift.as_result(result, input_features=["a", "b"])


@pytest.mark.parametrize("bad_selected", ["False", 1, np.nan, pd.NA])
def test_knockoff_adapter_rejects_non_boolean_selected_values(bad_selected):
    W = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "selected_index": [0, 1],
            "W": [1.0, -0.5],
            "selected": [bad_selected, False],
        }
    )
    result = _knockoff_result(W)

    with pytest.raises(ValueError, match="non-missing booleans"):
        sift.as_result(result, input_features=["a", "b"])


def test_duplicate_raw_labels_are_preserved_and_selected_positionally():
    result = _full_filter_result(
        ["dup", "dup", "other"],
        selected_indices=[1],
    )
    view = sift.as_result(
        result,
        input_features=["dup", "dup", "other"],
    )

    assert view.raw_features == ["dup", "dup", "other"]
    assert view.table["feature"].tolist() == ["dup", "dup", "other"]
    assert view.indices == [1]
    np.testing.assert_array_equal(view.support_, [False, True, False])
    assert view.table["path_rank"].tolist() == [pd.NA, 1, pd.NA]


def test_result_only_curve_has_exact_empty_schema(real_filter_result):
    curve = sift.as_result(real_filter_result).curve

    assert curve.empty
    assert tuple(curve.columns) == CURVE_COLUMNS
    assert tuple(curve.columns) == ("k", "criterion", "criterion_se", "selected")


def test_result_only_optional_operations_fail_with_actionable_messages(
    real_filter_result,
):
    view = sift.as_result(real_filter_result)

    with pytest.raises(NotImplementedError, match="result-only view.*fitted selector"):
        view.transform(np.zeros((1, 1)))
    with pytest.raises(NotImplementedError, match="inverse_transform.*fitted inverse encoder"):
        view.inverse_transform(np.zeros((1, 1)))
    with pytest.raises(NotImplementedError, match="store_proxies=True"):
        view.proxies(view.features[0])


def test_json_payload_is_strict_for_mixed_numpy_and_nan_labels():
    labels = [1, "1", np.int64(2), np.nan, ("pair", np.int32(3))]
    result = _full_filter_result(
        labels,
        selected_indices=[0, 3],
        relevance=np.array([1.0, np.nan, np.inf, -np.inf, np.float32(0.5)]),
    )
    view = sift.as_result(result, input_features=labels)

    payload = view.to_dict()
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))

    assert encoded == json.dumps(
        view.to_dict(),
        allow_nan=False,
        separators=(",", ":"),
    )
    assert payload["schema_version"] == "1"
    assert payload["raw_table"]["columns"] == [
        "feature",
        "selected_index",
        "path_rank",
        "selected",
        "relevance",
    ]
    assert payload["raw_table"]["data"][1][-1] is None
    assert payload["raw_table"]["data"][2][-1] is None
    assert payload["raw_table"]["data"][3][-1] is None


def test_columns_hash_is_deterministic_order_and_type_sensitive():
    labels = [1, "1", np.int64(2), np.nan, ("pair", np.int32(3))]
    first = sift.as_result(_full_filter_result(labels), input_features=labels)
    second = sift.as_result(_full_filter_result(list(labels)), input_features=list(labels))
    reversed_labels = list(reversed(labels))
    reordered = sift.as_result(
        _full_filter_result(reversed_labels),
        input_features=reversed_labels,
    )
    integer = sift.as_result(_full_filter_result([1]), input_features=[1])
    string = sift.as_result(_full_filter_result(["1"]), input_features=["1"])

    assert first.raw_input["columns_hash"] == second.raw_input["columns_hash"]
    assert first.raw_input["columns_hash"] != reordered.raw_input["columns_hash"]
    assert integer.raw_input["columns_hash"] != string.raw_input["columns_hash"]


def test_columns_hash_distinguishes_numpy_temporal_nat_types():
    # Both temporal scalars carry an explicit unit: NumPy 2.5 deprecates the
    # generic ("bare") timedelta64 unit.  The unit is incidental here -- the
    # contract under test is that None, a datetime64 NaT, and a timedelta64 NaT
    # hash to three different values.
    labels = [
        (None,),
        (np.datetime64("NaT", "ns"),),
        (np.timedelta64("NaT", "ns"),),
    ]
    hashes = {
        sift.as_result(_full_filter_result([label]), input_features=[label])
        .raw_input["columns_hash"]
        for label in labels
    }

    assert len(hashes) == len(labels)


@pytest.mark.parametrize("legacy", [[], ["a"], ("a",), ("a", "b")])
def test_as_result_rejects_legacy_lists_and_tuples(legacy):
    with pytest.raises(TypeError, match="return_result=True"):
        sift.as_result(legacy)


def test_non_catboost_result_dispatch_does_not_import_optional_catboost(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sift.catboost_common":
            raise AssertionError("unrelated result dispatch imported CatBoost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    view = sift.as_result(_full_filter_result(["a"], selected_indices=[0]))

    assert view.features == ["a"]


def _boruta_result(**overrides) -> sift.BorutaResult:
    values = {
        "feature_names": ["dup", "dup", "noise", "tentative"],
        "status": np.array([-1, 1, -1, 0], dtype=np.int8),
        "hits": np.array([0, 3, 1, 2], dtype=np.int64),
        "n_iter": 3,
        "shadow_thresholds": np.array([0.4, 0.5, 0.6]),
        "mean_importance": np.array([0.1, 1.5, np.nan, 0.3]),
    }
    values.update(overrides)
    return sift.BorutaResult(**values)


def _path_result(**overrides) -> sift.FeaturePathEvaluationResult:
    diagnostics = pd.DataFrame(
        {
            "k": [1, 3],
            "score": [1.0, 2.0],
            "std": [0.2, 0.4],
            "n_finite": [4, 4],
            "n_splits": [4, 4],
            "best_score": [1.0, 1.0],
        }
    )
    values = {
        "feature_path": ["b", "a", "c"],
        "k": [1, 3],
        "features": ["b"],
        "scores": {1: 1.0, 3: 2.0},
        "best_k": 1,
        "diagnostics": diagnostics,
    }
    values.update(overrides)
    return sift.FeaturePathEvaluationResult(**values)


def _single_path_result(
    *,
    score,
    std,
    n_finite,
    n_splits,
    best_k,
) -> sift.FeaturePathEvaluationResult:
    best_score = score if best_k else np.nan
    return sift.FeaturePathEvaluationResult(
        feature_path=["a"],
        k=[1],
        features=["a"] if best_k else [],
        scores={1: score},
        best_k=best_k,
        diagnostics=pd.DataFrame(
            {
                "k": [1],
                "score": pd.Series([score], dtype=object),
                "std": pd.Series([std], dtype=object),
                "n_finite": [n_finite],
                "n_splits": [n_splits],
                "best_score": pd.Series([best_score], dtype=object),
            }
        ),
    )


def test_boruta_result_view_is_complete_and_positionally_preserves_duplicates():
    result = _boruta_result()
    view = sift.as_result(result)

    _assert_five_accessor_lines(view)
    assert view.features == ["dup"]
    assert view.indices == [1]
    assert view.k == 1
    assert view.raw_features == ["dup", "dup", "noise", "tentative"]
    np.testing.assert_array_equal(view.support_, [False, True, False, False])
    assert view.table["selected_index"].tolist() == [0, 1, 2, 3]
    assert view.table["path_rank"].tolist() == [pd.NA, 1, pd.NA, pd.NA]
    assert view.table["boruta_status"].tolist() == [
        "rejected",
        "accepted",
        "rejected",
        "tentative",
    ]
    np.testing.assert_allclose(
        view.table["gain"].to_numpy(),
        result.mean_importance,
        equal_nan=True,
    )
    assert view.table["hits"].tolist() == [0, 3, 1, 2]
    assert view.metadata["adapter"] == "BorutaResult"
    assert view.metadata["table_complete"] is True
    assert view.metadata["input_kind"] == "unknown"
    assert view.curve.empty
    assert view.diagnostics["n_iter"] == 3
    np.testing.assert_array_equal(
        view.diagnostics["shadow_thresholds"],
        result.shadow_thresholds,
    )
    assert result.result_view().to_dict() == view.to_dict()
    json.dumps(view.to_dict(), allow_nan=False)


def test_real_boruta_and_feature_path_results_adapt(selection_data):
    X, y = selection_data
    boruta_result = sift.select_boruta(
        X,
        y,
        n_estimators=20,
        max_iter=3,
        random_state=19,
        verbose=False,
        return_result=True,
    )
    path_result = sift.evaluate_feature_path(
        X,
        y,
        feature_path=["f2", "f0", "f1"],
        k_grid=[1, 2, 3],
        random_state=19,
    )

    boruta_view = sift.as_result(boruta_result, input_features=X.columns)
    path_view = sift.as_result(path_result, input_features=X.columns)

    _assert_five_accessor_lines(boruta_view)
    _assert_five_accessor_lines(path_view)
    assert boruta_view.metadata["table_complete"] is True
    assert path_view.metadata["table_complete"] is True
    assert path_view.indices == [X.columns.get_loc(name) for name in path_view.features]
    assert path_view.curve["selected"].sum() == 1
    assert path_view.curve["criterion_se"].isna().all()


@pytest.mark.parametrize(
    ("input_features", "match"),
    [
        (["dup", "dup", "noise"], "length"),
        (["dup", "noise", "dup", "tentative"], "exact order"),
        (["dup", "dup", "noise", 1], "exact order"),
    ],
)
def test_boruta_adapter_rejects_incompatible_explicit_identity(
    input_features,
    match,
):
    with pytest.raises(ValueError, match=match):
        sift.as_result(_boruta_result(), input_features=input_features)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"status": np.array([1, -1])}, "status.*length"),
        ({"status": np.array([1.0, -1.0, -1.0, 0.0])}, "status values.*integer"),
        ({"status": np.array([1, -1, 2, 0])}, "-1, 0, or 1"),
        ({"hits": np.array([0, -1, 1, 2])}, "hits values.*>= 0"),
        ({"hits": np.array([0, 4, 1, 2])}, "cannot exceed"),
        ({"hits": [0, 2**100, 1, 2]}, "signed 64-bit integer"),
        ({"n_iter": True}, "n_iter must be an integer"),
        ({"shadow_thresholds": np.array([0.4])}, "shadow_thresholds.*length 3"),
        ({"mean_importance": np.array([0.1, 0.2])}, "mean_importance.*length 4"),
        (
            {"mean_importance": np.array([1.0 + 2.0j, 0.2, 0.1, 0.0])},
            "real non-boolean numeric values",
        ),
        (
            {"mean_importance": np.array([10**1000, 0.2, 0.1, 0.0], dtype=object)},
            "representable as float64",
        ),
    ],
)
def test_boruta_adapter_rejects_malformed_result_arrays(overrides, match):
    with pytest.raises(ValueError, match=match):
        sift.as_result(_boruta_result(**overrides))


def test_boruta_legacy_shape_and_pickle_remain_unchanged():
    expected_fields = [
        "feature_names",
        "status",
        "hits",
        "n_iter",
        "shadow_thresholds",
        "mean_importance",
    ]
    result = _boruta_result()
    restored = pickle.loads(pickle.dumps(result))

    assert [field.name for field in fields(sift.BorutaResult)] == expected_fields
    assert callable(result.selected_features)
    assert result.selected_features() == ["dup"]
    assert type(restored) is sift.BorutaResult
    assert restored.feature_names == result.feature_names
    np.testing.assert_array_equal(restored.status, result.status)
    np.testing.assert_array_equal(restored.hits, result.hits)
    np.testing.assert_array_equal(restored.shadow_thresholds, result.shadow_thresholds)
    np.testing.assert_allclose(
        restored.mean_importance,
        result.mean_importance,
        equal_nan=True,
    )


def test_feature_path_view_is_partial_without_input_identity():
    result = _path_result()
    view = sift.as_result(result)

    _assert_five_accessor_lines(view)
    assert view.features == ["b"]
    assert view.indices is None
    assert view.support_ is None
    assert view.raw_features is None
    assert view.table["feature"].tolist() == ["b", "a", "c"]
    assert view.table["selected_index"].isna().all()
    assert view.table["path_rank"].tolist() == [1, pd.NA, pd.NA]
    assert view.table["feature_path_rank"].tolist() == [1, 2, 3]
    assert view.metadata["table_complete"] is False
    assert result.result_view().to_dict() == view.to_dict()


def test_feature_path_view_maps_unique_explicit_identity_and_curve():
    result = _path_result()
    view = sift.as_result(result, input_features=["a", "b", "c", "unused"])

    assert view.features == ["b"]
    assert view.indices == [1]
    np.testing.assert_array_equal(view.support_, [False, True, False, False])
    assert view.table["feature"].tolist() == ["a", "b", "c", "unused"]
    assert view.table["path_rank"].tolist() == [pd.NA, 1, pd.NA, pd.NA]
    assert view.table["feature_path_rank"].tolist() == [2, 1, 3, pd.NA]
    assert view.metadata["table_complete"] is True
    assert view.metadata["criterion_direction"] == "minimize"
    assert view.curve["k"].tolist() == [1, 3]
    assert view.curve["criterion"].tolist() == [1.0, 2.0]
    np.testing.assert_allclose(
        view.curve["criterion_se"],
        np.array([0.2, 0.4]) / np.sqrt(3.0),
    )
    assert view.curve["selected"].tolist() == [True, False]
    json.dumps(view.to_dict(), allow_nan=False)


def test_feature_path_all_failed_result_has_empty_selection_and_no_curve_winner():
    diagnostics = pd.DataFrame(
        {
            "k": [1, 2],
            "score": [np.inf, np.inf],
            "std": [np.nan, np.nan],
            "n_finite": [0, 0],
            "n_splits": [2, 2],
            "best_score": [np.nan, np.nan],
        }
    )
    result = _path_result(
        feature_path=["a", "b"],
        k=[1, 2],
        features=[],
        scores={1: np.inf, 2: np.inf},
        best_k=0,
        diagnostics=diagnostics,
    )
    view = sift.as_result(result, input_features=["a", "b", "unused"])

    assert view.features == []
    assert view.indices == []
    assert view.k == 0
    np.testing.assert_array_equal(view.support_, [False, False, False])
    assert not view.curve["selected"].any()
    assert view.curve["criterion_se"].isna().all()
    json.dumps(view.to_dict(), allow_nan=False)


def test_feature_path_duplicate_labels_remain_partial_but_cannot_claim_positions():
    result = _path_result(
        feature_path=["dup", "dup", "z"],
        features=["dup"],
    )

    view = sift.as_result(result)
    assert view.table["feature"].tolist() == ["dup", "dup", "z"]
    assert view.indices is None

    with pytest.raises(ValueError, match="missing or ambiguous"):
        sift.as_result(result, input_features=["dup", "dup", "z"])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"k": [1, 1]}, "unique positions"),
        ({"k": [True, 3]}, "integer positions"),
        ({"scores": {1: 1.0}}, "score keys"),
        ({"best_k": 3, "features": ["b", "a", "c"]}, "best_k does not match"),
        ({"features": ["a"]}, r"feature_path\[:best_k\]"),
        (
            {
                "diagnostics": _path_result()
                .diagnostics.iloc[::-1]
                .reset_index(drop=True)
            },
            "diagnostics k order",
        ),
    ],
)
def test_feature_path_adapter_rejects_inconsistent_results(overrides, match):
    with pytest.raises(ValueError, match=match):
        sift.as_result(_path_result(**overrides))


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (
            _single_path_result(
                score=np.nan,
                std=np.nan,
                n_finite=0,
                n_splits=1,
                best_k=0,
            ),
            "finite or positive infinity",
        ),
        (
            _single_path_result(
                score=-np.inf,
                std=np.nan,
                n_finite=0,
                n_splits=1,
                best_k=0,
            ),
            "finite or positive infinity",
        ),
        (
            _single_path_result(
                score=np.inf,
                std=0.0,
                n_finite=1,
                n_splits=1,
                best_k=0,
            ),
            "infinite.*at least one failed split",
        ),
        (
            _single_path_result(
                score=1.0,
                std=np.nan,
                n_finite=1,
                n_splits=1,
                best_k=1,
            ),
            "finite.*std to be finite",
        ),
        (
            _path_result(
                diagnostics=pd.DataFrame(
                    {
                        "k": [1, 3],
                        "score": ["1.0", "2.0"],
                        "std": ["0.2", "0.4"],
                        "n_finite": [4, 4],
                        "n_splits": [4, 4],
                        "best_score": ["1.0", "1.0"],
                    }
                )
            ),
            "real non-boolean numeric values",
        ),
        (
            _path_result(scores={1: np.complex64(1.0 + 2.0j), 3: 2.0}),
            "real non-boolean numbers",
        ),
        (
            _single_path_result(
                score=10**1000,
                std=0.0,
                n_finite=1,
                n_splits=1,
                best_k=1,
            ),
            "representable as float64",
        ),
        (
            _single_path_result(
                score=1.0,
                std=123.0,
                n_finite=1,
                n_splits=1,
                best_k=1,
            ),
            "single-split.*std 0",
        ),
    ],
)
def test_feature_path_adapter_rejects_impossible_producer_states(result, match):
    with pytest.raises(ValueError, match=match):
        sift.as_result(result)


def test_feature_path_adapter_rejects_inconsistent_split_count():
    diagnostics = _path_result().diagnostics.copy()
    diagnostics["n_splits"] = [4, 5]
    diagnostics["n_finite"] = [4, 5]

    with pytest.raises(ValueError, match="n_splits must be constant"):
        sift.as_result(_path_result(diagnostics=diagnostics))


@pytest.mark.parametrize(
    "input_features",
    [
        ["a", "c", "unused"],
        ["a", "b", "c", "b"],
        ["a", 1, "c", "unused"],
    ],
)
def test_feature_path_adapter_rejects_missing_or_ambiguous_explicit_identity(
    input_features,
):
    with pytest.raises(ValueError, match="missing or ambiguous"):
        sift.as_result(_path_result(), input_features=input_features)


def test_feature_path_legacy_shape_and_pickle_remain_unchanged():
    expected_fields = [
        "feature_path",
        "k",
        "features",
        "scores",
        "best_k",
        "diagnostics",
    ]
    result = _path_result()
    restored = pickle.loads(pickle.dumps(result))

    assert [
        field.name for field in fields(sift.FeaturePathEvaluationResult)
    ] == expected_fields
    assert type(restored) is sift.FeaturePathEvaluationResult
    assert restored.feature_path == result.feature_path
    assert restored.k == result.k
    assert restored.features == result.features
    assert restored.scores == result.scores
    assert restored.best_k == result.best_k
    pd.testing.assert_frame_equal(restored.diagnostics, result.diagnostics)


def _catboost_result(**overrides) -> CatBoostSelectionResult:
    values = {
        "selected_features": ["b", "a"],
        "best_k": 2,
        "scores_by_k": {1: 0.62, 2: 0.58, 3: 0.57},
        "scores_std_by_k": {1: 0.02, 2: 0.02, 3: 0.02},
        "feature_importances": pd.Series({"b": 0.8, "a": 0.4}),
        "features_by_k": {
            1: ["b"],
            2: ["b", "a"],
            3: ["b", "a", "c"],
        },
        "stability_scores": pd.Series({"b": 1.0, "a": 0.75, "c": 0.25}),
        "prefilter_features": ["b", "a", "c", "d"],
        "metric": "RMSE",
        "higher_is_better": False,
        "all_scores": {
            1: [0.60, 0.64],
            2: [0.56, 0.60],
            3: [0.55, 0.59],
        },
        "selection_patience": 3,
    }
    values.update(overrides)
    return CatBoostSelectionResult(**values)


def test_catboost_result_view_is_partial_without_input_identity():
    result = _catboost_result()
    view = sift.as_result(result)

    _assert_five_accessor_lines(view)
    assert view.features == ["b", "a"]
    assert view.indices is None
    assert view.support_ is None
    assert view.raw_features is None
    assert view.table["feature"].tolist() == ["b", "a", "c", "d"]
    assert view.table["selected_index"].isna().all()
    assert view.table["path_rank"].tolist() == [1, 2, pd.NA, pd.NA]
    assert view.table["selected"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(
        view.table["gain"].to_numpy(),
        [0.8, 0.4, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        view.table["selection_frequency"].to_numpy(),
        [1.0, 0.75, 0.25, np.nan],
        equal_nan=True,
    )
    assert view.table["prefiltered_first_split"].tolist() == [True] * 4
    assert view.metadata["adapter"] == "CatBoostSelectionResult"
    assert view.metadata["table_complete"] is False
    assert view.metadata["criterion_direction"] == "minimize"
    assert view.metadata["target_k"] == 2
    assert view.metadata["selected_feature_count"] == 2
    assert view.metadata["best_scoring_k"] == 3
    assert view.metadata["gain_source"] == "final_model_feature_importance"
    assert view.curve["k"].tolist() == [1, 2, 3]
    assert view.curve["criterion"].tolist() == [0.62, 0.58, 0.57]
    np.testing.assert_allclose(view.curve["criterion_se"], [0.02, 0.02, 0.02])
    assert view.curve["selected"].tolist() == [False, True, False]
    assert view.diagnostics["prefilter_scope"] == "first_split_only"
    assert view.diagnostics["stability_scope"] == "target_k_split_frequency"
    assert result.result_view().to_dict() == view.to_dict()
    json.dumps(view.to_dict(), allow_nan=False)


def test_catboost_result_view_maps_explicit_raw_identity_without_guessing_provenance():
    result = _catboost_result()
    raw_features = ["d", "a", "unused", "b", "c"]
    view = sift.as_result(result, input_features=raw_features)

    assert view.features == ["b", "a"]
    assert view.indices == [3, 1]
    np.testing.assert_array_equal(view.support_, [False, True, False, True, False])
    assert view.raw_features == raw_features
    assert view.table["feature"].tolist() == raw_features
    assert view.table["selected_index"].tolist() == [0, 1, 2, 3, 4]
    assert view.table["path_rank"].tolist() == [pd.NA, 2, pd.NA, 1, pd.NA]
    assert view.metadata["table_complete"] is True
    assert view.metadata["input_kind"] == "unknown"
    assert view.metadata["raw_columns_hash"] is not None
    assert result.result_view(input_features=raw_features).to_dict() == view.to_dict()


def test_catboost_result_view_allows_duplicate_unobserved_raw_labels_positionally():
    raw_features = ["unused", "unused", "a", "b", "c", "d"]
    view = sift.as_result(_catboost_result(), input_features=raw_features)

    assert view.indices == [3, 2]
    assert view.table["selected_index"].tolist() == list(range(len(raw_features)))
    assert view.metadata["table_complete"] is True


@pytest.mark.parametrize(
    "input_features",
    [
        ["a", "b", "c"],
        ["a", "b", "b", "c", "d"],
    ],
)
def test_catboost_result_view_rejects_missing_or_ambiguous_known_features(
    input_features,
):
    with pytest.raises(ValueError, match="missing or ambiguous"):
        sift.as_result(_catboost_result(), input_features=input_features)


def test_catboost_curve_preserves_higher_is_better_direction_and_missing_se():
    result = _catboost_result(
        best_k=2,
        scores_by_k={1: 0.7, 2: 0.8},
        scores_std_by_k={},
        all_scores=None,
        metric="AUC",
        higher_is_better=True,
    )
    view = sift.as_result(result)

    assert view.curve["criterion"].tolist() == [0.7, 0.8]
    assert view.curve["criterion_se"].isna().all()
    assert view.curve["selected"].tolist() == [False, True]
    assert view.metadata["criterion_direction"] == "maximize"
    assert view.metadata["best_scoring_k"] == 2


def test_catboost_curve_filters_failed_split_scores_before_standard_error():
    result = _catboost_result(
        scores_by_k={2: 0.60},
        scores_std_by_k={2: 0.02},
        all_scores={2: [0.58, np.nan, np.inf, -np.inf, 0.62]},
    )
    view = sift.as_result(result)

    assert view.curve["criterion"].tolist() == [0.60]
    np.testing.assert_allclose(view.curve["criterion_se"], [0.02])


def test_catboost_curve_leaves_one_finite_split_standard_error_missing():
    result = _catboost_result(
        scores_by_k={2: 0.60},
        scores_std_by_k={2: 0.0},
        all_scores={2: [np.nan, 0.60, np.inf]},
    )
    view = sift.as_result(result)

    assert view.curve["criterion_se"].isna().all()


def test_catboost_target_k_can_exceed_actual_stable_feature_count():
    result = _catboost_result(
        selected_features=["b"],
        feature_importances=pd.Series({"b": 0.8}),
    )
    view = sift.as_result(result)

    assert view.k == 1
    assert view.metadata["target_k"] == 2
    assert view.metadata["selected_feature_count"] == 1
    assert view.curve.loc[view.curve["selected"], "k"].tolist() == [2]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"selected_features": ["b", "b"]}, "unique feature identities"),
        ({"selected_features": []}, "must be non-empty"),
        ({"best_k": True}, "best_k must be an integer"),
        ({"best_k": 1}, "more than best_k"),
        ({"best_k": 4}, "present in scores_by_k"),
        ({"scores_by_k": {}}, "at least one finite score"),
        ({"scores_by_k": {0: 1.0}}, ">= 1"),
        ({"scores_by_k": {2: True}}, "real non-boolean"),
        ({"scores_by_k": {2: np.inf}}, "must be finite"),
        ({"scores_by_k": {2: 1.0 + 2.0j}}, "real non-boolean"),
        (
            {"scores_std_by_k": {2: -0.1}},
            "scores_std_by_k values must be non-negative",
        ),
        (
            {"scores_std_by_k": {1: 0.02, 2: 0.02, 3: 0.02, 4: 0.0}},
            "unexpected",
        ),
        (
            {"all_scores": {2: [0.50, 0.70]}},
            "must match the finite all_scores mean",
        ),
        (
            {"all_scores": {2: [np.nan, np.inf]}},
            "finite observation",
        ),
        (
            {"features_by_k": {2: ["b"]}},
            "exactly 2 features",
        ),
        (
            {"feature_importances": [0.8, 0.4]},
            "feature_importances must be a pandas Series",
        ),
        (
            {"feature_importances": pd.Series([0.8, 0.4], index=["b", "b"])},
            "unique feature identities",
        ),
        (
            {"feature_importances": pd.Series({"b": 0.8})},
            "cover selected_features exactly",
        ),
        (
            {"stability_scores": pd.Series({"b": 1.2, "a": 0.5})},
            "between 0 and 1",
        ),
        (
            {"stability_scores": pd.Series({"b": 1.0, "c": 0.5})},
            "present in stability_scores",
        ),
        ({"prefilter_features": ["b", "b"]}, "unique feature identities"),
        ({"metric": ""}, "metric must be a non-empty string"),
        ({"higher_is_better": 1}, "higher_is_better must be boolean"),
        ({"selection_patience": 0}, "selection_patience must be >= 1"),
    ],
)
def test_catboost_adapter_rejects_malformed_result_states(overrides, match):
    with pytest.raises(ValueError, match=match):
        sift.as_result(_catboost_result(**overrides))


def test_catboost_legacy_shape_and_pickle_remain_unchanged():
    expected_fields = [
        "selected_features",
        "best_k",
        "scores_by_k",
        "scores_std_by_k",
        "feature_importances",
        "features_by_k",
        "stability_scores",
        "prefilter_features",
        "metric",
        "higher_is_better",
        "all_scores",
        "selection_patience",
    ]
    result = _catboost_result()
    restored = pickle.loads(pickle.dumps(result))

    assert [field.name for field in fields(CatBoostSelectionResult)] == expected_fields
    assert type(restored) is CatBoostSelectionResult
    assert restored.selected_features == result.selected_features
    assert restored.best_k == result.best_k
    assert restored.scores_by_k == result.scores_by_k
    assert restored.scores_std_by_k == result.scores_std_by_k
    pd.testing.assert_series_equal(restored.feature_importances, result.feature_importances)
    assert restored.features_by_k == result.features_by_k
    pd.testing.assert_series_equal(restored.stability_scores, result.stability_scores)
    assert restored.prefilter_features == result.prefilter_features
    assert restored.metric == result.metric
    assert restored.higher_is_better == result.higher_is_better
    assert restored.all_scores == result.all_scores
    assert restored.selection_patience == result.selection_patience


# --------------------------------------------------------------------------
# Stage 1.4 / R2: auto-k producers retain a full ranking and a normalized curve
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def auto_k_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(400, 12)),
        columns=[f"f{i}" for i in range(12)],
    )
    y = (
        2.0 * X["f0"].to_numpy()
        - 1.5 * X["f3"].to_numpy()
        + rng.normal(scale=0.5, size=len(X))
    )
    return X, y


def test_auto_k_filter_view_is_complete_and_carries_a_normalized_curve(auto_k_frame):
    X, y = auto_k_frame
    result = sift.select_cefsplus(X, y, k="auto", return_result=True, verbose=False)
    view = sift.as_result(result)

    assert len(view.table) == X.shape[1]
    assert view.metadata["table_complete"] is True
    assert view.table["selected_index"].tolist() == list(range(X.shape[1]))

    curve = view.curve
    assert list(curve.columns) == list(CURVE_COLUMNS)
    assert not curve.empty
    assert view.metadata["curve_available"] is True
    assert view.metadata["criterion"] == "penalized_score"
    assert view.metadata["criterion_direction"] == "higher_is_better"
    assert curve.loc[curve["selected"], "k"].tolist() == [len(view.features)]

    # The legacy result keeps its own fields; the curve lives in diagnostics_.
    assert type(result) is sift.FilterSelectionResult
    assert result.ranking_ is not None
    payload = result.diagnostics_["auto_k_curve"]
    assert payload["available"] is True
    assert payload["criterion"] == "penalized_score"


@pytest.mark.parametrize(
    ("k_method", "config_kwargs", "criterion", "direction"),
    [
        ("elbow", {}, "objective", "higher_is_better"),
        ("penalized_objective", {}, "penalized_score", "higher_is_better"),
        ("k_posterior", {}, "post", "higher_is_better"),
        ("chi2_stop", {}, "p_max", "lower_is_better"),
        ("forward_stop", {}, "Y_running_mean", "lower_is_better"),
        ("changepoint", {}, "log_scaled_gain", "higher_is_better"),
        ("perm_gap", {}, "gap", "higher_is_better"),
        ("stability", {}, "phi", "higher_is_better"),
        ("gaussian_cv", {"strategy": "kfold"}, "score", "higher_is_better"),
        ("xfit_objective", {"strategy": "kfold"}, "score", "higher_is_better"),
    ],
)
def test_auto_k_routes_publish_their_criterion_curve(
    auto_k_frame, k_method, config_kwargs, criterion, direction
):
    X, y = auto_k_frame
    config = sift.AutoKConfig(k_method=k_method, max_k=8, **config_kwargs)
    result = sift.select_cefsplus(
        X, y, k="auto", auto_k_config=config, return_result=True, verbose=False
    )
    view = sift.as_result(result)

    assert view.metadata["curve_available"] is True
    assert view.metadata["criterion"] == criterion
    assert view.metadata["criterion_direction"] == direction
    curve = view.curve
    assert list(curve.columns) == list(CURVE_COLUMNS)
    assert curve["k"].is_monotonic_increasing
    assert not curve["k"].duplicated().any()

    diagnostics = result.diagnostics_["auto_k_diagnostics"]
    assert curve["k"].tolist() == sorted(int(k) for k in diagnostics["k"])
    # ``selected`` marks the k the route actually returned.
    selected_k = curve.loc[curve["selected"], "k"].tolist()
    assert selected_k in ([], [len(view.features)])
    assert view.metadata["table_complete"] is True
    assert len(view.table) == X.shape[1]


# The 12-feature ``auto_k_frame`` makes the four consensus submethods disagree
# by 3x, so the ``consensus`` parameterization incidentally trips auto-k's
# ill-determined-k advisory.  That advisory is the library behaving as designed
# and is asserted directly in ``tests/test_auto_k_v2.py``; this test is about
# the curve-unavailability metadata, so the exact message is exempted here.
@pytest.mark.filterwarnings(
    "ignore:consensus auto-k methods disagree by more than 2x:UserWarning"
)
@pytest.mark.parametrize("k_method", ["knockoff_path", "consensus"])
def test_routes_without_a_k_curve_say_why(auto_k_frame, k_method):
    X, y = auto_k_frame
    config = sift.AutoKConfig(k_method=k_method, max_k=8)
    result = sift.select_cefsplus(
        X, y, k="auto", auto_k_config=config, return_result=True, verbose=False
    )
    view = sift.as_result(result)

    assert view.metadata["curve_available"] is False
    assert view.curve.empty
    assert list(view.curve.columns) == list(CURVE_COLUMNS)
    reason = view.metadata["curve_unavailable_reason"]
    assert "one row per" in reason
    # The route is still fully ranked and its table is complete.
    assert view.metadata["table_complete"] is True
    assert result.ranking_ is not None


def test_auto_k_router_curve_follows_the_routed_method(auto_k_frame):
    X, y = auto_k_frame
    result = sift.select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=sift.AutoKConfig(k_method="auto", max_k=8),
        return_result=True,
        verbose=False,
    )
    payload = result.diagnostics_["auto_k_curve"]
    routed = result.diagnostics_["auto_k"]["routed_method"]

    assert payload["route"] == routed
    assert payload["available"] is True
    view = sift.as_result(result)
    assert view.metadata["curve_route"] == routed


def test_classic_auto_k_retains_ranking_and_curve(auto_k_frame):
    X, y = auto_k_frame
    groups = np.repeat(np.arange(5), len(X) // 5)
    result = sift.select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="classic",
        auto_k_config=sift.AutoKConfig(
            k_method="evaluate", max_k=6, strategy="group_cv"
        ),
        groups=groups,
        subsample=None,
        n_jobs=1,
        return_result=True,
        verbose=False,
    )
    view = sift.as_result(result)

    assert result.ranking_ is not None
    assert list(result.ranking_["feature"]) != []
    assert view.metadata["table_complete"] is True
    assert len(view.table) == X.shape[1]
    assert view.metadata["curve_available"] is True
    assert view.metadata["criterion"] == "score"
    assert view.metadata["criterion_direction"] == "higher_is_better"


def test_fixed_k_filter_view_keeps_legacy_ranking_and_no_curve(selection_data):
    X, y = selection_data
    result = sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        estimator="classic",
        subsample=None,
        n_jobs=1,
        return_result=True,
        verbose=False,
    )
    view = sift.as_result(result)

    assert "auto_k_curve" not in (result.diagnostics_ or {})
    assert view.metadata["curve_available"] is False
    assert "criterion" not in view.metadata
    assert view.curve.empty
    assert list(view.curve.columns) == list(CURVE_COLUMNS)


def test_malformed_auto_k_curve_payloads_are_rejected(auto_k_frame):
    X, y = auto_k_frame
    result = sift.select_cefsplus(X, y, k="auto", return_result=True, verbose=False)
    good = result.diagnostics_["auto_k_curve"]

    result.diagnostics_["auto_k_curve"] = "not-a-mapping"
    with pytest.raises(ValueError, match="curve payload must be a mapping"):
        sift.as_result(result)

    result.diagnostics_["auto_k_curve"] = {**good, "criterion_direction": "maximize"}
    with pytest.raises(ValueError, match="criterion_direction must be"):
        sift.as_result(result)

    result.diagnostics_["auto_k_curve"] = {**good, "curve": None}
    with pytest.raises(ValueError, match="must carry a DataFrame curve"):
        sift.as_result(result)

    result.diagnostics_["auto_k_curve"] = {
        **good,
        "curve": good["curve"].drop(columns=["criterion_se"]),
    }
    with pytest.raises(ValueError, match="missing required columns"):
        sift.as_result(result)


# --------------------------------------------------------------------------
# Stage 1.4 / R3: knockoff raw width without input_features
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def knockoff_constant_result(auto_k_frame):
    X, y = auto_k_frame
    X = X.copy()
    X["f7"] = 1.0
    return X, sift.select_fdr(X, y, q=0.3, verbose=False)


def test_knockoff_metadata_separates_raw_and_post_filter_width(
    knockoff_constant_result,
):
    X, result = knockoff_constant_result
    metadata = result.selector_metadata

    assert metadata["n_features_input"] == X.shape[1]
    assert metadata["n_features"] == X.shape[1] - 1
    assert metadata["dropped_feature_positions"] == [7]
    assert metadata["dropped_feature_reasons"] == ["constant"]


def test_knockoff_view_builds_raw_support_without_input_features(
    knockoff_constant_result,
):
    X, result = knockoff_constant_result
    view = sift.as_result(result)

    assert view.support_ is not None
    assert view.support_.shape == (X.shape[1],)
    assert view.raw_input["n_features"] == X.shape[1]
    assert view.metadata["table_complete"] is True

    table = view.table
    assert len(table) == X.shape[1]
    assert table["selected_index"].tolist() == list(range(X.shape[1]))
    dropped = table.loc[table["reason_dropped"].notna()]
    assert dropped["selected_index"].tolist() == [7]
    assert dropped["reason_dropped"].tolist() == ["constant"]
    # A dropped column is never reported as selected.
    assert not bool(dropped["selected"].any())
    assert not bool(view.support_[7])


def test_knockoff_view_names_dropped_columns_when_identity_is_supplied(
    knockoff_constant_result,
):
    X, result = knockoff_constant_result
    named = sift.as_result(result, input_features=X.columns)

    dropped = named.table.loc[named.table["reason_dropped"].notna()]
    assert dropped["feature"].tolist() == ["f7"]
    assert named.metadata["table_complete"] is True
    assert named.raw_features == list(X.columns)


def test_knockoff_view_rejects_input_features_of_the_wrong_raw_width(
    knockoff_constant_result,
):
    X, result = knockoff_constant_result
    with pytest.raises(ValueError, match="n_features_input"):
        sift.as_result(result, input_features=list(X.columns) + ["extra"])


def test_legacy_knockoff_results_without_raw_width_stay_partial():
    W = pd.DataFrame(
        {
            "feature": ["a", "c"],
            "selected_index": [0, 2],
            "W": [2.0, -0.5],
            "selected": [True, False],
        }
    )
    result = sift.KnockoffSelectionResult(
        selected_features=["a"],
        selected_indices=[0],
        selector_metadata={"selector": "knockoff_fdr", "n_features": 2},
        W=W,
        threshold=1.0,
        selection_frequency=None,
    )
    view = sift.as_result(result)

    assert view.support_ is None
    assert view.metadata["table_complete"] is False
    assert "reason_dropped" not in view.table.columns


# --------------------------------------------------------------------------
# Stage 1.4 / R4: collision-free JSON conversion, no repr() fallback
# --------------------------------------------------------------------------


def test_mixed_key_mapping_survives_a_json_round_trip():
    payload = _json_safe({1: "int", "1": "str"})
    restored = json.loads(json.dumps(payload))

    assert restored["__sift_mapping__"] == "typed_key_entries"
    entries = restored["entries"]
    assert len(entries) == 2
    assert entries[0] == {"key": {"type": "builtins.int", "value": 1}, "value": "int"}
    assert entries[1] == {"key": {"type": "builtins.str", "value": "1"}, "value": "str"}

    # Both entries survive; the legacy str(key) form merged them into one.
    rebuilt = {
        (entry["key"]["type"], entry["key"]["value"]): entry["value"]
        for entry in entries
    }
    assert rebuilt == {("builtins.int", 1): "int", ("builtins.str", "1"): "str"}


def test_ordinary_string_key_mappings_keep_their_plain_json_shape():
    payload = _json_safe({"a": 1, "b": [1, 2], "c": {"d": None}})

    assert payload == {"a": 1, "b": [1, 2], "c": {"d": None}}
    assert "__sift_mapping__" not in payload
    assert "__sift_mapping__" not in payload["c"]


def test_view_payload_root_and_metadata_stay_plain_json_objects(selection_data):
    X, y = selection_data
    result = sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        estimator="classic",
        subsample=None,
        n_jobs=1,
        return_result=True,
        verbose=False,
    )
    payload = sift.as_result(result, input_features=X.columns).to_dict()

    assert "__sift_mapping__" not in payload
    assert "__sift_mapping__" not in payload["metadata"]
    assert payload["schema_version"] == "1"
    assert payload["metadata"]["schema_version"] == "1"
    assert all(isinstance(key, str) for key in payload)
    json.dumps(payload, allow_nan=False)


def test_nested_mixed_key_mapping_is_wrapped_where_it_occurs():
    payload = _json_safe({"outer": {2: "two", "2": "str-two"}})

    assert set(payload) == {"outer"}
    assert payload["outer"]["__sift_mapping__"] == "typed_key_entries"
    assert [entry["value"] for entry in payload["outer"]["entries"]] == [
        "two",
        "str-two",
    ]
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (pd.NA, None),
        (pd.NaT, None),
        (float("nan"), None),
        (float("inf"), None),
        (datetime.datetime(2020, 1, 2, 3, 4, 5), "2020-01-02T03:04:05"),
        (datetime.date(2020, 1, 2), "2020-01-02"),
        (datetime.time(3, 4, 5), "03:04:05"),
        (datetime.timedelta(days=1, seconds=5), "P1DT0H0M5S"),
        (np.datetime64("2020-01-02"), "2020-01-02"),
        (pd.Timestamp("2020-01-02T03:04:05"), "2020-01-02T03:04:05"),
    ],
)
def test_scalar_conversions_are_json_native(value, expected):
    assert _json_safe({"v": value}) == {"v": expected}


def test_dataclasses_serialize_through_asdict():
    @dataclasses.dataclass
    class Inner:
        count: int
        when: datetime.datetime

    @dataclasses.dataclass
    class Outer:
        name: str
        inner: Inner

    payload = _json_safe(Outer("x", Inner(1, datetime.datetime(2021, 5, 6))))

    assert payload == {
        "name": "x",
        "inner": {"count": 1, "when": "2021-05-06T00:00:00"},
    }
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize("value", [b"ab", bytearray(b"ab"), object(), 1 + 2j])
def test_unsupported_objects_raise_instead_of_leaking_repr(value):
    with pytest.raises(TypeError, match="no JSON-safe representation"):
        _json_safe(value)


def test_unsupported_mapping_keys_raise_too():
    with pytest.raises(TypeError, match="no JSON-safe representation"):
        _json_safe({object(): 1, "ok": 2})


def test_knockoff_view_with_dropped_inputs_appends_rows_without_pandas_warnings():
    W = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "selected_index": [0, 1, 3],
            "W": [1.5, -0.2, 0.9],
            "selected": [True, False, True],
            "relevance": [0.8, 0.1, 0.6],
            "feature_group": [0, 0, 1],
        }
    )
    result = sift.KnockoffSelectionResult(
        selected_features=["a", "c"],
        selected_indices=[0, 3],
        selector_metadata={
            "selector": "knockoff_fdr",
            "n_features": 3,
            "n_features_input": 4,
            "dropped_feature_positions": [2],
            "dropped_feature_reasons": ["constant"],
        },
        W=W,
        threshold=0.5,
        selection_frequency=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view = sift.as_result(result, input_features=["a", "b", "const", "c"])
    table = view.table
    assert table["selected_index"].tolist() == [0, 1, 2, 3]
    dropped = table.loc[table["selected_index"] == 2].iloc[0]
    assert dropped["feature"] == "const"
    assert dropped["reason_dropped"] == "constant"
    assert not dropped["selected"]
    assert pd.isna(dropped["path_rank"]) and pd.isna(dropped["gain"])
    assert pd.isna(dropped["relevance"]) and pd.isna(dropped["feature_group"])
    assert str(table["path_rank"].dtype) == "Int64"
    assert table["selected"].dtype == bool
    # Appending a row that leaves ``feature_group`` missing widens the numpy
    # int64 column to its nullable counterpart; that promotion is the documented
    # contract of ``_append_rows_like``, so pin it rather than leaving it loose.
    assert str(table["feature_group"].dtype) == "Int64"
    assert table.loc[table["selected_index"] != 2, "feature_group"].tolist() == [0, 0, 1]
    assert view.metadata["table_complete"] is True


def test_knockoff_view_without_dropped_inputs_keeps_numpy_integer_dtypes():
    """No appended rows means no widening: ``feature_group`` stays numpy int64."""
    W = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "selected_index": [0, 1, 2],
            "W": [1.5, -0.2, 0.9],
            "selected": [True, False, True],
            "relevance": [0.8, 0.1, 0.6],
            "feature_group": [0, 0, 1],
        }
    )
    result = sift.KnockoffSelectionResult(
        selected_features=["a", "c"],
        selected_indices=[0, 2],
        selector_metadata={
            "selector": "knockoff_fdr",
            "n_features": 3,
            "n_features_input": 3,
        },
        W=W,
        threshold=0.5,
        selection_frequency=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        view = sift.as_result(result, input_features=["a", "b", "c"])
    table = view.table
    assert table["feature_group"].dtype == np.dtype("int64")
    assert table["selected"].dtype == bool
    assert table["feature_group"].tolist() == [0, 0, 1]
    assert "reason_dropped" not in table.columns
    assert view.metadata["table_complete"] is True


def test_append_rows_like_is_an_identity_on_empty_rows():
    """``_append_rows_like(table, [])`` returns the table with dtypes untouched."""
    table = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "count": np.array([1, 2], dtype=np.int64),
            "selected": np.array([True, False]),
            "gain": np.array([0.5, 1.5]),
            "rank": pd.array([1, pd.NA], dtype="Int64"),
        }
    )
    before = table.dtypes.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        appended = _append_rows_like(table, [])

    assert appended is table
    pd.testing.assert_series_equal(appended.dtypes, before)
    pd.testing.assert_frame_equal(appended, table)
    assert appended["count"].dtype == np.dtype("int64")
    assert appended["selected"].dtype == np.dtype("bool")
