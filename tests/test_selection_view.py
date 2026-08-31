"""Contracts for additive normalized views over legacy selection results."""

from __future__ import annotations

from dataclasses import fields
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import sift
from sift.selection.view import CURVE_COLUMNS


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
    labels = [(None,), (np.datetime64("NaT"),), (np.timedelta64("NaT"),)]
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
