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
