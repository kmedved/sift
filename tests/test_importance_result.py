import inspect
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import sift
from sift.importance import ImportanceResult, permutation_importance


class _MutableLabel:
    def __init__(self, value):
        self.value = value


class _ConstantReprLabel:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "constant-label-repr"


def _regression_problem(*, dataframe: bool = True):
    rng = np.random.default_rng(20260831)
    values = rng.normal(size=(80, 3))
    y = 2.5 * values[:, 0] - 0.75 * values[:, 1]
    if dataframe:
        return pd.DataFrame(values, columns=["signal", "weak", "noise"]), y
    return values, y


@pytest.mark.parametrize("dataframe", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_return_result_default_preserves_legacy_dataframe(dataframe, weighted):
    X, y = _regression_problem(dataframe=dataframe)
    weights = np.linspace(0.5, 2.0, len(y)) if weighted else None
    model = LinearRegression().fit(X, y, sample_weight=weights)

    omitted = permutation_importance(
        model,
        X,
        y,
        sample_weight=weights,
        n_repeats=4,
        n_jobs=1,
        random_state=17,
    )
    explicit = permutation_importance(
        model,
        X,
        y,
        sample_weight=weights,
        n_repeats=4,
        n_jobs=1,
        random_state=17,
        return_result=False,
    )

    assert type(omitted) is pd.DataFrame
    pd.testing.assert_frame_equal(omitted, explicit, check_exact=True)
    assert list(omitted.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    ]


def test_importance_result_reproduces_legacy_table_and_repeat_aggregates():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    options = dict(n_repeats=5, n_jobs=1, random_state=23)

    legacy = permutation_importance(model, X, y, **options)
    result = permutation_importance(model, X, y, return_result=True, **options)

    assert type(result) is ImportanceResult
    pd.testing.assert_frame_equal(result.ranking_, legacy, check_exact=True)
    assert result.importances_.shape == (X.shape[1], options["n_repeats"])
    assert result.feature_names == list(X.columns)
    assert set(result.ranking_indices) == set(range(X.shape[1]))
    np.testing.assert_allclose(
        result.ranking_["importance_mean"],
        result.importances_.mean(axis=1)[result.ranking_indices],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.ranking_["importance_std"],
        result.importances_.std(axis=1)[result.ranking_indices],
        rtol=0.0,
        atol=0.0,
    )


def test_importance_result_accessors_are_defensive_and_pickle_stable():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        n_jobs=1,
        random_state=7,
        return_result=True,
    )

    ranking = result.ranking_
    ranking.iloc[0, 1] = -999.0
    importances = result.importances_
    importances[0, 0] = -999.0
    names = result.feature_names
    names[0] = "changed"
    indices = result.ranking_indices
    indices[0] = 999
    metadata = result.selector_metadata
    metadata["selector"] = "changed"
    diagnostics = result.diagnostics_
    diagnostics["std_ddof"] = 99

    assert result.ranking_.iloc[0, 1] != -999.0
    assert result.importances_[0, 0] != -999.0
    assert result.feature_names[0] == "signal"
    assert 999 not in result.ranking_indices
    assert result.selector_metadata["selector"] == "permutation_importance"
    assert result.diagnostics_["std_ddof"] == 0

    restored = pickle.loads(pickle.dumps(result))
    assert type(restored) is ImportanceResult
    pd.testing.assert_frame_equal(restored.ranking_, result.ranking_, check_exact=True)
    np.testing.assert_array_equal(restored.importances_, result.importances_)
    assert restored.selector_metadata == result.selector_metadata


def test_importance_result_view_is_complete_ranking_not_subset_selection():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=4,
        n_jobs=1,
        random_state=5,
        return_result=True,
    )

    view = sift.as_result(result)
    assert result.result_view().features == view.features
    assert view.features == result.ranking_["feature"].tolist()
    assert view.indices == result.ranking_indices
    np.testing.assert_array_equal(view.support_, np.ones(X.shape[1], dtype=bool))
    assert view.k == X.shape[1]
    assert view.curve.empty
    assert view.metadata["selection_semantics"] == "ranking_only"
    assert view.metadata["gain_source"] == "permutation_importance_mean"
    assert view.metadata["table_complete"] is True
    assert view.metadata["input_kind"] == "dataframe"
    table = view.table
    assert table["selected_index"].tolist() == list(range(X.shape[1]))
    assert table["selected"].all()
    np.testing.assert_allclose(table["gain"], result.importances_.mean(axis=1))
    np.testing.assert_array_equal(
        view.diagnostics["permutation_importance_repeats"],
        result.importances_,
    )
    json.dumps(view.to_dict())

    with pytest.raises(NotImplementedError, match="transform is unavailable"):
        view.transform(X)
    with pytest.raises(NotImplementedError, match="inverse_transform is unavailable"):
        view.inverse_transform(X)


def test_duplicate_dataframe_labels_remain_distinct_by_position():
    # A duplicate-labelled frame cannot be handed to an sklearn estimator at
    # all: from scikit-learn 1.9 its dataframe validation goes through narwhals,
    # which rejects repeated column names in ``fit`` and ``predict`` alike. SIFT
    # passes X through to ``model.predict`` untouched, so the estimator here is
    # a positional stub -- addressing columns by position is precisely the
    # contract under test, since ``X["dup"]`` is ambiguous by construction.
    class _PositionalPredictor:
        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            return 3.0 * X.iloc[:, 1].to_numpy()

    rng = np.random.default_rng(91)
    values = rng.normal(size=(70, 3))
    X = pd.DataFrame(values, columns=["dup", "dup", "noise"])
    y = 3.0 * values[:, 1]
    model = _PositionalPredictor()

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=4,
        n_jobs=1,
        random_state=2,
        return_result=True,
    )
    view = result.result_view()

    assert result.feature_names == ["dup", "dup", "noise"]
    assert sorted(result.ranking_indices) == [0, 1, 2]
    assert view.table["selected_index"].tolist() == [0, 1, 2]
    assert view.table["feature"].tolist() == ["dup", "dup", "noise"]
    assert view.indices == result.ranking_indices


def test_positional_result_can_receive_display_names_without_losing_positions():
    X, y = _regression_problem(dataframe=False)
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        n_jobs=1,
        random_state=3,
        return_result=True,
    )

    assert result.feature_names == [0, 1, 2]
    view = result.result_view(input_features=["a", "b", "c"])
    assert view.metadata["input_kind"] == "positional"
    assert view.raw_features == ["a", "b", "c"]
    assert view.features == [["a", "b", "c"][index] for index in result.ranking_indices]
    with pytest.raises(ValueError, match="length must match"):
        result.result_view(input_features=["a", "b"])


def test_importance_result_records_resolved_method_without_retaining_context_arrays():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    groups = np.repeat(np.arange(8), 10)
    result = permutation_importance(
        model,
        X,
        y,
        groups=groups,
        permute_method="auto",
        n_repeats=2,
        n_jobs=1,
        random_state=4,
        return_result=True,
    )

    metadata = result.selector_metadata
    assert metadata["permute_method_requested"] == "auto"
    assert metadata["permute_method"] == "within_group"
    assert metadata["groups_supplied"] is True
    assert all(not isinstance(value, np.ndarray) for value in metadata.values())


def test_rich_threads_and_processes_match():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    options = dict(
        n_repeats=3,
        n_jobs=2,
        random_state=11,
        return_result=True,
    )

    threads = permutation_importance(
        model,
        X,
        y,
        parallel_backend="threads",
        **options,
    )
    processes = permutation_importance(
        model,
        X,
        y,
        parallel_backend="processes",
        **options,
    )

    pd.testing.assert_frame_equal(threads.ranking_, processes.ranking_, check_exact=True)
    np.testing.assert_array_equal(threads.importances_, processes.importances_)


def test_importance_result_api_validation_and_export_boundary():
    assert inspect.signature(permutation_importance).parameters["return_result"].default is False
    assert "ImportanceResult" in __import__("sift.importance", fromlist=["__all__"]).__all__
    assert "ImportanceResult" not in sift.__all__
    assert not hasattr(sift, "ImportanceResult")

    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    legacy = permutation_importance(
        model,
        X,
        y,
        n_repeats=2,
        n_jobs=1,
        random_state=0,
    )
    with pytest.raises(TypeError, match="return_result=True"):
        sift.as_result(legacy)
    with pytest.raises(ValueError, match="return_result must be a boolean"):
        permutation_importance(
            model,
            X,
            y,
            n_repeats=2,
            n_jobs=1,
            return_result="yes",
        )


def test_importance_result_view_rejects_corrupted_repeat_shape():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        n_jobs=1,
        random_state=0,
        return_result=True,
    )
    object.__setattr__(result, "_importances", np.zeros((2, 3)))

    with pytest.raises(ValueError, match="shape must be"):
        sift.as_result(result)


@pytest.mark.parametrize(
    "importances",
    [
        np.asarray([[1.0 + 2.0j], [2.0 + 0.0j]]),
        np.asarray([[True], [False]]),
        np.asarray([["1.0"], ["2.0"]]),
    ],
)
def test_importance_result_constructor_rejects_lossy_repeat_coercions(importances):
    ranking = pd.DataFrame(
        {
            "feature": ["b", "a"],
            "importance_mean": [2.0, 1.0],
            "importance_std": [0.0, 0.0],
            "baseline_score": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="real non-boolean numeric"):
        ImportanceResult(
            ranking=ranking,
            importances=importances,
            feature_names=["a", "b"],
            ranking_indices=[1, 0],
            baseline_score=0.0,
            selector_metadata={},
        )


def test_importance_result_constructor_rejects_fractional_ranking_indices():
    ranking = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance_mean": [1.0, 0.0],
            "importance_std": [0.0, 0.0],
            "baseline_score": [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="integer positions"):
        ImportanceResult(
            ranking=ranking,
            importances=np.asarray([[1.0], [0.0]]),
            feature_names=["a", "b"],
            ranking_indices=[0.9, 1],
            baseline_score=0.0,
            selector_metadata={},
        )


def test_importance_result_isolates_mutable_feature_label_objects():
    left = _MutableLabel("left")
    right = _MutableLabel("right")
    X = pd.DataFrame(np.arange(40.0).reshape(20, 2), columns=[left, right])
    y = X.iloc[:, 0].to_numpy()
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=2,
        n_jobs=1,
        random_state=0,
        return_result=True,
    )

    returned_names = result.feature_names
    returned_names[0].value = "changed"
    returned_ranking = result.ranking_
    returned_label = returned_ranking["feature"].iloc[0]
    returned_label.value = "changed-again"

    assert [label.value for label in result.feature_names] == ["left", "right"]
    assert sorted(label.value for label in result.result_view().features) == [
        "left",
        "right",
    ]
    assert sorted(
        label.value for label in result.result_view(input_features=X.columns).features
    ) == ["left", "right"]
    assert sorted(
        label.value
        for label in result.result_view(input_features=result.feature_names).features
    ) == ["left", "right"]


def test_identity_only_labels_reject_unrelated_objects_with_the_same_repr():
    left = _ConstantReprLabel("left")
    right = _ConstantReprLabel("right")
    X = pd.DataFrame(np.arange(40.0).reshape(20, 2), columns=[left, right])
    y = X.iloc[:, 0].to_numpy()
    result = permutation_importance(
        LinearRegression().fit(X, y),
        X,
        y,
        n_repeats=2,
        n_jobs=1,
        random_state=0,
        return_result=True,
    )

    result.result_view(input_features=X.columns)
    result.result_view(input_features=result.feature_names)
    unrelated = [_ConstantReprLabel("other-1"), _ConstantReprLabel("other-2")]
    with pytest.raises(ValueError, match="exact order"):
        result.result_view(input_features=unrelated)

    replaced = result.feature_names
    replaced[0] = _ConstantReprLabel("replacement")
    with pytest.raises(ValueError, match="exact order"):
        result.result_view(input_features=replaced)

    mutated = result.feature_names
    mutated[0].value = "mutated"
    with pytest.raises(ValueError, match="exact order"):
        result.result_view(input_features=mutated)


def test_importance_result_metadata_does_not_retain_arbitrary_block_size_object():
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        permute_method="global",
        block_size=model,
        n_repeats=2,
        n_jobs=1,
        random_state=0,
        return_result=True,
    )

    assert isinstance(result.selector_metadata["block_size"], str)


@pytest.mark.parametrize(
    "corrupted",
    [
        np.asarray([[1.0 + 2.0j], [1.0], [1.0]], dtype=np.complex128),
        np.asarray([[True], [False], [True]]),
        np.asarray([["1.0"], ["1.0"], ["1.0"]]),
    ],
)
def test_importance_result_view_rejects_lossy_corrupted_repeat_types(corrupted):
    X, y = _regression_problem()
    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=1,
        n_jobs=1,
        random_state=0,
        return_result=True,
    )
    object.__setattr__(result, "_importances", corrupted)

    with pytest.raises(ValueError, match="real non-boolean numeric"):
        sift.as_result(result)
