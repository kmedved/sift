"""Categorical DataFrame compatibility across function and selector APIs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

import sift


def _binary_categorical_data():
    rng = np.random.default_rng(120)
    n = 72
    team = np.where(np.arange(n) % 3 == 0, "red", "blue")
    target = np.where(team == "red", "yes", "no")
    X = pd.DataFrame(
        {
            "team": pd.Categorical(team),
            "noise": rng.normal(size=n),
            "weak": rng.normal(size=n),
        }
    )
    sample_weight = np.linspace(0.5, 1.5, n)
    return X, target, sample_weight


@dataclass(frozen=True)
class CategoricalRoute:
    name: str
    function: object
    selector_class: type
    kwargs: dict


CATEGORY_ENCODER_ROUTES = (
    CategoricalRoute(
        "mrmr",
        sift.select_mrmr,
        sift.MRMRSelector,
        {"task": "regression", "estimator": "classic", "mrmr_backend": "serial"},
    ),
    CategoricalRoute(
        "jmi",
        sift.select_jmi,
        sift.JMISelector,
        {"task": "regression", "estimator": "r2"},
    ),
    CategoricalRoute(
        "jmim",
        sift.select_jmim,
        sift.JMIMSelector,
        {"task": "regression", "estimator": "r2"},
    ),
    CategoricalRoute(
        "cefsplus",
        sift.select_cefsplus,
        sift.CEFSPlusSelector,
        {},
    ),
)


BINARY_METADATA_KEYS = {
    "selector",
    "k_requested",
    "k",
    "top_m",
    "n_features",
    "auto_k",
    "loss",
    "weighted",
    "class_weight",
    "class_weight_scope",
    "ridge",
    "refit_every",
    "corr_prune",
    "subsample",
    "random_state",
    "cat_encoding",
    "loo_smoothing",
    "loo_clip_min",
    "loo_clip_max",
    "target_mapping",
}


@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_binary_function_loo_logit_categorical_contract(weighted):
    """The built-in categorical route stays usable without optional extras."""
    X, y, sample_weight = _binary_categorical_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sift.select_cefsplus_binary(
            X.copy(),
            y.copy(),
            k=1,
            cat_features=["team"],
            cat_encoding="loo_logit",
            allow_full_data_target_encoding=True,
            sample_weight=sample_weight.copy() if weighted else None,
            subsample=None,
            verbose=False,
            return_result=True,
        )

    assert [(item.category, str(item.message)) for item in caught] == []
    assert type(result) is sift.FilterSelectionResult
    assert result.selected_features == ["team"]
    assert result.selected_indices == [0]
    assert set(result.selector_metadata) == BINARY_METADATA_KEYS
    assert result.selector_metadata["selector"] == "cefsplus_binary"
    assert result.selector_metadata["cat_encoding"] == "loo_logit"
    assert result.selector_metadata["target_mapping"] == {"no": 0, "yes": 1}
    assert result.selector_metadata["weighted"] is weighted
    assert list(result.ranking_.columns) == [
        "feature",
        "rank",
        "selected",
        "selected_index",
        "relevance",
        "score",
        "selector",
    ]
    assert result.ranking_.loc[result.ranking_["selected"], "feature"].tolist() == [
        "team"
    ]


@pytest.mark.parametrize("weighted", (False, True), ids=("unweighted", "weighted"))
def test_binary_selector_loo_logit_fit_transform_contract(weighted):
    X, y, sample_weight = _binary_categorical_data()
    selector = sift.CEFSPlusBinarySelector(
        k=1,
        cat_features=["team"],
        cat_encoding="loo_logit",
        verbose=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        training_output = selector.fit_transform(
            X.copy(),
            y.copy(),
            sample_weight=sample_weight.copy() if weighted else None,
        )

    assert [(item.category, str(item.message)) for item in caught] == []
    assert selector.selected_features_ == ["team"]
    np.testing.assert_array_equal(
        selector.selected_indices_,
        np.array([0], dtype=np.int64),
    )
    assert selector.categorical_features_ == ["team"]
    assert type(selector.categorical_encoder_).__name__ == "LeaveOneOutLogitEncoder"
    assert type(training_output) is pd.DataFrame
    assert training_output.columns.tolist() == ["team"]
    assert training_output.index.equals(X.index)
    assert pd.api.types.is_float_dtype(training_output["team"])
    assert np.isfinite(training_output["team"]).all()

    transformed = selector.transform(X.iloc[:4].copy())
    assert type(transformed) is pd.DataFrame
    assert transformed.columns.tolist() == ["team"]
    assert transformed.index.equals(X.index[:4])
    assert np.isfinite(transformed["team"]).all()
    with pytest.raises(ValueError, match="transform also requires a DataFrame"):
        selector.transform(X.to_numpy())


def test_function_supervised_categorical_encoding_requires_explicit_opt_in():
    X, y, _ = _binary_categorical_data()
    with pytest.raises(
        ValueError,
        match="allow_full_data_target_encoding=True",
    ):
        sift.select_cefsplus_binary(
            X,
            y,
            k=1,
            cat_features=["team"],
            cat_encoding="loo_logit",
            subsample=None,
            verbose=False,
        )


def test_function_categorical_arguments_reject_ndarray_input():
    X, y, _ = _binary_categorical_data()
    with pytest.raises(
        TypeError,
        match="cat_features/cat_encoding require X to be a pandas DataFrame",
    ):
        sift.select_cefsplus_binary(
            X.to_numpy(),
            y,
            k=1,
            cat_features=["team"],
            cat_encoding="loo_logit",
            allow_full_data_target_encoding=True,
            subsample=None,
            verbose=False,
        )


def _nonnumeric_default_error(call):
    with pytest.raises(ValueError, match="Non-numeric columns") as error:
        call()
    return str(error.value)


def test_function_omitted_categorical_defaults_match_explicit_none():
    X, y_binary, _ = _binary_categorical_data()
    y = (y_binary == "yes").astype(float)
    omitted = _nonnumeric_default_error(
        lambda: sift.select_cefsplus(X.copy(), y.copy(), k=1, verbose=False)
    )
    explicit = _nonnumeric_default_error(
        lambda: sift.select_cefsplus(
            X.copy(),
            y.copy(),
            k=1,
            cat_features=None,
            cat_encoding="none",
            verbose=False,
        )
    )
    assert omitted == explicit


def test_selector_omitted_categorical_defaults_match_explicit_none():
    X, y_binary, _ = _binary_categorical_data()
    y = (y_binary == "yes").astype(float)
    omitted = _nonnumeric_default_error(
        lambda: sift.CEFSPlusSelector(k=1, verbose=False).fit(X.copy(), y.copy())
    )
    explicit = _nonnumeric_default_error(
        lambda: sift.CEFSPlusSelector(
            k=1,
            cat_features=None,
            cat_encoding="none",
            verbose=False,
        ).fit(X.copy(), y.copy())
    )
    assert omitted == explicit


def test_category_encoder_function_rejects_unconsumed_sample_weight():
    X, y_binary, sample_weight = _binary_categorical_data()
    y = (y_binary == "yes").astype(float)
    with pytest.raises(ValueError, match="sample_weight.*loo_logit"):
        sift.select_cefsplus(
            X,
            y,
            k=1,
            cat_features=["team"],
            cat_encoding="target",
            allow_full_data_target_encoding=True,
            sample_weight=sample_weight,
            subsample=None,
            verbose=False,
        )


def test_missing_category_encoders_has_clean_dependency_error():
    if importlib.util.find_spec("category_encoders") is not None:
        pytest.skip("category_encoders is installed; positive cells cover this path")
    X, y_binary, _ = _binary_categorical_data()
    y = (y_binary == "yes").astype(float)
    with pytest.raises(ImportError, match="category_encoders.*pip install"):
        sift.select_cefsplus(
            X,
            y,
            k=1,
            cat_features=["team"],
            cat_encoding="target",
            allow_full_data_target_encoding=True,
            subsample=None,
            verbose=False,
        )


@pytest.mark.parametrize("route", CATEGORY_ENCODER_ROUTES, ids=lambda route: route.name)
@pytest.mark.categorical
def test_category_encoder_backed_function_and_selector_routes(route):
    pytest.importorskip("category_encoders")
    rng = np.random.default_rng(121)
    n = 90
    category = np.resize(np.array(["low", "mid", "high"], dtype=object), n)
    target = np.select(
        [category == "low", category == "mid"],
        [-2.0, 0.5],
        default=3.0,
    ) + rng.normal(scale=0.02, size=n)
    X = pd.DataFrame(
        {
            "category": category,
            "noise": rng.normal(size=n),
            "weak": rng.normal(size=n),
        }
    )

    with warnings.catch_warnings(record=True) as function_warnings:
        warnings.simplefilter("always")
        function_result = route.function(
            X.copy(),
            target.copy(),
            k=1,
            cat_features=["category"],
            cat_encoding="target",
            allow_full_data_target_encoding=True,
            subsample=None,
            verbose=False,
            return_result=True,
            **route.kwargs,
        )
    selector_kwargs = dict(route.kwargs)
    selector_kwargs.pop("mrmr_backend", None)
    selector = route.selector_class(
        k=1,
        cat_features=["category"],
        cat_encoding="target",
        subsample=None,
        verbose=False,
        **selector_kwargs,
    )
    with warnings.catch_warnings(record=True) as selector_warnings:
        warnings.simplefilter("always")
        training_output = selector.fit_transform(X.copy(), target.copy())

    assert [(item.category, str(item.message)) for item in function_warnings] == []
    assert [(item.category, str(item.message)) for item in selector_warnings] == []
    assert type(function_result) is sift.FilterSelectionResult
    assert function_result.selected_features == ["category"]
    assert function_result.selected_indices == [0]
    assert selector.selected_features_ == ["category"]
    np.testing.assert_array_equal(selector.selected_indices_, np.array([0]))
    assert type(training_output) is pd.DataFrame
    assert training_output.columns.tolist() == ["category"]
    assert np.isfinite(training_output["category"]).all()
