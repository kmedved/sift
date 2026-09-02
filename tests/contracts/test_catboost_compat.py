"""Compatibility contracts for the optional CatBoost entry points."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

pytestmark = pytest.mark.catboost

import sift  # noqa: E402
from sift.catboost import (  # noqa: E402
    CatBoostSelectionResult,
    _build_catboost_model_params,
    catboost_classif,
    catboost_regression,
    catboost_select,
)


@pytest.fixture
def catboost_contract_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(901)
    n_rows = 96
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=n_rows),
            "proxy": rng.normal(size=n_rows),
            "weak": rng.normal(size=n_rows),
            "noise": rng.normal(size=n_rows),
        }
    )
    y_reg = pd.Series(
        2.0 * X["signal"]
        - 0.5 * X["proxy"]
        + rng.normal(scale=0.2, size=n_rows)
    )
    y_class = pd.Series(
        (
            X["signal"]
            + 0.4 * X["proxy"]
            + rng.normal(scale=0.1, size=n_rows)
            > 0
        ).astype(int)
    )
    return X, y_reg, y_class


@pytest.fixture
def bounded_catboost_kwargs() -> dict[str, object]:
    return {
        "n_splits": 2,
        "prefilter_k": None,
        "n_estimators": 20,
        "algorithm": "shap",
        "steps": 2,
        "train_early_stopping_rounds": 5,
        "n_jobs": 1,
        "random_state": 0,
        "verbose": False,
    }


def test_catboost_public_defaults_are_pinned() -> None:
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(catboost_select).parameters.items()
    }

    assert defaults["k"] is None
    assert defaults["task"] == "regression"
    assert defaults["algorithm"] == "shap"
    assert defaults["n_splits"] == 3
    assert defaults["n_estimators"] == 500
    assert defaults["random_state"] is None
    assert defaults["n_jobs"] == -1
    assert defaults["verbose"] is True
    assert defaults["groups"] is None
    assert defaults["time"] is None
    assert defaults["sample_weight"] is None
    assert sift.catboost_select is catboost_select
    assert sift.catboost_regression is catboost_regression
    assert sift.catboost_classif is catboost_classif


def test_catboost_result_contract(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = catboost_select(
            X.copy(),
            y_reg.copy(),
            k=2,
            task="regression",
            **bounded_catboost_kwargs,
        )

    assert caught == []
    assert type(result) is CatBoostSelectionResult
    assert result.selected_features == ["signal", "proxy"]
    assert result.best_k == 2
    assert set(result.scores_by_k) == {2}
    assert set(result.scores_std_by_k) == {2}
    assert result.features_by_k == {2: ["signal", "proxy"]}
    assert result.metric == "RMSE"
    assert result.higher_is_better is False
    assert result.selection_patience == 3
    assert list(result.feature_importances.index) == ["signal", "proxy"]

    view = sift.as_result(result, input_features=X.columns)
    assert view.features == result.selected_features
    assert view.indices == [0, 1]
    np.testing.assert_array_equal(view.support_, [True, True, False, False])
    assert view.table["feature"].tolist() == X.columns.tolist()
    assert view.metadata["adapter"] == "CatBoostSelectionResult"
    assert view.metadata["table_complete"] is True
    assert view.metadata["metric"] == "RMSE"
    assert view.curve["k"].tolist() == [2]
    assert view.curve["selected"].tolist() == [True]
    assert result.result_view(input_features=X.columns).to_dict() == view.to_dict()


def test_catboost_omitted_task_matches_explicit_regression_default(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data

    omitted = sift.catboost_select(
        X.copy(),
        y_reg.copy(),
        k=2,
        **bounded_catboost_kwargs,
    )
    explicit = catboost_select(
        X.copy(),
        y_reg.copy(),
        k=2,
        task="regression",
        **bounded_catboost_kwargs,
    )

    assert type(omitted) is type(explicit) is CatBoostSelectionResult
    assert omitted.selected_features == explicit.selected_features
    assert omitted.best_k == explicit.best_k
    assert omitted.scores_by_k == explicit.scores_by_k
    assert omitted.scores_std_by_k == explicit.scores_std_by_k
    assert omitted.features_by_k == explicit.features_by_k
    assert omitted.metric == explicit.metric
    assert omitted.higher_is_better == explicit.higher_is_better


@pytest.mark.parametrize(
    ("helper", "target_index"),
    [(catboost_regression, 1), (catboost_classif, 2)],
)
def test_catboost_helpers_preserve_list_contract_and_task_routing(
    helper,
    target_index: int,
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X = catboost_contract_data[0]
    y = catboost_contract_data[target_index]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = helper(
            X.copy(),
            y.copy(),
            k=2,
            **bounded_catboost_kwargs,
        )
        explicit_none = helper(
            X.copy(),
            y.copy(),
            k=2,
            callback=None,
            **bounded_catboost_kwargs,
        )

    assert caught == []
    assert type(selected) is type(explicit_none) is list
    assert selected == explicit_none == ["signal", "proxy"]


def test_catboost_direct_row_arrays_match_legacy_columns(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data
    groups = np.repeat(np.arange(8), 12)
    weights = np.linspace(0.5, 1.5, len(X))
    with_metadata = X.assign(group=groups, weight=weights)

    legacy = catboost_select(
        with_metadata,
        y_reg,
        k=2,
        group_col="group",
        sample_weight_col="weight",
        **bounded_catboost_kwargs,
    )
    direct = catboost_select(
        X,
        y_reg,
        k=2,
        groups=groups,
        sample_weight=weights,
        **bounded_catboost_kwargs,
    )

    assert direct.selected_features == legacy.selected_features
    assert direct.best_k == legacy.best_k
    assert direct.scores_by_k == legacy.scores_by_k
    assert direct.features_by_k == legacy.features_by_k


def test_catboost_time_column_is_removed_and_orders_rows(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data
    order = np.arange(len(X))[::-1]
    with_time = X.assign(when=order)

    result = catboost_select(
        with_time,
        y_reg,
        k=2,
        time="when",
        **bounded_catboost_kwargs,
    )
    assert "when" not in result.selected_features
    assert all("when" not in features for features in result.features_by_k.values())


def test_catboost_array_alias_conflicts_raise_before_fitting(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data
    with_metadata = X.assign(group=np.arange(len(X)), weight=1.0)
    with pytest.raises(ValueError, match="both groups and group_col"):
        catboost_select(
            with_metadata,
            y_reg,
            k=2,
            groups=np.arange(len(X)),
            group_col="group",
            **bounded_catboost_kwargs,
        )
    with pytest.raises(ValueError, match="both sample_weight and sample_weight_col"):
        catboost_select(
            with_metadata,
            y_reg,
            k=2,
            sample_weight=np.ones(len(X)),
            sample_weight_col="weight",
            **bounded_catboost_kwargs,
        )


def test_catboost_params_collision_warns_and_dictionary_still_wins(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    _X, y_reg, _ = catboost_contract_data
    with pytest.warns(UserWarning, match="depth.*iterations"):
        params, metric, higher_is_better = _build_catboost_model_params(
            task="regression",
            y=y_reg,
            n_estimators=30,
            learning_rate=None,
            max_depth=6,
            eval_metric=None,
            loss_function=None,
            catboost_params={"iterations": 7, "depth": 2},
            higher_is_better=None,
            random_state=0,
            gpu=False,
            n_jobs=1,
        )
    assert params["iterations"] == 7
    assert params["depth"] == 2
    assert metric == "RMSE"
    assert higher_is_better is False


def test_catboost_collision_warning_is_caller_facing_through_wrapper(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data
    with pytest.warns(UserWarning, match="iterations") as caught:
        catboost_regression(
            X,
            y_reg,
            k=2,
            catboost_params={"iterations": 7},
            **bounded_catboost_kwargs,
        )
    assert caught[0].filename == __file__


def test_catboost_none_random_state_warning_is_caller_facing(
    catboost_contract_data: tuple[pd.DataFrame, pd.Series, pd.Series],
    bounded_catboost_kwargs: dict[str, object],
) -> None:
    X, y_reg, _ = catboost_contract_data
    options = dict(bounded_catboost_kwargs)
    options.pop("random_state")
    with pytest.warns(FutureWarning, match="SIFT 1.0") as caught:
        catboost_select(X, y_reg, k=2, **options)
    assert caught[0].filename == __file__

    with pytest.warns(FutureWarning, match="SIFT 1.0") as wrapper_caught:
        catboost_regression(X, y_reg, k=2, **options)
    assert wrapper_caught[0].filename == __file__
