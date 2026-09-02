"""Acceptance contracts for additive 0.9 B2 conventions."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import GroupKFold

import sift
from sift.boruta import BorutaSelector
from sift.selection.auto_k_nested import NestedAutoKFold, select_k_nested


def _two_argument_mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


@pytest.fixture
def b2_data():
    rng = np.random.default_rng(20260901)
    n_rows = 72
    signal = rng.normal(size=n_rows)
    weak = rng.normal(size=n_rows)
    noise = rng.normal(size=n_rows)
    y = 2.0 * signal - 0.3 * weak + rng.normal(scale=0.15, size=n_rows)
    groups = np.repeat(np.arange(8), 9)
    time = np.tile(np.arange(9), 8)
    weights = np.linspace(0.5, 1.5, n_rows)
    X = pd.DataFrame(
        {
            "signal": signal,
            "weak": weak,
            "noise": noise,
            "group": groups,
            "time": time,
        }
    )
    return X, y, groups, time, weights


def _evaluate_config(metric="rmse", *, nested=False):
    return sift.AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        metric=metric,
        min_k=1,
        max_k=3,
        n_splits=4,
        auto_k_mode="nested" if nested else "prefix_only",
    )


def test_filter_and_select_k_auto_group_column_sugar_matches_arrays(b2_data):
    X, y, groups, _time, _weights = b2_data
    features = X[["signal", "weak", "noise"]]
    config = _evaluate_config()

    from_column = sift.select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        groups="group",
        auto_k_config=config,
        estimator="classic",
        mrmr_backend="serial",
        verbose=False,
    )
    from_array = sift.select_mrmr(
        features,
        y,
        k="auto",
        task="regression",
        groups=groups,
        auto_k_config=config,
        estimator="classic",
        mrmr_backend="serial",
        verbose=False,
    )
    assert from_column == from_array
    assert "group" not in from_column

    path = ["signal", "weak", "noise"]
    column_result = sift.select_k_auto(X, y, path, config, groups="group")
    array_result = sift.select_k_auto(features, y, path, config, groups=groups)
    assert column_result[:2] == array_result[:2]
    pd.testing.assert_frame_equal(column_result[2], array_result[2], check_exact=True)


def test_fixed_k_filter_still_rejects_string_metadata(b2_data):
    X, y, _groups, _time, _weights = b2_data
    with pytest.raises(ValueError, match="only meaningful for auto-k"):
        sift.select_mrmr(
            X,
            y,
            k=2,
            task="regression",
            groups="group",
            verbose=False,
        )


def test_selector_auto_k_sugar_excludes_metadata_and_transform_accepts_original(b2_data):
    X, y, _groups, _time, _weights = b2_data
    selector = sift.MRMRSelector(
        k="auto",
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        auto_k_config=_evaluate_config(),
        verbose=False,
    ).fit(X, y, groups="group")

    assert selector.feature_names_in_ == ["signal", "weak", "noise", "time"]
    assert "group" not in selector.selected_features_
    transformed = selector.transform(X)
    assert transformed.shape == (len(X), len(selector.selected_features_))


def test_stability_fit_and_threshold_tuning_accept_column_sugar_and_scorer(b2_data):
    X, y, groups, _time, _weights = b2_data
    features = X[["signal", "weak", "noise", "time"]]
    options = dict(
        n_bootstrap=4,
        alpha=0.2,
        threshold=0.4,
        n_jobs=1,
        random_state=0,
        verbose=False,
    )
    from_column = sift.StabilitySelector(**options).fit(X, y, groups="group")
    from_array = sift.StabilitySelector(**options).fit(features, y, groups=groups)
    np.testing.assert_array_equal(
        from_column.selection_frequencies_,
        from_array.selection_frequencies_,
    )
    assert from_column.feature_names_in_ == list(features.columns)

    threshold, diagnostics = from_column.tune_threshold(
        X,
        y,
        thresholds=(0.4, 0.6),
        cv=3,
        scoring=get_scorer("r2"),
        groups="group",
    )
    assert threshold in {0.4, 0.6}
    assert diagnostics["n_finite"].min() == 3


def test_boruta_selector_accepts_group_and_time_column_sugar(b2_data):
    X, y, _groups, _time, _weights = b2_data
    selector = BorutaSelector(
        n_estimators=10,
        max_iter=2,
        early_stop_rounds=1,
        random_state=0,
        verbose=False,
    ).fit(X, y, groups="group", time="time")

    assert selector.feature_names_in_ == ["signal", "weak", "noise"]
    assert selector.transform(X).shape[0] == len(X)


def test_permutation_importance_metadata_sugar_matches_arrays(b2_data):
    X, y, groups, time, _weights = b2_data
    features = X[["signal", "weak", "noise"]]
    model = LinearRegression().fit(features, y)
    options = dict(n_repeats=3, n_jobs=1, random_state=0)

    from_columns = sift.permutation_importance(
        model,
        X,
        y,
        groups="group",
        time="time",
        **options,
    )
    from_arrays = sift.permutation_importance(
        model,
        features,
        y,
        groups=groups,
        time=time,
        **options,
    )
    pd.testing.assert_frame_equal(from_columns, from_arrays, check_exact=True)


def test_evaluate_feature_path_metadata_sugar_and_sklearn_scorer(b2_data):
    X, y, groups, _time, weights = b2_data
    features = X[["signal", "weak", "noise"]]
    path = ["signal", "weak", "noise"]
    options = dict(
        feature_path=path,
        k_grid=[1, 2, 3],
        estimator=LinearRegression(),
        scoring=get_scorer("neg_mean_squared_error"),
        splitter=GroupKFold(n_splits=4),
        sample_weight=weights,
    )

    from_column = sift.evaluate_feature_path(X, y, groups="group", **options)
    from_array = sift.evaluate_feature_path(features, y, groups=groups, **options)
    assert from_column.best_k == from_array.best_k
    assert from_column.features == from_array.features
    assert np.isfinite(list(from_column.scores.values())).all()
    pd.testing.assert_frame_equal(
        from_column.diagnostics,
        from_array.diagnostics,
        check_exact=True,
    )
    assert from_column.diagnostics["scoring"].str.startswith("sklearn:").all()


def test_auto_k_sklearn_scorer_matches_equivalent_loss_ordering(b2_data):
    X, y, groups, _time, weights = b2_data
    features = X[["signal", "weak", "noise"]]
    path = list(features.columns)
    scorer_config = _evaluate_config(get_scorer("neg_mean_squared_error"))
    named_config = _evaluate_config("rmse")

    scorer_result = sift.select_k_auto(
        features,
        y,
        path,
        scorer_config,
        groups=groups,
        sample_weight=weights,
    )
    named_result = sift.select_k_auto(
        features,
        y,
        path,
        named_config,
        groups=groups,
        sample_weight=weights,
    )
    assert scorer_result[:2] == named_result[:2]
    assert scorer_result[2]["metric"].str.startswith("sklearn:").all()
    assert np.isfinite(scorer_result[2]["score"]).all()


def test_nested_auto_k_accepts_sklearn_scorer(b2_data):
    X, y, groups, _time, weights = b2_data
    values = X[["signal", "weak", "noise"]].to_numpy()
    config = _evaluate_config(get_scorer("neg_mean_squared_error"), nested=True)

    def build_fold_path(train_idx, val_idx, max_k):
        return NestedAutoKFold(
            train_path=values[train_idx, :max_k],
            val_path=values[val_idx, :max_k],
            feature_path=["signal", "weak", "noise"][:max_k],
        )

    result = select_k_nested(
        values,
        y,
        n_features=3,
        config=config,
        build_fold_path=build_fold_path,
        groups=groups,
        sample_weight=weights,
    )
    assert result.selected_k in {1, 2, 3}
    assert result.diagnostics["metric"].startswith("sklearn:")
    assert np.isfinite(result.diagnostics["scores"]["score"]).all()


def test_weighted_scorer_without_weight_support_raises_clearly(b2_data):
    X, y, groups, _time, weights = b2_data
    features = X[["signal", "weak", "noise"]]
    path = list(features.columns)
    scorer = make_scorer(_two_argument_mse, greater_is_better=False)

    with pytest.raises(TypeError, match="does not accept sample_weight"):
        sift.evaluate_feature_path(
            features,
            y,
            path,
            [1, 2],
            estimator=LinearRegression(),
            scoring=scorer,
            splitter=GroupKFold(n_splits=4),
            groups=groups,
            sample_weight=weights,
        )
    with pytest.raises(TypeError, match="does not accept sample_weight"):
        sift.select_k_auto(
            features,
            y,
            path,
            _evaluate_config(scorer),
            groups=groups,
            sample_weight=weights,
        )

    selector = sift.StabilitySelector(
        n_bootstrap=2,
        alpha=0.2,
        threshold=0.0,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(features, y, sample_weight=weights)
    with pytest.raises(TypeError, match="does not accept sample_weight"):
        selector.tune_threshold(
            features,
            y,
            thresholds=(0.0,),
            cv=2,
            scoring=scorer,
            sample_weight=weights,
        )

    unweighted = sift.evaluate_feature_path(
        features,
        y,
        path,
        [1, 2],
        estimator=LinearRegression(),
        scoring=scorer,
        splitter=GroupKFold(n_splits=4),
        groups=groups,
    )
    assert np.isfinite(list(unweighted.scores.values())).all()


def test_stability_penalty_alias_is_clone_safe_and_matches_alpha(b2_data):
    X, y, _groups, _time, _weights = b2_data
    features = X[["signal", "weak", "noise"]]
    common = dict(
        n_bootstrap=4,
        threshold=0.4,
        n_jobs=1,
        random_state=0,
        verbose=False,
    )
    alpha = sift.StabilitySelector(alpha=0.2, **common).fit(features, y)
    penalty_template = sift.StabilitySelector(penalty=0.2, **common)
    assert penalty_template.get_params()["penalty"] == 0.2
    penalty = clone(penalty_template).fit(features, y)
    np.testing.assert_array_equal(
        alpha.selection_frequencies_,
        penalty.selection_frequencies_,
    )
    assert alpha.alpha_ == penalty.alpha_ == 0.2

    equal = sift.StabilitySelector(alpha=0.2, penalty=0.2, **common).fit(
        features,
        y,
    )
    assert equal.alpha_ == 0.2
    with pytest.raises(ValueError, match="alpha and penalty"):
        sift.StabilitySelector(alpha=0.2, penalty=0.3, **common).fit(features, y)


@pytest.mark.parametrize(
    "method",
    ("cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"),
)
def test_select_cached_return_result_is_additive_and_complete(b2_data, method):
    X, y, _groups, _time, _weights = b2_data
    features = X[["signal", "weak", "noise"]]
    cache = sift.build_cache(features, subsample=None)
    view = sift.select_cached(
        cache,
        y,
        k=2,
        method=method,
        warn_noise_floor=False,
        return_result=True,
    )

    assert type(view) is sift.SelectionView
    assert view.metadata["cache_backed"] is True
    assert view.metadata["method"] == method
    assert view.metadata["table_complete"] is True
    assert view.raw_input["n_features"] == 3
    assert len(view.features) == len(view.indices) == 2
    assert view.support_.sum() == 2
    assert view.table["selected_index"].tolist() == [0, 1, 2]
    assert np.isfinite(view.diagnostics["objective"]).all()


def test_select_cached_result_flag_contract_and_default():
    parameter = inspect.signature(sift.select_cached).parameters["return_result"]
    assert parameter.default is False

    X = pd.DataFrame({"a": np.arange(12.0), "b": np.arange(12.0) ** 2})
    y = X["a"].to_numpy()
    cache = sift.build_cache(X, subsample=None)
    with pytest.raises(ValueError, match="cannot be combined"):
        sift.select_cached(
            cache,
            y,
            k=1,
            return_result=True,
            return_indices=True,
        )


def test_select_cached_rejects_duplicate_nan_feature_names():
    X = pd.DataFrame(
        np.arange(36.0).reshape(12, 3),
        columns=[float("nan"), float("nan"), "b"],
    )
    cache = sift.build_cache(X, subsample=None)
    y = np.arange(len(X), dtype=np.float64)
    with pytest.raises(ValueError, match="Duplicate feature names"):
        sift.select_cached(cache, y, k=1)
    with pytest.raises(ValueError, match="Duplicate feature names"):
        sift.select_cached(cache, y, k=1, return_result=True)


def test_metadata_sugar_rejects_missing_ambiguous_and_positional_names(b2_data):
    X, y, _groups, _time, _weights = b2_data
    model = LinearRegression().fit(X[["signal", "weak", "noise"]], y)
    with pytest.raises(ValueError, match="was not found"):
        sift.permutation_importance(
            model,
            X,
            y,
            groups="missing",
            n_repeats=1,
            n_jobs=1,
            random_state=0,
        )
    with pytest.raises(ValueError, match="requires X to be a pandas DataFrame"):
        sift.permutation_importance(
            model,
            X[["signal", "weak", "noise"]].to_numpy(),
            y,
            groups="group",
            n_repeats=1,
            n_jobs=1,
            random_state=0,
        )

    duplicate = X.copy()
    duplicate.columns = ["signal", "weak", "noise", "group", "group"]
    with pytest.raises(ValueError, match="ambiguous"):
        sift.select_mrmr(
            duplicate,
            y,
            k="auto",
            task="regression",
            groups="group",
            auto_k_config=_evaluate_config(),
            verbose=False,
        )


def test_none_random_state_warnings_are_caller_facing(b2_data):
    X, y, _groups, _time, _weights = b2_data
    features = X[["signal", "weak", "noise"]]
    model = LinearRegression().fit(features, y)
    with pytest.warns(FutureWarning, match="SIFT 1.0") as importance_warning:
        sift.permutation_importance(
            model,
            features,
            y,
            n_repeats=1,
            n_jobs=1,
        )
    assert importance_warning[0].filename == __file__

    with pytest.warns(FutureWarning, match="SIFT 1.0") as stability_warning:
        sift.StabilitySelector(
            n_bootstrap=2,
            alpha=0.2,
            n_jobs=1,
            verbose=False,
        ).fit(features, y)
    assert stability_warning[0].filename == __file__

    with pytest.warns(FutureWarning, match="SIFT 1.0") as wrapper_warning:
        sift.stability_regression(
            features,
            y,
            k=2,
            n_bootstrap=2,
            alpha=0.2,
            n_jobs=1,
            verbose=False,
        )
    assert wrapper_warning[0].filename == __file__
