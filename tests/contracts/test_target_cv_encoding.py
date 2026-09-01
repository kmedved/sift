"""Additive leakage-safe categorical encoding contracts."""

from __future__ import annotations

import importlib.util
import pickle

import numpy as np
import pandas as pd
import pytest

import sift
from sift._preprocess import TargetCVEncoder
from sift.selection.auto_k import AutoKConfig, select_k_auto


def _regression_categorical_data(seed: int = 1701):
    rng = np.random.default_rng(seed)
    n = 90
    category = np.resize(np.array(["low", "mid", "high"], dtype=object), n)
    y = np.select(
        [category == "low", category == "mid"],
        [-2.0, 0.5],
        default=3.0,
    ) + rng.normal(scale=0.03, size=n)
    X = pd.DataFrame(
        {
            "category": pd.Series(category, dtype="string"),
            "noise": rng.normal(size=n),
            "weak": rng.normal(size=n),
        }
    )
    return X, y


FUNCTION_ROUTES = (
    pytest.param(
        sift.select_mrmr,
        {"task": "regression", "estimator": "classic", "mrmr_backend": "serial"},
        id="mrmr",
    ),
    pytest.param(
        sift.select_jmi,
        {"task": "regression", "estimator": "r2"},
        id="jmi",
    ),
    pytest.param(
        sift.select_jmim,
        {"task": "regression", "estimator": "r2"},
        id="jmim",
    ),
    pytest.param(sift.select_cefsplus, {}, id="cefsplus"),
)


@pytest.mark.parametrize(("selector", "kwargs"), FUNCTION_ROUTES)
def test_target_cv_function_routes_need_no_optional_encoder(selector, kwargs):
    X, y = _regression_categorical_data()

    result = selector(
        X,
        y,
        k=1,
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
        return_result=True,
        **kwargs,
    )

    assert type(result) is sift.FilterSelectionResult
    assert result.selected_features == ["category"]
    assert result.selected_indices == [0]
    assert result.selector_metadata["cat_encoding"] == "target_cv"
    assert result.selector_metadata["encoding_cv"] == {
        "kind": "fixed_k",
        "n_splits": 5,
    }


def test_target_cv_binary_function_and_selector_contract():
    rng = np.random.default_rng(1702)
    n = 80
    category = np.resize(np.array(["red", "blue"], dtype=object), n)
    y = (category == "red").astype(np.int64)
    X = pd.DataFrame(
        {
            "category": pd.Categorical(category),
            "noise": rng.normal(size=n),
        }
    )

    result = sift.select_cefsplus_binary(
        X,
        y,
        k=1,
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
        return_result=True,
    )
    selector = sift.CEFSPlusBinarySelector(
        k=1,
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )
    training = selector.fit_transform(X, y)

    assert result.selected_features == ["category"]
    assert result.selector_metadata["encoding_cv"] == {
        "kind": "fixed_k",
        "n_splits": 5,
    }
    assert selector.categorical_encoding_metadata_ == result.selector_metadata["encoding_cv"]
    assert type(selector.categorical_encoder_) is TargetCVEncoder
    assert training.columns.tolist() == ["category"]
    assert np.isfinite(training.to_numpy()).all()


def test_target_cv_selector_uses_oof_training_and_target_blind_inference():
    n = 50
    X = pd.DataFrame({"id": [f"row_{i}" for i in range(n)]})
    y = np.linspace(-3.0, 4.0, n)
    selector = sift.MRMRSelector(
        k=1,
        task="regression",
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )

    training = selector.fit_transform(X, y)
    inference = selector.transform(X)

    assert type(training) is pd.DataFrame
    assert training.index.equals(X.index)
    assert training.columns.tolist() == ["id"]
    assert not np.allclose(training.to_numpy(), inference.to_numpy())
    assert selector.categorical_encoding_metadata_ == {
        "kind": "fixed_k",
        "n_splits": 5,
    }

    restored = pickle.loads(pickle.dumps(selector))
    probe = pd.DataFrame({"id": ["row_0", "unseen"]})
    np.testing.assert_allclose(
        restored.transform(probe).to_numpy(),
        selector.transform(probe).to_numpy(),
    )


def test_target_cv_unknown_and_missing_categories_have_stable_inference_rules():
    X = pd.DataFrame(
        {"category": ["a", "a", None, np.nan, "b", "b", "c", "c", "d", "d"]}
    )
    y = np.array([0.0, 0.0, 10.0, 10.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0])
    encoder = TargetCVEncoder(["category"], target_type="continuous", cv=2)
    training = encoder.fit_transform(X, y)

    transformed = encoder.transform(
        pd.DataFrame({"category": ["unseen", None, np.nan, "a"]})
    )
    global_mean = float(encoder.encoder_.target_mean_)

    assert training.index.equals(X.index)
    assert pd.api.types.is_float_dtype(training["category"])
    assert transformed.iloc[0, 0] == pytest.approx(global_mean)
    assert transformed.iloc[1, 0] == pytest.approx(transformed.iloc[2, 0])
    assert transformed.iloc[1, 0] != pytest.approx(global_mean)
    assert transformed.iloc[3, 0] != pytest.approx(global_mean)


def test_target_cv_rejects_multiclass_until_block_expansion_exists():
    X = pd.DataFrame({"category": np.resize(["a", "b", "c"], 45)})
    y = np.resize([0, 1, 2], 45)

    with pytest.raises(ValueError, match="does not yet support multiclass.*block-aware"):
        sift.MRMRSelector(
            k=1,
            task="classification",
            cat_encoding="target_cv",
            subsample=None,
            verbose=False,
        ).fit(X, y)


def test_target_cv_high_cardinality_id_does_not_beat_real_signal():
    rng = np.random.default_rng(1703)
    n = 240
    signal = rng.normal(size=n)
    y = signal + rng.normal(scale=0.75, size=n)
    X = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(n)],
            "signal": signal,
            "noise": rng.normal(size=n),
        }
    )

    safe = sift.select_mrmr(
        X,
        y,
        1,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_features=["id"],
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )
    assert safe == ["signal"]

    if importlib.util.find_spec("category_encoders") is not None:
        unsafe = sift.select_mrmr(
            X,
            y,
            1,
            task="regression",
            estimator="classic",
            mrmr_backend="serial",
            cat_features=["id"],
            cat_encoding="target",
            allow_full_data_target_encoding=True,
            subsample=None,
            verbose=False,
        )
        assert unsafe == ["id"]


def test_target_cv_auto_k_split_evaluation_refits_encoder_inside_split():
    X, y = _regression_categorical_data(1704)
    config = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        selection_rule="best",
    )

    best_k, selected, diagnostics = select_k_auto(
        X,
        y,
        list(X.columns[:2]),
        config,
        time=np.arange(len(X)),
        task="regression",
        cat_encoding="target_cv",
        target_cv_smoothing=20.0,
    )

    assert best_k in {1, 2}
    assert selected == list(X.columns[:best_k])
    assert diagnostics["selected"].sum() == 1


def test_target_cv_weighted_map_matches_integer_row_replication():
    X = pd.DataFrame({"category": ["a", "a", "b", "b", None, None]})
    y = np.array([0.0, 4.0, 3.0, 9.0, -2.0, 2.0])
    weight = np.array([1, 3, 2, 1, 4, 2])
    probe = pd.DataFrame({"category": ["a", "b", None, "unseen"]})

    weighted = TargetCVEncoder(
        ["category"],
        target_type="continuous",
        smooth=2.5,
        cv=3,
    ).fit(X, y, sample_weight=weight)
    repeated = TargetCVEncoder(
        ["category"],
        target_type="continuous",
        smooth=2.5,
        cv=3,
    ).fit(X.loc[X.index.repeat(weight)].reset_index(drop=True), np.repeat(y, weight))

    np.testing.assert_allclose(
        weighted.transform(probe).to_numpy(),
        repeated.transform(probe).to_numpy(),
        rtol=1e-13,
        atol=1e-13,
    )


def test_target_cv_zero_weight_targets_do_not_affect_custom_encoding():
    X = pd.DataFrame({"category": ["a", "a", "b", "b", "c", "c"]})
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    weight = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    changed = y.copy()
    changed[-2:] = [1e9, -1e9]

    first = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=2
    )
    second = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=2
    )
    first_oof = first.fit_transform(X, y, sample_weight=weight)
    second_oof = second.fit_transform(X, changed, sample_weight=weight)

    np.testing.assert_allclose(first_oof.to_numpy(), second_oof.to_numpy())
    np.testing.assert_allclose(
        first.transform(X).to_numpy(),
        second.transform(X).to_numpy(),
    )


def test_target_cv_group_folds_never_use_the_held_out_groups_targets():
    groups = np.repeat(np.arange(6), 3)
    X = pd.DataFrame(
        {"category": np.tile(["shared", "shared", "private"], 6)}
    )
    y = np.arange(len(X), dtype=float)
    changed = y.copy()
    changed[groups == 2] += 100_000.0

    first = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=3
    )
    second = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=3
    )
    first_oof = first.fit_transform(X, y, groups=groups)
    second_oof = second.fit_transform(X, changed, groups=groups)

    np.testing.assert_allclose(
        first_oof.loc[groups == 2].to_numpy(),
        second_oof.loc[groups == 2].to_numpy(),
    )
    assert first.encoding_cv_ == {"kind": "group", "n_splits": 3}
    assert first.effective_sample_weight_ is None


def test_target_cv_time_folds_use_strict_history_and_zero_weight_warmup():
    X = pd.DataFrame({"category": ["a"] * 6})
    y = np.array([100.0, 200.0, 1.0, 3.0, 10.0, 14.0])
    time = np.repeat(np.arange(3), 2)

    encoder = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=0.0, cv=3
    )
    training = encoder.fit_transform(X, y, time=time)

    np.testing.assert_allclose(training.iloc[:2, 0], 0.0)
    np.testing.assert_allclose(training.iloc[2:4, 0], 150.0)
    np.testing.assert_allclose(training.iloc[4:, 0], 76.0)
    np.testing.assert_array_equal(
        encoder.effective_sample_weight_,
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert encoder.encoding_cv_ == {"kind": "time", "n_splits": 3}


def test_target_cv_time_prior_is_target_independent_and_keeps_warmup():
    X = pd.DataFrame({"category": ["a"] * 6})
    y = np.array([100.0, 200.0, 1.0, 3.0, 10.0, 14.0])
    time = np.repeat(np.arange(3), 2)
    encoder = TargetCVEncoder(
        ["category"],
        target_type="continuous",
        smooth=0.0,
        cv=3,
        target_prior=-7.5,
    )

    training = encoder.fit_transform(X, y, time=time)

    np.testing.assert_allclose(training.iloc[:2, 0], -7.5)
    np.testing.assert_allclose(encoder.effective_sample_weight_, np.ones(len(X)))


def test_target_cv_time_exclude_policy_removes_warmup_from_selection_weight():
    X = pd.DataFrame({"category": ["a"] * 6})
    y = np.array([100.0, 200.0, 1.0, 3.0, 10.0, 14.0])
    time = np.repeat(np.arange(3), 2)
    encoder = TargetCVEncoder(
        ["category"],
        target_type="continuous",
        smooth=0.0,
        cv=3,
        warmup_policy="exclude",
    )

    encoder.fit_transform(X, y, time=time)

    np.testing.assert_array_equal(
        encoder.effective_sample_weight_,
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )


def test_target_cv_custom_modes_require_explicit_smoothing():
    X, y = _regression_categorical_data(1705)
    with pytest.raises(ValueError, match="target_cv_smoothing.*explicit"):
        TargetCVEncoder(["category"], target_type="continuous").fit_transform(
            X,
            y,
            groups=np.repeat(np.arange(9), 10),
        )


@pytest.mark.parametrize(("selector", "kwargs"), FUNCTION_ROUTES)
def test_target_cv_weighted_function_routes_use_custom_cross_fitting(
    selector,
    kwargs,
):
    X, y = _regression_categorical_data(1707)
    result = selector(
        X,
        y,
        1,
        cat_encoding="target_cv",
        target_cv_smoothing=2.0,
        sample_weight=np.linspace(0.5, 2.0, len(X)),
        subsample=None,
        verbose=False,
        return_result=True,
        **kwargs,
    )

    assert result.selected_features == ["category"]
    assert result.selector_metadata["encoding_cv"] == {
        "kind": "fixed_k",
        "n_splits": 5,
    }


def test_target_cv_weighted_binary_function_route_uses_custom_cross_fitting():
    X, y_reg = _regression_categorical_data(1708)
    y = (y_reg > np.median(y_reg)).astype(np.int64)
    result = sift.select_cefsplus_binary(
        X,
        y,
        1,
        cat_encoding="target_cv",
        target_cv_smoothing=2.0,
        sample_weight=np.linspace(0.5, 2.0, len(X)),
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert result.selected_features == ["category"]
    assert result.selector_metadata["encoding_cv"] == {
        "kind": "fixed_k",
        "n_splits": 5,
    }


@pytest.mark.parametrize(
    ("strategy", "context_name", "context", "expected_kind"),
    [
        ("group_cv", "groups", np.repeat(np.arange(9), 10), "group"),
        ("time_holdout", "time", np.repeat(np.arange(18), 5), "time"),
    ],
)
def test_target_cv_contextual_function_route_reports_actual_fold_kind(
    strategy,
    context_name,
    context,
    expected_kind,
):
    X, y = _regression_categorical_data(1709)
    config = AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        min_k=1,
        max_k=2,
        n_splits=3,
        selection_rule="best",
    )
    result = sift.select_mrmr(
        X,
        y,
        "auto",
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_encoding="target_cv",
        target_cv_n_splits=3,
        target_cv_smoothing=2.0,
        auto_k_config=config,
        subsample=None,
        verbose=False,
        return_result=True,
        **{context_name: context},
    )

    assert result.selector_metadata["encoding_cv"] == {
        "kind": expected_kind,
        "n_splits": 3,
    }


def test_boruta_target_cv_retains_encoder_for_inference():
    X, y = _regression_categorical_data(1706)
    selector = sift.BorutaSelector(
        n_estimators=20,
        max_iter=2,
        cat_encoding="target_cv",
        verbose=False,
    ).fit(X, y)

    assert type(selector.categorical_encoder_) is TargetCVEncoder
    assert selector.categorical_encoding_metadata_ == {
        "kind": "fixed_k",
        "n_splits": 5,
    }
    transformed = selector.transform(X.iloc[:4])
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 4
