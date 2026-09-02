"""Additive leakage-safe categorical encoding contracts."""

from __future__ import annotations

import importlib.util
import inspect
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import sift
from sift._preprocess import TargetCVEncoder
from sift.selection.auto_k import AutoKConfig, select_k_auto


def _fold_marker_data(seed: int, n: int = 600):
    """Unique-ID fixture used by the §1.1 centering acceptance tests.

    ``id`` is unique per row, so under raw fold-prior emissions every row was
    encoded with its complement folds' prior and the column became an
    anti-correlated fold marker that entered mRMR's top three.
    """
    rng = np.random.default_rng(seed)
    cities = np.array([f"city_{i}" for i in range(8)], dtype=object)
    city = rng.choice(cities, size=n)
    effects = dict(zip(cities.tolist(), rng.normal(size=8).tolist()))
    x1 = rng.normal(size=n)
    y = x1 + np.array([effects[value] for value in city]) + rng.normal(
        scale=0.3, size=n
    )
    X = pd.DataFrame(
        {
            "id": [f"id_{i}" for i in range(n)],
            "city": pd.Series(city, dtype=object),
            "x1": x1,
            "x_noise": rng.normal(size=n),
        }
    )
    return X, y


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
    X = pd.DataFrame({"team": [f"team_{i % 5}" for i in range(n)]})
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
    assert training.columns.tolist() == ["team"]
    # Out-of-fold training values come from fold-local maps; target-blind
    # inference reuses the full-fit centered map, so the two differ.
    assert not np.allclose(training.to_numpy(), inference.to_numpy())
    assert selector.categorical_encoding_metadata_ == {
        "kind": "fixed_k",
        "n_splits": 5,
    }

    restored = pickle.loads(pickle.dumps(selector))
    probe = pd.DataFrame({"team": ["team_0", "unseen"]})
    np.testing.assert_allclose(
        restored.transform(probe).to_numpy(),
        selector.transform(probe).to_numpy(),
    )
    # An unseen inference category emits the zero centered effect.
    assert float(selector.transform(probe).iloc[1, 0]) == 0.0


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

    assert training.index.equals(X.index)
    assert pd.api.types.is_float_dtype(training["category"])
    # Values are centered effects, so an unseen category emits exactly zero
    # (the raw global-mean estimate before centering) instead of a prior that
    # could identify the fitting rows.
    assert transformed.iloc[0, 0] == 0.0
    # Missing values stay one learned category with its own nonzero effect, and
    # every pandas missing sentinel maps to the same value.
    assert transformed.iloc[1, 0] == pytest.approx(transformed.iloc[2, 0])
    assert transformed.iloc[1, 0] != pytest.approx(0.0)
    assert transformed.iloc[3, 0] != pytest.approx(0.0)
    # The full-fit centered map is exactly the raw estimate minus its prior.
    assert encoder.category_maps_["category"]["a"] == pytest.approx(
        float(transformed.iloc[3, 0])
    )


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


def test_target_cv_fit_reports_the_split_count_fit_transform_would_use():
    """``fit`` counts folds over active rows, exactly like ``fit_transform``.

    ``fit`` used to run the effective-split calculation over *all* rows, so a
    frame whose first six rows carry zero weight advertised ``n_splits=5`` while
    ``fit_transform`` cross-fitted with four.
    """
    X = pd.DataFrame({"category": list("aabbccddee")})
    y = np.arange(10, dtype=np.float64)
    weight = np.array([0.0] * 6 + [1.0] * 4)

    fitted = TargetCVEncoder(["category"], target_type="continuous", smooth=1.0, cv=5)
    fitted.fit(X, y, sample_weight=weight)
    cross_fitted = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=5
    )
    cross_fitted.fit_transform(X, y, sample_weight=weight)

    assert fitted.encoding_cv_ == {"kind": "fixed_k", "n_splits": 4}
    assert fitted.encoding_cv_ == cross_fitted.encoding_cv_
    assert fitted.n_splits_ == cross_fitted.n_splits_


def test_target_cv_fit_split_count_matches_fit_transform_on_a_binary_target():
    X = pd.DataFrame({"category": list("aabbccddeeff")})
    y = np.array([0, 1] * 6)
    # Only three rows of each class stay active, so both routes must clamp to 3.
    weight = np.array([0.0] * 6 + [1.0] * 6)

    fitted = TargetCVEncoder(["category"], target_type="binary", smooth=1.0, cv=5)
    fitted.fit(X, y, sample_weight=weight)
    cross_fitted = TargetCVEncoder(["category"], target_type="binary", smooth=1.0, cv=5)
    cross_fitted.fit_transform(X, y, sample_weight=weight)

    assert fitted.encoding_cv_ == cross_fitted.encoding_cv_ == {
        "kind": "fixed_k",
        "n_splits": 3,
    }


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


def _time_fold_fixture():
    """Two categories over three timestamp blocks.

    With centering a single-category column collapses to all zeros, so the
    fixture carries two categories and pins the centered contrast instead.
    """
    X = pd.DataFrame({"category": ["a", "b", "a", "b", "a", "b"]})
    y = np.array([100.0, 200.0, 1.0, 3.0, 10.0, 14.0])
    time = np.repeat(np.arange(3), 2)
    return X, y, time


def test_target_cv_time_folds_use_strict_history_and_zero_weight_warmup():
    X, y, time = _time_fold_fixture()

    encoder = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=0.0, cv=3
    )
    training = encoder.fit_transform(X, y, time=time)

    # Block 0 has no history and stays at the neutral centered effect.
    np.testing.assert_allclose(training.iloc[:2, 0], 0.0)
    # Block 1 is encoded from block 0 alone: prior 150, so a -> -50, b -> +50.
    np.testing.assert_allclose(training.iloc[2:4, 0], [-50.0, 50.0])
    # Block 2 is encoded from blocks 0-1: prior 76, a mean 50.5, b mean 101.5.
    np.testing.assert_allclose(training.iloc[4:, 0], [-25.5, 25.5])
    np.testing.assert_array_equal(
        encoder.effective_sample_weight_,
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    )
    assert encoder.encoding_cv_ == {"kind": "time", "n_splits": 3}


def test_target_cv_single_category_time_column_carries_no_signal():
    X = pd.DataFrame({"category": ["a"] * 6})
    y = np.array([100.0, 200.0, 1.0, 3.0, 10.0, 14.0])
    time = np.repeat(np.arange(3), 2)

    encoder = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=0.0, cv=3
    )
    training = encoder.fit_transform(X, y, time=time)

    # Before centering these rows carried each fold's history prior, which is a
    # pure timestamp marker. Centered, the constant column is exactly zero.
    np.testing.assert_allclose(training.to_numpy(), 0.0)


def test_target_cv_time_prior_is_target_independent_and_keeps_warmup():
    X, y, time = _time_fold_fixture()
    encoder = TargetCVEncoder(
        ["category"],
        target_type="continuous",
        smooth=0.0,
        cv=3,
        target_prior=-7.5,
    )

    training = encoder.fit_transform(X, y, time=time)

    # An explicit target-independent prior lets the warmup rows stay in the
    # selection fit; centered against their own prior they emit a zero effect.
    np.testing.assert_allclose(training.iloc[:2, 0], 0.0)
    np.testing.assert_allclose(training.iloc[2:4, 0], [-50.0, 50.0])
    np.testing.assert_allclose(encoder.effective_sample_weight_, np.ones(len(X)))


def test_target_cv_time_exclude_policy_removes_warmup_from_selection_weight():
    X, y, time = _time_fold_fixture()
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


@pytest.mark.parametrize(
    ("context_name", "context", "expected_kind"),
    [
        ("sample_weight", np.linspace(0.5, 2.0, 90), "fixed_k"),
        ("groups", np.repeat(np.arange(9), 10), "group"),
        ("time", np.repeat(np.arange(9), 10), "time"),
    ],
)
def test_target_cv_auto_smoothing_is_available_on_every_contextual_mode(
    context_name,
    context,
    expected_kind,
):
    """``smooth="auto"`` is defined by weighted row mass, so it works everywhere.

    Before this was fixed, every contextual call raised
    ``ValueError: target_cv_smoothing must be an explicit non-negative float``,
    which made the documented weighted generalization unreachable.
    """
    X, y = _regression_categorical_data(1705)
    encoder = TargetCVEncoder(["category"], target_type="continuous", smooth="auto")

    encoded = encoder.fit_transform(X, y, **{context_name: context})

    assert encoder.encoding_cv_["kind"] == expected_kind
    assert np.isfinite(encoded["category"].to_numpy()).all()


def test_target_cv_auto_smoothing_weighted_map_matches_integer_row_replication():
    """The weighted ``"auto"`` prior is the integer formula with weighted mass.

    Mirrors ``test_target_cv_weighted_map_matches_integer_row_replication`` for
    the empirical-Bayes path: replicating a row ``m`` times must equal giving it
    weight ``m``.
    """
    X = pd.DataFrame({"category": ["a", "a", "b", "b", None, None]})
    y = np.array([0.0, 4.0, 3.0, 9.0, -2.0, 2.0])
    weight = np.array([1, 3, 2, 1, 4, 2])
    probe = pd.DataFrame({"category": ["a", "b", None, "unseen"]})

    weighted = TargetCVEncoder(
        ["category"], target_type="continuous", smooth="auto", cv=3
    ).fit(X, y, sample_weight=weight)
    repeated = TargetCVEncoder(
        ["category"], target_type="continuous", smooth="auto", cv=3
    ).fit(X.loc[X.index.repeat(weight)].reset_index(drop=True), np.repeat(y, weight))

    np.testing.assert_allclose(
        weighted.transform(probe).to_numpy(),
        repeated.transform(probe).to_numpy(),
        rtol=1e-13,
        atol=1e-13,
    )


def test_target_cv_auto_smoothing_unit_weights_match_the_unweighted_path():
    X, y = _regression_categorical_data(1706)

    weighted = TargetCVEncoder(["category"], target_type="continuous", smooth="auto")
    unweighted = TargetCVEncoder(["category"], target_type="continuous", smooth="auto")
    weighted_oof = weighted.fit_transform(X, y, sample_weight=np.ones(len(X)))
    unweighted_oof = unweighted.fit_transform(X, y)

    np.testing.assert_array_equal(
        weighted_oof.to_numpy(dtype=np.float64),
        unweighted_oof.to_numpy(dtype=np.float64),
    )
    assert weighted.encoding_cv_ == unweighted.encoding_cv_


def test_target_cv_auto_smoothing_binary_route_accepts_balanced_class_weight():
    """``class_weight="balanced"`` feeds weights in, which used to reject "auto"."""
    X, y_reg = _regression_categorical_data(1709)
    y = (y_reg > np.median(y_reg)).astype(np.int64)

    result = sift.select_cefsplus_binary(
        X,
        y,
        1,
        cat_encoding="target_cv",
        class_weight="balanced",
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert result.selected_features == ["category"]
    assert result.selector_metadata["encoding_cv"] == {
        "kind": "fixed_k",
        "n_splits": 5,
    }

    selector = sift.CEFSPlusBinarySelector(
        k=1,
        cat_features=["category"],
        cat_encoding="target_cv",
        class_weight="balanced",
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y)
    assert list(selector.get_feature_names_out()) == ["category"]


def test_target_cv_auto_smoothing_still_rejects_a_massless_fit():
    """The one genuinely undefined case: no positive weight mass.

    Neither the weighted prior ``sum(w*y)/sum(w)`` nor the weighted target
    variance the empirical-Bayes shrinkage needs exists then, so ``"auto"``
    keeps raising -- for exactly the same reason an explicit float does.
    """
    X, y = _regression_categorical_data(1710)

    for smooth in ("auto", 2.0):
        with pytest.raises(ValueError, match="at least one positive value"):
            TargetCVEncoder(
                ["category"], target_type="continuous", smooth=smooth
            ).fit_transform(X, y, sample_weight=np.zeros(len(X)))


def test_target_cv_rejects_smoothing_values_that_are_neither_auto_nor_a_float():
    X, y = _regression_categorical_data(1711)

    with pytest.raises(ValueError, match="'auto' or a non-negative float"):
        TargetCVEncoder(
            ["category"], target_type="continuous", smooth="soft"
        ).fit_transform(X, y)
    with pytest.raises(ValueError, match="finite and >= 0"):
        TargetCVEncoder(
            ["category"], target_type="continuous", smooth=-1.0
        ).fit_transform(X, y)


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


# --- 1.1 centering acceptance ---------------------------------------------


def test_unique_id_column_has_one_constant_oof_value_and_zero_variance():
    """Acceptance 1.1.1."""
    X, y = _fold_marker_data(0)

    encoder = TargetCVEncoder(["id"], target_type="continuous", cv=5)
    training = encoder.fit_transform(X.loc[:, ["id"]], y)
    column = training["id"].to_numpy()

    assert np.unique(column).size == 1
    assert float(column.var()) == 0.0
    np.testing.assert_allclose(column, 0.0)
    # Cross-fitting is still real: the full-fit inference map is not constant.
    assert float(encoder.transform(X.loc[:, ["id"]])["id"].to_numpy().var()) > 0.0


@pytest.mark.parametrize("k", [2, 3])
def test_unique_id_never_outranks_a_nonconstant_noise_feature_in_mrmr(k):
    """Acceptance 1.1.2 for mRMR across k>1 on the reproduced 8-seed design."""
    for seed in range(8):
        X, y = _fold_marker_data(seed)
        selected = sift.select_mrmr(
            X,
            y,
            k,
            task="regression",
            estimator="classic",
            mrmr_backend="serial",
            cat_encoding="target_cv",
            subsample=None,
            verbose=False,
        )
        assert "id" not in selected, f"seed={seed} k={k} selected {selected}"
        assert len(selected) == k


def test_unique_id_carries_zero_relevance_in_the_reported_ranking():
    """Acceptance 1.1.2: zero relevance, ranked below nonconstant noise."""
    X, y = _fold_marker_data(0)

    result = sift.select_mrmr(
        X,
        y,
        3,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
        return_result=True,
    )
    ranking = result.ranking_.set_index("feature")

    assert float(ranking.loc["id", "relevance"]) == pytest.approx(0.0, abs=1e-12)
    assert float(ranking.loc["x_noise", "relevance"]) > 0.0
    assert int(ranking.loc["id", "rank"]) > int(ranking.loc["x_noise", "rank"])


def _near_unique_id_data(seed: int, *, shared_target: bool, rows_per_id: int = 2):
    """300 identifiers x ``rows_per_id`` rows, so every level recurs in-fold.

    With ``shared_target``, an identifier's rows share a latent target, so the
    encoding legitimately carries sibling rows' targets.  Without it, the target
    is independent of the identifier and only fold-marker leakage could make the
    column relevant.
    """
    rng = np.random.default_rng(seed)
    n_ids = 300
    ids = np.repeat(np.arange(n_ids), rows_per_id)
    n = ids.size
    if shared_target:
        y = np.repeat(rng.normal(size=n_ids), rows_per_id) + rng.normal(
            scale=0.05, size=n
        )
    else:
        y = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "id": [f"id_{value}" for value in ids],
            "real": 0.5 * y + rng.normal(size=n),
        }
    )
    return X, y


def test_near_unique_ids_with_a_shared_target_stay_selectable_by_design():
    """The documented boundary of the centering guarantee.

    Centering neutralizes only *unseen-in-fold* emissions.  A level that appears
    twice in training still transmits its sibling row's target, which is
    ordinary high-cardinality target-encoding behavior rather than the fold
    marker 1.1 closed -- so this column is genuinely informative here and is
    expected to be selected.  Callers who must not have that cross-row
    information reach selection drop ID-like columns or pass ``groups=``.
    """
    X, y = _near_unique_id_data(11, shared_target=True)

    encoded = TargetCVEncoder(
        ["id"], target_type="continuous", smooth="auto"
    ).fit_transform(X.loc[:, ["id"]], y)["id"].to_numpy()

    assert float(np.var(encoded)) > 0.0
    assert abs(float(np.corrcoef(encoded, y)[0, 1])) > 0.5

    # k=1 makes the assertion falsifiable: the encoded near-unique ID must beat
    # the numeric feature outright, not merely appear in a two-of-two selection.
    selected = sift.select_mrmr(
        X,
        y,
        1,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )
    assert selected == ["id"]


def test_near_unique_ids_without_a_shared_target_are_not_selected():
    """The same shape, no real signal: the residual is information, not a marker."""
    X, y = _near_unique_id_data(12, shared_target=False)

    encoded = TargetCVEncoder(
        ["id"], target_type="continuous", smooth="auto"
    ).fit_transform(X.loc[:, ["id"]], y)["id"].to_numpy()

    assert abs(float(np.corrcoef(encoded, y)[0, 1])) < 0.1

    selected = sift.select_mrmr(
        X,
        y,
        1,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )
    assert selected == ["real"]


def test_grouping_an_identifiers_rows_into_one_fold_removes_the_residual():
    """``groups=`` is the documented remedy: no sibling row is ever in-fold."""
    X, y = _near_unique_id_data(11, shared_target=True)
    ids = X["id"].to_numpy()

    encoded = TargetCVEncoder(
        ["id"], target_type="continuous", smooth="auto"
    ).fit_transform(X.loc[:, ["id"]], y, groups=ids)["id"].to_numpy()

    np.testing.assert_allclose(encoded, 0.0)


@pytest.mark.parametrize(
    "auto_k_config",
    [
        pytest.param(None, id="no_config_router"),
        pytest.param(
            AutoKConfig(k_method="penalized_objective", min_k=1, max_k=4),
            id="penalized_objective",
        ),
        pytest.param(
            AutoKConfig(k_method="gaussian_cv", strategy="kfold", min_k=1, max_k=4),
            id="gaussian_cv",
        ),
    ],
)
def test_unique_id_never_outranks_noise_on_cefsplus_routes(auto_k_config):
    """Acceptance 1.1.2 for select_cefsplus auto-k routes."""
    for seed in range(8):
        X, y = _fold_marker_data(seed)
        selected = sift.select_cefsplus(
            X,
            y,
            k="auto",
            auto_k_config=auto_k_config,
            cat_encoding="target_cv",
            subsample=None,
            verbose=False,
        )
        # Exclusion is not required once k exhausts every nonconstant
        # candidate, which cannot happen here: the centered id column is
        # constant and therefore never a viable candidate at all.
        assert "id" not in selected, f"seed={seed} selected {selected}"


def test_requesting_every_column_still_succeeds_without_requiring_exclusion():
    """Acceptance 1.1.2 carve-out: exclusion is not required at full width.

    With ``k`` equal to the input width the contract only requires the call to
    succeed and to keep every nonconstant candidate; whether the constant,
    zero-relevance id column is returned as filler is not pinned.
    """
    X, y = _fold_marker_data(0)

    selected = sift.select_mrmr(
        X,
        y,
        4,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )

    assert {"city", "x1", "x_noise"}.issubset(selected)
    assert set(selected).issubset(set(X.columns))


def test_group_proxy_gains_no_relevance_from_complement_fold_priors():
    """Acceptance 1.1.3 for grouped folds."""
    groups = np.repeat(np.arange(20), 30)
    rng = np.random.default_rng(1710)
    y = groups.astype(float) + rng.normal(scale=0.5, size=groups.size)
    X = pd.DataFrame({"group_proxy": [f"g_{value}" for value in groups]})

    encoder = TargetCVEncoder(
        ["group_proxy"], target_type="continuous", smooth=1.0, cv=5
    )
    column = encoder.fit_transform(X, y, groups=groups)["group_proxy"].to_numpy()

    # Under raw fold priors this column reached |corr| ~ 0.38 with y purely
    # from the complement folds' means.
    assert float(column.var()) == 0.0
    np.testing.assert_allclose(column, 0.0)


def test_timestamp_proxy_gains_no_relevance_from_complement_fold_priors():
    """Acceptance 1.1.3 for time folds."""
    time = np.repeat(np.arange(20), 30)
    rng = np.random.default_rng(1711)
    y = time.astype(float) + rng.normal(scale=0.5, size=time.size)
    X = pd.DataFrame({"time_proxy": [f"t_{value}" for value in time]})

    encoder = TargetCVEncoder(
        ["time_proxy"], target_type="continuous", smooth=1.0, cv=5
    )
    column = encoder.fit_transform(X, y, time=time)["time_proxy"].to_numpy()

    # Under raw history priors this column reached |corr| ~ 0.97 with y.
    assert float(column.var()) == 0.0
    np.testing.assert_allclose(column, 0.0)


def test_mutating_held_out_group_targets_leaves_centered_effects_unchanged():
    """Acceptance 1.1.4 for grouped folds."""
    groups = np.repeat(np.arange(6), 4)
    X = pd.DataFrame({"category": np.tile(["shared", "other"], 12)})
    y = np.arange(len(X), dtype=float)
    mutated = y.copy()
    mutated[groups == 3] += 1e6

    baseline = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=3
    ).fit_transform(X, y, groups=groups)
    changed = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=1.0, cv=3
    ).fit_transform(X, mutated, groups=groups)

    held_out = groups == 3
    np.testing.assert_allclose(
        baseline.loc[held_out, "category"].to_numpy(),
        changed.loc[held_out, "category"].to_numpy(),
    )


def test_mutating_future_targets_leaves_earlier_centered_effects_unchanged():
    """Acceptance 1.1.4 for time folds."""
    X, y, time = _time_fold_fixture()
    mutated = y.copy()
    mutated[4:] += 1e6

    baseline = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=0.0, cv=3
    ).fit_transform(X, y, time=time)
    changed = TargetCVEncoder(
        ["category"], target_type="continuous", smooth=0.0, cv=3
    ).fit_transform(X, mutated, time=time)

    # Blocks 0 and 1 are encoded from strictly earlier history only.
    np.testing.assert_allclose(
        baseline.iloc[:4, 0].to_numpy(), changed.iloc[:4, 0].to_numpy()
    )


def test_inference_categories_are_deterministic_after_pickle_round_trip():
    """Acceptance 1.1.5."""
    X = pd.DataFrame(
        {"category": ["a", "a", "b", "b", None, np.nan, "c", "c", "a", "b"]}
    )
    y = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0])
    encoder = TargetCVEncoder(["category"], target_type="continuous", cv=2)
    encoder.fit_transform(X, y)

    probe = pd.DataFrame({"category": ["a", None, np.nan, pd.NA, "unseen"]})
    before = encoder.transform(probe).to_numpy()
    restored = pickle.loads(pickle.dumps(encoder))
    after = restored.transform(probe).to_numpy()

    np.testing.assert_array_equal(before, after)
    # Known, missing, and unseen each keep a stable, distinct rule.
    assert before[4, 0] == 0.0
    assert before[1, 0] == before[2, 0] == before[3, 0]
    assert before[0, 0] != before[1, 0]
    np.testing.assert_array_equal(before, encoder.transform(probe).to_numpy())


# --- 1.2 metadata acceptance ----------------------------------------------


def _binary_time_block_fixture():
    """Six timestamp blocks with the two earliest zero-weighted."""
    rng = np.random.default_rng(1712)
    n = 120
    time = np.repeat(np.arange(6), 20)
    category = np.resize(np.array(["p", "q", "r"], dtype=object), n)
    y = np.resize(np.array([0, 1, 1, 0], dtype=np.int64), n)
    X = pd.DataFrame(
        {
            "category": pd.Series(category, dtype=object),
            "noise": rng.normal(size=n),
        }
    )
    sample_weight = np.ones(n)
    sample_weight[time < 2] = 0.0
    return X, y, time, sample_weight


def test_binary_time_route_reports_the_encoders_active_fold_count():
    """§1.2: the encoder and the public result both report four active folds."""
    X, y, time, sample_weight = _binary_time_block_fixture()

    encoder = TargetCVEncoder(
        ["category"], target_type="binary", smooth=2.0, cv=5
    )
    encoder.fit_transform(
        X, y.astype(float), sample_weight=sample_weight, time=time
    )
    result = sift.select_cefsplus_binary(
        X,
        y,
        k="auto",
        auto_k_config=AutoKConfig(
            k_method="evaluate",
            strategy="time_holdout",
            min_k=1,
            max_k=2,
            n_splits=2,
            selection_rule="best",
        ),
        time=time,
        sample_weight=sample_weight,
        cat_encoding="target_cv",
        target_cv_smoothing=2.0,
        subsample=None,
        verbose=False,
        return_result=True,
    )

    # Two of the six timestamp blocks carry zero weight, so only four blocks
    # are active. Reconstructing the count from all rows reported five.
    assert encoder.encoding_cv_ == {"kind": "time", "n_splits": 4}
    assert result.selector_metadata["encoding_cv"] == {
        "kind": "time",
        "n_splits": 4,
    }


@pytest.mark.parametrize(("selector", "kwargs"), FUNCTION_ROUTES)
def test_target_cv_metadata_emits_no_stray_top_level_fold_keys(selector, kwargs):
    """§1.2: only the nested encoding_cv shape survives."""
    X, y = _regression_categorical_data(1713)

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

    assert result.selector_metadata["encoding_cv"] == {
        "kind": "fixed_k",
        "n_splits": 5,
    }
    assert "kind" not in result.selector_metadata
    assert "n_splits" not in result.selector_metadata


def test_requested_but_absent_categorical_column_does_not_break_rich_results():
    """§1.2: matches the silent legacy convention instead of raising."""
    rng = np.random.default_rng(1714)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = rng.normal(size=60)
    y_binary = (y > 0).astype(np.int64)

    binary_result = sift.select_cefsplus_binary(
        X,
        y_binary,
        k=2,
        cat_features=["missing_cat"],
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
        return_result=True,
    )
    filter_result = sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        cat_features=["missing_cat"],
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
        return_result=True,
    )

    # No encoding ran, so no encoding metadata is attached anywhere.
    assert "encoding_cv" not in binary_result.selector_metadata
    assert "encoding_cv" not in filter_result.selector_metadata
    assert len(binary_result.selected_features) == 2
    assert len(filter_result.selected_features) == 2


# --- C3 / C4 rejections ----------------------------------------------------


def test_function_routes_reject_target_cv_with_full_data_escape_hatch():
    """C3 at the function entry points."""
    X, y = _regression_categorical_data(1715)
    y_binary = (y > np.median(y)).astype(np.int64)
    message = "cannot be combined with allow_full_data_target_encoding=True"

    for selector, kwargs in (
        (sift.select_mrmr, {"task": "regression"}),
        (sift.select_jmi, {"task": "regression"}),
        (sift.select_jmim, {"task": "regression"}),
        (sift.select_cefsplus, {}),
    ):
        with pytest.raises(ValueError, match=message):
            selector(
                X,
                y,
                k=1,
                cat_encoding="target_cv",
                allow_full_data_target_encoding=True,
                verbose=False,
                **kwargs,
            )

    with pytest.raises(ValueError, match=message):
        sift.select_cefsplus_binary(
            X,
            y_binary,
            k=1,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        )


def test_selector_classes_and_boruta_reject_the_full_data_escape_hatch():
    """C3 at the selector-class and Boruta entry points."""
    X, y = _regression_categorical_data(1716)
    message = "cannot be combined with allow_full_data_target_encoding=True"

    with pytest.raises(ValueError, match=message):
        sift.MRMRSelector(
            k=1,
            task="regression",
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        ).fit(X, y)

    with pytest.raises(ValueError, match=message):
        sift.CEFSPlusSelector(
            k=1,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        ).fit(X, y)

    with pytest.raises(ValueError, match=message):
        sift.CEFSPlusBinarySelector(
            k=1,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        ).fit(X, (y > np.median(y)).astype(np.int64))

    with pytest.raises(ValueError, match=message):
        sift.BorutaSelector(
            n_estimators=10,
            max_iter=2,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        ).fit(X, y)

    with pytest.raises(ValueError, match=message):
        sift.select_boruta(
            X,
            y,
            n_estimators=10,
            max_iter=2,
            cat_encoding="target_cv",
            allow_full_data_target_encoding=True,
            verbose=False,
        )


def _knockoff_categorical_data(seed: int = 1717, n: int = 240):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "team": pd.Series(
                np.resize(np.array(["a", "b", "c", "d"], dtype=object), n),
                dtype=object,
            ),
            "signal": signal,
            "noise": rng.normal(size=n),
        }
    )
    y = signal + 0.4 * rng.normal(size=n)
    return X, y


def test_knockoff_selector_rejects_target_cv():
    """C4: target-derived preprocessing has no Model-X claim."""
    X, y = _knockoff_categorical_data()

    with pytest.raises(ValueError, match="does not support cat_encoding='target_cv'"):
        sift.KnockoffSelector(q=0.2, cat_encoding="target_cv", verbose=False).fit(X, y)

    # Function parity is deliberately not the fix: select_fdr gains no
    # cat_encoding parameter.
    assert "cat_encoding" not in inspect.signature(sift.select_fdr).parameters


def test_knockoff_legacy_supervised_encoding_warns_and_drops_the_fdr_claim():
    """C4: 0.8 compatibility retained only with a warning and no FDR claim."""
    X, y = _knockoff_categorical_data()
    y_binary = (y > 0).astype(np.int64)

    selector = sift.KnockoffSelector(q=0.2, cat_encoding="loo_logit", verbose=False)
    with pytest.warns(UserWarning, match="no FDR claim applies"):
        selector.fit(X, y_binary)

    metadata = selector.result_.selector_metadata
    assert metadata["fdr_control"] == "none"
    assert metadata["per_draw_fdr_control"] == "none"
    assert metadata["aggregation_preserves_per_draw_fdr"] is False
    assert metadata["cat_encoding"] == "loo_logit"
    assert "Model-X exchangeability" in metadata["validity_note"]


def test_knockoff_without_supervised_encoding_keeps_its_fdr_claim():
    """C4 guard: the downgrade is scoped to supervised encodings only."""
    X, y = _knockoff_categorical_data()
    numeric = X.drop(columns=["team"])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        selector = sift.KnockoffSelector(q=0.2, verbose=False).fit(numeric, y)

    metadata = selector.result_.selector_metadata
    assert metadata["fdr_control"] == "approximate_plugin"
    assert "validity_note" not in metadata
