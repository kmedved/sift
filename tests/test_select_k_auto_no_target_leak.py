import importlib.util
import warnings
import inspect

import numpy as np
import pandas as pd
import pytest

from sift import build_cache, select_cefsplus, select_cefsplus_binary, select_jmi, select_jmim, select_mrmr
import sift.selection.auto_k as auto_k_module
import sift.selection.filter_api as filter_api
import sift.selection.filter_payloads as filter_payloads
from sift.selection.auto_k import (
    AutoKConfig,
    choose_k_from_score_curve,
    select_k_auto,
    select_k_penalized_objective,
)
from sift.selection.auto_k_nested import NestedAutoKFold, select_k_nested


NESTED_MODE_ERROR = "auto_k_mode='nested'.*not implemented"


def _numeric_auto_k_data():
    rng = np.random.default_rng(123)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
    y = X["x0"].to_numpy() + 0.25 * rng.normal(size=n)
    time = np.arange(n)
    return X, y, time


def _binary_auto_k_data():
    X, y_reg, time = _numeric_auto_k_data()
    y = (y_reg > np.median(y_reg)).astype(int)
    return X, y, time


def test_select_k_auto_prefix_only_matches_default():
    X, y, time = _numeric_auto_k_data()
    feature_path = list(X.columns)

    default_cfg = AutoKConfig(
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )
    explicit_cfg = AutoKConfig(
        auto_k_mode="prefix_only",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    default_result = select_k_auto(X, y, feature_path, default_cfg, time=time)
    explicit_result = select_k_auto(X, y, feature_path, explicit_cfg, time=time)

    assert default_result[0] == explicit_result[0]
    assert default_result[1] == explicit_result[1]
    pd.testing.assert_frame_equal(default_result[2], explicit_result[2])


def test_select_k_auto_nested_mode_raises():
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    with pytest.raises(NotImplementedError, match=NESTED_MODE_ERROR):
        select_k_auto(X, y, list(X.columns), cfg, time=time)


def test_select_k_auto_clamps_min_k_to_path_length():
    X, y, time = _numeric_auto_k_data()
    X = X.iloc[:, :3]
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=10,
        max_k=10,
        val_frac=0.25,
    )

    best_k, selected, diag = select_k_auto(
        X,
        y,
        list(X.columns),
        cfg,
        time=time,
        task="regression",
    )

    assert best_k == 3
    assert selected == list(X.columns)
    assert diag["k"].tolist() == [3]
    assert diag["selected"].tolist() == [True]


def test_select_k_auto_clamps_min_k_after_filtering_missing_path_features():
    X, y, time = _numeric_auto_k_data()
    X = X.iloc[:, :2]
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=5,
        max_k=5,
        val_frac=0.25,
    )

    best_k, selected, diag = select_k_auto(
        X,
        y,
        ["missing0", "x0", "missing1", "x1", "missing2"],
        cfg,
        time=time,
        task="regression",
    )

    assert best_k in {1, 2}
    assert selected == list(X.columns[:best_k])
    assert not diag.empty
    assert diag["k"].min() >= 1
    assert diag["k"].max() <= 2


def test_select_k_nested_clamps_min_k_to_feature_count():
    X, y, time = _numeric_auto_k_data()
    X = X.iloc[:, :3]
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        auto_k_mode="nested",
        min_k=10,
        max_k=10,
        val_frac=0.25,
    )

    def build_fold_path(train_idx, val_idx, max_k):
        return NestedAutoKFold(
            train_path=X.iloc[train_idx, :max_k].to_numpy(),
            val_path=X.iloc[val_idx, :max_k].to_numpy(),
            feature_path=list(X.columns[:max_k]),
        )

    result = select_k_nested(
        X,
        y,
        n_features=X.shape[1],
        config=cfg,
        build_fold_path=build_fold_path,
        time=time,
        task="regression",
    )

    assert result.selected_k == 3
    assert result.diagnostics["scores"]["k"].tolist() == [3]


def test_select_k_auto_rejects_elbow_method():
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        k_method="elbow",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    with pytest.raises(ValueError, match="select_k_auto.*k_method='evaluate'"):
        select_k_auto(X, y, list(X.columns), cfg, time=time)


def test_select_k_auto_rejects_duplicate_dataframe_labels():
    X, y, time = _numeric_auto_k_data()
    X = X.copy()
    X.columns = ["dup", "dup", *list(X.columns[2:])]
    cfg = AutoKConfig(
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    with pytest.raises(ValueError, match="unique DataFrame column labels"):
        select_k_auto(X, y, ["dup", "x2", "x3"], cfg, time=time)


def test_select_k_auto_evaluate_honors_sample_weight():
    rng = np.random.default_rng(321)
    n_train = 40
    n_val = 20
    n = n_train + n_val
    x0 = rng.normal(size=n)
    x1 = np.zeros(n)
    x1[:n_train] = np.tile([0.0, 1.0], n_train // 2)
    x1[n_train:] = 1.0

    y = np.empty(n)
    y[:n_train] = 2.0 * x0[:n_train] + 10.0 * x1[:n_train]
    y[n_train : n - 1] = 2.0 * x0[n_train : n - 1]
    y[n - 1] = 2.0 * x0[n - 1] + 10.0

    X = pd.DataFrame({"x0": x0, "x1": x1})
    time = np.arange(n)
    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="rmse",
        min_k=1,
        max_k=2,
        val_frac=n_val / n,
    )

    unweighted_k, _, unweighted_diag = select_k_auto(
        X,
        y,
        ["x0", "x1"],
        cfg,
        time=time,
        task="regression",
    )

    sample_weight = np.ones(n)
    sample_weight[-1] = 1000.0
    weighted_k, _, weighted_diag = select_k_auto(
        X,
        y,
        ["x0", "x1"],
        cfg,
        time=time,
        task="regression",
        sample_weight=sample_weight,
    )

    assert unweighted_k == 1
    assert weighted_k == 2
    assert unweighted_diag.loc[unweighted_diag["k"] == 1, "score"].iloc[0] < (
        unweighted_diag.loc[unweighted_diag["k"] == 2, "score"].iloc[0]
    )
    assert weighted_diag.loc[weighted_diag["k"] == 2, "score"].iloc[0] < (
        weighted_diag.loc[weighted_diag["k"] == 1, "score"].iloc[0]
    )
    assert "score_mean" in weighted_diag.columns
    np.testing.assert_allclose(weighted_diag["score"], weighted_diag["score_mean"])


def test_choose_k_from_score_curve_plateau_and_best_rules():
    diag = pd.DataFrame(
        {
            "k": [5, 10, 25, 50, 100, 150, 200, 250, 300],
            "score_mean": [8.15, 8.14, 8.13, 8.12, 8.13, 8.119, 8.118, 8.119, 8.122],
            "score_std": np.nan,
            "score_se": np.nan,
            "n_splits": 1,
            "n_finite": 1,
        }
    )

    best_k, best_diag = choose_k_from_score_curve(
        diag,
        AutoKConfig(selection_rule="best", min_k=5, max_k=300),
    )
    plateau_k, plateau_diag = choose_k_from_score_curve(
        diag,
        AutoKConfig(
            selection_rule="plateau",
            score_rel_tol=0.001,
            plateau_prefer="smallest",
            min_k=5,
            max_k=300,
        ),
    )
    center_k, _ = choose_k_from_score_curve(
        diag,
        AutoKConfig(
            selection_rule="plateau",
            score_rel_tol=0.001,
            plateau_prefer="center",
            min_k=5,
            max_k=300,
        ),
    )

    assert best_k == 200
    assert plateau_k == 150
    assert center_k == 250
    np.testing.assert_allclose(best_diag["score"], best_diag["score_mean"])
    assert plateau_diag.loc[plateau_diag["k"] == 150, "in_selected_plateau"].iloc[0]


def test_choose_k_from_score_curve_one_se_and_ties_choose_smaller_k():
    diag = pd.DataFrame(
        {
            "k": [1, 2, 3],
            "score_mean": [0.99, 0.90, 0.90],
            "score_std": [0.01, 0.20, 0.20],
            "score_se": [0.005, 0.10, 0.10],
            "n_splits": 4,
            "n_finite": 4,
        }
    )

    best_k, _ = choose_k_from_score_curve(
        diag,
        AutoKConfig(selection_rule="best", min_k=1, max_k=3),
    )
    one_se_k, one_se_diag = choose_k_from_score_curve(
        diag,
        AutoKConfig(selection_rule="one_se", min_k=1, max_k=3),
    )

    assert best_k == 2
    assert one_se_k == 1
    assert one_se_diag.loc[one_se_diag["k"] == 1, "within_tolerance"].iloc[0]


def test_choose_k_from_score_curve_bounds_and_sorts_diagnostics():
    diag = pd.DataFrame(
        {
            "k": [10, 1, 5, 3],
            "score_mean": [0.20, 0.01, 0.30, 0.10],
            "score_std": np.nan,
            "score_se": np.nan,
            "n_splits": 1,
            "n_finite": 1,
        }
    )

    selected_k, selected_diag = choose_k_from_score_curve(
        diag,
        AutoKConfig(selection_rule="best", min_k=3, max_k=8),
    )

    assert selected_k == 3
    assert selected_diag["k"].tolist() == [3, 5]
    assert selected_diag.loc[selected_diag["selected"], "k"].tolist() == [3]


def test_select_k_auto_one_se_single_holdout_warns_and_falls_back_to_best():
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        selection_rule="one_se",
        strategy="time_holdout",
        min_k=1,
        max_k=4,
        val_frac=0.25,
    )

    with pytest.warns(UserWarning, match="falling back"):
        best_k, selected, diag = select_k_auto(X, y, list(X.columns), cfg, time=time)

    assert best_k == int(diag.loc[diag["selected"], "k"].iloc[0])
    assert len(selected) == best_k
    assert bool(diag["one_se_unavailable"].iloc[0])


def test_select_k_penalized_objective_bic_and_custom_ties():
    objective = np.array([0.10, 0.18, 0.23, 0.231, 0.232])
    cfg = AutoKConfig(k_method="penalized_objective", min_k=1, max_k=5)

    best_k, diag = select_k_penalized_objective(
        objective,
        cfg,
        objective_scale="n_eff",
        n_samples=100,
    )
    assert best_k == 3
    assert diag.loc[diag["selected"], "k"].iloc[0] == 3
    assert diag["n_eff_source"].iloc[0] == "selector_weight_sum"

    tie_cfg = AutoKConfig(
        k_method="penalized_objective",
        objective_penalty="custom",
        objective_penalty_weight=0.0,
        min_k=1,
        max_k=3,
    )
    tie_k, _ = select_k_penalized_objective(
        np.array([1.0, 1.0, 1.0]),
        tie_cfg,
        objective_scale=1.0,
        n_samples=10,
    )
    assert tie_k == 1


def test_select_k_penalized_objective_all_invalid_warns_and_uses_effective_min():
    cfg = AutoKConfig(k_method="penalized_objective", min_k=2, max_k=5)

    with pytest.warns(UserWarning, match="non-finite"):
        selected_k, diag = select_k_penalized_objective(
            np.array([np.nan, -np.inf, np.nan]),
            cfg,
            objective_scale="n_eff",
            n_samples=25,
        )

    assert selected_k == 2
    assert bool(diag["all_penalized_scores_invalid"].iloc[0])
    assert diag["n_finite_penalized_score"].iloc[0] == 0
    assert diag.loc[diag["selected"], "k"].tolist() == [2]


def test_penalized_objective_ignores_irrelevant_plateau_tolerance_validation():
    cfg = AutoKConfig(
        k_method="penalized_objective",
        selection_rule="plateau",
        min_k=1,
        max_k=3,
    )

    selected_k, diag = select_k_penalized_objective(
        np.array([0.2, 0.21, 0.205]),
        cfg,
        objective_scale=1.0,
        n_samples=20,
    )

    assert selected_k in {1, 2, 3}
    assert not diag.empty


def test_public_auto_k_passes_sample_weight_to_prefix_evaluation(monkeypatch):
    X, y, time = _numeric_auto_k_data()
    sample_weight = np.linspace(1.0, 3.0, len(y))
    captured = {}

    def fake_select_k_auto(
        X,
        y,
        feature_path,
        config,
        *,
        sample_weight=None,
        **kwargs,
    ):
        captured["sample_weight"] = np.asarray(sample_weight)
        return 1, feature_path[:1], pd.DataFrame({"k": [1], "score": [0.0]})

    monkeypatch.setattr(auto_k_module, "select_k_auto", fake_select_k_auto)
    cfg = AutoKConfig(
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.25,
    )

    selected = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        time=time,
        auto_k_config=cfg,
        sample_weight=sample_weight,
        subsample=None,
        verbose=False,
    )

    assert len(selected) == 1
    assert "sample_weight" in captured
    assert captured["sample_weight"].shape == sample_weight.shape
    assert np.isclose(captured["sample_weight"].mean(), 1.0)
    np.testing.assert_allclose(
        captured["sample_weight"] / captured["sample_weight"][0],
        sample_weight / sample_weight[0],
    )


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"task": "regression"}),
        (select_jmi, {"task": "regression"}),
        (select_jmim, {"task": "regression"}),
        (select_cefsplus, {}),
    ],
)
def test_public_selectors_reject_nested_auto_k_mode(selector, kwargs):
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=6,
        val_frac=0.25,
    )

    with pytest.raises(NotImplementedError, match=NESTED_MODE_ERROR):
        selector(
            X,
            y,
            k="auto",
            time=time,
            auto_k_config=cfg,
            verbose=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    "config_kwargs, match",
    [
        ({"k_method": "bad"}, "k_method"),
        ({"strategy": "bad"}, "strategy"),
        ({"val_frac": 1.0}, "val_frac"),
        ({"val_frac": "0.2"}, "val_frac"),
        ({"min_k": 5, "max_k": 3}, "min_k"),
        ({"min_k": True}, "min_k"),
        ({"elbow_min_rel_gain": "0.02"}, "elbow_min_rel_gain"),
        ({"selection_rule": "bad"}, "selection_rule"),
        ({"selection_rule": "plateau"}, "score_abs_tol or score_rel_tol"),
        ({"selection_rule": "tolerance"}, "score_abs_tol or score_rel_tol"),
        ({"score_rel_tol": -0.1}, "score_rel_tol"),
        ({"plateau_prefer": "bad"}, "plateau_prefer"),
        ({"objective_penalty": "bad"}, "objective_penalty"),
        ({"objective_penalty": "custom"}, "objective_penalty_weight"),
        ({"objective_penalty_weight": 1.0}, "objective_penalty_weight"),
        ({"objective_n_eff": 1.0}, "objective_n_eff"),
        ({"binary_objective_mode": "bad"}, "binary_objective_mode"),
    ],
)
def test_public_selectors_validate_auto_k_config(config_kwargs, match):
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(**config_kwargs)

    with pytest.raises(ValueError, match=match):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            time=time,
            auto_k_config=cfg,
            verbose=False,
        )


def test_filter_spec_auto_k_handler_contract():
    assert set(filter_api.MRMR_CLASSIC_SPEC.auto_k_handlers) == {"evaluate"}
    assert set(filter_api.MRMR_GAUSSIAN_SPEC.auto_k_handlers) == {
        "auto",
        "evaluate",
        "elbow",
        "gaussian_cv",
        "stability",
        "xfit_objective",
    }

    for specs in (filter_api.JMI_CLASSIC_SPECS, filter_api.JMIM_CLASSIC_SPECS):
        assert set(specs) == {"r2", "binned", "ksg"}
        for spec in specs.values():
            assert set(spec.auto_k_handlers) == {"evaluate"}

    assert set(filter_api.JMI_GAUSSIAN_SPEC.auto_k_handlers) == {
        "auto",
        "evaluate",
        "elbow",
        "gaussian_cv",
        "stability",
        "xfit_objective",
    }
    assert set(filter_api.JMIM_GAUSSIAN_SPEC.auto_k_handlers) == {
        "auto",
        "evaluate",
        "elbow",
        "gaussian_cv",
        "stability",
        "xfit_objective",
    }
    assert set(filter_api.CEFSPLUS_SPEC.auto_k_handlers) == {
        "auto",
        "changepoint",
        "chi2_stop",
        "consensus",
        "evaluate",
        "elbow",
        "forward_stop",
        "gaussian_cv",
        "k_posterior",
        "knockoff_path",
        "penalized_objective",
        "perm_gap",
        "stability",
        "xfit_objective",
    }
    assert set(filter_api.CEFSPLUS_BINARY_SPEC.auto_k_handlers) == {
        "auto",
        "changepoint",
        "evaluate",
        "elbow",
        "k_posterior",
        "penalized_objective",
    }


def test_filter_api_selector_kwargs_match_public_signatures():
    common = filter_api._COMMON_REQUEST_LOCAL_NAMES

    def selector_kwargs(fn):
        return tuple(
            name
            for name in inspect.signature(fn).parameters
            if name not in common
        )

    assert selector_kwargs(filter_api.select_mrmr) == filter_api.MRMR_SELECTOR_KWARGS
    assert selector_kwargs(filter_api.select_jmi) == filter_api.JMI_SELECTOR_KWARGS
    assert selector_kwargs(filter_api.select_jmim) == filter_api.JMI_SELECTOR_KWARGS
    assert selector_kwargs(filter_api.select_cefsplus) == filter_api.CEFSPLUS_SELECTOR_KWARGS
    assert (
        selector_kwargs(filter_api.select_cefsplus_binary)
        == filter_api.CEFSPLUS_BINARY_SELECTOR_KWARGS
    )


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"task": "regression", "estimator": "classic"}),
        (select_jmi, {"task": "regression", "estimator": "r2"}),
        (select_jmim, {"task": "regression", "estimator": "r2"}),
    ],
)
def test_classic_public_auto_k_rejects_elbow_method(selector, kwargs, monkeypatch):
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(
        k_method="elbow",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    def fail_prepare(*args, **kwargs):
        raise AssertionError("classic payload prep should not be called")

    monkeypatch.setattr(filter_payloads, "_prepare_xy_classic", fail_prepare)

    with pytest.raises(ValueError, match="does not support k_method='elbow'"):
        selector(
            X,
            y,
            k="auto",
            time=time,
            auto_k_config=cfg,
            verbose=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("strategy", "match"),
    [
        ("time_holdout", "requires time parameter"),
        ("group_cv", "requires groups parameter"),
    ],
)
def test_classic_evaluate_auto_k_rejects_missing_split_context_before_prep(
    strategy,
    match,
    monkeypatch,
):
    X, y, _ = _numeric_auto_k_data()
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    def fail_prepare(*args, **kwargs):
        raise AssertionError("classic payload prep should not be called")

    monkeypatch.setattr(filter_payloads, "_prepare_xy_classic", fail_prepare)

    with pytest.raises(ValueError, match=match):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="classic",
            auto_k_config=cfg,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("strategy", "match"),
    [
        ("time_holdout", "requires time parameter"),
        ("group_cv", "requires groups parameter"),
    ],
)
def test_gaussian_evaluate_auto_k_rejects_missing_split_context_before_cache(
    strategy,
    match,
    monkeypatch,
):
    X, y, _ = _numeric_auto_k_data()
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    def fail_cache(*args, **kwargs):
        raise AssertionError("gaussian cache construction should not be called")

    monkeypatch.setattr(filter_payloads, "_cache_for_gaussian", fail_cache)

    with pytest.raises(ValueError, match=match):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("strategy", "match"),
    [
        ("time_holdout", "requires time parameter"),
        ("group_cv", "requires groups parameter"),
    ],
)
def test_binary_logloss_evaluate_auto_k_rejects_missing_split_context_before_path(
    strategy,
    match,
    monkeypatch,
):
    X, y, _ = _binary_auto_k_data()
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    def fail_path(*args, **kwargs):
        raise AssertionError("binary logloss path construction should not be called")

    monkeypatch.setattr(filter_payloads, "build_binary_logloss_path", fail_path)

    with pytest.raises(ValueError, match=match):
        select_cefsplus_binary(
            X,
            y,
            k="auto",
            loss="logloss",
            auto_k_config=cfg,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("strategy", "match"),
    [
        ("time_holdout", "requires time parameter"),
        ("group_cv", "requires groups parameter"),
    ],
)
def test_binary_brier_evaluate_auto_k_rejects_missing_split_context_before_problem(
    strategy,
    match,
    monkeypatch,
):
    X, y, _ = _binary_auto_k_data()
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy=strategy,
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    def fail_prepare_problem(*args, **kwargs):
        raise AssertionError("binary problem preparation should not be called")

    monkeypatch.setattr(filter_api, "prepare_binary_problem", fail_prepare_problem)

    with pytest.raises(ValueError, match=match):
        select_cefsplus_binary(
            X,
            y,
            k="auto",
            loss="brier",
            auto_k_config=cfg,
            verbose=False,
        )


@pytest.mark.parametrize("route", ["classic", "gaussian", "binary_logloss", "binary_brier"])
def test_function_style_evaluate_auto_k_rejects_duplicate_labels_before_work(
    route,
    monkeypatch,
):
    X, y_reg, time = _numeric_auto_k_data()
    X = X.copy()
    X.columns = ["dup", "dup", *list(X.columns[2:])]
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )

    def fail_work(*args, **kwargs):
        raise AssertionError("selector work should not be called")

    with pytest.raises(ValueError, match="unique DataFrame column labels"):
        if route == "classic":
            monkeypatch.setattr(filter_payloads, "_prepare_xy_classic", fail_work)
            select_mrmr(
                X,
                y_reg,
                k="auto",
                task="regression",
                estimator="classic",
                time=time,
                auto_k_config=cfg,
                verbose=False,
            )
        elif route == "gaussian":
            monkeypatch.setattr(filter_payloads, "_cache_for_gaussian", fail_work)
            select_mrmr(
                X,
                y_reg,
                k="auto",
                task="regression",
                estimator="gaussian",
                time=time,
                auto_k_config=cfg,
                verbose=False,
            )
        elif route == "binary_logloss":
            monkeypatch.setattr(filter_payloads, "build_binary_logloss_path", fail_work)
            select_cefsplus_binary(
                X,
                (y_reg > np.median(y_reg)).astype(int),
                k="auto",
                loss="logloss",
                time=time,
                auto_k_config=cfg,
                verbose=False,
            )
        else:
            monkeypatch.setattr(filter_api, "prepare_binary_problem", fail_work)
            select_cefsplus_binary(
                X,
                (y_reg > np.median(y_reg)).astype(int),
                k="auto",
                loss="brier",
                time=time,
                auto_k_config=cfg,
                verbose=False,
            )


def test_public_auto_k_rejects_penalized_objective_for_non_cefsplus_routes():
    X, y, time = _numeric_auto_k_data()
    cfg = AutoKConfig(k_method="penalized_objective", min_k=1, max_k=3)

    with pytest.raises(ValueError, match="does not support k_method='penalized_objective'"):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="classic",
            time=time,
            auto_k_config=cfg,
            verbose=False,
        )
    with pytest.raises(ValueError, match="does not support k_method='penalized_objective'"):
        select_jmi(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            verbose=False,
        )


def test_gaussian_non_cefsplus_rejects_penalized_objective_before_cache(monkeypatch):
    X, y, _ = _numeric_auto_k_data()
    cfg = AutoKConfig(k_method="penalized_objective", min_k=1, max_k=3)

    def fail_build_cache(*args, **kwargs):
        raise AssertionError("build_cache should not be called")

    monkeypatch.setattr(filter_payloads, "build_cache", fail_build_cache)

    with pytest.raises(ValueError, match="does not support k_method='penalized_objective'"):
        select_jmim(
            X,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            auto_k_config=cfg,
            verbose=False,
        )


def test_classic_rejects_penalized_objective_before_supervised_encoding_error():
    X, y, _ = _numeric_auto_k_data()
    X = X.assign(cat=np.where(np.arange(len(X)) % 2 == 0, "a", "b"))
    cfg = AutoKConfig(k_method="penalized_objective", min_k=1, max_k=3)

    with pytest.raises(ValueError, match="does not support k_method='penalized_objective'"):
        select_mrmr(
            X,
            y,
            k="auto",
            task="regression",
            estimator="classic",
            cat_features=["cat"],
            cat_encoding="loo",
            auto_k_config=cfg,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("selector", "kwargs"),
    [
        (select_mrmr, {"task": "regression", "estimator": "not-real"}),
        (select_jmi, {"task": "regression", "estimator": "not-real"}),
        (select_jmim, {"task": "regression", "estimator": "not-real"}),
    ],
)
def test_public_filter_selectors_reject_invalid_estimators(selector, kwargs):
    X, y, _ = _numeric_auto_k_data()
    with pytest.raises(ValueError, match="estimator"):
        selector(X, y, k=2, verbose=False, **kwargs)


def test_gaussian_auto_k_elbow_still_works_without_split_context():
    X, y, _ = _numeric_auto_k_data()
    cfg = AutoKConfig(k_method="elbow", min_k=1, max_k=4)

    cefs = select_cefsplus(X, y, k="auto", auto_k_config=cfg, verbose=False)
    gaussian_mrmr = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        auto_k_config=cfg,
        verbose=False,
    )

    assert 1 <= len(cefs) <= 4
    assert 1 <= len(gaussian_mrmr) <= 4


def test_select_k_auto_target_encoding_not_leaky():
    pytest.importorskip("category_encoders")

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"id": [f"id_{i}" for i in range(n)]})
    y = rng.normal(size=n)
    feature_path = ["id"]

    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="rmse",
        val_frac=0.25,
        min_k=1,
        max_k=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, diag = select_k_auto(
            X=X,
            y=y,
            feature_path=feature_path,
            config=cfg,
            time=np.arange(n),
            task="regression",
            cat_features=["id"],
            cat_encoding="target",
        )

    assert float(diag["score"].iloc[0]) > 0.5


def test_select_k_auto_cat_encoding_requires_category_encoders():
    if importlib.util.find_spec("category_encoders") is not None:
        pytest.skip("category_encoders installed; skipping dependency error test")

    X = pd.DataFrame({"id": ["a", "b", "c", "d"]})
    y = np.array([0.1, 0.2, 0.3, 0.4])
    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="rmse",
        val_frac=0.5,
        min_k=1,
        max_k=1,
    )

    with pytest.raises(ImportError):
        select_k_auto(
            X=X,
            y=y,
            feature_path=["id"],
            config=cfg,
            time=np.arange(len(y)),
            task="regression",
            cat_features=["id"],
            cat_encoding="target",
        )


def test_select_k_auto_loo_logit_uses_builtin_binary_encoder():
    rng = np.random.default_rng(33)
    n = 120
    team = np.where(np.arange(n) % 3 == 0, "a", "b")
    y = (team == "a").astype(int)
    X = pd.DataFrame({"team": team, "noise": rng.normal(size=n)})
    sample_weight = np.ones(n)
    sample_weight[::5] = 2.0
    cfg = AutoKConfig(
        strategy="time_holdout",
        metric="logloss",
        val_frac=0.25,
        min_k=1,
        max_k=2,
    )

    best_k, selected, diag = select_k_auto(
        X=X,
        y=y,
        feature_path=["team", "noise"],
        config=cfg,
        time=np.arange(n),
        task="classification",
        cat_features=["team"],
        cat_encoding="loo_logit",
        sample_weight=sample_weight,
        loo_smoothing=7.0,
        loo_clip_min=1e-3,
        loo_clip_max=1.0 - 1e-3,
    )

    assert best_k in {1, 2}
    assert selected[0] == "team"
    assert np.isfinite(diag["score"]).all()


def test_gaussian_auto_k_rejects_cache_built_for_different_row_count():
    X1, y1, _ = _numeric_auto_k_data()
    cache = build_cache(X1, subsample=40, random_state=0)
    X2 = X1.iloc[:70].copy()
    y2 = y1[:70]
    cfg = AutoKConfig(k_method="evaluate", strategy="time_holdout", min_k=1, max_k=3)

    with pytest.raises(ValueError, match="cache was built with"):
        select_mrmr(
            X2,
            y2,
            k="auto",
            task="regression",
            estimator="gaussian",
            cache=cache,
            auto_k_config=cfg,
            time=np.arange(len(y2)),
            verbose=False,
        )


def test_gaussian_function_selector_maps_unnamed_cache_indices_to_feature_names():
    X, y, _ = _numeric_auto_k_data()
    cache = build_cache(X.to_numpy(), subsample=None)

    selected = select_cefsplus(X, y, k=3, cache=cache, verbose=False)
    result = select_cefsplus(X, y, k=3, cache=cache, return_result=True, verbose=False)

    assert all(feature.startswith("x") for feature in selected)
    assert result.selected_features == selected
    assert all(isinstance(index, int) for index in result.selected_indices)


def test_gaussian_auto_k_with_unnamed_cache_rejects_named_dataframe_evaluation():
    X, y, _ = _numeric_auto_k_data()
    cache = build_cache(X.to_numpy(), subsample=None)
    X_reordered = X[list(reversed(X.columns))].copy()
    cfg = AutoKConfig(k_method="evaluate", strategy="time_holdout", min_k=1, max_k=3)

    with pytest.raises(ValueError, match="cache built from unnamed/positional features"):
        select_mrmr(
            X_reordered,
            y,
            k="auto",
            task="regression",
            estimator="gaussian",
            cache=cache,
            auto_k_config=cfg,
            time=np.arange(len(y)),
            verbose=False,
        )


def test_gaussian_auto_k_with_unnamed_cache_accepts_positional_ndarray_evaluation():
    X, y, _ = _numeric_auto_k_data()
    X_arr = X.to_numpy()
    cache = build_cache(X_arr, subsample=None)
    cfg = AutoKConfig(k_method="evaluate", strategy="time_holdout", min_k=1, max_k=3)

    selected = select_mrmr(
        X_arr,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        cache=cache,
        auto_k_config=cfg,
        time=np.arange(len(y)),
        verbose=False,
    )

    assert selected
    assert all(feature.startswith("x") for feature in selected)


def test_gaussian_fixed_k_unnamed_cache_reordered_dataframe_stays_positional():
    rng = np.random.default_rng(0)
    X_cache = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    y = X_cache["d"].to_numpy() + 0.1 * rng.normal(size=len(X_cache))
    cache = build_cache(X_cache.to_numpy(), subsample=None)
    X_eval = X_cache[["d", "c", "b", "a"]].copy()

    result = select_cefsplus(
        X_eval,
        y,
        k=2,
        cache=cache,
        return_result=True,
        verbose=False,
    )

    assert all(feature.startswith("x") for feature in result.selected_features)
    assert result.selected_indices is not None
    assert result.selected_features == [f"x{i}" for i in result.selected_indices]


def test_gaussian_auto_k_return_result_indices_follow_input_column_order():
    rng = np.random.default_rng(0)
    X_cache = pd.DataFrame(rng.normal(size=(80, 4)), columns=["a", "b", "c", "d"])
    y = X_cache["d"].to_numpy() + 0.1 * rng.normal(size=len(X_cache))
    cache = build_cache(X_cache)
    X_eval = X_cache[["d", "c", "b", "a"]].copy()
    cfg = AutoKConfig(k_method="evaluate", strategy="time_holdout", min_k=1, max_k=3)

    result = select_cefsplus(
        X_eval,
        y,
        k="auto",
        auto_k_config=cfg,
        cache=cache,
        time=np.arange(len(X_eval)),
        return_result=True,
        verbose=False,
    )

    assert result.selected_indices is not None
    assert [X_eval.columns[i] for i in result.selected_indices] == result.selected_features


def test_gaussian_auto_k_named_x_columns_are_not_treated_as_synthetic():
    rng = np.random.default_rng(1)
    X_cache = pd.DataFrame(rng.normal(size=(80, 4)), columns=["x0", "x1", "x2", "x3"])
    y = X_cache["x3"].to_numpy() + 0.1 * rng.normal(size=len(X_cache))
    cache = build_cache(X_cache)
    X_eval = X_cache[["x3", "x2", "x1", "x0"]].copy()
    cfg = AutoKConfig(k_method="evaluate", strategy="time_holdout", min_k=1, max_k=3)

    result = select_cefsplus(
        X_eval,
        y,
        k="auto",
        auto_k_config=cfg,
        cache=cache,
        time=np.arange(len(X_eval)),
        return_result=True,
        verbose=False,
    )

    assert not cache.feature_names_are_synthetic
    assert result.selected_indices is not None
    assert [X_eval.columns[i] for i in result.selected_indices] == result.selected_features
