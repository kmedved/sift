"""Progress-callback contracts for long-running public selectors."""

from __future__ import annotations

import inspect
import logging
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

import sift
from sift import StabilitySelector, stability_regression
from sift.stability import stability_select
from sift.boruta import BorutaSelector, select_boruta
from sift import catboost as catboost_module
from sift.estimators.copula import build_cache
from sift.selection.auto_k import AutoKConfig
from sift.selection.cefsplus import (
    _cefsplus_loop_core,
    _cefsplus_loop_with_callback,
)
from sift.selection.loops import (
    FLOOR,
    _mrmr_loop_serial_with_callback,
    _mrmr_serial_callback_step,
    _standardize_columns_weighted,
    mrmr_loop_incremental,
)


def _regression_frame(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(2.0 * X["f0"] - X["f1"] + rng.normal(scale=0.2, size=len(X)))
    return X, y


def _greedy_bridge_fixture(
    seed: int, *, near_collinear: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic data shared by direct CEFS+ and mRMR bridge tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, 8))
    if near_collinear:
        latent = rng.normal(size=len(X))
        perturbation = rng.normal(size=len(X))
        X[:, 0] = latent + 1e-8 * perturbation
        X[:, 1] = latent - 1e-8 * perturbation
    y = 0.9 * X[:, 0] - 0.35 * X[:, 2] + rng.normal(scale=0.4, size=len(X))
    weights = rng.uniform(0.5, 1.5, size=len(X))
    return X, y, weights


def _mrmr_objective_path(
    Z: np.ndarray,
    relevance: np.ndarray,
    selected: np.ndarray,
    use_quotient: bool,
    weights: np.ndarray,
) -> np.ndarray:
    """Independently reconstruct the selected mRMR score at each step."""
    if len(selected) == 0:
        return np.empty(0, dtype=np.float64)
    red_sum = np.zeros(Z.shape[1], dtype=np.float64)
    objectives = [float(relevance[int(selected[0])])]
    weight_sum = float(weights.sum())
    for t in range(1, len(selected)):
        last = int(selected[t - 1])
        for j in range(Z.shape[1]):
            corr = 0.0
            for i in range(Z.shape[0]):
                corr += weights[i] * Z[i, j] * Z[i, last]
            red_sum[j] += abs(corr / weight_sum)
        chosen = int(selected[t])
        mean_red = red_sum[chosen] / t
        if use_quotient:
            objectives.append(float(relevance[chosen] / max(mean_red, FLOOR)))
        else:
            objectives.append(float(relevance[chosen] - mean_red))
    return np.asarray(objectives)


@pytest.mark.parametrize(
    ("seed", "near_collinear"),
    [(0, False), (7, False), (29, False), (101, True)],
)
def test_cefsplus_callback_bridge_matches_compiled_core(seed, near_collinear):
    X, y, _weights = _greedy_bridge_fixture(
        seed, near_collinear=near_collinear
    )
    joint_corr = np.corrcoef(np.column_stack([X, y]), rowvar=False)
    R = np.asarray(joint_corr[:-1, :-1], dtype=np.float64)
    r = np.asarray(joint_corr[:-1, -1], dtype=np.float64)
    tie_break_rel = -0.5 * np.log1p(-np.minimum(r * r, 1.0 - 1e-15))
    if near_collinear:
        tie_break_rel[1] = tie_break_rel[0]

    baseline_selected, baseline_objective = _cefsplus_loop_core(
        R, r, 6, tie_break_rel, True
    )
    events = []
    observed_selected, observed_objective = _cefsplus_loop_with_callback(
        R,
        r,
        6,
        tie_break_rel,
        lambda step, total, info: events.append((step, total, info)),
        want_objective=True,
    )

    np.testing.assert_array_equal(observed_selected, baseline_selected)
    np.testing.assert_array_equal(observed_objective, baseline_objective)
    assert [step for step, _, _ in events] == list(
        range(1, len(observed_selected) + 1)
    )


@pytest.mark.parametrize("use_quotient", [False, True])
@pytest.mark.parametrize(
    ("seed", "near_collinear"),
    [(0, False), (7, False), (29, False), (101, True)],
)
def test_serial_mrmr_callback_bridge_matches_compiled_core(
    seed, near_collinear, use_quotient
):
    X, y, weights = _greedy_bridge_fixture(seed, near_collinear=near_collinear)
    Z = _standardize_columns_weighted(X, weights)
    y_standardized = _standardize_columns_weighted(y[:, None], weights)[:, 0]
    relevance = np.abs((weights[:, None] * Z * y_standardized[:, None]).sum(axis=0))
    relevance /= weights.sum()
    if near_collinear:
        relevance[1] = relevance[0]

    baseline = mrmr_loop_incremental(Z, relevance, 6, use_quotient, weights)
    events = []
    observed = _mrmr_loop_serial_with_callback(
        Z,
        relevance,
        6,
        use_quotient,
        weights,
        lambda step, total, info: events.append((step, total, info)),
    )

    np.testing.assert_array_equal(observed, baseline)
    assert [step for step, _, _ in events] == list(range(1, len(observed) + 1))

    # Exercise the one-step kernel directly so score parity is protected in
    # addition to the public driver's selected-index parity.
    state_selected = np.empty(6, dtype=np.int64)
    state_selected[0] = int(baseline[0])
    state_mask = np.zeros(Z.shape[1], dtype=np.bool_)
    state_mask[state_selected[0]] = True
    state_red_sum = np.zeros(Z.shape[1], dtype=np.float64)
    bridge_objective = [float(relevance[state_selected[0]])]
    for t in range(1, 6):
        best_idx, best_score = _mrmr_serial_callback_step(
            Z,
            relevance,
            use_quotient,
            weights,
            t,
            int(state_selected[t - 1]),
            state_mask,
            state_red_sum,
        )
        state_selected[t] = best_idx
        state_mask[best_idx] = True
        bridge_objective.append(float(best_score))

    np.testing.assert_array_equal(state_selected, baseline)
    np.testing.assert_allclose(
        bridge_objective,
        _mrmr_objective_path(Z, relevance, baseline, use_quotient, weights),
        rtol=0.0,
        atol=1e-14,
    )


def _assert_fresh_events(events, *, total: int, stage: str, keys: set[str]) -> None:
    assert [step for step, _, _ in events] == list(range(1, len(events) + 1))
    assert {reported_total for _, reported_total, _ in events} == {total}
    assert len({id(info) for _, _, info in events}) == len(events)
    assert all(info["stage"] == stage for _, _, info in events)
    assert all(keys <= set(info) for _, _, info in events)

    if len(events) > 1:
        events[0][2]["stage"] = "mutated-by-caller"
        assert events[1][2]["stage"] == stage


def test_stability_callback_counts_fresh_info_and_preserves_result():
    X, y = _regression_frame()
    common = dict(
        n_bootstrap=4,
        threshold=0.25,
        alpha=0.02,
        random_state=4,
        n_jobs=1,
        verbose=False,
    )
    baseline = StabilitySelector(**common).fit(X, y)
    events = []
    observed = StabilitySelector(
        **common,
        callback=lambda step, total, info: events.append((step, total, info)),
    ).fit(X, y)

    np.testing.assert_array_equal(observed.selected_features_, baseline.selected_features_)
    np.testing.assert_array_equal(
        observed.selection_frequencies_, baseline.selection_frequencies_
    )
    assert len(events) == common["n_bootstrap"]
    _assert_fresh_events(
        events,
        total=common["n_bootstrap"],
        stage="bootstrap",
        keys={"task", "selected_features"},
    )


def test_stability_tune_threshold_keeps_fold_local_callbacks_silent():
    X, y = _regression_frame(2)
    events = []
    selector = StabilitySelector(
        n_bootstrap=2,
        threshold=0.25,
        alpha=0.02,
        random_state=2,
        n_jobs=1,
        verbose=False,
        callback=lambda step, total, info: events.append((step, total, info)),
    ).fit(X, y)

    assert [step for step, _, _ in events] == [1, 2]
    events.clear()

    selector.tune_threshold(X, y, thresholds=(0.5,), cv=2)

    assert events == []


def test_stability_wrapper_forwards_callback_and_callback_errors_propagate():
    X, y = _regression_frame(1)
    events = []
    stability_regression(
        X,
        y,
        k=3,
        n_bootstrap=3,
        threshold=0.0,
        alpha=0.02,
        random_state=1,
        n_jobs=1,
        verbose=False,
        callback=lambda step, total, info: events.append((step, total, info)),
    )
    assert [step for step, _, _ in events] == [1, 2, 3]

    selector = StabilitySelector(
        n_bootstrap=3,
        alpha=0.02,
        random_state=1,
        n_jobs=1,
        verbose=False,
        callback=lambda *_args: (_ for _ in ()).throw(RuntimeError("stop stability")),
    )
    with pytest.raises(RuntimeError, match="stop stability"):
        selector.fit(X, y)
    assert not hasattr(selector, "selection_frequencies_")


def _boruta_kwargs() -> dict:
    return {
        "estimator": RandomForestRegressor(
            n_estimators=12,
            max_depth=3,
            n_jobs=1,
            random_state=7,
        ),
        "n_estimators": 12,
        "max_iter": 3,
        "early_stop_rounds": 10,
        "random_state": 7,
        "verbose": False,
    }


def test_boruta_callback_counts_fresh_info_and_preserves_result():
    X, y = _regression_frame(2)
    common = _boruta_kwargs()
    baseline = BorutaSelector(**common).fit(X, y)
    events = []
    observed = BorutaSelector(
        **common,
        callback=lambda step, total, info: events.append((step, total, info)),
    ).fit(X, y)

    np.testing.assert_array_equal(observed.status_, baseline.status_)
    np.testing.assert_array_equal(observed.hits_, baseline.hits_)
    assert len(events) == observed.n_iter_ == common["max_iter"]
    _assert_fresh_events(
        events,
        total=common["max_iter"],
        stage="iteration",
        keys={
            "accepted",
            "rejected",
            "tentative",
            "shadow_threshold",
            "n_estimators",
        },
    )


def test_boruta_wrapper_forwards_callback_and_callback_errors_propagate():
    X, y = _regression_frame(3)
    events = []
    result = select_boruta(
        X,
        y,
        **_boruta_kwargs(),
        callback=lambda step, total, info: events.append((step, total, info)),
    )
    assert isinstance(result, list)
    assert [step for step, _, _ in events] == [1, 2, 3]

    selector = BorutaSelector(
        **_boruta_kwargs(),
        callback=lambda *_args: (_ for _ in ()).throw(RuntimeError("stop boruta")),
    )
    with pytest.raises(RuntimeError, match="stop boruta"):
        selector.fit(X, y)
    assert not hasattr(selector, "status_")


def test_estimator_callbacks_follow_sklearn_clone_contract():
    callback = lambda _step, _total, _info: None
    estimators = [
        StabilitySelector(callback=callback),
        BorutaSelector(callback=callback),
        sift.MRMRSelector(callback=callback),
        sift.JMISelector(callback=callback),
        sift.JMIMSelector(callback=callback),
        sift.CEFSPlusSelector(callback=callback),
        sift.CEFSPlusBinarySelector(callback=callback),
    ]

    for estimator in estimators:
        assert clone(estimator).callback is callback


@pytest.mark.parametrize(
    ("method", "expected_selector"),
    [
        ("cefsplus", "cefsplus"),
        ("jmi", "jmi"),
        ("jmim", "jmim"),
        ("mrmr_quot", "mrmr_quot"),
        ("mrmr_diff", "mrmr_diff"),
    ],
)
def test_select_cached_path_callback_preserves_every_gaussian_method(
    method, expected_selector
):
    X, y = _regression_frame(8)
    cache = build_cache(X, subsample=None)
    baseline = sift.select_cached(
        cache,
        y,
        4,
        method=method,
        return_indices=True,
        return_objective=True,
        warn_noise_floor=False,
    )
    events = []
    observed = sift.select_cached(
        cache,
        y,
        4,
        method=method,
        return_indices=True,
        return_objective=True,
        warn_noise_floor=False,
        callback=lambda step, total, info: events.append((step, total, info)),
    )

    assert observed[:2] == baseline[:2]
    np.testing.assert_allclose(observed[2], baseline[2], rtol=0.0, atol=1e-14)
    _assert_fresh_events(
        events,
        total=4,
        stage="path",
        keys={"selector"},
    )
    assert all(info["selector"] == expected_selector for _, _, info in events)


@pytest.mark.parametrize(
    ("selector", "kwargs", "expected_name"),
    [
        (
            sift.select_mrmr,
            {"task": "regression", "estimator": "classic", "mrmr_backend": "blas"},
            "mrmr",
        ),
        (
            sift.select_mrmr,
            {"task": "regression", "estimator": "classic", "mrmr_backend": "serial"},
            "mrmr",
        ),
        (
            sift.select_jmi,
            {"task": "regression", "estimator": "r2"},
            "jmi",
        ),
        (
            sift.select_jmim,
            {"task": "regression", "estimator": "r2"},
            "jmim",
        ),
        (sift.select_cefsplus, {}, "cefsplus"),
    ],
)
def test_filter_dispatcher_fixed_path_callback_preserves_result(
    selector, kwargs, expected_name
):
    X, y = _regression_frame(9)
    common = dict(k=3, subsample=None, verbose=False, **kwargs)
    baseline = selector(X, y, **common)
    events = []
    observed = selector(
        X,
        y,
        **common,
        callback=lambda step, total, info: events.append((step, total, info)),
    )

    assert observed == baseline
    _assert_fresh_events(
        events,
        total=3,
        stage="path",
        keys={"selector"},
    )
    assert all(info["selector"] == expected_name for _, _, info in events)


def test_callback_supplements_verbose_logging(caplog):
    X, y = _regression_frame(12)
    events = []
    caplog.set_level(logging.INFO)

    sift.select_mrmr(
        X,
        y,
        k=2,
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        subsample=None,
        verbose=True,
        callback=lambda step, total, info: events.append((step, total, info)),
    )

    assert [step for step, _, _ in events] == [1, 2]
    assert any(
        record.name == "sift" and "mRMR classic: selecting 2 features" in record.getMessage()
        for record in caplog.records
    )


def test_filter_dispatcher_auto_and_binary_paths_report_full_built_path():
    X, y = _regression_frame(10)
    config = AutoKConfig(k_method="elbow", min_k=1, max_k=4)
    events = []
    baseline = sift.select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=config,
        subsample=None,
        verbose=False,
    )
    observed = sift.select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=config,
        subsample=None,
        verbose=False,
        callback=lambda step, total, info: events.append((step, total, info)),
    )
    assert observed == baseline
    assert [step for step, _, _ in events] == [1, 2, 3, 4]
    assert {total for _, total, _ in events} == {4}

    labels = (y > y.median()).astype(int)
    binary_events = []
    binary_baseline = sift.select_cefsplus_binary(
        X, labels, k=3, subsample=None, verbose=False
    )
    binary_observed = sift.select_cefsplus_binary(
        X,
        labels,
        k=3,
        subsample=None,
        verbose=False,
        callback=lambda step, total, info: binary_events.append((step, total, info)),
    )
    assert binary_observed == binary_baseline
    assert [step for step, _, _ in binary_events] == [1, 2, 3]
    assert all(info["selector"] == "cefsplus_binary" for _, _, info in binary_events)


@pytest.mark.parametrize("callback_source", ["constructor", "fit"])
def test_nested_auto_selector_reports_only_the_final_refit(callback_source):
    X, y = _regression_frame(19)
    config = AutoKConfig(
        auto_k_mode="nested",
        strategy="time_holdout",
        min_k=1,
        max_k=3,
        val_frac=0.25,
    )
    events = []
    callback = lambda step, total, info: events.append((step, total, info))
    selector_kwargs = dict(
        k="auto",
        task="regression",
        estimator="classic",
        mrmr_backend="serial",
        auto_k_config=config,
        verbose=False,
    )
    fit_kwargs = {"time": np.arange(len(X))}
    if callback_source == "constructor":
        selector_kwargs["callback"] = callback
    else:
        fit_kwargs["callback"] = callback

    selector = sift.MRMRSelector(**selector_kwargs).fit(X, y, **fit_kwargs)

    _assert_fresh_events(
        events,
        total=selector.k_,
        stage="path",
        keys={"selector", "backend"},
    )
    assert len(events) == selector.k_


def test_filter_callback_exception_propagates_and_selector_state_is_cleared():
    X, y = _regression_frame(11)

    def fail(*_args):
        raise RuntimeError("stop path")

    with pytest.raises(RuntimeError, match="stop path"):
        sift.select_cached(build_cache(X, subsample=None), y, 3, callback=fail)

    selector = sift.CEFSPlusSelector(
        k=3,
        subsample=None,
        verbose=False,
        callback=fail,
    )
    with pytest.raises(RuntimeError, match="stop path"):
        selector.fit(X, y)
    assert not hasattr(selector, "selected_features_")


def test_callback_parameters_are_appended_without_rebinding_legacy_positionals():
    callback_last = [
        sift.StabilitySelector,
        sift.BorutaSelector,
        sift.select_boruta,
        sift.select_boruta_shap,
        sift.catboost_select,
        sift.select_cached,
        sift.select_mrmr,
        sift.select_jmi,
        sift.select_jmim,
        sift.select_cefsplus,
        sift.select_cefsplus_binary,
        sift.MRMRSelector,
        sift.JMISelector,
        sift.JMIMSelector,
        sift.CEFSPlusSelector,
        sift.CEFSPlusBinarySelector,
    ]
    additive_suffixes = {
        sift.StabilitySelector: ["penalty", "output_order", "store_proxies"],
        sift.BorutaSelector: ["output_order"],
        sift.catboost_select: ["groups", "time", "sample_weight"],
        sift.select_cached: ["return_result", "store_proxies", "include", "exclude", "candidates", "feature_blocks"],
        sift.MRMRSelector: ["output_order"],
        sift.JMISelector: ["output_order"],
        sift.JMIMSelector: ["output_order"],
        sift.CEFSPlusSelector: ["output_order"],
        sift.CEFSPlusBinarySelector: ["output_order"],
    }
    for callable_ in callback_last:
        parameters = inspect.signature(callable_).parameters
        expected_suffix = ["callback", *additive_suffixes.get(callable_, [])]
        assert list(parameters)[-len(expected_suffix):] == expected_suffix
        assert parameters["callback"].default is None

    callback_before_kwargs = [
        stability_select,
        sift.stability_regression,
        sift.stability_classif,
        catboost_module.catboost_regression,
        catboost_module.catboost_classif,
    ]
    for callable_ in callback_before_kwargs:
        parameters = list(inspect.signature(callable_).parameters.values())
        assert parameters[-2].name == "callback"
        assert parameters[-2].default is None
        assert parameters[-1].kind is inspect.Parameter.VAR_KEYWORD

    callback_positions = {
        sift.StabilitySelector: 18,
        sift.catboost_select: 36,
        sift.select_cached: 9,
    }
    for callable_, callback_position in callback_positions.items():
        signature = inspect.signature(callable_)
        names = list(signature.parameters)
        assert names.index("callback") == callback_position
        bound = signature.bind_partial(*range(callback_position))
        assert "callback" not in bound.arguments


def _run_fake_catboost_splits(monkeypatch, callback):
    X, y = _regression_frame(4)
    splits = [
        (np.arange(0, 50), np.arange(50, 65)),
        (np.arange(15, 65), np.arange(65, 80)),
    ]

    def fake_forward(*_args, **kwargs):
        counts = kwargs.get("feature_counts")
        if counts is None:
            counts = _args[5]
        selected = ["f0", "f1", "f2"]
        return {int(k): float(k) for k in counts}, selected

    monkeypatch.setattr(
        catboost_module,
        "_forward_select_single_split",
        fake_forward,
    )
    return catboost_module._run_catboost_split_evaluation(
        X_work=X,
        y=y,
        sample_weights=None,
        splits=splits,
        all_features=list(X.columns),
        counts=[3, 2],
        task="regression",
        model_params={},
        cat_features_final=[],
        text_feat=[],
        prefilter_k=None,
        prefilter_method="catboost",
        random_state=0,
        n_jobs=1,
        algorithm="forward",
        resolved_metric="RMSE",
        resolved_hib=False,
        train_early_stopping_rounds=3,
        steps=2,
        k_req=None,
        verbose=False,
        callback=callback,
    )


def test_catboost_split_callback_counts_fresh_info_and_preserves_result(monkeypatch):
    baseline = _run_fake_catboost_splits(monkeypatch, None)
    events = []
    observed = _run_fake_catboost_splits(
        monkeypatch,
        lambda step, total, info: events.append((step, total, info)),
    )

    assert observed == baseline
    assert len(events) == 2
    _assert_fresh_events(
        events,
        total=2,
        stage="split",
        keys={
            "train_rows",
            "validation_rows",
            "candidate_features",
            "evaluated_counts",
            "best_k",
            "best_score",
        },
    )


def test_catboost_split_callback_errors_propagate(monkeypatch):
    def fail(*_args):
        raise RuntimeError("stop catboost")

    with pytest.raises(RuntimeError, match="stop catboost"):
        _run_fake_catboost_splits(monkeypatch, fail)


@pytest.mark.catboost
def test_catboost_public_callback_preserves_result_when_installed():
    pytest.importorskip("catboost")
    X, y = _regression_frame(5)
    common = dict(
        k=2,
        n_splits=2,
        test_size=0.25,
        prefilter_k=None,
        n_estimators=8,
        max_depth=2,
        algorithm="prediction",
        train_early_stopping_rounds=3,
        random_state=5,
        n_jobs=1,
        verbose=False,
    )
    baseline = catboost_module.catboost_select(X, y, **common)
    events = []
    observed = catboost_module.catboost_select(
        X,
        y,
        **common,
        callback=lambda step, total, info: events.append((step, total, info)),
    )

    assert observed.selected_features == baseline.selected_features
    assert observed.scores_by_k == baseline.scores_by_k
    assert [step for step, _, _ in events] == [1, 2]
