from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from sift import ModelSelector, catboost as cb
from sift.selection import orchestration as _selection_orchestration
from sift.selection.orchestration import SelectionBackend


@pytest.mark.parametrize("bad_value", ["bad", None, [], 1])
def test_catboost_validate_choice_rejects_bad_values(bad_value):
    with pytest.raises(ValueError, match="algorithm=.*invalid"):
        cb._validate_choice("algorithm", bad_value, {"shap", "prediction"})


@pytest.mark.parametrize("bad_value", [1.0, 0.0, "0.5", None, np.nan])
def test_catboost_step_function_validation_rejects_bad_values(bad_value):
    with pytest.raises(ValueError, match="step_function"):
        cb._validate_step_function(bad_value)


@pytest.mark.parametrize(
    ("n_bootstrap", "stability_threshold"),
    [
        (0, 0.5),
        (True, 0.5),
        (10, "0.5"),
        (10, np.nan),
        (10, 1.5),
    ],
)
def test_catboost_stability_validation_rejects_bad_values(
    n_bootstrap, stability_threshold
):
    with pytest.raises(ValueError):
        cb._validate_stability_params(n_bootstrap, stability_threshold)


@pytest.mark.catboost
def test_catboost_select_features_reorders_survivors():
    pytest.importorskip("catboost")

    rng = np.random.default_rng(0)
    n, p = 300, 20
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.normal(size=n))

    X_train = X.iloc[:200]
    X_val = X.iloc[200:]
    y_train = y.iloc[:200]
    y_val = y.iloc[200:]

    feature_counts = [5, 10]
    model_params = {
        "iterations": 50,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "random_seed": 0,
    }

    scores, features_by_k = cb._select_features_single_split(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task="regression",
        model_params=model_params,
        feature_counts=feature_counts,
        features=list(X.columns),
        cat_features=[],
        text_features=[],
        eval_metric="RMSE",
        higher_is_better=False,
        algorithm="prediction",
        steps=2,
        train_early_stopping_rounds=10,
    )

    assert scores
    for k in feature_counts:
        assert k in features_by_k
        assert len(features_by_k[k]) == k


def _choose(scores, *, hib, tolerance=0.01, patience=3, k_req=None):
    return cb._choose_catboost_target_k(
        {k: [v] for k, v in scores.items()},
        k_req=k_req,
        resolved_hib=hib,
        tolerance=tolerance,
        selection_patience=patience,
        verbose=False,
    )


def test_choose_target_k_global_argbest_lower_is_better_optimum_at_smallest_prefix():
    # Regression: the old scan walked down from the largest k and stopped after
    # `selection_patience` non-improvements, so it reported k=10 / 0.5 here.
    target_k, best_k, best_score, scores_mean, _ = _choose(
        {10: 0.5, 8: 0.5, 6: 0.5, 4: 0.5, 2: 0.1}, hib=False
    )
    assert (best_k, best_score) == (2, 0.1)
    assert target_k == 2
    assert scores_mean == {10: 0.5, 8: 0.5, 6: 0.5, 4: 0.5, 2: 0.1}


def test_choose_target_k_global_argbest_higher_is_better_optimum_at_smallest_prefix():
    target_k, best_k, best_score, _, _ = _choose(
        {12: 0.5, 10: 0.7, 8: 0.7, 6: 0.7, 4: 0.7, 2: 0.9}, hib=True
    )
    assert (best_k, best_score) == (2, 0.9)
    assert target_k == 2


def test_choose_target_k_global_argbest_optimum_in_the_middle_both_directions():
    target_k, best_k, best_score, _, _ = _choose(
        {10: 0.5, 5: 0.3, 3: 0.4},
        hib=False,
        tolerance=0.0,
    )
    assert (target_k, best_k, best_score) == (5, 5, 0.3)
    target_k, best_k, best_score, _, _ = _choose(
        {10: 0.5, 5: 0.9, 3: 0.7},
        hib=True,
        tolerance=0.0,
    )
    assert (target_k, best_k, best_score) == (5, 5, 0.9)


def test_choose_target_k_parsimony_uses_tolerance_band_and_patience():
    scores = {20: 0.30, 15: 0.31, 10: 0.305, 5: 0.302, 3: 0.60}
    # tolerance 1% of 0.30 = 0.003: k=5 (0.302) is inside, 15/10 are outside,
    # patience 3 tolerates the two misses on the way down.
    target_k, best_k, best_score, _, _ = _choose(
        scores,
        hib=False,
        tolerance=0.01,
        patience=3,
    )
    assert (best_k, best_score) == (20, 0.30)
    assert target_k == 5
    # patience 1 gives up at the first miss and keeps the best-scoring k.
    target_k, best_k, _, _, _ = _choose(
        scores,
        hib=False,
        tolerance=0.01,
        patience=1,
    )
    assert (target_k, best_k) == (20, 20)
    # tolerance 0 only admits exact ties, and exact ties prefer fewer features.
    target_k, best_k, _, _, _ = _choose(
        {10: 0.5, 5: 0.5, 3: 0.6},
        hib=False,
        tolerance=0.0,
    )
    assert (target_k, best_k) == (5, 5)


def test_choose_target_k_higher_is_better_parsimony_with_negative_scores():
    # neg_logloss-style scores: best is -0.20 at k=8; k=4 (-0.21) is within 10%.
    target_k, best_k, best_score, _, _ = _choose(
        {8: -0.20, 6: -0.25, 4: -0.21, 2: -0.40}, hib=True, tolerance=0.10
    )
    assert (best_k, best_score) == (8, -0.20)
    assert target_k == 4


def test_choose_target_k_requested_k_still_wins_and_ignores_nan_scores():
    target_k, best_k, _, _, _ = _choose(
        {10: 0.5, 5: 0.3, 3: 0.4},
        hib=False,
        k_req=7,
    )
    assert (target_k, best_k) == (5, 5)
    target_k, best_k, best_score, scores_mean, _ = cb._choose_catboost_target_k(
        {5: [float("nan")], 3: [1.0]},
        k_req=None,
        resolved_hib=False,
        tolerance=0.0,
        selection_patience=3,
        verbose=False,
    )
    assert (target_k, best_k, best_score) == (3, 3, 1.0)
    assert scores_mean == {3: 1.0}


@pytest.mark.parametrize(
    ("tolerance", "patience", "message"),
    [
        (-0.1, 3, "tolerance"),
        (np.nan, 3, "tolerance"),
        (True, 3, "tolerance"),
        (0.0, 0, "selection_patience"),
        (0.0, 2.5, "selection_patience"),
    ],
)
def test_choose_target_k_validates_selection_params(tolerance, patience, message):
    with pytest.raises(ValueError, match=message):
        _choose({10: 0.5, 5: 0.3}, hib=False, tolerance=tolerance, patience=patience)


def test_catboost_select_native_preset_uses_module_helpers(monkeypatch):
    """Public preset keeps monkeypatchable catboost.py helper names."""
    from sift.catboost_common import CatBoostSelectionResult

    monkeypatch.setattr(cb, "CatBoostRegressor", object)
    calls: list[str] = []

    def fake_evaluate(**kwargs):
        calls.append("evaluate")
        assert kwargs["algorithm"] == "prediction"
        return {1: [0.2, 0.3]}, {1: [["a", "b"], ["a", "c"]]}, ["a", "b"]

    def fake_importance(**kwargs):
        calls.append("importance")
        assert kwargs["selected_features"] == ["a"]
        return pd.Series({"a": 1.0, "b": 0.4})

    def fake_select(*args, **kwargs):
        calls.append("select")
        return {1: 0.25}, {1: ["a"]}

    monkeypatch.setattr(cb, "_run_catboost_split_evaluation", fake_evaluate)
    monkeypatch.setattr(cb, "_compute_final_catboost_importances", fake_importance)
    monkeypatch.setattr(cb, "_select_features_single_split", fake_select)

    X = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4), columns=list("abcd"))
    y = pd.Series(np.arange(5, dtype=float))
    result = cb.catboost_select(
        X,
        y,
        k=1,
        algorithm="prediction",
        prefilter_k=None,
        n_splits=2,
        n_estimators=10,
        random_state=0,
        verbose=False,
        train_early_stopping_rounds=3,
        n_jobs=1,
    )
    assert calls == ["evaluate", "importance"]
    assert "select" not in calls
    assert type(result) is CatBoostSelectionResult
    assert result.selected_features == ["a"]
    assert result.best_k == 1
    assert result.metric == "RMSE"
    assert result.higher_is_better is False


def test_catboost_select_monkeypatched_split_helper_is_honored(monkeypatch):
    from sift.catboost_common import CatBoostSelectionResult

    monkeypatch.setattr(cb, "CatBoostRegressor", object)
    seen = {"select": 0}

    def fake_select(*args, **kwargs):
        seen["select"] += 1
        return {1: 0.4}, {1: ["a"]}

    def fake_importance(**kwargs):
        return pd.Series({"a": 2.0})

    monkeypatch.setattr(cb, "_select_features_single_split", fake_select)
    monkeypatch.setattr(cb, "_compute_final_catboost_importances", fake_importance)

    X = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4), columns=list("abcd"))
    y = pd.Series(np.arange(5, dtype=float))
    result = cb.catboost_select(
        X,
        y,
        k=1,
        algorithm="prediction",
        prefilter_k=None,
        n_splits=2,
        n_estimators=10,
        random_state=0,
        verbose=False,
        train_early_stopping_rounds=3,
        n_jobs=1,
    )
    assert seen["select"] == 2
    assert type(result) is CatBoostSelectionResult
    assert result.selected_features == ["a"]
    assert result.best_k == 1


def _stub_native_catboost(monkeypatch, *, scores=None, paths=None, prefilter=None):
    monkeypatch.setattr(cb, "CatBoostRegressor", object)

    def fake_evaluate(**kwargs):
        del kwargs
        return (
            scores if scores is not None else {1: [0.2, 0.3]},
            paths if paths is not None else {1: [["a", "b"], ["a", "c"]]},
            prefilter if prefilter is not None else ["a", "b"],
        )

    def fake_importance(**kwargs):
        names = kwargs["selected_features"]
        return pd.Series({name: float(i + 1) for i, name in enumerate(names)})

    monkeypatch.setattr(cb, "_run_catboost_split_evaluation", fake_evaluate)
    monkeypatch.setattr(cb, "_compute_final_catboost_importances", fake_importance)


def test_public_routes_use_shared_run_selection(monkeypatch):
    seen: list[str] = []
    real = _selection_orchestration.run_selection

    def spy(backend, X, y, **context):
        seen.append(type(backend).__name__)
        assert isinstance(backend, SelectionBackend)
        return real(backend, X, y, **context)

    monkeypatch.setattr(_selection_orchestration, "run_selection", spy)
    _stub_native_catboost(monkeypatch)

    rng = np.random.default_rng(0)
    Xg = rng.normal(size=(24, 4))
    yg = Xg[:, 0] + 0.05 * rng.normal(size=24)
    ModelSelector(Ridge(), n_features_to_select=2, random_state=0).fit(Xg, yg)

    Xc = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4), columns=list("abcd"))
    yc = pd.Series(np.arange(5, dtype=float))
    cb.catboost_select(
        Xc,
        yc,
        k=1,
        algorithm="prediction",
        prefilter_k=None,
        n_splits=2,
        n_estimators=10,
        random_state=0,
        verbose=False,
        train_early_stopping_rounds=3,
        n_jobs=1,
    )
    assert seen == ["_GenericModelBackend", "_CatBoostNativePreset"]
    assert issubclass(cb._CatBoostNativePreset, SelectionBackend)


def test_catboost_select_count_shortfall_warning_points_to_public_caller(monkeypatch):
    _stub_native_catboost(
        monkeypatch,
        scores={3: [0.4, 0.5]},
        paths={3: [["a", "b", "c"], ["a", "b", "c"]]},
        prefilter=["a", "b", "c"],
    )
    X = pd.DataFrame(np.arange(15, dtype=float).reshape(5, 3), columns=list("abc"))
    y = pd.Series(np.arange(5, dtype=float))
    with pytest.warns(UserWarning, match=r"k=5 exceeds max evaluated") as caught:
        result = cb.catboost_select(
            X,
            y,
            k=5,
            algorithm="prediction",
            prefilter_k=None,
            n_splits=2,
            n_estimators=10,
            random_state=0,
            verbose=False,
            train_early_stopping_rounds=3,
            n_jobs=1,
        )
    assert result.best_k == 3
    assert Path(caught[0].filename) == Path(__file__)


def test_catboost_select_unknown_cat_features_warning_points_to_public_caller(
    monkeypatch,
):
    _stub_native_catboost(monkeypatch)
    X = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4), columns=list("abcd"))
    y = pd.Series(np.arange(5, dtype=float))
    with pytest.warns(UserWarning, match="cat_features not found") as caught:
        cb.catboost_select(
            X,
            y,
            k=1,
            algorithm="prediction",
            cat_features=["missing"],
            prefilter_k=None,
            n_splits=2,
            n_estimators=10,
            random_state=0,
            verbose=False,
            train_early_stopping_rounds=3,
            n_jobs=1,
        )
    assert Path(caught[0].filename) == Path(__file__)


def test_choose_target_k_shortfall_warning_points_to_helper_caller():
    with pytest.warns(UserWarning, match=r"k=5 exceeds max evaluated") as caught:
        target_k, best_k, *_ = cb._choose_catboost_target_k(
            {3: [0.4], 1: [0.2]},
            k_req=5,
            resolved_hib=False,
            tolerance=0.0,
            selection_patience=3,
            verbose=False,
        )
    assert (target_k, best_k) == (3, 1)
    assert Path(caught[0].filename) == Path(__file__)
