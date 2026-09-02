import numpy as np
import pandas as pd
import pytest

from sift import catboost as cb


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
