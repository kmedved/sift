import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sift.importance import permutation_importance


def test_permutation_importance_auto_with_groups_and_time():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = X["f0"] * 2.0 + rng.normal(size=n) * 0.1
    groups = np.repeat(np.arange(5), 20)
    time = np.tile(np.arange(20), 5)

    model = LinearRegression().fit(X, y)
    result = permutation_importance(
        model,
        X,
        y.values,
        groups=groups,
        time=time,
        n_repeats=3,
        n_jobs=1,
        random_state=0,
    )

    assert set(result.columns) == {
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    }
    assert result.iloc[0]["feature"] == "f0"


def test_permutation_importance_requires_time_for_block():
    rng = np.random.default_rng(1)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = X["f0"] + rng.normal(size=n) * 0.1
    groups = np.repeat(np.arange(4), 10)

    model = LinearRegression().fit(X, y)
    try:
        permutation_importance(
            model,
            X,
            y.values,
            groups=groups,
            permute_method="block",
            n_repeats=2,
            n_jobs=1,
        )
    except ValueError as exc:
        assert "requires time" in str(exc)
    else:
        raise AssertionError("Expected ValueError when time is missing for block permutation.")
