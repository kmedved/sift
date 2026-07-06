import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import sift.scoring as scoring_module
from sift.importance import permutation_importance
from sift.scoring import ScoringSpec


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


def test_permutation_importance_auto_with_time_only_is_deterministic():
    rng = np.random.default_rng(6)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = X["f0"] * 1.5 + rng.normal(size=n) * 0.1
    time = np.arange(n)

    model = LinearRegression().fit(X, y)
    result1 = permutation_importance(
        model,
        X,
        y.values,
        time=time,
        n_repeats=3,
        n_jobs=1,
        random_state=0,
    )
    result2 = permutation_importance(
        model,
        X,
        y.values,
        time=time,
        n_repeats=3,
        n_jobs=1,
        random_state=0,
    )

    assert result1.equals(result2)
    assert set(result1.columns) == {
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    }
    assert len(result1) == X.shape[1]
    assert result1.iloc[0]["feature"] == "f0"


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


def test_permutation_importance_passes_dataframes_to_predict_preserving_metadata():
    class DataFrameOnlyClassifier:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            assert isinstance(X, pd.DataFrame)
            assert list(X.columns) == ["cat", "label", "num"]
            assert isinstance(X["cat"].dtype, pd.CategoricalDtype)
            assert X["label"].dtype == object
            self.calls += 1
            return np.where(
                (X["cat"].astype(str) == "hot") & (X["label"] == "keep"),
                1,
                0,
            )

    X = pd.DataFrame(
        {
            "cat": pd.Categorical(["hot", "cold", "hot", "cold", "hot", "cold"]),
            "label": pd.Series(
                ["keep", "drop", "keep", "drop", "drop", "keep"],
                dtype=object,
            ),
            "num": [1.0, 0.0, 1.0, 0.0, 0.5, 0.25],
        }
    )
    X_before = X.copy(deep=True)
    model = DataFrameOnlyClassifier()
    y = model.predict(X)
    model.calls = 0

    result = permutation_importance(
        model,
        X,
        y,
        scoring="accuracy",
        n_repeats=2,
        n_jobs=1,
        parallel_backend="threads",
        random_state=0,
    )

    assert set(result.columns) == {
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    }
    assert set(result["feature"]) == set(X.columns)
    assert model.calls == 1 + X.shape[1] * 2
    pd.testing.assert_frame_equal(X, X_before)


def test_permutation_importance_ndarray_path_still_predicts_arrays():
    class ArrayOnlyClassifier:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            assert isinstance(X, np.ndarray)
            self.calls += 1
            return (X[:, 0] > 0).astype(int)

    rng = np.random.default_rng(5)
    X = rng.normal(size=(60, 3))
    X_before = X.copy()
    model = ArrayOnlyClassifier()
    y = model.predict(X)
    model.calls = 0

    result = permutation_importance(
        model,
        X,
        y,
        scoring="accuracy",
        n_repeats=3,
        n_jobs=1,
        parallel_backend="threads",
        random_state=0,
    )

    assert set(result.columns) == {
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    }
    assert set(result["feature"]) == {0, 1, 2}
    assert result.iloc[0]["feature"] == 0
    assert model.calls == 1 + X.shape[1] * 3
    np.testing.assert_array_equal(X, X_before)


def test_permutation_importance_rejects_unknown_parallel_backend():
    rng = np.random.default_rng(13)
    X = pd.DataFrame(rng.normal(size=(30, 2)), columns=["f0", "f1"])
    y = X["f0"].to_numpy()
    model = LinearRegression().fit(X, y)

    try:
        permutation_importance(
            model,
            X,
            y,
            n_repeats=2,
            n_jobs=1,
            parallel_backend="loky",
        )
    except ValueError as exc:
        assert "parallel_backend" in str(exc)
        assert "threads" in str(exc)
        assert "processes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown parallel_backend.")


def test_permutation_importance_processes_match_threads_for_dataframe_and_array():
    rng = np.random.default_rng(14)
    n = 70
    X_df = pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=["f0", "f1", "f2", "f3"],
    )
    y = 2.0 * X_df["f0"].to_numpy() - X_df["f1"].to_numpy()
    model = LinearRegression().fit(X_df, y)

    df_threads = permutation_importance(
        model,
        X_df,
        y,
        n_repeats=2,
        n_jobs=2,
        parallel_backend="threads",
        random_state=0,
    )
    df_processes = permutation_importance(
        model,
        X_df,
        y,
        n_repeats=2,
        n_jobs=2,
        parallel_backend="processes",
        random_state=0,
    )
    pd.testing.assert_frame_equal(
        df_threads.sort_values("feature").reset_index(drop=True),
        df_processes.sort_values("feature").reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    X_arr = X_df.to_numpy()
    array_model = LinearRegression().fit(X_arr, y)
    array_threads = permutation_importance(
        array_model,
        X_arr,
        y,
        n_repeats=2,
        n_jobs=2,
        parallel_backend="threads",
        random_state=0,
    )
    array_processes = permutation_importance(
        array_model,
        X_arr,
        y,
        n_repeats=2,
        n_jobs=2,
        parallel_backend="processes",
        random_state=0,
    )
    pd.testing.assert_frame_equal(
        array_threads.sort_values("feature").reset_index(drop=True),
        array_processes.sort_values("feature").reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_permutation_importance_accuracy_classification_ranks_signal():
    rng = np.random.default_rng(2)
    n = 120
    f0 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "f0": f0,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    y = np.where(f0 > 0, "win", "loss")

    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="accuracy",
        n_repeats=5,
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
    assert result.loc[result["feature"] == "f0", "importance_mean"].iloc[0] > 0
    assert result["baseline_score"].iloc[0] == 1.0


def test_permutation_importance_neg_error_classification_ranks_signal_with_weights():
    rng = np.random.default_rng(3)
    n = 120
    f0 = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "f0": f0,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    y = (f0 > 0).astype(int)
    sample_weight = np.where(y == 1, 2.0, 0.5)

    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(
        X,
        y,
        sample_weight=sample_weight,
    )
    result = permutation_importance(
        model,
        X,
        y,
        sample_weight=sample_weight,
        scoring="neg_error",
        n_repeats=5,
        n_jobs=1,
        random_state=0,
    )

    assert result.iloc[0]["feature"] == "f0"
    assert result.loc[result["feature"] == "f0", "importance_mean"].iloc[0] > 0
    assert result["baseline_score"].iloc[0] == 0.0


def test_permutation_importance_balanced_accuracy_ranks_signal_on_imbalanced_data():
    rng = np.random.default_rng(9)
    n = 180
    f0 = rng.normal(size=n)
    y = np.where(f0 > 1.0, 1, 0)
    X = pd.DataFrame(
        {
            "f0": f0,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )

    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="balanced_accuracy",
        n_repeats=5,
        n_jobs=1,
        random_state=0,
    )

    assert result.iloc[0]["feature"] == "f0"
    assert result.loc[result["feature"] == "f0", "importance_mean"].iloc[0] > 0


def test_permutation_importance_neg_logloss_ranks_signal():
    rng = np.random.default_rng(10)
    n = 180
    f0 = rng.normal(size=n)
    logits = 3.0 * f0
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p, size=n)
    X = pd.DataFrame(
        {
            "f0": f0,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )

    model = LogisticRegression(max_iter=1000).fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_logloss",
        n_repeats=5,
        n_jobs=1,
        random_state=0,
    )

    assert result.iloc[0]["feature"] == "f0"
    assert result.loc[result["feature"] == "f0", "importance_mean"].iloc[0] > 0
    assert result["baseline_score"].iloc[0] < 0.0


def test_permutation_importance_neg_logloss_multiclass_ranks_signal():
    rng = np.random.default_rng(12)
    n_per_class = 60
    y = np.repeat([0, 1, 2], n_per_class)
    f0 = np.concatenate(
        [
            rng.normal(-2.0, 0.5, size=n_per_class),
            rng.normal(0.0, 0.5, size=n_per_class),
            rng.normal(2.0, 0.5, size=n_per_class),
        ]
    )
    order = rng.permutation(len(y))
    X = pd.DataFrame(
        {
            "f0": f0[order],
            "f1": rng.normal(size=len(y)),
            "f2": rng.normal(size=len(y)),
        }
    )
    y = y[order]

    model = LogisticRegression(max_iter=1000).fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_logloss",
        n_repeats=5,
        n_jobs=1,
        random_state=0,
    )

    assert result.iloc[0]["feature"] == "f0"
    assert result.loc[result["feature"] == "f0", "importance_mean"].iloc[0] > 0
    assert result["baseline_score"].iloc[0] < 0.0


def test_permutation_importance_honors_lower_is_better_scoring(monkeypatch):
    def mse_loss(model, X, y, w):
        pred = np.asarray(model.predict(X), dtype=np.float64)
        return float(np.average((pred - y) ** 2, weights=w))

    monkeypatch.setitem(
        scoring_module._SCORING_REGISTRY,
        "mse_loss_for_test",
        ScoringSpec("mse_loss_for_test", mse_loss, higher_is_better=False),
    )
    rng = np.random.default_rng(15)
    n = 120
    X = pd.DataFrame(
        {
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = 2.5 * X["signal"].to_numpy()
    model = LinearRegression().fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        scoring="mse_loss_for_test",
        n_repeats=5,
        n_jobs=1,
        random_state=0,
    )

    assert result.iloc[0]["feature"] == "signal"
    assert result.loc[result["feature"] == "signal", "importance_mean"].iloc[0] > 0


def test_permutation_importance_rejects_plain_error_scoring():
    rng = np.random.default_rng(4)
    n = 40
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f0", "f1"])
    y = (X["f0"] > 0).astype(int).to_numpy()

    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    try:
        permutation_importance(
            model,
            X,
            y,
            scoring="error",
            n_repeats=2,
            n_jobs=1,
        )
    except ValueError as exc:
        assert "Unknown scoring" in str(exc)
        assert "neg_error" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported scoring='error'.")


def test_permutation_importance_logloss_requires_predict_proba():
    class PredictOnlyClassifier:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    X = pd.DataFrame({"f0": [0.0, 1.0, 2.0, 3.0], "f1": [1.0, 1.0, 0.0, 0.0]})
    y = np.array([0, 0, 1, 1])
    model = PredictOnlyClassifier()

    try:
        permutation_importance(
            model,
            X,
            y,
            scoring="neg_logloss",
            n_repeats=2,
            n_jobs=1,
        )
    except ValueError as exc:
        assert "predict_proba" in str(exc)
    else:
        raise AssertionError("Expected ValueError for neg_logloss without predict_proba.")


def test_permutation_importance_rejects_plain_logloss_scoring():
    rng = np.random.default_rng(11)
    n = 100
    f0 = rng.normal(size=n)
    y = (f0 + rng.normal(size=n) * 0.5 > 0).astype(int)
    X = pd.DataFrame(
        {
            "f0": f0,
            "f1": rng.normal(size=n),
        }
    )

    model = LogisticRegression(max_iter=1000).fit(X, y)
    try:
        permutation_importance(
            model,
            X,
            y,
            scoring="logloss",
            n_repeats=3,
            n_jobs=1,
            random_state=0,
        )
    except ValueError as exc:
        assert "Unknown scoring" in str(exc)
        assert "neg_logloss" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported scoring='logloss'.")


def test_permutation_importance_rejects_length_mismatch():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["f0", "f1"])
    y = rng.normal(size=19)
    model = LinearRegression().fit(X.iloc[:19], y)

    try:
        permutation_importance(model, X, y, n_repeats=2, n_jobs=1)
    except ValueError as exc:
        assert "X has 20 rows but y has 19" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched X/y lengths.")


def test_permutation_importance_rejects_nonpositive_repeats():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["f0", "f1"])
    y = X["f0"].to_numpy()
    model = LinearRegression().fit(X, y)

    for n_repeats in [0, -1]:
        try:
            permutation_importance(model, X, y, n_repeats=n_repeats, n_jobs=1)
        except ValueError as exc:
            assert "n_repeats" in str(exc)
        else:
            raise AssertionError("Expected ValueError for invalid n_repeats.")
