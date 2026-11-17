"""Behaviour of ``sift.sample_rows``: exact draws, correct probabilities, unbiased weights."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import sift
from sift import RowSample, sample_rows
from sift.sampling.smart import _capped_pi, _class_gradient, _resolve_task


def _data(n=3_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.standard_t(3, size=n) * (0.3 + np.abs(X[:, 2]))
    return X, y


def test_fraction_and_count_draw_exact_sorted_unique_rows():
    X, y = _data()
    for size, expected in ((0.2, 600), (0.05, 150), (777, 777)):
        s = sample_rows(X, y, size, pilot=X[:, 0], random_state=0)
        assert isinstance(s, RowSample)
        assert len(s) == expected
        assert s.n_rows == len(y)
        assert np.all(np.diff(s.indices) > 0)
        assert s.indices.dtype == np.int64
        assert s.weights.mean() == pytest.approx(1.0)
        assert np.all((s.inclusion_prob > 0) & (s.inclusion_prob <= 1))
        np.testing.assert_allclose(s.weights * s.inclusion_prob, (s.weights * s.inclusion_prob)[0])
        assert s.fraction == pytest.approx(expected / len(y))


def test_size_at_or_above_n_keeps_every_row_unweighted():
    X, y = _data(n=200)
    for size in (1.0, 200, 10_000):
        s = sample_rows(X, y, size)
        np.testing.assert_array_equal(s.indices, np.arange(200))
        np.testing.assert_array_equal(s.weights, np.ones(200))
        np.testing.assert_array_equal(s.inclusion_prob, np.ones(200))


def test_take_handles_arrays_frames_and_series():
    X, y = _data(n=500)
    df = pd.DataFrame(X, columns=list("abc"), index=np.arange(1_000, 1_500))
    s = sample_rows(df, pd.Series(y, index=df.index), 0.1, pilot=LinearRegression(), random_state=1)
    np.testing.assert_array_equal(s.take(X), X[s.indices])
    assert s.take(df).index.tolist() == (1_000 + s.indices).tolist()
    assert s.take(pd.Series(y)).tolist() == y[s.indices].tolist()
    assert s.take(list(y)).tolist() == y[s.indices].tolist()


def test_same_seed_same_draw_and_different_seed_different_draw():
    X, y = _data()
    a = sample_rows(X, y, 0.1, random_state=5)
    b = sample_rows(X, y, 0.1, random_state=5)
    c = sample_rows(X, y, 0.1, pilot=X[:, 0], random_state=6)
    d = sample_rows(X, y, 0.1, pilot=X[:, 0], random_state=5)
    assert not np.array_equal(a.indices, d.indices)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert not np.array_equal(a.indices, c.indices)


@pytest.mark.parametrize("structure", ["none", "groups_time"])
def test_empirical_inclusion_frequency_matches_inclusion_probability(structure):
    rng = np.random.default_rng(3)
    n, k = 120, 30
    y = rng.standard_t(2, size=n)
    pred = np.zeros(n)
    groups = rng.integers(0, 9, n) if structure == "groups_time" else None
    time = rng.integers(0, 20, n) if structure == "groups_time" else None
    counts = np.zeros(n)
    draws = 4_000
    pi = None
    for seed in range(draws):
        s = sample_rows(None, y, k, groups=groups, time=time, pilot=pred, random_state=seed)
        counts[s.indices] += 1
        if pi is None:
            pi = np.zeros(n)
        pi[s.indices] = s.inclusion_prob
    score = np.sqrt(y ** 2 + np.mean(y ** 2))
    expected = _capped_pi(score, k)
    np.testing.assert_allclose(pi[pi > 0], expected[pi > 0])
    se = np.sqrt(expected * (1 - expected) / draws) + 1e-9
    assert np.max(np.abs(counts / draws - expected) / se) < 5.0


def test_weighted_estimates_are_unbiased_for_full_data_totals():
    X, y = _data(n=4_000, seed=1)
    est = [np.average(y[s.indices], weights=s.weights)
           for s in (sample_rows(X, y, 0.1, pilot=X[:, 0], random_state=r) for r in range(300))]
    se = np.std(est) / np.sqrt(len(est))
    assert abs(np.mean(est) - y.mean()) < 4 * se


def test_larger_residuals_get_larger_inclusion_probability():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1_000)
    s = sample_rows(None, y, 1_000 - 1, pilot=np.zeros(1_000), random_state=0)
    order = np.argsort(np.abs(y[s.indices]))
    assert np.all(np.diff(s.inclusion_prob[order]) >= -1e-12)


def test_mvs_score_bounds_weights_even_with_outliers():
    X, y = _data(n=5_000)
    y = y.copy()
    y[:5] = 1e6
    s = sample_rows(X, y, 0.2, pilot=LinearRegression(), random_state=0)
    assert s.weights.max() / s.weights.min() < 1e6
    assert set(range(5)) <= set(s.indices.tolist())


def test_uniform_mix_one_and_no_pilot_give_equal_probabilities():
    X, y = _data()
    for kwargs in ({"uniform_mix": 1.0}, {"pilot": None}):
        s = sample_rows(X, y, 0.1, random_state=0, **{"pilot": X[:, 0], **kwargs})
        np.testing.assert_allclose(s.inclusion_prob, 0.1)
        np.testing.assert_allclose(s.weights, 1.0)
    assert sample_rows(X, None, 0.1, pilot=None).task == "uniform"


def test_equal_probability_draw_spreads_exactly_across_groups():
    rng = np.random.default_rng(0)
    n = 10_000
    groups = rng.integers(0, 40, n)
    time = rng.integers(0, 1_000, n)
    s = sample_rows(None, None, 0.1, groups=groups, time=time, pilot=None, random_state=0)
    sizes = np.bincount(groups, minlength=40)
    got = np.bincount(groups[s.indices], minlength=40)
    assert np.all(np.abs(got - 0.1 * sizes) < 1.0 + 1e-9)


def test_time_order_spreads_draw_along_each_timeline():
    n = 1_000
    groups = np.repeat(np.arange(10), 100)
    time = np.tile(np.arange(100), 10)[::-1].copy()
    s = sample_rows(None, None, 100, groups=groups, time=time, pilot=None, random_state=2)
    for g in range(10):
        t = np.sort(time[s.indices][groups[s.indices] == g])
        assert len(t) == 10
        assert np.all(np.diff(t) == 10)


def test_missing_groups_are_a_group_not_dropped():
    n = 600
    groups = np.array(["a", "b", None] * (n // 3), dtype=object)
    s = sample_rows(None, None, 0.5, groups=groups, pilot=None, random_state=0)
    assert pd.isna(groups[s.indices]).sum() == 100


def test_always_include_rows_are_kept_with_probability_one():
    X, y = _data()
    keep = np.zeros(len(y), dtype=bool)
    keep[[3, 50, 999]] = True
    s = sample_rows(X, y, 100, pilot=X[:, 0], always_include=keep, random_state=0)
    assert len(s) == 100
    assert {3, 50, 999} <= set(s.indices.tolist())
    assert np.all(s.inclusion_prob[np.isin(s.indices, [3, 50, 999])] == 1.0)
    with pytest.raises(ValueError, match="always_include marks 3 rows but size allows only 2"):
        sample_rows(X, y, 2, always_include=keep)
    s = sample_rows(X, y, 3, always_include=keep)
    np.testing.assert_array_equal(s.indices, [3, 50, 999])


def test_auto_task_detection():
    _, y = _data(n=2_000)
    assert _resolve_task(y, "auto") == "regression"
    assert _resolve_task((y > 0).astype(int), "auto") == "classification"
    assert _resolve_task((y > 0).astype(float), "auto") == "classification"
    assert _resolve_task(y > 0, "auto") == "classification"
    assert _resolve_task(np.where(y > 0, "yes", "no"), "auto") == "classification"
    assert _resolve_task(pd.Series(["a", "b"] * 5, dtype="category").to_numpy(), "auto") == "classification"
    assert _resolve_task(np.round(y).astype(int), "auto") == "regression"
    assert _resolve_task(np.digitize(y, [-1, 1]), "auto") == "regression"
    assert _resolve_task(np.digitize(y, [-1, 1]), "classification") == "classification"
    s = sample_rows(None, (y > 0).astype(int), 0.1, pilot=np.full(len(y), 0.5))
    assert s.task == "classification"


def test_class_gradient_is_norm_of_logloss_gradient():
    classes = np.array([0, 1, 2])
    y = np.array([0, 2, 1])
    proba = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.1, 0.1, 0.8]])
    onehot = np.eye(3)[y]
    np.testing.assert_allclose(_class_gradient(y, proba, classes), np.linalg.norm(onehot - proba, axis=1))
    binary = _class_gradient(np.array([0, 1]), np.array([0.25, 0.25]), np.array([0, 1]))
    np.testing.assert_allclose(binary, np.sqrt(2) * np.array([0.25, 0.75]))


def test_classification_pilot_survives_class_missing_from_pilot_slice():
    rng = np.random.default_rng(0)
    n = 5_000
    X = rng.normal(size=(n, 2))
    y = np.where(X[:, 0] > 0, "a", "b").astype(object)
    y[:2] = "rare"
    for seed in range(3):
        s = sample_rows(X, y, 0.2, task="classification", pilot=LogisticRegression(),
                        pilot_size=1_000, random_state=seed)
        assert len(s) == 1_000
        assert np.isfinite(s.weights).all()


def test_custom_pilot_estimator_is_cloned_not_fitted():
    X, y = _data()
    pilot = LinearRegression()
    s = sample_rows(X, y, 0.1, pilot=pilot)
    assert len(s) == 300
    assert not hasattr(pilot, "coef_")
    clf = LogisticRegression()
    s = sample_rows(X, (y > 0).astype(int), 0.1, pilot=clf)
    assert s.task == "classification"
    assert not hasattr(clf, "coef_")


def test_classification_pilot_without_predict_proba_is_rejected():
    X, y = _data()
    with pytest.raises(TypeError, match="predict_proba"):
        sample_rows(X, (y > 0).astype(int), 0.1, pilot=LinearRegression())


def test_precomputed_pilot_matches_fitted_pilot_contract():
    X, y = _data()
    s = sample_rows(None, y, 0.1, pilot=X[:, 0])
    assert len(s) == 300
    yc = (y > 0).astype(int)
    s = sample_rows(None, yc, 0.1, pilot=np.full(len(y), 0.5))
    assert s.task == "classification"
    with pytest.raises(ValueError, match="precomputed pilot has 10 rows"):
        sample_rows(None, y, 0.1, pilot=np.zeros(10))
    bad = X[:, 0].copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sample_rows(None, y, 0.1, pilot=bad)


def test_perfect_pilot_falls_back_to_equal_probability_with_warning():
    _, y = _data()
    with pytest.warns(RuntimeWarning, match="pilot fits every row exactly"):
        s = sample_rows(None, y, 0.1, pilot=y)
    np.testing.assert_allclose(s.weights, 1.0)


def test_auto_pilot_accepts_missing_features_but_not_infinite_ones():
    X, y = _data()
    X = X.copy()
    X[::7, 1] = np.nan
    assert len(sample_rows(X, y, 0.1)) == 300
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        sample_rows(X, y, 0.1)


def test_auto_pilot_rejects_non_numeric_features():
    X, y = _data(n=300)
    df = pd.DataFrame({"a": X[:, 0], "b": ["x", "y", "z"] * 100})
    with pytest.raises(TypeError, match="numeric features"):
        sample_rows(df, y, 0.1)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"size": 0}, ValueError, "row count >= 1"),
        ({"size": 0.0}, ValueError, r"fraction in \(0, 1\]"),
        ({"size": 1.5}, ValueError, r"fraction in \(0, 1\]"),
        ({"size": float("nan")}, ValueError, r"fraction in \(0, 1\]"),
        ({"size": True}, TypeError, "int row count or a float"),
        ({"size": "10%"}, TypeError, "int row count or a float"),
        ({"pilot_size": -3}, ValueError, "pilot_size"),
        ({"uniform_mix": 1.5}, ValueError, "uniform_mix"),
        ({"uniform_mix": "half"}, TypeError, "uniform_mix"),
        ({"task": "ranking"}, ValueError, "task must be one of"),
        ({"pilot": 3}, TypeError, "pilot must be"),
        ({"random_state": 1.5}, TypeError, "random_state"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, error, match):
    X, y = _data(n=100)
    with pytest.raises(error, match=match):
        sample_rows(X, y, **{"size": 0.1, **kwargs})


def test_target_and_shape_validation():
    X, y = _data(n=100)
    with pytest.raises(ValueError, match="y is required"):
        sample_rows(X, None, 0.1)
    with pytest.raises(ValueError, match="needs at least one of"):
        sample_rows(None, None, 0.1, pilot=None)
    bad = y.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite for regression"):
        sample_rows(X, bad, 0.1)
    labels = np.array(["a", None] * 50, dtype=object)
    with pytest.raises(ValueError, match="missing labels"):
        sample_rows(X, labels, 0.1, task="classification")
    with pytest.raises(ValueError, match="groups must be 1-D"):
        sample_rows(X, y, 0.1, groups=np.zeros(99))
    with pytest.raises(ValueError, match="numeric"):
        sample_rows(X, np.array(["a"] * 100, dtype=object), 0.1, task="regression")


def test_verbose_logs_pilot_and_sample_summary(caplog):
    X, y = _data()
    with caplog.at_level(logging.INFO, logger="sift"):
        sample_rows(X, y, 0.1, verbose=True)
    text = caplog.text
    assert "Pilot fitted on 1,000 of 3,000 rows" in text
    assert "Sampled 300 of 3,000 rows" in text


def test_no_warnings_on_the_default_path():
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample_rows(X, y)


def test_top_level_exports():
    assert sift.sample_rows is sample_rows
    assert sift.RowSample is RowSample
