"""Behaviour of ``sift.sample_rows``: exact draws, correct probabilities, mean-one weights."""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import sift
from sift import RowSample, sample_rows
from sift.sampling.smart import _capped_pi, _class_gradient, _ordered_draw, _resolve_task


def _data(n=3_000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] - 0.5 * X[:, 1] + rng.standard_t(3, size=n) * (0.3 + np.abs(X[:, 2]))
    return X, y


def test_fraction_and_count_draw_exact_sorted_unique_rows():
    X, y = _data()
    for size, expected in ((0.2, 600), (0.05, 150), (777, 777)):
        s = sample_rows(X, y, size, pilot=X[:, 0], random_state=0, method="residual")
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
        s = sample_rows(X, y, size, method="residual")
        np.testing.assert_array_equal(s.indices, np.arange(200))
        np.testing.assert_array_equal(s.weights, np.ones(200))
        np.testing.assert_array_equal(s.inclusion_prob, np.ones(200))


def test_take_handles_arrays_frames_and_series():
    X, y = _data(n=500)
    df = pd.DataFrame(X, columns=list("abc"), index=np.arange(1_000, 1_500))
    s = sample_rows(df, pd.Series(y, index=df.index), 0.1, pilot=LinearRegression(), random_state=1, method="residual")
    np.testing.assert_array_equal(s.take(X), X[s.indices])
    assert s.take(df).index.tolist() == (1_000 + s.indices).tolist()
    assert s.take(pd.Series(y)).tolist() == y[s.indices].tolist()
    assert s.take(list(y)).tolist() == y[s.indices].tolist()


def test_same_seed_same_draw_and_different_seed_different_draw():
    X, y = _data()
    a = sample_rows(X, y, 0.1, random_state=5, method="residual")
    b = sample_rows(X, y, 0.1, random_state=5, method="residual")
    c = sample_rows(X, y, 0.1, pilot=X[:, 0], random_state=6, method="residual")
    d = sample_rows(X, y, 0.1, pilot=X[:, 0], random_state=5, method="residual")
    assert not np.array_equal(a.indices, d.indices)
    np.testing.assert_array_equal(a.indices, b.indices)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert not np.array_equal(a.indices, c.indices)


def test_empirical_inclusion_frequency_matches_inclusion_probability():
    rng = np.random.default_rng(3)
    n, k = 120, 30
    y = rng.standard_t(2, size=n)
    pred = np.zeros(n)
    counts = np.zeros(n)
    draws = 4_000
    pi = np.zeros(n)
    for seed in range(draws):
        s = sample_rows(None, y, k, pilot=pred, random_state=seed, method="residual")
        assert s.method == "residual"
        counts[s.indices] += 1
        pi[s.indices] = s.inclusion_prob
    expected = 0.5 * _capped_pi(np.abs(y), k) + 0.5 * k / n
    np.testing.assert_allclose(pi[pi > 0], expected[pi > 0])
    se = np.sqrt(expected * (1 - expected) / draws) + 1e-9
    assert np.max(np.abs(counts / draws - expected) / se) < 5.0


@pytest.mark.parametrize("method", ["residual", "residual_timeline", "uniform", "chronological"])
def test_raw_inverse_probability_totals_are_unbiased(method):
    # Horvitz-Thompson: sum(y / inclusion_prob) over the sample estimates sum(y),
    # including forced rows (probability 1).
    X, y = _data(n=4_000, seed=1)
    forced = np.zeros(len(y), bool)
    forced[np.argsort(y)[-5:]] = True
    groups = np.repeat(np.arange(40), 100)
    kwargs = {"residual": {"pilot": X[:, 0]}, "uniform": {},
              "residual_timeline": {"pilot": X[:, 0], "groups": groups, "method": "residual"},
              "chronological": {"groups": groups}}[method]
    kwargs.setdefault("method", method)
    totals = []
    for r in range(300):
        s = sample_rows(X, y, 0.1, always_include=forced, random_state=r, **kwargs)
        totals.append(np.sum(y[s.indices] / s.inclusion_prob))
    se = np.std(totals) / np.sqrt(len(totals))
    assert abs(np.mean(totals) - y.sum()) < 4 * se


def test_forced_rows_make_equal_probability_weights_unequal():
    y = np.arange(5.0)
    forced = np.array([True, False, False, False, False])
    for kwargs in ({"method": "uniform"}, {"method": "chronological", "time": np.arange(5)}):
        s = sample_rows(None, y, 2, always_include=forced, random_state=0, **kwargs)
        assert s.indices[0] == 0 and len(s) == 2
        np.testing.assert_allclose(s.inclusion_prob, [1.0, 0.25])
        np.testing.assert_allclose(s.weights, [0.4, 1.6])
        assert np.all(sample_rows(None, y, 2, random_state=0, **kwargs).weights == 1.0)


def test_residual_draw_does_not_lock_onto_a_repeating_series():
    # y repeats [0, 0, 0, 0, 1] and the pilot predicts 0, so each 1 has
    # probability 0.6, each 0 has 0.1, and every 5 rows hold one pick. Along a
    # fixed order one shared start keeps all ones or none.
    n = 5_000
    y = (np.arange(n) % 5 == 4).astype(float)
    means = []
    for seed in range(100):
        s = sample_rows(None, y, 0.2, pilot=np.zeros(n), random_state=seed, method="residual")
        assert s.method == "residual"
        assert len(s) == 1_000 and np.unique(s.indices).size == 1_000
        means.append(np.average(y[s.indices], weights=s.weights))
    assert 0.15 < min(means) and max(means) < 0.25


def _mixed_pi(score, k, uniform_mix):
    return (1 - uniform_mix) * _capped_pi(score, k) + uniform_mix * k / score.size


@pytest.mark.parametrize("uniform_mix", [0.5, 0.02, 1e-9])
def test_ordered_draw_is_exact_and_matches_inclusion_probability(uniform_mix):
    # A small uniform_mix puts the capped rows' probability at or next to 1,
    # where a row can fill a whole unit of cumulative probability.
    rng = np.random.default_rng(11)
    n, k, draws = 400, 60, 20_000
    score = np.abs(rng.standard_t(2, size=n))
    score[rng.choice(n, 8, replace=False)] = 1e4
    pi = _mixed_pi(score, k, uniform_mix)
    assert pi.max() > 0.98 - 0.5 * (uniform_mix == 0.5)
    order = rng.permutation(n)
    counts = np.zeros(n)
    for _ in range(draws):
        drawn = _ordered_draw(order, pi, k, rng)
        assert drawn.size == k and np.unique(drawn).size == k
        counts[drawn] += 1
    se = np.sqrt(pi * (1 - pi) / draws) + 1e-4
    assert np.max(np.abs(counts / draws - pi) / se) < 5.0


def test_ordered_draw_keeps_every_prefix_within_one_row_of_its_probability():
    # One row per unit of cumulative probability: along the order, the rows
    # drawn so far never differ from the probability so far by more than one.
    rng = np.random.default_rng(5)
    n, k = 3_000, 450
    pi = _mixed_pi(np.abs(rng.standard_t(3, size=n)), k, 0.5)
    order = rng.permutation(n)
    for _ in range(50):
        taken = np.zeros(n)
        taken[_ordered_draw(order, pi, k, rng)] = 1
        assert np.abs(np.cumsum(taken[order]) - np.cumsum(pi[order])).max() <= 1 + 1e-9


def test_residual_timeline_draw_does_not_lock_onto_a_repeating_series():
    # The same repeating series as above, now drawn in time order, where one
    # shared start would keep all the ones or none of them.
    n = 5_000
    y = (np.arange(n) % 5 == 4).astype(float)
    means = []
    for seed in range(100):
        s = sample_rows(None, y, 0.2, pilot=np.zeros(n), time=np.arange(n), random_state=seed,
                        method="residual")
        assert len(s) == 1_000 and np.unique(s.indices).size == 1_000
        means.append(np.average(y[s.indices], weights=s.weights))
    assert 0.15 < min(means) and max(means) < 0.25
    # Equal probabilities and a period of exactly 1 / fraction rows.
    weekday = np.arange(n) % 5
    for seed in range(20):
        s = sample_rows(None, y, 0.2, pilot=np.zeros(n), time=np.arange(n), uniform_mix=1.0,
                        random_state=seed, method="residual")
        np.testing.assert_allclose(s.weights, 1.0)
        assert np.all(np.abs(np.bincount(weekday[s.indices], minlength=5) / 1_000 - 0.2) < 0.06)


def test_residual_timeline_draw_spreads_over_strata_entities_and_time():
    rng = np.random.default_rng(8)
    entities, seasons, games = 40, 4, 50
    groups = np.repeat(np.arange(entities), seasons * games)
    strata = np.tile(np.repeat(np.arange(seasons), games), entities)
    time = np.tile(np.arange(seasons * games), entities)
    y = rng.standard_t(3, size=groups.size) * (1 + (strata == 3))
    k = groups.size // 5
    pi = _mixed_pi(np.abs(y), k, 0.5)
    cell = groups * seasons + strata
    for seed in range(20):
        s = sample_rows(None, y, k, pilot=np.zeros(y.size), groups=groups, time=time[::-1].copy(),
                        strata=strata, random_state=seed, method="residual")
        assert len(s) == k
        np.testing.assert_allclose(s.inclusion_prob, pi[s.indices])
        taken = np.zeros(y.size)
        taken[s.indices] = 1
        # Each season, and each entity-season inside it, is one stretch of the order.
        assert np.abs(np.bincount(strata, taken) - np.bincount(strata, pi)).max() < 2
        assert np.abs(np.bincount(cell, taken) - np.bincount(cell, pi)).max() < 2
    again = sample_rows(None, y, k, pilot=np.zeros(y.size), groups=groups, time=time,
                        strata=strata, random_state=3, method="residual")
    same = sample_rows(None, y, k, pilot=np.zeros(y.size), groups=groups, time=time,
                       strata=strata, random_state=3, method="residual")
    np.testing.assert_array_equal(again.indices, same.indices)
    # Two seasons of three rows each and three draws: never all from one season.
    two = np.repeat([0, 1], 3)
    for seed in range(50):
        s = sample_rows(None, np.arange(6.0), 3, pilot=np.zeros(6), task="regression",
                        uniform_mix=1.0, groups=np.tile(np.arange(3), 2), time=np.arange(6),
                        strata=two, random_state=seed, method="residual")
        assert sorted(np.bincount(two[s.indices], minlength=2)) == [1, 2]


def test_residual_timeline_draw_accepts_dates_and_missing_groups():
    X, y = _data(n=600)
    dates = np.array(pd.date_range("2024-01-01", periods=600))
    groups = np.where(np.arange(600) % 3 == 0, None, "a").astype(object)
    s = sample_rows(X, y, 60, pilot=X[:, 0], groups=groups, time=dates, method="residual",
                    random_state=0)
    assert len(s) == 60 and np.unique(s.indices).size == 60
    assert 10 <= pd.isna(groups[s.indices]).sum() <= 30


def test_larger_residuals_get_larger_inclusion_probability():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1_000)
    s = sample_rows(None, y, 1_000 - 1, pilot=np.zeros(1_000), random_state=0, method="residual")
    order = np.argsort(np.abs(y[s.indices]))
    assert np.all(np.diff(s.inclusion_prob[order]) >= -1e-12)


def test_uniform_share_bounds_weights_even_with_outliers():
    X, y = _data(n=5_000)
    y = y.copy()
    y[:5] = 1e6
    s = sample_rows(X, y, 0.2, pilot=LinearRegression(), random_state=0, method="residual")
    # Every row keeps at least uniform_mix * k / n = 0.1, so raw 1/pi <= 10; a capped
    # row has pi = 0.5 * 1 + 0.5 * 0.2 = 0.6.
    raw = 1.0 / s.inclusion_prob
    assert raw.max() <= 10.0 + 1e-9
    assert raw.min() == pytest.approx(1 / 0.6)
    outliers = np.isin(s.indices, np.arange(5))
    np.testing.assert_allclose(s.inclusion_prob[outliers], 0.6)


def test_forced_rows_raise_the_raw_weight_cap_to_the_remaining_fraction():
    # 100 rows, k = 20, 19 forced: one draw from the 81 other rows, so the
    # remaining fraction is 1/81, not k/n = 0.2.
    rng = np.random.default_rng(0)
    n, k, n_forced = 100, 20, 19
    X, y = rng.normal(size=(n, 3)), rng.normal(size=n)
    forced = np.arange(n) < n_forced
    f = (k - n_forced) / (n - n_forced)
    for method, extra in (("uniform", {}), ("chronological", {"time": np.arange(n)})):
        s = sample_rows(X, y, k, method=method, always_include=forced, random_state=1, **extra)
        raw = 1.0 / s.inclusion_prob
        np.testing.assert_allclose(raw[np.isin(s.indices, np.flatnonzero(forced))], 1.0)
        assert raw.max() == pytest.approx(1 / f)
    worst = 0.0
    for seed in range(50):
        noise = np.random.default_rng(seed).normal(size=n)
        s = sample_rows(X, y, k, method="residual", pilot=y + noise**3, always_include=forced,
                        random_state=seed)
        worst = max(worst, (1.0 / s.inclusion_prob).max())
    assert 1 / (0.5 * k / n) < worst <= 1 / (0.5 * f) + 1e-9


def test_rows_the_pilot_fits_exactly_stay_drawable():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2_000)
    pred = np.where(np.arange(2_000) < 1_000, y, 0.0)  # first half fitted exactly
    seen = np.zeros(2_000, dtype=bool)
    for seed in range(20):
        s = sample_rows(None, y, 0.2, pilot=pred, random_state=seed, method="residual")
        assert s.inclusion_prob.min() >= 0.5 * 0.2 - 1e-12
        seen[s.indices] = True
    assert seen[:1_000].mean() > 0.5


def test_equal_probability_methods_and_uniform_mix_one():
    X, y = _data()
    groups = np.repeat(np.arange(30), 100)
    cases = [
        sample_rows(X, y, 0.1, pilot=X[:, 0], uniform_mix=1.0, random_state=0, method="residual"),
        sample_rows(size=0.1, method="uniform", always_include=np.zeros(len(y), bool)),
        sample_rows(size=0.1, method="chronological", groups=groups),
    ]
    for s in cases:
        assert len(s) == 300
        np.testing.assert_allclose(s.inclusion_prob, 0.1)
        np.testing.assert_allclose(s.weights, 1.0)
    assert [s.method for s in cases] == ["residual", "uniform", "chronological"]
    assert [s.task for s in cases] == ["regression", None, None]


def test_uniform_method_is_a_simple_random_sample():
    n, draws = 60, 6_000
    counts = np.zeros(n)
    for seed in range(draws):
        s = sample_rows(size=12, method="uniform", always_include=np.zeros(n, bool), random_state=seed)
        counts[s.indices] += 1
    assert np.abs(counts / draws - 0.2).max() < 5 * np.sqrt(0.2 * 0.8 / draws)


def test_chronological_method_calls_chronological_sample():
    rng = np.random.default_rng(0)
    n = 4_000
    frame = pd.DataFrame({
        "player": rng.integers(0, 80, n),
        "day": rng.integers(0, 400, n),
        "season": rng.integers(2019, 2024, n),
    })
    for seed in range(3):
        s = sample_rows(size=0.15, method="chronological", groups=frame["player"],
                        time=frame["day"], strata=frame["season"], random_state=seed)
        expected = sift.sampling.chronological_sample(
            frame, 600, group_col="player", time_col="day", stratum_col="season",
            random_state=seed)
        np.testing.assert_array_equal(s.indices, expected)
    seasons = np.repeat([0, 1], 3)
    players = np.tile(np.arange(3), 2)
    for seed in range(50):
        s = sample_rows(size=3, method="chronological", groups=players, time=np.arange(6),
                        strata=seasons, random_state=seed)
        assert sorted(np.bincount(seasons[s.indices], minlength=2)) == [1, 2]


def test_chronological_method_spreads_along_each_timeline():
    groups = np.repeat(np.arange(10), 100)
    time = np.tile(np.arange(100), 10)[::-1].copy()
    s = sample_rows(size=100, method="chronological", groups=groups, time=time, random_state=2)
    for g in range(10):
        t = np.sort(time[s.indices][groups[s.indices] == g])
        assert len(t) == 10
        gaps = np.r_[np.diff(t), t[0] + 100 - t[-1]]  # one pick per (rotated) block of 10
        assert gaps.max() < 20
    single = sample_rows(size=0.2, method="chronological", time=np.arange(5_000), random_state=0)
    weekday = np.arange(5_000) % 5
    assert np.all(np.abs(np.bincount(weekday[single.indices], minlength=5) / 1_000 - 0.2) < 0.05)


def test_missing_groups_are_a_group_not_dropped():
    n = 600
    groups = np.array(["a", "b", None] * (n // 3), dtype=object)
    s = sample_rows(size=0.5, method="chronological", groups=groups, random_state=0)
    assert pd.isna(groups[s.indices]).sum() == 100


def test_always_include_rows_are_kept_with_probability_one():
    X, y = _data()
    keep = np.zeros(len(y), dtype=bool)
    keep[[3, 50, 999]] = True
    groups = np.repeat(np.arange(30), 100)
    for kwargs in ({"X": X, "y": y, "pilot": X[:, 0], "method": "residual"}, {"method": "uniform"},
                   {"X": X, "y": y, "pilot": X[:, 0], "method": "residual", "groups": groups},
                   {"method": "chronological", "groups": groups}):
        s = sample_rows(size=100, always_include=keep, random_state=0, **kwargs)
        assert len(s) == 100
        assert {3, 50, 999} <= set(s.indices.tolist())
        assert np.all(s.inclusion_prob[np.isin(s.indices, [3, 50, 999])] == 1.0)
    with pytest.raises(ValueError, match="always_include marks 3 rows but size allows only 2"):
        sample_rows(X, y, 2, always_include=keep, method="residual")
    with pytest.raises(ValueError, match="the whole budget"):
        sample_rows(X, y, 3, always_include=keep, method="residual")
    s = sample_rows(X[:3], y[:3], 3, always_include=np.ones(3, bool), method="residual")
    np.testing.assert_array_equal(s.indices, [0, 1, 2])


def test_structure_arguments_do_not_apply_to_the_uniform_method():
    X, y = _data(n=500)
    for kwargs in ({"groups": np.zeros(500)}, {"time": np.arange(500)}, {"strata": np.zeros(500)}):
        with pytest.raises(ValueError, match="method='uniform' draws rows in random order"):
            sample_rows(size=0.1, method="uniform", **kwargs)
    with pytest.raises(ValueError, match="needs groups or time"):
        sample_rows(X, y, 0.1, method="chronological")
    with pytest.raises(ValueError, match="strata needs groups or time"):
        sample_rows(X, y, 0.1, method="residual", strata=np.zeros(500))
    with pytest.raises(ValueError, match="time must not contain missing values"):
        sample_rows(X, y, 0.1, method="residual", time=np.r_[np.nan, np.arange(499.0)])
    with pytest.raises(ValueError, match="needs a pilot"):
        sample_rows(X, y, 0.1, pilot=None, method="residual")
    with pytest.raises(ValueError, match="method must be one of"):
        sample_rows(X, y, 0.1, method="smart")


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
    s = sample_rows(None, (y > 0).astype(int), 0.1, pilot=np.full(len(y), 0.5), method="residual")
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
                        pilot_size=1_000, random_state=seed, method="residual")
        assert len(s) == 1_000
        assert np.isfinite(s.weights).all()


@pytest.mark.parametrize("pilot", ["auto", LogisticRegression()])
@pytest.mark.parametrize("rare_label", [0, 1])
def test_pilot_slice_with_one_class_scores_the_other_class_highest(pilot, rare_label):
    rng = np.random.default_rng(0)
    n = 6_000
    X = rng.normal(size=(n, 4))
    y = np.full(n, 1 - rare_label)
    y[1234] = rare_label
    with pytest.warns(RuntimeWarning, match="single class"):
        s = sample_rows(X, y, 0.2, task="classification", pilot=pilot, random_state=0, method="residual")
    assert len(s) == 1_200
    common = s.indices != 1234
    np.testing.assert_allclose(s.inclusion_prob[common], 0.5 * 1_199 / 5_999 + 0.5 * 0.2)
    assert s.inclusion_prob[~common] == pytest.approx([0.5 + 0.5 * 0.2] * (~common).sum())


def test_custom_pilot_estimator_is_cloned_not_fitted():
    X, y = _data()
    pilot = LinearRegression()
    s = sample_rows(X, y, 0.1, pilot=pilot, method="residual")
    assert len(s) == 300
    assert not hasattr(pilot, "coef_")
    clf = LogisticRegression()
    s = sample_rows(X, (y > 0).astype(int), 0.1, pilot=clf, method="residual")
    assert s.task == "classification"
    assert not hasattr(clf, "coef_")


def test_classification_pilot_without_predict_proba_is_rejected():
    X, y = _data()
    with pytest.raises(TypeError, match="predict_proba"):
        sample_rows(X, (y > 0).astype(int), 0.1, pilot=LinearRegression(), method="residual")


def test_precomputed_pilot_matches_fitted_pilot_contract():
    X, y = _data()
    s = sample_rows(None, y, 0.1, pilot=X[:, 0], method="residual")
    assert len(s) == 300
    yc = (y > 0).astype(int)
    s = sample_rows(None, yc, 0.1, pilot=np.full(len(y), 0.5), method="residual")
    assert s.task == "classification"
    with pytest.raises(ValueError, match="precomputed pilot has 10 rows"):
        sample_rows(None, y, 0.1, pilot=np.zeros(10), method="residual")
    bad = X[:, 0].copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sample_rows(None, y, 0.1, pilot=bad, method="residual")


def test_perfect_pilot_falls_back_to_equal_probability_with_warning():
    _, y = _data()
    with pytest.warns(RuntimeWarning, match="pilot fits every row exactly"):
        s = sample_rows(None, y, 0.1, pilot=y, method="residual")
    np.testing.assert_allclose(s.weights, 1.0)


def test_auto_pilot_accepts_missing_features_but_not_infinite_ones():
    X, y = _data()
    X = X.copy()
    X[::7, 1] = np.nan
    assert len(sample_rows(X, y, 0.1, method="residual")) == 300
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        sample_rows(X, y, 0.1, method="residual")


def test_auto_pilot_rejects_non_numeric_features():
    X, y = _data(n=300)
    df = pd.DataFrame({"a": X[:, 0], "b": ["x", "y", "z"] * 100})
    with pytest.raises(TypeError, match=r"non-numeric: \['b'\]"):
        sample_rows(df, y, 0.1, method="residual")


_PILOT_COLUMNS = {
    "int": lambda n: np.arange(n),
    "bool": lambda n: np.arange(n) % 2 == 0,
    "nullable_int": lambda n: pd.array(np.arange(n), dtype="Int64"),
    "nullable_bool": lambda n: pd.array(np.arange(n) % 2 == 0, dtype="boolean"),
    "datetime": lambda n: pd.date_range("2020-01-01", periods=n),
    "datetime_tz": lambda n: pd.date_range("2020-01-01", periods=n, tz="UTC"),
    "timedelta": lambda n: pd.to_timedelta(np.arange(n), unit="D"),
    "text": lambda n: np.array(["x", "y"] * (n // 2), dtype=object),
    "category": lambda n: pd.Categorical(["x", "y"] * (n // 2)),
}
_PILOT_ACCEPTS = {"int", "bool", "nullable_int", "nullable_bool"}


@pytest.mark.parametrize("kind", sorted(_PILOT_COLUMNS))
def test_auto_pilot_column_rule_is_shared_by_sample_rows_and_smart_sample(kind):
    # Numeric and boolean columns feed the default pilot; datetime, timedelta,
    # text and categorical columns are rejected by both entry points, by name,
    # instead of being cast to low-precision floats.
    n = 200
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "c": _PILOT_COLUMNS[kind](n), "y": rng.normal(size=n)})
    calls = (
        lambda: sample_rows(df[["a", "c"]], df["y"], 0.2, method="residual", random_state=0),
        lambda: sift.smart_sample(df, ["a", "c"], "y", method="residual", random_state=0),
    )
    for call in calls:
        if kind in _PILOT_ACCEPTS:
            assert len(call()) == 40
        else:
            with pytest.raises(TypeError, match=r"needs numeric or boolean features; non-numeric: \['c'\]"):
                call()


def test_auto_pilot_column_rule_covers_arrays():
    n = 200
    y = np.random.default_rng(0).normal(size=n)
    assert len(sample_rows(np.c_[np.arange(n) % 2 == 0, np.arange(n)], y, 0.2, method="residual")) == 40
    stamps = np.array(pd.date_range("2020-01-01", periods=n)).reshape(-1, 1)
    for X in (stamps, np.array([["x"]] * n)):
        with pytest.raises(TypeError, match=r"non-numeric: \['<array>'\]"):
            sample_rows(X, y, 0.2, method="residual")
    # An explicit pilot estimator or precomputed predictions bypass the rule.
    assert len(sample_rows(None, y, 0.2, method="residual", pilot=np.zeros(n))) == 40


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
        ({"uniform_mix": 0.0}, ValueError, r"uniform_mix must be in \(0, 1\]"),
        ({"uniform_mix": "half"}, TypeError, "uniform_mix"),
        ({"task": "ranking"}, ValueError, "task must be one of"),
        ({"pilot": 3}, TypeError, "pilot must be"),
        ({"random_state": 1.5}, TypeError, "random_state"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, error, match):
    X, y = _data(n=100)
    with pytest.raises(error, match=match):
        sample_rows(X, y, method="residual", **{"size": 0.1, **kwargs})


def test_target_and_shape_validation():
    X, y = _data(n=100)
    with pytest.raises(ValueError, match="y is required"):
        sample_rows(X, None, 0.1, method="residual")
    with pytest.raises(ValueError, match="needs at least one of"):
        sample_rows(None, None, 0.1, pilot=None, method="residual")
    bad = y.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite for regression"):
        sample_rows(X, bad, 0.1, method="residual")
    labels = np.array(["a", None] * 50, dtype=object)
    with pytest.raises(ValueError, match="missing labels"):
        sample_rows(X, labels, 0.1, task="classification", method="residual")
    with pytest.raises(ValueError, match="groups must be 1-D"):
        sample_rows(X, y, 0.1, groups=np.zeros(99), method="residual")
    with pytest.raises(ValueError, match="numeric"):
        sample_rows(X, np.array(["a"] * 100, dtype=object), 0.1, task="regression", method="residual")


def test_verbose_logs_pilot_and_sample_summary(caplog):
    X, y = _data()
    with caplog.at_level(logging.INFO, logger="sift"):
        sample_rows(X, y, 0.1, verbose=True, method="residual")
    text = caplog.text
    assert "Pilot fitted on 1,000 of 3,000 rows" in text
    assert "Sampled 300 of 3,000 rows" in text


def test_no_warnings_on_the_default_or_residual_path():
    X, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sample_rows(X, y)
        sample_rows(X, y, method="residual")


def test_default_method_is_a_uniform_draw_without_a_pilot():
    X, y = _data(n=1_000)
    s = sample_rows(X, y, 0.1, random_state=0)
    assert s.method == "uniform" and s.task is None
    assert len(s) == 100 and np.all(s.weights == 1.0)
    np.testing.assert_array_equal(s.indices, sample_rows(size=0.1, method="uniform",
                                                         always_include=np.zeros(1_000, bool),
                                                         random_state=0).indices)
    assert sift.SmartSamplerConfig().method == "uniform"


@pytest.mark.parametrize("kwargs", [
    {"pilot": None}, {"pilot": LinearRegression()}, {"pilot": np.zeros(500)},
    {"pilot_size": 0.05}, {"uniform_mix": 0.3},
])
def test_residual_only_arguments_are_rejected_for_other_methods(kwargs):
    X, y = _data(n=500)
    with pytest.raises(ValueError, match="apply only to method='residual'"):
        sample_rows(X, y, 0.1, **kwargs)
    with pytest.raises(ValueError, match="apply only to method='residual'"):
        sample_rows(X, y, 0.1, method="chronological", time=np.arange(500), **kwargs)
    with pytest.raises(ValueError, match="apply only to method='residual'"):
        sift.SmartSamplerConfig(**kwargs)


def test_top_level_exports():
    assert sift.sample_rows is sample_rows
    assert sift.RowSample is RowSample
