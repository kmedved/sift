"""Public contracts for purged/embargoed time-series splitters."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from sklearn import config_context
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

from sift import GroupPurgedTimeSeriesSplit, PurgedTimeSeriesSplit

SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])


def test_forward_split_expanding_unique_times_and_get_n_splits():
    time = np.arange(12)
    X = np.zeros((12, 2))
    cv = PurgedTimeSeriesSplit(n_splits=3)
    assert cv.get_n_splits() == 3
    folds = list(cv.split(X, time=time))
    assert [(tr.tolist(), va.tolist()) for tr, va in folds] == [
        ([0, 1, 2], [3, 4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7, 8]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]),
    ]
    for train, val in folds:
        assert train.dtype == np.int64
        assert val.dtype == np.int64
        assert np.max(time[train]) < np.min(time[val])
        assert np.intersect1d(train, val).size == 0


def test_unsorted_input_maps_back_to_original_positions():
    time = np.array([4, 0, 2, 1, 3, 5])
    X = np.zeros((6, 1))
    cv = PurgedTimeSeriesSplit(n_splits=2, test_size=2)
    folds = list(cv.split(X, time=time))
    assert len(folds) == 2
    train0, val0 = folds[0]
    assert sorted(time[val0].tolist()) == [2, 3]
    assert sorted(time[train0].tolist()) == [0, 1]
    assert set(train0.tolist()) == {1, 3}


def test_tied_timestamps_are_one_boundary_unit():
    time = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    X = np.zeros((8, 1))
    cv = PurgedTimeSeriesSplit(n_splits=2, test_size=1)
    folds = list(cv.split(X, time=time))
    for train, val in folds:
        val_times = set(time[val].tolist())
        assert len(val_times) == 1
        assert set(time[train]).isdisjoint(val_times)
        assert np.max(time[train]) < np.min(time[val])


def test_closed_interval_purge_drops_exact_boundary_and_horizon():
    start = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    end = start + 2
    X = np.zeros((8, 1))
    cv = PurgedTimeSeriesSplit(n_splits=2, test_size=2)
    train, val = list(cv.split(X, time=start, event_end=end))[0]
    assert val.tolist() == [4, 5]
    # val intervals [4,6] and [5,7]; train 3 has [3,5] overlapping 4.
    assert 3 not in train
    assert 2 not in train
    assert set(train.tolist()) <= {0, 1}


def test_embargo_is_past_side_duration_not_sample_gap():
    start = np.arange(10)
    X = np.zeros((10, 1))
    base = PurgedTimeSeriesSplit(n_splits=2, test_size=2, embargo=0)
    embargoed = PurgedTimeSeriesSplit(n_splits=2, test_size=2, embargo=2)
    tr0, va0 = list(base.split(X, time=start))[0]
    tr1, va1 = list(embargoed.split(X, time=start))[0]
    assert va0.tolist() == va1.tolist() == [6, 7]
    assert tr0.tolist() == [0, 1, 2, 3, 4, 5]
    assert tr1.tolist() == [0, 1, 2, 3]
    assert set(tr1).issubset(set(tr0))


def test_datetime_and_integer_precision_and_timedelta_embargo():
    time = np.arange("2020-01-01", "2020-01-13", dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )
    X = np.zeros((12, 1))
    cv = PurgedTimeSeriesSplit(n_splits=3, embargo=np.timedelta64(1, "D"))
    folds = list(cv.split(X, time=time))
    assert len(folds) == 3
    for train, val in folds:
        assert np.max(time[train]) < np.min(time[val])
    big = np.array(
        [10**15, 10**15 + 1, 10**15 + 2, 10**15 + 3, 10**15 + 4, 10**15 + 5],
        dtype=np.int64,
    )
    cv_i = PurgedTimeSeriesSplit(n_splits=2, test_size=2, embargo=1)
    tr, va = list(cv_i.split(np.zeros((6, 1)), time=big))[0]
    assert big[va].dtype == np.int64
    assert np.max(big[tr]) < np.min(big[va])
    t32 = np.arange(12, dtype=np.float32)
    t64 = np.arange(12, dtype=np.float64)
    a = list(PurgedTimeSeriesSplit(n_splits=3).split(np.zeros((12, 1)), time=t32))
    b = list(PurgedTimeSeriesSplit(n_splits=3).split(np.zeros((12, 1)), time=t64))
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)


def test_y_is_ignored_and_groups_rejected_on_ungrouped_splitter():
    time = np.arange(12)
    X = np.zeros((12, 1))
    y_a = np.arange(12)
    y_b = np.ones(12) * 99
    cv = PurgedTimeSeriesSplit(n_splits=3)
    a = list(cv.split(X, y=y_a, time=time))
    b = list(cv.split(X, y=y_b, time=time))
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)
    with pytest.raises(ValueError, match="does not accept groups"):
        list(cv.split(X, groups=np.zeros(12, dtype=int), time=time))
    with pytest.raises(ValueError, match="time is required"):
        list(cv.split(X))


def test_misaligned_missing_and_inverted_event_end_raise():
    X = np.zeros((6, 1))
    cv = PurgedTimeSeriesSplit(n_splits=2, test_size=2)
    with pytest.raises(ValueError, match="time has 5 rows"):
        list(cv.split(X, time=np.arange(5)))
    with pytest.raises(ValueError, match="missing"):
        list(cv.split(X, time=np.array([0, 1, np.nan, 3, 4, 5])))
    with pytest.raises(ValueError, match="event_end must be at or after time"):
        list(
            cv.split(
                X,
                time=np.arange(6),
                event_end=np.array([0, 1, 1, 2, 3, 2]),
            )
        )
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        PurgedTimeSeriesSplit(n_splits=1)


def test_purged_kfold_is_bidirectional_and_opt_in():
    time = np.arange(9)
    X = np.zeros((9, 1))
    cv = PurgedTimeSeriesSplit(n_splits=3, mode="purged_kfold")
    folds = list(cv.split(X, time=time))
    assert len(folds) == 3
    train0, val0 = folds[0]
    assert val0.tolist() == [0, 1, 2]
    assert np.min(time[train0]) > np.max(time[val0])
    train_last, val_last = folds[-1]
    assert np.max(time[train_last]) < np.min(time[val_last])
    with pytest.raises(ValueError, match="purged_kfold"):
        PurgedTimeSeriesSplit(n_splits=2, mode="rolling")


def test_empty_train_after_purge_raises():
    start = np.array([0, 1, 2, 3])
    end = np.array([10, 10, 10, 10])
    X = np.zeros((4, 1))
    cv = PurgedTimeSeriesSplit(n_splits=2, test_size=1)
    with pytest.raises(ValueError, match="training fold"):
        list(cv.split(X, time=start, event_end=end))


def test_group_variant_keeps_identities_disjoint_without_ordering_groups():
    time = np.arange(8)
    groups = np.array([3, 3, 1, 1, 2, 2, 0, 0])
    X = np.zeros((8, 1))
    cv = GroupPurgedTimeSeriesSplit(n_splits=2, test_size=2)
    folds = list(cv.split(X, groups=groups, time=time))
    assert [(tr.tolist(), va.tolist()) for tr, va in folds] == [
        ([0, 1, 2, 3], [4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7]),
    ]
    for train, val in folds:
        assert set(groups[train]).isdisjoint(set(groups[val]))
        assert np.max(time[train]) < np.min(time[val])
    with pytest.raises(ValueError, match="requires groups"):
        list(cv.split(X, time=time))
    shuffled_groups = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    tr, va = list(cv.split(X, groups=shuffled_groups, time=time))[0]
    assert set(shuffled_groups[tr]).isdisjoint(set(shuffled_groups[va]))
    assert GroupPurgedTimeSeriesSplit(n_splits=2, test_size=2).get_n_splits() == 2
    g32 = np.arange(8, dtype=np.float32)
    tr32, va32 = list(cv.split(X, groups=g32, time=time))[0]
    assert set(g32[tr32]).isdisjoint(set(g32[va32]))


def test_purged_kfold_embargo_keeps_both_sides():
    t = np.arange(12)
    X = np.zeros((12, 1))
    folds = list(
        PurgedTimeSeriesSplit(n_splits=3, mode="purged_kfold", embargo=1).split(
            X, time=t
        )
    )
    assert len(folds) == 3
    tr, va = folds[1]
    assert t[tr].min() < t[va].min() and t[tr].max() > t[va].max()
    lo, hi = t[va].min() - 1, t[va].max() + 1
    assert not ((t[tr] >= lo) & (t[tr] <= hi)).any()
    first_tr, first_va = folds[0]
    last_tr, last_va = folds[-1]
    assert t[first_tr].min() > t[first_va].max()
    assert t[last_tr].max() < t[last_va].min()
    days = np.arange("2020-01-01", "2020-01-13", dtype="datetime64[D]")
    dt_folds = list(
        PurgedTimeSeriesSplit(
            n_splits=3, mode="purged_kfold", embargo=np.timedelta64(1, "D")
        ).split(X, time=days)
    )
    tr_d, va_d = dt_folds[1]
    assert days[tr_d].min() < days[va_d].min()
    assert days[tr_d].max() > days[va_d].max()


def test_purged_kfold_test_size_and_minimal_folds():
    t = np.arange(12)
    X = np.zeros((12, 1))
    none_folds = list(PurgedTimeSeriesSplit(n_splits=3, mode="purged_kfold").split(X, time=t))
    sized = list(
        PurgedTimeSeriesSplit(n_splits=3, mode="purged_kfold", test_size=2).split(
            X, time=t
        )
    )
    assert [len(va) for _, va in none_folds] == [4, 4, 4]
    assert [len(va) for _, va in sized] == [2, 2, 2]
    tiny = np.arange(3)
    folds = list(
        PurgedTimeSeriesSplit(n_splits=3, mode="purged_kfold").split(
            np.zeros((3, 1)), time=tiny
        )
    )
    assert len(folds) == 3
    assert all(len(va) == 1 and len(tr) == 2 for tr, va in folds)


@pytest.mark.parametrize("dtype", [np.uint64, np.uint32, np.int64, np.int8])
def test_integer_embargo_cutoffs_do_not_wrap(dtype):
    t = np.arange(9, dtype=dtype)
    X = np.zeros((9, 1))
    cv = PurgedTimeSeriesSplit(n_splits=3, test_size=2, embargo=4)
    with pytest.raises(ValueError, match="training fold"):
        list(cv.split(X, time=t))
    wide = np.arange(10, dtype=dtype)
    cv_ok = PurgedTimeSeriesSplit(n_splits=2, test_size=1, embargo=1)
    tr, va = list(cv_ok.split(np.zeros((10, 1)), time=wide))[0]
    assert np.max(wide[tr].astype(np.int64)) < int(wide[va].min()) - 1


@pytest.mark.skipif(SKLEARN_VERSION < (1, 4), reason="CV metadata routing needs sklearn>=1.4")
def test_modern_metadata_routing_matches_precomputed_splits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2))
    y = rng.normal(size=12)
    t = np.arange(12)
    groups = np.repeat(np.arange(6), 2)
    pre = list(PurgedTimeSeriesSplit(n_splits=3).split(X, time=t))
    with config_context(enable_metadata_routing=True):
        cv = PurgedTimeSeriesSplit(n_splits=3)
        cv.set_split_request(time=True)
        routed = cross_validate(Ridge(), X, y, cv=cv, params={"time": t})
        direct = cross_validate(Ridge(), X, y, cv=pre)
        np.testing.assert_allclose(routed["test_score"], direct["test_score"])
        with pytest.raises(Exception, match="time"):
            cross_validate(Ridge(), X, y, cv=PurgedTimeSeriesSplit(n_splits=3), params={"time": t})
        gcv = GroupPurgedTimeSeriesSplit(n_splits=2, test_size=2)
        assert "groups" in str(gcv.get_metadata_routing())
        gcv.set_split_request(time=True)
        g_pre = list(gcv.split(X, groups=groups, time=t))
        g_routed = cross_validate(
            Ridge(), X, y, cv=gcv, params={"time": t, "groups": groups}
        )
        g_direct = cross_validate(Ridge(), X, y, cv=g_pre)
        np.testing.assert_allclose(g_routed["test_score"], g_direct["test_score"])


@pytest.mark.parametrize("mode", ["forward", "purged_kfold"])
def test_mixed_integer_start_float_event_end_embargo_matches_all_float(mode):
    t_int = np.arange(18, dtype=np.int64)
    end_mixed = t_int + 0.5
    t_float = np.arange(18, dtype=np.float64)
    end_float = t_float + 0.5
    X = np.zeros((18, 1))
    cv = PurgedTimeSeriesSplit(n_splits=3, embargo=1, mode=mode)
    mixed = list(cv.split(X, time=t_int, event_end=end_mixed))
    all_float = list(cv.split(X, time=t_float, event_end=end_float))
    assert len(mixed) == len(all_float) == 3
    for (tr_m, va_m), (tr_f, va_f) in zip(mixed, all_float):
        np.testing.assert_array_equal(tr_m, tr_f)
        np.testing.assert_array_equal(va_m, va_f)
