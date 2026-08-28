import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from sift.selection.auto_k_core import time_holdout_split
from sift.stability import StabilitySelector, _block_bootstrap_indices
from sift.sampling.stability import _stationary_block_sample


def test_block_bootstrap_indices_basic():
    n = 120
    groups = np.repeat(np.arange(6), 20)
    time = np.tile(np.arange(20), 6)
    splits = list(
        _block_bootstrap_indices(
            n=n,
            n_bootstrap=3,
            groups=groups,
            time=time,
            block_size=5,
            block_method="moving",
            random_state=0,
        )
    )

    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert train_idx.dtype == np.int64
        assert val_idx.dtype == np.int64
        assert train_idx.min() >= 0
        assert train_idx.max() < n
        assert val_idx.min() >= 0
        assert val_idx.max() < n


def test_stability_selector_block_bootstrap_runs():
    rng = np.random.default_rng(0)
    n, p = 120, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n) * 0.2
    groups = np.repeat(np.arange(6), 20)
    time = np.tile(np.arange(20), 6)

    selector = StabilitySelector(
        n_bootstrap=5,
        threshold=0.1,
        alpha=0.01,
        random_state=0,
        n_jobs=1,
        verbose=False,
        block_size=5,
        block_method="moving",
    )
    selector.fit(X, y, groups=groups, time=time)

    assert selector.n_features_selected_ > 0


def test_stationary_block_sample_wraps_right_edge():
    class FakeRng:
        def integers(self, low, high=None):
            return 4

        def geometric(self, p):
            return 3

    sample = _stationary_block_sample(np.arange(5), mean_block_size=2, n=5, rng=FakeRng())

    assert sample[:3] == [4, 0, 1]


@pytest.mark.parametrize(
    "time",
    [
        np.array([3, 1, 2, 1, 3, 2, 4]),
        np.array(["c", "a", "b", "a", "c", "b", "d"]),
        np.array(
            ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01",
             "2024-01-03", "2024-01-02", "2024-01-04"],
            dtype="datetime64[D]",
        ),
    ],
    ids=["integer", "string", "datetime64"],
)
def test_time_holdout_split_keeps_equal_timestamps_together(time):
    train, val = time_holdout_split(time, val_frac=0.35)

    assert np.all(time[train][:-1] <= time[train][1:])
    assert np.all(time[val][:-1] <= time[val][1:])
    assert set(time[train]).isdisjoint(set(time[val]))
    assert max(time[train]) < min(time[val])


def test_time_holdout_split_uses_smaller_equidistant_boundary():
    time = np.array([2, 1, 1, 2, 3, 3, 4, 4])

    train, val = time_holdout_split(time, val_frac=0.375)

    np.testing.assert_array_equal(train, np.array([1, 2, 0, 3]))
    np.testing.assert_array_equal(val, np.array([4, 5, 6, 7]))


@pytest.mark.parametrize("time_vals", [np.ones(4), np.array([1])])
def test_time_holdout_split_rejects_degenerate_time_vectors(time_vals):
    with pytest.raises(ValueError, match="time_holdout_split"):
        time_holdout_split(time_vals, val_frac=0.2)


def test_time_holdout_split_rejects_unorderable_timestamp_values():
    with pytest.raises(TypeError, match="orderable"):
        time_holdout_split(np.array([1, "a", 2], dtype=object), val_frac=0.2)


@pytest.mark.parametrize(
    "time_vals",
    [
        np.array([0.0, np.nan, 1.0]),
        np.array(["2024-01-01", "NaT", "2024-01-03"], dtype="datetime64[D]"),
        np.array([0, None, 1], dtype=object),
        np.array([0, pd.NA, 1], dtype=object),
    ],
    ids=["nan", "nat", "none", "pandas-na"],
)
def test_time_holdout_split_rejects_missing_timestamp_values(time_vals):
    with pytest.raises(ValueError, match="time values.*missing"):
        time_holdout_split(time_vals, val_frac=0.2)


def test_block_bootstrap_rejects_missing_time_values():
    groups = np.zeros(4, dtype=np.int64)
    time = np.array([0.0, np.nan, 1.0, 2.0])

    with pytest.raises(ValueError, match="time values.*missing"):
        list(
            _block_bootstrap_indices(
                n=4,
                n_bootstrap=1,
                groups=groups,
                time=time,
                block_size=2,
                min_oob=0,
                random_state=0,
            )
        )


def test_block_bootstrap_sample_frac_sets_rounded_panel_draw_budget():
    groups = np.repeat(np.arange(3), [5, 11, 20])
    time = np.concatenate([np.arange(size) for size in [5, 11, 20]])

    half = list(
        _block_bootstrap_indices(
            len(groups), 2, groups, time, block_size=3, random_state=17,
            min_oob=0, sample_frac=0.5,
        )
    )
    full = list(
        _block_bootstrap_indices(
            len(groups), 2, groups, time, block_size=3, random_state=17,
            min_oob=0, sample_frac=1.0,
        )
    )

    assert all(len(train) == 18 for train, _ in half)
    assert all(len(train) == 36 for train, _ in full)
    for train, val in half + full:
        assert np.intersect1d(np.unique(train), val).size == 0
        assert np.all((train >= 0) & (train < len(groups)))
        assert np.all((val >= 0) & (val < len(groups)))
    repeat = list(
        _block_bootstrap_indices(
            len(groups), 2, groups, time, block_size=3, random_state=17,
            min_oob=0, sample_frac=0.5,
        )
    )
    assert all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(half, repeat))


def test_stability_selector_get_feature_names_out_checks_fitted_input_names():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(size=40) * 0.1
    selector = StabilitySelector(
        n_bootstrap=2, sample_frac=0.5, threshold=0.0, alpha=0.1,
        n_jobs=1, random_state=0, verbose=False,
    )
    with pytest.raises(NotFittedError):
        selector.get_feature_names_out()
    selector.fit(X, y)
    assert selector.get_feature_names_out().tolist() == ["x0", "x1", "x2"]
    assert selector.get_feature_names_out(["x0", "x1", "x2"]).tolist() == ["x0", "x1", "x2"]
    with pytest.raises(ValueError, match="input_features"):
        selector.get_feature_names_out(["wrong", "x1", "x2"])
