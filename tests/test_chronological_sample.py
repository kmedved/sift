import numpy as np
import pandas as pd
import pytest

from sift.sampling import chronological_sample


def test_exact_budget_and_group_coverage_on_shuffled_panel():
    frame = pd.DataFrame({
        "season": np.repeat([2020, 2021], [27, 24]),
        "player": np.repeat([1, 2, 3, 1, 2, 4], [10, 15, 2, 6, 17, 1]),
        "time": np.arange(51),
    }).sample(frac=1, random_state=12)
    frame.index = [7] * len(frame)  # Positions are independent of index labels.

    for seed in range(8):
        positions = chronological_sample(
            frame, 10, group_col="player", time_col="time",
            stratum_col="season", random_state=seed,
        )
        assert len(positions) == 10
        assert np.all(np.diff(positions) > 0)
        for keys in [["season"], ["season", "player"]]:
            population = frame.groupby(keys).size()
            counts = frame.iloc[positions].groupby(keys).size()
            error = counts.reindex(population.index, fill_value=0) - population * 10 / 51
            assert np.all(np.abs(error) < 1)


def test_circular_time_blocks_bound_gaps():
    frame = pd.DataFrame({"player": 1, "time": np.arange(50)}).sample(frac=1, random_state=7)
    positions = chronological_sample(
        frame, 10, group_col="player", time_col="time", random_state=4,
    )
    times = np.sort(frame.iloc[positions]["time"].to_numpy())
    gaps = np.diff(np.r_[times, times[0] + len(frame)])
    assert gaps.max() <= 9


def test_many_small_groups_and_missing_ids():
    frame = pd.DataFrame({"player": [None, None, "a", "b", "c", "d"], "time": range(6)})
    kwargs = {"group_col": "player", "time_col": "time", "random_state": 4}
    assert chronological_sample(frame, 0, **kwargs).size == 0
    np.testing.assert_array_equal(chronological_sample(frame, 6, **kwargs), np.arange(6))
    first = chronological_sample(frame, 2, **kwargs)
    np.testing.assert_array_equal(first, chronological_sample(frame, 2, **kwargs))
    assert len(np.unique(first)) == 2
    with pytest.raises(ValueError, match="number of rows"):
        chronological_sample(frame, 7, **kwargs)
    with pytest.raises(ValueError, match="missing"):
        chronological_sample(frame.assign(time=np.nan), 2, **kwargs)
