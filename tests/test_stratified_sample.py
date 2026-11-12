import numpy as np
import pandas as pd
import pytest

from sift.sampling import stratified_sample


def test_exact_sample_preserves_zero_and_predictor_bands():
    # Each joint cell contains ten observations. Ties must stay together.
    frame = pd.DataFrame({
        "y": np.repeat([0.0, 1.0, 9.0], 20),
        "x": np.tile(np.repeat([3.0, np.nan], 10), 3),
    }).sample(frac=1, random_state=11)
    frame.index = [7] * len(frame)
    kwargs = {"target_col": "y", "balance_cols": ["x"], "random_state": 5}
    positions = stratified_sample(frame, 6, **kwargs)
    np.testing.assert_array_equal(positions, stratified_sample(frame, 6, **kwargs))
    assert len(positions) == 6
    assert np.all(np.diff(positions) > 0)
    counts = frame.iloc[positions].groupby(["y", "x"], dropna=False).size()
    assert len(counts) == 6
    assert (counts == 1).all()


def test_rare_cells_when_budget_is_smaller_than_cell_count():
    frame = pd.DataFrame({"y": np.arange(101), "x": np.arange(101) % 5})
    for seed in range(8):
        positions = stratified_sample(
            frame, 3, target_col="y", balance_cols=["x"], random_state=seed,
        )
        assert len(np.unique(positions)) == 3


def test_constant_labels_and_invalid_targets():
    frame = pd.DataFrame({"y": [1, 1, 1], "x": [np.nan] * 3})
    kwargs = {"target_col": "y", "balance_cols": ["x"]}
    assert stratified_sample(frame, 0, **kwargs).size == 0
    np.testing.assert_array_equal(stratified_sample(frame, 3, **kwargs), np.arange(3))
    assert len(stratified_sample(frame, 1, **kwargs)) == 1
    with pytest.raises(ValueError, match="finite"):
        stratified_sample(frame.assign(y=[0, np.nan, 2]), 1, **kwargs)
    with pytest.raises(ValueError, match="distinct"):
        stratified_sample(frame, 1, target_col="y", balance_cols=["y"])
