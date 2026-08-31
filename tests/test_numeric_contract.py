from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sift import (
    BorutaSelector,
    StabilitySelector,
    build_cache,
    select_jmi,
    select_jmim,
    select_mrmr,
)
from sift._preprocess import validate_inputs


@pytest.mark.parametrize(
    "selector, kwargs",
    [
        (select_mrmr, {"estimator": "classic", "relevance": "f"}),
        (select_jmi, {"estimator": "r2", "relevance": "f"}),
        (select_jmim, {"estimator": "r2", "relevance": "f"}),
    ],
)
def test_classic_selectors_preserve_large_offset_signal(selector, kwargs):
    rng = np.random.default_rng(481)
    n = 160
    signal = np.linspace(-1.0, 1.0, n)
    y = signal + rng.normal(scale=0.02, size=n)
    X = np.column_stack([signal, rng.normal(size=(n, 2))])
    shifted = X.copy()
    shifted[:, 0] += 1e8

    baseline = selector(X, y, k=2, task="regression", subsample=None, verbose=False, **kwargs)
    actual = selector(
        shifted, y, k=2, task="regression", subsample=None, verbose=False, **kwargs
    )
    assert actual == baseline
    assert actual[0] == "x0"


@pytest.mark.parametrize(
    "column",
    [
        pd.date_range("2024-01-01", periods=4),
        pd.to_timedelta([0, 1, 2, 3], unit="D"),
    ],
    ids=["datetime64", "timedelta64"],
)
def test_validate_inputs_rejects_datetime_like_feature_columns(column):
    X = pd.DataFrame({"temporal": column, "legitimate_numeric": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="Datetime or timedelta"):
        validate_inputs(X, np.arange(4, dtype=np.float64), task="regression")


@pytest.mark.parametrize(
    "column",
    [
        pd.date_range("2024-01-01", periods=4),
        pd.to_timedelta([0, 1, 2, 3], unit="D"),
    ],
    ids=["datetime64", "timedelta64"],
)
def test_gaussian_cache_rejects_datetime_like_feature_columns(column):
    X = pd.DataFrame({"temporal": column, "legitimate_numeric": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="Datetime or timedelta"):
        build_cache(X, subsample=None)


@pytest.mark.parametrize("dtype", ["datetime64[D]", "timedelta64[D]"])
def test_numpy_datetime_like_feature_arrays_are_rejected_everywhere(dtype):
    X = np.arange(4).astype(dtype)[:, None]
    y = np.arange(4, dtype=np.float64)
    for call in (
        lambda: validate_inputs(X, y, task="regression"),
        lambda: build_cache(X, subsample=None),
        lambda: BorutaSelector(max_iter=1, verbose=False).fit(X, y),
        lambda: StabilitySelector(
            n_bootstrap=1,
            alpha=0.1,
            n_jobs=1,
            verbose=False,
        ).fit(X, y),
    ):
        with pytest.raises(ValueError, match="Datetime or timedelta"):
            call()


@pytest.mark.parametrize(
    "values",
    [
        [datetime(2024, 1, day) for day in range(1, 5)],
        [timedelta(days=day) for day in range(4)],
    ],
    ids=["python-datetime", "python-timedelta"],
)
def test_object_arrays_reject_datetime_like_feature_values(values):
    X = np.empty((4, 1), dtype=object)
    X[:, 0] = values
    y = np.arange(4, dtype=np.float64)

    for call in (
        lambda: validate_inputs(X, y, task="regression"),
        lambda: build_cache(X, subsample=None),
        lambda: BorutaSelector(max_iter=1, verbose=False).fit(X, y),
        lambda: StabilitySelector(
            n_bootstrap=1,
            alpha=0.1,
            n_jobs=1,
            verbose=False,
        ).fit(X, y),
    ):
        with pytest.raises(ValueError, match="Datetime or timedelta"):
            call()


@pytest.mark.parametrize("unit", ["ns", "s"])
def test_arrow_duration_feature_columns_are_rejected_everywhere(unit):
    pytest.importorskip("pyarrow")
    duration = pd.Series(
        pd.to_timedelta(np.arange(4), unit=unit),
        dtype=f"duration[{unit}][pyarrow]",
    )
    X = pd.DataFrame(
        {
            "temporal": duration,
            "legitimate_numeric": np.arange(4, dtype=np.float64),
        }
    )
    y = np.arange(4, dtype=np.float64)

    for call in (
        lambda: validate_inputs(X, y, task="regression"),
        lambda: build_cache(X, subsample=None),
        lambda: BorutaSelector(max_iter=1, verbose=False).fit(X, y),
        lambda: StabilitySelector(
            n_bootstrap=1,
            alpha=0.1,
            n_jobs=1,
            verbose=False,
        ).fit(X, y),
    ):
        with pytest.raises(ValueError, match="Datetime or timedelta"):
            call()


def test_validate_inputs_keeps_legitimate_numeric_features_float64():
    X = pd.DataFrame({"integer": [1, 2, 3], "float": [0.1, 0.2, 0.3]})
    X_arr, y_arr, names = validate_inputs(X, np.array([0.0, 1.0, 2.0]), "regression")
    assert X_arr.dtype == np.float64
    assert y_arr.dtype == np.float64
    assert names == ["integer", "float"]
