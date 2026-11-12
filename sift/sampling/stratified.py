"""Exact-size sampling within joint target and predictor bands."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    column = frame[name]
    if (
        not isinstance(column, pd.Series)
        or not is_numeric_dtype(column.dtype)
        or is_complex_dtype(column.dtype)
    ):
        raise TypeError(f"{name!r} must name one real numeric column.")
    return column.to_numpy(dtype=np.float64, na_value=np.nan)


def _bands(values: np.ndarray, quantiles: Sequence[float], *, separate_zero=False) -> np.ndarray:
    observed = np.isfinite(values)
    result = np.full(len(values), -1, dtype=np.intp)
    if observed.any():
        finite = values[observed]
        cuts = np.quantile(finite, quantiles, method="inverted_cdf")
        if separate_zero and finite.min() == 0:
            cuts = np.r_[0, cuts]
        result[observed] = np.searchsorted(np.unique(cuts), finite, side="left")
    return result


def stratified_sample(
    frame: pd.DataFrame,
    n_samples: int,
    *,
    target_col: str,
    balance_cols: Sequence[str] = (),
    random_state: int | None = None,
) -> np.ndarray:
    """Draw training rows proportionally within target and predictor bands.

    Pass only eligible training rows with finite numeric targets. The target
    bands use the 25th, 50th, 75th, 90th and 99th percentiles, and separate
    zeros for nonnegative targets. Supply up to two numeric training
    predictors in `balance_cols` to split each target band by predictor
    quartiles. Ties stay together and nonfinite predictor values have their
    own band.

    Draw without replacement within each joint cell. Each cell and each
    target band receives the floor or ceiling of its proportional share.
    The draw has exactly `n_samples` rows with equal marginal inclusion
    probabilities, so no reweighting is needed for a row-weighted objective.

    Returns sorted integer positions relative to `frame`, usable with
    `frame.iloc`. DataFrame index labels are ignored.
    """
    if not isinstance(n_samples, Integral) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer.")
    n_samples = int(n_samples)
    n_rows = len(frame)
    if not 0 <= n_samples <= n_rows:
        raise ValueError("n_samples must be between 0 and the number of rows.")
    if isinstance(balance_cols, str):
        raise TypeError("balance_cols must be a sequence of column names.")
    columns = tuple(balance_cols)
    if len(columns) > 2 or len(set(columns)) != len(columns) or target_col in columns:
        raise ValueError("Pass at most two distinct balance columns other than the target.")

    target = _numeric_column(frame, target_col)
    if not np.isfinite(target).all():
        raise ValueError("target_col must contain only finite training labels.")
    predictors = [_numeric_column(frame, name) for name in columns]
    if n_samples == 0 or n_samples == n_rows:
        return np.arange(n_samples, dtype=np.intp)

    bands = [_bands(target, (0.25, 0.5, 0.75, 0.90, 0.99), separate_zero=True)]
    bands.extend(_bands(values, (0.25, 0.5, 0.75)) for values in predictors)
    order = np.lexsort(bands[::-1])
    ordered_bands = np.asarray(bands)[:, order]
    boundaries = np.flatnonzero(np.any(ordered_bands[:, 1:] != ordered_bands[:, :-1], axis=0)) + 1

    rng = np.random.default_rng(random_state)
    offset = int(rng.integers(n_rows))
    selected = []
    start = 0
    for end in [*boundaries.tolist(), n_rows]:
        # Difference of rounded cumulative quotas. Target bands are
        # contiguous, so the rounding bound applies to their totals too.
        quota = (end * n_samples + offset) // n_rows - (start * n_samples + offset) // n_rows
        if quota:
            selected.append(rng.choice(order[start:end], size=quota, replace=False))
        start = end
    return np.sort(np.concatenate(selected))
