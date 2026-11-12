"""Exact-size sampling for grouped observations in time order."""

from numbers import Integral

import numpy as np
import pandas as pd


def chronological_sample(
    frame: pd.DataFrame,
    n_samples: int,
    *,
    group_col: str,
    time_col: str,
    stratum_col: str | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw training rows spread across each group's timeline.

    Pass only eligible training rows. When `stratum_col` is set (for example,
    to a season), allocate separately to each group within each stratum. The
    draw takes exactly `n_samples` distinct rows. Each stratum and each group
    within it receives the floor or ceiling of its proportional share, so
    small groups can receive zero rows. Missing group or stratum values form
    their own groups. Time values must be present and sortable.

    Within each group, randomly rotate its time-ordered rows and draw one row
    from each of its roughly equal blocks. Equal times use input row order.
    Every row has the same marginal inclusion probability, so use equal
    weights for a row-weighted training objective.

    Returns sorted integer positions relative to `frame`, usable with
    `frame.iloc`. DataFrame index labels are ignored.
    """
    if not isinstance(n_samples, Integral) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer.")
    n_samples = int(n_samples)
    n_rows = len(frame)
    if not 0 <= n_samples <= n_rows:
        raise ValueError("n_samples must be between 0 and the number of rows.")

    times = frame[time_col]
    if not isinstance(times, pd.Series) or times.isna().any():
        raise ValueError("time_col must name one column with no missing values.")
    groups, group_values = pd.factorize(frame[group_col], use_na_sentinel=False)
    strata = (
        pd.factorize(frame[stratum_col], use_na_sentinel=False)[0]
        if stratum_col is not None
        else np.zeros(n_rows, dtype=np.intp)
    )
    if n_samples == 0 or n_samples == n_rows:
        return np.arange(n_samples, dtype=np.intp)

    rng = np.random.default_rng(random_state)
    group_order = rng.permutation(len(group_values))[groups]
    order = pd.DataFrame(
        {"stratum": strata, "group": group_order, "time": times.array}
    ).sort_values(["stratum", "group", "time"]).index.to_numpy(dtype=np.intp)
    sorted_groups = groups[order]
    sorted_strata = strata[order]
    boundaries = np.flatnonzero(
        (sorted_groups[1:] != sorted_groups[:-1])
        | (sorted_strata[1:] != sorted_strata[:-1])
    ) + 1

    selected = np.empty(n_samples, dtype=np.intp)
    written = 0
    # One randomized remainder gives each group and each contiguous stratum a
    # proportional quota rounded up or down, with an exact global total.
    carry = int(rng.integers(n_rows))
    for positions in np.split(order, boundaries):
        size = len(positions)
        quota, carry = divmod(carry + n_samples * size, n_rows)
        if quota == 0:
            continue
        if quota == size:
            draw = positions
        else:
            edges = np.arange(quota + 1, dtype=np.int64) * size // quota
            slots = rng.integers(edges[:-1], edges[1:])
            draw = positions[(slots + rng.integers(size)) % size]
        selected[written:written + quota] = draw
        written += quota

    return np.sort(selected)
