"""Purged and embargoed time-series cross-validators.

These splitters are additive sklearn-compatible CV objects. They do not
change SIFT's existing holdout helpers or default split routing.
Timestamps are supplied on each ``split`` call so the same instance can be
reused on a later fold-local or nested row subset. sklearn 1.3
``GridSearchCV`` / ``cross_val_score`` call ``split(X, y, groups)`` and
cannot route ``time``; pass ``time=`` (and ``event_end=``) on a direct
``split`` call or precompute the index pairs. On sklearn >= 1.4 with
``enable_metadata_routing=True``, ``set_split_request(time=True)`` plus
``params={'time': ...}`` routes correctly; grouped splitters also consume
``groups`` by default. Unrequested routed metadata is rejected, not ignored.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import indexable


_FORWARD = "forward"
_PURGED_KFOLD = "purged_kfold"
_VALID_MODES = (_FORWARD, _PURGED_KFOLD)


def _n_samples(X) -> int:
    (X_idx,) = indexable(X)
    if hasattr(X_idx, "shape") and len(getattr(X_idx, "shape", ())) >= 1:
        return int(X_idx.shape[0])
    return int(len(X_idx))


def _as_1d(values, n_rows: int, *, name: str, role: str = "time") -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required")
    if isinstance(values, pd.Series):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"{name} has {int(arr.shape[0])} rows but X has {n_rows}")
    try:
        missing = np.asarray(pd.isna(arr), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must not contain missing values") from exc
    if missing.any():
        raise ValueError(f"{name} must not contain missing values")
    if role == "time" and arr.dtype.kind == "f" and int(arr.dtype.itemsize) < 8:
        arr = np.asarray(arr, dtype=np.float64)
    return arr


def _is_zero_embargo(embargo) -> bool:
    if embargo is None:
        return True
    if isinstance(embargo, (bool, np.bool_)):
        raise TypeError("embargo must be a duration, not a boolean")
    if isinstance(embargo, (np.timedelta64, pd.Timedelta, timedelta)):
        delta = pd.Timedelta(embargo)
        if delta.value < 0:
            raise ValueError("embargo must be non-negative")
        return delta.value == 0
    if isinstance(embargo, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(embargo)):
            raise ValueError("embargo must be finite")
        if float(embargo) < 0.0:
            raise ValueError("embargo must be non-negative")
        return float(embargo) == 0.0
    raise TypeError(
        "embargo must be 0, a numeric duration matching numeric timestamps, "
        "or a timedelta matching datetime timestamps"
    )


def _embargo_threshold(val_start_min, embargo):
    if _is_zero_embargo(embargo):
        return None
    time_dtype = getattr(val_start_min, "dtype", None)
    kind = getattr(time_dtype, "kind", None)
    if kind == "M":
        if not isinstance(embargo, (np.timedelta64, pd.Timedelta, timedelta)):
            raise TypeError(
                "datetime timestamps require a timedelta embargo; a numeric "
                "embargo has no meaning on a datetime timeline"
            )
        return val_start_min - np.timedelta64(int(pd.Timedelta(embargo).value), "ns")
    if kind in {"i", "u"}:
        _require_integer_embargo(embargo)
        # Python int so a fractional event_end can compare without wrapping
        # or recasting the integer timeline.
        return _python_int(val_start_min) - _python_int(embargo)
    if kind == "f":
        if not isinstance(embargo, (int, float, np.integer, np.floating)):
            raise TypeError("float timestamps require a numeric embargo")
        return val_start_min - np.asarray(embargo, dtype=np.float64)
    try:
        return val_start_min - embargo
    except TypeError as exc:
        raise TypeError(
            "embargo dtype is incompatible with the timestamp dtype"
        ) from exc


def _require_integer_embargo(embargo) -> None:
    if isinstance(embargo, (np.timedelta64, pd.Timedelta, timedelta)):
        raise TypeError(
            "integer timestamps require a non-negative integer embargo so "
            "the timeline is not cast to float"
        )
    if not isinstance(embargo, (int, np.integer)) or isinstance(
        embargo, (bool, np.bool_)
    ):
        raise TypeError(
            "integer timestamps require a non-negative integer embargo so "
            "the timeline is not cast to float"
        )


def _python_int(value) -> int:
    arr = np.asarray(value)
    return int(arr.reshape(()).item())


def _int_lt_cutoff(values: np.ndarray, cutoff: int) -> np.ndarray:
    info = np.iinfo(values.dtype)
    if cutoff > int(info.max):
        return np.ones(values.shape, dtype=bool)
    if cutoff <= int(info.min):
        return np.zeros(values.shape, dtype=bool)
    return values < np.asarray(cutoff, dtype=values.dtype)


def _int_gt_cutoff(values: np.ndarray, cutoff: int) -> np.ndarray:
    info = np.iinfo(values.dtype)
    if cutoff >= int(info.max):
        return np.zeros(values.shape, dtype=bool)
    if cutoff < int(info.min):
        return np.ones(values.shape, dtype=bool)
    return values > np.asarray(cutoff, dtype=values.dtype)


def _integer_dtype_kind(value) -> bool:
    kind = getattr(getattr(value, "dtype", None), "kind", None)
    return kind in {"i", "u"}


def _past_integer_cutoff(val_start_min, embargo) -> int:
    extra = _python_int(embargo)
    if _integer_dtype_kind(val_start_min):
        return _python_int(val_start_min) - extra
    return int(np.ceil(float(np.asarray(val_start_min).reshape(()).item()) - extra))


def _future_integer_cutoff(val_end_max, embargo) -> int:
    extra = _python_int(embargo)
    if _integer_dtype_kind(val_end_max):
        return _python_int(val_end_max) + extra
    return int(np.floor(float(np.asarray(val_end_max).reshape(()).item()) + extra))


def _unique_time_ids(start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(start.shape[0])
    try:
        order = np.argsort(start, kind="mergesort")
        ordered = start[order]
        if start.dtype.kind == "O":
            for previous, current in zip(ordered[:-1], ordered[1:]):
                if bool(current < previous):
                    raise TypeError("time values are not monotonically orderable")
        elif np.asarray(ordered[1:] < ordered[:-1], dtype=bool).any():
            raise TypeError("time values are not monotonically orderable")
        if start.dtype.kind == "O":
            change = np.ones(n, dtype=bool)
            for i in range(1, n):
                change[i] = bool(ordered[i] != ordered[i - 1])
        else:
            change = np.ones(n, dtype=bool)
            if n > 1:
                change[1:] = np.asarray(ordered[1:] != ordered[:-1], dtype=bool)
    except (TypeError, ValueError) as exc:
        raise TypeError("time values must be orderable") from exc
    uniq = ordered[change]
    ordered_ids = np.cumsum(change.astype(np.int64)) - 1
    time_id = np.empty(n, dtype=np.int64)
    time_id[order] = ordered_ids
    return time_id, uniq


def _extreme(values, *, op):
    extreme = values[0]
    for value in values[1:]:
        if bool(op(value, extreme)):
            extreme = value
    return extreme


def _purge_mask(train_start, train_end, val_start, val_end) -> np.ndarray:
    """Closed overlap for a contiguous validation timestamp block.

    Candidate training starts lie entirely before or after the validation
    start-time block, so pairwise overlap reduces to
    ``train_end < min(val_start)`` or ``train_start > max(val_end)``.
    """
    if train_start.size == 0 or val_start.size == 0:
        return np.ones(train_start.shape[0], dtype=bool)
    val_start_min = _extreme(val_start, op=lambda a, b: a < b)
    val_end_max = _extreme(val_end, op=lambda a, b: a > b)
    return (train_end < val_start_min) | (train_start > val_end_max)


def _validate_mode(mode: str) -> str:
    if mode not in _VALID_MODES:
        raise ValueError(
            "mode must be 'forward' (chronological expanding windows) or "
            f"'purged_kfold' (opt-in bidirectional purged CV); got {mode!r}"
        )
    return str(mode)


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Forward purged/embargoed time-series split on explicit timestamps.

    Folds are expanding windows over *distinct start timestamps*, not raw
    row counts. Tied timestamps are one boundary unit and are never cut
    apart. Training information intervals are then purged of any closed
    overlap with validation intervals. ``embargo`` is an extra past-side
    duration dropped from train immediately before validation; it is not
    sklearn ``TimeSeriesSplit(gap=)`` sample skipping.

    Parameters
    ----------
    n_splits : int, default 5
        Number of train/validation folds. Must be at least 2.
    max_train_size : int or None, default None
        Optional cap on the number of distinct *training* timestamps kept
        in the candidate window before purge/embargo. Forward mode keeps
        the most recent eligible timestamps. ``purged_kfold`` keeps the
        ``max_train_size`` unique-time *indices* nearest the validation
        block (index distance, not elapsed time). ``None`` keeps every
        eligible training timestamp.
    test_size : int or None, default None
        Distinct timestamps in each validation block. In ``forward`` mode,
        ``None`` uses ``n_unique // (n_splits + 1)``, the sklearn
        ``TimeSeriesSplit`` default applied to unique times. In
        ``purged_kfold``, ``None`` partitions the unique timeline into
        ``n_splits`` contiguous blocks; an explicit value is each block's
        width, laid out sequentially from the earliest timestamp, and
        leftover later timestamps stay in train.
    embargo : 0, number, or timedelta, default 0
        Extra exclusion in the same domain as ``time``. ``0`` is
        purge-only. A datetime ``time`` requires a timedelta embargo;
        integer ``time`` requires an integer embargo. Forward mode only
        embargoes the past side of validation. ``purged_kfold`` keeps a
        row if it lies wholly before *or* wholly after the embargoed
        validation window.
    mode : {'forward', 'purged_kfold'}, default 'forward'
        ``'forward'`` is chronological: train timestamps are strictly
        before validation. ``'purged_kfold'`` is opt-in bidirectional
        purged CV: train may include future non-overlapping rows, with
        embargo on both sides of the validation window. It is not
        forward-only.

    Notes
    -----
    Call ``split(X, time=..., event_end=...)``. ``y`` is ignored. ``groups``
    is rejected here; use ``GroupPurgedTimeSeriesSplit``. sklearn 1.3
    drivers cannot route ``time``; pass it to ``split`` or precompute
    pairs. On sklearn >= 1.4 with ``enable_metadata_routing=True``,
    ``set_split_request(time=True)`` plus ``params={'time': ...}`` routes
    ``time``. Unrequested routed metadata is rejected.

    Each row's information interval is ``[time, event_end]`` closed on both
    ends. When ``event_end`` is omitted, the row is a point observation
    ``[time, time]``. Exact boundary equality is overlap and is purged.

    See Also
    --------
    GroupPurgedTimeSeriesSplit : Same chronology/interval rules with disjoint groups.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import PurgedTimeSeriesSplit
    >>> time = np.arange(12)
    >>> X = np.zeros((12, 1))
    >>> cv = PurgedTimeSeriesSplit(n_splits=3)
    >>> cv.get_n_splits()
    3
    >>> [(tr.tolist(), va.tolist()) for tr, va in cv.split(X, time=time)]
    [([0, 1, 2], [3, 4, 5]), ([0, 1, 2, 3, 4, 5], [6, 7, 8]), ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11])]
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        embargo=0,
        mode: str = _FORWARD,
    ):
        if not isinstance(n_splits, (int, np.integer)) or isinstance(
            n_splits, (bool, np.bool_)
        ):
            raise TypeError("n_splits must be an integer")
        n_splits = int(n_splits)
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if max_train_size is not None:
            if not isinstance(max_train_size, (int, np.integer)) or isinstance(
                max_train_size, (bool, np.bool_)
            ):
                raise TypeError("max_train_size must be an integer or None")
            max_train_size = int(max_train_size)
            if max_train_size < 1:
                raise ValueError("max_train_size must be positive when given")
        if test_size is not None:
            if not isinstance(test_size, (int, np.integer)) or isinstance(
                test_size, (bool, np.bool_)
            ):
                raise TypeError("test_size must be an integer or None")
            test_size = int(test_size)
            if test_size < 1:
                raise ValueError("test_size must be positive when given")
        _is_zero_embargo(embargo)
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.embargo = embargo
        self.mode = _validate_mode(mode)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the configured fold count.

        Parameters
        ----------
        X, y, groups : object
            Ignored; exist for sklearn compatibility.

        Returns
        -------
        n_splits : int
            The ``n_splits`` constructor value. Empty folds raise from
            ``split``, they are not dropped here.
        """
        return int(self.n_splits)

    def split(self, X, y=None, groups=None, *, time=None, event_end=None):
        """Yield original-row train and validation indices.

        Parameters
        ----------
        X : array-like
            Feature matrix or any indexable with a row count.
        y : array-like or None, default None
            Ignored. Fold construction does not read the target.
        groups : array-like or None, default None
            Rejected on this class. Use ``GroupPurgedTimeSeriesSplit``.
        time : array-like
            Per-row start timestamps, keyword-only, aligned to original
            rows of this ``X``. Numeric or datetime64. Not stored on the
            instance.
        event_end : array-like or None, default None
            Optional per-row information-interval ends, same length and
            domain as ``time``. ``None`` means point observations.

        Yields
        ------
        train : ndarray of int64
            Training indices into the original row order of ``X``.
        test : ndarray of int64
            Validation indices into the original row order of ``X``.
        """
        if groups is not None:
            raise ValueError(
                "PurgedTimeSeriesSplit does not accept groups; "
                "use GroupPurgedTimeSeriesSplit"
            )
        yield from self._iter_splits(
            X, groups=None, time=time, event_end=event_end, require_groups=False
        )

    def _iter_splits(self, X, *, groups, time, event_end, require_groups: bool):
        n = _n_samples(X)
        start = _as_1d(time, n, name="time")
        if event_end is None:
            end = start
        else:
            end = _as_1d(event_end, n, name="event_end")
            try:
                inverted = np.asarray(end < start, dtype=bool)
            except TypeError as exc:
                raise TypeError(
                    "event_end must be comparable with time and use the same domain"
                ) from exc
            if inverted.any():
                raise ValueError("event_end must be at or after time on every row")
        if require_groups:
            group_arr = _as_1d(groups, n, name="groups", role="groups")
        elif groups is not None:
            raise ValueError(
                "PurgedTimeSeriesSplit does not accept groups; "
                "use GroupPurgedTimeSeriesSplit"
            )
        else:
            group_arr = None

        time_id, uniq = _unique_time_ids(start)
        n_unique = int(uniq.shape[0])
        n_splits = int(self.n_splits)
        mode = _validate_mode(self.mode)
        if mode == _FORWARD:
            test_size = (
                int(self.test_size)
                if self.test_size is not None
                else n_unique // (n_splits + 1)
            )
            if test_size < 1:
                raise ValueError(
                    "Not enough distinct timestamps to build a validation block; "
                    f"got {n_unique} distinct times for n_splits={n_splits}"
                )
            if n_unique - test_size * n_splits <= 0:
                raise ValueError(
                    f"Too many splits={n_splits} for distinct timestamps="
                    f"{n_unique} with test_size={test_size}"
                )
            val_id_ranges = [
                (n_unique - (n_splits - i) * test_size, test_size)
                for i in range(n_splits)
            ]
        else:
            if n_unique < n_splits:
                raise ValueError(
                    f"purged_kfold requires at least n_splits distinct timestamps; "
                    f"got {n_unique} for n_splits={n_splits}"
                )
            if self.test_size is None:
                val_id_ranges = _contiguous_val_ranges(n_unique, n_splits)
            else:
                test_size = int(self.test_size)
                if test_size * n_splits > n_unique:
                    raise ValueError(
                        f"purged_kfold test_size={test_size} cannot fill "
                        f"{n_splits} validation blocks from {n_unique} "
                        "distinct timestamps"
                    )
                val_id_ranges = [
                    (i * test_size, test_size) for i in range(n_splits)
                ]

        rows = np.arange(n, dtype=np.int64)
        for fold_i, (val_start_id, val_width) in enumerate(val_id_ranges):
            val_ids = np.arange(val_start_id, val_start_id + val_width, dtype=np.int64)
            if mode == _FORWARD:
                train_ids = np.arange(0, val_start_id, dtype=np.int64)
                if self.max_train_size is not None and train_ids.size > self.max_train_size:
                    train_ids = train_ids[-int(self.max_train_size) :]
            else:
                train_ids = np.setdiff1d(
                    np.arange(n_unique, dtype=np.int64), val_ids, assume_unique=True
                )
                if self.max_train_size is not None and train_ids.size > self.max_train_size:
                    # Unique-time *index* distance, not elapsed-time ranking.
                    dist = np.minimum(
                        np.abs(train_ids - val_ids[0]),
                        np.abs(train_ids - val_ids[-1]),
                    )
                    keep = np.argsort(dist, kind="mergesort")[: int(self.max_train_size)]
                    train_ids = np.sort(train_ids[keep], kind="mergesort")

            val_mask = np.isin(time_id, val_ids)
            train_mask = np.isin(time_id, train_ids)
            val_idx = rows[val_mask]
            cand_idx = rows[train_mask]
            if val_idx.size == 0:
                raise ValueError(
                    f"validation fold {fold_i} is empty after timestamp blocking"
                )
            keep = _purge_mask(
                start[cand_idx], end[cand_idx], start[val_idx], end[val_idx]
            )
            train_idx = cand_idx[keep]
            train_idx = _apply_embargo(
                train_idx,
                start,
                end,
                start[val_idx],
                end[val_idx],
                self.embargo,
                mode=mode,
            )
            if group_arr is not None:
                val_groups = group_arr[val_idx]
                train_idx = train_idx[~np.isin(group_arr[train_idx], val_groups)]
            if train_idx.size == 0:
                raise ValueError(
                    f"training fold {fold_i} is empty after purge, embargo, "
                    "or group exclusion"
                )
            if mode == _FORWARD and np.max(start[train_idx]) >= np.min(start[val_idx]):
                raise ValueError(
                    "forward mode produced a training timestamp at or after "
                    "validation; this is a chronology violation"
                )
            yield (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64),
            )


def _contiguous_val_ranges(n_unique: int, n_splits: int) -> list[tuple[int, int]]:
    base, extra = divmod(n_unique, n_splits)
    if base < 1:
        raise ValueError(
            f"purged_kfold cannot form {n_splits} validation blocks from "
            f"{n_unique} distinct timestamps"
        )
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n_splits):
        width = base + (1 if i < extra else 0)
        if start == 0 and width == n_unique:
            raise ValueError(
                "purged_kfold validation block would consume every timestamp, "
                "leaving an empty training candidate set"
            )
        ranges.append((start, width))
        start += width
    return ranges


def _apply_embargo(
    train_idx: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    val_start: np.ndarray,
    val_end: np.ndarray,
    embargo,
    *,
    mode: str,
) -> np.ndarray:
    if _is_zero_embargo(embargo) or train_idx.size == 0:
        return train_idx
    val_start_min = val_start[0]
    for value in val_start[1:]:
        if bool(value < val_start_min):
            val_start_min = value
    val_end_max = val_end[0]
    for value in val_end[1:]:
        if bool(value > val_end_max):
            val_end_max = value
    past_ok = _keep_before_embargo(end[train_idx], val_start_min, embargo)
    if mode != _PURGED_KFOLD:
        return train_idx[past_ok]
    future_ok = _keep_after_embargo(start[train_idx], val_end_max, embargo)
    return train_idx[past_ok | future_ok]


def _keep_before_embargo(end, val_start_min, embargo) -> np.ndarray:
    kind = getattr(getattr(end, "dtype", None), "kind", None)
    if kind in {"i", "u"}:
        _require_integer_embargo(embargo)
        return _int_lt_cutoff(end, _past_integer_cutoff(val_start_min, embargo))
    past_cut = _embargo_threshold(val_start_min, embargo)
    return end < past_cut


def _keep_after_embargo(start, val_end_max, embargo) -> np.ndarray:
    kind = getattr(getattr(start, "dtype", None), "kind", None)
    if kind in {"i", "u"}:
        _require_integer_embargo(embargo)
        return _int_gt_cutoff(start, _future_integer_cutoff(val_end_max, embargo))
    if kind == "M":
        if not isinstance(embargo, (np.timedelta64, pd.Timedelta, timedelta)):
            raise TypeError(
                "datetime timestamps require a timedelta embargo; a numeric "
                "embargo has no meaning on a datetime timeline"
            )
        future_cut = val_end_max + np.timedelta64(
            int(pd.Timedelta(embargo).value), "ns"
        )
        return start > future_cut
    if kind == "f":
        if not isinstance(embargo, (int, float, np.integer, np.floating)):
            raise TypeError("float timestamps require a numeric embargo")
        return start > (val_end_max + np.asarray(embargo, dtype=np.float64))
    try:
        return start > (val_end_max + embargo)
    except TypeError as exc:
        raise TypeError(
            "embargo dtype is incompatible with the timestamp dtype"
        ) from exc


class GroupPurgedTimeSeriesSplit(PurgedTimeSeriesSplit):
    """Purged time-series split that also holds out validation groups.

    Chronology, closed-interval purge, embargo, tied timestamps, and
    ``mode`` follow ``PurgedTimeSeriesSplit``. Groups are identities, not
    a time axis: a group that appears in validation is excluded from
    training entirely.

    Parameters
    ----------
    n_splits : int, default 5
        Number of train/validation folds. Must be at least 2.
    max_train_size : int or None, default None
        Optional cap on distinct training timestamps before purge/embargo.
    test_size : int or None, default None
        Distinct timestamps in each validation block. See
        ``PurgedTimeSeriesSplit``.
    embargo : 0, number, or timedelta, default 0
        Extra exclusion duration in the same domain as ``time``.
    mode : {'forward', 'purged_kfold'}, default 'forward'
        Chronological expanding windows, or opt-in bidirectional purged CV.

    Notes
    -----
    ``split`` requires ``groups`` (sklearn's third argument) and keyword-only
    ``time``. ``y`` is ignored. On sklearn >= 1.4 with metadata routing
    enabled, ``groups`` is consumed by default and ``time`` is requested
    with ``set_split_request(time=True)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import GroupPurgedTimeSeriesSplit
    >>> time = np.arange(8)
    >>> groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    >>> X = np.zeros((8, 1))
    >>> cv = GroupPurgedTimeSeriesSplit(n_splits=2, test_size=2)
    >>> [(tr.tolist(), va.tolist()) for tr, va in cv.split(X, groups=groups, time=time)]
    [([0, 1, 2, 3], [4, 5]), ([0, 1, 2, 3, 4, 5], [6, 7])]
    """

    __metadata_request__split = {"groups": True}

    def split(self, X, y=None, groups=None, *, time=None, event_end=None):
        """Yield original-row train and validation indices.

        Parameters
        ----------
        X : array-like
            Feature matrix or any indexable with a row count.
        y : array-like or None, default None
            Ignored.
        groups : array-like
            Per-row group identities, aligned to original rows. Required.
        time : array-like
            Per-row start timestamps, keyword-only.
        event_end : array-like or None, default None
            Optional information-interval ends.

        Yields
        ------
        train : ndarray of int64
            Training indices with no validation group and no interval overlap.
        test : ndarray of int64
            Validation indices.
        """
        if groups is None:
            raise ValueError("GroupPurgedTimeSeriesSplit requires groups")
        yield from self._iter_splits(
            X, groups=groups, time=time, event_end=event_end, require_groups=True
        )
