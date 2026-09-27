"""Closure fixes: ordinal level order, cross-dtype transform, tiny weights."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from sift import MRMRSelector, select_cefsplus
from sift._preprocess import CatEncoding, OneHotBlockEncoder, validate_inputs
from sift._unsupervised_cat import UnsupervisedCatEncoder


def _codes(values, *, col="c", weights=None):
    """Ordinal codes for a raw column, as a plain list of floats."""
    frame = pd.DataFrame({col: values})
    enc = UnsupervisedCatEncoder([col], method="ordinal").fit(frame, sample_weight=weights)
    return enc.transform(frame)[col].to_numpy().tolist()


def test_integer_levels_code_in_ascending_numeric_order():
    levels = list(range(1, 13))
    codes = _codes(levels * 2)
    # 1..12 are their own rank; the old repr order gave 1, 10, 11, 12, 2, ...
    assert codes == [float(v - 1) for v in levels] * 2


def test_numeric_levels_order_by_value_across_int_and_float():
    assert _codes([-2, -10, 0, 1]) == [1.0, 0.0, 2.0, 3.0]
    assert _codes([-2.5, -10.0, 0.0, 3.5, 1.0]) == [1.0, 0.0, 2.0, 4.0, 3.0]
    # Numerically equal int and float stay distinct levels, integer first.
    mixed = pd.Series([1.0, 1, 2], dtype=object)
    assert _codes(mixed) == [1.0, 0.0, 2.0]


def test_strings_keep_string_order_and_never_mix_with_numbers():
    assert _codes(["007", "10", "9"]) == [0.0, 1.0, 2.0]
    # Numbers rank before strings, each group in its own natural order.
    assert _codes([10, 9, "10", "9"]) == [1.0, 0.0, 2.0, 3.0]
    assert _codes(["1", 1, True]) == [2.0, 1.0, 0.0]


def test_datetime_bytes_and_other_kinds_order_by_value():
    stamps = [datetime(2021, 1, 2), datetime(2020, 6, 1), datetime(2021, 1, 1)]
    assert _codes(stamps) == [2.0, 0.0, 1.0]
    assert _codes([timedelta(days=3), timedelta(hours=1)]) == [1.0, 0.0]
    assert _codes([b"z", b"a"]) == [1.0, 0.0]


def test_datetimes_outside_the_nanosecond_range_order_by_value():
    # pandas cannot hold these in nanoseconds; they used to rank as "other".
    assert _codes([datetime(2020, 1, 1), datetime(1500, 1, 1), datetime(2021, 1, 1)]) == [
        1.0, 0.0, 2.0,
    ]
    assert _codes([date(2020, 1, 1), date(3000, 1, 1), date(1000, 1, 1)]) == [1.0, 2.0, 0.0]
    stamps = pd.Series(
        [np.datetime64("2020-01-01"), np.datetime64("1500-01-01"), np.datetime64("2021-01-01")],
        dtype=object,
    )
    assert _codes(stamps) == [1.0, 0.0, 2.0]
    assert _codes([timedelta(days=10**6), timedelta(days=1), timedelta(days=-(10**6))]) == [
        2.0, 1.0, 0.0,
    ]
    # An aware value orders by its UTC instant, as in-range timestamps do.
    aware = datetime(1500, 1, 1, 6, tzinfo=timezone(timedelta(hours=5)))  # 01:00 UTC
    assert _codes([aware, datetime(1500, 1, 1, 0, 30), datetime(1500, 1, 1, 1, 30)]) == [
        1.0, 0.0, 2.0,
    ]
    # Still datetime-like: after numbers, before strings.
    assert _codes(["x", datetime(1500, 1, 1), 5, datetime(2000, 1, 1)]) == [3.0, 1.0, 0.0, 2.0]


def test_datetime_likes_at_the_calendar_and_int64_extremes_order_by_value():
    # An aware datetime whose UTC instant falls before year 1 used to rank
    # after strings; a numpy scalar beyond int64 microseconds wrapped around;
    # a non-nanosecond pd.Timedelta beyond int64 microseconds ranked "other".
    east = timezone(timedelta(hours=5))
    assert _codes([datetime(1, 1, 1, 2, tzinfo=east), datetime(1, 1, 1, 12), "x"]) == [
        0.0, 1.0, 2.0,
    ]
    west = timezone(timedelta(hours=-5))
    assert _codes([datetime(9999, 12, 31, 22, tzinfo=west), datetime(9999, 12, 31, 23)]) == [
        1.0, 0.0,
    ]
    stamps = pd.Series(
        [
            np.datetime64(298030, "Y"),
            np.datetime64("2020-01-01"),
            np.datetime64("1500-01-01"),
            np.datetime64(-(2**62), "D"),
        ],
        dtype=object,
    )
    assert _codes(stamps) == [3.0, 2.0, 1.0, 0.0]
    td = np.timedelta64
    durations = pd.Series(
        [
            td(2**62, "s"),
            td(-(2**62), "s"),
            td(2**62, "D"),
            td(1, "D"),
            td(2**63 - 1, "ns"),
            td(-(2**63) + 1, "ns"),
            pd.Timedelta(td(2**62, "ms")),
        ],
        dtype=object,
    )
    assert _codes(durations) == [5.0, 0.0, 6.0, 2.0, 3.0, 1.0, 4.0]
    # Below a nanosecond stays exact.
    assert _codes(pd.Series([td(2, "ns"), td(1500, "ps"), td(1, "ns")], dtype=object)) == [
        2.0, 1.0, 0.0,
    ]


def _exact_order_key(value):
    """Independent oracle: (0, instant) or (1, duration) in exact nanoseconds."""
    ns_per = {
        unit: int(np.timedelta64(1, unit).astype("timedelta64[ns]").astype(np.int64))
        for unit in ("W", "D", "h", "m", "s", "ms", "us", "ns")
    }
    micro = timedelta(microseconds=1)
    if isinstance(value, pd.Timestamp):
        return (0, _exact_order_key(value.to_pydatetime(warn=False))[1] + value.nanosecond)
    if isinstance(value, pd.Timedelta):
        unit, _ = np.datetime_data(value.asm8.dtype)
        return (1, int(value.asm8.astype(np.int64)) * ns_per[unit])
    if isinstance(value, datetime):
        offset = value.utcoffset() or timedelta(0)
        since_year_one = (value.replace(tzinfo=None) - datetime(1, 1, 1)) // micro
        epoch = (datetime(1970, 1, 1) - datetime(1, 1, 1)) // micro
        return (0, (since_year_one - epoch - offset // micro) * 1000)
    if isinstance(value, date):
        return (0, (value - date(1970, 1, 1)).days * ns_per["D"])
    if isinstance(value, timedelta):
        return (1, (value // micro) * 1000)
    unit, count = np.datetime_data(value.dtype)
    ticks = int(value.astype(np.int64)) * count
    if isinstance(value, np.timedelta64):
        if unit in ("Y", "M"):
            # numpy's own conversion of a calendar duration to seconds.
            return (1, int(value.astype("timedelta64[s]").astype(np.int64)) * ns_per["s"])
        return (1, ticks * ns_per[unit])
    if unit in ("Y", "M"):
        # numpy's own calendar arithmetic, exact while the days fit int64.
        return (0, int(value.astype("datetime64[D]").astype(np.int64)) * ns_per["D"])
    return (0, ticks * ns_per[unit])


def _random_datetime_like(rng):
    kind = int(rng.integers(9))
    if kind == 0:
        stamp = datetime(int(rng.integers(1, 10000)), int(rng.integers(1, 13)), 1)
        return stamp + timedelta(days=int(rng.integers(28)), microseconds=int(rng.integers(86_400 * 10**6)))
    if kind == 1:
        base = datetime(1, 1, 1, 12) if rng.random() < 0.3 else datetime(9999, 12, 31, 12)
        if rng.random() < 0.4:
            base = datetime(int(rng.integers(1, 10000)), 6, 1)
        minutes = int(rng.integers(-1439, 1440))
        moved = base + timedelta(minutes=int(rng.integers(-600, 600)))
        return moved.replace(tzinfo=timezone(timedelta(minutes=minutes)))
    if kind == 2:
        return date(int(rng.integers(1, 10000)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    if kind == 3:
        return timedelta(
            days=int(rng.integers(-999_999_999, 1_000_000_000)),
            microseconds=int(rng.integers(86_400 * 10**6)),
        )
    if kind == 4:
        unit = ["s", "ms", "us", "ns"][int(rng.integers(4))]
        if unit == "ns":
            ns = int(rng.integers(pd.Timestamp.min.value, pd.Timestamp.max.value))
            return pd.Timestamp(ns, tz="UTC" if rng.random() < 0.5 else None)
        naive = datetime(int(rng.integers(1, 10000)), int(rng.integers(1, 13)), 1, 7, 8, 9)
        return pd.Timestamp(np.datetime64(naive, unit))
    if kind == 5:
        unit = ["s", "ms", "us", "ns"][int(rng.integers(4))]
        return pd.Timedelta(np.timedelta64(int(rng.integers(-(2**63) + 1, 2**63)), unit))
    if kind == 6:
        unit = ["W", "D", "h", "m", "s", "ms", "us", "ns"][int(rng.integers(8))]
        return np.datetime64(int(rng.integers(-(2**63) + 1, 2**63)), unit)
    if kind == 7:
        unit = ["Y", "M"][int(rng.integers(2))]
        return np.datetime64(int(rng.integers(-(10**12), 10**12)), unit)
    unit = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"][int(rng.integers(10))]
    bound = 10**9 if unit in ("Y", "M") else 2**63
    return np.timedelta64(int(rng.integers(-bound + 1, bound)), unit)


def test_datetime_like_order_matches_an_exact_oracle_on_random_mixes():
    rng = np.random.default_rng(20260927)
    for _ in range(300):
        values = [_random_datetime_like(rng) for _ in range(int(rng.integers(2, 8)))]
        codes = _codes(pd.Series(values + ["x"], dtype=object))
        assert codes[-1] == max(codes)
        keys = [_exact_order_key(value) for value in values]
        for key_a, code_a, a in zip(keys, codes, values):
            for key_b, code_b, b in zip(keys, codes, values):
                if key_a < key_b:
                    assert code_a < code_b, (a, b)


def test_datetime_like_keys_in_the_nanosecond_range_are_pandas_values():
    # Everything pandas holds keeps exactly the key (and so the order) it had.
    from sift._unsupervised_cat import _datetime_like_value

    rng = np.random.default_rng(7)
    tz = timezone(timedelta(hours=-7, minutes=-30))
    for _ in range(300):
        day = 86_400 * 10**9
        ns = int(rng.integers(pd.Timestamp.min.value + day, pd.Timestamp.max.value - day))
        stamp = pd.Timestamp(ns)
        micro_stamp = stamp.floor("us").to_pydatetime()
        for value in (stamp, micro_stamp, micro_stamp.replace(tzinfo=tz), stamp.date(),
                      stamp.to_datetime64(), stamp.to_datetime64().astype("datetime64[s]")):
            assert _datetime_like_value(value) == (0, pd.Timestamp(value).value), value
        span = pd.Timedelta(int(rng.integers(-(2**63) + 1, 2**63)))
        for value in (span, span.to_timedelta64(), span.floor("us").to_pytimedelta()):
            assert _datetime_like_value(value) == (1, pd.Timedelta(value).value), value


def test_decimal_and_fraction_levels_order_as_numbers():
    assert _codes([Decimal(10), Decimal(2), 3]) == [2.0, 0.0, 1.0]
    assert _codes([Fraction(1, 2), Fraction(1, 3), 1]) == [1.0, 0.0, 2.0]
    assert _codes([Fraction(7, 3), Decimal("2.5"), 2, 2.4]) == [1.0, 3.0, 0.0, 2.0]
    assert _codes(["x", Decimal(5), 1, "a"]) == [3.0, 1.0, 0.0, 2.0]
    # Numerically equal levels stay distinct: int, then float, then the other
    # real types by repr.
    assert _codes([Fraction(1), Decimal(1), 1.0, 1]) == [3.0, 2.0, 1.0, 0.0]
    assert _codes([Decimal("Infinity"), Decimal(1), 10**400]) == [2.0, 0.0, 1.0]
    # A NaN Decimal is missing and takes the last code.
    assert _codes([Decimal("NaN"), Decimal(1), "a"]) == [2.0, 0.0, 1.0]


@pytest.mark.parametrize("signaling", [Decimal("sNaN"), Decimal("-sNaN"), Decimal("sNaN12")])
def test_signaling_nan_decimal_is_missing_like_a_quiet_nan(signaling):
    from sift._preprocess import LeaveOneOutLogitEncoder, TargetCVEncoder

    rng = np.random.default_rng(4)
    n = 60
    levels = [Decimal(i % 3) for i in range(n)]
    x = rng.normal(size=n)
    y = x + rng.normal(size=n)
    y_binary = (y > np.median(y)).astype(int)

    def frame(nan):
        values = list(levels)
        values[5] = values[17] = nan
        return pd.DataFrame({"c": pd.Series(values, dtype=object), "x": x})

    quiet, loud = frame(Decimal("NaN")), frame(signaling)
    # pd.isna raises InvalidOperation on a signaling NaN; it used to crash
    # every encoder SIFT implements.
    encoders = (
        lambda: UnsupervisedCatEncoder(["c"], method="ordinal"),
        lambda: UnsupervisedCatEncoder(["c"], method="frequency"),
        lambda: OneHotBlockEncoder(["c"]),
    )
    for make in encoders:
        pd.testing.assert_frame_equal(
            make().fit_transform(loud), make().fit_transform(quiet)
        )
    assert _codes(loud["c"])[5] == 3.0
    pd.testing.assert_frame_equal(
        TargetCVEncoder(["c"], cv=3).fit_transform(loud, y),
        TargetCVEncoder(["c"], cv=3).fit_transform(quiet, y),
    )
    pd.testing.assert_frame_equal(
        LeaveOneOutLogitEncoder(["c"]).fit_transform(loud, y_binary),
        LeaveOneOutLogitEncoder(["c"]).fit_transform(quiet, y_binary),
    )
    for encoding in ("target_cv", "ordinal"):
        kw = {"cat_features": ["c"], "cat_encoding": encoding, "verbose": False}
        loud_result = select_cefsplus(loud, y, 2, return_result=True, **kw)
        quiet_result = select_cefsplus(quiet, y, 2, return_result=True, **kw)
        pd.testing.assert_frame_equal(loud_result.ranking_, quiet_result.ranking_)


def test_numpy_timedelta_scalars_in_an_object_column_encode_as_durations():
    # np.timedelta64 subclasses np.integer, so these used to crash in int().
    raw = [
        np.timedelta64(3, "D"),
        np.timedelta64(36, "h"),
        np.timedelta64(1, "D"),
        np.timedelta64(24, "h"),
        np.timedelta64("NaT"),
    ]
    values = pd.Series(raw, dtype=object)
    # One day and 24 hours are one level; NaT is missing and takes the last code.
    assert _codes(values) == [2.0, 1.0, 0.0, 0.0, 3.0]
    typed = pd.Series(raw, dtype="timedelta64[ns]")
    assert _codes(typed) == _codes(values)

    frame = pd.DataFrame({"c": values})
    freq = UnsupervisedCatEncoder(["c"], method="frequency").fit(frame)
    assert freq.transform(frame)["c"].tolist() == [0.2, 0.2, 0.4, 0.4, 0.2]
    onehot = OneHotBlockEncoder(["c"]).fit(frame)
    dummies = onehot.transform(frame)
    assert dummies.shape == (5, 4)
    assert dummies.to_numpy().sum() == 5
    # The object column and the typed column share one vocabulary.
    np.testing.assert_array_equal(
        onehot.transform(pd.DataFrame({"c": typed})).to_numpy(), dummies.to_numpy()
    )


def test_numpy_calendar_durations_equal_in_numpy_are_one_level():
    td = np.timedelta64
    # numpy equates a year with twelve months; pandas holds neither unit.
    assert td(1, "Y") == td(12, "M")
    values = pd.Series(
        [td(1, "Y"), td(12, "M"), td(13, "M"), td(11, "M"), td(2, "Y"), td(24, "M")],
        dtype=object,
    )
    assert _codes(values) == [1.0, 1.0, 2.0, 0.0, 3.0, 3.0]
    frame = pd.DataFrame({"c": values})
    freq = UnsupervisedCatEncoder(["c"], method="frequency").fit(frame)
    assert freq.transform(frame)["c"].tolist() == [2 / 6, 2 / 6, 1 / 6, 1 / 6, 2 / 6, 2 / 6]
    onehot = OneHotBlockEncoder(["c"]).fit(frame)
    assert list(onehot.transform(frame).columns) == [
        "c__12 months", "c__24 months", "c__11 months", "c__13 months",
    ]
    # A calendar duration still ranks among durations, at numpy's average
    # month (365.2425 / 12 days), and never merges with an integer level.
    mixed = pd.Series([td(1, "M"), pd.Timedelta(days=30), 1, pd.Timedelta(days=31)], dtype=object)
    assert _codes(mixed) == [2.0, 1.0, 0.0, 3.0]


def _encodings(values):
    """Ordinal codes, frequencies and one-hot column names of a raw object column."""
    frame = pd.DataFrame({"c": pd.Series(values, dtype=object)})
    ordinal = UnsupervisedCatEncoder(["c"], method="ordinal").fit_transform(frame)
    frequency = UnsupervisedCatEncoder(["c"], method="frequency").fit_transform(frame)
    onehot = OneHotBlockEncoder(["c"]).fit(frame).transform(frame)
    return ordinal["c"].tolist(), frequency["c"].tolist(), list(onehot.columns)


def _scalar(count, dtype):
    """A numpy scalar with a count multiplier, e.g. one tick of ``m8[3D]``."""
    return np.array([count], dtype=dtype)[0]


def test_numpy_durations_are_one_level_per_exact_length():
    td = np.timedelta64
    # A count multiplier is part of the length: one tick of m8[3D] is three
    # days (it used to merge with one day), of m8[15m] fifteen minutes.
    assert _encodings([_scalar(1, "m8[3D]"), td(1, "D"), td(3, "D"), td(2, "D")]) == (
        [2.0, 0.0, 2.0, 1.0],
        [0.5, 0.25, 0.5, 0.25],
        ["c__3 days 00:00:00", "c__1 days 00:00:00", "c__2 days 00:00:00"],
    )
    assert _codes(pd.Series([_scalar(3, "m8[15m]"), td(3, "m"), td(45, "m")], dtype=object)) == [
        1.0, 0.0, 1.0,
    ]
    assert _codes(pd.Series([_scalar(1, "m8[2Y]"), td(24, "M"), td(1, "Y")], dtype=object)) == [
        1.0, 1.0, 0.0,
    ]
    # Equal lengths below a nanosecond, or beyond pandas' range, are one
    # level too; they used to stay apart as raw numpy scalars.
    assert _encodings([td(5000, "ps"), td(5, "ns")]) == (
        [0.0, 0.0], [1.0, 1.0], ["c__0 days 00:00:00.000000005"],
    )
    assert _encodings([td(1, "ns"), td(1000, "ps"), td(1001, "ps")]) == (
        [0.0, 0.0, 1.0],
        [2 / 3, 2 / 3, 1 / 3],
        ["c__0 days 00:00:00.000000001", "c__1001 picoseconds"],
    )
    assert _encodings([td(2, "ps"), td(2000, "fs")]) == ([0.0, 0.0], [1.0, 1.0], ["c__2 picoseconds"])
    assert _encodings([td(2**47, "D"), td(2**47 * 24, "h"), td(2**47 * 24 + 1, "h")]) == (
        [0.0, 0.0, 1.0],
        [2 / 3, 2 / 3, 1 / 3],
        ["c__140737488355328 days", "c__3377699720527873 hours"],
    )
    # A calendar duration never meets a fixed one of the same raw count: this
    # mix used to raise numpy's "Cannot get a common metadata divisor".
    assert _encodings([td(12, "fs"), td(1, "Y"), td(12, "M")]) == (
        [0.0, 1.0, 1.0],
        [1 / 3, 2 / 3, 2 / 3],
        ["c__12 months", "c__12 femtoseconds"],
    )
    assert _encodings([td(60, "as"), td(5, "Y"), _scalar(1, "m8[5Y]")]) == (
        [0.0, 1.0, 1.0],
        [1 / 3, 2 / 3, 2 / 3],
        ["c__60 months", "c__60 attoseconds"],
    )


@pytest.mark.parametrize("resolution", ["s", "ms", "us", "ns"])
def test_typed_timedelta_columns_share_levels_with_numpy_durations(resolution):
    # A typed column hands over pd.Timedelta values in its own resolution; a
    # numpy duration of the same length, in any unit, is the same level.
    rng = np.random.default_rng(11)
    td = np.timedelta64
    for _ in range(50):
        raw = [int(v) for v in rng.integers(-(2**62), 2**62, size=3)]
        raw += [int(v) for v in rng.integers(-50, 50, size=3)]
        typed = pd.Series(np.array(raw, dtype=f"m8[{resolution}]"))
        per = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}[resolution]
        spelled = []
        for value in raw:
            nanoseconds = value * per
            if abs(nanoseconds * 1000) < 2**63 and rng.random() < 0.5:
                spelled.append(td(nanoseconds * 1000, "ps"))
                continue
            # The same length in the coarsest unit that holds it exactly.
            for unit, unit_ns in (("D", 86_400 * 10**9), ("h", 3_600 * 10**9), ("s", 10**9),
                                  ("ms", 10**6), ("us", 10**3), ("ns", 1)):
                if nanoseconds % unit_ns == 0 and abs(nanoseconds // unit_ns) < 2**63:
                    spelled.append(td(nanoseconds // unit_ns, unit))
                    break
        typed_frame = pd.DataFrame({"c": typed})
        numpy_frame = pd.DataFrame({"c": pd.Series(spelled, dtype=object)})
        for method in ("ordinal", "frequency"):
            fitted = UnsupervisedCatEncoder(["c"], method=method).fit(typed_frame)
            pd.testing.assert_frame_equal(
                fitted.transform(numpy_frame), fitted.transform(typed_frame)
            )
        onehot = OneHotBlockEncoder(["c"]).fit(typed_frame)
        assert list(OneHotBlockEncoder(["c"]).fit(numpy_frame).transform(numpy_frame).columns) == list(
            onehot.transform(typed_frame).columns
        )
        np.testing.assert_array_equal(
            onehot.transform(numpy_frame).to_numpy(), onehot.transform(typed_frame).to_numpy()
        )


def test_numpy_datetimes_of_different_units_never_compare():
    dt = np.datetime64
    # numpy hashes a datetime as its raw count and overflows converting years
    # to attoseconds, so these crashed every encoder with OverflowError.
    assert _encodings([dt("1975", "Y"), dt(5, "as")]) == (
        [1.0, 0.0],
        [0.5, 0.5],
        ["c__1970-01-01T00:00:00.000000000000000005", "c__1975"],
    )
    assert _encodings([dt(-6, "ps"), dt("1969-12-26")]) == (
        [1.0, 0.0], [0.5, 0.5], ["c__1969-12-26", "c__1969-12-31T23:59:59.999999999994"],
    )
    # Each unit stays its own level with numpy's own text, as in 1.0, and
    # equal-weight levels keep 1.0's one-hot order.
    assert _encodings([_scalar(1, "M8[3D]"), dt("1970-01-02"), dt("1970-01-04"), dt("1970-01-04")]) == (
        [2.0, 0.0, 1.0, 1.0],
        [0.25, 0.25, 0.5, 0.5],
        ["c__1970-01-04", "c__1970-01-02", "c__1970-01-04__2"],
    )
    assert _encodings([dt("2021-01-01"), dt("2020-06-01T00:00:00"), dt("2020-06-01T00:00:00")]) == (
        [1.0, 0.0, 0.0], [1 / 3, 2 / 3, 2 / 3], ["c__2020-06-01T00:00:00", "c__2021-01-01"],
    )
    assert _encodings([dt("2021-01-01"), dt("2020-06-01T00:00:00"), 7]) == (
        [2.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3], ["c__2020-06-01T00:00:00", "c__2021-01-01", "c__7"],
    )


def test_aware_timestamp_beyond_python_years_encodes():
    # pandas cannot repr an aware non-nanosecond Timestamp after year 9999,
    # which the level tie-break used, so every encoder raised
    # NotImplementedError; its str still works.
    late = pd.Timestamp(np.datetime64("12000-01-01T05:06:07", "s")).tz_localize("UTC")
    ahead = late.tz_convert("America/New_York")
    early = pd.Timestamp(np.datetime64("0500-01-01", "s")).tz_localize("UTC")
    assert _encodings([late, "x", early, late]) == (
        [1.0, 2.0, 0.0, 1.0],
        [0.5, 0.25, 0.25, 0.5],
        ["c__12000-01-01 05:06:07+00:00", "c__0500-01-01 00:00:00+00:00", "c__x"],
    )
    # One instant in two zones: equal Timestamps, one level, as in 1.0.
    ordinal, _, names = _encodings([late, ahead])
    assert ordinal == [0.0, 0.0]
    assert names == ["c__12000-01-01 05:06:07+00:00"]


def test_repeated_wall_clock_hour_is_two_levels_in_any_row_order():
    from zoneinfo import ZoneInfo

    new_york = ZoneInfo("America/New_York")
    first = datetime(2021, 11, 7, 1, 30, tzinfo=new_york)  # 05:30 UTC
    second = first.replace(fold=1)  # 06:30 UTC
    between = datetime(2021, 11, 7, 6, tzinfo=timezone.utc)
    # Python calls the two readings equal, so they used to merge into one
    # level whose instant (and so code) came from whichever row was first.
    assert first == second
    names = [
        "c__2021-11-07 01:30:00-05:00",
        "c__2021-11-07 01:30:00-04:00",
        "c__2021-11-07 06:00:00+00:00",
    ]
    assert _encodings([first, between, second]) == ([0.0, 1.0, 2.0], [1 / 3] * 3, names)
    assert _encodings([second, between, first]) == ([2.0, 1.0, 0.0], [1 / 3] * 3, names)
    # Across zones the levels are as in 1.0: a reading inside the fold never
    # equals another zone's value (PEP 495), an unambiguous one merges with
    # the same instant elsewhere.
    same_instant = datetime(2021, 11, 7, 5, 30, tzinfo=timezone.utc)
    assert _encodings([first, same_instant]) == (
        [0.0, 1.0], [0.5, 0.5], ["c__2021-11-07 01:30:00-04:00", "c__2021-11-07 05:30:00+00:00"],
    )
    summer = datetime(2021, 6, 1, 8, tzinfo=new_york)
    assert _encodings([summer, summer.astimezone(timezone.utc)]) == (
        [0.0, 0.0], [1.0, 1.0], ["c__2021-06-01 08:00:00-04:00"],
    )


_MICRO = timedelta(microseconds=1)
_EPOCH = datetime(1970, 1, 1)


def _instant_truth(value):
    """Seconds since the epoch, exactly, from numpy's and pandas' own conversions.

    Nothing here multiplies raw ticks by unit sizes or counts calendar days:
    numpy casts a datetime of any unit and count multiplier to microseconds
    (or to attoseconds near the epoch), and Python measures its own values.
    """
    if isinstance(value, pd.Timestamp):
        return Fraction(value.value, 10**9)
    if isinstance(value, datetime):
        return Fraction((value - _EPOCH) // _MICRO, 10**6)
    if np.datetime_data(value.dtype)[0] in ("ns", "ps", "fs", "as"):
        return Fraction(int(value.astype("M8[as]").astype(np.int64)), 10**18)
    return Fraction(int(value.astype("M8[us]").astype(np.int64)), 10**6)


def _anchored_datetimes(rng):
    """numpy datetimes of many units near calendar edges, and their neighbours."""
    anchors = [
        date(1970, 1, 1), date(1969, 12, 31), date(1969, 11, 1), date(1968, 3, 1),
        date(2000, 2, 29), date(2000, 3, 1), date(2020, 2, 29), date(2020, 3, 1),
        date(1900, 2, 28), date(1900, 3, 1), date(2100, 2, 28), date(2100, 3, 1),
        date(1600, 3, 1), date(4, 3, 1), date(1, 3, 1), date(9999, 12, 1),
        date(1583, 1, 1), date(2400, 3, 1),
    ]
    anchor = anchors[int(rng.integers(len(anchors)))]
    day = np.datetime64(anchor.isoformat(), "D")
    month = np.datetime64(anchor.isoformat()[:7], "M")
    hours = int(rng.integers(-60, 61))
    choices = [
        lambda: day,
        lambda: month,
        lambda: np.datetime64(anchor.isoformat()[:4], "Y"),
        lambda: day.astype("M8[W]"),
        lambda: day.astype("M8[3D]"),
        lambda: month.astype("M8[2M]"),
        lambda: month.astype("M8[5Y]"),
        lambda: day.astype("M8[h]") + np.timedelta64(hours, "h"),
        lambda: day.astype("M8[m]") + np.timedelta64(hours * 60 + 7, "m"),
        lambda: (day.astype("M8[h]") + np.timedelta64(hours, "h")).astype("M8[7h]"),
        lambda: datetime.combine(anchor, datetime.min.time()) + timedelta(hours=hours),
    ]
    return choices[int(rng.integers(len(choices)))]()


def _subnanosecond_datetime(rng):
    """A value within a few nanoseconds of the epoch, in ns and finer units."""
    kind = int(rng.integers(4))
    if kind == 0:
        return pd.Timestamp(int(rng.integers(-4, 5)))
    unit, per_ns = [("ps", 10**3), ("fs", 10**6), ("as", 10**9)][kind - 1]
    return np.datetime64(int(rng.integers(-4 * per_ns, 4 * per_ns + 1)), unit)


def _assert_ordinal_matches_truth(values, truth, same_level):
    codes = _codes(pd.Series(values, dtype=object))
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            if truth[i] < truth[j]:
                assert codes[i] < codes[j], (a, b)
            if same_level(a, b, truth[i] == truth[j]):
                assert codes[i] == codes[j], (a, b)
            else:
                assert codes[i] != codes[j], (a, b)


def _same_datetime_level(a, b, equal):
    # One level per type and numpy unit (count included), as in 1.0.
    return equal and type(a) is type(b) and getattr(a, "dtype", None) == getattr(b, "dtype", None)


def test_numpy_datetimes_order_like_numpy_calendar_on_random_mixes():
    rng = np.random.default_rng(20260928)
    for _ in range(400):
        values = [_anchored_datetimes(rng) for _ in range(int(rng.integers(2, 9)))]
        truth = [_instant_truth(value) for value in values]
        _assert_ordinal_matches_truth(values, truth, _same_datetime_level)
    for _ in range(200):
        values = [_subnanosecond_datetime(rng) for _ in range(int(rng.integers(2, 9)))]
        truth = [_instant_truth(value) for value in values]
        _assert_ordinal_matches_truth(values, truth, _same_datetime_level)


def _length_truth(value):
    """``(is calendar, seconds)`` from numpy's and pandas' own conversions."""
    if isinstance(value, pd.Timedelta):
        return False, Fraction(value.value, 10**9)
    unit = np.datetime_data(value.dtype)[0]
    if unit in ("Y", "M"):
        # numpy's own average month (365.2425 / 12 days).
        return True, Fraction(int(value.astype("m8[s]").astype(np.int64)))
    if unit in ("ns", "ps", "fs", "as"):
        return False, Fraction(int(value.astype("m8[as]").astype(np.int64)), 10**18)
    return False, Fraction(value.astype("m8[us]").item() // _MICRO, 10**6)


def _random_duration(rng):
    td = np.timedelta64
    month = 2_629_746  # numpy's average month in seconds
    choices = [
        lambda: td(int(rng.integers(-3, 4)), ["Y", "M"][int(rng.integers(2))]),
        lambda: _scalar(int(rng.integers(-2, 3)), ["m8[2Y]", "m8[5M]"][int(rng.integers(2))]),
        lambda: td(int(rng.integers(-3, 4)) * month + int(rng.integers(-2, 3)), "s"),
        lambda: td(int(rng.integers(-3, 4)) * 30 + int(rng.integers(-1, 2)), "D"),
        lambda: td(int(rng.integers(-3, 4)) * 365 + int(rng.integers(-1, 2)), "D"),
        lambda: td(int(rng.integers(-20, 21)), ["W", "D"][int(rng.integers(2))]),
        lambda: _scalar(int(rng.integers(-7, 8)), ["m8[3D]", "m8[7D]", "m8[2W]"][int(rng.integers(3))]),
        lambda: td(int(rng.integers(-6, 7)) * 12, ["h", "m"][int(rng.integers(2))]),
        lambda: pd.Timedelta(int(rng.integers(-3, 4)), "D"),
        lambda: td(int(rng.integers(-3000, 3001)), "ps"),
        lambda: td(int(rng.integers(-3 * 10**6, 3 * 10**6 + 1)), "fs"),
        lambda: td(int(rng.integers(-3, 4)), "ns"),
        lambda: _scalar(int(rng.integers(-6, 7)), ["m8[500ps]", "m8[250fs]"][int(rng.integers(2))]),
    ]
    return choices[int(rng.integers(len(choices)))]()


def test_numpy_duration_levels_match_numpy_lengths_on_random_mixes():
    rng = np.random.default_rng(20260929)

    def same_level(a, b, equal):
        # Equal lengths are one level in any unit; calendar and fixed
        # durations never are, even at numpy's average month.
        return equal and _length_truth(a)[0] == _length_truth(b)[0]

    for _ in range(600):
        values = [_random_duration(rng) for _ in range(int(rng.integers(2, 9)))]
        truth = [_length_truth(value)[1] for value in values]
        _assert_ordinal_matches_truth(values, truth, same_level)
    # Tie-sensitive month and sub-nanosecond neighbours, pinned.
    td = np.timedelta64
    assert _codes(pd.Series(
        [td(2_629_747, "s"), td(1, "M"), td(2_629_745, "s"), td(30, "D"), td(31, "D")], dtype=object
    )) == [3.0, 2.0, 1.0, 0.0, 4.0]
    assert _codes(pd.Series(
        [td(1200, "ps"), td(1_100_000, "fs"), td(1, "ns"), td(-1500, "ps"), td(-1, "ns")], dtype=object
    )) == [4.0, 3.0, 2.0, 0.0, 1.0]


def test_ordered_categorical_uses_declared_order():
    ordered = pd.Categorical(
        ["mid", "low", "high", "mid"],
        categories=["low", "mid", "high"],
        ordered=True,
    )
    assert _codes(ordered) == [1.0, 0.0, 2.0, 1.0]

    unordered = pd.Categorical(
        ["mid", "low", "high", "mid"],
        categories=["low", "mid", "high"],
        ordered=False,
    )
    assert _codes(unordered) == [2.0, 1.0, 0.0, 2.0]

    # Declared order wins over numeric order, and codes stay dense when a
    # declared category is never observed.
    numeric = pd.Categorical([1, 10, 2], categories=[10, 2, 1, 7], ordered=True)
    assert _codes(numeric) == [2.0, 0.0, 1.0]


def test_missing_level_always_takes_the_last_code():
    assert _codes(["b", None, "a"]) == [1.0, 2.0, 0.0]
    assert _codes([5, np.nan, 1]) == [1.0, 2.0, 0.0]
    ordered = pd.Categorical(
        ["high", None, "low"], categories=["low", "high"], ordered=True
    )
    assert _codes(ordered) == [1.0, 2.0, 0.0]


def test_level_order_is_independent_of_row_order():
    values = [7, "b", True, 2.5, None, b"x", "a", 7, datetime(2000, 1, 1)]
    baseline = _codes(values)
    rng = np.random.default_rng(0)
    for _ in range(3):
        order = rng.permutation(len(values))
        permuted = [values[i] for i in order]
        assert _codes(permuted) == [baseline[i] for i in order]


def test_mixed_kinds_order_by_kind_then_value():
    values = ["b", "b", "b", 1, 1, 2.0, True, None]
    # bool < numeric < str < missing, so True=0, 1=1, 2.0=2, "b"=3, missing=4.
    assert _codes(values) == [3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 0.0, 4.0]


def test_frequency_values_and_onehot_vocabulary_are_unchanged():
    values = ["b", "b", "b", 1, 1, 2.0, True, None]
    frame = pd.DataFrame({"c": values})
    freq = UnsupervisedCatEncoder(["c"], method="frequency").fit(frame)
    shares = freq.transform(frame)["c"].to_numpy().tolist()
    assert shares == [3 / 8, 3 / 8, 3 / 8, 2 / 8, 2 / 8, 1 / 8, 1 / 8, 1 / 8]

    onehot = OneHotBlockEncoder(["c"]).fit(frame)
    # Descending mass, ties broken by repr of the identity: ('bool', True) <
    # ('float', 2.0) < ('missing',).
    assert list(onehot.transform(frame).columns) == [
        "c__b",
        "c__1",
        "c__True",
        "c__2.0",
        "c__missing",
    ]


def test_monotone_ordinal_feature_beats_a_weaker_decoy():
    rng = np.random.default_rng(7)
    n = 200
    level = rng.integers(1, 13, size=n)
    y = level + rng.normal(0, 1.0, size=n)
    X = pd.DataFrame({"level": level, "decoy": 0.35 * y + rng.normal(size=n)})
    # The decoy is genuinely the weaker feature on the raw levels.
    assert abs(np.corrcoef(X["decoy"], y)[0, 1]) < abs(np.corrcoef(level, y)[0, 1])
    picked = select_cefsplus(
        X, y, k=1, cat_features=["level"], cat_encoding="ordinal",
        subsample=None, verbose=False,
    )
    assert list(picked) == ["level"]


def _cross_dtype_frames():
    train_float = pd.DataFrame({"g": [1.0, 2.0, 1.0, np.nan, 2.0, 1.0] * 4})
    score_int = pd.DataFrame({"g": [1, 2, 1, 2] * 2})
    return train_float, score_int


@pytest.mark.parametrize("method", ["ordinal", "frequency"])
def test_encoder_fitted_on_floats_recognises_an_integer_batch(method):
    train_float, score_int = _cross_dtype_frames()
    enc = UnsupervisedCatEncoder(["g"], method=method).fit(train_float)
    got = enc.transform(score_int)["g"].to_numpy()
    matched = enc.transform(score_int.astype(float))["g"].to_numpy()
    np.testing.assert_allclose(got, matched)
    # Hand-computed: 3 of 6 training rows are 1.0, 2 are 2.0, 1 is missing, and
    # the scoring batch alternates 1, 2.
    expected = {"ordinal": [0.0, 1.0], "frequency": [3 / 6, 2 / 6]}[method]
    np.testing.assert_allclose(got, expected * (len(score_int) // 2))
    assert enc.vocabulary_["g"]["identities"] == (("float", 1.0), ("float", 2.0), ("missing",))


@pytest.mark.parametrize("method", ["ordinal", "frequency"])
def test_encoder_fitted_on_integers_recognises_a_float_batch(method):
    train_int = pd.DataFrame({"g": [1, 2, 1, 2, 1, 3]})
    score_float = pd.DataFrame({"g": [1.0, 3.0, 2.0]})
    enc = UnsupervisedCatEncoder(["g"], method=method).fit(train_int)
    got = enc.transform(score_float)["g"].to_numpy()
    matched = enc.transform(pd.DataFrame({"g": [1, 3, 2]}))["g"].to_numpy()
    np.testing.assert_allclose(got, matched)
    expected = {"ordinal": [0.0, 2.0, 1.0], "frequency": [3 / 6, 1 / 6, 2 / 6]}[method]
    np.testing.assert_allclose(got, expected)
    assert enc.vocabulary_["g"]["identities"] == (("int", 1), ("int", 2), ("int", 3))


def test_onehot_fitted_on_floats_recognises_an_integer_batch():
    train_float, score_int = _cross_dtype_frames()
    enc = OneHotBlockEncoder(["g"]).fit(train_float)
    got = enc.transform(score_int)
    matched = enc.transform(score_int.astype(float))
    # Fitted dummy names stay in the float vocabulary; only lookup changes.
    assert list(got.columns) == ["g__1.0", "g__2.0", "g__missing"]
    np.testing.assert_allclose(got.to_numpy(), matched.to_numpy())
    assert got.to_numpy().sum() == len(score_int)


def test_cross_dtype_fallback_never_crosses_other_kinds():
    train = pd.DataFrame({"g": [1.0, 2.0, 1.0, 2.5]})
    enc = UnsupervisedCatEncoder(["g"], method="ordinal").fit(train)
    probe = pd.DataFrame({"g": ["1", True, 3, 2.5, 1]})
    got = enc.transform(probe)["g"].to_numpy().tolist()
    # str "1", bool True and the unfitted 3 stay unknown; 2.5 and int 1 match.
    assert got == [-1.0, -1.0, -1.0, 2.0, 0.0]
    big = pd.DataFrame({"g": [float(2**53)]})
    enc_big = UnsupervisedCatEncoder(["g"], method="ordinal").fit(big)
    # 2**53 + 1 is not representable as that float, so it must stay unknown.
    assert enc_big.transform(pd.DataFrame({"g": [2**53 + 1]}))["g"].tolist() == [-1.0]
    assert enc_big.transform(pd.DataFrame({"g": [2**53]}))["g"].tolist() == [0.0]


@pytest.mark.parametrize("encoding", ["ordinal", "frequency", "onehot"])
def test_selector_wrapper_transform_handles_an_integer_batch(encoding):
    train_float, score_int = _cross_dtype_frames()
    train = train_float.assign(noise=np.linspace(-1.0, 1.0, len(train_float)))
    score = score_int.assign(noise=np.linspace(-1.0, 1.0, len(score_int)))
    y = np.arange(len(train), dtype=float)
    selector = MRMRSelector(
        k=2, task="regression", cat_features=["g"], cat_encoding=encoding,
        subsample=None, verbose=False,
    ).fit(train, y)
    got = np.asarray(selector.transform(score), dtype=np.float64)
    matched = np.asarray(selector.transform(score.astype(float)), dtype=np.float64)
    np.testing.assert_allclose(got, matched)
    names = [str(name) for name in selector.get_feature_names_out()]
    columns = [i for i, name in enumerate(names) if name == "g" or name.startswith("g__")]
    assert columns
    block = got[:, columns]
    if encoding == "onehot":
        np.testing.assert_allclose(block.sum(axis=1), np.ones(len(score)))
    else:
        assert not np.any(block == (-1.0 if encoding == "ordinal" else 0.0))


def test_non_numeric_error_lists_every_accepted_encoding():
    X = pd.DataFrame({"c": ["a", "b"], "n": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Non-numeric columns found") as exc:
        validate_inputs(X, np.array([0.0, 1.0]), "regression")
    message = str(exc.value)
    assert "'ordinal'" in message and "'frequency'" in message
    # Every accepted value is offered; 'none' would leave the columns raw.
    for name in get_args(CatEncoding):
        assert (repr(name) in message) == (name != "none")


def test_tiny_positive_weights_keep_their_levels():
    frame = pd.DataFrame({"c": ["a", "b", "c"]})
    weights = np.array([1e308, 1e-300, 1e-300])
    ordinal = UnsupervisedCatEncoder(["c"], method="ordinal").fit(frame, sample_weight=weights)
    assert ordinal.vocabulary_["c"]["identities"] == (
        ("str", "a"),
        ("str", "b"),
        ("str", "c"),
    )
    assert ordinal.transform(frame)["c"].to_numpy().tolist() == [0.0, 1.0, 2.0]

    freq = UnsupervisedCatEncoder(["c"], method="frequency").fit(frame, sample_weight=weights)
    mapping = freq.vocabulary_["c"]["mapping"]
    assert set(mapping) == {("str", "a"), ("str", "b"), ("str", "c")}
    assert mapping[("str", "a")] == pytest.approx(1.0)
    # The shares of the tiny levels underflow, but they are fitted levels.
    assert mapping[("str", "b")] == 0.0
    # Zero-weight rows are still dropped from the vocabulary.
    zero = UnsupervisedCatEncoder(["c"], method="ordinal").fit(
        frame, sample_weight=np.array([1.0, 0.0, 1.0])
    )
    assert ("str", "b") not in zero.vocabulary_["c"]["mapping"]
    assert zero.transform(frame)["c"].to_numpy().tolist() == [0.0, -1.0, 1.0]
