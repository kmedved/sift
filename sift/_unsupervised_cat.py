"""Target-blind width-preserving encoders for ``ordinal`` and ``frequency``.

Private implementation for ``cat_encoding='ordinal'`` / ``'frequency'``. Each
raw categorical becomes one numeric column; ``ordinal`` is one-to-one over the
fitted levels, ``frequency`` is not (levels of equal mass share one value).
Public exports and one-hot behavior are unchanged.
"""

from __future__ import annotations

import numbers
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any, List, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sift._preprocess import (
    _fitted_level_identity,
    _onehot_level_identity,
    ensure_weights,
    require_unique_encoding_columns,
)

UnsupervisedCatEncoding = Literal["ordinal", "frequency"]

UNSUPERVISED_CAT_ENCODINGS = frozenset({"ordinal", "frequency"})
UNSUPERVISED_UNKNOWN = {"ordinal": -1.0, "frequency": 0.0}
MISSING_IDENTITY = ("missing",)
# Resampled auto-k rules score subsets of a globally encoded matrix and have
# no train-fold encoder remap. In-sample rules (elbow, penalized, chi2, ...)
# treat the call's X as training data. Held-out rules are wired separately.
UNSUPPORTED_UNSUPERVISED_AUTO_K = frozenset(
    {"stability", "knockoff_path", "consensus"}
)
UNSUPPORTED_UNSUPERVISED_AUTO_K_MESSAGE = (
    "cannot use this auto-k rule: the method resamples or draws knockoffs "
    "from a globally encoded matrix and has no training-partition encoder "
    "remap, so validation/OOB frequencies would enter the map. Use "
    "evaluate, gaussian_cv, xfit_objective, elbow, penalized_objective, "
    "or k_method='auto'. Nested evaluate learns maps on outer training folds."
)


def is_unsupervised_cat_encoding(method: str | None) -> bool:
    return method in UNSUPERVISED_CAT_ENCODINGS


def require_unsupervised_auto_k(method: str | None) -> None:
    """Reject resampled auto-k that would silently reuse global frequencies."""
    if method is None or method not in UNSUPPORTED_UNSUPERVISED_AUTO_K:
        return
    raise ValueError(
        f"cat_encoding in {{'ordinal', 'frequency'}} is not supported with "
        f"k_method={method!r}: " + UNSUPPORTED_UNSUPERVISED_AUTO_K_MESSAGE
    )


def _positive_mass(series: pd.Series, weights: np.ndarray) -> dict[tuple, float]:
    """Accumulate positive-weight mass with a safe scale before summation.

    Individual finite weights such as ``1e308`` overflow a raw sum even though
    they are valid for ``ensure_weights(..., normalize=True)``. Scaling by the
    maximum positive weight keeps proportions identical (rescaling and
    integer-replication invariant) without float32 quantization.
    """
    observed = series.to_numpy(dtype=object, copy=False)
    identities = [_onehot_level_identity(value) for value in observed]
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    positive = w > 0.0
    if not np.any(positive):
        return {}
    scale = float(np.max(w[positive]))
    if not np.isfinite(scale) or scale <= 0.0:
        return {}
    mass: dict[tuple, float] = {}
    for ident, weight, keep in zip(identities, w, positive):
        if not keep:
            continue
        # A tiny positive weight can underflow to 0.0 against a huge maximum
        # (1e-300 against 1e308). The level still has a positive-weight row, so
        # it keeps a vocabulary entry with a 0.0 share instead of turning into
        # an unknown at transform time.
        mass[ident] = mass.get(ident, 0.0) + float(weight) / scale
    return mass


_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_MICROSECOND = timedelta(microseconds=1)
_DAY_NS = 86_400 * 10**9
# Nanoseconds per numpy datetime unit. Units below a nanosecond stay exact as
# fractions; a generic unit counts nanoseconds, as pandas reads it.
_UNIT_NS: dict[str, int | Fraction] = {
    "W": 7 * _DAY_NS,
    "D": _DAY_NS,
    "h": 3_600 * 10**9,
    "m": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
    "ps": Fraction(1, 10**3),
    "fs": Fraction(1, 10**6),
    "as": Fraction(1, 10**9),
    "generic": 1,
}
# A calendar duration (years, months) has no fixed length; numpy converts one
# at the average Gregorian month of 365.2425 / 12 days.
_MONTH_NS = 2_629_746 * 10**9
_DAYS_BEFORE_MONTH = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)


def _days_from_epoch(year: int, month: int) -> int:
    """Days from 1970-01-01 to the first of ``month``, for any integer year.

    Proleptic Gregorian with a year 0, as numpy counts, so it also covers the
    years ``datetime`` cannot hold.
    """
    before = year - 1
    days = before * 365 + before // 4 - before // 100 + before // 400
    days += _DAYS_BEFORE_MONTH[month - 1]
    if month > 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        days += 1
    return days + 1 - _EPOCH_ORDINAL


def _numpy_nanoseconds(value: np.datetime64 | np.timedelta64) -> int | Fraction | None:
    """Exact nanoseconds of a numpy scalar in any unit, with no int64 overflow."""
    unit, count = np.datetime_data(value.dtype)
    steps = int(value.astype(np.int64)) * count
    if unit in ("Y", "M"):
        months = steps * 12 if unit == "Y" else steps
        if isinstance(value, np.timedelta64):
            return months * _MONTH_NS
        years, month = divmod(months, 12)
        return _days_from_epoch(1970 + years, month + 1) * _DAY_NS
    factor = _UNIT_NS.get(unit)
    return None if factor is None else steps * factor


def _datetime_like_value(value: Any) -> tuple[int, int | Fraction] | None:
    """``(sub-kind, nanoseconds)`` for datetime-like/timedelta-like values.

    Exact integer arithmetic, so no year or duration is out of range: pandas'
    nanosecond range, ``datetime``'s years 1-9999 and numpy's int64 ticks all
    stop mattering. Instants are counted from the epoch, and an aware value
    counts its UTC instant, as ``pd.Timestamp.value`` does wherever it exists.
    """
    try:
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            # Its own unit and, for an aware stamp, its UTC instant.
            value = value.asm8
        if isinstance(value, (np.datetime64, np.timedelta64)):
            nanoseconds = _numpy_nanoseconds(value)
            if nanoseconds is None:
                return None
            return (0 if isinstance(value, np.datetime64) else 1), nanoseconds
        if isinstance(value, datetime):
            naive = value.replace(tzinfo=None)
            seconds = (naive.toordinal() - _EPOCH_ORDINAL) * 86_400 + (
                naive.hour * 3_600 + naive.minute * 60 + naive.second
            )
            micros = seconds * 10**6 + naive.microsecond
            offset = value.utcoffset()
            if offset is not None:
                micros -= offset // _MICROSECOND
            return 0, micros * 1_000
        if isinstance(value, date):
            return 0, (value.toordinal() - _EPOCH_ORDINAL) * _DAY_NS
        if isinstance(value, timedelta):
            return 1, (value // _MICROSECOND) * 1_000
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def _real_value(value: Any) -> Any | None:
    """Exactly comparable value of a real number outside the int/float kinds.

    ``Decimal`` (not registered as ``numbers.Real``) and rationals such as
    ``Fraction`` compare exactly with ints and floats; any other real type is
    compared through ``float``. A NaN has no place in the order.
    """
    try:
        if isinstance(value, Decimal):
            return None if value.is_nan() else value
        if isinstance(value, numbers.Rational):
            return Fraction(value.numerator, value.denominator)
        if isinstance(value, numbers.Real):
            as_float = float(value)
            return None if np.isnan(as_float) else as_float
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        return None
    return None


def _natural_order_key(identity: tuple) -> tuple:
    """Sort key ordering one level the way its kind is normally ordered.

    Kinds rank bool < numeric < datetime-like < str < bytes < everything else,
    so the key stays total across mixed columns. Integers, floats and other
    real numbers (``Decimal``, ``Fraction``) share the numeric rank and compare
    by value (exactly, without a float cast); numerically equal levels order
    int, then float, then the other real types. Datetime-likes compare by
    instant and durations by length, exactly and at any magnitude, with every
    datetime before every duration. The trailing ``repr`` only breaks ties
    between identities a kind cannot separate, for example two timestamps
    that denote the same instant in different time zones.
    """
    kind = identity[0]
    if kind == "bool":
        primary: tuple = (0, bool(identity[1]))
    elif kind in {"int", "float"}:
        primary = (1, identity[1], 0 if kind == "int" else 1)
    elif kind == "str":
        primary = (3, identity[1])
    elif kind == "bytes":
        primary = (4, identity[1])
    else:
        # Datetime-likes first: numpy registers np.timedelta64 as an integer.
        moment = _datetime_like_value(identity[2])
        real = None if moment is not None else _real_value(identity[2])
        if moment is not None:
            primary = (2,) + moment
        elif real is not None:
            primary = (1, real, 2)
        else:
            primary = (5, identity[1])
    return primary + (repr(identity),)


def _declared_order(series: pd.Series) -> dict[tuple, int] | None:
    """Level ranks declared by an ordered pandas Categorical, else ``None``."""
    dtype = getattr(series, "dtype", None)
    if not isinstance(dtype, pd.CategoricalDtype) or not dtype.ordered:
        return None
    ranks: dict[tuple, int] = {}
    for position, category in enumerate(dtype.categories):
        ranks.setdefault(_onehot_level_identity(category), position)
    return ranks


def _canonical_identities(
    mass: dict[tuple, float], declared: dict[tuple, int] | None = None
) -> tuple[tuple, ...]:
    """Vocabulary order: declared order if any, else natural, missing last."""

    def key(identity: tuple) -> tuple:
        if identity == MISSING_IDENTITY:
            return (2,)
        if declared is not None:
            rank = declared.get(identity)
            if rank is not None:
                return (0, rank)
        return (1,) + _natural_order_key(identity)

    return tuple(sorted(mass, key=key))


class UnsupervisedCatEncoder(BaseEstimator, TransformerMixin):
    """Target-independent encoder: one numeric column per raw categorical.

    Fitted vocabulary uses rows with positive ``sample_weight`` only (a level
    with any strictly positive weight is kept, however small). Identities reuse
    one-hot level identity (missing is ``("missing",)`` when observed in that
    positive-weight mass), so ``1``, ``1.0``, ``"1"`` and ``True`` stay four
    levels. Declared-but-unobserved pandas Categorical levels are ignored.

    Ordinal codes are ``0..C-1`` in natural level order, independent of row
    order: an ordered Categorical uses its declared category order, otherwise
    levels group by kind as bool < numeric (ints, floats, ``Decimal`` and
    ``Fraction`` by value) < datetime-like (datetimes by instant, then
    durations by length, exactly at any magnitude) < str < bytes <
    everything else (by type name and ``repr``).
    Missing, when fitted, always takes the last code. Ordinal unknown is
    ``-1``; frequency unknown is ``0``. Frequency values are the level's share
    of positive training mass (scale-invariant), so equal-mass levels share one
    value and the map is not one-to-one. At transform time a value whose exact
    identity is unknown falls back to a numerically equal fitted level of the
    other numeric kind (``1`` against a fitted ``1.0``); the fitted vocabulary
    is never changed. ``y`` is ignored. Zero-weight rows are still transformed
    so row count is preserved.
    """

    def __init__(self, cols: List[Any], *, method: UnsupervisedCatEncoding):
        if method not in UNSUPERVISED_CAT_ENCODINGS:
            raise ValueError(
                "unsupervised cat_encoding method must be 'ordinal' or "
                f"'frequency'; got {method!r}"
            )
        self.cols = list(cols)
        self.method = method

    def fit(self, X: pd.DataFrame, y=None, sample_weight=None):
        del y
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"cat_encoding={self.method!r} requires a pandas DataFrame"
            )
        require_unique_encoding_columns(X, encoding=self.method)
        missing = [col for col in self.cols if col not in X.columns]
        if missing:
            raise ValueError(
                f"{self.method} cat_features are not columns of X: {missing[:5]}"
            )
        n = len(X)
        weights = (
            np.ones(n, dtype=np.float64)
            if sample_weight is None
            else ensure_weights(sample_weight, n, normalize=False)
        )
        unknown = UNSUPERVISED_UNKNOWN[self.method]
        vocab: dict[Any, dict[str, Any]] = {}
        for col in self.cols:
            column = X[col]
            mass = _positive_mass(column, weights)
            if not mass:
                raise ValueError(
                    f"{self.method} column {col!r} has no positive-weight rows "
                    "to learn a vocabulary"
                )
            identities = _canonical_identities(mass, _declared_order(column))
            if self.method == "ordinal":
                mapping = {
                    ident: float(code) for code, ident in enumerate(identities)
                }
            else:
                total = float(sum(mass.values()))
                if not np.isfinite(total) or total <= 0.0:
                    raise ValueError(
                        f"{self.method} column {col!r} has no finite positive "
                        "weight mass to learn frequencies"
                    )
                mapping = {ident: float(mass[ident]) / total for ident in identities}
            vocab[col] = {
                "identities": identities,
                "mapping": mapping,
                "unknown": unknown,
                "mass": mass,
            }
        self.vocabulary_ = vocab
        self.feature_names_in_ = tuple(X.columns)
        self.n_features_in_ = int(X.shape[1])
        self.output_names_ = tuple(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["vocabulary_", "feature_names_in_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"cat_encoding={self.method!r} transform requires a pandas DataFrame"
            )
        require_unique_encoding_columns(X, encoding=self.method)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"{self.method} transform column identity does not match the "
                "fitted frame"
            )
        pieces: list[pd.Series] = []
        cat_set = set(self.cols)
        for col in X.columns:
            if col in cat_set:
                pieces.append(self._transform_column(X[col], col))
            else:
                pieces.append(X[col])
        return pd.concat(pieces, axis=1)

    def fit_transform(self, X: pd.DataFrame, y=None, sample_weight=None) -> pd.DataFrame:
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def _transform_column(self, series: pd.Series, col: Any) -> pd.Series:
        spec = self.vocabulary_[col]
        mapping: dict[tuple, float] = spec["mapping"]
        unknown = float(spec["unknown"])
        observed = series.to_numpy(dtype=object, copy=False)
        out = np.empty(len(series), dtype=np.float64)
        for i, value in enumerate(observed):
            ident = _fitted_level_identity(_onehot_level_identity(value), mapping)
            out[i] = unknown if ident is None else mapping[ident]
        return pd.Series(out, index=series.index, name=series.name, dtype=np.float64)
