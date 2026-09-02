"""Additive, normalized views over SIFT's legacy result objects."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Set
from numbers import Real
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sift._selector_compat import ordered_indices, validate_output_order
from sift.selection.proxies import normalize_proxy_frame


SCHEMA_VERSION = "1"
CURVE_COLUMNS = ("k", "criterion", "criterion_se", "selected")
_INPUT_KINDS = {"dataframe", "positional", "unknown"}

# Tag marking a serialized mapping that could not be represented as a plain
# JSON object because at least one of its keys is not a string.  See
# ``_json_safe`` for the envelope format.
_MAPPING_ENVELOPE_TAG = "__sift_mapping__"
_MAPPING_ENVELOPE_KIND = "typed_key_entries"


def _label_token(value: Any) -> Any:
    if isinstance(value, np.datetime64):
        return {
            "type": "numpy.datetime64",
            "dtype": str(value.dtype),
            "value": str(value),
        }
    if isinstance(value, np.timedelta64):
        return {
            "type": "numpy.timedelta64",
            "dtype": str(value.dtype),
            "value": str(value),
        }
    if isinstance(value, np.generic):
        value = value.item()
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    if value is None or isinstance(value, (bool, int, str)):
        payload: Any = value
    elif isinstance(value, float):
        if math.isnan(value):
            payload = "NaN"
        elif math.isinf(value):
            payload = "Infinity" if value > 0 else "-Infinity"
        else:
            payload = value
    elif isinstance(value, tuple):
        payload = [_label_token(item) for item in value]
    elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
        payload = value.isoformat()
    else:
        payload = repr(value)
    return {"type": type_name, "value": payload}


def _columns_hash(features: Iterable[Any]) -> str:
    encoded = json.dumps(
        [_label_token(feature) for feature in features],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _labels_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.generic) and not isinstance(
        left, (np.datetime64, np.timedelta64)
    ):
        left = left.item()
    if isinstance(right, np.generic) and not isinstance(
        right, (np.datetime64, np.timedelta64)
    ):
        right = right.item()
    if type(left) is not type(right):
        return False
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _labels_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if left is None or isinstance(
        left,
        (
            bool,
            int,
            float,
            str,
            bytes,
            pd.Timestamp,
            pd.Timedelta,
            np.datetime64,
            np.timedelta64,
        ),
    ):
        return _label_token(left) == _label_token(right)
    values = np.empty(2, dtype=object)
    values[:] = [left, right]
    try:
        index = pd.Index(values, dtype=object, tupleize_cols=False)
        return bool(index.duplicated()[1])
    except (TypeError, ValueError):
        return _label_token(left) == _label_token(right)


def _coerce_feature_names(input_features: Any) -> list[Any] | None:
    if input_features is None:
        return None
    if isinstance(input_features, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("input_features must be an ordered one-dimensional iterable")
    if isinstance(input_features, np.ndarray) and input_features.ndim != 1:
        raise ValueError("input_features must be one-dimensional")
    try:
        names = list(input_features)
    except TypeError as exc:
        raise TypeError("input_features must be an ordered one-dimensional iterable") from exc
    return names


def _coerce_indices(indices: Any, *, label: str) -> list[int] | None:
    if indices is None:
        return None
    out: list[int] = []
    for value in list(indices):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must contain integer positions")
        out.append(int(value))
    if len(set(out)) != len(out):
        raise ValueError(f"{label} must contain unique positions")
    if any(value < 0 for value in out):
        raise ValueError(f"{label} must contain non-negative positions")
    return out


def _strict_integer(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    out = int(value)
    bounds = np.iinfo(np.int64)
    if out < bounds.min or out > bounds.max:
        raise ValueError(f"{label} must fit in a signed 64-bit integer")
    if minimum is not None and out < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return out


def _strict_integer_vector(
    values: Any,
    *,
    label: str,
    length: int,
    minimum: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{label} must be one-dimensional with length {length}")
    return np.asarray(
        [
            _strict_integer(value, label=f"{label} values", minimum=minimum)
            for value in array.tolist()
        ],
        dtype=np.int64,
    )


def _numeric_vector(values: Any, *, label: str, length: int) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{label} must be one-dimensional with length {length}")
    out: list[float] = []
    for value in array.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise ValueError(f"{label} must contain real non-boolean numeric values")
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{label} must contain real non-boolean numeric values representable "
                "as float64"
            ) from exc
        if (
            isinstance(value, np.floating)
            and np.isfinite(value)
            and not math.isfinite(converted)
        ):
            raise ValueError(f"{label} values must be representable as float64")
        out.append(converted)
    return np.asarray(out, dtype=np.float64)


def _coerce_position_series(
    values: pd.Series,
    *,
    label: str,
    allow_missing: bool,
) -> pd.Series:
    out: list[Any] = []
    for value in values.tolist():
        missing = value is None or value is pd.NA or (
            isinstance(value, (float, np.floating)) and math.isnan(float(value))
        )
        if missing:
            if allow_missing:
                out.append(pd.NA)
                continue
            raise ValueError(f"{label} values must be non-missing integers")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} values must be integers")
        position = int(value)
        if position < 0:
            raise ValueError(f"{label} values must be non-negative")
        out.append(position)
    return pd.Series(pd.array(out, dtype="Int64"), index=values.index)


def _coerce_boolean_series(values: pd.Series, *, label: str) -> pd.Series:
    out: list[bool] = []
    for value in values.tolist():
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} values must be non-missing booleans")
        out.append(bool(value))
    return pd.Series(out, index=values.index, dtype=bool)


def _validate_selected_identity(
    features: list[Any],
    indices: list[int] | None,
    input_features: list[Any] | None,
) -> list[int] | None:
    if indices is not None and len(indices) != len(features):
        raise ValueError("selected feature names and indices must have the same length")
    if input_features is None:
        return indices
    if indices is None:
        tokens = [_label_token(value) for value in input_features]
        resolved: list[int] = []
        for feature in features:
            token = _label_token(feature)
            matches = [idx for idx, candidate in enumerate(tokens) if candidate == token]
            if len(matches) != 1:
                raise ValueError(
                    f"selected feature {feature!r} is missing or ambiguous in input_features; "
                    "adapt a result carrying positional selected_indices"
                )
            resolved.append(matches[0])
        indices = resolved
    if any(index >= len(input_features) for index in indices):
        raise ValueError("selected_indices contains a position outside input_features")
    for feature, index in zip(features, indices):
        if not _labels_equal(feature, input_features[index]):
            raise ValueError(
                "selected feature names do not match input_features at selected_indices"
            )
    return indices


def _json_key_token(key: Any) -> dict[str, Any]:
    """Return a collision-free, JSON-safe token for a mapping key.

    The token records the key's concrete type alongside its JSON-safe value so
    that ``1`` and ``"1"`` remain distinguishable.  Unsupported key objects
    raise the same ``TypeError`` as unsupported values.
    """
    return {
        "type": f"{type(key).__module__}.{type(key).__qualname__}",
        "value": _json_safe(key),
    }


def _json_safe(value: Any) -> Any:
    """Convert ``value`` into a JSON-serializable structure (schema version 1).

    Conversions
    -----------
    - ``None``/``bool``/``int``/``str`` pass through unchanged; non-finite
      floats, ``pd.NA``, and ``pd.NaT`` become ``None``.
    - Dates, times, datetimes, and timedeltas become ISO 8601 strings.
    - ``pathlib.Path`` becomes its string form.
    - NumPy scalars/arrays, pandas ``Series``/``DataFrame``, and dataclasses
      become their plain Python equivalents (``DataFrame`` uses
      ``orient="split"``; dataclasses use :func:`dataclasses.asdict`).
    - Sequences and sets become lists.

    Mapping envelope
    ----------------
    A mapping whose keys are **all** strings serializes as an ordinary JSON
    object, so the payload root and normal metadata keep their familiar shape.
    A mapping containing any non-string key would silently merge entries such
    as ``1`` and ``"1"`` under that representation, so it instead serializes as
    a tagged, order-preserving envelope::

        {
            "__sift_mapping__": "typed_key_entries",
            "entries": [
                {"key": {"type": "builtins.int", "value": 1}, "value": "int"},
                {"key": {"type": "builtins.str", "value": "1"}, "value": "str"},
            ],
        }

    Each ``key`` token carries the key's concrete type and its JSON-safe value,
    which keeps distinct keys distinct through ``json.dumps``/``json.loads``.
    Both forms belong to schema version ``"1"``; consumers that meet a mapping
    holding the ``"__sift_mapping__"`` tag must read ``entries`` instead of the
    object's own keys.

    Raises
    ------
    TypeError
        If ``value`` (or a mapping key) has no defined JSON representation.
        Emitting ``repr()`` would leak memory addresses and produce payloads
        that cannot be read back, so unsupported objects fail loudly.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="split"))
    if isinstance(value, pd.Series):
        return {
            "name": _json_safe(value.name),
            "index": _json_safe(value.index.tolist()),
            "data": _json_safe(value.tolist()),
        }
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _json_safe(item) for key, item in value.items()}
        return {
            _MAPPING_ENVELOPE_TAG: _MAPPING_ENVELOPE_KIND,
            "entries": [
                {"key": _json_key_token(key), "value": _json_safe(item)}
                for key, item in value.items()
            ],
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    raise TypeError(
        f"{type(value).__module__}.{type(value).__qualname__} has no JSON-safe "
        "representation in SelectionView schema version "
        f"{SCHEMA_VERSION!r}; convert it to a primitive, list, mapping, "
        "dataclass, NumPy/pandas container, or datetime before serializing."
    )


def _validate_table_selection(
    table: pd.DataFrame,
    features: list[Any],
    indices: list[int] | None,
) -> None:
    required = {"feature", "selected_index", "path_rank", "selected"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"raw_table is missing required columns: {missing}")
    selected_mask = _coerce_boolean_series(table["selected"], label="raw_table selected")
    selected_rows = table.loc[selected_mask]
    if len(selected_rows) != len(features):
        raise ValueError("raw_table selected rows do not match features")
    if indices is not None:
        positions = _coerce_position_series(
            selected_rows["selected_index"],
            label="selected raw_table selected_index",
            allow_missing=False,
        )
        by_position = {
            int(position): feature
            for position, feature in zip(positions, selected_rows["feature"])
        }
        if set(by_position) != set(indices):
            raise ValueError("raw_table selected positions do not match indices")
        for feature, position in zip(features, indices):
            if not _labels_equal(feature, by_position[position]):
                raise ValueError("raw_table selected feature identities do not match features")
        return
    expected = sorted(
        (json.dumps(_label_token(feature), sort_keys=True) for feature in features)
    )
    observed = sorted(
        json.dumps(_label_token(feature), sort_keys=True)
        for feature in selected_rows["feature"]
    )
    if observed != expected:
        raise ValueError("raw_table selected feature identities do not match features")


class SelectionView:
    """Normalized, non-replacing view over a SIFT selection result.

    Legacy result objects remain the public return values.  This class copies
    their available metadata and diagnostics, never the caller's feature
    matrix.
    """

    __slots__ = (
        "_curve",
        "_diagnostics",
        "_encoded_features",
        "_encoded_indices",
        "_encoded_support",
        "_encoded_table",
        "_features",
        "_indices",
        "_inverse_transformer",
        "_metadata",
        "_n_raw_features",
        "_proxy_correlations",
        "_raw_features",
        "_raw_table",
        "_support",
        "_transformer",
    )

    def __init__(
        self,
        *,
        features: Iterable[Any],
        indices: Iterable[int] | None,
        raw_features: Iterable[Any] | None,
        n_raw_features: int | None,
        raw_table: pd.DataFrame,
        curve: pd.DataFrame | None = None,
        metadata: Mapping[str, Any] | None = None,
        diagnostics: Any = None,
        encoded_features: Iterable[Any] | None = None,
        encoded_indices: Iterable[int] | None = None,
        encoded_table: pd.DataFrame | None = None,
        transformer: Callable[[Any], Any] | None = None,
        inverse_transformer: Callable[[Any], Any] | None = None,
        proxy_correlations: pd.DataFrame | None = None,
    ) -> None:
        selected = list(features)
        selected_indices = _coerce_indices(indices, label="indices")
        if selected_indices is not None and len(selected_indices) != len(selected):
            raise ValueError("features and indices must have the same length")

        raw_names = None if raw_features is None else list(raw_features)
        if n_raw_features is None and raw_names is not None:
            n_raw_features = len(raw_names)
        if n_raw_features is not None:
            if isinstance(n_raw_features, (bool, np.bool_)) or not isinstance(
                n_raw_features, (int, np.integer)
            ):
                raise ValueError("n_raw_features must be a non-negative integer or None")
            n_raw_features = int(n_raw_features)
            if n_raw_features < 0:
                raise ValueError("n_raw_features must be a non-negative integer or None")
            if raw_names is not None and len(raw_names) != n_raw_features:
                raise ValueError("raw_features length must equal n_raw_features")
            if selected_indices is not None and any(
                index >= n_raw_features for index in selected_indices
            ):
                raise ValueError("indices contains a position outside the raw feature width")

        encoded_names = None if encoded_features is None else list(encoded_features)
        encoded_positions = _coerce_indices(encoded_indices, label="encoded_indices")
        if encoded_positions is not None and len(encoded_positions) != len(selected):
            raise ValueError("encoded_indices must align with selected features")
        if encoded_names is not None and encoded_positions is not None and any(
            index >= len(encoded_names) for index in encoded_positions
        ):
            raise ValueError("encoded_indices contains an out-of-bounds position")

        if not isinstance(raw_table, pd.DataFrame):
            raise TypeError("raw_table must be a pandas DataFrame")
        normalized_table = raw_table.copy(deep=True).reset_index(drop=True)
        required_table_columns = {"feature", "selected_index", "path_rank", "selected"}
        missing_table_columns = sorted(
            required_table_columns.difference(normalized_table.columns)
        )
        if missing_table_columns:
            raise ValueError(
                f"raw_table is missing required columns: {missing_table_columns}"
            )
        normalized_table["selected_index"] = _coerce_position_series(
            normalized_table["selected_index"],
            label="raw_table selected_index",
            allow_missing=True,
        )
        normalized_table["selected"] = _coerce_boolean_series(
            normalized_table["selected"],
            label="raw_table selected",
        )
        normalized_table["path_rank"] = _coerce_position_series(
            normalized_table["path_rank"],
            label="raw_table path_rank",
            allow_missing=True,
        )
        present_ranks = normalized_table["path_rank"].dropna().astype(int)
        if (
            (present_ranks < 1).any()
            or present_ranks.duplicated().any()
            or set(present_ranks) != set(range(1, len(selected) + 1))
            or not normalized_table["selected"].equals(
                normalized_table["path_rank"].notna()
            )
        ):
            raise ValueError(
                "raw_table path_rank must be unique, one-based, and present exactly for "
                "selected rows"
            )
        _validate_table_selection(normalized_table, selected, selected_indices)
        normalized_curve = (
            pd.DataFrame(columns=CURVE_COLUMNS)
            if curve is None
            else curve.copy(deep=True)
        )
        missing_curve = [column for column in CURVE_COLUMNS if column not in normalized_curve]
        if missing_curve:
            raise ValueError(f"curve is missing required columns: {missing_curve}")
        normalized_curve = normalized_curve.loc[:, list(CURVE_COLUMNS)].reset_index(drop=True)

        meta = copy.deepcopy(dict(metadata or {}))
        meta["schema_version"] = SCHEMA_VERSION
        input_kind = meta.setdefault("input_kind", "unknown")
        if input_kind not in _INPUT_KINDS:
            raise ValueError(
                "metadata['input_kind'] must be 'dataframe', 'positional', or 'unknown'"
            )
        meta.setdefault("table_complete", False)
        if not isinstance(meta["table_complete"], (bool, np.bool_)):
            raise ValueError("metadata['table_complete'] must be boolean")
        meta["table_complete"] = bool(meta["table_complete"])
        if meta["table_complete"]:
            if n_raw_features is None:
                raise ValueError("a complete raw table requires n_raw_features")
            positions = _coerce_position_series(
                normalized_table["selected_index"],
                label="raw_table selected_index",
                allow_missing=False,
            )
            if (
                positions.isna().any()
                or len(normalized_table) != n_raw_features
                or set(int(value) for value in positions) != set(range(n_raw_features))
            ):
                raise ValueError(
                    "metadata marks raw_table complete but it does not cover every raw position"
                )
        meta["transform_available"] = transformer is not None
        meta["inverse_transform_available"] = inverse_transformer is not None
        meta["raw_columns_hash"] = (
            _columns_hash(raw_names) if raw_names is not None else None
        )
        meta["encoded_columns_hash"] = (
            _columns_hash(encoded_names) if encoded_names is not None else None
        )

        support = None
        if n_raw_features is not None and selected_indices is not None:
            support = np.zeros(n_raw_features, dtype=bool)
            support[selected_indices] = True
        encoded_support = None
        if encoded_names is not None and encoded_positions is not None:
            encoded_support = np.zeros(len(encoded_names), dtype=bool)
            encoded_support[encoded_positions] = True

        self._features = tuple(selected)
        self._indices = None if selected_indices is None else tuple(selected_indices)
        self._raw_features = None if raw_names is None else tuple(raw_names)
        self._n_raw_features = n_raw_features
        self._support = support
        self._raw_table = normalized_table
        self._curve = normalized_curve
        self._metadata = meta
        self._diagnostics = copy.deepcopy(diagnostics)
        self._encoded_features = None if encoded_names is None else tuple(encoded_names)
        self._encoded_indices = (
            None if encoded_positions is None else tuple(encoded_positions)
        )
        self._encoded_support = encoded_support
        self._encoded_table = (
            None if encoded_table is None else encoded_table.copy(deep=True).reset_index(drop=True)
        )
        self._transformer = transformer
        self._inverse_transformer = inverse_transformer
        self._proxy_correlations, proxy_storage_bytes = normalize_proxy_frame(
            proxy_correlations,
            selected_indices=selected_indices,
            n_raw_features=n_raw_features,
        )
        self._metadata["proxy_correlations_stored"] = self._proxy_correlations is not None
        self._metadata["proxy_candidate_count"] = (
            0 if self._proxy_correlations is None else len(self._proxy_correlations)
        )
        self._metadata["proxy_storage_bytes"] = proxy_storage_bytes

    @property
    def features(self) -> list[Any]:
        return list(self._features)

    @property
    def indices(self) -> list[int] | None:
        return None if self._indices is None else list(self._indices)

    @property
    def support_(self) -> np.ndarray | None:
        return None if self._support is None else self._support.copy()

    @property
    def k(self) -> int:
        return len(self._features)

    @property
    def raw_features(self) -> list[Any] | None:
        return None if self._raw_features is None else list(self._raw_features)

    @property
    def raw_input(self) -> dict[str, Any]:
        return {
            "n_features": self._n_raw_features,
            "features": self.raw_features,
            "columns_hash": self._metadata["raw_columns_hash"],
        }

    @property
    def encoded_features(self) -> list[Any] | None:
        return (
            None if self._encoded_features is None else list(self._encoded_features)
        )

    @property
    def encoded_indices(self) -> list[int] | None:
        return None if self._encoded_indices is None else list(self._encoded_indices)

    @property
    def encoded_support_(self) -> np.ndarray | None:
        return None if self._encoded_support is None else self._encoded_support.copy()

    @property
    def encoded_output(self) -> dict[str, Any] | None:
        if self._encoded_features is None:
            return None
        return {
            "n_features": len(self._encoded_features),
            "features": self.encoded_features,
            "columns_hash": self._metadata["encoded_columns_hash"],
        }

    @property
    def raw_table(self) -> pd.DataFrame:
        return self._raw_table.copy(deep=True)

    @property
    def table(self) -> pd.DataFrame:
        return self.raw_table

    @property
    def encoded_table(self) -> pd.DataFrame | None:
        return (
            None if self._encoded_table is None else self._encoded_table.copy(deep=True)
        )

    @property
    def curve(self) -> pd.DataFrame:
        return self._curve.copy(deep=True)

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def diagnostics(self) -> Any:
        return copy.deepcopy(self._diagnostics)

    def transform(self, X: Any) -> Any:
        if self._transformer is None:
            raise NotImplementedError(
                "transform is unavailable for this result-only view; adapt a fitted selector "
                "that retains preprocessing state"
            )
        return self._transformer(X)

    def inverse_transform(self, X_selected: Any) -> Any:
        if self._inverse_transformer is None:
            raise NotImplementedError(
                "inverse_transform is unavailable because this view does not retain a fitted "
                "inverse encoder"
            )
        return self._inverse_transformer(X_selected)

    def proxies(self, feature: Any, r_min: float = 0.8) -> pd.DataFrame:
        if self._proxy_correlations is None:
            raise NotImplementedError(
                "proxy correlations were not stored; rerun selection with store_proxies=True"
            )
        raw_names = self.raw_features
        if raw_names is None:
            names = self.features
            positions = self.indices
        else:
            names = raw_names
            positions = list(range(len(raw_names)))
        assert positions is not None
        matches = [
            position
            for position, value in zip(positions, names)
            if _labels_equal(value, feature)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"feature {feature!r} is missing or ambiguous; use "
                "proxies_at(selected_index, ...) for positional access"
            )
        return self.proxies_at(matches[0], r_min=r_min)

    def proxies_at(self, selected_index: int, r_min: float = 0.8) -> pd.DataFrame:
        """Return unselected proxy candidates for one selected raw position."""
        if self._proxy_correlations is None:
            raise NotImplementedError(
                "proxy correlations were not stored; rerun selection with store_proxies=True"
            )
        if isinstance(selected_index, (bool, np.bool_)) or not isinstance(
            selected_index,
            (int, np.integer),
        ):
            raise ValueError("selected_index must be an integer raw-feature position")
        position = int(selected_index)
        if position not in self._proxy_correlations.columns:
            raise ValueError(f"raw feature position {position} is not a selected proxy feature")
        if isinstance(r_min, (bool, np.bool_)) or not isinstance(
            r_min,
            (Real, np.integer, np.floating),
        ):
            raise ValueError("r_min must be a finite number between 0 and 1")
        threshold = float(r_min)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("r_min must be a finite number between 0 and 1")

        values = self._proxy_correlations[position]
        candidate_positions = np.asarray(values.index, dtype=np.int64)
        selected_positions = set(self._indices or ())
        correlations = values.to_numpy(dtype=np.float64)
        mask = np.asarray(
            [candidate not in selected_positions for candidate in candidate_positions],
            dtype=bool,
        )
        mask &= np.abs(correlations) >= threshold
        candidate_positions = candidate_positions[mask]
        correlations = correlations[mask]
        order = np.lexsort((candidate_positions, -np.abs(correlations)))
        candidate_positions = candidate_positions[order]
        correlations = correlations[order]
        raw_names = self.raw_features
        labels = (
            candidate_positions.tolist()
            if raw_names is None
            else [raw_names[int(candidate)] for candidate in candidate_positions]
        )
        return pd.DataFrame(
            {
                "feature": labels,
                "selected_index": candidate_positions,
                "correlation": correlations,
            }
        )

    def plot(self, ax=None):
        curve_available = not self._curve.empty
        if not curve_available and not self._metadata["table_complete"]:
            raise NotImplementedError(
                "plot data is incomplete for this partial result view; supply a result "
                "with a complete raw table"
            )
        metric = None
        if not curve_available:
            metric = next(
                (name for name in ("gain", "relevance") if name in self._raw_table),
                None,
            )
            if metric is None:
                raise NotImplementedError("plot data is unavailable for this result view")
        if ax is None:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:  # pragma: no cover - optional dependency path
                raise ImportError("plot() requires matplotlib") from exc
            _, ax = plt.subplots()
        if curve_available:
            ax.plot(self._curve["k"], self._curve["criterion"], marker="o")
            selected = self._curve.loc[self._curve["selected"]]
            if not selected.empty:
                ax.scatter(selected["k"], selected["criterion"], zorder=3)
            ax.set_xlabel("k")
            ax.set_ylabel("criterion")
            return ax
        assert metric is not None
        table = self._raw_table.sort_values(metric, ascending=False, kind="mergesort")
        ax.bar(np.arange(len(table)), table[metric])
        ax.set_xlabel("feature rank")
        ax.set_ylabel(metric)
        return ax

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "features": self.features,
            "indices": self.indices,
            "support": self.support_,
            "raw_input": self.raw_input,
            "raw_table": self._raw_table,
            "encoded_features": self.encoded_features,
            "encoded_indices": self.encoded_indices,
            "encoded_support": self.encoded_support_,
            "encoded_output": self.encoded_output,
            "encoded_table": self._encoded_table,
            "curve": self._curve,
            "metadata": self._metadata,
            "diagnostics": self._diagnostics,
        }
        return _json_safe(payload)

    def __repr__(self) -> str:
        metadata = getattr(self, "_metadata", {})
        features = getattr(self, "_features", ())
        adapter = metadata.get("adapter", "unknown")
        return f"SelectionView(k={len(features)}, adapter={adapter!r})"


def _selection_path_ranks(
    table: pd.DataFrame,
    features: list[Any],
    indices: list[int] | None,
) -> pd.Series:
    ranks = pd.Series(pd.array([pd.NA] * len(table), dtype="Int64"), index=table.index)
    if indices is not None and "selected_index" in table:
        index_to_rank = {index: rank for rank, index in enumerate(indices, start=1)}
        for row_index, value in table["selected_index"].items():
            if pd.notna(value) and int(value) in index_to_rank:
                ranks.loc[row_index] = index_to_rank[int(value)]
        return ranks
    tokens = [_label_token(value) for value in features]
    for row_index, value in table["feature"].items():
        token = _label_token(value)
        matches = [idx for idx, candidate in enumerate(tokens) if candidate == token]
        if len(matches) == 1:
            ranks.loc[row_index] = matches[0] + 1
    return ranks


def _normalize_filter_table(
    result: Any,
    *,
    raw_features: list[Any] | None,
    n_raw_features: int | None,
    indices: list[int] | None,
) -> tuple[pd.DataFrame, list[Any] | None, bool]:
    ranking = result.get_feature_ranking()
    if not isinstance(ranking, pd.DataFrame) or "feature" not in ranking:
        raise ValueError("filter result ranking must be a DataFrame with a feature column")
    table = pd.DataFrame({"feature": ranking["feature"].tolist()})
    if "selected_index" in ranking:
        table["selected_index"] = _coerce_position_series(
            ranking["selected_index"],
            label="filter ranking selected_index",
            allow_missing=True,
        )
    elif indices is not None and len(ranking) == len(indices):
        table["selected_index"] = pd.array(indices, dtype="Int64")
    table["path_rank"] = _selection_path_ranks(
        table,
        list(result.selected_features),
        indices,
    )
    table["selected"] = table["path_rank"].notna()
    if "relevance" in ranking:
        relevance = pd.to_numeric(ranking["relevance"], errors="coerce")
        if relevance.notna().any():
            table["relevance"] = relevance

    if "selected_index" in table:
        known_positions = table["selected_index"].dropna().astype(int)
        if known_positions.duplicated().any():
            raise ValueError(
                "filter ranking selected_index values must be unique and non-negative"
            )
        if n_raw_features is not None and (known_positions >= n_raw_features).any():
            raise ValueError("filter ranking contains positions outside input_features")
        if raw_features is not None:
            for row_index, position in table["selected_index"].items():
                if pd.isna(position):
                    continue
                if not _labels_equal(
                    table.at[row_index, "feature"],
                    raw_features[int(position)],
                ):
                    raise ValueError(
                        "filter ranking feature identities do not match input_features"
                    )

    complete = False
    if n_raw_features is not None and "selected_index" in table:
        positions = table["selected_index"]
        complete = (
            not positions.isna().any()
            and len(table) == n_raw_features
            and set(int(value) for value in positions) == set(range(n_raw_features))
        )
    if complete:
        table = table.sort_values("selected_index", kind="mergesort").reset_index(drop=True)
        if raw_features is None:
            raw_features = table["feature"].tolist()
        elif any(
            not _labels_equal(feature, table.at[index, "feature"])
            for index, feature in enumerate(raw_features)
        ):
            raise ValueError("filter ranking feature identities do not match input_features")
    return table, raw_features, complete


_CRITERION_DIRECTIONS = {"higher_is_better", "lower_is_better"}


def _normalize_auto_k_curve(payload: Any) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Read a producer-side auto-k curve payload into curve plus metadata.

    Auto-k producers normalize their own route diagnostics (see
    ``sift.selection.filter_auto_k.build_auto_k_curve_payload``), so adapters
    never inspect route-specific diagnostic columns themselves.  A route with no
    k-indexed criterion reports an explicit reason instead of a fabricated curve.
    """
    if payload is None:
        return None, {"curve_available": False}
    if not isinstance(payload, Mapping):
        raise ValueError("auto-k curve payload must be a mapping")
    if not payload.get("available", False):
        reason = payload.get("unavailable_reason")
        return None, {
            "curve_available": False,
            "curve_unavailable_reason": None if reason is None else str(reason),
        }
    criterion = payload.get("criterion")
    direction = payload.get("criterion_direction")
    if not isinstance(criterion, str) or not criterion:
        raise ValueError("auto-k curve payload must name its criterion column")
    if direction not in _CRITERION_DIRECTIONS:
        raise ValueError(
            "auto-k curve payload criterion_direction must be 'higher_is_better' "
            "or 'lower_is_better'"
        )
    curve = payload.get("curve")
    if not isinstance(curve, pd.DataFrame):
        raise ValueError("an available auto-k curve payload must carry a DataFrame curve")
    missing = [column for column in CURVE_COLUMNS if column not in curve.columns]
    if missing:
        raise ValueError(f"auto-k curve payload is missing required columns: {missing}")
    normalized = pd.DataFrame(
        {
            "k": _coerce_position_series(
                curve["k"],
                label="auto-k curve k",
                allow_missing=False,
            ),
            "criterion": pd.to_numeric(curve["criterion"], errors="coerce").to_numpy(
                dtype=float
            ),
            "criterion_se": pd.to_numeric(
                curve["criterion_se"], errors="coerce"
            ).to_numpy(dtype=float),
            "selected": _coerce_boolean_series(
                curve["selected"],
                label="auto-k curve selected",
            ).to_numpy(),
        }
    ).reset_index(drop=True)
    if normalized["k"].duplicated().any():
        raise ValueError("auto-k curve payload k values must be unique")
    return normalized, {
        "curve_available": True,
        "criterion": criterion,
        "criterion_direction": direction,
        "curve_route": payload.get("route"),
    }


def _as_filter_result(result: Any, input_features: Any) -> SelectionView:
    from sift.selection.filter_auto_k import AUTO_K_CURVE_KEY
    from sift.selection.result import _PROXY_CORRELATIONS_ATTR

    selected = list(result.selected_features)
    selected_indices = _coerce_indices(result.selected_indices, label="selected_indices")
    raw_features = _coerce_feature_names(input_features)
    selected_indices = _validate_selected_identity(
        selected,
        selected_indices,
        raw_features,
    )
    metadata = copy.deepcopy(dict(result.selector_metadata))
    raw_width_value = metadata.get("n_features")
    if raw_width_value is not None and (
        isinstance(raw_width_value, (bool, np.bool_))
        or not isinstance(raw_width_value, (int, np.integer))
        or int(raw_width_value) < 0
    ):
        raise ValueError("selector_metadata['n_features'] must be a non-negative integer")
    metadata_width = (
        int(raw_width_value)
        if isinstance(raw_width_value, (int, np.integer))
        and not isinstance(raw_width_value, (bool, np.bool_))
        and int(raw_width_value) >= 0
        else None
    )
    if (
        raw_features is not None
        and metadata_width is not None
        and len(raw_features) != metadata_width
    ):
        raise ValueError(
            "input_features length does not match selector_metadata['n_features']"
        )
    raw_width = len(raw_features) if raw_features is not None else metadata_width
    table, raw_features, complete = _normalize_filter_table(
        result,
        raw_features=raw_features,
        n_raw_features=raw_width,
        indices=selected_indices,
    )
    diagnostics = result.diagnostics_
    curve, curve_metadata = _normalize_auto_k_curve(
        diagnostics.get(AUTO_K_CURVE_KEY) if isinstance(diagnostics, Mapping) else None
    )
    metadata.update(
        {
            "adapter": "FilterSelectionResult",
            "table_complete": complete,
            "input_kind": "unknown",
        }
    )
    metadata.update(curve_metadata)
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=raw_width,
        raw_table=table,
        curve=curve,
        metadata=metadata,
        diagnostics=diagnostics,
        proxy_correlations=getattr(result, _PROXY_CORRELATIONS_ATTR, None),
    )



def _append_rows_like(table: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Append ``rows`` to ``table`` with every column cast to the table's dtype.

    Columns the rows do not mention become missing values of the matching
    dtype.  Aligning dtypes first keeps the concatenation free of pandas'
    all-NA-entry deprecation, which fires whenever an all-missing column would
    otherwise be ignored while inferring the result dtype.
    """
    missing = pd.DataFrame(rows)
    count = len(missing)
    aligned: dict[str, object] = {}
    for column in table.columns:
        dtype = table[column].dtype
        if column in missing.columns:
            values = missing[column]
            if pd.api.types.is_extension_array_dtype(dtype):
                aligned[column] = pd.array(values.tolist(), dtype=dtype)
            elif pd.api.types.is_object_dtype(dtype):
                aligned[column] = pd.Series(values.tolist(), dtype=object)
            elif values.isna().any():
                nullable = table[column].convert_dtypes().dtype
                table = table.astype({column: nullable})
                aligned[column] = pd.array(values.tolist(), dtype=nullable)
            else:
                aligned[column] = values.to_numpy().astype(dtype, copy=False)
        elif pd.api.types.is_extension_array_dtype(dtype):
            aligned[column] = pd.array([pd.NA] * count, dtype=dtype)
        elif pd.api.types.is_object_dtype(dtype):
            aligned[column] = pd.Series([None] * count, dtype=object)
        elif pd.api.types.is_float_dtype(dtype):
            aligned[column] = np.full(count, np.nan, dtype=dtype)
        else:
            nullable = table[column].convert_dtypes().dtype
            table = table.astype({column: nullable})
            aligned[column] = pd.array([pd.NA] * count, dtype=nullable)
    return pd.concat([table, pd.DataFrame(aligned)], ignore_index=True)

def _knockoff_dropped_inputs(metadata: Mapping[str, Any]) -> dict[int, str]:
    """Return ``{raw position: reason}`` for columns knockoffs could not use."""
    positions = metadata.get("dropped_feature_positions")
    reasons = metadata.get("dropped_feature_reasons")
    if positions is None and reasons is None:
        return {}
    if positions is None or reasons is None:
        raise ValueError(
            "knockoff metadata must carry both 'dropped_feature_positions' and "
            "'dropped_feature_reasons'"
        )
    dropped_positions = _coerce_indices(
        positions,
        label="selector_metadata['dropped_feature_positions']",
    )
    reason_list = list(reasons)
    if dropped_positions is None or len(dropped_positions) != len(reason_list):
        raise ValueError(
            "knockoff metadata 'dropped_feature_positions' and "
            "'dropped_feature_reasons' must have the same length"
        )
    dropped = dict(zip(dropped_positions, (str(reason) for reason in reason_list)))
    if len(dropped) != len(dropped_positions):
        raise ValueError(
            "knockoff metadata 'dropped_feature_positions' must be unique"
        )
    return dropped


def _as_knockoff_result(result: Any, input_features: Any) -> SelectionView:
    selected = list(result.selected_features)
    selected_indices = _coerce_indices(result.selected_indices, label="selected_indices")
    raw_features = _coerce_feature_names(input_features)
    selected_indices = _validate_selected_identity(
        selected,
        selected_indices,
        raw_features,
    )
    if not isinstance(result.W, pd.DataFrame):
        raise ValueError("knockoff result W must be a pandas DataFrame")
    required = {"feature", "selected_index", "W", "selected"}
    missing = sorted(required.difference(result.W.columns))
    if missing:
        raise ValueError(f"knockoff result W is missing required columns: {missing}")

    source_metadata = copy.deepcopy(dict(result.selector_metadata))
    # ``n_features`` is the post-screening count the filter ran on; only the
    # additive ``n_features_input`` establishes the caller's raw matrix width.
    raw_width = None
    if source_metadata.get("n_features_input") is not None:
        raw_width = _strict_integer(
            source_metadata["n_features_input"],
            label="selector_metadata['n_features_input']",
            minimum=0,
        )
        if raw_features is not None and len(raw_features) != raw_width:
            raise ValueError(
                "input_features length does not match "
                "selector_metadata['n_features_input']"
            )
    elif raw_features is not None:
        raw_width = len(raw_features)
    dropped_inputs = _knockoff_dropped_inputs(source_metadata)
    if raw_width is not None and any(
        position >= raw_width for position in dropped_inputs
    ):
        raise ValueError(
            "knockoff metadata 'dropped_feature_positions' contains a position "
            "outside the raw input width"
        )

    positions = _coerce_position_series(
        result.W["selected_index"],
        label="knockoff W selected_index",
        allow_missing=False,
    ).astype(int)
    if positions.duplicated().any():
        raise ValueError("knockoff W selected_index values must be unique and non-negative")
    if raw_features is not None and (positions >= len(raw_features)).any():
        raise ValueError("knockoff W contains positions outside input_features")
    if raw_width is not None and (positions >= raw_width).any():
        raise ValueError("knockoff W contains positions outside the raw input width")
    if raw_features is not None:
        for feature, position in zip(result.W["feature"], positions):
            if not _labels_equal(feature, raw_features[int(position)]):
                raise ValueError("knockoff W feature identities do not match input_features")

    selected_values = _coerce_boolean_series(
        result.W["selected"],
        label="knockoff W selected",
    )
    table = pd.DataFrame(
        {
            "feature": result.W["feature"].tolist(),
            "selected_index": pd.array(positions, dtype="Int64"),
        }
    )
    table["path_rank"] = _selection_path_ranks(table, selected, selected_indices)
    table["selected"] = selected_values.to_numpy()
    selected_from_w = set(int(value) for value in positions[selected_values])
    if selected_indices is not None and selected_from_w != set(selected_indices):
        raise ValueError("selected features do not match the W selected mask")
    gain = pd.to_numeric(result.W["W"], errors="raise")
    if not np.isfinite(gain.to_numpy(dtype=float)).all():
        raise ValueError("knockoff W statistics must be finite")
    table["gain"] = gain.to_numpy(copy=True)
    for column in ("relevance", "selection_frequency", "feature_group"):
        if column in result.W:
            values = result.W[column]
            if values.notna().any():
                table[column] = values.to_numpy(copy=True)
    if dropped_inputs:
        present = {int(value) for value in table["selected_index"]}
        # Columns dropped before knockoff construction have no W row at all;
        # give them an explicit positional row instead of leaving a silent gap.
        missing_rows = [
            {
                "feature": raw_features[position] if raw_features is not None else None,
                "selected_index": position,
                "path_rank": pd.NA,
                "selected": False,
            }
            for position in sorted(set(dropped_inputs).difference(present))
        ]
        if missing_rows:
            table = _append_rows_like(table, missing_rows)
        table["reason_dropped"] = [
            dropped_inputs.get(int(position)) for position in table["selected_index"]
        ]
    table = table.sort_values("selected_index", kind="mergesort").reset_index(drop=True)

    covered = {int(value) for value in table["selected_index"]}
    complete = bool(raw_width is not None and covered == set(range(raw_width)))
    metadata = copy.deepcopy(dict(result.selector_metadata))
    metadata.update(
        {
            "adapter": "KnockoffSelectionResult",
            "curve_available": False,
            "table_complete": complete,
            "input_kind": "unknown",
            "threshold": result.threshold,
        }
    )
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=raw_width,
        raw_table=table,
        metadata=metadata,
        diagnostics=result.diagnostics_,
    )


def _as_boruta_result(result: Any, input_features: Any) -> SelectionView:
    feature_names = _coerce_feature_names(result.feature_names)
    if feature_names is None:
        raise ValueError("BorutaResult feature_names must be an ordered iterable")
    n_features = len(feature_names)
    status = _strict_integer_vector(
        result.status,
        label="BorutaResult status",
        length=n_features,
    )
    invalid_status = sorted(set(status.tolist()).difference({-1, 0, 1}))
    if invalid_status:
        raise ValueError(
            f"BorutaResult status values must be -1, 0, or 1; got {invalid_status}"
        )
    n_iter = _strict_integer(result.n_iter, label="BorutaResult n_iter", minimum=0)
    hits = _strict_integer_vector(
        result.hits,
        label="BorutaResult hits",
        length=n_features,
        minimum=0,
    )
    if (hits > n_iter).any():
        raise ValueError("BorutaResult hits values cannot exceed n_iter")
    mean_importance = _numeric_vector(
        result.mean_importance,
        label="BorutaResult mean_importance",
        length=n_features,
    )
    shadow_thresholds = _numeric_vector(
        result.shadow_thresholds,
        label="BorutaResult shadow_thresholds",
        length=n_iter,
    )

    supplied_features = _coerce_feature_names(input_features)
    if supplied_features is not None:
        if len(supplied_features) != n_features:
            raise ValueError(
                "input_features length does not match BorutaResult feature_names"
            )
        if any(
            not _labels_equal(expected, observed)
            for expected, observed in zip(feature_names, supplied_features)
        ):
            raise ValueError(
                "input_features must match BorutaResult feature_names in exact order"
            )
        raw_features = supplied_features
    else:
        raw_features = feature_names

    selected_indices = np.flatnonzero(status == 1).astype(int).tolist()
    selected = [feature_names[index] for index in selected_indices]
    path_rank = pd.Series(
        pd.array([pd.NA] * n_features, dtype="Int64"),
    )
    for rank, index in enumerate(selected_indices, start=1):
        path_rank.iloc[index] = rank
    status_names = {-1: "rejected", 0: "tentative", 1: "accepted"}
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": path_rank,
            "selected": status == 1,
            "gain": mean_importance.copy(),
            "hits": hits.copy(),
            "boruta_status": [status_names[int(value)] for value in status],
        }
    )
    metadata = {
        "adapter": "BorutaResult",
        "selector": "boruta",
        "curve_available": False,
        "table_complete": True,
        "input_kind": "unknown",
        "n_iter": n_iter,
    }
    diagnostics = {
        "n_iter": n_iter,
        "shadow_thresholds": shadow_thresholds.copy(),
    }
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def _path_result_scores(result: Any, tested_k: list[int]) -> dict[int, float]:
    if not isinstance(result.scores, Mapping):
        raise ValueError("FeaturePathEvaluationResult scores must be a mapping")
    scores: dict[int, float] = {}
    for key, value in result.scores.items():
        key_int = _strict_integer(
            key,
            label="FeaturePathEvaluationResult score keys",
            minimum=1,
        )
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (Real, np.integer, np.floating)
        ):
            raise ValueError(
                "FeaturePathEvaluationResult scores must be real non-boolean numbers"
            )
        try:
            score = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "FeaturePathEvaluationResult scores must be real non-boolean numbers "
                "representable as float64"
            ) from exc
        if (
            isinstance(value, np.floating)
            and np.isfinite(value)
            and not math.isfinite(score)
        ):
            raise ValueError(
                "FeaturePathEvaluationResult scores must be representable as float64"
            )
        if math.isnan(score) or score == float("-inf"):
            raise ValueError(
                "FeaturePathEvaluationResult scores must be finite or positive infinity"
            )
        scores[key_int] = score
    if set(scores) != set(tested_k) or len(scores) != len(tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult score keys must match the tested k grid"
        )
    return scores


def _numeric_values_equal(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))


def _validate_path_diagnostics(
    diagnostics: Any,
    *,
    tested_k: list[int],
    scores: Mapping[int, float],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(diagnostics, pd.DataFrame):
        raise ValueError("FeaturePathEvaluationResult diagnostics must be a DataFrame")
    required = {"k", "score", "std", "n_finite", "n_splits", "best_score"}
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics is missing required columns: "
            f"{missing}"
        )
    if len(diagnostics) != len(tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics rows must match the tested k grid"
        )
    diagnostic_k = _strict_integer_vector(
        diagnostics["k"],
        label="FeaturePathEvaluationResult diagnostics k",
        length=len(tested_k),
        minimum=1,
    ).tolist()
    if diagnostic_k != tested_k:
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics k order must match the tested grid"
        )
    diagnostic_scores = _numeric_vector(
        diagnostics["score"],
        label="FeaturePathEvaluationResult diagnostics score",
        length=len(tested_k),
    )
    if any(
        not _numeric_values_equal(score, scores[k])
        for k, score in zip(tested_k, diagnostic_scores)
    ):
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics scores must match scores"
        )
    std = _numeric_vector(
        diagnostics["std"],
        label="FeaturePathEvaluationResult diagnostics std",
        length=len(tested_k),
    )
    if np.isinf(std).any() or (std[np.isfinite(std)] < 0.0).any():
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics std must be non-negative or NaN"
        )
    n_finite = _strict_integer_vector(
        diagnostics["n_finite"],
        label="FeaturePathEvaluationResult diagnostics n_finite",
        length=len(tested_k),
        minimum=0,
    )
    n_splits = _strict_integer_vector(
        diagnostics["n_splits"],
        label="FeaturePathEvaluationResult diagnostics n_splits",
        length=len(tested_k),
        minimum=1,
    )
    if (n_finite > n_splits).any():
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics n_finite cannot exceed n_splits"
        )
    if len(set(n_splits.tolist())) != 1:
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics n_splits must be constant"
        )
    for score, spread, finite, splits in zip(
        diagnostic_scores, std, n_finite, n_splits
    ):
        if np.isfinite(score):
            if finite != splits or not np.isfinite(spread):
                raise ValueError(
                    "finite FeaturePathEvaluationResult scores require every split "
                    "to be finite and std to be finite"
                )
            if finite == 1 and spread != 0.0:
                raise ValueError(
                    "a finite single-split FeaturePathEvaluationResult must have std 0"
                )
            continue
        if score != float("inf") or finite >= splits:
            raise ValueError(
                "infinite FeaturePathEvaluationResult scores require at least one "
                "failed split"
            )
        if (finite > 1 and not np.isfinite(spread)) or (
            finite <= 1 and not np.isnan(spread)
        ):
            raise ValueError(
                "FeaturePathEvaluationResult diagnostics std is inconsistent with "
                "n_finite"
            )
    return diagnostics.copy(deep=True), diagnostic_scores, std, n_finite


def _resolve_path_positions(
    feature_path: list[Any],
    raw_features: list[Any],
) -> list[int]:
    raw_tokens = [_label_token(feature) for feature in raw_features]
    positions: list[int] = []
    for feature in feature_path:
        token = _label_token(feature)
        matches = [
            index for index, candidate in enumerate(raw_tokens) if candidate == token
        ]
        if len(matches) != 1:
            raise ValueError(
                f"feature_path feature {feature!r} is missing or ambiguous in input_features; "
                "FeaturePathEvaluationResult does not retain raw positions"
            )
        positions.append(matches[0])
    if len(set(positions)) != len(positions):
        raise ValueError(
            "feature_path entries do not map to unique positions in input_features"
        )
    return positions


def _as_feature_path_result(result: Any, input_features: Any) -> SelectionView:
    feature_path = _coerce_feature_names(result.feature_path)
    selected = _coerce_feature_names(result.features)
    if feature_path is None or not feature_path:
        raise ValueError("FeaturePathEvaluationResult feature_path must be non-empty")
    if selected is None:
        raise ValueError(
            "FeaturePathEvaluationResult features must be an ordered iterable"
        )

    tested_k = _coerce_indices(result.k, label="FeaturePathEvaluationResult k")
    if tested_k is None or not tested_k or any(k < 1 for k in tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult k must contain unique positive integers"
        )
    if any(k > len(feature_path) for k in tested_k):
        raise ValueError(
            "FeaturePathEvaluationResult k cannot exceed the feature_path length"
        )
    scores = _path_result_scores(result, tested_k)
    diagnostics, diagnostic_scores, std, n_finite = _validate_path_diagnostics(
        result.diagnostics,
        tested_k=tested_k,
        scores=scores,
    )

    finite_candidates = [
        (score, k) for k, score in scores.items() if np.isfinite(score)
    ]
    if finite_candidates:
        expected_best_score, expected_best_k = min(
            finite_candidates,
            key=lambda item: (item[0], item[1]),
        )
    else:
        expected_best_k = 0
        expected_best_score = float("nan")
    best_k = _strict_integer(
        result.best_k,
        label="FeaturePathEvaluationResult best_k",
        minimum=0,
    )
    if best_k != expected_best_k:
        raise ValueError(
            "FeaturePathEvaluationResult best_k does not match the lower-is-better scores"
        )
    expected_features = feature_path[:best_k]
    if len(selected) != best_k or any(
        not _labels_equal(expected, observed)
        for expected, observed in zip(expected_features, selected)
    ):
        raise ValueError(
            "FeaturePathEvaluationResult features must equal feature_path[:best_k]"
        )
    best_score_values = _numeric_vector(
        diagnostics["best_score"],
        label="FeaturePathEvaluationResult diagnostics best_score",
        length=len(tested_k),
    )
    if any(
        not _numeric_values_equal(value, expected_best_score)
        for value in best_score_values
    ):
        raise ValueError(
            "FeaturePathEvaluationResult diagnostics best_score is inconsistent"
        )

    criterion_se = np.full(len(tested_k), np.nan, dtype=np.float64)
    for index, (score, spread, finite) in enumerate(
        zip(diagnostic_scores, std, n_finite)
    ):
        if np.isfinite(score) and np.isfinite(spread) and finite >= 2:
            criterion_se[index] = float(spread) / math.sqrt(int(finite) - 1)
    curve = pd.DataFrame(
        {
            "k": tested_k,
            "criterion": diagnostic_scores.copy(),
            "criterion_se": criterion_se,
            "selected": [best_k > 0 and k == best_k for k in tested_k],
        }
    )

    raw_features = _coerce_feature_names(input_features)
    if raw_features is None:
        selected_indices = None
        table = pd.DataFrame(
            {
                "feature": feature_path,
                "selected_index": pd.array([pd.NA] * len(feature_path), dtype="Int64"),
                "path_rank": pd.array(
                    list(range(1, best_k + 1)) + [pd.NA] * (len(feature_path) - best_k),
                    dtype="Int64",
                ),
                "selected": [index < best_k for index in range(len(feature_path))],
                "feature_path_rank": pd.array(
                    range(1, len(feature_path) + 1), dtype="Int64"
                ),
            }
        )
        table_complete = False
        n_raw_features = None
    else:
        path_positions = _resolve_path_positions(feature_path, raw_features)
        selected_indices = path_positions[:best_k]
        n_raw_features = len(raw_features)
        path_rank = pd.Series(pd.array([pd.NA] * n_raw_features, dtype="Int64"))
        feature_path_rank = pd.Series(pd.array([pd.NA] * n_raw_features, dtype="Int64"))
        for rank, position in enumerate(path_positions, start=1):
            feature_path_rank.iloc[position] = rank
        for rank, position in enumerate(selected_indices, start=1):
            path_rank.iloc[position] = rank
        selected_mask = np.zeros(n_raw_features, dtype=bool)
        selected_mask[selected_indices] = True
        table = pd.DataFrame(
            {
                "feature": raw_features,
                "selected_index": pd.array(range(n_raw_features), dtype="Int64"),
                "path_rank": path_rank,
                "selected": selected_mask,
                "feature_path_rank": feature_path_rank,
            }
        )
        table_complete = True

    metadata = {
        "adapter": "FeaturePathEvaluationResult",
        "curve_available": True,
        "criterion_direction": "minimize",
        "best_k": best_k,
        "best_score": expected_best_score,
        "tested_k": tested_k,
        "table_complete": table_complete,
        "input_kind": "unknown",
        "criterion_se_definition": "population_std/sqrt(n_finite-1)",
    }
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=n_raw_features,
        raw_table=table,
        curve=curve,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def _catboost_label_key(value: Any) -> str:
    return json.dumps(
        _label_token(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _catboost_feature_list(
    values: Any,
    *,
    label: str,
    allow_none: bool = False,
) -> list[Any] | None:
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{label} must be an ordered iterable")
    try:
        features = _coerce_feature_names(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ordered one-dimensional iterable") from exc
    if features is None:
        raise ValueError(f"{label} must be an ordered iterable")
    keys = [_catboost_label_key(feature) for feature in features]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} must contain unique feature identities")
    return features


def _catboost_float(
    value: Any,
    *,
    label: str,
    finite: bool,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a real non-boolean number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be representable as float64") from exc
    if (
        isinstance(value, np.floating)
        and np.isfinite(value)
        and not math.isfinite(converted)
    ):
        raise ValueError(f"{label} must be representable as float64")
    if finite and not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _catboost_score_mapping(values: Any, *, label: str) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    scores: dict[int, float] = {}
    for key, value in values.items():
        k = _strict_integer(key, label=f"{label} keys", minimum=1)
        if k in scores:
            raise ValueError(f"{label} keys must be unique positive integers")
        scores[k] = _catboost_float(value, label=f"{label} values", finite=True)
    return scores


def _catboost_std_mapping(values: Any, *, score_keys: set[int]) -> dict[int, float]:
    std_by_k = _catboost_score_mapping(values, label="CatBoost scores_std_by_k")
    extra = sorted(set(std_by_k).difference(score_keys))
    if extra:
        raise ValueError(
            "CatBoost scores_std_by_k keys must be present in scores_by_k; "
            f"unexpected {extra}"
        )
    if any(value < 0.0 for value in std_by_k.values()):
        raise ValueError("CatBoost scores_std_by_k values must be non-negative")
    return std_by_k


def _catboost_all_scores(values: Any) -> dict[int, list[float]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError("CatBoost all_scores must be a mapping or None")
    all_scores: dict[int, list[float]] = {}
    for key, raw_values in values.items():
        k = _strict_integer(key, label="CatBoost all_scores keys", minimum=1)
        if k in all_scores:
            raise ValueError(
                "CatBoost all_scores keys must be unique positive integers"
            )
        if isinstance(raw_values, (str, bytes, bytearray, Mapping, Set)):
            raise ValueError("CatBoost all_scores values must be ordered iterables")
        try:
            score_values = list(raw_values)
        except TypeError as exc:
            raise ValueError(
                "CatBoost all_scores values must be ordered iterables"
            ) from exc
        all_scores[k] = [
            _catboost_float(
                value,
                label="CatBoost all_scores observations",
                finite=False,
            )
            for value in score_values
        ]
    return all_scores


def _catboost_features_by_k(values: Any) -> dict[int, list[Any]]:
    if not isinstance(values, Mapping):
        raise ValueError("CatBoost features_by_k must be a mapping")
    features_by_k: dict[int, list[Any]] = {}
    for key, raw_features in values.items():
        k = _strict_integer(key, label="CatBoost features_by_k keys", minimum=1)
        if k in features_by_k:
            raise ValueError(
                "CatBoost features_by_k keys must be unique positive integers"
            )
        features = _catboost_feature_list(
            raw_features,
            label=f"CatBoost features_by_k[{k}]",
        )
        assert features is not None
        if len(features) != k:
            raise ValueError(
                f"CatBoost features_by_k[{k}] must contain exactly {k} features"
            )
        features_by_k[k] = features
    return features_by_k


def _catboost_numeric_series(
    values: Any,
    *,
    label: str,
    allow_none: bool = False,
    unit_interval: bool = False,
) -> pd.Series | None:
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{label} must be a pandas Series")
    if not isinstance(values, pd.Series):
        raise ValueError(f"{label} must be a pandas Series")
    features = _catboost_feature_list(values.index, label=f"{label} index")
    assert features is not None
    numeric = np.asarray(
        [
            _catboost_float(value, label=f"{label} values", finite=True)
            for value in values.tolist()
        ],
        dtype=np.float64,
    )
    if unit_interval and ((numeric < 0.0).any() or (numeric > 1.0).any()):
        raise ValueError(f"{label} values must be between 0 and 1")
    return pd.Series(numeric, index=pd.Index(features), name=values.name)


def _catboost_curve(
    *,
    scores_by_k: Mapping[int, float],
    scores_std_by_k: Mapping[int, float],
    all_scores: Mapping[int, list[float]] | None,
    best_k: int,
) -> pd.DataFrame:
    if not scores_by_k:
        raise ValueError("CatBoost scores_by_k must contain at least one finite score")
    if best_k not in scores_by_k:
        raise ValueError("CatBoost best_k must be present in scores_by_k")

    criterion_se: dict[int, float] = {k: float("nan") for k in scores_by_k}
    if all_scores is not None:
        for k, observations in all_scores.items():
            finite_values = np.asarray(observations, dtype=np.float64)
            finite_values = finite_values[np.isfinite(finite_values)]
            if k not in scores_by_k:
                if finite_values.size:
                    raise ValueError(
                        "CatBoost all_scores with finite observations must have a "
                        "matching scores_by_k entry"
                    )
                continue
            if not finite_values.size:
                raise ValueError(
                    "CatBoost all_scores must contain a finite observation for each "
                    "stored score"
                )
            mean = float(np.mean(finite_values))
            if not np.isclose(mean, scores_by_k[k], rtol=1e-12, atol=1e-15):
                raise ValueError(
                    "CatBoost scores_by_k must match the finite all_scores mean"
                )
            spread = float(np.std(finite_values))
            if k in scores_std_by_k and not np.isclose(
                spread,
                scores_std_by_k[k],
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "CatBoost scores_std_by_k must match the finite all_scores "
                    "population standard deviation"
                )
            if finite_values.size >= 2:
                criterion_se[k] = spread / math.sqrt(int(finite_values.size) - 1)

    ks = sorted(scores_by_k)
    return pd.DataFrame(
        {
            "k": ks,
            "criterion": [scores_by_k[k] for k in ks],
            "criterion_se": [criterion_se[k] for k in ks],
            "selected": [k == best_k for k in ks],
        }
    )


def _catboost_known_features(
    *,
    selected: list[Any],
    features_by_k: Mapping[int, list[Any]],
    feature_importances: pd.Series,
    stability_scores: pd.Series | None,
    prefilter_features: list[Any] | None,
) -> list[Any]:
    known: list[Any] = []
    seen: set[str] = set()

    def extend(values: Iterable[Any]) -> None:
        for feature in values:
            key = _catboost_label_key(feature)
            if key not in seen:
                seen.add(key)
                known.append(feature)

    extend(selected)
    for k in sorted(features_by_k, reverse=True):
        extend(features_by_k[k])
    if stability_scores is not None:
        extend(stability_scores.index.tolist())
    extend(feature_importances.index.tolist())
    if prefilter_features is not None:
        extend(prefilter_features)
    return known


def _catboost_resolve_positions(
    known_features: list[Any],
    raw_features: list[Any],
) -> dict[str, int]:
    raw_keys = [_catboost_label_key(feature) for feature in raw_features]
    positions: dict[str, int] = {}
    for feature in known_features:
        key = _catboost_label_key(feature)
        matches = [index for index, candidate in enumerate(raw_keys) if candidate == key]
        if len(matches) != 1:
            raise ValueError(
                f"CatBoost feature {feature!r} is missing or ambiguous in input_features"
            )
        positions[key] = matches[0]
    return positions


def _as_catboost_result(result: Any, input_features: Any) -> SelectionView:
    selected = _catboost_feature_list(
        result.selected_features,
        label="CatBoost selected_features",
    )
    assert selected is not None
    if not selected:
        raise ValueError("CatBoost selected_features must be non-empty")
    best_k = _strict_integer(result.best_k, label="CatBoost best_k", minimum=1)
    if len(selected) > best_k:
        raise ValueError("CatBoost selected_features cannot contain more than best_k")

    scores_by_k = _catboost_score_mapping(
        result.scores_by_k,
        label="CatBoost scores_by_k",
    )
    if not scores_by_k:
        raise ValueError("CatBoost scores_by_k must contain at least one finite score")
    scores_std_by_k = _catboost_std_mapping(
        result.scores_std_by_k,
        score_keys=set(scores_by_k),
    )
    all_scores = _catboost_all_scores(result.all_scores)
    curve = _catboost_curve(
        scores_by_k=scores_by_k,
        scores_std_by_k=scores_std_by_k,
        all_scores=all_scores,
        best_k=best_k,
    )
    features_by_k = _catboost_features_by_k(result.features_by_k)
    feature_importances = _catboost_numeric_series(
        result.feature_importances,
        label="CatBoost feature_importances",
    )
    assert feature_importances is not None
    stability_scores = _catboost_numeric_series(
        result.stability_scores,
        label="CatBoost stability_scores",
        allow_none=True,
        unit_interval=True,
    )
    prefilter_features = _catboost_feature_list(
        result.prefilter_features,
        label="CatBoost prefilter_features",
        allow_none=True,
    )

    selected_keys = {_catboost_label_key(feature) for feature in selected}
    if best_k in features_by_k and not selected_keys.issubset(
        {_catboost_label_key(feature) for feature in features_by_k[best_k]}
    ):
        raise ValueError(
            "CatBoost selected_features must be contained in features_by_k[best_k]"
        )
    if not feature_importances.empty and selected_keys != {
        _catboost_label_key(feature) for feature in feature_importances.index
    }:
        raise ValueError(
            "non-empty CatBoost feature_importances must cover selected_features exactly"
        )
    if stability_scores is not None and not selected_keys.issubset(
        {_catboost_label_key(feature) for feature in stability_scores.index}
    ):
        raise ValueError(
            "CatBoost selected_features must be present in stability_scores"
        )

    if not isinstance(result.metric, str) or not result.metric:
        raise ValueError("CatBoost metric must be a non-empty string")
    if not isinstance(result.higher_is_better, (bool, np.bool_)):
        raise ValueError("CatBoost higher_is_better must be boolean")
    higher_is_better = bool(result.higher_is_better)
    selection_patience = _strict_integer(
        result.selection_patience,
        label="CatBoost selection_patience",
        minimum=1,
    )

    known_features = _catboost_known_features(
        selected=selected,
        features_by_k=features_by_k,
        feature_importances=feature_importances,
        stability_scores=stability_scores,
        prefilter_features=prefilter_features,
    )
    raw_features = _coerce_feature_names(input_features)
    if raw_features is None:
        selected_indices = None
        table_features = known_features
        selected_index = pd.array([pd.NA] * len(table_features), dtype="Int64")
        table_complete = False
        n_raw_features = None
        row_positions = {
            _catboost_label_key(feature): index
            for index, feature in enumerate(table_features)
        }
    else:
        raw_positions = _catboost_resolve_positions(known_features, raw_features)
        selected_indices = [
            raw_positions[_catboost_label_key(feature)] for feature in selected
        ]
        table_features = raw_features
        selected_index = pd.array(range(len(raw_features)), dtype="Int64")
        table_complete = True
        n_raw_features = len(raw_features)
        row_positions = raw_positions

    selected_ranks = {
        _catboost_label_key(feature): rank
        for rank, feature in enumerate(selected, start=1)
    }
    path_rank = pd.Series(pd.array([pd.NA] * len(table_features), dtype="Int64"))
    selected_mask = np.zeros(len(table_features), dtype=bool)
    for feature in selected:
        key = _catboost_label_key(feature)
        row = row_positions[key]
        path_rank.iloc[row] = selected_ranks[key]
        selected_mask[row] = True
    table = pd.DataFrame(
        {
            "feature": table_features,
            "selected_index": selected_index,
            "path_rank": path_rank,
            "selected": selected_mask,
        }
    )

    def add_metric(column: str, values: pd.Series) -> None:
        if values.empty:
            return
        metric_values = np.full(len(table), np.nan, dtype=np.float64)
        for feature, value in values.items():
            metric_values[row_positions[_catboost_label_key(feature)]] = float(value)
        table[column] = metric_values

    add_metric("gain", feature_importances)
    if stability_scores is not None:
        add_metric("selection_frequency", stability_scores)
    if prefilter_features is not None:
        prefiltered = {
            _catboost_label_key(feature) for feature in prefilter_features
        }
        table["prefiltered_first_split"] = [
            _catboost_label_key(feature) in prefiltered for feature in table_features
        ]

    if higher_is_better:
        best_scoring_k = min(
            scores_by_k,
            key=lambda k: (-scores_by_k[k], k),
        )
    else:
        best_scoring_k = min(
            scores_by_k,
            key=lambda k: (scores_by_k[k], k),
        )
    metadata = {
        "adapter": "CatBoostSelectionResult",
        "selector": "catboost",
        "curve_available": True,
        "criterion_direction": "maximize" if higher_is_better else "minimize",
        "criterion_se_definition": "population_std/sqrt(n_finite-1)",
        "metric": result.metric,
        "higher_is_better": higher_is_better,
        "target_k": best_k,
        "selected_feature_count": len(selected),
        "best_scoring_k": best_scoring_k,
        "best_scoring_score": scores_by_k[best_scoring_k],
        "gain_source": "final_model_feature_importance",
        "table_complete": table_complete,
        "input_kind": "unknown",
    }
    diagnostics = {
        "scores_std_by_k": scores_std_by_k,
        "all_scores": all_scores,
        "features_by_k": features_by_k,
        "stability_scores": stability_scores,
        "stability_scope": (
            "target_k_split_frequency" if stability_scores is not None else None
        ),
        "prefilter_features": prefilter_features,
        "prefilter_scope": (
            "first_split_only" if prefilter_features is not None else None
        ),
        "selection_patience": selection_patience,
    }
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=n_raw_features,
        raw_table=table,
        curve=curve,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def _as_stability_selector(selector: Any, input_features: Any) -> SelectionView:
    from sklearn.utils.validation import check_is_fitted

    required = [
        "feature_names_in_",
        "n_features_in_",
        "selected_features_",
        "selected_feature_names_",
        "n_features_selected_",
        "selection_frequencies_",
        "mean_abs_coef_",
        "alpha_",
        "alpha_rule_effective_",
    ]
    check_is_fitted(selector, required)

    n_features = _strict_integer(
        selector.n_features_in_,
        label="StabilitySelector n_features_in_",
        minimum=1,
    )
    raw_features = _coerce_feature_names(selector.feature_names_in_)
    if raw_features is None or len(raw_features) != n_features:
        raise ValueError(
            "StabilitySelector feature_names_in_ must match n_features_in_"
        )
    raw_values = np.empty(n_features, dtype=object)
    raw_values[:] = raw_features
    raw_index = pd.Index(raw_values, dtype=object, tupleize_cols=False)
    if raw_index.duplicated().any():
        raise ValueError("StabilitySelector feature_names_in_ must be unique")

    supplied_features = _coerce_feature_names(input_features)
    if supplied_features is not None:
        if len(supplied_features) != n_features or any(
            not _labels_equal(expected, observed)
            for expected, observed in zip(raw_features, supplied_features)
        ):
            raise ValueError(
                "input_features must match StabilitySelector feature_names_in_ "
                "in exact order"
            )

    selected_indices = _coerce_indices(
        selector.selected_features_,
        label="StabilitySelector selected_features_",
    )
    assert selected_indices is not None
    selected = _coerce_feature_names(selector.selected_feature_names_)
    if selected is None:
        raise ValueError(
            "StabilitySelector selected_feature_names_ must be an ordered iterable"
        )
    selected_indices = _validate_selected_identity(
        selected,
        selected_indices,
        raw_features,
    )
    if any(index >= n_features for index in selected_indices):
        raise ValueError(
            "StabilitySelector selected_features_ contains an out-of-bounds position"
        )
    selected_count = _strict_integer(
        selector.n_features_selected_,
        label="StabilitySelector n_features_selected_",
        minimum=0,
    )
    if selected_count != len(selected_indices):
        raise ValueError(
            "StabilitySelector n_features_selected_ must match selected_features_"
        )

    frequencies = _numeric_vector(
        selector.selection_frequencies_,
        label="StabilitySelector selection_frequencies_",
        length=n_features,
    )
    if not np.isfinite(frequencies).all() or (
        (frequencies < 0.0) | (frequencies > 1.0)
    ).any():
        raise ValueError(
            "StabilitySelector selection_frequencies_ must be finite values in [0, 1]"
        )
    mean_abs_coef = _numeric_vector(
        selector.mean_abs_coef_,
        label="StabilitySelector mean_abs_coef_",
        length=n_features,
    )
    if not np.isfinite(mean_abs_coef).all() or (mean_abs_coef < 0.0).any():
        raise ValueError(
            "StabilitySelector mean_abs_coef_ must contain finite non-negative values"
        )
    mean_abs_coef_values = np.asarray(selector.mean_abs_coef_).copy()

    if selector.task not in {"regression", "classification"}:
        raise ValueError(
            "StabilitySelector task must be 'regression' or 'classification'"
        )
    if not isinstance(selector.threshold, Real):
        raise ValueError("StabilitySelector threshold must be a real value in [0, 1]")
    threshold = float(selector.threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("StabilitySelector threshold must be a finite value in [0, 1]")
    if selector.max_features is None:
        max_features = None
    else:
        max_features = _strict_integer(
            selector.max_features,
            label="StabilitySelector max_features",
            minimum=1,
        )
    if not isinstance(selector.alpha_, Real):
        raise ValueError("StabilitySelector alpha_ must be a finite positive value")
    alpha = float(selector.alpha_)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("StabilitySelector alpha_ must be a finite positive value")
    if selector.alpha_rule_effective_ not in {"fixed", "one_se", "best"}:
        raise ValueError(
            "StabilitySelector alpha_rule_effective_ must be 'fixed', 'one_se', or 'best'"
        )
    if not isinstance(selector.coef_threshold, Real):
        raise ValueError(
            "StabilitySelector coef_threshold must be a finite non-negative value"
        )
    coef_threshold = float(selector.coef_threshold)
    if not math.isfinite(coef_threshold) or coef_threshold < 0.0:
        raise ValueError(
            "StabilitySelector coef_threshold must be a finite non-negative value"
        )
    n_bootstrap_requested = _strict_integer(
        selector.n_bootstrap,
        label="StabilitySelector n_bootstrap",
        minimum=1,
    )

    expected_mask = frequencies >= threshold
    if max_features is not None and expected_mask.sum() > max_features:
        top_indices = np.argsort(-frequencies, kind="mergesort")[:max_features]
        expected_mask = np.zeros(n_features, dtype=bool)
        expected_mask[top_indices] = True
    expected_indices = np.flatnonzero(expected_mask)
    expected_indices = expected_indices[
        np.argsort(-frequencies[expected_indices], kind="mergesort")
    ].tolist()
    if selected_indices != expected_indices:
        raise ValueError(
            "StabilitySelector selected_features_ is inconsistent with its "
            "threshold, max_features, and selection_frequencies_"
        )
    selected_mask = np.zeros(n_features, dtype=bool)
    selected_mask[selected_indices] = True

    # The fitted selector's ``output_order`` drives transform, get_support,
    # and get_feature_names_out.  The view must not silently disagree with it,
    # so features, indices, and path_rank all follow the same order.
    output_order = validate_output_order(selector.output_order)
    view_indices = [int(index) for index in ordered_indices(selected_indices, output_order)]
    view_selected = [raw_features[index] for index in view_indices]

    path_rank = pd.Series(
        pd.array([pd.NA] * n_features, dtype="Int64"),
    )
    for rank, index in enumerate(view_indices, start=1):
        path_rank.iloc[index] = rank
    table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": path_rank,
            "selected": selected_mask,
            "selection_frequency": frequencies.copy(),
            "mean_abs_coef": mean_abs_coef_values.copy(),
            "gain": mean_abs_coef_values.copy(),
        }
    )

    has_generated_marker = hasattr(selector, "_fit_feature_names_generated_")
    has_input_kind_marker = hasattr(selector, "_fit_input_kind_")
    if not (has_generated_marker and has_input_kind_marker):
        generated_names = False
        input_kind = "unknown"
    else:
        generated_names = selector._fit_feature_names_generated_
        fit_input_kind = selector._fit_input_kind_
        if not isinstance(generated_names, (bool, np.bool_)):
            raise ValueError(
                "StabilitySelector _fit_feature_names_generated_ must be boolean"
            )
        if fit_input_kind not in {"dataframe", "positional"}:
            raise ValueError(
                "StabilitySelector _fit_input_kind_ must be 'dataframe' or 'positional'"
            )
        generated_names = bool(generated_names)
        input_kind = fit_input_kind
    generated_names = bool(generated_names)

    # Freeze only the fitted state used by transform.  This preserves sklearn's
    # set_output wrapping without retaining X, coefficient matrices, callbacks,
    # or a live reference whose behavior could change after refit.
    transform_selector = type(selector)()
    # Match the fitted selector's public dtype contract: sklearn requires
    # ``feature_names_in_`` to be a one-dimensional object ndarray.
    transform_selector.feature_names_in_ = raw_values.copy()
    transform_selector.n_features_in_ = n_features
    transform_selector.selected_features_ = np.asarray(
        selected_indices, dtype=np.int64
    )
    transform_selector.selected_feature_names_ = list(selected)
    transform_selector.output_order = output_order
    transform_selector._fit_feature_names_generated_ = generated_names
    if hasattr(selector, "_sklearn_output_config"):
        transform_selector._sklearn_output_config = copy.deepcopy(
            selector._sklearn_output_config
        )

    coefs_available = hasattr(selector, "coef_bootstrap_")
    completed_bootstraps = None
    coef_shape = None
    if coefs_available:
        coef_matrix = np.asarray(selector.coef_bootstrap_)
        if (
            coef_matrix.ndim != 2
            or coef_matrix.shape[0] < 1
            or coef_matrix.shape[1] != n_features
            or not np.issubdtype(coef_matrix.dtype, np.number)
            or not np.isfinite(coef_matrix).all()
        ):
            raise ValueError(
                "StabilitySelector coef_bootstrap_ must be a finite numeric matrix "
                "with one column per fitted feature"
            )
        completed_bootstraps = int(coef_matrix.shape[0])
        coef_shape = tuple(int(value) for value in coef_matrix.shape)

    metadata = {
        "adapter": "StabilitySelector",
        "selector": "stability",
        "curve_available": False,
        "table_complete": True,
        "input_kind": input_kind,
        "raw_namespace": "fitted_candidate_features",
        "output_order": output_order,
        "task": selector.task,
        "threshold": threshold,
        "max_features": max_features,
        "selected_feature_count": selected_count,
        "alpha": alpha,
        "alpha_rule_effective": selector.alpha_rule_effective_,
        "coef_threshold": coef_threshold,
        "gain_source": "mean_abs_coef",
        "selection_frequency_source": "completed_bootstrap_fraction",
        "n_bootstrap_requested": n_bootstrap_requested,
        "coefs_available": coefs_available,
    }
    diagnostics = {
        "fit_context": {
            "sample_weight": bool(
                getattr(selector, "_fit_used_sample_weight_", False)
            ),
            "groups": bool(getattr(selector, "_fit_used_groups_", False)),
            "time": bool(getattr(selector, "_fit_used_time_", False)),
            "smart_sampler": bool(selector.use_smart_sampler),
        },
        "sampled_n": getattr(selector, "sampled_n_", None),
        "coefs_available": coefs_available,
        "coef_bootstrap_shape": coef_shape,
        "completed_bootstraps": completed_bootstraps,
    }
    return SelectionView(
        features=view_selected,
        indices=view_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        metadata=metadata,
        diagnostics=diagnostics,
        transformer=transform_selector.transform,
    )


def _as_importance_result(result: Any, input_features: Any) -> SelectionView:
    snapshot = result._adapter_snapshot()
    ranking = snapshot["ranking"]
    if not isinstance(ranking, pd.DataFrame):
        raise ValueError("ImportanceResult ranking must be a pandas DataFrame")
    expected_columns = [
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    ]
    if list(ranking.columns) != expected_columns:
        raise ValueError(
            "ImportanceResult ranking must have exactly the legacy columns "
            f"{expected_columns}"
        )

    feature_names = _coerce_feature_names(snapshot["feature_names"])
    if feature_names is None:
        raise ValueError("ImportanceResult feature_names must be an ordered iterable")
    n_features = len(feature_names)
    if len(ranking) != n_features:
        raise ValueError(
            "ImportanceResult ranking rows must match the original feature width"
        )
    ranking_indices = _coerce_indices(
        snapshot["ranking_indices"],
        label="ImportanceResult ranking_indices",
    )
    if ranking_indices is None or set(ranking_indices) != set(range(n_features)):
        raise ValueError(
            "ImportanceResult ranking_indices must contain every raw feature position"
        )
    for row, position in enumerate(ranking_indices):
        if not _labels_equal(ranking["feature"].iloc[row], feature_names[position]):
            raise ValueError(
                "ImportanceResult ranking feature identities do not match ranking_indices"
            )

    metadata = snapshot["selector_metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("ImportanceResult selector_metadata must be a mapping")
    metadata = copy.deepcopy(dict(metadata))
    if metadata.get("selector") != "permutation_importance":
        raise ValueError(
            "ImportanceResult selector_metadata must identify permutation_importance"
        )
    metadata_n_features = _strict_integer(
        metadata.get("n_features"),
        label="ImportanceResult selector_metadata n_features",
        minimum=0,
    )
    if metadata_n_features != n_features:
        raise ValueError(
            "ImportanceResult selector_metadata n_features must match feature_names"
        )
    n_repeats = _strict_integer(
        metadata.get("n_repeats"),
        label="ImportanceResult selector_metadata n_repeats",
        minimum=1,
    )
    input_kind = metadata.get("input_kind")
    if input_kind not in {"dataframe", "positional"}:
        raise ValueError(
            "ImportanceResult selector_metadata input_kind must be 'dataframe' or "
            "'positional'"
        )
    if metadata.get("selection_semantics") != "ranking_only":
        raise ValueError(
            "ImportanceResult selector_metadata selection_semantics must be "
            "'ranking_only'"
        )

    try:
        raw_importances = np.asarray(snapshot["importances"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "ImportanceResult importances_ must be a numeric matrix"
        ) from exc
    if raw_importances.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            "ImportanceResult importances_ must contain real non-boolean numeric values"
        )
    importances = raw_importances.astype(np.float64, copy=True)
    if importances.ndim != 2 or importances.shape != (n_features, n_repeats):
        raise ValueError(
            "ImportanceResult importances_ shape must be "
            "(n_features, n_repeats)"
        )

    ranking_means = _numeric_vector(
        ranking["importance_mean"],
        label="ImportanceResult importance_mean",
        length=n_features,
    )
    ranking_stds = _numeric_vector(
        ranking["importance_std"],
        label="ImportanceResult importance_std",
        length=n_features,
    )
    ranking_baselines = _numeric_vector(
        ranking["baseline_score"],
        label="ImportanceResult baseline_score",
        length=n_features,
    )
    baseline_value = snapshot["baseline_score"]
    if isinstance(baseline_value, (bool, np.bool_)) or not isinstance(
        baseline_value,
        (Real, np.integer, np.floating),
    ):
        raise ValueError("ImportanceResult baseline_score must be a real number")
    baseline_score = float(baseline_value)
    if not np.allclose(
        ranking_baselines,
        baseline_score,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError(
            "ImportanceResult ranking baseline_score must match baseline_score"
        )
    raw_means = np.mean(importances, axis=1)
    raw_stds = np.std(importances, axis=1)
    if not np.allclose(
        ranking_means,
        raw_means[ranking_indices],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ) or not np.allclose(
        ranking_stds,
        raw_stds[ranking_indices],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError(
            "ImportanceResult ranking mean/std must match importances_ using ddof=0"
        )

    supplied_features = _coerce_feature_names(input_features)
    if input_kind == "dataframe":
        if input_features is not None and not result._matches_original_features(
            input_features
        ):
            raise ValueError(
                "input_features must match ImportanceResult DataFrame columns in exact order"
            )
        raw_features = feature_names
    else:
        if any(
            not _labels_equal(feature, position)
            for position, feature in enumerate(feature_names)
        ):
            raise ValueError(
                "positional ImportanceResult feature_names must equal raw positions"
            )
        if supplied_features is not None and len(supplied_features) != n_features:
            raise ValueError(
                "input_features length must match the positional ImportanceResult width"
            )
        raw_features = feature_names if supplied_features is None else supplied_features

    selected = [raw_features[position] for position in ranking_indices]
    rank_by_position = np.empty(n_features, dtype=np.int64)
    for rank, position in enumerate(ranking_indices, start=1):
        rank_by_position[position] = rank
    table = pd.DataFrame(
        {
            "feature": raw_features,
            "selected_index": pd.array(range(n_features), dtype="Int64"),
            "path_rank": pd.array(rank_by_position, dtype="Int64"),
            "selected": np.ones(n_features, dtype=bool),
            "gain": raw_means.copy(),
            "importance_mean": raw_means.copy(),
            "importance_std": raw_stds.copy(),
            "baseline_score": baseline_score,
        }
    )
    result_diagnostics = snapshot["diagnostics"]
    if not isinstance(result_diagnostics, Mapping):
        raise ValueError("ImportanceResult diagnostics_ must be a mapping")
    diagnostics = copy.deepcopy(dict(result_diagnostics))
    diagnostics["permutation_importance_repeats"] = importances.copy()
    metadata.update(
        {
            "adapter": "ImportanceResult",
            "curve_available": False,
            "gain_source": "permutation_importance_mean",
            "table_complete": True,
            "input_kind": input_kind,
        }
    )
    return SelectionView(
        features=selected,
        indices=ranking_indices,
        raw_features=raw_features,
        n_raw_features=n_features,
        raw_table=table,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def as_result(obj: Any, input_features: Any = None) -> SelectionView:
    """Return an additive :class:`SelectionView` for a supported SIFT result.

    Passing an existing view is an identity operation.  Legacy list and tuple
    returns are intentionally not guessed; request the corresponding result
    object with ``return_result=True`` first.
    """

    if isinstance(obj, SelectionView):
        if input_features is not None:
            raise ValueError("input_features cannot be supplied when obj is already a SelectionView")
        return obj

    from sift.boruta import BorutaResult
    from sift.importance import ImportanceResult
    from sift.selection.knockoff_filter import KnockoffSelectionResult
    from sift.selection.path_eval import FeaturePathEvaluationResult
    from sift.selection.result import FilterSelectionResult

    if type(obj) is FilterSelectionResult:
        return _as_filter_result(obj, input_features)
    if type(obj) is KnockoffSelectionResult:
        return _as_knockoff_result(obj, input_features)
    if type(obj) is BorutaResult:
        return _as_boruta_result(obj, input_features)
    if type(obj) is FeaturePathEvaluationResult:
        return _as_feature_path_result(obj, input_features)
    if type(obj) is ImportanceResult:
        return _as_importance_result(obj, input_features)
    obj_type = type(obj)
    if (
        obj_type.__module__ == "sift.stability"
        and obj_type.__qualname__ == "StabilitySelector"
    ):
        from sift.stability import StabilitySelector

        if obj_type is StabilitySelector:
            return _as_stability_selector(obj, input_features)
    if (
        obj_type.__module__ == "sift.catboost_common"
        and obj_type.__qualname__ == "CatBoostSelectionResult"
    ):
        from sift.catboost_common import CatBoostSelectionResult

        if obj_type is CatBoostSelectionResult:
            return _as_catboost_result(obj, input_features)
    if isinstance(obj, (list, tuple)):
        raise TypeError(
            "as_result cannot infer a result protocol from a legacy list/tuple; rerun the "
            "selector with return_result=True"
        )
    if isinstance(obj, pd.DataFrame):
        raise TypeError(
            "as_result cannot recover repeat-level identity from a permutation-importance "
            "DataFrame; rerun permutation_importance with return_result=True"
        )
    raise TypeError(
        f"as_result does not yet support {type(obj).__module__}.{type(obj).__qualname__}"
    )


__all__ = ["SelectionView", "as_result"]
