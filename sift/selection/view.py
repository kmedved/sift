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

from sift.selection.proxies import (
    normalize_proxy_frame,
    normalize_resample_selections,
    proxy_cluster_frame,
    redundancy_report_frame,
    validate_r_min,
)


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
      ``orient="split"``; dataclasses use ``dataclasses.asdict``).
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

    One shape for every selector: whichever of SIFT's result families produced
    a selection, the view exposes the same five accessors -- ``features``,
    ``indices``, ``k``, ``table``, and ``metadata`` -- so
    downstream code does not branch on the selector.  Build one with
    ``as_result`` or a result object's ``result_view()`` rather than
    calling this constructor, which exists for the adapters.  Every accessor
    returns a defensive copy, so a view is safe to hand around and cannot be
    mutated through what it hands back.

    Parameters
    ----------
    features : iterable
        Selected feature labels in the selector's own order.
    indices : iterable of int or None
        Their positions in the raw feature matrix, aligned to ``features``.
        ``None`` when the source result cannot prove positions.
    raw_features : iterable or None
        Ordered labels of every raw input column, or ``None`` when unknown.
    n_raw_features : int or None
        Raw input width.  Inferred from ``raw_features`` when that is given;
        required whenever ``metadata["table_complete"]`` is true.
    raw_table : DataFrame
        Per-feature table.  Must carry ``feature``, ``selected_index``,
        ``path_rank`` and ``selected``; ``path_rank`` must be unique,
        one-based, and present on exactly the selected rows.
    curve : DataFrame or None, default None
        Selection curve with columns ``k``, ``criterion``, ``criterion_se``
        and ``selected``.  ``None`` stores an empty frame with those columns.
    metadata : mapping or None, default None
        Selector metadata to copy.  ``schema_version``, ``transform_available``,
        ``inverse_transform_available``, the column hashes, and the proxy
        counters are always (re)written by the constructor; ``input_kind``
        must be ``"dataframe"``, ``"positional"`` or ``"unknown"``, and
        ``table_complete`` must be boolean.
    diagnostics : any, default None
        Selector diagnostics, deep-copied as given.
    encoded_features : iterable or None, default None
        Labels of the post-encoding feature space, when the selector had one.
    encoded_indices : iterable of int or None, default None
        Selected positions in that encoded space, aligned to ``features``.
    encoded_table : DataFrame or None, default None
        Per-feature table in the encoded space.
    transformer : callable or None, default None
        ``X -> X_selected`` callable backing ``transform``.  ``None``
        leaves the method raising ``NotImplementedError``.
    inverse_transformer : callable or None, default None
        Callable backing ``inverse_transform``, under the same rule.
    proxy_correlations : DataFrame or None, default None
        Candidate-by-selected correlation block backing ``proxies``,
        ``proxies_at``, ``redundancy_report``, and ``proxy_clusters``.
        Normalized and size-checked on construction.
    resample_selections : ndarray or None, default None
        Completed-resample boolean matrix of shape
        ``(n_resamples, n_raw_features)`` used for cluster selection
        frequencies. Copied and size-checked; never retains ``X``.

    Attributes
    ----------
    features : list
        Selected feature labels, in selection order.
    indices : list of int or None
        Their raw-matrix positions, or ``None`` when unknown.
    support_ : ndarray of shape (n_raw_features,), bool, or None
        Boolean mask over the raw columns, or ``None`` when either the raw
        width or the indices are unknown.
    k : int
        Number of selected features.
    raw_features : list or None
        Ordered labels of every raw input column, or ``None``.
    raw_input : dict
        ``{"n_features", "features", "columns_hash"}`` describing the raw
        input identity.
    encoded_features : list or None
        Labels of the encoded feature space, or ``None``.
    encoded_indices : list of int or None
        Selected positions in the encoded space, or ``None``.
    encoded_support_ : ndarray of bool or None
        Boolean mask over the encoded columns, or ``None``.
    encoded_output : dict or None
        ``{"n_features", "features", "columns_hash"}`` for the encoded space,
        or ``None`` when there is none.
    raw_table : DataFrame
        The normalized per-feature table.
    table : DataFrame
        Alias of ``raw_table``.
    encoded_table : DataFrame or None
        Per-feature table in the encoded space, or ``None``.
    curve : DataFrame
        Selection curve with columns ``k``, ``criterion``, ``criterion_se``
        and ``selected``; empty when the route reported none.
    metadata : dict
        Copied selector metadata plus ``schema_version``, ``input_kind``,
        ``table_complete``, ``transform_available``,
        ``inverse_transform_available``, ``raw_columns_hash``,
        ``encoded_columns_hash``, the ``proxy_*`` counters, and resample
        cluster-frequency availability.
    diagnostics : any
        Copied selector diagnostics.

    See Also
    --------
    as_result : Build a view from a supported result object.
    sift.select_cached : Returns a view directly with ``return_result=True``.
    sift.KnockoffSelectionResult : One of the result families adapted here.

    Notes
    -----
    The view is *additive*: it never replaces or mutates the result it was
    built from, and it never retains the caller's feature matrix -- only
    labels, positions, tables, metadata and diagnostics.  Result-only sources
    cannot prove whether their input was named or positional, so those views
    report ``metadata["input_kind"] == "unknown"``; passing ``input_features``
    to ``as_result`` establishes an ordered raw identity and a
    ``raw_columns_hash`` without rewriting that provenance.  Likewise, only a
    view built from a fitted selector can transform data: result-only views
    raise ``NotImplementedError`` from ``transform`` and report
    ``metadata["transform_available"] is False``.  A partial table -- one that
    does not cover every raw position -- is marked by
    ``metadata["table_complete"] is False`` and limits ``plot``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import as_result, select_cefsplus
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    >>> y = X["a"] + 0.7 * X["d"] + 0.1 * rng.normal(size=200)
    >>> result = select_cefsplus(X, y, k=2, verbose=False, return_result=True)
    >>> view = as_result(result, input_features=list(X.columns))
    >>> view.k, view.features, view.indices
    (2, ['a', 'd'], [0, 3])
    >>> view.support_.tolist()
    [True, False, False, True, False]
    >>> list(view.table.columns)
    ['feature', 'selected_index', 'path_rank', 'selected', 'relevance']
    >>> view.metadata["table_complete"], view.metadata["transform_available"]
    (True, False)
    >>> view
    SelectionView(k=2, adapter='FilterSelectionResult')
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
        "_resample_selections",
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
        resample_selections: np.ndarray | None = None,
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
        if encoded_positions is not None and encoded_names is None:
            raise ValueError("encoded_indices requires encoded_features")
        if encoded_positions is not None:
            if any(index < 0 for index in encoded_positions):
                raise ValueError("encoded_indices contains an out-of-bounds position")
            if len(set(encoded_positions)) != len(encoded_positions):
                raise ValueError("encoded_indices must be unique")
        if encoded_names is not None and encoded_positions is not None and any(
            index >= len(encoded_names) for index in encoded_positions
        ):
            raise ValueError("encoded_indices contains an out-of-bounds position")
        if (
            encoded_positions is not None
            and selected_indices is not None
            and n_raw_features is not None
            and len(encoded_positions) < len(selected)
        ):
            raise ValueError(
                "encoded_indices must cover every selected raw feature after expansion"
            )

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
        if resample_selections is not None and proxy_correlations is None:
            raise ValueError(
                "resample_selections require proxy_correlations; cluster "
                "frequencies cannot be advertised without a stored proxy block"
            )
        self._proxy_correlations, proxy_storage_bytes = normalize_proxy_frame(
            proxy_correlations,
            selected_indices=selected_indices,
            n_raw_features=n_raw_features,
        )
        self._resample_selections, resample_storage_bytes = normalize_resample_selections(
            resample_selections,
            n_features=n_raw_features,
        )
        self._metadata["proxy_correlations_stored"] = self._proxy_correlations is not None
        self._metadata["proxy_candidate_count"] = (
            0 if self._proxy_correlations is None else len(self._proxy_correlations)
        )
        self._metadata["proxy_storage_bytes"] = proxy_storage_bytes
        self._metadata["cluster_frequencies_available"] = (
            self._resample_selections is not None
        )
        self._metadata["n_resamples_stored"] = (
            0 if self._resample_selections is None else int(self._resample_selections.shape[0])
        )
        self._metadata["resample_selection_storage_bytes"] = resample_storage_bytes

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
        """Reduce a feature matrix to the selected columns.

        Available only on views built from a fitted selector that retained its
        preprocessing state; result-only views have nothing to apply.  Check
        ``metadata["transform_available"]`` before calling.

        Parameters
        ----------
        X : DataFrame or ndarray
            Matrix in the same feature space the selector was fitted on.

        Returns
        -------
        DataFrame or ndarray
            ``X`` restricted to the selected columns, in the fitted
            selector's own output order.

        Raises
        ------
        NotImplementedError
            If this view carries no fitted transformer.

        See Also
        --------
        SelectionView.inverse_transform : The reverse mapping, when retained.
        SelectionView.features : The selected labels, always available.
        """
        if self._transformer is None:
            raise NotImplementedError(
                "transform is unavailable for this result-only view; adapt a fitted selector "
                "that retains preprocessing state"
            )
        return self._transformer(X)

    def inverse_transform(self, X_selected: Any) -> Any:
        """Map selected columns back to the original feature space.

        Available only on views that retained a fitted inverse encoder, which
        no current SIFT adapter provides; check
        ``metadata["inverse_transform_available"]`` first.

        Parameters
        ----------
        X_selected : DataFrame or ndarray
            Matrix over the selected columns, as returned by
            ``transform``.

        Returns
        -------
        DataFrame or ndarray
            The same rows expressed in the original feature space.

        Raises
        ------
        NotImplementedError
            If this view carries no fitted inverse transformer.

        See Also
        --------
        SelectionView.transform : The forward mapping.
        """
        if self._inverse_transformer is None:
            raise NotImplementedError(
                "inverse_transform is unavailable because this view does not retain a fitted "
                "inverse encoder"
            )
        return self._inverse_transformer(X_selected)

    def proxies(self, feature: Any, r_min: float = 0.8) -> pd.DataFrame:
        """Return unselected stand-ins for one selected feature, by label.

        Answers "what else could have been picked instead?" -- the unselected
        candidates whose copula correlation with ``feature`` is large enough
        that they carry much the same signal.  Requires the selection to have
        run with ``store_proxies=True``.

        Parameters
        ----------
        feature : hashable
            Label of the selected feature.  Resolved against
            ``raw_features`` when known, otherwise against
            ``features``; an ambiguous or missing label is an error.
        r_min : float, default 0.8
            Minimum absolute correlation to report, in ``[0, 1]``.

        Returns
        -------
        DataFrame
            Columns ``feature``, ``selected_index``, and ``correlation``,
            sorted by descending absolute correlation then raw position.
            Empty when nothing clears ``r_min``.

        Raises
        ------
        NotImplementedError
            If proxy correlations were not stored.
        ValueError
            If ``feature`` is missing or ambiguous -- use ``proxies_at``
            for positional access -- or if ``r_min`` is outside ``[0, 1]``.

        See Also
        --------
        SelectionView.proxies_at : The positional form of this lookup.
        SelectionView.redundancy_report : Every qualifying proxy edge.
        SelectionView.proxy_clusters : Selected-anchored proxy components.

        Examples
        --------
        >>> import numpy as np
        >>> from sift import build_cache, select_cached
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(200, 5))
        >>> y = X[:, 0] + 0.1 * rng.normal(size=200)
        >>> view = select_cached(build_cache(X, compute_Rxx=True), y, k=1,
        ...                      return_result=True, store_proxies=True)
        >>> list(view.proxies("x0", r_min=0.9).columns)
        ['feature', 'selected_index', 'correlation']
        """
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
        """Return unselected proxy candidates for one selected raw position.

        Positional form of ``proxies``, and the one to use when labels are
        duplicated or absent.  Requires the selection to have run with
        ``store_proxies=True``.

        Parameters
        ----------
        selected_index : int
            Raw-matrix position of a *selected* feature; a position that was
            not selected has no stored proxy column.
        r_min : float, default 0.8
            Minimum absolute correlation to report, in ``[0, 1]``.

        Returns
        -------
        DataFrame
            Columns ``feature``, ``selected_index``, and ``correlation``,
            sorted by descending absolute correlation then raw position.
            Selected features are excluded, so only genuine stand-ins appear.

        Raises
        ------
        NotImplementedError
            If proxy correlations were not stored.
        ValueError
            If ``selected_index`` is not an integer or is not a selected proxy
            position, or if ``r_min`` is not a finite number in ``[0, 1]``.

        See Also
        --------
        SelectionView.proxies : The label-based form of this lookup.
        """
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
        threshold = validate_r_min(r_min)

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

    def redundancy_report(self, r_min: float = 0.8) -> pd.DataFrame:
        """Return every qualifying unselected-candidate ↔ selected-feature edge.

        This is the all-selected companion to ``proxies`` / ``proxies_at``:
        one row per stored copula correlation whose absolute value is at
        least ``r_min``.  Selected features never appear as proxy
        candidates.  Requires ``store_proxies=True``.

        Parameters
        ----------
        r_min : float, default 0.8
            Minimum absolute correlation to report, in ``[0, 1]``.

        Returns
        -------
        DataFrame
            Columns ``selected_feature``, ``selected_index`` (the selected
            raw position), ``feature``, ``candidate_index`` (the unselected
            raw position), and signed ``correlation``.  Sorted by selected
            path order, then descending absolute correlation, then candidate
            raw position.  Empty when nothing qualifies.

        Raises
        ------
        NotImplementedError
            If proxy correlations were not stored.
        ValueError
            If ``r_min`` is not a finite number in ``[0, 1]``.
        """
        block = self._require_proxy_block()
        threshold = validate_r_min(r_min)
        selected = [] if self._indices is None else list(self._indices)
        raw_names = None if self._raw_features is None else list(self._raw_features)
        return redundancy_report_frame(
            block,
            selected_indices=selected,
            raw_features=raw_names,
            r_min=threshold,
        )

    def proxy_clusters(self, r_min: float = 0.8) -> pd.DataFrame:
        """Return selected-anchored connected components of proxy edges.

        Nodes are selected features plus unselected candidates with
        ``|correlation| >= r_min`` to at least one selected feature.
        Edges come from that stored candidate×selected block, including
        qualifying selected↔selected correlations: this is not an all-pairs
        clustering of unselected columns.  A candidate linked to two selected
        anchors joins those anchors, as does a direct selected-selected edge.
        Each selected feature is at least a singleton cluster.

        When the view carries completed-resample selection indicators,
        ``cluster_frequency`` is the fraction of those resamples in which
        any cluster member was selected.  Otherwise the column is nullable
        ``Float64`` and entirely missing.

        Parameters
        ----------
        r_min : float, default 0.8
            Absolute-correlation threshold for an edge, in ``[0, 1]``.
            Signed correlations are preserved in ``redundancy_report``;
            clustering uses the absolute value.

        Returns
        -------
        DataFrame
            One row per member, columns ``cluster_id`` (dense, 0-based,
            ordered by first selected path member), ``feature``,
            ``selected_index``, ``selected``, and ``cluster_frequency``.

        Raises
        ------
        NotImplementedError
            If proxy correlations were not stored.
        ValueError
            If ``r_min`` is not a finite number in ``[0, 1]``.
        """
        block = self._require_proxy_block()
        threshold = validate_r_min(r_min)
        selected = [] if self._indices is None else list(self._indices)
        raw_names = None if self._raw_features is None else list(self._raw_features)
        return proxy_cluster_frame(
            block,
            selected_indices=selected,
            raw_features=raw_names,
            r_min=threshold,
            resample_selections=self._resample_selections,
        )

    def _require_proxy_block(self) -> pd.DataFrame:
        if self._proxy_correlations is None:
            raise NotImplementedError(
                "proxy correlations were not stored; rerun selection with store_proxies=True"
            )
        return self._proxy_correlations

    def plot(self, ax=None):
        """Plot the selection curve, or the per-feature metric as a fallback.

        Draws ``curve`` (``criterion`` against ``k``, with the chosen
        points marked) when the route reported one.  Otherwise it falls back
        to a bar chart of ``gain`` or ``relevance`` from the table, which
        requires that table to cover every raw position.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, default None
            Axes to draw on.  ``None`` creates a new figure and axes, which is
            the only path that imports matplotlib.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that were drawn on.

        Raises
        ------
        NotImplementedError
            If there is no curve and the table is partial, or has no ``gain``
            or ``relevance`` column to plot.
        ImportError
            If ``ax`` is ``None`` and matplotlib is not installed.

        See Also
        --------
        SelectionView.curve : The underlying curve data.
        SelectionView.table : The per-feature table used as a fallback.
        """
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
        """Serialize the view to a versioned, JSON-safe payload.

        Everything the view holds except the proxy-correlation block and any
        resample-selection indicators, which are deliberately omitted because
        they are bounded working data rather than part of the result.

        Returns
        -------
        dict
            Keys ``schema_version``, ``features``, ``indices``, ``support``,
            ``raw_input``, ``raw_table``, ``encoded_features``,
            ``encoded_indices``, ``encoded_support``, ``encoded_output``,
            ``encoded_table``, ``curve``, ``metadata``, and ``diagnostics``.
            Tables are emitted in pandas ``orient="split"`` form, and mappings
            with non-string keys use a tagged envelope rather than losing
            their key types.

        Raises
        ------
        TypeError
            If a diagnostics or metadata value has no defined JSON
            representation.  Nothing is coerced to its ``repr`` silently.

        See Also
        --------
        SelectionView.metadata : The metadata copied into the payload.
        SelectionView.reproducibility_ : Provenance manifest, not this snapshot.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from sift import as_result, select_cefsplus
        >>> rng = np.random.default_rng(0)
        >>> X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
        >>> y = X["a"] + 0.7 * X["d"] + 0.1 * rng.normal(size=200)
        >>> view = as_result(select_cefsplus(X, y, k=2, verbose=False,
        ...                                  return_result=True))
        >>> payload = view.to_dict()
        >>> payload["schema_version"], payload["features"]
        ('1', ['a', 'd'])
        """
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

    def reproducibility_(self, *, X=None, hash_data: bool = False) -> dict[str, Any]:
        """Return a JSON-safe reproducibility manifest for this view.

        Package versions, BLAS identity, and git commit are captured at
        export time and labelled as such. They are not selection-time
        context. Shape, typed column hash, cache provenance, effective
        configuration, and seeds come only from facts this view already
        retained. The caller's feature matrix is never stored. Data hashing
        is off unless ``hash_data=True`` and ``X`` is supplied.

        Parameters
        ----------
        X : DataFrame or ndarray, optional
            Caller-supplied matrix used only when hashing data or filling
            unknown shape at export. Not retained.
        hash_data : bool, default False
            If True, hash ``X``. Raises if ``X`` is omitted.

        Returns
        -------
        dict
            Schema ``"1"`` payload with ``environment``, ``input``,
            ``configuration``, and ``folds``. Safe for ``json.dumps``.

        Raises
        ------
        ValueError
            If ``hash_data=True`` without ``X``, or if ``X`` disagrees with
            a known feature width.

        See Also
        --------
        SelectionView.to_dict : Full result snapshot, not a manifest.
        """
        from sift.selection.reproducibility import manifest_from_view

        return manifest_from_view(self, X=X, hash_data=hash_data)

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


def _append_rows_like(table: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Append ``rows`` to ``table`` with every column cast to the table's dtype.

    Columns the rows do not mention become missing values of the matching
    dtype.  Aligning dtypes first keeps the concatenation free of pandas'
    all-NA-entry deprecation, which fires whenever an all-missing column would
    otherwise be ignored while inferring the result dtype.

    Widening a column to hold the new missing values is part of that alignment:
    a numpy ``int64`` or ``bool`` column that the appended rows do not mention
    (or that they fill with a missing value) is promoted to its nullable
    counterpart, ``Int64`` or ``boolean``.  That promotion happens **only when
    rows are actually appended** -- with an empty ``rows`` the helper is an
    identity and returns ``table`` itself, dtypes and values untouched.
    """
    if not rows:
        return table
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
            elif pd.api.types.is_float_dtype(dtype):
                # Floats hold NaN natively, so a mentioned missing value keeps
                # the table's float dtype instead of widening to ``Float64``.
                aligned[column] = values.to_numpy(dtype=dtype, na_value=np.nan)
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


def as_result(obj: Any, input_features: Any = None) -> SelectionView:
    """Return an additive ``SelectionView`` for a supported SIFT result.

    Passing an existing view is an identity operation.  Legacy list and tuple
    returns are intentionally not guessed; request the corresponding result
    object with ``return_result=True`` first.

    This is the single entry point that turns any of SIFT's result families
    into one normalized shape, so downstream code can read ``features``,
    ``indices``, ``k``, ``table`` and ``metadata`` without knowing which
    selector ran.  The original object is neither modified nor replaced; the
    view is a separate copy that never retains the caller's feature matrix.

    Parameters
    ----------
    obj : result object or SelectionView
        A ``sift.selection.result.FilterSelectionResult``,
        ``sift.KnockoffSelectionResult``, ``BorutaResult``,
        ``FeaturePathEvaluationResult``, ``ImportanceResult``,
        ``CatBoostSelectionResult``, a fitted ``StabilitySelector``, a
        fitted ``Stabilized`` selector, or a fitted ``ModelSelector``.  An
        existing ``SelectionView`` is returned unchanged.  Subclasses of
        the result types are not accepted: dispatch is by exact type so an
        adapter cannot silently mis-read an extended object.
    input_features : sequence or None, default None
        Ordered labels of every raw input column.  Supplying them establishes
        the view's raw feature identity and ``raw_columns_hash`` for a result
        that cannot prove its own, which is what lets ``support_`` and a
        complete table exist.  Must not be given when ``obj`` is already a
        ``SelectionView``.

    Returns
    -------
    SelectionView
        Normalized view over ``obj``.  For an existing view, ``obj`` itself.

    Raises
    ------
    TypeError
        If ``obj`` is a bare list or tuple (rerun the selector with
        ``return_result=True``), a permutation-importance DataFrame (rerun
        ``sift.permutation_importance`` with ``return_result=True``), or
        any other unsupported type.
    ValueError
        If ``input_features`` is supplied together with an existing
        ``SelectionView``, or if it contradicts the identity, width, or
        positions the result already carries.

    See Also
    --------
    SelectionView : The returned view type and its accessors.
    sift.KnockoffSelectionResult.result_view : Method form for that result.
    sift.select_cached : Returns a view directly with ``return_result=True``.

    Notes
    -----
    Adaptation is additive and lossless in one direction only: the view reads
    what the result already proved and refuses to invent the rest.  A legacy
    result cannot say whether its input was named or positional, so those
    views report ``metadata["input_kind"] == "unknown"`` even when
    ``input_features`` is supplied.  Result-only sources also carry no fitted
    preprocessing state, so ``SelectionView.transform`` raises and
    ``metadata["transform_available"]`` is ``False``; only a fitted selector
    yields a transforming view.  Metrics a selector never reported stay
    ``NaN`` rather than being guessed, and a table that cannot cover every raw
    position is marked ``table_complete=False``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import as_result, select_cefsplus
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    >>> y = X["a"] + 0.7 * X["d"] + 0.1 * rng.normal(size=200)
    >>> result = select_cefsplus(X, y, k=2, verbose=False, return_result=True)
    >>> view = as_result(result, input_features=list(X.columns))
    >>> view.k, view.features, view.metadata["input_kind"]
    (2, ['a', 'd'], 'unknown')
    >>> as_result(view) is view  # identity for an existing view
    True
    >>> try:  # a legacy list is never guessed at
    ...     as_result(["a", "d"])
    ... except TypeError as exc:
    ...     print(type(exc).__name__)
    TypeError
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
        from sift.selection.view_filter import _as_filter_result

        return _as_filter_result(obj, input_features)
    if type(obj) is KnockoffSelectionResult:
        from sift.selection.view_knockoff import _as_knockoff_result

        return _as_knockoff_result(obj, input_features)
    if type(obj) is BorutaResult:
        from sift.selection.view_boruta import _as_boruta_result

        return _as_boruta_result(obj, input_features)
    if type(obj) is FeaturePathEvaluationResult:
        from sift.selection.view_path import _as_feature_path_result

        return _as_feature_path_result(obj, input_features)
    if type(obj) is ImportanceResult:
        from sift.selection.view_importance import _as_importance_result

        return _as_importance_result(obj, input_features)
    obj_type = type(obj)
    if (
        obj_type.__module__ == "sift.stability"
        and obj_type.__qualname__ == "StabilitySelector"
    ):
        from sift.stability import StabilitySelector

        if obj_type is StabilitySelector:
            from sift.selection.view_stability import _as_stability_selector

            return _as_stability_selector(obj, input_features)
    if (
        obj_type.__module__ == "sift.selection.stabilized"
        and obj_type.__qualname__ == "Stabilized"
    ):
        from sift.selection.stabilized import Stabilized

        if obj_type is Stabilized:
            from sift.selection.view_stabilized import _as_stabilized_selector

            return _as_stabilized_selector(obj, input_features)
    if (
        obj_type.__module__ == "sift.selection.model_selector"
        and obj_type.__qualname__ == "ModelSelector"
    ):
        from sift.selection.model_selector import ModelSelector

        if obj_type is ModelSelector:
            from sift.selection.view_model import _as_model_selector

            return _as_model_selector(obj, input_features)
    if (
        obj_type.__module__ == "sift.catboost_common"
        and obj_type.__qualname__ == "CatBoostSelectionResult"
    ):
        from sift.catboost_common import CatBoostSelectionResult

        if obj_type is CatBoostSelectionResult:
            from sift.selection.view_catboost import _as_catboost_result

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
