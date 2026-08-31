"""Additive, normalized views over SIFT's legacy result objects."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Set
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1"
CURVE_COLUMNS = ("k", "criterion", "criterion_se", "selected")
_INPUT_KINDS = {"dataframe", "positional", "unknown"}


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
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
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


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
        self._proxy_correlations = (
            None if proxy_correlations is None else proxy_correlations.copy(deep=True)
        )

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
        if not 0.0 <= float(r_min) <= 1.0:
            raise ValueError("r_min must be between 0 and 1")
        columns = list(self._proxy_correlations.columns)
        matches = [idx for idx, value in enumerate(columns) if _labels_equal(value, feature)]
        if len(matches) != 1:
            raise ValueError(
                f"feature {feature!r} is missing or ambiguous; use positional proxy access"
            )
        values = self._proxy_correlations.iloc[:, matches[0]]
        mask = np.abs(values.to_numpy(dtype=float)) >= float(r_min)
        return pd.DataFrame(
            {
                "feature": self._proxy_correlations.index[mask],
                "correlation": values.to_numpy(dtype=float)[mask],
            }
        ).reset_index(drop=True)

    def plot(self, ax=None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("plot() requires matplotlib") from exc
        if ax is None:
            _, ax = plt.subplots()
        if not self._curve.empty:
            ax.plot(self._curve["k"], self._curve["criterion"], marker="o")
            selected = self._curve.loc[self._curve["selected"]]
            if not selected.empty:
                ax.scatter(selected["k"], selected["criterion"], zorder=3)
            ax.set_xlabel("k")
            ax.set_ylabel("criterion")
            return ax
        metric = next(
            (name for name in ("gain", "relevance") if name in self._raw_table),
            None,
        )
        if metric is None:
            raise NotImplementedError("plot data is unavailable for this partial view")
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


def _as_filter_result(result: Any, input_features: Any) -> SelectionView:
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
    metadata.update(
        {
            "adapter": "FilterSelectionResult",
            "curve_available": False,
            "table_complete": complete,
            "input_kind": "unknown",
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

    positions = _coerce_position_series(
        result.W["selected_index"],
        label="knockoff W selected_index",
        allow_missing=False,
    ).astype(int)
    if positions.duplicated().any():
        raise ValueError("knockoff W selected_index values must be unique and non-negative")
    if raw_features is not None:
        if (positions >= len(raw_features)).any():
            raise ValueError("knockoff W contains positions outside input_features")
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
    table = table.sort_values("selected_index", kind="mergesort").reset_index(drop=True)

    complete = bool(
        raw_features is not None
        and len(table) == len(raw_features)
        and set(int(value) for value in table["selected_index"]) == set(range(len(raw_features)))
    )
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
        n_raw_features=len(raw_features) if raw_features is not None else None,
        raw_table=table,
        metadata=metadata,
        diagnostics=result.diagnostics_,
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

    from sift.selection.knockoff_filter import KnockoffSelectionResult
    from sift.selection.result import FilterSelectionResult

    if type(obj) is FilterSelectionResult:
        return _as_filter_result(obj, input_features)
    if type(obj) is KnockoffSelectionResult:
        return _as_knockoff_result(obj, input_features)
    if isinstance(obj, (list, tuple)):
        raise TypeError(
            "as_result cannot infer a result protocol from a legacy list/tuple; rerun the "
            "selector with return_result=True"
        )
    raise TypeError(
        f"as_result does not yet support {type(obj).__module__}.{type(obj).__qualname__}"
    )


__all__ = ["SelectionView", "as_result"]
