"""Knockoff result adapter for additive SelectionView construction."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sift.selection.view import (
    SelectionView,
    _append_rows_like,
    _coerce_boolean_series,
    _coerce_feature_names,
    _coerce_indices,
    _coerce_position_series,
    _labels_equal,
    _selection_path_ranks,
    _strict_integer,
    _validate_selected_identity,
)


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
    for column in ("relevance", "selection_frequency", "feature_group", "evalue"):
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
