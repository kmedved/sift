"""Filter result adapter for additive SelectionView construction."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from sift.selection.view import (
    CURVE_COLUMNS,
    SelectionView,
    _coerce_boolean_series,
    _coerce_feature_names,
    _coerce_indices,
    _coerce_position_series,
    _labels_equal,
    _selection_path_ranks,
    _validate_selected_identity,
)


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
    for column in ("within_relevance", "between_relevance"):
        if column in ranking:
            values = pd.to_numeric(ranking[column], errors="coerce")
            if values.notna().any():
                table[column] = values
    if "block_id" in ranking:
        table["block_id"] = ranking["block_id"].tolist()

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
    encoded_names = metadata.get("encoded_feature_names")
    encoded_positions = metadata.get("encoded_selected_indices")
    encoded_table = None
    if isinstance(diagnostics, Mapping):
        ranking = diagnostics.get("encoded_ranking")
        if isinstance(ranking, pd.DataFrame):
            encoded_table = ranking.copy()
    return SelectionView(
        features=selected,
        indices=selected_indices,
        raw_features=raw_features,
        n_raw_features=raw_width,
        raw_table=table,
        curve=curve,
        metadata=metadata,
        diagnostics=diagnostics,
        encoded_features=encoded_names,
        encoded_indices=encoded_positions,
        encoded_table=encoded_table,
        proxy_correlations=getattr(result, _PROXY_CORRELATIONS_ATTR, None),
    )
