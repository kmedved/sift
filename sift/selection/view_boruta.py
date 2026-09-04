"""Boruta result adapter for additive SelectionView construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sift.selection.view import (
    SelectionView,
    _coerce_feature_names,
    _labels_equal,
    _numeric_vector,
    _strict_integer,
    _strict_integer_vector,
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
