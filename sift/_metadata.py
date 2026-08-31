"""Shared row-metadata conventions for public SIFT entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResolvedRowMetadata:
    """Feature input and positional row metadata after column-name sugar."""

    X: Any
    groups: Any
    time: Any
    sample_weight: Any
    extracted_columns: tuple[Any, ...]


def _labels_equal(left: Any, right: Any) -> bool:
    values = np.empty(2, dtype=object)
    values[:] = [left, right]
    try:
        return bool(
            pd.Index(values, dtype=object, tupleize_cols=False).duplicated()[1]
        )
    except (TypeError, ValueError):
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False


def _column_position(X: pd.DataFrame, column: Any, *, argument: str) -> int:
    matches = [
        position
        for position, candidate in enumerate(X.columns)
        if _labels_equal(candidate, column)
    ]
    if not matches:
        raise ValueError(f"{argument}={column!r} was not found in X")
    if len(matches) > 1:
        raise ValueError(
            f"{argument}={column!r} is ambiguous because X contains duplicate "
            "column labels"
        )
    return matches[0]


def resolve_row_metadata(
    X: Any,
    *,
    groups: Any = None,
    time: Any = None,
    sample_weight: Any = None,
    group_col: Any = None,
    time_col: Any = None,
    sample_weight_col: Any = None,
) -> ResolvedRowMetadata:
    """Resolve DataFrame column-name metadata without aligning row arrays.

    Scalar strings supplied through ``groups`` or ``time`` are the additive
    0.9 shorthand for a DataFrame column. Legacy ``*_col`` arguments use the
    same resolver. Direct arrays remain positional and are deliberately left
    untouched for their downstream validator.
    """

    conflicts = (
        ("groups", groups, "group_col", group_col),
        ("time", time, "time_col", time_col),
        ("sample_weight", sample_weight, "sample_weight_col", sample_weight_col),
    )
    for direct_name, direct_value, alias_name, alias_value in conflicts:
        if direct_value is not None and alias_value is not None:
            raise ValueError(
                f"Cannot specify both {direct_name} and {alias_name}"
            )

    requests: list[tuple[str, Any, str]] = []
    if isinstance(groups, str):
        requests.append(("groups", groups, "groups"))
    elif group_col is not None:
        requests.append(("groups", group_col, "group_col"))
    if isinstance(time, str):
        requests.append(("time", time, "time"))
    elif time_col is not None:
        requests.append(("time", time_col, "time_col"))
    if sample_weight_col is not None:
        requests.append(("sample_weight", sample_weight_col, "sample_weight_col"))

    if requests and not isinstance(X, pd.DataFrame):
        names = ", ".join(argument for _, _, argument in requests)
        raise ValueError(
            f"{names} requires X to be a pandas DataFrame when used as "
            "column-name row metadata"
        )

    if not requests:
        return ResolvedRowMetadata(
            X=X,
            groups=groups,
            time=time,
            sample_weight=sample_weight,
            extracted_columns=(),
        )

    assert isinstance(X, pd.DataFrame)
    resolved = {
        "groups": groups,
        "time": time,
        "sample_weight": sample_weight,
    }
    positions_to_drop: set[int] = set()
    extracted_columns: list[Any] = []
    for target, column, argument in requests:
        position = _column_position(X, column, argument=argument)
        resolved[target] = X.iloc[:, position].to_numpy(copy=True)
        positions_to_drop.add(position)
        if not any(_labels_equal(column, existing) for existing in extracted_columns):
            extracted_columns.append(column)

    keep = [
        position
        for position in range(X.shape[1])
        if position not in positions_to_drop
    ]
    return ResolvedRowMetadata(
        X=X.iloc[:, keep].copy(),
        groups=resolved["groups"],
        time=resolved["time"],
        sample_weight=resolved["sample_weight"],
        extracted_columns=tuple(extracted_columns),
    )


def drop_fitted_metadata_columns(
    X: Any,
    columns: tuple[Any, ...] | list[Any],
) -> Any:
    """Drop metadata columns recorded during fit when they are still present."""

    if not columns or not isinstance(X, pd.DataFrame):
        return X

    present: list[int] = []
    missing: list[Any] = []
    for column in columns:
        matches = [
            position
            for position, candidate in enumerate(X.columns)
            if _labels_equal(candidate, column)
        ]
        if len(matches) > 1:
            raise ValueError(
                f"metadata column {column!r} is ambiguous because X contains "
                "duplicate column labels"
            )
        if matches:
            present.append(matches[0])
        else:
            missing.append(column)

    if present and missing:
        raise ValueError(
            "transform input contains only some fitted row-metadata columns; "
            f"missing: {missing[:5]}"
        )
    if not present:
        return X
    present_set = set(present)
    keep = [position for position in range(X.shape[1]) if position not in present_set]
    return X.iloc[:, keep]


__all__ = [
    "ResolvedRowMetadata",
    "drop_fitted_metadata_columns",
    "resolve_row_metadata",
]
