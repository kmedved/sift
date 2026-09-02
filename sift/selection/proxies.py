"""Internal contracts for bounded selection-time proxy correlations."""

from __future__ import annotations

from numbers import Real
from typing import Iterable

import numpy as np
import pandas as pd


MAX_PROXY_CORRELATION_BYTES = 64 * 1024**2


def _positions(values: Iterable[object], *, label: str) -> list[int]:
    positions: list[int] = []
    for value in list(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(f"{label} must contain integer raw-feature positions")
        position = int(value)
        if position < 0:
            raise ValueError(f"{label} must contain non-negative positions")
        positions.append(position)
    if len(set(positions)) != len(positions):
        raise ValueError(f"{label} must contain unique positions")
    return positions


def _check_storage_size(n_candidates: int, n_selected: int) -> int:
    storage_bytes = int(n_candidates) * int(n_selected) * np.dtype(np.float32).itemsize
    if storage_bytes > MAX_PROXY_CORRELATION_BYTES:
        limit_mib = MAX_PROXY_CORRELATION_BYTES / 1024**2
        requested_mib = storage_bytes / 1024**2
        raise ValueError(
            "store_proxies=True would retain "
            f"{requested_mib:.2f} MiB of correlations, exceeding the "
            f"{limit_mib:.0f} MiB limit"
        )
    return storage_bytes


def _numeric_correlations(values: np.ndarray, *, label: str) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind in {"i", "u", "f"}:
        array = source.astype(np.float64, copy=False)
    else:
        flat = np.asarray(values, dtype=object).ravel()
        converted: list[float] = []
        for value in flat.tolist():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                (Real, np.integer, np.floating),
            ):
                raise ValueError(f"{label} must contain real non-boolean values")
            converted.append(float(value))
        array = np.asarray(converted, dtype=np.float64).reshape(source.shape)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    if np.any(np.abs(array) > 1.0 + 1e-6):
        raise ValueError(f"{label} values must be correlations in [-1, 1]")
    return np.clip(array, -1.0, 1.0)


def proxy_frame_from_panel(
    correlations: np.ndarray,
    *,
    candidate_indices: Iterable[int],
    selected_indices: Iterable[int],
) -> pd.DataFrame:
    """Extract a float32 candidate-by-selected block from a candidate panel."""
    candidates = _positions(candidate_indices, label="candidate_indices")
    selected = _positions(selected_indices, label="selected_indices")
    matrix = np.asarray(correlations)
    if matrix.ndim != 2 or matrix.shape != (len(candidates), len(candidates)):
        raise ValueError(
            "candidate correlation matrix must be square and align with candidate_indices"
        )
    local_by_raw = {raw: local for local, raw in enumerate(candidates)}
    missing = [position for position in selected if position not in local_by_raw]
    if missing:
        raise ValueError(f"selected proxy positions are absent from the candidate panel: {missing}")
    _check_storage_size(len(candidates), len(selected))
    numeric = _numeric_correlations(matrix, label="candidate correlation matrix")
    block = np.empty((len(candidates), len(selected)), dtype=np.float32)
    for column, raw_position in enumerate(selected):
        block[:, column] = numeric[:, local_by_raw[raw_position]]
    return pd.DataFrame(
        block,
        index=pd.Index(candidates, name="selected_index"),
        columns=pd.Index(selected, name="selected_index"),
    )


def normalize_proxy_frame(
    frame: pd.DataFrame | None,
    *,
    selected_indices: Iterable[int] | None,
    n_raw_features: int | None,
) -> tuple[pd.DataFrame | None, int]:
    """Validate and defensively copy a positional proxy block."""
    if frame is None:
        return None, 0
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("proxy_correlations must be a pandas DataFrame")
    if selected_indices is None or n_raw_features is None:
        raise ValueError(
            "proxy correlations require known selected indices and raw feature width"
        )
    selected = _positions(selected_indices, label="selected_indices")
    candidates = _positions(frame.index.tolist(), label="proxy candidate index")
    columns = _positions(frame.columns.tolist(), label="proxy selected columns")
    if columns != selected:
        raise ValueError(
            "proxy selected columns must equal selected indices in selection order"
        )
    if any(position >= int(n_raw_features) for position in candidates + columns):
        raise ValueError("proxy correlations contain a position outside the raw feature width")
    missing = [position for position in selected if position not in set(candidates)]
    if missing:
        raise ValueError(f"selected proxy positions are absent from candidate rows: {missing}")
    storage_bytes = _check_storage_size(len(candidates), len(selected))
    values = _numeric_correlations(
        frame.to_numpy(copy=False),
        label="proxy correlations",
    )
    normalized = pd.DataFrame(
        values.astype(np.float32, copy=False),
        index=pd.Index(candidates, name="selected_index"),
        columns=pd.Index(selected, name="selected_index"),
    )
    return normalized, storage_bytes


__all__: list[str] = []
