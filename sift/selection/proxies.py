"""Internal contracts for bounded selection-time proxy correlations."""

from __future__ import annotations

from numbers import Real
from typing import Iterable

import numpy as np
import pandas as pd


MAX_PROXY_CORRELATION_BYTES = 64 * 1024**2
MAX_RESAMPLE_SELECTION_BYTES = 16 * 1024**2


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


def weighted_correlation_columns(
    Z: np.ndarray,
    w: np.ndarray,
    column_locals: Iterable[int],
    *,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Weighted correlations of every column of ``Z`` with a selected subset.

    Computes only the ``p × k`` block. Self-correlations of selected columns
    are forced to 1.0 after clipping off-diagonals to ``±0.999999``, matching
    ``weighted_correlation_matrix``.
    """
    from threadpoolctl import threadpool_limits

    Z64 = np.ascontiguousarray(Z, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64).ravel()
    cols = np.asarray(list(column_locals), dtype=np.int64)
    if Z64.ndim != 2:
        raise ValueError("Z must be 2-d")
    n, p = Z64.shape
    if w64.shape[0] != n:
        raise ValueError("w length must match Z rows")
    k = int(cols.size)
    if k == 0:
        return np.zeros((p, 0), dtype=np.float64)
    if np.any((cols < 0) | (cols >= p)):
        raise ValueError("column_locals must be valid Z column positions")
    w_sum = float(w64.sum())
    if w_sum <= 0.0:
        raise ValueError("Weights must sum to > 0")
    sqrt_w = np.sqrt(w64)
    gram = np.zeros((p, k), dtype=np.float64)
    batch_size = max(1, int(batch_size))
    with threadpool_limits(limits=1):
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            zw = Z64[start:stop] * sqrt_w[start:stop, None]
            gram += zw.T @ zw[:, cols]
    gram /= w_sum
    np.clip(gram, -0.999999, 0.999999, out=gram)
    for j, local in enumerate(cols.tolist()):
        gram[int(local), j] = 1.0
    return gram


def reject_unavailable_proxy_positions(
    selected_indices: Iterable[int],
    *,
    available_original: Iterable[int],
    feature_names: Iterable[object] | None = None,
) -> None:
    """Reject proxy retention when selected raw columns have no cache correlations.

    Atomic blocks may still expand to cache-dropped constant members. Those
    columns stay in the selection; they cannot appear in a finite copula
    proxy block.
    """
    selected = _positions(selected_indices, label="selected_indices")
    available = {int(i) for i in _positions(available_original, label="available_original")}
    missing = [position for position in selected if position not in available]
    if not missing:
        return
    names = None if feature_names is None else list(feature_names)
    refs = []
    for position in missing:
        if names is not None and 0 <= position < len(names):
            refs.append(names[position])
        else:
            refs.append(position)
    raise ValueError(
        "store_proxies=True cannot retain finite copula correlations for "
        f"cache-dropped constant or otherwise unavailable block members: {refs} "
        f"(positions {missing}). Atomic selection still expands those raw "
        "columns; omit store_proxies or drop unavailable members from the block"
    )


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


def validate_r_min(r_min: object) -> float:
    """Reject bools and non-finite values; require ``r_min`` in ``[0, 1]``."""
    if isinstance(r_min, (bool, np.bool_)) or not isinstance(
        r_min,
        (Real, np.integer, np.floating),
    ):
        raise ValueError("r_min must be a finite number between 0 and 1")
    threshold = float(r_min)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("r_min must be a finite number between 0 and 1")
    return threshold


def normalize_resample_selections(
    values: np.ndarray | None,
    *,
    n_features: int | None,
) -> tuple[np.ndarray | None, int]:
    """Validate and copy a completed-resample selection indicator matrix."""
    if values is None:
        return None, 0
    if n_features is None:
        raise ValueError("resample selections require a known raw feature width")
    n_features = int(n_features)
    if n_features < 0:
        raise ValueError("n_raw_features must be a non-negative integer or None")
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("resample selections must be a 2-d indicator matrix")
    n_resamples, n_cols = array.shape
    if n_cols != n_features:
        raise ValueError(
            "resample selections must have one column per raw feature"
        )
    if n_resamples < 1:
        raise ValueError("resample selections must contain at least one completed resample")
    if array.dtype == np.bool_ or array.dtype == bool:
        indicators = np.ascontiguousarray(array.astype(bool, copy=True))
    else:
        if array.dtype.kind not in {"i", "u", "b"}:
            raise ValueError("resample selections must be boolean or integer 0/1 indicators")
        unique = np.unique(array)
        if unique.size > 2 or np.any((unique != 0) & (unique != 1)):
            raise ValueError("resample selections must be boolean or integer 0/1 indicators")
        indicators = np.ascontiguousarray(array.astype(bool, copy=True))
    storage_bytes = int(indicators.nbytes)
    if storage_bytes > MAX_RESAMPLE_SELECTION_BYTES:
        limit_mib = MAX_RESAMPLE_SELECTION_BYTES / 1024**2
        requested_mib = storage_bytes / 1024**2
        raise ValueError(
            "store_proxies=True would retain "
            f"{requested_mib:.2f} MiB of resample selection indicators, "
            f"exceeding the {limit_mib:.0f} MiB limit"
        )
    return indicators, storage_bytes


def redundancy_report_frame(
    block: pd.DataFrame,
    *,
    selected_indices: list[int],
    raw_features: list[object] | None,
    r_min: float,
) -> pd.DataFrame:
    """Every qualifying unselected-candidate ↔ selected-feature edge."""
    columns = {
        "selected_feature": pd.Series(dtype=object),
        "selected_index": pd.Series(dtype="int64"),
        "feature": pd.Series(dtype=object),
        "candidate_index": pd.Series(dtype="int64"),
        "correlation": pd.Series(dtype="float64"),
    }
    if not selected_indices:
        return pd.DataFrame(columns)
    selected_set = set(int(i) for i in selected_indices)
    selected_labels = _labels_for(selected_indices, raw_features)
    records: list[tuple[int, float, int, int, object, object, float]] = []
    for path_rank, selected_pos in enumerate(selected_indices):
        if selected_pos not in block.columns:
            continue
        values = block[selected_pos]
        for candidate_pos, correlation in zip(
            np.asarray(values.index, dtype=np.int64),
            values.to_numpy(dtype=np.float64),
        ):
            candidate_pos = int(candidate_pos)
            if candidate_pos in selected_set:
                continue
            if abs(float(correlation)) < r_min:
                continue
            records.append(
                (
                    path_rank,
                    -abs(float(correlation)),
                    candidate_pos,
                    int(selected_pos),
                    selected_labels[path_rank],
                    _label_at(candidate_pos, raw_features),
                    float(correlation),
                )
            )
    records.sort()
    return pd.DataFrame(
        {
            "selected_feature": [item[4] for item in records],
            "selected_index": [item[3] for item in records],
            "feature": [item[5] for item in records],
            "candidate_index": [item[2] for item in records],
            "correlation": [item[6] for item in records],
        }
    )


def proxy_cluster_frame(
    block: pd.DataFrame,
    *,
    selected_indices: list[int],
    raw_features: list[object] | None,
    r_min: float,
    resample_selections: np.ndarray | None = None,
) -> pd.DataFrame:
    """Selected-anchored connected components of qualifying proxy edges."""
    columns = {
        "cluster_id": pd.Series(dtype="int64"),
        "feature": pd.Series(dtype=object),
        "selected_index": pd.Series(dtype="int64"),
        "selected": pd.Series(dtype=bool),
        "cluster_frequency": pd.Series(dtype="Float64"),
    }
    if not selected_indices:
        return pd.DataFrame(columns)
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    selected_set = set(int(i) for i in selected_indices)
    for selected_pos in selected_indices:
        parent.setdefault(int(selected_pos), int(selected_pos))
        if selected_pos not in block.columns:
            continue
        values = block[selected_pos]
        for candidate_pos, correlation in zip(
            np.asarray(values.index, dtype=np.int64),
            values.to_numpy(dtype=np.float64),
        ):
            candidate_pos = int(candidate_pos)
            if candidate_pos == int(selected_pos):
                continue
            if abs(float(correlation)) < r_min:
                continue
            union(int(selected_pos), candidate_pos)

    root_to_id: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    for selected_pos in selected_indices:
        root = find(int(selected_pos))
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
            members[root] = []
        members[root].append(int(selected_pos))
    for node in parent:
        if node in selected_set:
            continue
        root = find(node)
        if root in members:
            members[root].append(int(node))

    cluster_freq: dict[int, float] | None = None
    if resample_selections is not None:
        n_resamples = int(resample_selections.shape[0])
        cluster_freq = {}
        for root, cluster_id in root_to_id.items():
            positions = np.asarray(members[root], dtype=np.int64)
            hit = np.any(resample_selections[:, positions], axis=1)
            cluster_freq[cluster_id] = float(np.count_nonzero(hit) / n_resamples)

    rows: list[dict[str, object]] = []
    emitted: set[int] = set()
    for selected_pos in selected_indices:
        root = find(int(selected_pos))
        cluster_id = root_to_id[root]
        if cluster_id in emitted:
            continue
        emitted.add(cluster_id)
        cluster_members = members[root]
        selected_members = [pos for pos in selected_indices if pos in cluster_members]
        candidate_members = sorted(
            pos for pos in cluster_members if pos not in selected_set
        )
        freq: object = pd.NA if cluster_freq is None else cluster_freq[cluster_id]
        for pos in selected_members + candidate_members:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "feature": _label_at(pos, raw_features),
                    "selected_index": int(pos),
                    "selected": pos in selected_set,
                    "cluster_frequency": freq,
                }
            )
    frame = pd.DataFrame(rows, columns=list(columns))
    frame["cluster_frequency"] = frame["cluster_frequency"].astype("Float64")
    return frame


def _labels_for(positions: list[int], raw_features: list[object] | None) -> list[object]:
    return [_label_at(position, raw_features) for position in positions]


def _label_at(position: int, raw_features: list[object] | None) -> object:
    if raw_features is None:
        return int(position)
    return raw_features[int(position)]


__all__: list[str] = []
