"""Shared helpers for SIFT benchmark scripts."""

from __future__ import annotations

import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = "sift-benchmark-promotion-v1"


def regression_frame(
    n: int,
    p: int,
    *,
    seed: int,
    signal_count: int = 8,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create deterministic regression data with early-column signal."""
    rng = np.random.default_rng(seed)
    X_arr = rng.normal(size=(n, p)).astype(np.float32)
    n_signal = min(signal_count, p)
    coefs = np.linspace(2.5, 0.4, n_signal, dtype=np.float32)
    y = X_arr[:, :n_signal] @ coefs + rng.normal(scale=0.25, size=n).astype(np.float32)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(p)])
    return X, y


def measure(
    fn: Callable,
    *,
    repeat: int = 3,
    measure_memory: bool = False,
):
    """Measure median/best wall time and optional tracemalloc peak memory."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    walls: list[float] = []
    peaks: list[float | None] = []
    result = None
    for _ in range(repeat):
        if measure_memory:
            tracemalloc.start()
        peak_mb = None
        start = time.perf_counter()
        try:
            result = fn()
        finally:
            wall = time.perf_counter() - start
            if measure_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = peak / (1024 * 1024)
        walls.append(wall)
        peaks.append(peak_mb)

    finite_peaks = [peak for peak in peaks if peak is not None]
    return {
        "median_seconds": float(statistics.median(walls)),
        "best_seconds": float(min(walls)),
        "peak_memory_mb": float(max(finite_peaks)) if finite_peaks else None,
        "result": result,
    }


def promotion_status(
    *,
    parity: bool,
    baseline_seconds: float,
    current_seconds: float,
    baseline_peak_mb: float | None = None,
    current_peak_mb: float | None = None,
    wall_tolerance: float = 0.05,
    memory_tolerance: float = 0.05,
) -> str:
    """Return promoted/blocked status under the benchmark gate."""
    if not parity:
        return "blocked: parity"
    if current_seconds > baseline_seconds * (1.0 + wall_tolerance):
        return "blocked: wall"
    if (
        baseline_peak_mb is not None
        and current_peak_mb is not None
        and current_peak_mb > baseline_peak_mb * (1.0 + memory_tolerance)
    ):
        return "blocked: memory"
    return "promoted"


def write_json(path: Path, records: Iterable[dict]) -> None:
    """Write records with the shared promotion schema envelope."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "records": list(records),
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def _jsonable(value):
    """Convert NumPy/pandas scalars and containers to JSON-native values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def markdown_table(records: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a simple markdown table from ``(header, key)`` columns."""
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, sep]
    for row in records:
        values = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
