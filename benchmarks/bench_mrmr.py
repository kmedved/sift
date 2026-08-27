#!/usr/bin/env python
"""Benchmark mRMR backends and verify selected-feature parity."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sift import select_mrmr  # noqa: E402
from benchmarks.bench_utils import promotion_status, write_json  # noqa: E402


Case = tuple[str, int, int, int]


def _parse_n_jobs(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--n-jobs must contain at least one integer")
    if 0 in values:
        raise argparse.ArgumentTypeError("n_jobs=0 is invalid")
    return values


def _parse_blas_threads(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--blas-threads must contain at least one integer")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("--blas-threads values must be >= 1")
    return values


def _cases(full: bool) -> list[Case]:
    quick_cases: list[Case] = [("classic", 5_000, 200, 20), ("gaussian", 5_000, 200, 20)]
    if not full:
        return quick_cases
    return quick_cases + [
        ("classic", 20_000, 2_000, 50),
        ("classic", 50_000, 2_000, 100),
        ("gaussian", 20_000, 2_000, 50),
    ]


def _make_data(n: int, p: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_arr = rng.normal(size=(n, p)).astype(np.float32)
    signal_count = min(12, p)
    coefs = np.linspace(2.5, 0.5, signal_count, dtype=np.float32)
    y = X_arr[:, :signal_count] @ coefs + rng.normal(scale=0.2, size=n).astype(np.float32)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(p)])
    return X, y


def _time_select(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    method: str,
    backend: str,
    n_jobs: int,
    k: int,
    blas_threads: int | None = None,
    repeat: int = 3,
) -> tuple[float, list[str]]:
    thread_context = (
        threadpool_limits(limits=blas_threads)
        if backend == "blas" and blas_threads is not None
        else nullcontext()
    )
    best_wall = float("inf")
    best_selected: list[str] = []
    with thread_context:
        for _ in range(repeat):
            start = time.perf_counter()
            selected = select_mrmr(
                X,
                y,
                k=k,
                task="regression",
                estimator="classic" if method == "classic" else "gaussian",
                formula="quotient",
                top_m=max(5 * k, 250),
                subsample=None,
                n_jobs=n_jobs,
                mrmr_backend=backend,
                verbose=False,
            )
            wall = time.perf_counter() - start
            if wall < best_wall:
                best_wall = wall
                best_selected = selected
    return best_wall, best_selected


def _warmup() -> None:
    X, y = _make_data(128, 12, seed=999)
    select_mrmr(X, y, k=3, task="regression", mrmr_backend="serial", verbose=False)
    select_mrmr(X, y, k=3, task="regression", mrmr_backend="blas", verbose=False)
    select_mrmr(
        X,
        y,
        k=3,
        task="regression",
        estimator="gaussian",
        subsample=None,
        mrmr_backend="serial",
        verbose=False,
    )


def _markdown_table(records: Iterable[dict]) -> str:
    rows = [
        "| method | backend | n | p | k | n_jobs | BLAS threads | wall seconds | speedup vs serial | selected-feature parity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        speedup = row["speedup_vs_serial"]
        row_display = dict(row)
        row_display["blas_threads"] = row["blas_threads"] if row["blas_threads"] is not None else "-"
        rows.append(
            "| {method} | {backend} | {n} | {p} | {k} | {n_jobs} | {blas_threads} | {wall_seconds:.3f} | "
            "{speedup:.2f}x | {parity} |".format(
                speedup=speedup,
                parity="yes" if row["selected_feature_parity"] else "NO",
                **row_display,
            )
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the fast local benchmark matrix.")
    mode.add_argument("--full", action="store_true", help="Run larger benchmark cases.")
    parser.add_argument(
        "--n-jobs",
        type=_parse_n_jobs,
        default=_parse_n_jobs("2"),
        help="Comma-separated process worker counts for the processes backend, e.g. 2,4,-1.",
    )
    parser.add_argument(
        "--blas-threads",
        type=_parse_blas_threads,
        default=None,
        help=(
            "Comma-separated native BLAS thread limits for the BLAS backend, "
            "e.g. 1,2,4. Omit to run one unbounded BLAS row."
        ),
    )
    parser.add_argument("--repeat", type=int, default=3, help="Run count per row; best time is reported.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    full = bool(args.full)
    blas_thread_values = args.blas_threads if args.blas_threads is not None else [None]
    _warmup()

    records: list[dict] = []
    for case_idx, (method, n, p, k) in enumerate(_cases(full)):
        X, y = _make_data(n, p, seed=10_000 + case_idx)

        serial_time, serial_selected = _time_select(
            X, y, method=method, backend="serial", n_jobs=1, k=k, repeat=args.repeat
        )
        records.append(
            {
                "benchmark": "mrmr",
                "benchmark_kind": "baseline",
                "method": method,
                "backend": "serial",
                "n": n,
                "p": p,
                "k": k,
                "n_jobs": 1,
                "blas_threads": None,
                "wall_seconds": serial_time,
                "baseline_wall_seconds": serial_time,
                "current_wall_seconds": serial_time,
                "baseline_peak_memory_mb": None,
                "current_peak_memory_mb": None,
                "speedup_vs_serial": 1.0,
                "selected_feature_parity": True,
                "promotion_status": "baseline",
                "selected_features": serial_selected,
            }
        )

        for blas_threads in blas_thread_values:
            wall, selected = _time_select(
                X,
                y,
                method=method,
                backend="blas",
                n_jobs=1,
                k=k,
                blas_threads=blas_threads,
                repeat=args.repeat,
            )
            parity = selected == serial_selected
            blas_kind = "promotion" if method == "classic" else "parity"
            records.append(
                {
                    "benchmark": "mrmr",
                    "benchmark_kind": blas_kind,
                    "method": method,
                    "backend": "blas",
                    "n": n,
                    "p": p,
                    "k": k,
                    "n_jobs": 1,
                    "blas_threads": blas_threads,
                    "wall_seconds": wall,
                    "baseline_wall_seconds": serial_time,
                    "current_wall_seconds": wall,
                    "baseline_peak_memory_mb": None,
                    "current_peak_memory_mb": None,
                    "speedup_vs_serial": serial_time / wall if wall > 0 else float("inf"),
                    "selected_feature_parity": parity,
                    "promotion_status": (
                        promotion_status(
                            parity=parity,
                            baseline_seconds=serial_time,
                            current_seconds=wall,
                        )
                        if blas_kind == "promotion"
                        else ("parity" if parity else "blocked: parity")
                    ),
                    "selected_features": selected,
                }
            )

        for process_jobs in args.n_jobs:
            wall, selected = _time_select(
                X,
                y,
                method=method,
                backend="processes",
                n_jobs=process_jobs,
                k=k,
                repeat=args.repeat,
            )
            parity = selected == serial_selected
            records.append(
                {
                    "benchmark": "mrmr",
                    "benchmark_kind": "parity",
                    "method": method,
                    "backend": "processes",
                    "n": n,
                    "p": p,
                    "k": k,
                    "n_jobs": process_jobs,
                    "blas_threads": None,
                    "wall_seconds": wall,
                    "baseline_wall_seconds": serial_time,
                    "current_wall_seconds": wall,
                    "baseline_peak_memory_mb": None,
                    "current_peak_memory_mb": None,
                    "speedup_vs_serial": serial_time / wall if wall > 0 else float("inf"),
                    "selected_feature_parity": parity,
                    "promotion_status": "parity" if parity else "blocked: parity",
                    "selected_features": selected,
                }
            )

    print(_markdown_table(records))

    if args.output is not None:
        write_json(args.output, records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
