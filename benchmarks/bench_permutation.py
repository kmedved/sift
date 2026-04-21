#!/usr/bin/env python
"""Benchmark permutation importance DataFrame and ndarray paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sift.importance import permutation_importance  # noqa: E402
from benchmarks.bench_utils import write_json  # noqa: E402


class LinearPredictor:
    def __init__(self, coef: np.ndarray) -> None:
        self.coef = np.asarray(coef)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(copy=False)
        return np.asarray(X) @ self.coef


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if 0 in values:
        raise argparse.ArgumentTypeError("n_jobs=0 is invalid")
    return values


def _parse_backends(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one backend")
    invalid = [value for value in values if value not in {"threads", "processes"}]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown backend(s): {invalid}. Expected threads or processes."
        )
    return values


def _make_data(
    n_samples: int,
    n_features: int,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, LinearPredictor]:
    rng = np.random.default_rng(seed)
    X_arr = rng.normal(size=(n_samples, n_features)).astype(np.float64)
    coef = np.zeros(n_features, dtype=np.float64)
    signal_count = min(10, n_features)
    coef[:signal_count] = np.linspace(2.0, 0.25, signal_count)
    y = X_arr @ coef + rng.normal(scale=0.1, size=n_samples)
    X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    return X_df, X_arr, y, LinearPredictor(coef)


def _time_importance(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    model: LinearPredictor,
    *,
    n_repeats: int,
    n_jobs: int,
    backend: str,
) -> tuple[float, str | int, float]:
    start = time.perf_counter()
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mse",
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        parallel_backend=backend,
        random_state=0,
    )
    wall_seconds = time.perf_counter() - start
    top_feature = result["feature"].iloc[0]
    baseline_score = float(result["baseline_score"].iloc[0])
    return wall_seconds, top_feature, baseline_score


def _markdown_table(records: list[dict]) -> str:
    rows = [
        "| input | backend | n | p | repeats | n_jobs | wall seconds | top feature | baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in records:
        rows.append(
            "| {input} | {backend} | {n_samples} | {n_features} | {n_repeats} | "
            "{n_jobs} | {wall_seconds:.3f} | {top_feature} | {baseline_score:.6f} |".format(
                **row
            )
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run a quick local benchmark.")
    mode.add_argument("--full", action="store_true", help="Run a larger benchmark.")
    parser.add_argument("--n-samples", type=int, help="Override sample count.")
    parser.add_argument("--n-features", type=int, help="Override feature count.")
    parser.add_argument("--n-repeats", type=int, help="Override repeat count.")
    parser.add_argument(
        "--n-jobs",
        type=_parse_csv_ints,
        default=_parse_csv_ints("1,2"),
        help="Comma-separated job counts, e.g. 1,2,-1.",
    )
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=_parse_backends("threads,processes"),
        help="Comma-separated backend preferences: threads,processes.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    full = bool(args.full)
    n_samples = args.n_samples or (10_000 if full else 2_000)
    n_features = args.n_features or (120 if full else 40)
    n_repeats = args.n_repeats or (5 if full else 3)

    X_df, X_arr, y, model = _make_data(n_samples, n_features, seed=1234)

    records = []
    for input_name, X in [("dataframe", X_df), ("ndarray", X_arr)]:
        for backend in args.backends:
            for n_jobs in args.n_jobs:
                wall_seconds, top_feature, baseline_score = _time_importance(
                    X,
                    y,
                    model,
                    n_repeats=n_repeats,
                    n_jobs=n_jobs,
                    backend=backend,
                )
                records.append(
                    {
                        "benchmark": "permutation",
                        "input": input_name,
                        "backend": backend,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_repeats": n_repeats,
                        "n_jobs": n_jobs,
                        "wall_seconds": wall_seconds,
                        "top_feature": top_feature,
                        "baseline_score": baseline_score,
                    }
                )

    baselines = {
        row["input"]: row
        for row in records
        if row["backend"] == "threads" and row["n_jobs"] == 1
    }
    for row in records:
        baseline = baselines.get(row["input"], row)
        parity = row["top_feature"] == baseline["top_feature"]
        is_baseline = row is baseline
        row["benchmark_kind"] = "baseline" if is_baseline else "parity"
        row["baseline_wall_seconds"] = baseline["wall_seconds"]
        row["current_wall_seconds"] = row["wall_seconds"]
        row["baseline_peak_memory_mb"] = None
        row["current_peak_memory_mb"] = None
        row["selected_feature_parity"] = parity
        row["promotion_status"] = (
            "baseline"
            if is_baseline
            else ("parity" if parity else "blocked: parity")
        )

    print(_markdown_table(records))

    if args.output is not None:
        write_json(args.output, records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
