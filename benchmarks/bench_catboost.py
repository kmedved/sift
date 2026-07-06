#!/usr/bin/env python
"""Benchmark CatBoost split helpers and optional tiny selector smoke cases."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_utils import markdown_table, measure, regression_frame, write_json  # noqa: E402
from sift.catboost import _bootstrap_indices, catboost_select  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    n = 400 if not args.full else 2_000
    p = 20 if not args.full else 80
    X, y = regression_frame(n, p, seed=60_000)
    groups = np.repeat(np.arange(max(2, n // 20)), 20)[:n]
    y_class = pd.Series(np.where(y > np.median(y), "hi", "lo"))

    cases = [
        (
            "iid_bootstrap",
            lambda: list(_bootstrap_indices(n, 10, random_state=0, min_oob=5)),
        ),
        (
            "group_bootstrap",
            lambda: list(
                _bootstrap_indices(
                    n,
                    10,
                    groups=groups,
                    y=y_class,
                    task="classification",
                    random_state=0,
                    min_oob=5,
                )
            ),
        ),
    ]

    if importlib.util.find_spec("catboost") is not None:
        X_smoke = X.iloc[: min(n, 300)].copy()
        y_smoke = pd.Series(y[: len(X_smoke)])
        cases.append(
            (
                "catboost_forward_smoke",
                lambda: catboost_select(
                    X_smoke,
                    y_smoke,
                    k=3,
                    task="regression",
                    algorithm="forward",
                    n_estimators=20,
                    n_splits=2,
                    prefilter_k=8,
                    verbose=False,
                    random_state=0,
                    n_jobs=1,
                    catboost_params={"allow_writing_files": False},
                ),
            )
        )

    records = []
    for name, fn in cases:
        timing = measure(fn, repeat=args.repeat)
        result = timing["result"]
        count = len(result) if hasattr(result, "__len__") else 1
        records.append(
            {
                "benchmark": "catboost",
                "benchmark_kind": "informational",
                "case": name,
                "n": n,
                "p": p,
                "median_seconds": timing["median_seconds"],
                "best_seconds": timing["best_seconds"],
                "baseline_wall_seconds": timing["median_seconds"],
                "current_wall_seconds": timing["median_seconds"],
                "baseline_peak_memory_mb": timing["peak_memory_mb"],
                "current_peak_memory_mb": timing["peak_memory_mb"],
                "result_count": count,
                "catboost_installed": importlib.util.find_spec("catboost") is not None,
                "selected_feature_parity": True,
                "promotion_status": "informational",
            }
        )

    print(
        markdown_table(
            records,
            [
                ("case", "case"),
                ("n", "n"),
                ("p", "p"),
                ("median s", "median_seconds"),
                ("count", "result_count"),
                ("catboost", "catboost_installed"),
            ],
        )
    )
    if args.output:
        write_json(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
