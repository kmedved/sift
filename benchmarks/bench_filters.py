#!/usr/bin/env python
"""Benchmark end-to-end filter selectors with promotion-style JSON output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_utils import markdown_table, measure, regression_frame, write_json  # noqa: E402
from sift import select_cefsplus, select_jmi, select_jmim, select_mrmr  # noqa: E402


def _cases(full: bool) -> list[tuple[str, str, int, int, int]]:
    quick = [
        ("mrmr", "classic", 2_000, 120, 15),
        ("jmi", "r2", 1_500, 100, 12),
        ("jmim", "r2", 1_500, 100, 12),
        ("cefsplus", "gaussian", 1_500, 100, 12),
    ]
    if not full:
        return quick
    return quick + [
        ("mrmr", "gaussian", 5_000, 250, 25),
        ("jmi", "r2", 5_000, 250, 25),
        ("jmim", "r2", 5_000, 250, 25),
        ("cefsplus", "gaussian", 5_000, 250, 25),
    ]


def _select(selector: str, estimator: str, X, y, k: int) -> list[str]:
    common = dict(k=k, top_m=max(5 * k, 100), subsample=None, verbose=False)
    if selector == "mrmr":
        return select_mrmr(
            X,
            y,
            task="regression",
            estimator=estimator,
            mrmr_backend="blas" if estimator == "classic" else "serial",
            **common,
        )
    if selector == "jmi":
        return select_jmi(X, y, task="regression", estimator=estimator, **common)
    if selector == "jmim":
        return select_jmim(X, y, task="regression", estimator=estimator, **common)
    if selector == "cefsplus":
        return select_cefsplus(X, y, **common)
    raise ValueError(selector)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict] = []
    for i, (selector, estimator, n, p, k) in enumerate(_cases(bool(args.full))):
        X, y = regression_frame(n, p, seed=30_000 + i)
        timing = measure(lambda: _select(selector, estimator, X, y, k), repeat=args.repeat)
        selected = timing["result"]
        records.append(
            {
                "benchmark": "filters",
                "benchmark_kind": "informational",
                "selector": selector,
                "estimator": estimator,
                "n": n,
                "p": p,
                "k": k,
                "median_seconds": timing["median_seconds"],
                "best_seconds": timing["best_seconds"],
                "baseline_wall_seconds": timing["median_seconds"],
                "current_wall_seconds": timing["median_seconds"],
                "baseline_peak_memory_mb": timing["peak_memory_mb"],
                "current_peak_memory_mb": timing["peak_memory_mb"],
                "selected_count": len(selected),
                "selected_feature_parity": True,
                "promotion_status": "informational",
                "selected_features": selected,
            }
        )

    print(
        markdown_table(
            records,
            [
                ("selector", "selector"),
                ("estimator", "estimator"),
                ("n", "n"),
                ("p", "p"),
                ("k", "k"),
                ("median s", "median_seconds"),
                ("selected", "selected_count"),
                ("status", "promotion_status"),
            ],
        )
    )
    if args.output:
        write_json(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
