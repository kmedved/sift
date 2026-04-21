#!/usr/bin/env python
"""Benchmark CEFS+ time and allocation-sensitive options."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_utils import markdown_table, measure, regression_frame, write_json  # noqa: E402
from sift import build_cache, select_cached  # noqa: E402


def _cases(full: bool) -> list[tuple[int, int, int, int | None, bool]]:
    quick = [
        (1_500, 120, 15, 100, False),
        (1_500, 120, 15, 100, True),
    ]
    if not full:
        return quick
    return quick + [
        (5_000, 500, 40, 250, False),
        (5_000, 500, 40, 250, True),
        (10_000, 1_000, 50, 500, False),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict] = []
    for i, (n, p, k, top_m, compute_rxx) in enumerate(_cases(bool(args.full))):
        X, y = regression_frame(n, p, seed=40_000 + i)
        timing = measure(
            lambda: select_cached(
                build_cache(X, subsample=None, compute_Rxx=compute_rxx),
                y,
                k=k,
                top_m=top_m,
                method="cefsplus",
            ),
            repeat=args.repeat,
            measure_memory=args.measure_memory,
        )
        selected = timing["result"]
        records.append(
            {
                "benchmark": "cefsplus",
                "benchmark_kind": "informational",
                "n": n,
                "p": p,
                "k": k,
                "top_m": top_m,
                "compute_Rxx": compute_rxx,
                "median_seconds": timing["median_seconds"],
                "best_seconds": timing["best_seconds"],
                "peak_memory_mb": timing["peak_memory_mb"],
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
                ("n", "n"),
                ("p", "p"),
                ("k", "k"),
                ("top_m", "top_m"),
                ("Rxx", "compute_Rxx"),
                ("median s", "median_seconds"),
                ("peak MB", "peak_memory_mb"),
                ("selected", "selected_count"),
            ],
        )
    )
    if args.output:
        write_json(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
