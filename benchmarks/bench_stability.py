#!/usr/bin/env python
"""Benchmark stability-selection split streaming and fit memory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_utils import markdown_table, measure, write_json  # noqa: E402
from sift import StabilitySelector  # noqa: E402
from sift.sampling.smart import SmartSamplerConfig, smart_sample  # noqa: E402
from sift.stability import _block_bootstrap_indices, _bootstrap_indices  # noqa: E402


def _data(n: int, p: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.25, size=n).astype(np.float32)
    groups = np.repeat(np.arange(max(2, n // 50)), 50)[:n]
    if len(groups) < n:
        groups = np.r_[groups, np.full(n - len(groups), groups[-1] + 1)]
    time = np.tile(np.arange(50), int(np.ceil(n / 50)))[:n]
    return X, y, groups, time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    n, p, n_bootstrap = (400, 30, 8) if not args.full else (2_000, 80, 30)
    X, y, groups, time = _data(n, p, seed=50_000)

    split_cases = [
        (
            "iid_splits",
            lambda: list(
                _bootstrap_indices(
                    n=n,
                    n_bootstrap=n_bootstrap,
                    sample_frac=0.5,
                    random_state=0,
                )
            ),
        ),
        (
            "block_splits",
            lambda: list(
                _block_bootstrap_indices(
                    n=n,
                    n_bootstrap=n_bootstrap,
                    groups=groups,
                    time=time,
                    block_size=5,
                    random_state=0,
                    min_oob=5,
                )
            ),
        ),
    ]
    fit_cases = [
        (
            "fit_store_coefs",
            lambda: StabilitySelector(
                n_bootstrap=n_bootstrap,
                threshold=0.1,
                alpha=0.02,
                store_coefs=True,
                n_jobs=1,
                random_state=0,
                verbose=False,
            ).fit(X, y),
        ),
        (
            "fit_no_store_coefs",
            lambda: StabilitySelector(
                n_bootstrap=n_bootstrap,
                threshold=0.1,
                alpha=0.02,
                store_coefs=False,
                n_jobs=1,
                random_state=0,
                verbose=False,
            ).fit(X, y),
        ),
    ]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["group"] = groups
    df["label"] = np.where(y > np.median(y), "hi", "lo")
    smart_cases = [
        (
            "smart_sample_residual_off",
            lambda: smart_sample(
                df,
                [f"f{i}" for i in range(p)],
                "label",
                config=SmartSamplerConfig(
                    sample_frac=0.25,
                    group_col="group",
                    min_per_group=1,
                    residual_weight_cap=0.0,
                    random_state=0,
                    verbose=False,
                ),
            ),
        )
    ]

    records: list[dict] = []
    for name, fn in split_cases + fit_cases + smart_cases:
        timing = measure(fn, repeat=args.repeat, measure_memory=args.measure_memory)
        result = timing["result"]
        if hasattr(result, "n_features_selected_"):
            selected_count = int(result.n_features_selected_)
        elif isinstance(result, pd.DataFrame):
            selected_count = len(result)
        else:
            selected_count = len(result)
        records.append(
            {
                "benchmark": "stability",
                "benchmark_kind": "informational",
                "case": name,
                "n": n,
                "p": p,
                "n_bootstrap": n_bootstrap,
                "median_seconds": timing["median_seconds"],
                "best_seconds": timing["best_seconds"],
                "peak_memory_mb": timing["peak_memory_mb"],
                "baseline_wall_seconds": timing["median_seconds"],
                "current_wall_seconds": timing["median_seconds"],
                "baseline_peak_memory_mb": timing["peak_memory_mb"],
                "current_peak_memory_mb": timing["peak_memory_mb"],
                "selected_count": selected_count,
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
                ("boot", "n_bootstrap"),
                ("median s", "median_seconds"),
                ("peak MB", "peak_memory_mb"),
                ("count", "selected_count"),
            ],
        )
    )
    if args.output:
        write_json(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
