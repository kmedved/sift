#!/usr/bin/env python
"""Benchmark Gaussian-copula knockoff selection timings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_utils import markdown_table, regression_frame, write_json  # noqa: E402
from sift import build_cache  # noqa: E402
from sift.estimators.copula import weighted_corr_with_vector, weighted_rank_gauss_1d  # noqa: E402
from sift.estimators.knockoffs import (  # noqa: E402
    fit_gaussian_knockoffs,
    gaussian_knockoff_mean,
    sample_gaussian_knockoffs,
)
from sift.selection.knockoff_filter import (  # noqa: E402
    _build_context,
    _get_statistic,
    _weighted_variance,
    knockoff_threshold,
)


def _cases(full: bool) -> list[tuple[int, int, int, str]]:
    quick = [
        (2_000, 100, 1, "relevance"),
        (2_000, 100, 5, "relevance"),
        (2_000, 100, 1, "cefsplus"),
    ]
    if not full:
        return quick
    return quick + [
        (50_000, 500, 1, "relevance"),
        (50_000, 500, 11, "relevance"),
        (50_000, 500, 1, "cefsplus"),
        (50_000, 2_000, 1, "relevance"),
        (50_000, 2_000, 11, "relevance"),
        (50_000, 2_000, 1, "cefsplus"),
    ]


def _timed_case(*, n: int, p: int, n_draws: int, statistic: str, seed: int) -> dict:
    X, y = regression_frame(n, p, seed=seed)

    start = time.perf_counter()
    cache = build_cache(X, subsample=None, compute_Rxx=True)
    cache_seconds = time.perf_counter() - start

    w = np.asarray(cache.sample_weight, dtype=np.float64)
    active = _weighted_variance(cache.Z, w) > 1e-12
    R_active = np.asarray(cache.Rxx, dtype=np.float64)[np.ix_(active, active)]
    Z_active = (
        np.asarray(cache.Z, dtype=np.float32)
        if bool(active.all())
        else np.ascontiguousarray(cache.Z[:, active], dtype=np.float32)
    )

    start = time.perf_counter()
    model = fit_gaussian_knockoffs(R_active)
    fit_seconds = time.perf_counter() - start

    zy = weighted_rank_gauss_1d(np.asarray(y, dtype=np.float32)[cache.row_idx], w)
    r = weighted_corr_with_vector(Z_active, zy, w)
    stat_spec = _get_statistic(statistic)
    seeds = np.random.SeedSequence(seed).spawn(n_draws)
    start = time.perf_counter()
    mean_active = gaussian_knockoff_mean(Z_active, model) if n_draws > 1 else None
    mean_seconds = time.perf_counter() - start

    sample_seconds = 0.0
    stat_seconds = 0.0
    threshold_seconds = 0.0
    selected_by_draw = np.zeros((n_draws, Z_active.shape[1]), dtype=bool)
    thresholds: list[float] = []

    for draw_idx, child in enumerate(seeds):
        rng = np.random.default_rng(child)
        start = time.perf_counter()
        Zt_active = sample_gaussian_knockoffs(Z_active, model, rng, mean=mean_active)
        sample_seconds += time.perf_counter() - start

        start = time.perf_counter()
        context = _build_context(
            Z_active,
            Zt_active,
            zy,
            w,
            model,
            screen_pairs=2000 if stat_spec.needs_screening else None,
            options={"min_gain_ratio": 1e-4} if statistic == "cefsplus" else {},
            n_jobs=1,
            rng=rng,
            build_augmented=stat_spec.needs_screening,
            statistic_name=statistic,
            r=r,
        )
        W = stat_spec.fn(context)
        stat_seconds += time.perf_counter() - start

        start = time.perf_counter()
        threshold = knockoff_threshold(W, 0.1, offset=1)
        threshold_seconds += time.perf_counter() - start
        thresholds.append(threshold)
        if np.isfinite(threshold):
            selected_by_draw[draw_idx] = W >= threshold

    if n_draws == 1:
        selected_count = int(selected_by_draw[0].sum())
    else:
        selected_count = int((selected_by_draw.mean(axis=0) >= 0.5).sum())

    threshold_median = float(np.median(thresholds))
    return {
        "cache_seconds": cache_seconds,
        "fit_seconds": fit_seconds,
        "mean_seconds": mean_seconds,
        "sample_seconds": sample_seconds,
        "stat_seconds": stat_seconds,
        "threshold_seconds": threshold_seconds,
        "total_seconds": cache_seconds + fit_seconds + mean_seconds + sample_seconds + stat_seconds + threshold_seconds,
        "selected_count": selected_count,
        "threshold_median": threshold_median if np.isfinite(threshold_median) else None,
        "s_mean": float(np.mean(model.s)),
        "gamma": float(model.gamma),
        "max_relevance": float(np.max(np.abs(r))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records: list[dict] = []
    for i, (n, p, n_draws, statistic) in enumerate(_cases(bool(args.full))):
        timing = _timed_case(n=n, p=p, n_draws=n_draws, statistic=statistic, seed=90_000 + i)
        records.append(
            {
                "benchmark": "knockoffs",
                "benchmark_kind": "informational",
                "statistic": statistic,
                "n": n,
                "p": p,
                "n_draws": n_draws,
                "median_seconds": timing["total_seconds"],
                "best_seconds": timing["total_seconds"],
                "peak_memory_mb": None,
                "promotion_status": "informational",
                **timing,
            }
        )

    print(
        markdown_table(
            records,
            [
                ("stat", "statistic"),
                ("n", "n"),
                ("p", "p"),
                ("draws", "n_draws"),
                ("cache s", "cache_seconds"),
                ("fit s", "fit_seconds"),
                ("mean s", "mean_seconds"),
                ("sample s", "sample_seconds"),
                ("stat s", "stat_seconds"),
                ("total s", "total_seconds"),
                ("selected", "selected_count"),
            ],
        )
    )
    if args.output:
        write_json(args.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
