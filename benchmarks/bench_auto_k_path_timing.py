#!/usr/bin/env python
"""Measure the fixed-k CEFS+ path denominator used by the Auto-K G5 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from threadpoolctl import threadpool_info, threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.auto_k_designs import DESIGNS  # noqa: E402
from benchmarks.bench_auto_k import _design_max_k  # noqa: E402
from sift import __version__ as sift_version  # noqa: E402
from sift import build_cache  # noqa: E402
from sift.selection.cefsplus import select_cached  # noqa: E402


CSV_COLUMNS = ("design", "seed", "benchmark", "runtime_s")
BENCHMARK = "fixed_k_select_cached"
PROVENANCE_SCHEMA = "sift-auto-k-path-timing-provenance-v1"
SOURCE_PATHS = (
    "benchmarks/bench_auto_k_path_timing.py",
    "benchmarks/bench_auto_k.py",
    "benchmarks/auto_k_designs.py",
    "sift/estimators/copula.py",
    "sift/selection/cefsplus.py",
)
THREAD_ENV_KEYS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"sift": sift_version}
    for distribution in ("numpy", "pandas", "scikit-learn", "scipy", "numba", "threadpoolctl"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _source_hashes() -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        hashes[relative] = _sha256(path) if path.is_file() else None
    return hashes


def _capture_source_state() -> dict[str, object]:
    """Snapshot the code state that will execute before timing begins."""
    status = _git_output("status", "--short", "--untracked-files=all")
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_output("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "source_sha256": _source_hashes(),
    }


def _run_select_cached(cache, y, *, k: int, top_m: int):
    return select_cached(
        cache,
        y,
        k,
        method="cefsplus",
        top_m=top_m,
        return_indices=True,
        return_objective=True,
    )


def measure_d9_path_timing(
    *,
    full: bool,
    seeds: Sequence[int],
    timing_repeats: int,
    warmup_runs: int,
    thread_limit: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return strict G5 rows and detailed per-seed measurement provenance."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(int(seed) < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative")
    if timing_repeats <= 0:
        raise ValueError("timing_repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if thread_limit is not None and thread_limit <= 0:
        raise ValueError("thread_limit must be positive when supplied")

    rows: list[dict[str, object]] = []
    measurements: list[dict[str, object]] = []
    design = DESIGNS["D9"]

    for raw_seed in seeds:
        seed = int(raw_seed)
        X, y, meta = design.make(seed, full)
        n_rows, n_features = X.shape
        k = _design_max_k(n_features, meta)
        top_m = max(5 * k, 250)
        subsample = None if n_rows <= 50_000 else 50_000
        compute_rxx = n_features <= 4000

        cache_start = time.perf_counter()
        cache = build_cache(
            X,
            subsample=subsample,
            random_state=seed,
            compute_Rxx=compute_rxx,
        )
        cache_preparation_s = time.perf_counter() - cache_start

        context = (
            nullcontext()
            if thread_limit is None
            else threadpool_limits(limits=thread_limit)
        )
        with context:
            timing_threadpools = threadpool_info()
            expected_indices: tuple[int, ...] | None = None
            for _ in range(warmup_runs):
                _names, indices, objective = _run_select_cached(cache, y, k=k, top_m=top_m)
                expected_indices = tuple(int(index) for index in indices)
                if len(objective) != len(indices):
                    raise RuntimeError("select_cached returned inconsistent path/objective lengths")

            samples: list[float] = []
            for _ in range(timing_repeats):
                start = time.perf_counter()
                _names, indices, objective = _run_select_cached(cache, y, k=k, top_m=top_m)
                runtime_s = time.perf_counter() - start
                actual_indices = tuple(int(index) for index in indices)
                if expected_indices is not None and actual_indices != expected_indices:
                    raise RuntimeError("select_cached path changed across timing repetitions")
                expected_indices = actual_indices
                if len(objective) != len(indices):
                    raise RuntimeError("select_cached returned inconsistent path/objective lengths")
                samples.append(float(runtime_s))

        median_runtime = float(statistics.median(samples))
        rows.append(
            {
                "design": "D9",
                "seed": seed,
                "benchmark": BENCHMARK,
                "runtime_s": median_runtime,
            }
        )
        measurements.append(
            {
                "design": "D9",
                "seed": seed,
                "shape": [int(n_rows), int(n_features)],
                "resolved_k": int(k),
                "top_m": int(top_m),
                "selected_path_length": len(expected_indices or ()),
                "cache_preparation_s": float(cache_preparation_s),
                "runtime_samples_s": samples,
                "runtime_median_s": median_runtime,
                "threadpools_during_timing": timing_threadpools,
            }
        )

    return rows, measurements


def write_timing_artifacts(
    output_path: Path,
    provenance_path: Path,
    *,
    rows: Sequence[dict[str, object]],
    measurements: Sequence[dict[str, object]],
    full: bool,
    timing_repeats: int,
    warmup_runs: int,
    thread_limit: int | None,
    command_argv: Sequence[str],
    source_state: dict[str, object] | None = None,
) -> None:
    """Write the strict timing CSV and a checksum-bound JSON sidecar."""
    if output_path.resolve() == provenance_path.resolve():
        raise ValueError("CSV output and provenance output must be different paths")

    # CLI runs capture this before measurement begins. The fallback keeps the
    # helper convenient for callers while still preceding artifact creation.
    if source_state is None:
        source_state = _capture_source_state()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    seeds = [int(row["seed"]) for row in rows]
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "columns": list(CSV_COLUMNS),
            "row_count": len(rows),
        },
        "command": {
            "argv": list(command_argv),
            "shell": shlex.join(command_argv),
            "cwd": str(Path.cwd()),
        },
        "configuration": {
            "design": "D9",
            "full": bool(full),
            "seeds": seeds,
            "benchmark": BENCHMARK,
            "timing_repeats": int(timing_repeats),
            "warmup_runs": int(warmup_runs),
            "runtime_aggregation": "median_per_seed",
            "timer": "time.perf_counter",
            "timing_thread_limit": thread_limit,
            "denominator_scope": (
                "complete select_cached call on a pre-built FeatureCache; "
                "cache construction is excluded"
            ),
            "selector": {
                "method": "cefsplus",
                "k_rule": "bench_auto_k._design_max_k",
                "top_m_rule": "max(5 * k, 250)",
                "corr_prune": "auto",
                "return_indices": True,
                "return_objective": True,
            },
            "cache": {
                "subsample_rule": "None when n_rows <= 50000 else 50000",
                "compute_Rxx_rule": "n_features <= 4000",
                "random_state": "seed",
            },
        },
        "git": {
            "captured_at_utc": source_state["captured_at_utc"],
            "capture_scope": "before measurement and artifact creation",
            "commit": source_state["commit"],
            "dirty": source_state["dirty"],
            "status_porcelain": source_state["status_porcelain"],
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "package_versions": _package_versions(),
            "thread_environment": {
                key: os.environ[key] for key in THREAD_ENV_KEYS if key in os.environ
            },
            "threadpools_ambient": threadpool_info(),
        },
        "source_sha256": source_state["source_sha256"],
        "measurements": list(measurements),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="use the 10k x 500 D9 smoke design")
    mode.add_argument("--full", action="store_true", help="use the 50k x 2000 gate design")
    parser.add_argument("--seeds", type=_positive_int, default=2, help="number of consecutive seeds")
    parser.add_argument("--seed-start", type=_nonnegative_int, default=0)
    parser.add_argument("--timing-repeats", type=_positive_int, default=5)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=1)
    parser.add_argument(
        "--thread-limit",
        type=_positive_int,
        help="optional BLAS/OpenMP thread limit applied only to warm-up and timed calls",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="JSON sidecar path (default: replace the CSV suffix with .provenance.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    provenance_output = args.provenance_output or args.output.with_suffix(".provenance.json")
    source_state = _capture_source_state()
    rows, measurements = measure_d9_path_timing(
        full=bool(args.full),
        seeds=seeds,
        timing_repeats=args.timing_repeats,
        warmup_runs=args.warmup_runs,
        thread_limit=args.thread_limit,
    )
    effective_argv = [sys.executable, sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    write_timing_artifacts(
        args.output,
        provenance_output,
        rows=rows,
        measurements=measurements,
        full=bool(args.full),
        timing_repeats=args.timing_repeats,
        warmup_runs=args.warmup_runs,
        thread_limit=args.thread_limit,
        command_argv=effective_argv,
        source_state=source_state,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
