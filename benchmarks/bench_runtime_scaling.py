#!/usr/bin/env python
"""Measure representative end-to-end SIFT selector scaling with provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCHEMA = "sift-runtime-scaling-v1"
PROVENANCE_SCHEMA = "sift-runtime-scaling-provenance-v1"
CSV_COLUMNS = (
    "workload",
    "method",
    "n",
    "p",
    "k",
    "warmup_runs",
    "timing_repeats",
    "p50_s",
    "p95_s",
    "p99_s",
    "peak_rss_mb",
    "million_cells_per_s",
    "selected_count",
    "selection_sha256",
    "data_sha256",
)
METHODS = (
    "mrmr_classic",
    "jmi_r2",
    "jmim_r2",
    "cefsplus",
    "cefsplus_binary",
    "fdr_relevance",
)
FULL_WORKLOADS = (
    ("baseline", 2_000, 100),
    ("tall", 20_000, 100),
    ("wide", 2_000, 500),
)
QUICK_WORKLOADS = (("smoke", 300, 20),)
THREAD_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
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
    return completed.stdout.rstrip("\n")


def _source_hashes() -> dict[str, str]:
    paths = [REPO_ROOT / "benchmarks" / "bench_runtime_scaling.py"]
    paths.extend(sorted((REPO_ROOT / "sift").rglob("*.py")))
    paths.append(REPO_ROOT / "pyproject.toml")
    return {
        str(path.relative_to(REPO_ROOT)): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def _capture_source_state() -> dict[str, object]:
    status = _git_output("status", "--short", "--untracked-files=all")
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_output("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "source_sha256": _source_hashes(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in (
        "sift-feature-selection",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "numba",
        "threadpoolctl",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("at least one timing sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def _make_data(n: int, p: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, p)).astype(np.float64)
    n_signal = min(8, p)
    coefs = np.linspace(2.5, 0.4, n_signal, dtype=np.float64)
    linear = values[:, :n_signal] @ coefs
    y_reg = linear + rng.normal(scale=0.5, size=n)
    latent = linear + rng.normal(scale=1.0, size=n)
    y_binary = (latent >= np.median(latent)).astype(np.int8)
    columns = [f"f{i}" for i in range(p)]
    X = pd.DataFrame(values, columns=columns)
    return X, y_reg, y_binary


def _data_fingerprint(X, y) -> str:
    import numpy as np

    digest = hashlib.sha256()
    values = np.ascontiguousarray(X.to_numpy(dtype="<f8", copy=False))
    target = np.ascontiguousarray(np.asarray(y))
    digest.update(values.tobytes())
    digest.update(target.dtype.str.encode("ascii"))
    digest.update(target.tobytes())
    digest.update("\0".join(map(str, X.columns)).encode("utf-8"))
    return digest.hexdigest()


def _run_selector(method: str, X, y_reg, y_binary, k: int):
    from sift import (
        select_cefsplus,
        select_cefsplus_binary,
        select_fdr,
        select_jmi,
        select_jmim,
        select_mrmr,
    )

    common = {
        "k": k,
        "subsample": None,
        "random_state": 0,
        "verbose": False,
    }
    if method == "mrmr_classic":
        return select_mrmr(
            X,
            y_reg,
            task="regression",
            estimator="classic",
            mrmr_backend="blas",
            n_jobs=1,
            **common,
        )
    if method == "jmi_r2":
        return select_jmi(
            X, y_reg, task="regression", estimator="r2", **common
        )
    if method == "jmim_r2":
        return select_jmim(
            X, y_reg, task="regression", estimator="r2", **common
        )
    if method == "cefsplus":
        return select_cefsplus(X, y_reg, **common)
    if method == "cefsplus_binary":
        return select_cefsplus_binary(X, y_binary, loss="logloss", **common)
    if method == "fdr_relevance":
        return select_fdr(
            X,
            y_reg,
            q=0.1,
            statistic="relevance",
            n_draws=1,
            subsample=None,
            random_state=0,
            n_jobs=1,
            verbose=False,
        ).selected_features
    raise ValueError(f"unknown method {method!r}")


def _selection_fingerprint(selected: Sequence[object]) -> str:
    encoded = json.dumps(list(selected), ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(encoded.encode("utf-8"))


def _measure_worker(
    *,
    method: str,
    workload: str,
    n: int,
    p: int,
    k: int,
    seed: int,
    warmup_runs: int,
    timing_repeats: int,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")

    from threadpoolctl import threadpool_info, threadpool_limits

    X, y_reg, y_binary = _make_data(n, p, seed)
    y = y_binary if method == "cefsplus_binary" else y_reg
    data_sha256 = _data_fingerprint(X, y)
    expected_fingerprint: str | None = None
    selected_count: int | None = None

    with threadpool_limits(limits=1):
        threadpools = threadpool_info()
        for _ in range(warmup_runs):
            selected = list(_run_selector(method, X, y_reg, y_binary, k))
            expected_fingerprint = _selection_fingerprint(selected)
            selected_count = len(selected)

        samples: list[float] = []
        for _ in range(timing_repeats):
            start = time.perf_counter()
            selected = list(_run_selector(method, X, y_reg, y_binary, k))
            samples.append(float(time.perf_counter() - start))
            fingerprint = _selection_fingerprint(selected)
            if expected_fingerprint is not None and fingerprint != expected_fingerprint:
                raise RuntimeError(
                    f"{method} selection changed across timing repetitions"
                )
            expected_fingerprint = fingerprint
            selected_count = len(selected)

    p50 = _percentile(samples, 0.50)
    return {
        "schema": SCHEMA,
        "workload": workload,
        "method": method,
        "n": int(n),
        "p": int(p),
        "k": int(k),
        "seed": int(seed),
        "warmup_runs": int(warmup_runs),
        "timing_repeats": int(timing_repeats),
        "runtime_samples_s": samples,
        "p50_s": p50,
        "p95_s": _percentile(samples, 0.95),
        "p99_s": _percentile(samples, 0.99),
        "peak_rss_mb": _peak_rss_mb(),
        "million_cells_per_s": (n * p) / p50 / 1_000_000.0,
        "selected_count": int(selected_count or 0),
        "selection_sha256": str(expected_fingerprint),
        "data_sha256": data_sha256,
        "threadpools_during_timing": threadpools,
    }


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {unknown}")
    if not methods:
        raise argparse.ArgumentTypeError("at least one method is required")
    if len(methods) != len(set(methods)):
        raise argparse.ArgumentTypeError("methods must not repeat")
    return methods


def _worker_command(
    *,
    method: str,
    workload: str,
    n: int,
    p: int,
    k: int,
    seed: int,
    warmup_runs: int,
    timing_repeats: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method,
        "--workload",
        workload,
        "--n",
        str(n),
        "--p",
        str(p),
        "--k",
        str(k),
        "--seed",
        str(seed),
        "--warmup-runs",
        str(warmup_runs),
        "--timing-repeats",
        str(timing_repeats),
    ]


def _invoke_worker(**kwargs) -> dict[str, object]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONWARNINGS"] = "error"
    completed = subprocess.run(
        _worker_command(**kwargs),
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _csv_row(measurement: dict[str, object]) -> dict[str, object]:
    return {column: measurement[column] for column in CSV_COLUMNS}


def _write_artifacts(
    output: Path,
    provenance_output: Path,
    *,
    measurements: Sequence[dict[str, object]],
    source_state: dict[str, object],
    command_argv: Sequence[str],
    full: bool,
    methods: Sequence[str],
) -> None:
    if output.resolve() == provenance_output.resolve():
        raise ValueError("CSV output and provenance output must be different paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_row(item) for item in measurements)

    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "columns": list(CSV_COLUMNS),
            "row_count": len(measurements),
        },
        "command": {
            "argv": list(command_argv),
            "shell": shlex.join(command_argv),
            "cwd": str(Path.cwd()),
        },
        "configuration": {
            "full": bool(full),
            "workloads": [
                {"name": name, "n": n, "p": p}
                for name, n, p in (FULL_WORKLOADS if full else QUICK_WORKLOADS)
            ],
            "methods": list(methods),
            "k": "min(10, p)",
            "seed": 20260903,
            "timer": "time.perf_counter",
            "runtime_aggregation": "linear-interpolated p50/p95/p99",
            "process_isolation": "one fresh worker per method/workload",
            "peak_rss_scope": (
                "whole worker process after data generation, warm-up, and timed calls; "
                "not incremental selector allocation"
            ),
            "throughput": "n * p / p50_s, reported as million input cells/s",
            "thread_environment": THREAD_ENV,
            "selector_options": {
                "common": {
                    "subsample": None,
                    "random_state": 0,
                    "verbose": False,
                },
                "mrmr_classic": {"task": "regression", "mrmr_backend": "blas", "n_jobs": 1},
                "jmi_r2": {"task": "regression", "estimator": "r2"},
                "jmim_r2": {"task": "regression", "estimator": "r2"},
                "cefsplus": {"target": "regression"},
                "cefsplus_binary": {"loss": "logloss"},
                "fdr_relevance": {
                    "q": 0.1,
                    "statistic": "relevance",
                    "n_draws": 1,
                    "n_jobs": 1,
                },
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
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "package_versions": _package_versions(),
        },
        "source_sha256": source_state["source_sha256"],
        "measurements": list(measurements),
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown_table(rows: Sequence[dict[str, str]]) -> str:
    lines = [
        "| workload | n | p | method | p50 s | p95 s | peak RSS MB | M cells/s | selected |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {n:,} | {p:,} | `{method}` | {p50:.4f} | "
            "{p95:.4f} | {rss:.1f} | {rate:.2f} | {selected} |".format(
                workload=row["workload"],
                n=int(row["n"]),
                p=int(row["p"]),
                method=row["method"],
                p50=float(row["p50_s"]),
                p95=float(row["p95_s"]),
                rss=float(row["peak_rss_mb"]),
                rate=float(row["million_cells_per_s"]),
                selected=int(row["selected_count"]),
            )
        )
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(f"unexpected runtime-scaling CSV columns in {path}")
        return list(reader)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--methods", type=_parse_methods, default=METHODS)
    parser.add_argument("--warmup-runs", type=_nonnegative_int, default=1)
    parser.add_argument("--timing-repeats", type=_positive_int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provenance-output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--workload", help=argparse.SUPPRESS)
    parser.add_argument("--n", type=_positive_int, help=argparse.SUPPRESS)
    parser.add_argument("--p", type=_positive_int, help=argparse.SUPPRESS)
    parser.add_argument("--k", type=_positive_int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=_nonnegative_int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        required = (args.method, args.workload, args.n, args.p, args.k, args.seed)
        if any(value is None for value in required):
            parser.error("worker mode requires method, workload, n, p, k, and seed")
        result = _measure_worker(
            method=args.method,
            workload=args.workload,
            n=args.n,
            p=args.p,
            k=args.k,
            seed=args.seed,
            warmup_runs=args.warmup_runs,
            timing_repeats=args.timing_repeats,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.quick == args.full:
        parser.error("choose exactly one of --quick or --full")
    if args.output is None:
        parser.error("--output is required")
    provenance_output = args.provenance_output or args.output.with_suffix(
        ".provenance.json"
    )
    source_state = _capture_source_state()
    workloads = FULL_WORKLOADS if args.full else QUICK_WORKLOADS
    measurements: list[dict[str, object]] = []
    for workload, n, p in workloads:
        k = min(10, p)
        for method in args.methods:
            print(f"measuring {workload}/{method} (n={n}, p={p}, k={k})", flush=True)
            measurements.append(
                _invoke_worker(
                    method=method,
                    workload=workload,
                    n=n,
                    p=p,
                    k=k,
                    seed=20260903,
                    warmup_runs=args.warmup_runs,
                    timing_repeats=args.timing_repeats,
                )
            )

    effective_argv = [sys.executable, sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    _write_artifacts(
        args.output,
        provenance_output,
        measurements=measurements,
        source_state=source_state,
        command_argv=effective_argv,
        full=bool(args.full),
        methods=args.methods,
    )
    print(render_markdown_table([_csv_row(item) for item in measurements]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
