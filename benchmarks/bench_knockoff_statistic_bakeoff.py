#!/usr/bin/env python
"""Seeded public-API knockoff statistic quality/runtime bakeoff.

Compares relevance, lsm, ridge, and cefsplus through the actual
``sift.select_fdr`` call on four shared designs. This is evidence for the
documented 1.0 default-statistic decision. It does not change 0.9 defaults
and does not upgrade ``approximate_plugin`` validity. Adaptive CEFS+ and
tied/truncated LSM have no general sign-flip proof; realized FDP here is not
a proof of those assumptions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import shlex
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

from threadpoolctl import threadpool_info, threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_runtime_scaling as runtime_scaling  # noqa: E402
from sift import select_fdr  # noqa: E402


PROVENANCE_SCHEMA = "sift-knockoff-statistic-bakeoff-v1"
STATISTICS = ("relevance", "lsm", "ridge", "cefsplus")
DESIGNS = ("independent", "ar1", "block", "dense_weak")
SELECT_FDR_FIXED = {
    "q": 0.1,
    "offset": 1,
    "s_method": "equi",
    "n_draws": 1,
    "n_jobs": 1,
    "verbose": False,
}
CSV_COLUMNS = (
    "study",
    "design",
    "statistic",
    "seed",
    "n",
    "p",
    "n_signal",
    "q",
    "offset",
    "s_method",
    "n_draws",
    "status",
    "n_discoveries",
    "fdp",
    "power",
    "runtime_s",
    "fdr_control",
    "warning_count",
    "warning_messages",
    "error",
    "data_sha256",
    "selection_sha256",
)
RUNNER_RELATIVE = "benchmarks/bench_knockoff_statistic_bakeoff.py"
HELPER_RELATIVE = "benchmarks/bench_runtime_scaling.py"
THREAD_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FULL_N = 800
FULL_P = 40
FULL_SEEDS = 30
SMOKE_N = 160
SMOKE_P = 16
SMOKE_SEEDS = 2


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
    return runtime_scaling._sha256_file(path)


def _source_hashes() -> dict[str, str]:
    hashes = dict(runtime_scaling._source_hashes())
    hashes[RUNNER_RELATIVE] = _sha256_file(REPO_ROOT / RUNNER_RELATIVE)
    hashes[HELPER_RELATIVE] = _sha256_file(REPO_ROOT / HELPER_RELATIVE)
    return hashes


def _source_identity() -> dict[str, Any]:
    state = runtime_scaling._capture_source_state()
    return {
        "commit": state["commit"],
        "source_sha256": _source_hashes(),
    }


def verify_source_unchanged(start_environment: dict[str, Any]) -> dict[str, Any]:
    current = _source_identity()
    if current["commit"] != start_environment.get("commit"):
        raise RuntimeError("git commit changed during the bakeoff")
    if current["source_sha256"] != start_environment.get("source_sha256"):
        raise RuntimeError("hashed source files changed during the bakeoff")
    return current


def _ar1_cov(p: int, rho: float) -> Any:
    import numpy as np

    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _block_cov(p: int, block: int, rho: float) -> Any:
    import numpy as np

    sigma = np.eye(p)
    for start in range(0, p, block):
        end = min(start + block, p)
        size = end - start
        sigma[start:end, start:end] = (1.0 - rho) * np.eye(size) + rho
    return sigma


def design_spec(name: str, p: int) -> dict[str, Any]:
    if name == "independent":
        n_signal = max(6, min(12, p // 2))
        return {
            "covariance": "identity",
            "n_signal": n_signal,
            "amplitude": (1.8, 1.1),
        }
    if name == "ar1":
        n_signal = max(6, min(12, p // 2))
        return {
            "covariance": "ar1",
            "rho": 0.5,
            "n_signal": n_signal,
            "amplitude": (1.8, 1.1),
        }
    if name == "block":
        n_signal = max(6, min(12, p // 2))
        return {
            "covariance": "block",
            "block": 5,
            "rho": 0.7,
            "n_signal": n_signal,
            "amplitude": (1.8, 1.1),
        }
    if name == "dense_weak":
        n_signal = max(8, min(20, (3 * p) // 4))
        return {
            "covariance": "identity",
            "n_signal": n_signal,
            "amplitude": (0.70, 0.45),
        }
    raise ValueError(f"unknown design {name!r}")


def make_design(
    name: str,
    seed: int,
    *,
    n: int,
    p: int,
) -> tuple[Any, Any, set[int], dict[str, Any]]:
    import numpy as np

    spec = design_spec(name, p)
    rng = np.random.default_rng(int(seed))
    if spec["covariance"] == "identity":
        sigma = np.eye(p)
    elif spec["covariance"] == "ar1":
        sigma = _ar1_cov(p, float(spec["rho"]))
    else:
        sigma = _block_cov(p, int(spec["block"]), float(spec["rho"]))
    x = rng.multivariate_normal(np.zeros(p), sigma, size=n)
    n_signal = int(spec["n_signal"])
    low, high = spec["amplitude"]
    beta = np.zeros(p)
    beta[:n_signal] = np.linspace(low, high, n_signal)
    y = x @ beta + rng.normal(scale=1.0, size=n)
    return x, y, set(range(n_signal)), spec


def _data_fingerprint(x, y) -> str:
    import numpy as np

    digest = hashlib.sha256()
    values = np.ascontiguousarray(np.asarray(x, dtype="<f8"))
    target = np.ascontiguousarray(np.asarray(y, dtype="<f8"))
    digest.update(values.tobytes())
    digest.update(target.tobytes())
    return digest.hexdigest()


def _selection_fingerprint(selected: Sequence[object] | None) -> str | None:
    if selected is None:
        return None
    encoded = json.dumps(list(selected), ensure_ascii=False, separators=(",", ":"))
    return _sha256_bytes(encoded.encode("utf-8"))


def _fdp_power(selected: Sequence[int] | None, truth: set[int]) -> tuple[float | None, float | None, int | None]:
    if selected is None:
        return None, None, None
    chosen = set(int(i) for i in selected)
    n_disc = len(chosen)
    false = len(chosen - truth)
    true = len(chosen & truth)
    fdp = false / max(1, n_disc)
    power = true / len(truth)
    return float(fdp), float(power), int(n_disc)


def _format_warning_line(item: dict[str, Any]) -> str:
    return (
        f"{item['phase']}[{item['repeat']}] {item['category']}: {item['message']}"
    )


def evaluate_one(
    *,
    design: str,
    statistic: str,
    seed: int,
    n: int,
    p: int,
    warmup_runs: int,
    timing_repeats: int,
    select_fn=select_fdr,
) -> dict[str, Any]:
    if statistic not in STATISTICS:
        raise ValueError(f"unknown statistic {statistic!r}")
    x, y, truth, spec = make_design(design, seed, n=n, p=p)
    data_sha256 = _data_fingerprint(x, y)
    kwargs = {
        **SELECT_FDR_FIXED,
        "statistic": statistic,
        "random_state": int(seed),
    }
    warning_events: list[dict[str, Any]] = []

    def _call(phase: str, repeat: int):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = select_fn(x, y, **kwargs)
            finally:
                for item in caught:
                    warning_events.append(
                        {
                            "phase": phase,
                            "repeat": int(repeat),
                            "category": item.category.__name__,
                            "message": str(item.message),
                        }
                    )
        selected = list(result.selected_indices or [])
        metadata = dict(result.selector_metadata)
        return selected, metadata

    error = ""
    status = "ok"
    selected = None
    metadata: dict[str, Any] = {}
    samples: list[float] = []
    with threadpool_limits(limits=1):
        threadpools = threadpool_info()
        try:
            expected = None
            for warmup_idx in range(int(warmup_runs)):
                selected, metadata = _call("warmup", warmup_idx)
                expected = _selection_fingerprint(selected)
            if int(timing_repeats) < 1:
                raise ValueError("timing_repeats must be >= 1")
            for timing_idx in range(int(timing_repeats)):
                start = time.perf_counter()
                selected, metadata = _call("timing", timing_idx)
                elapsed = float(time.perf_counter() - start)
                if not math.isfinite(elapsed):
                    raise RuntimeError("non-finite timing sample")
                samples.append(elapsed)
                fingerprint = _selection_fingerprint(selected)
                if expected is not None and fingerprint != expected:
                    raise RuntimeError("select_fdr selection changed across timing repetitions")
                expected = fingerprint
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            selected = None
            metadata = {}

    fdp, power, n_disc = _fdp_power(selected, truth)
    runtime = None if not samples else float(statistics.median(samples))
    effective_num_threads = sorted(
        {int(pool.get("num_threads", 1)) for pool in threadpools}
    )
    return {
        "study": "knockoff_statistic_bakeoff",
        "design": design,
        "statistic": statistic,
        "seed": int(seed),
        "n": int(n),
        "p": int(p),
        "n_signal": int(spec["n_signal"]),
        "q": SELECT_FDR_FIXED["q"],
        "offset": SELECT_FDR_FIXED["offset"],
        "s_method": SELECT_FDR_FIXED["s_method"],
        "n_draws": SELECT_FDR_FIXED["n_draws"],
        "status": status,
        "n_discoveries": n_disc,
        "fdp": fdp,
        "power": power,
        "runtime_s": runtime,
        "fdr_control": metadata.get("fdr_control"),
        "warning_count": len(warning_events),
        "warning_messages": " | ".join(_format_warning_line(item) for item in warning_events),
        "error": error,
        "data_sha256": data_sha256,
        "selection_sha256": _selection_fingerprint(selected),
        "selected_indices": None if selected is None else [int(i) for i in selected],
        "runtime_samples_s": samples,
        "warnings": warning_events,
        "effective_num_threads": effective_num_threads,
        "threadpools_during_timing": threadpools,
        "truth_size": len(truth),
    }


def _mean_se(values: list[float]) -> tuple[float | None, float | None, int]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(finite)
    if n == 0:
        return None, None, 0
    mean = float(statistics.fmean(finite))
    if n == 1:
        return mean, None, n
    se = float(statistics.stdev(finite) / math.sqrt(n))
    return mean, se, n


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        by_cell.setdefault((row["design"], row["statistic"]), []).append(row)
    cells = []
    for design in DESIGNS:
        for statistic in STATISTICS:
            rows = by_cell.get((design, statistic), [])
            ok = [row for row in rows if row["status"] == "ok"]
            failed = [row for row in rows if row["status"] != "ok"]
            warned = [row for row in ok if int(row["warning_count"]) > 0]
            fdp_mean, fdp_se, n_fdp = _mean_se([row["fdp"] for row in ok])
            power_mean, power_se, n_power = _mean_se([row["power"] for row in ok])
            disc_mean, disc_se, _ = _mean_se(
                [row["n_discoveries"] for row in ok if row["n_discoveries"] is not None]
            )
            time_mean, time_se, _ = _mean_se(
                [row["runtime_s"] for row in ok if row["runtime_s"] is not None]
            )
            cells.append(
                {
                    "design": design,
                    "statistic": statistic,
                    "n_ok": len(ok),
                    "n_failed": len(failed),
                    "n_warned": len(warned),
                    "fdp_mean": fdp_mean,
                    "fdp_se": fdp_se,
                    "power_mean": power_mean,
                    "power_se": power_se,
                    "n_discoveries_mean": disc_mean,
                    "n_discoveries_se": disc_se,
                    "runtime_s_mean": time_mean,
                    "runtime_s_se": time_se,
                    "n_metrics": n_fdp if n_fdp == n_power else min(n_fdp, n_power),
                }
            )
    paired = []
    for design in DESIGNS:
        rel = {
            int(row["seed"]): row
            for row in by_cell.get((design, "relevance"), [])
            if row["status"] == "ok"
        }
        ridge = {
            int(row["seed"]): row
            for row in by_cell.get((design, "ridge"), [])
            if row["status"] == "ok"
        }
        shared = sorted(set(rel) & set(ridge))
        d_power = [ridge[seed]["power"] - rel[seed]["power"] for seed in shared]
        d_fdp = [ridge[seed]["fdp"] - rel[seed]["fdp"] for seed in shared]
        d_time = [
            ridge[seed]["runtime_s"] - rel[seed]["runtime_s"]
            for seed in shared
            if ridge[seed]["runtime_s"] is not None and rel[seed]["runtime_s"] is not None
        ]
        p_mean, p_se, n_p = _mean_se(d_power)
        f_mean, f_se, n_f = _mean_se(d_fdp)
        t_mean, t_se, n_t = _mean_se(d_time)
        paired.append(
            {
                "design": design,
                "comparison": "ridge_minus_relevance",
                "n_paired": n_p,
                "power_mean_diff": p_mean,
                "power_se_diff": p_se,
                "fdp_mean_diff": f_mean,
                "fdp_se_diff": f_se,
                "runtime_s_mean_diff": t_mean,
                "runtime_s_se_diff": t_se,
                "n_runtime_paired": n_t,
                "n_fdp_paired": n_f,
            }
        )
    return {"cells": cells, "paired_ridge_minus_relevance": paired}


def _select_fdr_defaults() -> dict[str, Any]:
    defaults = {}
    for name, parameter in inspect.signature(select_fdr).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        defaults[name] = parameter.default
    return {
        key: defaults.get(key)
        for key in ("q", "offset", "s_method", "n_draws", "statistic", "eta", "aggregation")
    }


def capture_environment() -> dict[str, Any]:
    state = runtime_scaling._capture_source_state()
    return {
        "captured_at_utc": state["captured_at_utc"],
        "commit": state["commit"],
        "dirty": state["dirty"],
        "status_porcelain": list(state["status_porcelain"]),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": runtime_scaling._package_versions(),
        "thread_env": {key: os.environ.get(key) for key in THREAD_ENV},
        "select_fdr_defaults": _select_fdr_defaults(),
        "select_fdr_fixed": dict(SELECT_FDR_FIXED),
        "source_sha256": _source_hashes(),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, type(None), str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be serialized")
        return float(value)
    if hasattr(value, "item"):
        return _json_ready(value.item())
    return value


def _record_for_provenance(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **csv_row(record),
        "selected_indices": record.get("selected_indices"),
        "runtime_samples_s": list(record.get("runtime_samples_s") or []),
        "warnings": list(record.get("warnings") or []),
        "effective_num_threads": list(record.get("effective_num_threads") or []),
    }


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    return {column: record.get(column) for column in CSV_COLUMNS}


def write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_row(record))


def write_provenance(
    path: Path,
    *,
    csv_path: Path,
    records: Sequence[dict[str, Any]],
    study: str,
    n: int,
    p: int,
    seeds: Sequence[int],
    warmup_runs: int,
    timing_repeats: int,
    environment: dict[str, Any],
) -> dict[str, Any]:
    env = copy.deepcopy(environment)
    unique_pools = []
    seen_pools = set()
    for record in records:
        pools = record.get("threadpools_during_timing") or []
        key = json.dumps(_json_ready(pools), sort_keys=True)
        if key in seen_pools:
            continue
        seen_pools.add(key)
        unique_pools.append(pools)
    try:
        csv_relative = str(csv_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        csv_relative = str(csv_path)
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "study": study,
        "command": shlex.join(sys.argv),
        "n": int(n),
        "p": int(p),
        "seeds": [int(seed) for seed in seeds],
        "designs": list(DESIGNS),
        "statistics": list(STATISTICS),
        "warmup_runs": int(warmup_runs),
        "timing_repeats": int(timing_repeats),
        "environment": env,
        "git": {
            "commit": env.get("commit"),
            "dirty": env.get("dirty"),
            "status_porcelain": list(env.get("status_porcelain") or []),
            "source_sha256": dict(env.get("source_sha256") or {}),
        },
        "artifact": {
            "csv": csv_relative,
            "sha256": _sha256_file(csv_path),
            "row_count": len(records),
            "columns": list(CSV_COLUMNS),
        },
        "threadpools_during_timing": unique_pools[0] if len(unique_pools) == 1 else unique_pools,
        "records": [_record_for_provenance(record) for record in records],
        "summary": summarize(records),
        "caveats": [
            "Measured FDP is empirical on these Gaussian designs; it does not prove Model-X exchangeability.",
            "SIFT reports approximate_plugin for the default ungrouped single-draw path; this study does not upgrade that claim.",
            "Adaptive CEFS+ and tied/truncated LSM do not have a general sign-flip proof; their rows are quality/runtime measurements, not a validity certificate.",
            "The 0.9 default remains statistic=relevance. Any 1.0 flip is an owner decision from the retained full run.",
        ],
    }
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def render_summary_markdown(summary: dict[str, Any], *, study: str) -> str:
    lines = [
        f"Study `{study}`. Cells are mean ± SE over completed seeds; failed seeds are counted, not converted to empty selections.",
        "",
        "| design | statistic | n ok | n failed | n warned | FDP | power | discoveries | runtime s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in summary["cells"]:
        def _fmt(mean, se):
            if mean is None:
                return "—"
            if se is None:
                return f"{mean:.3f}"
            return f"{mean:.3f} ± {se:.3f}"

        lines.append(
            "| {design} | `{statistic}` | {n_ok} | {n_failed} | {n_warned} | {fdp} | {power} | {disc} | {runtime} |".format(
                design=cell["design"],
                statistic=cell["statistic"],
                n_ok=cell["n_ok"],
                n_failed=cell["n_failed"],
                n_warned=cell["n_warned"],
                fdp=_fmt(cell["fdp_mean"], cell["fdp_se"]),
                power=_fmt(cell["power_mean"], cell["power_se"]),
                disc=_fmt(cell["n_discoveries_mean"], cell["n_discoveries_se"]),
                runtime=_fmt(cell["runtime_s_mean"], cell["runtime_s_se"]),
            )
        )
    lines.extend(
        [
            "",
            "Paired `ridge - relevance` on shared seeds:",
            "",
            "| design | n paired | power diff | FDP diff | runtime s diff |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["paired_ridge_minus_relevance"]:
        def _fmt(mean, se):
            if mean is None:
                return "—"
            if se is None:
                return f"{mean:.3f}"
            return f"{mean:.3f} ± {se:.3f}"

        lines.append(
            "| {design} | {n} | {power} | {fdp} | {runtime} |".format(
                design=row["design"],
                n=row["n_paired"],
                power=_fmt(row["power_mean_diff"], row["power_se_diff"]),
                fdp=_fmt(row["fdp_mean_diff"], row["fdp_se_diff"]),
                runtime=_fmt(row["runtime_s_mean_diff"], row["runtime_s_se_diff"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_study(
    *,
    full: bool,
    seeds: Sequence[int] | None = None,
    n: int | None = None,
    p: int | None = None,
    warmup_runs: int | None = None,
    timing_repeats: int | None = None,
    designs: Sequence[str] = DESIGNS,
    statistics: Sequence[str] = STATISTICS,
    select_fn=select_fdr,
) -> list[dict[str, Any]]:
    n_rows = FULL_N if full else SMOKE_N
    n_cols = FULL_P if full else SMOKE_P
    if n is not None:
        n_rows = int(n)
    if p is not None:
        n_cols = int(p)
    n_seeds = FULL_SEEDS if full else SMOKE_SEEDS
    seed_list = list(seeds) if seeds is not None else list(range(n_seeds))
    warm = 1 if full else 0
    repeats = 1
    if warmup_runs is not None:
        warm = int(warmup_runs)
    if timing_repeats is not None:
        repeats = int(timing_repeats)
    records = []
    for design in designs:
        for seed in seed_list:
            for statistic in statistics:
                records.append(
                    evaluate_one(
                        design=design,
                        statistic=statistic,
                        seed=int(seed),
                        n=n_rows,
                        p=n_cols,
                        warmup_runs=warm,
                        timing_repeats=repeats,
                        select_fn=select_fn,
                    )
                )
    return records


def _apply_thread_env() -> None:
    for key, value in THREAD_ENV.items():
        os.environ.setdefault(key, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="cheap 2-seed functional study")
    mode.add_argument("--full", action="store_true", help="retained 30-seed study")
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV path; provenance is written beside it as *.provenance.json",
    )
    parser.add_argument("--warmup-runs", type=_nonnegative_int)
    parser.add_argument("--timing-repeats", type=_positive_int)
    args = parser.parse_args(argv)
    if not args.smoke and not args.full:
        parser.error("pass --smoke or --full")
    full = bool(args.full)
    output = args.output
    if output is None:
        name = "knockoff_statistic_bakeoff.csv" if full else "knockoff_statistic_bakeoff_smoke.csv"
        output = REPO_ROOT / "benchmarks" / "results" / name
    _apply_thread_env()
    n = FULL_N if full else SMOKE_N
    p = FULL_P if full else SMOKE_P
    seeds = list(range(FULL_SEEDS if full else SMOKE_SEEDS))
    warmup = 1 if full else 0
    repeats = 1
    if args.warmup_runs is not None:
        warmup = args.warmup_runs
    if args.timing_repeats is not None:
        repeats = args.timing_repeats
    start_environment = copy.deepcopy(capture_environment())
    records = run_study(
        full=full,
        seeds=seeds,
        n=n,
        p=p,
        warmup_runs=warmup,
        timing_repeats=repeats,
    )
    verify_source_unchanged(start_environment)
    write_csv(output, records)
    provenance_path = output.with_suffix(".provenance.json")
    payload = write_provenance(
        provenance_path,
        csv_path=output,
        records=records,
        study="full" if full else "smoke",
        n=n,
        p=p,
        seeds=seeds,
        warmup_runs=warmup,
        timing_repeats=repeats,
        environment=start_environment,
    )
    print(render_summary_markdown(payload["summary"], study="full" if full else "smoke"))
    print(f"wrote {output}")
    print(f"wrote {provenance_path}")
    n_failed = sum(1 for row in records if row["status"] != "ok")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
