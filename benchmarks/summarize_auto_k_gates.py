#!/usr/bin/env python
"""Regenerate the Auto-K v2 G1-G6 gate table from recorded raw inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


RAW_COLUMNS = (
    "design",
    "seed",
    "method",
    "k_hat",
    "k_oracle",
    "k_star",
    "rmse_hat",
    "rmse_oracle",
    "regret_frac",
    "support_precision",
    "support_recall",
    "support_f1",
    "k_dispersion_group",
    "saturated_min",
    "saturated_max",
    "runtime_s",
    "notes",
)

PATH_TIMING_COLUMNS = ("design", "seed", "benchmark", "runtime_s")
PATH_TIMING_BENCHMARK = "fixed_k_select_cached"
PATH_TIMING_PROVENANCE_SCHEMA = "sift-auto-k-path-timing-provenance-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

GATE_COLUMNS = (
    "method",
    "G1_accuracy",
    "G2_or_D5_null",
    "G3_dense_weak",
    "G4_structure",
    "G5_runtime",
    "G5_runtime_ratio",
    "G6_stability",
    "program_mean_regret_D1_D3_D7",
    "program_std_k_over_oracle",
    "D5_main_p_gt3",
    "D5_main_max_k",
    "program_success",
)

REFERENCE_METHODS = (
    "elbow",
    "penalized/bic",
    "evaluate/time_holdout/best",
    "evaluate/one_se",
    "fixed_k=50",
    "oracle",
)
COMPARISON_BASELINES = REFERENCE_METHODS[:4]
CANDIDATE_METHODS = (
    "penalized/ebic",
    "penalized/ric",
    "chi2_stop",
    "forward_stop",
    "perm_gap",
    "knockoff_path",
    "xfit_objective",
    "gaussian_cv",
    "k_posterior",
    "stability",
    "changepoint",
    "consensus",
    "gaussian_cv/best",
)

G1_DESIGNS = ("D1", "D2", "D3", "D7")
G6_DESIGNS = ("D1", "D2", "D3")
REQUIRED_MAIN_DESIGNS = (*G1_DESIGNS, "D4", "D5", "D6", "D8")
PROGRAM_DESIGNS = G1_DESIGNS

DEEP_NULL_METHODS = (
    "penalized/ebic",
    "chi2_stop",
    "forward_stop",
    "perm_gap",
    "knockoff_path",
)
CALIBRATION_LEVELS = {
    "chi2_stop": 0.05,
    "forward_stop": 0.05,
    "perm_gap": 0.05,
    "knockoff_path": 0.2,
}

PATH_ONLY_METHODS = {
    "penalized/ebic",
    "penalized/ric",
    "chi2_stop",
    "forward_stop",
    "k_posterior",
    "changepoint",
}
FOLD_METHODS = {"xfit_objective", "gaussian_cv", "gaussian_cv/best"}


class GateSummaryError(ValueError):
    """Raised when campaign inputs cannot support an auditable gate table."""


@dataclass(frozen=True)
class BenchmarkRow:
    design: str
    seed: int
    method: str
    k_hat: int
    k_oracle: int
    regret_frac: float
    runtime_s: float


@dataclass(frozen=True)
class PathTimingRow:
    design: str
    seed: int
    runtime_s: float


def _schema_error(path: Path, expected: Sequence[str], actual: Sequence[str]) -> GateSummaryError:
    return GateSummaryError(
        f"{path}: CSV schema mismatch; expected {list(expected)!r}, got {list(actual)!r}"
    )


def _parse_int(path: Path, line: int, field: str, value: str, *, optional: bool = False) -> int | None:
    if optional and value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise GateSummaryError(f"{path}:{line}: {field} must be an integer") from exc
    return parsed


def _parse_float(
    path: Path,
    line: int,
    field: str,
    value: str,
    *,
    allow_nan: bool = False,
) -> float:
    if allow_nan and value == "":
        return math.nan
    try:
        parsed = float(value)
    except ValueError as exc:
        raise GateSummaryError(f"{path}:{line}: {field} must be numeric") from exc
    if math.isnan(parsed) and allow_nan:
        return parsed
    if not math.isfinite(parsed):
        raise GateSummaryError(f"{path}:{line}: {field} must be finite")
    return parsed


def _read_benchmark_csv(path: Path) -> list[BenchmarkRow]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise GateSummaryError(f"cannot read benchmark input {path}: {exc}") from exc

    rows: list[BenchmarkRow] = []
    seen: set[tuple[str, int, str]] = set()
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(RAW_COLUMNS):
            raise _schema_error(path, RAW_COLUMNS, reader.fieldnames or ())
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(raw[field] is None for field in RAW_COLUMNS):
                raise GateSummaryError(f"{path}:{line}: row does not match the CSV schema")
            design = raw["design"].strip()
            method = raw["method"].strip()
            if not design or not method:
                raise GateSummaryError(f"{path}:{line}: design and method must be non-empty")

            seed = _parse_int(path, line, "seed", raw["seed"])
            k_hat = _parse_int(path, line, "k_hat", raw["k_hat"])
            k_oracle = _parse_int(path, line, "k_oracle", raw["k_oracle"])
            assert seed is not None and k_hat is not None and k_oracle is not None
            if seed < 0 or k_hat < 0 or k_oracle < 0:
                raise GateSummaryError(
                    f"{path}:{line}: seed, k_hat, and k_oracle must be non-negative"
                )
            _parse_int(path, line, "k_star", raw["k_star"], optional=True)

            for field in ("rmse_hat", "rmse_oracle", "support_precision", "support_recall", "support_f1"):
                _parse_float(path, line, field, raw[field])
            regret = _parse_float(
                path,
                line,
                "regret_frac",
                raw["regret_frac"],
                allow_nan=True,
            )
            runtime = _parse_float(path, line, "runtime_s", raw["runtime_s"])
            if runtime < 0:
                raise GateSummaryError(f"{path}:{line}: runtime_s must be non-negative")
            for field in ("saturated_min", "saturated_max"):
                if raw[field] not in {"True", "False"}:
                    raise GateSummaryError(
                        f"{path}:{line}: {field} must be exactly True or False"
                    )

            key = (design, seed, method)
            if key in seen:
                raise GateSummaryError(f"{path}:{line}: duplicate row key {key!r}")
            seen.add(key)
            rows.append(
                BenchmarkRow(
                    design=design,
                    seed=seed,
                    method=method,
                    k_hat=k_hat,
                    k_oracle=k_oracle,
                    regret_frac=regret,
                    runtime_s=runtime,
                )
            )
    if not rows:
        raise GateSummaryError(f"{path}: benchmark input is empty")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha256(commit: str, relative: Path) -> str:
    """Hash a recorded source file from its provenance commit."""

    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GateSummaryError(
            "cannot verify recorded source at provenance commit "
            f"{commit!r}: {relative.as_posix()}"
            + (f" ({detail})" if detail else "")
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def _read_path_timing_provenance(
    path: Path,
    rows: Sequence[PathTimingRow],
    *,
    require_clean: bool,
    verify_source_hashes: bool,
) -> None:
    provenance_path = path.with_suffix(".provenance.json")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateSummaryError(
            f"cannot read fixed-k path timing provenance {provenance_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GateSummaryError(
            f"{provenance_path}: invalid JSON provenance: {exc}"
        ) from exc
    if not isinstance(provenance, dict):
        raise GateSummaryError(f"{provenance_path}: provenance must be a JSON object")
    if provenance.get("schema") != PATH_TIMING_PROVENANCE_SCHEMA:
        raise GateSummaryError(
            f"{provenance_path}: unsupported provenance schema; expected "
            f"{PATH_TIMING_PROVENANCE_SCHEMA!r}"
        )

    artifact = provenance.get("artifact")
    if not isinstance(artifact, dict):
        raise GateSummaryError(f"{provenance_path}: artifact provenance is required")
    if artifact.get("sha256") != _sha256(path):
        raise GateSummaryError(
            f"{provenance_path}: artifact checksum does not match {path}"
        )
    if artifact.get("columns") != list(PATH_TIMING_COLUMNS):
        raise GateSummaryError(f"{provenance_path}: artifact columns do not match the CSV schema")
    if artifact.get("row_count") != len(rows):
        raise GateSummaryError(f"{provenance_path}: artifact row_count does not match the CSV")

    configuration = provenance.get("configuration")
    if not isinstance(configuration, dict):
        raise GateSummaryError(f"{provenance_path}: configuration provenance is required")
    if configuration.get("full") is not True:
        raise GateSummaryError(
            f"{provenance_path}: full-size path timing provenance is required; quick runs "
            "cannot feed release gates"
        )
    if configuration.get("design") != "D9":
        raise GateSummaryError(f"{provenance_path}: configuration design must be D9")
    if configuration.get("benchmark") != PATH_TIMING_BENCHMARK:
        raise GateSummaryError(
            f"{provenance_path}: configuration benchmark must be "
            f"{PATH_TIMING_BENCHMARK!r}"
        )
    expected_seeds = [row.seed for row in rows]
    if configuration.get("seeds") != expected_seeds:
        raise GateSummaryError(
            f"{provenance_path}: configuration seeds do not match the CSV rows"
        )

    git = provenance.get("git")
    if not isinstance(git, dict):
        raise GateSummaryError(f"{provenance_path}: git provenance is required")
    if not isinstance(git.get("commit"), str) or not git["commit"]:
        raise GateSummaryError(f"{provenance_path}: git commit provenance is required")
    if require_clean and git.get("dirty") is not False:
        raise GateSummaryError(
            f"{provenance_path}: clean git provenance is required for release gates"
        )

    source_hashes = provenance.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise GateSummaryError(f"{provenance_path}: non-empty source_sha256 is required")
    if verify_source_hashes:
        commit = git["commit"]
        for relative, expected_hash in source_hashes.items():
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise GateSummaryError(
                    f"{provenance_path}: source_sha256 must map paths to checksums"
                )
            relative_path = Path(relative)
            source_path = (REPO_ROOT / relative_path).resolve()
            if relative_path.is_absolute() or not source_path.is_relative_to(REPO_ROOT):
                raise GateSummaryError(
                    f"{provenance_path}: recorded source path must stay inside the repository: "
                    f"{relative}"
                )
            if _git_blob_sha256(commit, relative_path) != expected_hash:
                raise GateSummaryError(
                    f"{provenance_path}: recorded source checksum does not match "
                    f"provenance commit {commit}: {relative}"
                )


def _read_path_timing_csv(
    path: Path | None,
    *,
    require_clean: bool = True,
    verify_source_hashes: bool = True,
) -> list[PathTimingRow]:
    if path is None:
        raise GateSummaryError(
            "fixed-k path timing provenance is required; the post-path "
            "fixed_k=50 rows in auto_k_v2_d9.csv are not a substitute"
        )
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise GateSummaryError(f"cannot read fixed-k path timing input {path}: {exc}") from exc

    rows: list[PathTimingRow] = []
    seen: set[int] = set()
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(PATH_TIMING_COLUMNS):
            raise _schema_error(path, PATH_TIMING_COLUMNS, reader.fieldnames or ())
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(raw[field] is None for field in PATH_TIMING_COLUMNS):
                raise GateSummaryError(f"{path}:{line}: row does not match the CSV schema")
            seed = _parse_int(path, line, "seed", raw["seed"])
            assert seed is not None
            if seed < 0:
                raise GateSummaryError(f"{path}:{line}: seed must be non-negative")
            if raw["design"] != "D9":
                raise GateSummaryError(f"{path}:{line}: path timing design must be D9")
            if raw["benchmark"] != PATH_TIMING_BENCHMARK:
                raise GateSummaryError(
                    f"{path}:{line}: benchmark must be {PATH_TIMING_BENCHMARK!r}"
                )
            runtime = _parse_float(path, line, "runtime_s", raw["runtime_s"])
            if runtime <= 0:
                raise GateSummaryError(f"{path}:{line}: runtime_s must be positive")
            if seed in seen:
                raise GateSummaryError(f"{path}:{line}: duplicate path timing seed {seed}")
            seen.add(seed)
            rows.append(PathTimingRow(design="D9", seed=seed, runtime_s=runtime))
    if not rows:
        raise GateSummaryError(f"{path}: fixed-k path timing input is empty")
    _read_path_timing_provenance(
        path,
        rows,
        require_clean=require_clean,
        verify_source_hashes=verify_source_hashes,
    )
    return rows


def _group(rows: Iterable[BenchmarkRow]) -> dict[tuple[str, str], list[BenchmarkRow]]:
    grouped: dict[tuple[str, str], list[BenchmarkRow]] = {}
    for row in rows:
        grouped.setdefault((row.design, row.method), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row.seed)
    return grouped


def _seed_set(rows: Sequence[BenchmarkRow]) -> set[int]:
    return {row.seed for row in rows}


def _validate_campaign(
    main: Sequence[BenchmarkRow],
    null: Sequence[BenchmarkRow],
    timing: Sequence[BenchmarkRow],
    path_timing: Sequence[PathTimingRow],
) -> tuple[dict[tuple[str, str], list[BenchmarkRow]], ...]:
    main_groups = _group(main)
    null_groups = _group(null)
    timing_groups = _group(timing)

    methods = {row.method for row in main}
    unknown = methods.difference(REFERENCE_METHODS, CANDIDATE_METHODS)
    if unknown:
        raise GateSummaryError(f"main input contains unsupported method(s): {sorted(unknown)!r}")
    candidates = [method for method in CANDIDATE_METHODS if method in methods]
    if not candidates:
        raise GateSummaryError("main input contains no Auto-K candidate methods")

    main_seed_sets: set[frozenset[int]] = set()
    required_methods = (*REFERENCE_METHODS, *candidates)
    for design in REQUIRED_MAIN_DESIGNS:
        for method in required_methods:
            rows = main_groups.get((design, method))
            if rows is None:
                raise GateSummaryError(f"main input is missing {design}/{method}")
            if len(rows) < 2:
                raise GateSummaryError(f"main input needs at least two seeds for {design}/{method}")
            main_seed_sets.add(frozenset(_seed_set(rows)))
    if len(main_seed_sets) != 1:
        raise GateSummaryError("main input must use one identical seed set for every gate design/method")

    for design in REQUIRED_MAIN_DESIGNS:
        seed_to_oracle: dict[int, int] = {}
        for method in required_methods:
            for row in main_groups[(design, method)]:
                previous = seed_to_oracle.setdefault(row.seed, row.k_oracle)
                if previous != row.k_oracle:
                    raise GateSummaryError(
                        f"main input has inconsistent k_oracle for {design}/seed={row.seed}"
                    )

    null_methods = {row.method for row in null}
    unexpected_null = null_methods.difference(candidates)
    if unexpected_null:
        raise GateSummaryError(
            f"null input contains method(s) absent from main: {sorted(unexpected_null)!r}"
        )
    needed_null = set(candidates).intersection(DEEP_NULL_METHODS)
    missing_null = needed_null.difference(null_methods)
    if missing_null:
        raise GateSummaryError(
            f"null input is missing deep-calibration method(s): {sorted(missing_null)!r}"
        )
    if {row.design for row in null} != {"D5"}:
        raise GateSummaryError("null input must contain D5 rows only")
    null_seed_sets = {
        frozenset(_seed_set(rows)) for (design, _method), rows in null_groups.items() if design == "D5"
    }
    if not null_seed_sets or len(null_seed_sets) != 1:
        raise GateSummaryError("null input must use one identical non-empty seed set for every method")

    if {row.design for row in timing} != {"D9"}:
        raise GateSummaryError("timing input must contain D9 rows only")
    timing_methods = {row.method for row in timing}
    unknown_timing = timing_methods.difference(REFERENCE_METHODS, CANDIDATE_METHODS)
    if unknown_timing:
        raise GateSummaryError(
            f"timing input contains unsupported method(s): {sorted(unknown_timing)!r}"
        )
    eval_rows = timing_groups.get(("D9", "evaluate/time_holdout/best"))
    if eval_rows is None:
        raise GateSummaryError("timing input is missing D9/evaluate/time_holdout/best")
    timing_seeds = _seed_set(eval_rows)
    missing_timing_baselines = [
        method for method in REFERENCE_METHODS if ("D9", method) not in timing_groups
    ]
    if missing_timing_baselines:
        raise GateSummaryError(
            f"timing input is missing baseline method(s): {missing_timing_baselines!r}"
        )
    if any(_seed_set(rows) != timing_seeds for rows in timing_groups.values()):
        raise GateSummaryError("timing input must use one identical seed set for every method")
    if {row.seed for row in path_timing} != timing_seeds:
        raise GateSummaryError("fixed-k path timing seeds must exactly match the D9 timing seeds")

    return main_groups, null_groups, timing_groups


def _finite_regrets(rows: Sequence[BenchmarkRow], *, label: str) -> list[float]:
    values = [row.regret_frac for row in rows if not math.isnan(row.regret_frac)]
    if len(values) != len(rows):
        raise GateSummaryError(f"{label} contains missing/NaN regret_frac values")
    return values


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def _median_abs_k_error(rows: Sequence[BenchmarkRow]) -> float:
    return float(statistics.median(abs(row.k_hat - row.k_oracle) for row in rows))


def _sample_std_k(rows: Sequence[BenchmarkRow]) -> float:
    return float(statistics.stdev(row.k_hat for row in rows))


def _program_oracle(rows: Sequence[BenchmarkRow], convention: str) -> float:
    values = [row.k_oracle for row in rows]
    if convention == "mean":
        return _mean(values)
    if convention == "median":
        return float(statistics.median(values))
    raise GateSummaryError(
        "oracle aggregation convention is required and must be exactly 'mean' or 'median'"
    )


def _g1(method: str, groups: dict[tuple[str, str], list[BenchmarkRow]]) -> bool:
    for design in G1_DESIGNS:
        candidate = groups[(design, method)]
        best_median_error = min(
            _median_abs_k_error(groups[(design, baseline)]) for baseline in COMPARISON_BASELINES
        )
        best_mean_regret = min(
            _mean(_finite_regrets(groups[(design, baseline)], label=f"{design}/{baseline}"))
            for baseline in COMPARISON_BASELINES
        )
        if _median_abs_k_error(candidate) > best_median_error:
            return False
        candidate_regret = _mean(_finite_regrets(candidate, label=f"{design}/{method}"))
        if candidate_regret > best_mean_regret + 0.01:
            return False
    return True


def _g2(
    method: str,
    main_groups: dict[tuple[str, str], list[BenchmarkRow]],
    null_groups: dict[tuple[str, str], list[BenchmarkRow]],
) -> bool:
    deep_rows = null_groups.get(("D5", method))
    if deep_rows is None:
        return max(row.k_hat for row in main_groups[("D5", method)]) <= 3

    if method not in CALIBRATION_LEVELS:
        return max(row.k_hat for row in deep_rows) <= 3
    level = CALIBRATION_LEVELS[method]
    probability = _mean(row.k_hat > 3 for row in deep_rows)
    standard_error = math.sqrt(level * (1.0 - level) / len(deep_rows))
    return probability <= level + 2.0 * standard_error


def _g3(method: str, groups: dict[tuple[str, str], list[BenchmarkRow]]) -> bool:
    candidate = _mean(_finite_regrets(groups[("D4", method)], label=f"D4/{method}"))
    baseline = _mean(
        _finite_regrets(
            groups[("D4", "evaluate/one_se")],
            label="D4/evaluate/one_se",
        )
    )
    return candidate <= baseline + 0.02


def _g4(method: str, groups: dict[tuple[str, str], list[BenchmarkRow]]) -> bool:
    d2 = _mean(_finite_regrets(groups[("D2", method)], label=f"D2/{method}"))
    d8 = _mean(_finite_regrets(groups[("D8", method)], label=f"D8/{method}"))
    return abs(d8 - d2) <= 0.02


def _g5(
    method: str,
    timing_groups: dict[tuple[str, str], list[BenchmarkRow]],
    path_baseline_s: float,
) -> tuple[bool | None, float | None]:
    method_rows = timing_groups.get(("D9", method))
    if method in PATH_ONLY_METHODS:
        if method_rows is None:
            return False, None
        ratio = _mean(row.runtime_s for row in method_rows) / path_baseline_s
        return ratio <= 1.5, ratio
    if method in FOLD_METHODS:
        if method_rows is None:
            return False, None
        evaluate_rows = timing_groups[("D9", "evaluate/time_holdout/best")]
        evaluate_runtime = _mean(row.runtime_s for row in evaluate_rows)
        if evaluate_runtime <= 0:
            raise GateSummaryError("D9 evaluate/time_holdout/best mean runtime must be positive")
        ratio = _mean(row.runtime_s for row in method_rows) / evaluate_runtime
        return ratio <= 0.5, ratio
    return None, None


def _g6(method: str, groups: dict[tuple[str, str], list[BenchmarkRow]]) -> bool:
    for design in G6_DESIGNS:
        best_baseline_std = min(
            _sample_std_k(groups[(design, baseline)]) for baseline in COMPARISON_BASELINES
        )
        if _sample_std_k(groups[(design, method)]) > 1.5 * best_baseline_std:
            return False
    return True


def summarize_gate_rows(
    main_path: Path,
    null_path: Path,
    timing_path: Path,
    *,
    path_timing_path: Path | None,
    oracle_aggregation: str | None,
) -> list[dict[str, object]]:
    """Compute canonical G1-G6 rows after validating all campaign inputs."""
    if oracle_aggregation not in {"mean", "median"}:
        raise GateSummaryError(
            "oracle aggregation convention is required and must be exactly 'mean' or 'median'"
        )

    main = _read_benchmark_csv(main_path)
    null = _read_benchmark_csv(null_path)
    timing = _read_benchmark_csv(timing_path)
    path_timing = _read_path_timing_csv(path_timing_path)
    main_groups, null_groups, timing_groups = _validate_campaign(
        main,
        null,
        timing,
        path_timing,
    )
    path_baseline_s = _mean(row.runtime_s for row in path_timing)
    candidates = [method for method in CANDIDATE_METHODS if ("D1", method) in main_groups]

    output: list[dict[str, object]] = []
    for method in candidates:
        g2 = _g2(method, main_groups, null_groups)
        g5, g5_ratio = _g5(method, timing_groups, path_baseline_s)
        program_rows = [
            row for design in PROGRAM_DESIGNS for row in main_groups[(design, method)]
        ]
        program_mean_regret = _mean(
            _finite_regrets(program_rows, label=f"program/{method}")
        )
        design_stability = []
        for design in PROGRAM_DESIGNS:
            rows = main_groups[(design, method)]
            oracle = _program_oracle(rows, oracle_aggregation)
            if oracle <= 0:
                raise GateSummaryError(
                    f"{design}/{method}: aggregated k_oracle must be positive for the program ratio"
                )
            design_stability.append(_sample_std_k(rows) / oracle)
        program_std = _mean(design_stability)
        d5_rows = main_groups[("D5", method)]
        d5_p_gt3 = _mean(row.k_hat > 3 for row in d5_rows)
        d5_max_k = max(row.k_hat for row in d5_rows)
        output.append(
            {
                "method": method,
                "G1_accuracy": _g1(method, main_groups),
                "G2_or_D5_null": g2,
                "G3_dense_weak": _g3(method, main_groups),
                "G4_structure": _g4(method, main_groups),
                "G5_runtime": g5,
                "G5_runtime_ratio": g5_ratio,
                "G6_stability": _g6(method, main_groups),
                "program_mean_regret_D1_D3_D7": program_mean_regret,
                "program_std_k_over_oracle": program_std,
                "D5_main_p_gt3": d5_p_gt3,
                "D5_main_max_k": d5_max_k,
                "program_success": program_mean_regret <= 0.02 and program_std <= 0.35 and g2,
            }
        )
    return output


def render_gate_csv(rows: Sequence[dict[str, object]]) -> bytes:
    """Render rows with a fixed header, ordering, newline, and scalar spelling."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=GATE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(GATE_COLUMNS):
            raise GateSummaryError("gate row keys do not match the output schema")
        writer.writerow({key: "" if row[key] is None else str(row[key]) for key in GATE_COLUMNS})
    return buffer.getvalue().encode("utf-8")


def regenerate_gate_csv(
    main_path: Path,
    null_path: Path,
    timing_path: Path,
    output_path: Path,
    *,
    path_timing_path: Path | None,
    oracle_aggregation: str | None,
) -> None:
    rows = summarize_gate_rows(
        main_path,
        null_path,
        timing_path,
        path_timing_path=path_timing_path,
        oracle_aggregation=oracle_aggregation,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_gate_csv(rows))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True, help="D1-D8 raw campaign CSV")
    parser.add_argument("--null", type=Path, required=True, help="D5 deep-null raw CSV")
    parser.add_argument("--timing", type=Path, required=True, help="D9 method-timing raw CSV")
    parser.add_argument(
        "--fixed-k-path-timing",
        type=Path,
        required=True,
        help="raw D9 fixed_k_select_cached path-build timing CSV",
    )
    parser.add_argument(
        "--oracle-aggregation",
        choices=("mean", "median"),
        required=True,
        help="denominator convention for program std(k_hat)/k_oracle",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        regenerate_gate_csv(
            args.main,
            args.null,
            args.timing,
            args.output,
            path_timing_path=args.fixed_k_path_timing,
            oracle_aggregation=args.oracle_aggregation,
        )
    except GateSummaryError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
