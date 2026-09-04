"""Contracts for the provenance-bound runtime/scaling reference."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from benchmarks import bench_runtime_scaling as runtime


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks" / "results" / "runtime_scaling_2026-09-03.csv"
PROVENANCE = RESULT.with_suffix(".provenance.json")
DOC = ROOT / "docs" / "runtime-scaling.md"
TABLE_START = "<!-- runtime-scaling-table:start -->\n"
TABLE_END = "\n<!-- runtime-scaling-table:end -->"


def test_percentile_uses_linear_interpolation() -> None:
    samples = [1.0, 2.0, 4.0, 8.0]
    assert runtime._percentile(samples, 0.0) == 1.0
    assert runtime._percentile(samples, 0.5) == 3.0
    assert runtime._percentile(samples, 0.95) == pytest.approx(7.4)
    assert runtime._percentile(samples, 1.0) == 8.0


def test_committed_runtime_evidence_and_documented_table_are_bound() -> None:
    with RESULT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == runtime.CSV_COLUMNS
        rows = list(reader)

    assert len(rows) == len(runtime.FULL_WORKLOADS) * len(runtime.METHODS)
    assert {row["workload"] for row in rows} == {
        name for name, _, _ in runtime.FULL_WORKLOADS
    }
    assert {row["method"] for row in rows} == set(runtime.METHODS)
    for row in rows:
        assert int(row["warmup_runs"]) == 1
        assert int(row["timing_repeats"]) == 7
        assert 0 < float(row["p50_s"]) <= float(row["p95_s"]) <= float(row["p99_s"])
        assert float(row["peak_rss_mb"]) > 0
        assert float(row["million_cells_per_s"]) > 0
        assert len(row["selection_sha256"]) == 64
        assert len(row["data_sha256"]) == 64

    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert provenance["schema"] == runtime.PROVENANCE_SCHEMA
    assert provenance["artifact"]["row_count"] == len(rows)
    assert provenance["artifact"]["columns"] == list(runtime.CSV_COLUMNS)
    assert provenance["artifact"]["sha256"] == hashlib.sha256(
        RESULT.read_bytes()
    ).hexdigest()
    assert provenance["configuration"]["full"] is True
    assert provenance["configuration"]["methods"] == list(runtime.METHODS)
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["git"]["commit"])
    assert isinstance(provenance["git"]["dirty"], bool)
    assert len(provenance["measurements"]) == len(rows)
    assert all(
        len(measurement["runtime_samples_s"]) == 7
        for measurement in provenance["measurements"]
    )
    assert all(
        pool["num_threads"] == 1
        for measurement in provenance["measurements"]
        for pool in measurement["threadpools_during_timing"]
    )

    expected_rows = [
        {column: str(measurement[column]) for column in runtime.CSV_COLUMNS}
        for measurement in provenance["measurements"]
    ]
    assert rows == expected_rows

    for relative, expected_hash in provenance["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash

    doc = DOC.read_text(encoding="utf-8")
    documented_table = doc.split(TABLE_START, 1)[1].split(TABLE_END, 1)[0]
    assert documented_table == runtime.render_markdown_table(rows)


def test_worker_executes_a_seeded_selector_twice_with_one_fingerprint() -> None:
    measurement = runtime._measure_worker(
        method="mrmr_classic",
        workload="test",
        n=120,
        p=12,
        k=3,
        seed=7,
        warmup_runs=0,
        timing_repeats=2,
    )

    assert measurement["selected_count"] == 3
    assert len(measurement["runtime_samples_s"]) == 2
    assert len(measurement["selection_sha256"]) == 64
