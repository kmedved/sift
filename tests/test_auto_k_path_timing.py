import csv
import hashlib
import json

import numpy as np

from benchmarks import bench_auto_k_path_timing as timing
from benchmarks.summarize_auto_k_gates import _read_path_timing_csv


class _TinyD9:
    def make(self, seed, full):
        assert seed == 4
        assert full is False
        X = np.arange(30, dtype=float).reshape(5, 6)
        y = np.arange(5, dtype=float)
        return X, y, {"k_star": 2}


def test_path_timing_runner_measures_complete_select_cached_calls(monkeypatch):
    calls = []
    cache = object()

    monkeypatch.setitem(timing.DESIGNS, "D9", _TinyD9())

    def fake_build_cache(X, **kwargs):
        calls.append(("build_cache", X.shape, kwargs))
        return cache

    def fake_select_cached(actual_cache, y, k, **kwargs):
        calls.append(("select_cached", actual_cache, y.shape, k, kwargs))
        indices = list(range(k))
        return [f"x{i}" for i in indices], indices, np.arange(k, dtype=float)

    monkeypatch.setattr(timing, "build_cache", fake_build_cache)
    monkeypatch.setattr(timing, "select_cached", fake_select_cached)

    rows, measurements = timing.measure_d9_path_timing(
        full=False,
        seeds=[4],
        timing_repeats=3,
        warmup_runs=1,
        thread_limit=None,
    )

    assert list(rows[0]) == list(timing.CSV_COLUMNS)
    assert rows[0]["design"] == "D9"
    assert rows[0]["seed"] == 4
    assert rows[0]["benchmark"] == timing.BENCHMARK
    assert rows[0]["runtime_s"] > 0
    assert calls[0] == (
        "build_cache",
        (5, 6),
        {"subsample": None, "random_state": 4, "compute_Rxx": True},
    )
    select_calls = [call for call in calls if call[0] == "select_cached"]
    assert len(select_calls) == 4
    assert all(call[3] == 6 for call in select_calls)
    assert all(
        call[4]
        == {
            "method": "cefsplus",
            "top_m": 250,
            "return_indices": True,
            "return_objective": True,
        }
        for call in select_calls
    )
    assert measurements[0]["resolved_k"] == 6
    assert measurements[0]["selected_path_length"] == 6
    assert len(measurements[0]["runtime_samples_s"]) == 3


def test_path_timing_artifacts_bind_strict_csv_to_provenance(monkeypatch, tmp_path):
    csv_path = tmp_path / "d9_path.csv"
    provenance_path = tmp_path / "d9_path.provenance.json"
    rows = [
        {
            "design": "D9",
            "seed": seed,
            "benchmark": timing.BENCHMARK,
            "runtime_s": 0.25 + seed,
        }
        for seed in (0, 1)
    ]
    measurements = [{"seed": row["seed"]} for row in rows]

    def fake_git_output(*args):
        return "abc123" if args == ("rev-parse", "HEAD") else " M tracked.py"

    monkeypatch.setattr(timing, "_git_output", fake_git_output)
    monkeypatch.setattr(timing, "_package_versions", lambda: {"sift": "test"})
    monkeypatch.setattr(timing, "_source_hashes", lambda: {"runner.py": "source-hash"})
    monkeypatch.setattr(timing, "threadpool_info", lambda: [{"num_threads": 1}])

    timing.write_timing_artifacts(
        csv_path,
        provenance_path,
        rows=rows,
        measurements=measurements,
        full=True,
        timing_repeats=5,
        warmup_runs=1,
        thread_limit=1,
        command_argv=["python", "benchmarks/bench_auto_k_path_timing.py", "--full"],
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(timing.CSV_COLUMNS)
        assert [int(row["seed"]) for row in reader] == [0, 1]

    # The strict runner output is accepted directly by the gate summarizer.
    parsed = _read_path_timing_csv(csv_path)
    assert [(row.seed, row.runtime_s) for row in parsed] == [(0, 0.25), (1, 1.25)]

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert provenance["schema"] == timing.PROVENANCE_SCHEMA
    assert provenance["artifact"]["sha256"] == expected_hash
    assert provenance["artifact"]["columns"] == list(timing.CSV_COLUMNS)
    assert provenance["configuration"]["full"] is True
    assert provenance["configuration"]["seeds"] == [0, 1]
    assert provenance["configuration"]["timing_thread_limit"] == 1
    assert provenance["git"]["commit"] == "abc123"
    assert provenance["git"]["dirty"] is True
    assert provenance["git"]["status_porcelain"] == [" M tracked.py"]
    assert provenance["git"]["capture_scope"] == "before measurement and artifact creation"
    assert provenance["git"]["captured_at_utc"]
    assert provenance["source_sha256"] == {"runner.py": "source-hash"}
