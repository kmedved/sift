import subprocess
import sys
from pathlib import Path

from benchmarks import run_benchmarks
from benchmarks.run_benchmarks import _blocked_promotion_rows


def test_benchmark_runner_blocks_only_promotion_failures():
    records = [
        {"benchmark": "jmi", "benchmark_kind": "promotion", "promotion_status": "blocked: wall"},
        {"benchmark": "mrmr", "benchmark_kind": "parity", "promotion_status": "blocked: wall"},
        {"benchmark": "filters", "benchmark_kind": "informational", "promotion_status": "informational"},
    ]

    blocked = _blocked_promotion_rows(records)

    assert blocked == [records[0]]


def _write_script_records(cmd, records):
    output = Path(cmd[cmd.index("--output") + 1])
    run_benchmarks.write_json(output, records)
    return subprocess.CompletedProcess(cmd, 0)


def test_benchmark_runner_allows_blocked_parity_rows(tmp_path, monkeypatch):
    records = [
        {
            "benchmark": "mrmr",
            "benchmark_kind": "parity",
            "promotion_status": "blocked: parity",
        }
    ]
    monkeypatch.setattr(run_benchmarks, "SCRIPTS", ["fake.py"])
    monkeypatch.setattr(
        run_benchmarks.subprocess,
        "run",
        lambda cmd, cwd, text: _write_script_records(cmd, records),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks.py", "--quick", "--output", str(tmp_path / "out.json")],
    )

    assert run_benchmarks.main() == 0


def test_benchmark_runner_blocks_promotion_rows(tmp_path, monkeypatch):
    records = [
        {
            "benchmark": "jmi",
            "benchmark_kind": "promotion",
            "promotion_status": "blocked: wall",
        }
    ]
    monkeypatch.setattr(run_benchmarks, "SCRIPTS", ["fake.py"])
    monkeypatch.setattr(
        run_benchmarks.subprocess,
        "run",
        lambda cmd, cwd, text: _write_script_records(cmd, records),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks.py", "--quick", "--output", str(tmp_path / "out.json")],
    )

    assert run_benchmarks.main() == 1


def test_benchmark_runner_fails_when_script_writes_no_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run_benchmarks, "SCRIPTS", ["fake.py"])
    monkeypatch.setattr(
        run_benchmarks.subprocess,
        "run",
        lambda cmd, cwd, text: subprocess.CompletedProcess(cmd, 0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks.py", "--quick", "--output", str(tmp_path / "out.json")],
    )

    assert run_benchmarks.main() == 1


def test_benchmark_runner_fails_on_malformed_script_output(tmp_path, monkeypatch):
    def write_invalid_json(cmd, cwd, text):
        output = Path(cmd[cmd.index("--output") + 1])
        output.write_text("{not json", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(run_benchmarks, "SCRIPTS", ["fake.py"])
    monkeypatch.setattr(run_benchmarks.subprocess, "run", write_invalid_json)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_benchmarks.py", "--quick", "--output", str(out_path)],
    )

    assert run_benchmarks.main() == 1
    assert out_path.exists()
