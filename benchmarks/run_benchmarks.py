#!/usr/bin/env python
"""Run the SIFT benchmark suite and aggregate promotion JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bench_utils import SCHEMA_VERSION, write_json  # noqa: E402


SCRIPTS = [
    "bench_mrmr.py",
    "bench_jmi.py",
    "bench_permutation.py",
    "bench_filters.py",
    "bench_cefsplus.py",
    "bench_knockoffs.py",
    "bench_stability.py",
    "bench_catboost.py",
]


def _load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if payload.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"Unexpected benchmark schema in {path}: {payload.get('schema')!r}")
    return list(payload.get("records", []))


def _blocked_promotion_rows(records: list[dict]) -> list[dict]:
    """Return benchmark rows that should fail the promotion gate."""
    blocked = []
    for row in records:
        status = str(row.get("promotion_status", ""))
        kind = row.get("benchmark_kind", "promotion")
        if kind == "promotion" and status.startswith("blocked"):
            blocked.append(row)
    return blocked


def _script_failure_reason(proc: subprocess.CompletedProcess, out_path: Path) -> str | None:
    """Classify subprocess failures before row-level promotion gates run."""
    if proc.returncode != 0:
        return f"exit code {proc.returncode}"
    if not out_path.exists():
        return "missing output JSON"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args()

    mode_flag = "--full" if args.full else "--quick"
    records: list[dict] = []
    failures: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="sift-bench-") as tmp:
        tmp_path = Path(tmp)
        for script in SCRIPTS:
            out_path = tmp_path / f"{script}.json"
            cmd = [
                args.python,
                str(REPO_ROOT / "benchmarks" / script),
                mode_flag,
                "--output",
                str(out_path),
            ]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True)
            failure_reason = _script_failure_reason(proc, out_path)
            if failure_reason is not None:
                failures.append((script, failure_reason))
                if not args.continue_on_fail:
                    break
            if out_path.exists():
                try:
                    records.extend(_load_records(out_path))
                except Exception as exc:
                    failures.append((script, f"invalid output JSON: {exc}"))
                    if not args.continue_on_fail:
                        break

    write_json(args.output, records)

    if failures:
        for script, reason in failures:
            print(f"{script} failed: {reason}", file=sys.stderr)
        return 1

    blocked_rows = _blocked_promotion_rows(records)
    if blocked_rows:
        for row in blocked_rows:
            label = row.get("benchmark") or row.get("case") or row.get("method") or "benchmark"
            print(
                f"{label} blocked promotion: {row.get('promotion_status')}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
