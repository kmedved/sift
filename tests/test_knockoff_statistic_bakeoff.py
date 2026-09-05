"""Behavior tests for the F8c public-API knockoff statistic bakeoff."""

from __future__ import annotations

import inspect
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import bench_knockoff_statistic_bakeoff as bakeoff
from sift import select_fdr


def test_paired_designs_share_data_across_statistics():
    first = None
    for statistic in bakeoff.STATISTICS:
        x, y, truth, spec = bakeoff.make_design("ar1", 7, n=40, p=12)
        if first is None:
            first = (x.copy(), y.copy(), set(truth), spec)
        else:
            np.testing.assert_allclose(x, first[0])
            np.testing.assert_allclose(y, first[1])
            assert truth == first[2]
            assert spec == first[3]
        assert statistic in bakeoff.STATISTICS
    assert len(first[2]) == spec["n_signal"]
    assert spec["n_signal"] * bakeoff.SELECT_FDR_FIXED["q"] < spec["n_signal"]


def test_public_defaults_are_the_documented_select_fdr_settings():
    defaults = inspect.signature(select_fdr).parameters
    assert defaults["q"].default == bakeoff.SELECT_FDR_FIXED["q"]
    assert defaults["offset"].default == bakeoff.SELECT_FDR_FIXED["offset"]
    assert defaults["s_method"].default == bakeoff.SELECT_FDR_FIXED["s_method"]
    assert defaults["n_draws"].default == bakeoff.SELECT_FDR_FIXED["n_draws"]
    assert defaults["statistic"].default == "relevance"
    assert defaults["statistic_options"].default is None


def test_evaluate_one_calls_public_select_fdr(monkeypatch):
    calls = []

    def fake_select(X, y, **kwargs):
        calls.append(kwargs)
        n_signal = 6
        return SimpleNamespace(
            selected_indices=list(range(n_signal)),
            selector_metadata={"fdr_control": "approximate_plugin"},
        )

    row = bakeoff.evaluate_one(
        design="independent",
        statistic="ridge",
        seed=3,
        n=40,
        p=12,
        warmup_runs=0,
        timing_repeats=1,
        select_fn=fake_select,
    )
    assert calls
    kwargs = calls[0]
    assert kwargs["q"] == 0.1
    assert kwargs["offset"] == 1
    assert kwargs["s_method"] == "equi"
    assert kwargs["n_draws"] == 1
    assert kwargs["statistic"] == "ridge"
    assert "statistic_options" not in kwargs
    assert kwargs["random_state"] == 3
    assert row["status"] == "ok"
    assert row["n_discoveries"] == 6
    assert row["fdp"] == 0.0
    assert row["power"] == 1.0
    assert row["selection_sha256"]


def test_failed_run_is_recorded_not_empty_selection(monkeypatch):
    def boom(X, y, **kwargs):
        raise RuntimeError("forced bakeoff failure")

    row = bakeoff.evaluate_one(
        design="independent",
        statistic="relevance",
        seed=0,
        n=30,
        p=10,
        warmup_runs=0,
        timing_repeats=1,
        select_fn=boom,
    )
    assert row["status"] == "failed"
    assert "forced bakeoff failure" in row["error"]
    assert row["n_discoveries"] is None
    assert row["fdp"] is None
    assert row["power"] is None
    assert row["selection_sha256"] is None
    assert row["selected_indices"] is None
    assert row["runtime_samples_s"] == []
    assert row["data_sha256"]


def test_warnings_are_retained(monkeypatch):
    def noisy(X, y, **kwargs):
        warnings.warn("visible bakeoff warning", UserWarning)
        return SimpleNamespace(
            selected_indices=[0, 1],
            selector_metadata={"fdr_control": "approximate_plugin"},
        )

    row = bakeoff.evaluate_one(
        design="independent",
        statistic="relevance",
        seed=1,
        n=30,
        p=10,
        warmup_runs=0,
        timing_repeats=1,
        select_fn=noisy,
    )
    assert row["status"] == "ok"
    assert row["warning_count"] >= 1
    assert "visible bakeoff warning" in row["warning_messages"]


def test_smoke_writes_csv_and_provenance(tmp_path):
    records = bakeoff.run_study(
        full=False,
        seeds=[0],
        n=48,
        p=12,
        warmup_runs=0,
        timing_repeats=1,
        designs=("independent",),
        statistics=("relevance", "ridge"),
    )
    assert len(records) == 2
    assert {row["statistic"] for row in records} == {"relevance", "ridge"}
    assert records[0]["data_sha256"] == records[1]["data_sha256"]
    csv_path = tmp_path / "bakeoff.csv"
    bakeoff.write_csv(csv_path, records)
    provenance_path = tmp_path / "bakeoff.provenance.json"
    environment = bakeoff.capture_environment()
    payload = bakeoff.write_provenance(
        provenance_path,
        csv_path=csv_path,
        records=records,
        study="smoke",
        n=48,
        p=12,
        seeds=[0],
        warmup_runs=0,
        timing_repeats=1,
        environment=environment,
    )
    assert payload["schema"] == bakeoff.PROVENANCE_SCHEMA
    assert payload["artifact"]["sha256"] == bakeoff._sha256_file(csv_path)
    assert payload["summary"]["paired_ridge_minus_relevance"]
    loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert loaded["caveats"]
    assert loaded["records"][0]["selected_indices"] == records[0]["selected_indices"]
    assert loaded["records"][0]["runtime_samples_s"] == records[0]["runtime_samples_s"]
    assert loaded["records"][0]["effective_num_threads"]
    assert bakeoff.RUNNER_RELATIVE in loaded["git"]["source_sha256"]
    assert bakeoff.HELPER_RELATIVE in loaded["git"]["source_sha256"]
    assert "sift/selection/knockoff_filter.py" in loaded["git"]["source_sha256"]
    assert "numba" in loaded["environment"]["packages"]


def test_summarize_counts_failures_separately():
    ok = {
        "design": "independent",
        "statistic": "relevance",
        "seed": 0,
        "status": "ok",
        "fdp": 0.1,
        "power": 0.5,
        "n_discoveries": 4,
        "runtime_s": 0.2,
        "warning_count": 0,
    }
    failed = dict(ok, status="failed", fdp=None, power=None, n_discoveries=None)
    ridge_ok = dict(ok, statistic="ridge", power=0.6, fdp=0.05, runtime_s=0.3)
    summary = bakeoff.summarize([ok, failed, ridge_ok])
    relevance = next(
        cell
        for cell in summary["cells"]
        if cell["design"] == "independent" and cell["statistic"] == "relevance"
    )
    assert relevance["n_ok"] == 1
    assert relevance["n_failed"] == 1
    paired = next(
        row
        for row in summary["paired_ridge_minus_relevance"]
        if row["design"] == "independent"
    )
    assert paired["n_paired"] == 1
    assert paired["power_mean_diff"] == pytest.approx(0.1)


def test_warning_before_failure_is_kept():
    def noisy_fail(X, y, **kwargs):
        warnings.warn("warning before failure", UserWarning)
        raise RuntimeError("forced bakeoff failure")

    row = bakeoff.evaluate_one(
        design="independent",
        statistic="relevance",
        seed=0,
        n=30,
        p=10,
        warmup_runs=0,
        timing_repeats=1,
        select_fn=noisy_fail,
    )
    assert row["status"] == "failed"
    assert row["warning_count"] == 1
    assert row["warnings"][0]["phase"] == "timing"
    assert row["warnings"][0]["repeat"] == 0
    assert "warning before failure" in row["warnings"][0]["message"]
    assert row["runtime_samples_s"] == []
    json.dumps(bakeoff._json_ready(bakeoff._record_for_provenance(row)), allow_nan=False)


def test_warmup_warning_survives_timing_failure():
    calls = {"n": 0}

    def flaky(X, y, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            warnings.warn("warmup only", UserWarning)
            return SimpleNamespace(
                selected_indices=[0, 1],
                selector_metadata={"fdr_control": "approximate_plugin"},
            )
        warnings.warn("timing fail", UserWarning)
        raise RuntimeError("timing failed")

    row = bakeoff.evaluate_one(
        design="independent",
        statistic="relevance",
        seed=1,
        n=30,
        p=10,
        warmup_runs=1,
        timing_repeats=1,
        select_fn=flaky,
    )
    assert row["status"] == "failed"
    assert [item["phase"] for item in row["warnings"]] == ["warmup", "timing"]
    assert row["selected_indices"] is None
    assert row["runtime_samples_s"] == []


def test_main_captures_source_before_run_and_write(monkeypatch, tmp_path):
    order = []
    environment = {
        "captured_at_utc": "t0",
        "commit": "abc",
        "dirty": True,
        "status_porcelain": [" M unrelated"],
        "python": "x",
        "platform": "x",
        "executable": "x",
        "packages": {"numba": "0.0"},
        "thread_env": {},
        "select_fdr_defaults": {},
        "select_fdr_fixed": dict(bakeoff.SELECT_FDR_FIXED),
        "source_sha256": {
            bakeoff.RUNNER_RELATIVE: "aa",
            bakeoff.HELPER_RELATIVE: "bb",
        },
    }

    def fake_capture():
        order.append("capture")
        return dict(environment)

    def fake_run(**kwargs):
        order.append("run")
        return [
            {
                "study": "knockoff_statistic_bakeoff",
                "design": "independent",
                "statistic": "relevance",
                "seed": 0,
                "n": 10,
                "p": 8,
                "n_signal": 4,
                "q": 0.1,
                "offset": 1,
                "s_method": "equi",
                "n_draws": 1,
                "status": "ok",
                "n_discoveries": 1,
                "fdp": 0.0,
                "power": 0.25,
                "runtime_s": 0.01,
                "fdr_control": "approximate_plugin",
                "warning_count": 0,
                "warning_messages": "",
                "error": "",
                "data_sha256": "d" * 64,
                "selection_sha256": "s" * 64,
                "selected_indices": [0],
                "runtime_samples_s": [0.01],
                "warnings": [],
                "effective_num_threads": [1],
                "threadpools_during_timing": [{"num_threads": 1}],
            }
        ]

    def fake_verify(start):
        order.append("verify")
        assert start["source_sha256"] == environment["source_sha256"]
        assert start["dirty"] is True
        return {"commit": start["commit"], "source_sha256": start["source_sha256"]}

    real_write = bakeoff.write_csv

    def fake_write(path, records):
        order.append("write_csv")
        return real_write(path, records)

    monkeypatch.setattr(bakeoff, "capture_environment", fake_capture)
    monkeypatch.setattr(bakeoff, "run_study", fake_run)
    monkeypatch.setattr(bakeoff, "verify_source_unchanged", fake_verify)
    monkeypatch.setattr(bakeoff, "write_csv", fake_write)
    output = tmp_path / "order.csv"
    assert bakeoff.main(["--smoke", "--output", str(output)]) == 0
    assert order == ["capture", "run", "verify", "write_csv"]
    payload = json.loads(output.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert payload["git"]["dirty"] is True
    assert payload["git"]["source_sha256"] == environment["source_sha256"]
    assert payload["records"][0]["selected_indices"] == [0]
    assert payload["records"][0]["runtime_samples_s"] == [0.01]
    assert payload["artifact"]["sha256"] == bakeoff._sha256_file(output)
