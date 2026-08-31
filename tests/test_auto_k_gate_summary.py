import csv
from pathlib import Path

import pytest

from benchmarks.summarize_auto_k_gates import (
    PATH_TIMING_COLUMNS,
    RAW_COLUMNS,
    GateSummaryError,
    regenerate_gate_csv,
)


def _raw_row(design, seed, method, *, k_hat, k_oracle, regret, runtime=0.01):
    return {
        "design": design,
        "seed": seed,
        "method": method,
        "k_hat": k_hat,
        "k_oracle": k_oracle,
        "k_star": k_oracle,
        "rmse_hat": 1.0,
        "rmse_oracle": 1.0,
        "regret_frac": regret,
        "support_precision": 1.0,
        "support_recall": 1.0,
        "support_f1": 1.0,
        "k_dispersion_group": f"{design}:{method}",
        "saturated_min": False,
        "saturated_max": False,
        "runtime_s": runtime,
        "notes": "fixture",
    }


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def gate_campaign(tmp_path):
    main_path = tmp_path / "main.csv"
    null_path = tmp_path / "null.csv"
    timing_path = tmp_path / "timing.csv"
    path_timing_path = tmp_path / "path_timing.csv"

    oracle_by_design = {
        "D1": 10,
        "D2": 10,
        "D3": 20,
        "D4": 40,
        "D5": 0,
        "D6": 10,
        "D7": 8,
        "D8": 10,
    }
    baselines = (
        "elbow",
        "penalized/bic",
        "evaluate/time_holdout/best",
        "evaluate/one_se",
        "fixed_k=50",
        "oracle",
    )
    main_rows = []
    for design, oracle in oracle_by_design.items():
        for seed in (0, 1):
            for baseline in baselines:
                if baseline == "fixed_k=50":
                    k_hat = 50
                    regret = 0.02
                elif baseline == "oracle":
                    k_hat = oracle
                    regret = 0.0
                else:
                    k_hat = oracle + 1
                    regret = 0.005
                main_rows.append(
                    _raw_row(
                        design,
                        seed,
                        baseline,
                        k_hat=k_hat,
                        k_oracle=oracle,
                        regret=regret,
                    )
                )
            main_rows.append(
                _raw_row(
                    design,
                    seed,
                    "penalized/ebic",
                    k_hat=seed if design == "D5" else oracle,
                    k_oracle=oracle,
                    regret=0.0 if design in {"D1", "D2", "D3", "D7"} else 0.01,
                )
            )
    _write_csv(main_path, RAW_COLUMNS, main_rows)

    null_rows = [
        _raw_row("D5", seed, "penalized/ebic", k_hat=seed, k_oracle=0, regret=0.0)
        for seed in (0, 1)
    ]
    _write_csv(null_path, RAW_COLUMNS, null_rows)

    timing_rows = []
    for seed in (0, 1):
        for baseline in baselines:
            timing_rows.append(
                _raw_row(
                    "D9",
                    seed,
                    baseline,
                    k_hat=20,
                    k_oracle=20,
                    regret=0.0,
                    runtime=0.4 if baseline == "evaluate/time_holdout/best" else 0.01,
                )
            )
        timing_rows.append(
            _raw_row(
                "D9",
                seed,
                "penalized/ebic",
                k_hat=20,
                k_oracle=20,
                regret=0.0,
                runtime=0.1,
            )
        )
    _write_csv(timing_path, RAW_COLUMNS, timing_rows)
    _write_csv(
        path_timing_path,
        PATH_TIMING_COLUMNS,
        [
            {
                "design": "D9",
                "seed": seed,
                "benchmark": "fixed_k_select_cached",
                "runtime_s": 0.2,
            }
            for seed in (0, 1)
        ],
    )
    return main_path, null_path, timing_path, path_timing_path


def test_gate_summary_is_byte_identical(gate_campaign, tmp_path):
    main_path, null_path, timing_path, path_timing_path = gate_campaign
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    reversed_main = tmp_path / "reversed_main.csv"
    with main_path.open(newline="", encoding="utf-8") as handle:
        main_rows = list(csv.DictReader(handle))
    _write_csv(reversed_main, RAW_COLUMNS, reversed(main_rows))
    kwargs = {
        "path_timing_path": path_timing_path,
        "oracle_aggregation": "mean",
    }

    regenerate_gate_csv(main_path, null_path, timing_path, first, **kwargs)
    regenerate_gate_csv(reversed_main, null_path, timing_path, second, **kwargs)

    expected = (
        b"method,G1_accuracy,G2_or_D5_null,G3_dense_weak,G4_structure,G5_runtime,"
        b"G5_runtime_ratio,G6_stability,program_mean_regret_D1_D3_D7,"
        b"program_std_k_over_oracle,D5_main_p_gt3,D5_main_max_k,program_success\n"
        b"penalized/ebic,True,True,True,True,True,0.5,True,0.0,0.0,0.0,1,True\n"
    )
    assert first.read_bytes() == expected
    assert second.read_bytes() == expected


def test_gate_summary_rejects_missing_provenance(gate_campaign, tmp_path):
    main_path, null_path, timing_path, _path_timing_path = gate_campaign

    with pytest.raises(GateSummaryError, match="fixed-k path timing provenance is required"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            tmp_path / "out.csv",
            path_timing_path=None,
            oracle_aggregation="mean",
        )

    with pytest.raises(GateSummaryError, match="oracle aggregation convention is required"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            tmp_path / "out.csv",
            path_timing_path=_path_timing_path,
            oracle_aggregation=None,
        )
    assert not (tmp_path / "out.csv").exists()


def test_gate_summary_rejects_raw_schema_drift(gate_campaign, tmp_path):
    _main_path, null_path, timing_path, path_timing_path = gate_campaign
    malformed_main = tmp_path / "malformed_main.csv"
    malformed_main.write_text("design,seed,method\nD1,0,penalized/ebic\n", encoding="utf-8")

    with pytest.raises(GateSummaryError, match="CSV schema mismatch"):
        regenerate_gate_csv(
            malformed_main,
            null_path,
            timing_path,
            tmp_path / "out.csv",
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )
