import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.summarize_auto_k_gates import (
    compare_gate_csv_files,
    PATH_TIMING_COLUMNS,
    PATH_TIMING_PROVENANCE_SCHEMA,
    RAW_COLUMNS,
    GateSummaryError,
    regenerate_gate_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(commit, relative):
    return subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _write_path_timing_provenance(path, *, full=True, dirty=False, source_hashes=None):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_hashes is None:
        source_relative = Path("benchmarks/summarize_auto_k_gates.py")
        source_hashes = {
            str(source_relative): hashlib.sha256(
                _git_blob(commit, source_relative)
            ).hexdigest()
        }
    provenance = {
        "schema": PATH_TIMING_PROVENANCE_SCHEMA,
        "artifact": {
            "sha256": _sha256(path),
            "columns": list(PATH_TIMING_COLUMNS),
            "row_count": len(rows),
        },
        "configuration": {
            "design": "D9",
            "full": full,
            "seeds": [int(row["seed"]) for row in rows],
            "benchmark": "fixed_k_select_cached",
        },
        "git": {"commit": commit, "dirty": dirty},
        "source_sha256": source_hashes,
    }
    path.with_suffix(".provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )


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
    _write_path_timing_provenance(path_timing_path)
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
        b"penalized/ebic,True,True,True,True,True,0.5,True,0,0,0,1,True\n"
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


def test_gate_summary_rejects_missing_or_untrusted_path_timing_sidecar(
    gate_campaign,
    tmp_path,
):
    main_path, null_path, timing_path, path_timing_path = gate_campaign
    output_path = tmp_path / "out.csv"
    sidecar_path = path_timing_path.with_suffix(".provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    original_csv = path_timing_path.read_bytes()

    sidecar_path.unlink()
    with pytest.raises(GateSummaryError, match="cannot read fixed-k path timing provenance"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            output_path,
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )

    sidecar["configuration"]["full"] = False
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(GateSummaryError, match="quick runs cannot feed release gates"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            output_path,
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )

    sidecar["configuration"]["full"] = True
    sidecar["git"]["dirty"] = True
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(GateSummaryError, match="clean git provenance is required"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            output_path,
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )

    sidecar["git"]["dirty"] = False
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    path_timing_path.write_bytes(original_csv + b"\n")
    with pytest.raises(GateSummaryError, match="artifact checksum does not match"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            output_path,
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )

    path_timing_path.write_bytes(original_csv)
    sidecar["source_sha256"] = {"benchmarks/summarize_auto_k_gates.py": "0" * 64}
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(GateSummaryError, match="source checksum does not match"):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            output_path,
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )

    assert not output_path.exists()


def test_committed_dated_gate_is_bound_to_its_full_clean_provenance(tmp_path):
    results = REPO_ROOT / "benchmarks/results"
    sidecar = json.loads(
        (results / "auto_k_v2_d9_fixed_k_path_2026-08-31.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    provenance_commit = str(sidecar["git"]["commit"])
    have_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{provenance_commit}^{{commit}}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not have_commit:
        pytest.skip(
            "provenance commit "
            f"{provenance_commit[:12]} is not available in this checkout (shallow "
            "clone or archive); CI runs this verification on a full-history checkout"
        )
    output_path = tmp_path / "gates.csv"
    regenerate_gate_csv(
        results / "auto_k_v2_main.csv",
        results / "auto_k_v2_null.csv",
        results / "auto_k_v2_d9.csv",
        output_path,
        path_timing_path=(
            results / "auto_k_v2_d9_fixed_k_path_2026-08-31.csv"
        ),
        oracle_aggregation="mean",
    )
    expected = results / "auto_k_v2_gates_mean_oracle_2026-08-31.csv"
    # Floats are rendered with 12 significant digits, which absorbs last-ulp
    # platform differences; compare numerically (exact for non-float cells).
    assert compare_gate_csv_files(output_path, expected) == []


def test_gate_summary_rejects_unavailable_provenance_commit(
    gate_campaign,
    tmp_path,
):
    main_path, null_path, timing_path, path_timing_path = gate_campaign
    sidecar_path = path_timing_path.with_suffix(".provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["git"]["commit"] = "f" * 40
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(
        GateSummaryError,
        match="cannot verify recorded source at provenance commit",
    ):
        regenerate_gate_csv(
            main_path,
            null_path,
            timing_path,
            tmp_path / "out.csv",
            path_timing_path=path_timing_path,
            oracle_aggregation="mean",
        )


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
