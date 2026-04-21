#!/usr/bin/env python
"""Benchmark classic JMI hot-loop paths and selected-feature parity."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sift.estimators import joint_mi as jmi_est  # noqa: E402
from sift.selection.loops import jmi_select  # noqa: E402
from benchmarks.bench_utils import promotion_status, write_json  # noqa: E402


Estimator = Literal["r2", "binned"]
Aggregation = Literal["sum", "min"]
Case = tuple[Estimator, int, int, int, int | None]


def _cases(full: bool) -> list[Case]:
    quick_cases: list[Case] = [
        ("r2", 3_000, 300, 20, 250),
        ("binned", 1_500, 200, 15, 150),
    ]
    if not full:
        return quick_cases
    return quick_cases + [
        ("r2", 20_000, 5_000, 50, 500),
        ("r2", 20_000, 5_000, 50, None),
        ("binned", 10_000, 1_000, 50, 500),
    ]


def _make_data(
    n: int,
    p: int,
    *,
    seed: int,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float64)
    signal_count = min(10, p)
    coefs = np.linspace(2.0, 0.3, signal_count)
    y = X[:, :signal_count] @ coefs + rng.normal(scale=0.35, size=n)
    if weighted:
        w = rng.uniform(0.25, 2.5, size=n).astype(np.float64)
    else:
        w = np.ones(n, dtype=np.float64)

    y_centered = y - np.average(y, weights=w)
    relevance = np.abs(X.T @ (w * y_centered)) / float(w.sum())
    relevance += np.linspace(1e-8, 2e-8, p)
    return X, y.astype(np.float64), relevance.astype(np.float64), w


def _candidate_view(
    X: np.ndarray,
    relevance: np.ndarray,
    top_m: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.where(relevance > 0)[0]
    X_valid = X[:, valid_idx]
    rel_valid = relevance[valid_idx]

    if top_m is not None and top_m < len(valid_idx):
        top_local = np.argpartition(rel_valid, -top_m)[-top_m:]
        return X_valid[:, top_local], rel_valid[top_local], valid_idx[top_local]
    return X_valid, rel_valid, valid_idx


def _legacy_jmi_select(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    relevance: np.ndarray,
    *,
    estimator: Estimator,
    aggregation: Aggregation,
    top_m: int | None,
    sample_weight: np.ndarray,
) -> np.ndarray:
    X_cand, rel_cand, idx_map = _candidate_view(X, relevance, top_m)
    m = X_cand.shape[1]
    k = min(k, m)
    if k <= 0:
        return np.empty(0, dtype=np.int64)

    if estimator == "binned":
        X_binned = jmi_est.quantile_bin_matrix(X_cand, n_bins=10)
        y_binned = jmi_est._quantile_bin(y, 10)
        n_y_bins = 10
    else:
        X_binned = None
        y_binned = None
        n_y_bins = None

    scores = np.zeros(m, dtype=np.float64)
    if aggregation == "min":
        scores.fill(np.inf)

    is_selected = np.zeros(m, dtype=bool)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(np.argmax(rel_cand))
    is_selected[selected[0]] = True
    count = 1

    for t in range(1, k):
        last = int(selected[t - 1])
        cand_indices = np.where(~is_selected)[0]
        if cand_indices.size == 0:
            break
        cand_idx64 = cand_indices.astype(np.int64, copy=False)

        if estimator == "r2":
            mi_values = jmi_est.r2_joint_mi_indexed(
                X_cand,
                cand_idx64,
                X_cand[:, last],
                y,
                sample_weight,
            )
        else:
            s_binned = jmi_est._quantile_bin(X_cand[:, last], 10)
            mi_values = jmi_est.binned_joint_mi_indexed_prebinned(
                X_binned,
                cand_idx64,
                s_binned,
                y_binned,
                sample_weight,
                n_bins=10,
                n_y_bins=n_y_bins,
            )

        for i, idx in enumerate(cand_indices):
            if aggregation == "sum":
                scores[idx] += mi_values[i]
            else:
                scores[idx] = min(scores[idx], mi_values[i])

        best_score = -np.inf
        best_idx = -1
        for idx in cand_indices:
            score = scores[idx] if np.isfinite(scores[idx]) else rel_cand[idx]
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        selected[t] = best_idx
        is_selected[best_idx] = True
        count += 1

    return idx_map[selected[:count]]


def _time_best(repeat: int, fn) -> tuple[float, np.ndarray]:
    best_time = float("inf")
    best_result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        wall = time.perf_counter() - start
        if wall < best_time:
            best_time = wall
            best_result = result
    return best_time, best_result


def _warmup() -> None:
    X, y, relevance, w = _make_data(128, 16, seed=999, weighted=False)
    jmi_est.r2_joint_mi_indexed(X, np.array([1, 2, 3], dtype=np.int64), X[:, 0], y, w)
    jmi_select(X, y, 3, relevance, mi_estimator="r2", top_m=12, sample_weight=w)
    jmi_select(
        X,
        y,
        3,
        relevance,
        mi_estimator="binned",
        top_m=12,
        sample_weight=w,
    )


def _markdown_table(records: Iterable[dict]) -> str:
    rows = [
        "| estimator | n | p | k | top_m | weights | legacy seconds | current seconds | speedup | parity |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in records:
        row_display = dict(row)
        row_display["top_m"] = row["top_m"] if row["top_m"] is not None else "-"
        row_display["weights"] = "yes" if row["weighted"] else "no"
        row_display["parity"] = "yes" if row["selected_feature_parity"] else "NO"
        rows.append(
            "| {estimator} | {n} | {p} | {k} | {top_m} | {weights} | "
            "{legacy_seconds:.3f} | {current_seconds:.3f} | {speedup:.2f}x | {parity} |".format(
                **row_display,
            )
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run the fast local benchmark cases.")
    mode.add_argument("--full", action="store_true", help="Run larger benchmark cases.")
    parser.add_argument("--repeat", type=int, default=1, help="Run count per row; best time is reported.")
    parser.add_argument("--weighted", action="store_true", help="Use non-uniform sample weights.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    _warmup()
    records: list[dict] = []
    for case_idx, (estimator, n, p, k, top_m) in enumerate(_cases(bool(args.full))):
        X, y, relevance, w = _make_data(
            n,
            p,
            seed=20_000 + case_idx,
            weighted=args.weighted,
        )

        legacy_seconds, legacy_selected = _time_best(
            args.repeat,
            lambda: _legacy_jmi_select(
                X,
                y,
                k,
                relevance,
                estimator=estimator,
                aggregation="sum",
                top_m=top_m,
                sample_weight=w,
            ),
        )
        current_seconds, current_selected = _time_best(
            args.repeat,
            lambda: jmi_select(
                X,
                y,
                k,
                relevance,
                mi_estimator=estimator,
                aggregation="sum",
                top_m=top_m,
                sample_weight=w,
            ),
        )

        records.append(
            {
                "benchmark": "jmi",
                "benchmark_kind": "promotion",
                "estimator": estimator,
                "n": n,
                "p": p,
                "k": k,
                "top_m": top_m,
                "weighted": bool(args.weighted),
                "legacy_seconds": legacy_seconds,
                "current_seconds": current_seconds,
                "baseline_wall_seconds": legacy_seconds,
                "current_wall_seconds": current_seconds,
                "baseline_peak_memory_mb": None,
                "current_peak_memory_mb": None,
                "speedup": legacy_seconds / current_seconds if current_seconds > 0 else float("inf"),
                "selected_feature_parity": np.array_equal(current_selected, legacy_selected),
                "promotion_status": promotion_status(
                    parity=np.array_equal(current_selected, legacy_selected),
                    baseline_seconds=legacy_seconds,
                    current_seconds=current_seconds,
                ),
                "legacy_selected": legacy_selected.tolist(),
                "current_selected": current_selected.tolist(),
            }
        )

    print(_markdown_table(records))

    if args.output is not None:
        write_json(args.output, records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
