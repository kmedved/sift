#!/usr/bin/env python
"""Benchmark automatic-k methods on synthetic ground-truth designs."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.auto_k_designs import DESIGNS, score_support  # noqa: E402
from sift import build_cache  # noqa: E402
from sift.selection.auto_k import (  # noqa: E402
    AutoKConfig,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
    select_k_posterior,
)
from sift.selection.auto_k_stop import (  # noqa: E402
    select_k_changepoint,
    select_k_chi2_stop,
    select_k_forward_stop,
)
from sift.selection.auto_k_resample import (  # noqa: E402
    bootstrap_paths,
    null_objective_paths,
    select_k_perm_gap,
    select_k_stability,
)
from sift.selection.auto_k_xfit import (  # noqa: E402
    gaussian_cv_curves,
    select_k_gaussian_cv,
    select_k_xfit_objective,
    xfit_objective_curves,
)
from sift.selection.auto_k_knockoff import select_k_knockoff_path  # noqa: E402
from sift.selection.cefsplus import select_cached  # noqa: E402


BASELINE_METHODS = (
    "elbow",
    "penalized/bic",
    "evaluate/time_holdout/best",
    "evaluate/one_se",
    "fixed_k=50",
    "oracle",
)

CSV_COLUMNS = (
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


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_methods(value: str) -> tuple[str, ...]:
    methods: list[str] = []
    for part in _parse_csv(value):
        if part == "baselines":
            methods.extend(BASELINE_METHODS)
        else:
            methods.append(part)
    return tuple(methods)


def _path_max_k(p: int, k_star: int | None) -> int:
    guess = 25 if k_star is None else max(1, int(k_star))
    return min(p, max(4 * guess, 100))


def _design_max_k(p: int, meta: dict) -> int:
    if "benchmark_max_k" in meta:
        return min(p, max(1, int(meta["benchmark_max_k"])))
    return _path_max_k(p, meta.get("k_star"))


def _risk_grid(max_k: int) -> list[int]:
    values = [0]
    values.extend(range(1, min(max_k, 30) + 1))
    if max_k > 30:
        values.extend(range(35, min(max_k, 100) + 1, 5))
    if max_k > 100:
        values.extend(range(125, max_k + 1, 25))
    if values[-1] != max_k:
        values.append(max_k)
    return sorted(set(int(v) for v in values if 0 <= v <= max_k))


def _risk_model(model_kind: str, *, seed: int):
    if model_kind == "ridge":
        alphas = np.logspace(-3, 3, 10)
        return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    if model_kind == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "--model catboost requires the optional catboost dependency; "
                "install with python -m pip install -e '.[catboost]'"
            ) from exc
        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=120,
            depth=4,
            learning_rate=0.05,
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            bootstrap_type="No",
            random_strength=0.0,
        )
    raise ValueError(f"Unsupported risk model {model_kind!r}")


def _fit_rmse_curve(
    X,
    y,
    X_test,
    y_test,
    path_indices: list[int],
    grid: list[int],
    *,
    model_kind: str,
    seed: int,
) -> dict[int, float]:
    y_mean = float(np.mean(y))
    null_rmse = float(np.sqrt(np.mean((y_test - y_mean) ** 2)))
    out = {0: null_rmse}
    for k in grid:
        if k == 0:
            continue
        cols = path_indices[:k]
        model = _risk_model(model_kind, seed=seed + int(k))
        model.fit(X.iloc[:, cols], y)
        pred = model.predict(X_test.iloc[:, cols])
        out[int(k)] = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    return out


def _fit_rmse_for_indices(
    X,
    y,
    X_test,
    y_test,
    selected_indices: list[int],
    *,
    model_kind: str,
    seed: int,
) -> float:
    if not selected_indices:
        y_mean = float(np.mean(y))
        return float(np.sqrt(np.mean((y_test - y_mean) ** 2)))
    model = _risk_model(model_kind, seed=seed)
    model.fit(X.iloc[:, selected_indices], y)
    pred = model.predict(X_test.iloc[:, selected_indices])
    return float(np.sqrt(np.mean((y_test - pred) ** 2)))


def _method_k(
    method: str,
    *,
    X,
    y,
    path_names: list[str],
    objective: np.ndarray,
    cache,
    meta: dict,
    max_k: int,
    seed: int,
) -> tuple[int, str, float, list[int] | None]:
    start = time.perf_counter()
    notes = ""
    selected_indices_override = None
    if method == "elbow":
        k_hat, _diag = select_k_elbow(objective, min_k=1, max_k=max_k)
    elif method.startswith("penalized/"):
        penalty = method.split("/", 1)[1]
        cfg = AutoKConfig(
            k_method="penalized_objective",
            objective_penalty=penalty,
            min_k=0 if penalty in {"ebic", "ric"} else 1,
            max_k=max_k,
        )
        k_hat, _diag = select_k_penalized_objective(
            objective,
            cfg,
            objective_scale="n_eff",
            n_samples=len(cache.sample_weight),
            sample_weight=cache.sample_weight,
            n_candidates=len(cache.valid_cols),
            max_k=max_k,
        )
    elif method == "k_posterior":
        cfg = AutoKConfig(k_method="k_posterior", min_k=0, max_k=max_k)
        k_hat, _diag = select_k_posterior(
            objective,
            cfg,
            objective_scale="n_eff",
            n_samples=len(cache.sample_weight),
            sample_weight=cache.sample_weight,
            n_candidates=len(cache.valid_cols),
            max_k=max_k,
        )
    elif method == "chi2_stop":
        cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=max_k)
        n_eff = float(len(cache.sample_weight))
        k_hat, _diag = select_k_chi2_stop(
            objective,
            cfg,
            n_eff=n_eff,
            p_candidates=len(cache.valid_cols),
        )
    elif method == "forward_stop":
        cfg = AutoKConfig(k_method="forward_stop", min_k=0, max_k=max_k)
        n_eff = float(len(cache.sample_weight))
        k_hat, _diag = select_k_forward_stop(
            objective,
            cfg,
            n_eff=n_eff,
            p_candidates=len(cache.valid_cols),
        )
    elif method == "changepoint":
        cfg = AutoKConfig(k_method="changepoint", min_k=1, max_k=max_k)
        n_eff = float(len(cache.sample_weight))
        k_hat, _diag = select_k_changepoint(
            objective,
            cfg,
            objective_scale=n_eff,
            n_eff=n_eff,
            p_candidates=len(cache.valid_cols),
        )
    elif method == "perm_gap":
        cfg = AutoKConfig(k_method="perm_gap", min_k=0, max_k=max_k, perm_B=10, random_state=seed)
        nulls = null_objective_paths(
            cache,
            y,
            B=cfg.perm_B,
            max_k=max_k,
            null=cfg.perm_null,
            time=meta.get("time"),
            groups=meta.get("groups"),
            top_m=max(5 * max_k, 250),
            corr_prune="auto",
            random_state=seed,
        )
        k_hat, _diag = select_k_perm_gap(objective, nulls, cfg)
    elif method == "knockoff_path":
        cfg = AutoKConfig(
            k_method="knockoff_path",
            min_k=0,
            max_k=max_k,
            knockoff_q=0.2,
            random_state=seed,
        )
        selected_valid, k_hat, _diag = select_k_knockoff_path(
            cache,
            y,
            cfg,
            top_m=max(5 * max_k, 250),
        )
        selected_indices_override = (
            np.asarray(cache.valid_cols, dtype=np.int64)[selected_valid].astype(int).tolist()
        )
    elif method == "xfit_objective":
        cfg = AutoKConfig(
            k_method="xfit_objective",
            strategy="kfold",
            selection_rule="best",
            min_k=1,
            max_k=max_k,
            xfit_folds=3,
            random_state=seed,
        )
        curves = xfit_objective_curves(
            cache,
            y,
            config=cfg,
            top_m=max(5 * max_k, 250),
            corr_prune="auto",
            method="cefsplus",
        )
        k_hat, _diag = select_k_xfit_objective(curves, cfg)
    elif method == "gaussian_cv" or method.startswith("gaussian_cv/"):
        parts = method.split("/")
        selection_rule = parts[1] if len(parts) >= 2 and parts[1] else "one_se"
        strategy = parts[2] if len(parts) >= 3 and parts[2] else "kfold"
        if selection_rule not in {"best", "one_se", "plateau", "tolerance"}:
            raise ValueError(f"Unsupported gaussian_cv selection rule {selection_rule!r}")
        if strategy not in {"kfold", "group_cv", "time_holdout"}:
            raise ValueError(f"Unsupported gaussian_cv strategy {strategy!r}")
        if strategy == "group_cv" and "groups" not in meta:
            notes = "group_cv_requested_without_groups;using_kfold"
            strategy = "kfold"
        if strategy == "time_holdout" and "time" not in meta:
            notes = "time_holdout_requested_without_time;using_kfold"
            strategy = "kfold"
        cfg = AutoKConfig(
            k_method="gaussian_cv",
            strategy=strategy,
            selection_rule=selection_rule,
            min_k=1,
            max_k=max_k,
            xfit_folds=3,
            random_state=seed,
        )
        curves = gaussian_cv_curves(
            cache,
            y,
            config=cfg,
            top_m=max(5 * max_k, 250),
            corr_prune="auto",
            method="cefsplus",
            groups=meta.get("groups"),
            time=meta.get("time"),
        )
        k_hat, _diag = select_k_gaussian_cv(curves, cfg)
    elif method == "stability":
        cfg = AutoKConfig(k_method="stability", min_k=1, max_k=max_k, boot_B=10, random_state=seed)
        paths = bootstrap_paths(
            cache,
            y,
            B=cfg.boot_B,
            max_k=max_k,
            boot_mode=cfg.boot_mode,
            top_m=max(5 * max_k, 250),
            corr_prune="auto",
            random_state=seed,
        )
        k_hat, _diag = select_k_stability(paths, len(cache.valid_cols), cfg)
    elif method == "consensus":
        cfg = AutoKConfig(
            k_method="consensus",
            min_k=0,
            max_k=max_k,
            perm_B=10,
            xfit_folds=3,
            consensus_methods=("ebic", "chi2_stop", "gaussian_cv"),
            random_state=seed,
        )
        votes = []
        ebic_cfg = AutoKConfig(
            k_method="penalized_objective",
            objective_penalty="ebic",
            min_k=0,
            max_k=max_k,
        )
        ebic_k, _diag = select_k_penalized_objective(
            objective,
            ebic_cfg,
            objective_scale="n_eff",
            n_samples=len(cache.sample_weight),
            sample_weight=cache.sample_weight,
            n_candidates=len(cache.valid_cols),
            max_k=max_k,
        )
        votes.append(int(ebic_k))
        chi_cfg = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=max_k)
        chi_k, _diag = select_k_chi2_stop(
            objective,
            chi_cfg,
            n_eff=float(len(cache.sample_weight)),
            p_candidates=len(cache.valid_cols),
        )
        votes.append(int(chi_k))
        cv_cfg = AutoKConfig(
            k_method="gaussian_cv",
            strategy="kfold",
            selection_rule="one_se",
            min_k=1,
            max_k=max_k,
            xfit_folds=3,
            random_state=seed,
        )
        curves = gaussian_cv_curves(
            cache,
            y,
            config=cv_cfg,
            top_m=max(5 * max_k, 250),
            corr_prune="auto",
            method="cefsplus",
        )
        cv_k, _diag = select_k_gaussian_cv(curves, cv_cfg)
        votes.append(int(cv_k))
        votes.sort()
        k_hat = votes[(len(votes) - 1) // 2]
    elif method == "evaluate/time_holdout/best":
        eval_time = np.asarray(meta["time"]) if "time" in meta else np.arange(len(y))
        cfg = AutoKConfig(
            k_method="evaluate",
            strategy="time_holdout",
            selection_rule="best",
            min_k=1,
            max_k=max_k,
            val_frac=0.2,
            random_state=seed,
        )
        k_hat, _selected, _diag = select_k_auto(X, y, path_names, cfg, time=eval_time)
    elif method == "evaluate/one_se":
        eval_groups = np.asarray(meta["groups"]) if "groups" in meta else np.arange(len(y))
        cfg = AutoKConfig(
            k_method="evaluate",
            strategy="group_cv",
            selection_rule="one_se",
            min_k=1,
            max_k=max_k,
            n_splits=5,
            random_state=seed,
        )
        k_hat, _selected, _diag = select_k_auto(X, y, path_names, cfg, groups=eval_groups)
        if "groups" not in meta:
            notes = "synthetic_group_cv"
    elif method.startswith("fixed_k="):
        k_hat = min(max_k, int(method.split("=", 1)[1]))
    else:
        raise ValueError(f"Unsupported benchmark method {method!r}")
    runtime = time.perf_counter() - start
    if method == "evaluate/time_holdout/best" and "time" not in meta and "groups" not in meta:
        notes = "synthetic_time_holdout"
    return int(k_hat), notes, float(runtime), selected_indices_override


def _row(
    *,
    design_id: str,
    seed: int,
    method: str,
    k_hat: int,
    k_oracle: int,
    k_star,
    rmse_hat: float,
    rmse_oracle: float,
    rmse_null: float,
    support_scores: tuple[float, float, float],
    max_k: int,
    runtime_s: float,
    notes: str,
) -> dict:
    denom = rmse_null - rmse_oracle
    if denom <= 1e-12:
        regret = 0.0 if rmse_hat <= rmse_oracle + 1e-12 else float("nan")
        notes = f"{notes};regret_denominator_zero".strip(";")
    else:
        regret = (rmse_hat - rmse_oracle) / denom
    precision, recall, f1 = support_scores
    return {
        "design": design_id,
        "seed": seed,
        "method": method,
        "k_hat": int(k_hat),
        "k_oracle": int(k_oracle),
        "k_star": "" if k_star is None else int(k_star),
        "rmse_hat": rmse_hat,
        "rmse_oracle": rmse_oracle,
        "regret_frac": regret,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "k_dispersion_group": f"{design_id}:{method}",
        "saturated_min": bool(k_hat <= 0),
        "saturated_max": bool(k_hat >= max_k),
        "runtime_s": runtime_s,
        "notes": notes,
    }


def run(args: argparse.Namespace) -> list[dict]:
    design_ids = _parse_csv(args.designs)
    methods = _parse_methods(args.methods)
    rows: list[dict] = []

    for design_id in design_ids:
        design = DESIGNS[design_id]
        for seed in range(int(args.seeds)):
            X, y, meta = design.make(seed, bool(args.full))
            max_k = _design_max_k(X.shape[1], meta)
            cache = build_cache(
                X,
                subsample=None if X.shape[0] <= 50_000 else 50_000,
                random_state=seed,
                compute_Rxx=X.shape[1] <= 4000,
            )
            path_names, path_indices, objective = select_cached(
                cache,
                y,
                max_k,
                method="cefsplus",
                top_m=max(5 * max_k, 250),
                return_indices=True,
                return_objective=True,
            )
            max_path_k = len(path_indices)
            if max_path_k == 0:
                continue
            risk_grid = _risk_grid(max_path_k)
            X_test, y_test = design.sample_test(seed, int(args.n_test), bool(args.full))
            rmse_curve = _fit_rmse_curve(
                X,
                y,
                X_test,
                y_test,
                path_indices,
                risk_grid,
                model_kind=args.model,
                seed=seed,
            )
            k_oracle = min(rmse_curve, key=lambda item: (rmse_curve[item], item))
            rmse_oracle = rmse_curve[k_oracle]
            rmse_null = rmse_curve[0]

            for method in methods:
                if method == "oracle":
                    k_hat, notes, runtime_s, selected_override = int(k_oracle), "", 0.0, None
                else:
                    method_kwargs = {
                        "X": X,
                        "y": y,
                        "path_names": path_names,
                        "objective": objective,
                        "cache": cache,
                        "meta": meta,
                        "max_k": max_path_k,
                        "seed": seed,
                    }
                    # Discard one warm-up run, then report a median rather than
                    # a noisy single cold measurement.
                    _method_k(method, **method_kwargs)
                    trials = [
                        _method_k(method, **method_kwargs)
                        for _ in range(int(getattr(args, "timing_repeats", 3)))
                    ]
                    k_hat, notes, _runtime, selected_override = trials[0]
                    runtime_s = float(np.median([trial[2] for trial in trials]))
                if selected_override is None:
                    selected_indices = path_indices[: max(0, min(k_hat, len(path_indices)))]
                    if k_hat in rmse_curve:
                        rmse_hat = rmse_curve[k_hat]
                    else:
                        rmse_hat = _fit_rmse_for_indices(
                            X,
                            y,
                            X_test,
                            y_test,
                            selected_indices,
                            model_kind=args.model,
                            seed=seed + 100_000 + int(k_hat),
                        )
                        notes = f"{notes};exact_off_grid_k".strip(";")
                else:
                    selected_indices = selected_override
                    rmse_hat = _fit_rmse_for_indices(
                        X,
                        y,
                        X_test,
                        y_test,
                        selected_indices,
                        model_kind=args.model,
                        seed=seed + 200_000 + len(selected_indices),
                    )
                support_scores = score_support(selected_indices, meta)
                rows.append(
                    _row(
                        design_id=design_id,
                        seed=seed,
                        method=method,
                        k_hat=k_hat,
                        k_oracle=int(k_oracle),
                        k_star=meta.get("k_star"),
                        rmse_hat=rmse_hat,
                        rmse_oracle=rmse_oracle,
                        rmse_null=rmse_null,
                        support_scores=support_scores,
                        max_k=max_path_k,
                        runtime_s=runtime_s,
                        notes=notes,
                    )
                )
    return rows


def _normalize_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    if args.quick and args.full:
        if parser is not None:
            parser.error("--quick and --full are mutually exclusive")
        raise ValueError("--quick and --full are mutually exclusive")
    if args.quick:
        args.seeds = min(int(args.seeds), 1)
        args.n_test = min(int(args.n_test), 1_000)
    timing_repeats = int(getattr(args, "timing_repeats", 3))
    if timing_repeats <= 0:
        if parser is not None:
            parser.error("--timing-repeats must be positive")
        raise ValueError("--timing-repeats must be positive")
    args.timing_repeats = timing_repeats
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--designs", default="D1,D2,D3,D5")
    parser.add_argument("--methods", default="baselines")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("ridge", "catboost"), default="ridge")
    parser.add_argument("--n-test", type=int, default=20_000)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    _normalize_args(args, parser)
    rows = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
