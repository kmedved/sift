"""Shared split, metric, and score-curve helpers for auto-k selection."""

from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


def build_k_grid(min_k: int, max_k: int) -> List[int]:
    """Build sensible k grid: dense early, sparse later."""
    if max_k <= 30:
        grid = list(range(min_k, max_k + 1, 2))
        if grid and grid[-1] != max_k:
            grid.append(max_k)
        return grid

    grid = set()
    grid.update(range(min_k, min(30, max_k) + 1, 5))
    grid.update(
        [40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1000]
    )
    grid.add(min_k)
    grid.add(max_k)

    return sorted(k for k in grid if min_k <= k <= max_k)


def resolve_metric(metric: str, task: str) -> str:
    """Resolve metric, defaulting based on task."""
    if metric == "auto":
        return "rmse" if task == "regression" else "logloss"
    if task == "regression":
        valid = ("rmse", "mae")
        if metric not in valid:
            raise ValueError(
                f"metric='{metric}' is invalid for task='regression'. "
                f"Valid metrics: {valid} or 'auto'"
            )
    elif task == "classification":
        valid = ("logloss", "error")
        if metric not in valid:
            raise ValueError(
                f"metric='{metric}' is invalid for task='classification'. "
                f"Valid metrics: {valid} or 'auto'"
            )
    else:
        raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
    return metric


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    *,
    y_proba: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute error metric (lower is better)."""
    if metric == "rmse":
        return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight)))
    if metric == "mae":
        return float(np.average(np.abs(y_true - y_pred), weights=sample_weight))
    if metric == "error":
        return float(np.average(y_true != y_pred, weights=sample_weight))
    if metric == "logloss":
        if y_proba is None:
            return float(np.inf)
        return float(log_loss(y_true, y_proba, sample_weight=sample_weight))
    raise ValueError(f"Unknown metric: {metric}")


def split_weights(w: np.ndarray, idx: np.ndarray, label: str) -> np.ndarray:
    """Return fold-local mean-one weights for fitting/scoring."""
    out = np.asarray(w[idx], dtype=np.float64)
    total = float(out.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} split has zero total sample_weight")
    mean = float(out.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"{label} split has invalid sample_weight mean")
    return out / mean


def time_holdout_split(
    time_vals: np.ndarray,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by time: train on past, validate on future."""
    order = np.argsort(time_vals)
    n = len(order)
    cut = int(np.floor((1.0 - val_frac) * n))
    cut = max(1, min(cut, n - 1))
    return order[:cut], order[cut:]


def numeric_train_val(X_train, X_val) -> tuple[np.ndarray, np.ndarray]:
    """Convert train/validation path matrices to scaled finite arrays."""
    if isinstance(X_train, pd.DataFrame):
        Xtr = X_train.to_numpy(dtype=np.float64, copy=False)
    else:
        Xtr = np.asarray(X_train, dtype=np.float64)
    if isinstance(X_val, pd.DataFrame):
        Xva = X_val.to_numpy(dtype=np.float64, copy=False)
    else:
        Xva = np.asarray(X_val, dtype=np.float64)

    with np.errstate(all="ignore"):
        col_means = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    mask_tr = ~np.isfinite(Xtr)
    if mask_tr.any():
        Xtr = Xtr.copy()
        Xtr[mask_tr] = col_means[np.where(mask_tr)[1]]

    mask_va = ~np.isfinite(Xva)
    if mask_va.any():
        Xva = Xva.copy()
        Xva[mask_va] = col_means[np.where(mask_va)[1]]

    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xva)


def build_score_curve_diagnostics(
    k_grid: List[int],
    split_scores: dict[int, list[float]],
) -> pd.DataFrame:
    """Summarize split-level prefix scores while preserving the old score column."""
    rows = []
    for k in k_grid:
        values = np.asarray(split_scores.get(k, []), dtype=np.float64)
        finite = values[np.isfinite(values)]
        score_mean = float(np.mean(values)) if values.size else float("inf")
        score_std = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
        score_se = score_std / float(np.sqrt(finite.size)) if finite.size >= 2 else float("nan")
        rows.append(
            {
                "k": int(k),
                "score": score_mean,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_se": score_se,
                "n_splits": int(values.size),
                "n_finite": int(finite.size),
                "split_scores": tuple(float(v) for v in values.tolist()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_numeric_prefixes(
    X_train_path,
    X_val_path,
    y_train: np.ndarray,
    y_val: np.ndarray,
    w_train: np.ndarray,
    w_val: np.ndarray,
    *,
    task: Literal["regression", "classification"],
    metric: str,
    k_grid: list[int],
) -> dict[int, float]:
    """Evaluate all prefix sizes on an already-built feature path."""
    if X_train_path.shape[1] == 0:
        return {k: np.inf for k in k_grid}

    Xtr_s, Xva_s = numeric_train_val(X_train_path, X_val_path)
    scores: dict[int, float] = {}
    alphas = np.logspace(-3, 3, 10)

    from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV

    for k in k_grid:
        if k > Xtr_s.shape[1]:
            scores[k] = np.inf
            continue
        try:
            if task == "classification" and len(np.unique(y_train)) < 2:
                scores[k] = np.inf
                continue

            if task == "regression":
                ridgecv = RidgeCV(alphas=alphas).fit(
                    Xtr_s[:, :k],
                    y_train,
                    sample_weight=w_train,
                )
                model = Ridge(alpha=float(ridgecv.alpha_))
            else:
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

            model.fit(Xtr_s[:, :k], y_train, sample_weight=w_train)

            if task == "classification" and metric == "logloss":
                if not np.isin(np.unique(y_val), model.classes_).all():
                    scores[k] = np.inf
                else:
                    proba = model.predict_proba(Xva_s[:, :k])
                    scores[k] = float(
                        log_loss(
                            y_val,
                            proba,
                            labels=model.classes_,
                            sample_weight=w_val,
                        )
                    )
            else:
                pred = model.predict(Xva_s[:, :k])
                scores[k] = compute_metric(
                    y_val,
                    pred,
                    metric,
                    sample_weight=w_val,
                )
        except Exception:
            scores[k] = np.inf
    return scores
