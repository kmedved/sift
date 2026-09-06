"""Shared split, metric, and score-curve helpers for auto-k selection."""

from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from sift.scoring import (
    UnsupportedScorerSampleWeightError,
    is_sklearn_scorer,
    score_with_sklearn_scorer,
)


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


def resolve_metric(metric: object, task: str) -> object:
    """Resolve metric, defaulting based on task."""
    if is_sklearn_scorer(metric):
        return metric
    if not isinstance(metric, str):
        raise ValueError(
            "metric must be a SIFT metric name or an sklearn scorer object"
        )
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


def _as_ridge_alpha(alpha) -> float | np.ndarray:
    """Return a Ridge-compatible alpha from RidgeCV (scalar or per-target)."""
    arr = np.asarray(alpha, dtype=np.float64)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return arr


def weighted_regression_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Row-weighted RMSE/MAE; 2-D ``y`` averages targets after the row reduction.

    For ``q=1`` this is the historical scalar formula. For ``q>=2`` each
    target is scored with the same row weights, then the ``q`` values are
    averaged with equal target weight. RMSE is the square root of the mean
    of those per-target MSEs (Frobenius), not the mean of per-target RMSEs.
    """
    if metric not in {"rmse", "mae"}:
        raise ValueError("scoring must be 'rmse' or 'mae'")
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape for regression scoring"
        )
    if metric == "rmse":
        err = (true - pred) ** 2
    else:
        err = np.abs(true - pred)
    per_target = np.average(err, axis=0, weights=sample_weight)
    reduced = float(np.mean(per_target))
    if metric == "rmse":
        return float(np.sqrt(reduced))
    return reduced


def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    *,
    y_proba: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Compute error metric (lower is better)."""
    if metric in {"rmse", "mae"}:
        return weighted_regression_score(
            y_true, y_pred, metric, sample_weight=sample_weight
        )
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
    """Split by time without separating equal timestamps.

    Rows are stably sorted by ``time_vals`` and the cut is moved to the
    nearest boundary between distinct timestamps. A holdout cannot be
    defined for fewer than two rows or for an all-tied timestamp vector, so
    those cases raise instead of silently leaking a timestamp across folds.
    """
    if not isinstance(val_frac, (int, float, np.integer, np.floating)) or isinstance(
        val_frac, (bool, np.bool_)
    ):
        raise TypeError("val_frac must be a real number in (0, 1)")
    val_frac = float(val_frac)
    if not np.isfinite(val_frac) or not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be finite and in (0, 1)")

    values = np.asarray(time_vals).reshape(-1)
    n = values.size
    if n < 2:
        raise ValueError("time_holdout_split requires at least two rows")
    try:
        missing = np.asarray(pd.isna(values), dtype=bool)
    except (TypeError, ValueError):
        missing = np.zeros(n, dtype=bool)
    if missing.any():
        raise ValueError("time values must not contain missing values")

    try:
        order = np.argsort(values, kind="mergesort")
        ordered = values[order]
        # Native dtypes can verify the sorted result in one vectorized pass.
        # Keep scalar comparisons for object arrays so mixed or exotic Python
        # values retain their existing comparison and rejection semantics.
        if values.dtype.kind == "O":
            for previous, current in zip(ordered[:-1], ordered[1:]):
                if bool(current < previous):
                    raise TypeError("time values are not monotonically orderable")
        elif np.asarray(ordered[1:] < ordered[:-1], dtype=bool).any():
            raise TypeError("time values are not monotonically orderable")
    except (TypeError, ValueError) as exc:
        raise TypeError("time values must be orderable") from exc
    try:
        distinct = np.asarray(ordered[1:] != ordered[:-1], dtype=bool)
    except Exception as exc:  # pragma: no cover - defensive for exotic objects
        raise TypeError("time values must be orderable") from exc
    boundaries = np.flatnonzero(distinct) + 1
    if boundaries.size == 0:
        raise ValueError("time_holdout_split requires at least two distinct timestamps")

    desired_train = (1.0 - val_frac) * n
    distance = np.abs(boundaries.astype(float) - desired_train)
    # In an exact tie, choose the smaller boundary: validation is still wholly
    # future and is not made implausibly smaller than requested.
    cut = int(boundaries[np.flatnonzero(distance == distance.min())[0]])
    return order[:cut].astype(np.int64, copy=False), order[cut:].astype(np.int64, copy=False)


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
        # Prefix sizes must be compared on the same validation coverage. A k
        # that succeeds on only one unusually favorable fold must not beat a k
        # evaluated on every fold. Preserve finite counts for diagnostics, but
        # make any partially failed row ineligible for selection.
        score_mean = (
            float(np.mean(values))
            if values.size and finite.size == values.size
            else float("inf")
        )
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
    metric: object,
    k_grid: list[int],
    ridge_alpha_strategy: Literal["per_prefix", "full_path"] = "per_prefix",
    sample_weight_supplied: bool = True,
) -> dict[int, float]:
    """Evaluate all prefix sizes on an already-built feature path."""
    if X_train_path.shape[1] == 0:
        return {k: np.inf for k in k_grid}
    if ridge_alpha_strategy not in {"per_prefix", "full_path"}:
        raise ValueError("ridge_alpha_strategy must be 'per_prefix' or 'full_path'")

    Xtr_s, Xva_s = numeric_train_val(X_train_path, X_val_path)
    scores: dict[int, float] = {}
    alphas = np.logspace(-3, 3, 10)

    from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV

    full_path_alpha = None
    if task == "regression" and ridge_alpha_strategy == "full_path":
        # An outer sklearn Pipeline may enable metadata routing globally.  This
        # private RidgeCV call receives weights directly and must keep its
        # historical, non-routed semantics on every supported sklearn version.
        with config_context(enable_metadata_routing=False):
            ridgecv = RidgeCV(alphas=alphas).fit(
                Xtr_s,
                y_train,
                sample_weight=w_train,
            )
        full_path_alpha = _as_ridge_alpha(ridgecv.alpha_)

    for k in k_grid:
        if k > Xtr_s.shape[1]:
            scores[k] = np.inf
            continue
        try:
            if task == "classification" and len(np.unique(y_train)) < 2:
                scores[k] = np.inf
                continue

            if task == "regression":
                if full_path_alpha is None:
                    with config_context(enable_metadata_routing=False):
                        ridgecv = RidgeCV(alphas=alphas).fit(
                            Xtr_s[:, :k],
                            y_train,
                            sample_weight=w_train,
                        )
                    alpha = _as_ridge_alpha(ridgecv.alpha_)
                else:
                    alpha = full_path_alpha
                model = Ridge(alpha=alpha)
            else:
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

            model.fit(Xtr_s[:, :k], y_train, sample_weight=w_train)

            if is_sklearn_scorer(metric):
                signed_score = score_with_sklearn_scorer(
                    metric,
                    model,
                    Xva_s[:, :k],
                    y_val,
                    sample_weight=w_val if sample_weight_supplied else None,
                )
                scores[k] = -signed_score
            elif task == "classification" and metric == "logloss":
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
        except UnsupportedScorerSampleWeightError:
            raise
        except Exception:
            scores[k] = np.inf
    return scores
