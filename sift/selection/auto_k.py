"""Automatic k selection for filter methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sift.estimators.copula import FeatureCache


@dataclass
class AutoKConfig:
    """Configuration for automatic k selection."""

    k_method: Literal["evaluate", "elbow"] = "evaluate"
    strategy: Literal["time_holdout", "group_cv"] = "time_holdout"
    metric: Literal["rmse", "mae", "logloss", "error", "auto"] = "auto"
    max_k: int = 100
    min_k: int = 5
    val_frac: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    elbow_min_rel_gain: float = 0.02
    elbow_patience: int = 3


def _build_k_grid(min_k: int, max_k: int) -> List[int]:
    """Build sensible k grid: dense early, sparse later."""
    if max_k <= 30:
        grid = list(range(min_k, max_k + 1, 2))
        if grid and grid[-1] != max_k:
            grid.append(max_k)
        return grid

    grid = set()
    grid.update(range(min_k, min(30, max_k) + 1, 5))
    grid.update([40, 50, 60, 75, 100, 125, 150])
    grid.add(min_k)
    grid.add(max_k)

    return sorted(k for k in grid if min_k <= k <= max_k)


def _resolve_metric(metric: str, task: str) -> str:
    """Resolve metric, defaulting based on task."""
    if metric == "auto":
        return "rmse" if task == "regression" else "logloss"
    if task == "regression" and metric in ("logloss", "error"):
        raise ValueError(f"metric='{metric}' is invalid for task='regression'")
    if task == "classification" and metric in ("rmse", "mae"):
        raise ValueError(f"metric='{metric}' is invalid for task='classification'")
    return metric


def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    *,
    y_proba: np.ndarray | None = None,
) -> float:
    """Compute error metric (lower is better)."""
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "error":
        return float(1.0 - np.mean(y_true == y_pred))
    if metric == "logloss":
        if y_proba is None:
            return float(np.inf)
        return float(log_loss(y_true, y_proba))
    raise ValueError(f"Unknown metric: {metric}")


def _time_holdout_split(
    time_vals: np.ndarray,
    val_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by time: train on past, validate on future."""
    order = np.argsort(time_vals)
    n = len(order)
    cut = int(np.floor((1.0 - val_frac) * n))
    cut = max(1, min(cut, n - 1))
    return order[:cut], order[cut:]


def select_k_auto(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_path: List[str],
    config: AutoKConfig,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    task: Literal["regression", "classification"] = "regression",
) -> Tuple[int, List[str], pd.DataFrame]:
    """Select optimal k by evaluating prefixes of feature_path."""
    if not feature_path:
        return 0, [], pd.DataFrame()

    y_arr = np.asarray(y).ravel()
    max_k = min(config.max_k, len(feature_path))
    min_k = max(1, min(config.min_k, max_k))

    valid_features = [f for f in feature_path if f in X.columns]
    if not valid_features:
        return 0, [], pd.DataFrame()

    max_k = min(max_k, len(valid_features))
    valid_features = valid_features[:max_k]
    k_grid = _build_k_grid(min_k, max_k)

    X_path = X[valid_features].to_numpy(dtype=np.float64, copy=False)

    metric = _resolve_metric(config.metric, task)
    alphas = np.logspace(-3, 3, 10)

    def _eval_split(train_idx: np.ndarray, val_idx: np.ndarray) -> dict:
        """Evaluate all k values for one train/val split."""
        Xtr = X_path[train_idx]
        Xva = X_path[val_idx]
        ytr = y_arr[train_idx]
        yva = y_arr[val_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        if task == "regression":
            ridgecv = RidgeCV(alphas=alphas).fit(Xtr_s, ytr)
            alpha = float(ridgecv.alpha_)
            model = Ridge(alpha=alpha)
        else:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

        split_scores = {}
        for k in k_grid:
            try:
                if task == "classification" and len(np.unique(ytr)) < 2:
                    split_scores[k] = np.inf
                    continue

                model.fit(Xtr_s[:, :k], ytr)

                if task == "classification" and metric == "logloss":
                    proba = model.predict_proba(Xva_s[:, :k])
                    if not np.isin(np.unique(yva), model.classes_).all():
                        split_scores[k] = np.inf
                    else:
                        split_scores[k] = float(log_loss(yva, proba, labels=model.classes_))
                else:
                    pred = model.predict(Xva_s[:, :k])
                    split_scores[k] = _compute_metric(yva, pred, metric)
            except Exception:
                split_scores[k] = np.inf
        return split_scores

    if config.strategy == "time_holdout":
        if time is None:
            raise ValueError("time_holdout strategy requires time parameter")

        train_idx, val_idx = _time_holdout_split(time, config.val_frac)
        scores = _eval_split(train_idx, val_idx)
        diag = pd.DataFrame({"k": list(scores.keys()), "score": list(scores.values())})

    elif config.strategy == "group_cv":
        if groups is None:
            raise ValueError("group_cv strategy requires groups parameter")

        n_unique = len(np.unique(groups))
        n_splits = min(config.n_splits, n_unique)
        if n_splits < 2:
            raise ValueError(f"group_cv requires at least 2 groups, got {n_unique}")

        gkf = GroupKFold(n_splits=n_splits)

        all_scores = {k: [] for k in k_grid}
        for train_idx, val_idx in gkf.split(X_path, y_arr, groups):
            fold_scores = _eval_split(train_idx, val_idx)
            for k, score in fold_scores.items():
                all_scores[k].append(score)

        diag = pd.DataFrame(
            {"k": k_grid, "score": [np.mean(all_scores[k]) for k in k_grid]}
        )

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    if diag.empty or not np.isfinite(diag["score"]).any():
        return max_k, valid_features[:max_k], diag

    best_idx = diag["score"].idxmin()
    best_k = int(diag.loc[best_idx, "k"])

    return best_k, valid_features[:best_k], diag


def select_k_elbow(
    objective_path: np.ndarray,
    min_k: int = 5,
    max_k: int = 100,
    min_rel_gain: float = 0.02,
    patience: int = 3,
) -> Tuple[int, pd.DataFrame]:
    """Select k via elbow detection on an objective path."""
    obj = np.asarray(objective_path).ravel()
    max_k = min(max_k, len(obj))

    if max_k <= 0:
        return 0, pd.DataFrame()

    delta = np.zeros_like(obj)
    delta[0] = obj[0]
    delta[1:] = obj[1:] - obj[:-1]

    rel_gain = np.zeros_like(obj)
    rel_gain[0] = np.inf
    denom = np.maximum(np.abs(obj[:-1]), 1.0)
    rel_gain[1:] = delta[1:] / denom

    best_k = max_k
    run = 0

    for k in range(max(min_k, 2), max_k + 1):
        if rel_gain[k - 1] < min_rel_gain:
            run += 1
            if run >= patience:
                best_k = k - patience + 1
                break
        else:
            run = 0

    diag = pd.DataFrame(
        {
            "k": np.arange(1, max_k + 1),
            "objective": obj[:max_k],
            "delta": delta[:max_k],
            "rel_gain": rel_gain[:max_k],
        }
    )

    return best_k, diag


def compute_objective_for_path(
    cache: "FeatureCache",
    y: np.ndarray,
    feature_path: List[str],
    *,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute objective path for an arbitrary ordered feature_path.

    Objective at step t:
        obj[t] = log|Σ_S| - log|Σ_{y,S}|
               = 2 * I(y; S)   (Gaussian MI proxy)
    """
    from sift.estimators.copula import (
        weighted_corr_with_vector,
        weighted_correlation_matrix,
        weighted_rank_gauss_1d,
    )
    from sift.selection.objective import objective_from_corr_path

    if not feature_path:
        return np.empty(0, dtype=np.float64)

    valid_cols = np.asarray(cache.valid_cols)
    orig_to_valid = {int(orig): int(pos) for pos, orig in enumerate(valid_cols)}

    name_to_orig = {}
    if cache.feature_names:
        name_to_orig = {name: i for i, name in enumerate(cache.feature_names)}

    path_valid_pos = []
    for f in feature_path:
        if isinstance(f, str):
            orig_idx = name_to_orig.get(f, None)
            if orig_idx is None:
                continue
        else:
            orig_idx = int(f)

        vpos = orig_to_valid.get(int(orig_idx), None)
        if vpos is None:
            continue
        path_valid_pos.append(vpos)

    if not path_valid_pos:
        return np.empty(0, dtype=np.float64)

    path_valid_pos = np.asarray(path_valid_pos, dtype=np.int64)

    y_arr = np.asarray(y).ravel()
    ys = y_arr[np.asarray(cache.row_idx)]
    zy = weighted_rank_gauss_1d(ys, cache.sample_weight)
    r_y_full = weighted_corr_with_vector(cache.Z, zy, cache.sample_weight).astype(np.float64)

    r_path = r_y_full[path_valid_pos].copy()
    np.clip(r_path, -0.999999, 0.999999, out=r_path)

    if cache.Rxx is not None:
        R_full = np.asarray(cache.Rxx, dtype=np.float64)
        R_path = np.ascontiguousarray(R_full[np.ix_(path_valid_pos, path_valid_pos)], dtype=np.float64)
    else:
        Z_path = np.ascontiguousarray(cache.Z[:, path_valid_pos], dtype=np.float64)
        R_path = weighted_correlation_matrix(
            Z_path,
            np.asarray(cache.sample_weight, dtype=np.float64),
            backend="blas",
        )

    return objective_from_corr_path(R_path, r_path, shrink=shrink, eps=eps)
