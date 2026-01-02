"""Automatic k selection for filter methods.

Design principle: Run selector ONCE to get ordered path, then evaluate prefixes.
This keeps filter methods fast while enabling principled k selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AutoKConfig:
    """Configuration for automatic k selection."""

    k_method: Literal["evaluate", "elbow"] = "evaluate"
    strategy: Literal["time_holdout", "group_cv"] = "time_holdout"
    metric: Literal["rmse", "mae"] = "rmse"
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
        return list(range(min_k, max_k + 1, 2))

    grid = set()
    grid.update(range(min_k, min(30, max_k) + 1, 5))
    grid.update([40, 50, 60, 75, 100, 125, 150])
    grid.add(min_k)
    grid.add(max_k)

    return sorted(k for k in grid if min_k <= k <= max_k)


def _compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute error metric (lower is better)."""
    err = y_true - y_pred
    if metric == "rmse":
        return float(np.sqrt(np.mean(err**2)))
    if metric == "mae":
        return float(np.mean(np.abs(err)))
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
    k_grid = _build_k_grid(min_k, max_k)

    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 10)),
    )
    results = []

    if config.strategy == "time_holdout":
        if time is None:
            raise ValueError("time_holdout strategy requires time parameter")

        train_idx, val_idx = _time_holdout_split(time, config.val_frac)

        for k in k_grid:
            feats = valid_features[:k]
            X_train = X.iloc[train_idx][feats].values
            X_val = X.iloc[val_idx][feats].values
            y_train = y_arr[train_idx]
            y_val = y_arr[val_idx]

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = _compute_metric(y_val, y_pred, config.metric)
                results.append({"k": k, "score": score})
            except Exception:
                results.append({"k": k, "score": np.inf})

    elif config.strategy == "group_cv":
        if groups is None:
            raise ValueError("group_cv strategy requires groups parameter")

        gkf = GroupKFold(n_splits=config.n_splits)

        for k in k_grid:
            feats = valid_features[:k]
            X_k = X[feats].values

            fold_scores = []
            try:
                for train_idx, val_idx in gkf.split(X_k, y_arr, groups):
                    model.fit(X_k[train_idx], y_arr[train_idx])
                    y_pred = model.predict(X_k[val_idx])
                    fold_scores.append(_compute_metric(y_arr[val_idx], y_pred, config.metric))
                results.append({"k": k, "score": np.mean(fold_scores)})
            except Exception:
                results.append({"k": k, "score": np.inf})

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    diag = pd.DataFrame(results)

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
    """Select k via elbow detection on an objective path (e.g. 2*I(y; S))."""
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

    This implementation is efficient:
      - builds R_path once (from cache.Rxx if available, else one BLAS corr)
      - uses Schur-complement logdet updates along the path
    """
    from sift.estimators.copula import (
        weighted_corr_with_vector,
        weighted_correlation_matrix,
        weighted_rank_gauss_1d,
    )

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
    k = int(path_valid_pos.size)

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

    if shrink > 0.0:
        R_path *= (1.0 - shrink)
        r_path *= (1.0 - shrink)
        np.fill_diagonal(R_path, 1.0)

    obj = np.empty(k, dtype=np.float64)

    logdet_S = 0.0
    inv_S = np.array([[1.0]], dtype=np.float64)

    r0 = float(r_path[0])
    det_yS = max(1.0 - r0 * r0, eps)
    logdet_yS = float(np.log(det_yS))
    inv_yS = (1.0 / det_yS) * np.array([[1.0, -r0], [-r0, 1.0]], dtype=np.float64)

    obj[0] = logdet_S - logdet_yS

    for t in range(1, k):
        b = R_path[:t, t].reshape(-1, 1)
        v = inv_S @ b
        s1 = float(1.0 - (b.T @ v)[0, 0])
        s1 = max(s1, eps)

        inv_S_new = np.empty((t + 1, t + 1), dtype=np.float64)
        inv_S_new[:t, :t] = inv_S + (v @ v.T) / s1
        inv_S_new[:t, t] = -v[:, 0] / s1
        inv_S_new[t, :t] = -v[:, 0] / s1
        inv_S_new[t, t] = 1.0 / s1
        inv_S = inv_S_new
        logdet_S += np.log(s1)

        b2 = np.empty((t + 1, 1), dtype=np.float64)
        b2[0, 0] = r_path[t]
        b2[1:, 0] = b[:, 0]

        v2 = inv_yS @ b2
        s2 = float(1.0 - (b2.T @ v2)[0, 0])
        s2 = max(s2, eps)

        inv_yS_new = np.empty((t + 2, t + 2), dtype=np.float64)
        inv_yS_new[: t + 1, : t + 1] = inv_yS + (v2 @ v2.T) / s2
        inv_yS_new[: t + 1, t + 1] = -v2[:, 0] / s2
        inv_yS_new[t + 1, : t + 1] = -v2[:, 0] / s2
        inv_yS_new[t + 1, t + 1] = 1.0 / s2
        inv_yS = inv_yS_new
        logdet_yS += np.log(s2)

        obj[t] = logdet_S - logdet_yS

    return obj
