"""Time-series-aware permutation importance."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def permutation_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    *,
    scoring: str | Callable = "neg_mse",
    n_repeats: int = 10,
    permute_method: str = "auto",
    block_size: int | str = "auto",
    n_jobs: int = -1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Permutation importance with optional time-series-aware strategies.

    Parameters
    ----------
    model : fitted estimator
        Must have .predict() method.
    X : DataFrame
        Features.
    y : array
        Target.
    sample_weight : array, optional
        Weights for scoring. Defaults to uniform.
    groups : array, optional
        Group labels. Enables within_group permutation.
    time : array, optional
        Time values. With groups, enables block/circular_shift.
    scoring : str or callable
        "neg_mse", "neg_mae", "r2", or callable(y, y_pred, w) -> float
    n_repeats : int
        Permutation repeats per feature.
    permute_method : str
        - "auto": circular_shift if groups+time, within_group if groups only, global otherwise
        - "global": standard shuffle
        - "within_group": shuffle within each group (requires groups)
        - "block": shuffle blocks within groups (requires groups + time)
        - "circular_shift": rotate within groups (requires groups + time)
    block_size : int or "auto"
        For block method.

    Returns
    -------
    DataFrame with: feature, importance_mean, importance_std, baseline_score
    """
    rng = np.random.default_rng(random_state)
    features = list(X.columns)
    X_arr = X.values.copy()
    n = len(y)

    w = np.ones(n, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    if permute_method == "auto":
        if groups is not None and time is not None:
            permute_method = "circular_shift"
        elif groups is not None:
            permute_method = "within_group"
        else:
            permute_method = "global"

    if permute_method in ("within_group", "block", "circular_shift") and groups is None:
        raise ValueError(f"permute_method='{permute_method}' requires groups")
    if permute_method in ("block", "circular_shift") and time is None:
        raise ValueError(f"permute_method='{permute_method}' requires time")

    group_info = _build_group_info(groups, time) if groups is not None else None

    baseline = _score(model, X_arr, y, w, scoring)

    def compute_importance(feat_idx: int) -> tuple[float, float]:
        col = X_arr[:, feat_idx].copy()
        drops = []

        for _ in range(n_repeats):
            seed = rng.integers(0, 2**31)
            permuted = _permute(col, group_info, permute_method, block_size, seed)

            X_perm = X_arr.copy()
            X_perm[:, feat_idx] = permuted

            score = _score(model, X_perm, y, w, scoring)
            drops.append(baseline - score)

        return float(np.mean(drops)), float(np.std(drops))

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_importance)(j) for j in range(len(features))
    )

    return pd.DataFrame({
        "feature": features,
        "importance_mean": [r[0] for r in results],
        "importance_std": [r[1] for r in results],
        "baseline_score": baseline,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _build_group_info(
    groups: np.ndarray | None,
    time: np.ndarray | None,
) -> dict | None:
    """Precompute time-sorted indices per group."""
    if groups is None:
        return None

    info = {}
    for g in np.unique(groups):
        mask = groups == g
        idx = np.where(mask)[0]
        if time is not None:
            order = np.argsort(time[idx])
            idx = idx[order]
        info[g] = idx
    return info


def _permute(
    x: np.ndarray,
    group_info: dict | None,
    method: str,
    block_size: int | str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if method == "global":
        return rng.permutation(x)
    if method == "within_group":
        return _permute_within_group(x, group_info, rng)
    if method == "block":
        return _permute_block(x, group_info, block_size, rng)
    if method == "circular_shift":
        return _permute_circular_shift(x, group_info, rng)
    raise ValueError(f"Unknown permute_method: {method}")


def _permute_within_group(x: np.ndarray, group_info: dict, rng) -> np.ndarray:
    result = x.copy()
    for idx in group_info.values():
        result[idx] = rng.permutation(x[idx])
    return result


def _permute_block(x: np.ndarray, group_info: dict, block_size: int | str, rng) -> np.ndarray:
    result = x.copy()
    for sorted_idx in group_info.values():
        n = len(sorted_idx)
        bs = int(np.sqrt(n)) if block_size == "auto" else min(block_size, n)
        bs = max(1, bs)

        n_blocks = max(1, int(np.ceil(n / bs)))
        blocks = [sorted_idx[i * bs: (i + 1) * bs] for i in range(n_blocks)]
        rng.shuffle(blocks)

        new_order = np.concatenate(blocks)
        result[sorted_idx] = x[new_order]
    return result


def _permute_circular_shift(x: np.ndarray, group_info: dict, rng) -> np.ndarray:
    result = x.copy()
    for sorted_idx in group_info.values():
        n = len(sorted_idx)
        if n <= 1:
            continue
        shift = rng.integers(1, n)
        result[sorted_idx] = np.roll(x[sorted_idx], shift)
    return result


def _score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable,
) -> float:
    y_pred = model.predict(X)

    if callable(scoring):
        return float(scoring(y, y_pred, w))

    if scoring == "neg_mse":
        return -float(np.average((y - y_pred) ** 2, weights=w))
    if scoring == "neg_mae":
        return -float(np.average(np.abs(y - y_pred), weights=w))
    if scoring == "r2":
        y_mean = np.average(y, weights=w)
        ss_res = np.average((y - y_pred) ** 2, weights=w)
        ss_tot = np.average((y - y_mean) ** 2, weights=w)
        return float(1 - ss_res / (ss_tot + 1e-10))
    raise ValueError(f"Unknown scoring: {scoring}")


__all__ = ["permutation_importance"]
