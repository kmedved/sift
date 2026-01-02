"""Time-series-aware permutation importance."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sift._permute import (
    PermutationMethod,
    build_group_info,
    permute_array,
    resolve_permutation_method,
)
from sift._preprocess import ensure_weights


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
    permute_method: PermutationMethod = "auto",
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
        "neg_mse", "neg_rmse", "neg_mae", "r2", or callable(y, y_pred, w) -> float
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
    features = list(X.columns)
    X_arr = X.values.copy()
    n = len(y)

    w = ensure_weights(sample_weight, n, normalize=True)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31, size=(len(features), n_repeats))

    permute_method = resolve_permutation_method(permute_method, groups=groups, time=time)

    if permute_method in ("within_group", "block", "circular_shift") and groups is None:
        raise ValueError(f"permute_method='{permute_method}' requires groups")
    if permute_method in ("block", "circular_shift") and time is None:
        raise ValueError(f"permute_method='{permute_method}' requires time")

    group_info = build_group_info(groups, time) if groups is not None else None

    baseline = _score(model, X_arr, y, w, scoring)

    def compute_importance(feat_idx: int) -> tuple[float, float]:
        X_work = X_arr.copy()
        orig_col = X_work[:, feat_idx].copy()
        drops = []

        for rep in range(n_repeats):
            seed = int(seeds[feat_idx, rep])
            permuted = permute_array(
                orig_col,
                method=permute_method,
                group_info=group_info,
                block_size=block_size,
                rng=np.random.default_rng(seed),
            )
            X_work[:, feat_idx] = permuted
            score = _score(model, X_work, y, w, scoring)
            drops.append(baseline - score)

        return float(np.mean(drops)), float(np.std(drops))

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_importance)(j) for j in range(len(features))
    )

    return (
        pd.DataFrame(
            {
                "feature": features,
                "importance_mean": [r[0] for r in results],
                "importance_std": [r[1] for r in results],
                "baseline_score": baseline,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


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
    if scoring == "neg_rmse":
        return -float(np.sqrt(np.average((y - y_pred) ** 2, weights=w)))
    if scoring == "neg_mae":
        return -float(np.average(np.abs(y - y_pred), weights=w))
    if scoring == "r2":
        y_mean = np.average(y, weights=w)
        ss_res = np.average((y - y_pred) ** 2, weights=w)
        ss_tot = np.average((y - y_mean) ** 2, weights=w)
        return float(1 - ss_res / (ss_tot + 1e-10))
    raise ValueError(f"Unknown scoring: {scoring}")


__all__ = ["permutation_importance"]
