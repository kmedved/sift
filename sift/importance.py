"""Time-series-aware permutation importance."""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from sift._permute import (
    PermutationMethod,
    build_group_info,
    permute_array,
    resolve_permutation_method,
)
from sift._preprocess import ensure_weights
from sift.scoring import get_scoring


ParallelBackend = Literal["threads", "processes"]
_VALID_PARALLEL_BACKENDS: tuple[ParallelBackend, ...] = ("threads", "processes")


def permutation_importance(
    model,
    X: pd.DataFrame | np.ndarray,
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
    parallel_backend: ParallelBackend = "threads",
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Permutation importance with optional time-series-aware strategies.

    Parameters
    ----------
    model : fitted estimator
        Must have .predict() method.
    X : DataFrame or ndarray
        Features.
    y : array
        Target.
    sample_weight : array, optional
        Weights for scoring. Defaults to uniform.
    groups : array, optional
        Group labels. Enables within_group permutation.
    time : array, optional
        Time values. Enables block/circular_shift. If provided without groups,
        the full dataset is treated as one ordered group.
    scoring : str or callable
        "neg_mse", "neg_rmse", "neg_mae", "r2", "accuracy",
        "balanced_accuracy", "neg_error", "neg_logloss", or
        callable(y, y_pred, w) -> float. Built-in scoring strings are
        higher-is-better.
    n_repeats : int
        Permutation repeats per feature.
    permute_method : str
        - "auto": circular_shift if time is provided, within_group if groups only, global otherwise
        - "global": standard shuffle
        - "within_group": shuffle within each group (requires groups)
        - "block": shuffle time-ordered blocks (requires time)
        - "circular_shift": rotate within time order (requires time)
    block_size : int or "auto"
        For block method.
    n_jobs : int
        Number of parallel jobs.
    parallel_backend : {"threads", "processes"}
        Joblib backend preference. "threads" avoids inter-process copies for
        many estimators; "processes" isolates workers when process parallelism
        is preferred.

    Returns
    -------
    DataFrame with: feature, importance_mean, importance_std, baseline_score
    """
    if isinstance(n_repeats, (bool, np.bool_)) or not isinstance(n_repeats, (int, np.integer)):
        raise ValueError("n_repeats must be a positive integer")
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    parallel_backend = _validate_parallel_backend(parallel_backend)

    n = len(y)
    X_arr = None
    if isinstance(X, pd.DataFrame):
        if X.shape[0] != n:
            raise ValueError(f"X has {X.shape[0]} rows but y has {n}")
        n_features = X.shape[1]
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or pandas DataFrame")
        if X_arr.shape[0] != n:
            raise ValueError(f"X has {X_arr.shape[0]} rows but y has {n}")
        n_features = X_arr.shape[1]

    w = ensure_weights(sample_weight, n, normalize=True)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31, size=(n_features, n_repeats))

    permute_method = resolve_permutation_method(permute_method, groups=groups, time=time)
    if permute_method in ("block", "circular_shift") and time is None:
        raise ValueError(f"permute_method='{permute_method}' requires time")
    group_info = (
        build_group_info(groups, time, n_samples=n) if permute_method != "global" else None
    )

    if isinstance(X, pd.DataFrame):
        return _permutation_importance_dataframe(
            model,
            X,
            y,
            w,
            scoring,
            n_repeats,
            permute_method,
            block_size,
            group_info,
            seeds,
            baseline=_score(model, X, y, w, scoring),
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

    assert X_arr is not None
    return _permutation_importance_array(
        model,
        X_arr,
        y,
        w,
        scoring,
        n_repeats,
        permute_method,
        block_size,
        group_info,
        seeds,
        baseline=_score(model, X_arr, y, w, scoring),
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )


def _permutation_importance_dataframe(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable,
    n_repeats: int,
    permute_method: PermutationMethod,
    block_size: int | str,
    group_info: dict | None,
    seeds: np.ndarray,
    *,
    baseline: float,
    n_jobs: int,
    parallel_backend: ParallelBackend,
) -> pd.DataFrame:
    features = list(X.columns)
    row_positions = np.arange(X.shape[0])
    chunks = _feature_chunks(len(features), n_jobs)

    def compute_chunk(feature_indices: np.ndarray) -> list[tuple[int, float, float]]:
        X_work = X.copy()
        chunk_results = []

        for feat_idx in feature_indices:
            orig_col = X.iloc[:, feat_idx].copy()
            drops = []

            for rep in range(n_repeats):
                seed = int(seeds[feat_idx, rep])
                source_positions = permute_array(
                    row_positions,
                    method=permute_method,
                    group_info=group_info,
                    block_size=block_size,
                    rng=np.random.default_rng(seed),
                )
                permuted = orig_col.iloc[source_positions].copy()
                permuted.index = X_work.index
                X_work.iloc[:, feat_idx] = permuted
                try:
                    score = _score(model, X_work, y, w, scoring)
                finally:
                    X_work.iloc[:, feat_idx] = orig_col
                drops.append(baseline - score)

            chunk_results.append((feat_idx, float(np.mean(drops)), float(np.std(drops))))

        return chunk_results

    result_chunks = Parallel(n_jobs=n_jobs, prefer=parallel_backend)(
        delayed(compute_chunk)(chunk) for chunk in chunks
    )
    results = _flatten_feature_results(result_chunks, len(features))

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


def _permutation_importance_array(
    model,
    X_arr: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable,
    n_repeats: int,
    permute_method: PermutationMethod,
    block_size: int | str,
    group_info: dict | None,
    seeds: np.ndarray,
    *,
    baseline: float,
    n_jobs: int,
    parallel_backend: ParallelBackend,
) -> pd.DataFrame:
    features = list(range(X_arr.shape[1]))
    chunks = _feature_chunks(len(features), n_jobs)

    def compute_chunk(feature_indices: np.ndarray) -> list[tuple[int, float, float]]:
        X_work = X_arr.copy()
        chunk_results = []

        for feat_idx in feature_indices:
            orig_col = X_arr[:, feat_idx]
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
                try:
                    score = _score(model, X_work, y, w, scoring)
                finally:
                    X_work[:, feat_idx] = orig_col
                drops.append(baseline - score)

            chunk_results.append((feat_idx, float(np.mean(drops)), float(np.std(drops))))

        return chunk_results

    result_chunks = Parallel(n_jobs=n_jobs, prefer=parallel_backend)(
        delayed(compute_chunk)(chunk) for chunk in chunks
    )
    results = _flatten_feature_results(result_chunks, len(features))

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


def _validate_parallel_backend(parallel_backend: str) -> ParallelBackend:
    if parallel_backend not in _VALID_PARALLEL_BACKENDS:
        raise ValueError(
            f"Unknown parallel_backend: {parallel_backend!r}. "
            f"Expected one of {list(_VALID_PARALLEL_BACKENDS)}."
        )
    return parallel_backend  # type: ignore[return-value]


def _feature_chunks(n_features: int, n_jobs: int) -> list[np.ndarray]:
    if n_features < 1:
        return []
    n_workers = min(n_features, max(1, effective_n_jobs(n_jobs)))
    return [
        chunk.astype(np.intp, copy=False)
        for chunk in np.array_split(np.arange(n_features, dtype=np.intp), n_workers)
        if chunk.size
    ]


def _flatten_feature_results(
    result_chunks: list[list[tuple[int, float, float]]],
    n_features: int,
) -> list[tuple[float, float]]:
    results: list[tuple[float, float] | None] = [None] * n_features
    for chunk in result_chunks:
        for feat_idx, mean, std in chunk:
            results[feat_idx] = (mean, std)

    if any(result is None for result in results):
        raise RuntimeError("Permutation importance did not return every feature result")
    return [result for result in results if result is not None]


def _score(
    model,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable,
) -> float:
    if callable(scoring):
        y_pred = model.predict(X)
        return float(scoring(y, y_pred, w))
    return get_scoring(scoring)(model, X, y, w)


__all__ = ["permutation_importance"]
