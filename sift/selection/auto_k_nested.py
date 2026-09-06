"""Nested selector-path auto-k evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from sift._preprocess import ensure_weights
from sift.selection.auto_k import (
    AutoKConfig,
    choose_k_from_score_curve,
    validate_auto_k_config,
    with_effective_k_bounds,
)
from sift.selection.auto_k_core import (
    build_k_grid,
    build_score_curve_diagnostics,
    evaluate_numeric_prefixes,
    resolve_metric,
    split_weights,
    time_holdout_split,
)
from sift.scoring import is_sklearn_scorer, sklearn_scorer_label


@dataclass(frozen=True)
class NestedAutoKFold:
    """Selector path and validation matrix for one nested auto-k fold."""

    train_path: Any
    val_path: Any
    feature_path: list[str]
    prefix_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class NestedAutoKResult:
    """Result from nested auto-k evaluation."""

    selected_k: int
    diagnostics: dict[str, Any]


NestedPathBuilder = Callable[[np.ndarray, np.ndarray, int], NestedAutoKFold]


def _build_nested_splits(
    X,
    y_arr: np.ndarray,
    *,
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    config: AutoKConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build nested train/validation splits according to AutoKConfig."""
    n = len(y_arr)
    if config.strategy == "time_holdout":
        if time is None:
            raise ValueError("auto_k_mode='nested' with time_holdout requires time")
        time_arr = np.asarray(time).reshape(-1)
        if len(time_arr) != n:
            raise ValueError(f"time has {len(time_arr)} rows but X/y have {n}")
        return [time_holdout_split(time_arr, config.val_frac)]

    if config.strategy == "group_cv":
        if groups is None:
            raise ValueError("auto_k_mode='nested' with group_cv requires groups")
        group_arr = np.asarray(groups).reshape(-1)
        if len(group_arr) != n:
            raise ValueError(f"groups has {len(group_arr)} rows but X/y have {n}")
        n_unique = len(np.unique(group_arr))
        n_splits = min(config.n_splits, n_unique)
        if n_splits < 2:
            raise ValueError(f"group_cv requires at least 2 groups, got {n_unique}")
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y_arr, group_arr))

    raise ValueError(f"Unknown auto_k strategy: {config.strategy}")


def select_k_nested(
    X,
    y: np.ndarray,
    *,
    n_features: int,
    config: AutoKConfig,
    build_fold_path: NestedPathBuilder,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    task: Literal["regression", "classification"] = "regression",
) -> NestedAutoKResult:
    """Select k by fitting a train-only selector path inside each validation split."""
    validate_auto_k_config(config)
    if config.auto_k_mode != "nested":
        raise ValueError("select_k_nested requires auto_k_mode='nested'")
    if config.k_method != "evaluate":
        raise ValueError("auto_k_mode='nested' currently supports only k_method='evaluate'")
    if int(n_features) < 0:
        raise ValueError("n_features must be non-negative")

    y_arr = np.asarray(y).reshape(-1)
    probe_max = min(int(config.max_k), max(int(n_features), 1))
    metric = resolve_metric(config.metric, task)
    sample_weight_supplied = sample_weight is not None
    w_arr = ensure_weights(sample_weight, len(y_arr), normalize=True)
    splits = _build_nested_splits(X, y_arr, groups=groups, time=time, config=config)

    built_folds: list[tuple[int, np.ndarray, np.ndarray, NestedAutoKFold]] = []
    block_depths: list[int] = []
    uses_blocks = False
    for split_id, (train_idx, val_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        fold = build_fold_path(train_idx, val_idx, probe_max)
        built_folds.append((int(split_id), train_idx, val_idx, fold))
        if fold.prefix_sizes is not None:
            uses_blocks = True
            n_steps = sum(1 for width in fold.prefix_sizes if int(width) > 0)
            if n_steps:
                block_depths.append(int(n_steps))

    if uses_blocks:
        common = min(block_depths) if block_depths else 0
        max_k = min(int(config.max_k), int(common))
        min_k = max(1, min(int(config.min_k), max_k)) if max_k else 0
    else:
        if int(n_features) < 1:
            raise ValueError("X must contain at least one feature")
        max_k = min(int(config.max_k), int(n_features))
        min_k = max(1, min(int(config.min_k), max_k))
    k_grid = build_k_grid(min_k, max_k) if max_k else []

    all_scores = {k: [] for k in k_grid}
    fold_rows = []
    for split_id, train_idx, val_idx, fold in built_folds:
        w_train = split_weights(w_arr, train_idx, "train")
        w_val = split_weights(w_arr, val_idx, "validation")
        if fold.prefix_sizes is not None:
            sizes = [int(w) for w in fold.prefix_sizes if int(w) > 0]
            fold_max = min(max_k, len(sizes))
            fold_min = max(1, min(min_k, fold_max)) if fold_max else 0
            fold_grid = [k for k in k_grid if fold_min <= int(k) <= fold_max]
            k_grid_eval = [int(sizes[int(k) - 1]) for k in fold_grid]
        else:
            fold_grid = k_grid
            k_grid_eval = k_grid
        if not fold_grid:
            continue

        split_scores = evaluate_numeric_prefixes(
            fold.train_path,
            fold.val_path,
            y_arr[train_idx],
            y_arr[val_idx],
            w_train,
            w_val,
            task=task,
            metric=metric,
            k_grid=k_grid_eval,
            sample_weight_supplied=sample_weight_supplied,
        )
        if fold.prefix_sizes is not None:
            size_to_step = {int(sizes[int(k) - 1]): int(k) for k in fold_grid}
            split_scores = {
                size_to_step.get(int(raw), int(raw)): score
                for raw, score in split_scores.items()
            }

        for k, score in split_scores.items():
            all_scores.setdefault(int(k), []).append(score)
            if fold.prefix_sizes is not None and int(k) > 0:
                raw_k = int(fold.prefix_sizes[min(int(k), len(fold.prefix_sizes)) - 1])
            else:
                raw_k = int(k)
            fold_rows.append(
                {
                    "split": split_id,
                    "k": int(k),
                    "score": score,
                    "path": tuple(fold.feature_path[: min(raw_k, len(fold.feature_path))]),
                }
            )

    score_df = build_score_curve_diagnostics(k_grid, all_scores)
    if score_df.empty:
        selected_k = max_k
        score_best_k = None
    else:
        curve_config = with_effective_k_bounds(config, min_k=min_k, max_k=max_k)
        selected_k, score_df = choose_k_from_score_curve(
            score_df,
            curve_config,
            lower_is_better=True,
        )
        score_best_k = None if score_df.empty else int(score_df["best_k"].iloc[0])

    return NestedAutoKResult(
        selected_k=selected_k,
        diagnostics={
            "mode": "nested",
            "strategy": config.strategy,
            "metric": (
                sklearn_scorer_label(metric)
                if is_sklearn_scorer(metric)
                else str(metric)
            ),
            "selection_rule": config.selection_rule,
            "selection_rule_effective": (
                None
                if score_df.empty
                else str(score_df["selection_rule_effective"].iloc[0])
            ),
            "best_k": score_best_k,
            "selected_k": selected_k,
            "scores": score_df,
            "folds": pd.DataFrame(fold_rows),
        },
    )
