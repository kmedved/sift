"""Bootstrap split generators for stability selection."""

from __future__ import annotations

from typing import Iterator
import warnings

import numpy as np
import pandas as pd


def _bootstrap_indices(
    n: int,
    n_bootstrap: int,
    sample_frac: float,
    y: np.ndarray | None = None,
    task: str = "regression",
    random_state: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    subsample_size = max(2, int(n * sample_frac))
    subsample_size = min(subsample_size, n)

    is_classification = task == "classification" and y is not None
    if is_classification:
        classes = np.unique(y)
        n_classes = len(classes)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        class_counts = np.array([len(class_indices[c]) for c in classes])
        if subsample_size < n_classes:
            subsample_size = n_classes

        def stratified_indices(rng_local):
            props = class_counts / class_counts.sum()
            raw = props * subsample_size
            counts = np.floor(raw).astype(int)
            counts = np.maximum(counts, 1)
            counts = np.minimum(counts, class_counts)

            total = counts.sum()
            frac = raw - np.floor(raw)

            if total < subsample_size:
                need = subsample_size - total
                room = class_counts - counts
                order = np.argsort(-frac)
                for j in order:
                    if need == 0:
                        break
                    if room[j] > 0:
                        add = min(room[j], need)
                        counts[j] += add
                        need -= add
            elif total > subsample_size:
                extra = total - subsample_size
                order = np.argsort(-counts)
                for j in order:
                    if extra == 0:
                        break
                    can_drop = counts[j] - 1
                    if can_drop > 0:
                        drop = min(can_drop, extra)
                        counts[j] -= drop
                        extra -= drop

            idx_list = [
                rng_local.choice(class_indices[c], size=counts[i], replace=False)
                for i, c in enumerate(classes)
                if counts[i] > 0
            ]
            return np.concatenate(idx_list)

    for _ in range(n_bootstrap):
        if is_classification:
            train_idx = stratified_indices(rng)
        else:
            train_idx = rng.choice(n, size=subsample_size, replace=False)

        in_bag = np.zeros(n, dtype=bool)
        in_bag[train_idx] = True
        val_idx = np.flatnonzero(~in_bag)
        yield train_idx.astype(np.int64), val_idx.astype(np.int64)


def _block_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    groups: np.ndarray,
    time: np.ndarray,
    block_size: int | str = "auto",
    block_method: str = "moving",
    y: np.ndarray | None = None,
    task: str = "regression",
    random_state: int | None = None,
    min_oob: int = 10,
    sample_frac: float = 1.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Block bootstrap respecting group/time structure.

    Parameters
    ----------
    groups : array
        Group labels (e.g., player_id).
    time : array
        Time values for ordering within groups.
    block_size : int or "auto"
        "auto" uses sqrt(n_per_group).
    block_method : str
        "moving", "circular", or "stationary"
    sample_frac : float
        Fraction of panel rows to draw per bootstrap. The rounded total draw
        budget is allocated across groups by largest remainder, with at least
        one draw per non-empty group when the budget permits. Draws are with
        replacement; duplicate rows are collapsed into weights by the
        stability fitter.
    """
    if not isinstance(sample_frac, (int, float, np.integer, np.floating)) or isinstance(
        sample_frac, (bool, np.bool_)
    ):
        raise TypeError("sample_frac must be a real number in (0, 1]")
    sample_frac = float(sample_frac)
    if not np.isfinite(sample_frac) or not 0.0 < sample_frac <= 1.0:
        raise ValueError("sample_frac must be finite and in (0, 1]")
    groups = np.asarray(groups).reshape(-1)
    time = np.asarray(time).reshape(-1)
    if groups.size != n or time.size != n:
        raise ValueError("groups and time must each have one value per row")
    if np.asarray(pd.isna(time), dtype=bool).any():
        raise ValueError("time values must not contain missing values")

    rng = np.random.default_rng(random_state)

    unique_groups, inverse_groups = np.unique(groups, return_inverse=True)
    order = np.lexsort((time, inverse_groups))
    sorted_inverse = inverse_groups[order]
    if len(order) == 0:
        group_data = {}
    else:
        starts = np.r_[0, np.flatnonzero(sorted_inverse[1:] != sorted_inverse[:-1]) + 1]
        stops = np.r_[starts[1:], len(order)]
        group_data = {
            unique_groups[int(sorted_inverse[start])]: order[start:stop]
            for start, stop in zip(starts, stops)
        }

    classes = set(np.unique(y)) if task == "classification" and y is not None else None

    valid = 0
    attempts = 0
    max_attempts = n_bootstrap * 10

    while valid < n_bootstrap and attempts < max_attempts:
        attempts += 1
        train_idx = []
        val_idx = []

        group_sizes = [len(idx) for idx in group_data.values()]
        target_total = max(1, min(n, int(np.floor(sample_frac * n + 0.5))))
        target_sizes = _allocate_group_draws(group_sizes, target_total, sample_frac)

        for group_pos, (g, sorted_idx) in enumerate(group_data.items()):
            n_g = len(sorted_idx)
            if n_g == 0:
                continue

            bs = int(np.sqrt(n_g)) if block_size == "auto" else min(block_size, n_g)
            bs = max(1, bs)
            target_n = int(target_sizes[group_pos])

            if block_method == "moving":
                in_bag = _moving_block_sample(sorted_idx, bs, target_n, rng)
            elif block_method == "circular":
                in_bag = _circular_block_sample(sorted_idx, bs, target_n, rng)
            elif block_method == "stationary":
                in_bag = _stationary_block_sample(sorted_idx, bs, target_n, rng)
            else:
                raise ValueError(f"Unknown block_method: {block_method}")

            in_bag_arr = np.asarray(in_bag, dtype=np.int64)
            oob = np.setdiff1d(sorted_idx, np.unique(in_bag_arr), assume_unique=True)

            train_idx.extend(in_bag_arr.tolist())
            val_idx.extend(oob.tolist())

        train_arr = np.array(train_idx, dtype=np.int64)
        val_arr = np.array(val_idx, dtype=np.int64)

        if len(val_arr) < min_oob:
            continue

        if classes is not None:
            if set(y[train_arr]) != classes or set(y[val_arr]) != classes:
                continue

        valid += 1
        yield train_arr, val_arr

    if valid < n_bootstrap:
        warnings.warn(f"Only generated {valid}/{n_bootstrap} valid block bootstrap splits.")


def _allocate_group_draws(
    group_sizes: list[int], target_total: int, sample_frac: float
) -> np.ndarray:
    """Allocate a rounded panel draw budget proportionally across groups."""
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if sizes.size == 0:
        return np.empty(0, dtype=np.int64)
    quotas = sizes.astype(np.float64) * float(sample_frac)
    targets = np.floor(quotas).astype(np.int64)
    target_total = int(min(int(sizes.sum()), max(1, target_total)))

    # Preserve representation of each group whenever the requested budget can
    # afford it. This matters for panel fits, while the tiny-budget case is
    # still handled coherently by largest remainder below.
    if target_total >= sizes.size:
        targets = np.maximum(targets, 1)
        targets = np.minimum(targets, sizes)

    def add_one(order):
        for pos in order:
            if int(targets[pos]) < int(sizes[pos]):
                targets[pos] += 1
                return True
        return False

    while int(targets.sum()) < target_total:
        remainder = quotas - targets
        if not add_one(np.argsort(-remainder, kind="mergesort")):
            break
    while int(targets.sum()) > target_total:
        # Remove from the least-deserving groups first, retaining one row per
        # group when the budget can cover all groups.
        floor = np.ones_like(targets) if target_total >= sizes.size else np.zeros_like(targets)
        removable = np.flatnonzero(targets > floor)
        if removable.size == 0:
            break
        pos = removable[np.argmin(quotas[removable] - targets[removable])]
        targets[pos] -= 1
    return targets.astype(np.int64, copy=False)


def _moving_block_sample(sorted_idx: np.ndarray, block_size: int, n: int, rng) -> list[int]:
    n = int(n)
    if n <= 0:
        return []
    source_n = len(sorted_idx)
    n_blocks = max(1, int(np.ceil(n / block_size)))
    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, source_n - block_size + 1))
        result.extend(sorted_idx[start:start + block_size].tolist())
    return result[:n]


def _circular_block_sample(sorted_idx: np.ndarray, block_size: int, n: int, rng) -> list[int]:
    n = int(n)
    if n <= 0:
        return []
    source_n = len(sorted_idx)
    n_blocks = max(1, int(np.ceil(n / block_size)))
    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, source_n)
        indices = [(start + i) % source_n for i in range(block_size)]
        result.extend(sorted_idx[indices].tolist())
    return result[:n]


def _stationary_block_sample(sorted_idx: np.ndarray, mean_block_size: int, n: int, rng) -> list[int]:
    n = int(n)
    if n <= 0:
        return []
    result = []
    p = 1.0 / max(1, mean_block_size)
    source_n = len(sorted_idx)
    while len(result) < n:
        start = rng.integers(0, source_n)
        length = int(rng.geometric(p))
        indices = [(start + i) % source_n for i in range(length)]
        result.extend(sorted_idx[indices].tolist())
    return result[:n]
