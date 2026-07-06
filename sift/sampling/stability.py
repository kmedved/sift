"""Bootstrap split generators for stability selection."""

from __future__ import annotations

from typing import Iterator
import warnings

import numpy as np


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
    """
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

        for g, sorted_idx in group_data.items():
            n_g = len(sorted_idx)
            if n_g == 0:
                continue

            bs = int(np.sqrt(n_g)) if block_size == "auto" else min(block_size, n_g)
            bs = max(1, bs)

            if block_method == "moving":
                in_bag = _moving_block_sample(sorted_idx, bs, n_g, rng)
            elif block_method == "circular":
                in_bag = _circular_block_sample(sorted_idx, bs, n_g, rng)
            elif block_method == "stationary":
                in_bag = _stationary_block_sample(sorted_idx, bs, n_g, rng)
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


def _moving_block_sample(sorted_idx: np.ndarray, block_size: int, n: int, rng) -> list[int]:
    n_blocks = max(1, int(np.ceil(n / block_size)))
    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n - block_size + 1))
        result.extend(sorted_idx[start:start + block_size].tolist())
    return result


def _circular_block_sample(sorted_idx: np.ndarray, block_size: int, n: int, rng) -> list[int]:
    n_blocks = max(1, int(np.ceil(n / block_size)))
    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, n)
        indices = [(start + i) % n for i in range(block_size)]
        result.extend(sorted_idx[indices].tolist())
    return result


def _stationary_block_sample(sorted_idx: np.ndarray, mean_block_size: int, n: int, rng) -> list[int]:
    result = []
    p = 1.0 / max(1, mean_block_size)
    while len(result) < n:
        start = rng.integers(0, n)
        length = int(rng.geometric(p))
        indices = [(start + i) % n for i in range(length)]
        result.extend(sorted_idx[indices].tolist())
    return result[:n]
