"""Permutation utilities for importance and Boruta."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

PermutationMethod = Literal["auto", "global", "within_group", "block", "circular_shift"]
PermutationAxis = Literal["columns", "rows"]


def resolve_permutation_method(
    method: PermutationMethod,
    *,
    groups: np.ndarray | None,
    time: np.ndarray | None,
) -> PermutationMethod:
    """Resolve 'auto' to concrete permutation method."""
    if method != "auto":
        return method
    if groups is not None and time is not None:
        return "circular_shift"
    if groups is not None:
        return "within_group"
    return "global"


def build_group_info(
    groups: np.ndarray,
    time: np.ndarray | None = None,
) -> dict[Any, np.ndarray]:
    """
    Build mapping from group value to row indices.

    If time is provided, indices are sorted by time within each group.

    Returns
    -------
    dict mapping group_value -> array of row indices
    """
    groups = np.asarray(groups).reshape(-1)
    if time is not None:
        time = np.asarray(time).reshape(-1)
        if time.shape[0] != groups.shape[0]:
            raise ValueError("time and groups must have same length")

    uniq, inv = np.unique(groups, return_inverse=True)

    if time is None:
        order = np.argsort(inv, kind="mergesort")
    else:
        order = np.lexsort((time, inv))

    inv_sorted = inv[order]
    cuts = np.flatnonzero(np.diff(inv_sorted)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [inv_sorted.size]])

    info: dict[Any, np.ndarray] = {}
    for s, e in zip(starts, ends):
        g_val = uniq[inv_sorted[s]]
        info[g_val] = order[s:e]
    return info


def permute_array(
    x: np.ndarray,
    *,
    method: PermutationMethod,
    group_info: dict | None,
    block_size: int | str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute a 1D array using the specified method.

    Parameters
    ----------
    x : array of shape (n,)
    method : permutation method
    group_info : output of build_group_info, required for non-global methods
    block_size : block size for 'block' method, or 'auto' for sqrt(n)
    rng : numpy random generator

    Returns
    -------
    Permuted copy of x
    """
    if method == "global":
        return rng.permutation(x)

    if group_info is None:
        raise ValueError(f"method='{method}' requires group_info")

    out = x.copy()

    if method == "within_group":
        for idx in group_info.values():
            out[idx] = rng.permutation(x[idx])
        return out

    if method == "circular_shift":
        for idx in group_info.values():
            n = len(idx)
            if n <= 1:
                continue
            shift = int(rng.integers(1, n))
            out[idx] = np.roll(out[idx], shift)
        return out

    if method == "block":
        for idx in group_info.values():
            n = len(idx)
            if n <= 1:
                continue
            bs = int(np.sqrt(n)) if block_size == "auto" else int(block_size)
            bs = max(1, min(bs, n))
            n_blocks = max(1, int(np.ceil(n / bs)))
            blocks = [idx[i * bs : (i + 1) * bs] for i in range(n_blocks)]
            rng.shuffle(blocks)
            new_order = np.concatenate(blocks)
            out[idx] = x[new_order]
        return out

    raise ValueError(f"Unknown permutation method: {method}")


def permute_rows(
    X: np.ndarray,
    *,
    method: PermutationMethod,
    group_info: dict[Any, np.ndarray] | None,
    block_size: int | str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute rows of a 2D matrix (same permutation for all columns).

    Preserves cross-feature covariance - recommended for time series.
    """
    if X.ndim != 2:
        raise ValueError("permute_rows expects a 2D array")

    n = X.shape[0]

    if method == "global":
        return X[rng.permutation(n)]

    if group_info is None:
        raise ValueError(f"method='{method}' requires group_info")

    out = X.copy()

    if method == "within_group":
        for idx in group_info.values():
            if idx.size <= 1:
                continue
            perm_idx = rng.permutation(idx)
            out[idx, :] = X[perm_idx, :]
        return out

    if method == "circular_shift":
        for idx in group_info.values():
            n_g = len(idx)
            if n_g <= 1:
                continue
            shift = int(rng.integers(1, n_g))
            out[idx, :] = np.roll(X[idx, :], shift, axis=0)
        return out

    if method == "block":
        for idx in group_info.values():
            n_g = len(idx)
            if n_g <= 1:
                continue
            bs = int(np.sqrt(n_g)) if block_size == "auto" else int(block_size)
            bs = max(1, min(bs, n_g))
            n_blocks = int(np.ceil(n_g / bs))
            blocks = [idx[i * bs : (i + 1) * bs] for i in range(n_blocks)]
            rng.shuffle(blocks)
            new_order = np.concatenate(blocks)
            out[idx, :] = X[new_order, :]
        return out

    raise ValueError(f"Unknown permutation method: {method}")


def permute_matrix(
    X: np.ndarray,
    *,
    method: PermutationMethod,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    group_info: dict | None = None,
    block_size: int | str,
    seed: int,
    axis: PermutationAxis = "columns",
) -> np.ndarray:
    """
    Permute a 2D matrix column-wise or row-wise.

    Parameters
    ----------
    X : array of shape (n, p)
    method : permutation method (should already be resolved from 'auto')
    groups : group labels, required for non-global methods (ignored if group_info provided)
    time : time values for ordering within groups (ignored if group_info provided)
    group_info : precomputed output of build_group_info (optional, for efficiency)
    block_size : block size for 'block' method
    seed : random seed
    axis : {"columns", "rows"}
        - "columns": permute each column independently (classic Boruta)
        - "rows": same row permutation for all columns (preserves cross-feature covariance)

    Returns
    -------
    Permuted copy of X with shape (n, p)
    """
    rng = np.random.default_rng(seed)
    _, p = X.shape

    if group_info is None and method in ("within_group", "block", "circular_shift"):
        if groups is None:
            raise ValueError(f"method='{method}' requires groups or group_info")
        if method in ("block", "circular_shift") and time is None:
            raise ValueError(f"method='{method}' requires time")
        group_info = build_group_info(groups, time)

    if axis == "rows":
        return permute_rows(
            X, method=method, group_info=group_info, block_size=block_size, rng=rng
        )

    out = np.empty_like(X)
    for j in range(p):
        out[:, j] = permute_array(
            X[:, j],
            method=method,
            group_info=group_info,
            block_size=block_size,
            rng=rng,
        )
    return out


__all__ = [
    "PermutationMethod",
    "PermutationAxis",
    "resolve_permutation_method",
    "build_group_info",
    "permute_array",
    "permute_rows",
    "permute_matrix",
]
