"""Weighted panel within-transforms applied before filter ranks.

``within="groups"`` subtracts per-entity weighted means of ``X`` and ``y``.
``within="two_way"`` alternates entity and time demeaning for a fixed
iteration count (``TWO_WAY_ITERATIONS``). Unseen entity ids at transform
time fall back to the training grand mean; unseen time ids add no extra
time effect (time effects are residual after entity demeaning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sift._preprocess import reject_datetime_like_features

WithinMode = Literal["groups", "two_way"]

TWO_WAY_ITERATIONS = 5
_VALID_WITHIN = frozenset({"groups", "two_way"})


def validate_within(within: str | None) -> str | None:
    """Return a canonical within mode or ``None``."""
    if within is None:
        return None
    if isinstance(within, (bool, np.bool_)) or not isinstance(within, str):
        raise ValueError("within must be None, 'groups', or 'two_way'")
    if within not in _VALID_WITHIN:
        raise ValueError("within must be None, 'groups', or 'two_way'")
    return within


def require_within_context(
    within: str | None,
    *,
    task: str | None = None,
    groups=None,
    time=None,
    X=None,
) -> str | None:
    """Validate a public within option before any path or scoring work."""
    resolved = validate_within(within)
    if resolved is None:
        return None
    if task is not None and task != "regression":
        raise ValueError("within is only supported for task='regression'")
    if groups is None:
        raise ValueError(f"within={resolved!r} requires groups")
    if resolved == "two_way" and time is None:
        raise ValueError("within='two_way' requires groups and time")
    if X is not None:
        reject_datetime_like_features(X)
    return resolved


def _require_finite_xy(X: np.ndarray, y: np.ndarray) -> None:
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(
            "within demeaning requires finite X and y; impute missing values first"
        )


def _positive_weights(sample_weight: np.ndarray, n_rows: int) -> np.ndarray:
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != n_rows:
        raise ValueError("sample_weight length must match X rows")
    if not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("sample_weight must be finite and non-negative")
    if float(w.sum()) <= 0.0:
        raise ValueError("sample_weight must sum to > 0")
    return w


def _factorize(ids: np.ndarray, *, label: str) -> tuple[pd.Index, np.ndarray]:
    ids_arr = np.asarray(ids).reshape(-1)
    codes, uniques = pd.factorize(ids_arr, sort=False)
    codes = np.asarray(codes, dtype=np.int64)
    if np.any(codes < 0):
        raise ValueError(f"{label} used with within must not contain missing values")
    return pd.Index(uniques), codes


def _weighted_mean_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return np.asarray([_weighted_mean_scalar(values, weights)], dtype=np.float64)
    w_sum = float(weights.sum())
    anchor = np.array(values[0], dtype=np.float64, copy=True)
    return anchor + (weights @ (values - anchor)) / w_sum


def _weighted_mean_scalar(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    w_sum = float(weights.sum())
    anchor = float(values[0])
    return anchor + float(weights @ (values - anchor) / w_sum)


def _level_means(
    values: np.ndarray,
    codes: np.ndarray,
    n_levels: int,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values.reshape(-1, 1)
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    wsum = np.bincount(codes, weights=weights, minlength=n_levels).astype(np.float64)
    first = np.full(n_levels, values.shape[0], dtype=np.int64)
    np.minimum.at(first, codes, np.arange(codes.size, dtype=np.int64))
    seen = first < values.shape[0]
    anchors = np.zeros((n_levels, values.shape[1]), dtype=np.float64)
    if np.any(seen):
        anchors[seen] = values[first[seen]]
    centered = values - anchors[codes]
    means = np.empty((n_levels, values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        means[:, j] = np.bincount(
            codes, weights=weights * centered[:, j], minlength=n_levels
        )
    positive = wsum > 0.0
    means[positive] /= wsum[positive, None]
    means[positive] += anchors[positive]
    if np.any(~positive):
        grand = _weighted_mean_rows(values, weights)
        means[~positive] = grand
    if squeeze:
        return means[:, 0]
    return means


@dataclass(frozen=True)
class WithinTransform:
    """Fitted within-demeaning map, reusable on validation rows."""

    mode: WithinMode
    group_index: pd.Index
    group_effects_X: np.ndarray
    group_effects_y: np.ndarray
    grand_mean_X: np.ndarray
    grand_mean_y: float
    time_index: pd.Index | None = None
    time_effects_X: np.ndarray | None = None
    time_effects_y: np.ndarray | None = None
    n_iterations: int = 1

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        time: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        else:
            X_arr = np.array(X_arr, dtype=np.float64, copy=True)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1).copy()
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X_arr.shape[1] != self.grand_mean_X.shape[0]:
            raise ValueError("X column count does not match the fitted within transform")
        _require_finite_xy(X_arr, y_arr)
        g_codes = self.group_index.get_indexer(np.asarray(groups).reshape(-1))
        if g_codes.shape[0] != X_arr.shape[0]:
            raise ValueError("groups length must match X rows")
        seen_g = g_codes >= 0
        X_out = X_arr
        y_out = y_arr
        if np.any(seen_g):
            X_out[seen_g] -= self.group_effects_X[g_codes[seen_g]]
            y_out[seen_g] -= self.group_effects_y[g_codes[seen_g]]
        if np.any(~seen_g):
            X_out[~seen_g] -= self.grand_mean_X
            y_out[~seen_g] -= self.grand_mean_y
        if self.mode == "two_way":
            if time is None:
                raise ValueError("within='two_way' requires time")
            if self.time_index is None or self.time_effects_X is None or self.time_effects_y is None:
                raise RuntimeError("two-way within transform is missing time effects")
            t_codes = self.time_index.get_indexer(np.asarray(time).reshape(-1))
            if t_codes.shape[0] != X_arr.shape[0]:
                raise ValueError("time length must match X rows")
            seen_t = t_codes >= 0
            if np.any(seen_t):
                X_out[seen_t] -= self.time_effects_X[t_codes[seen_t]]
                y_out[seen_t] -= self.time_effects_y[t_codes[seen_t]]
        return X_out, y_out


def fit_within_transform(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray | None,
    sample_weight: np.ndarray,
) -> WithinTransform:
    """Fit demeaning parameters on training rows only."""
    resolved = validate_within(mode)
    if resolved is None:
        raise ValueError("fit_within_transform requires within='groups' or 'two_way'")
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    else:
        X_arr = np.array(X_arr, dtype=np.float64, copy=True)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1).copy()
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    _require_finite_xy(X_arr, y_arr)
    w = _positive_weights(sample_weight, X_arr.shape[0])
    fit_mask = w > 0.0
    if not np.any(fit_mask):
        raise ValueError("within demeaning requires at least one positive-weight row")
    X_fit = X_arr[fit_mask]
    y_fit = y_arr[fit_mask]
    w_fit = w[fit_mask]
    groups_fit = np.asarray(groups).reshape(-1)[fit_mask]
    grand_X = _weighted_mean_rows(X_fit, w_fit)
    grand_y = _weighted_mean_scalar(y_fit, w_fit)
    group_index, g_codes = _factorize(groups_fit, label="groups")
    if resolved == "groups":
        group_X = _level_means(X_fit, g_codes, len(group_index), w_fit)
        group_y = _level_means(y_fit, g_codes, len(group_index), w_fit)
        return WithinTransform(
            mode="groups",
            group_index=group_index,
            group_effects_X=np.ascontiguousarray(group_X, dtype=np.float64),
            group_effects_y=np.ascontiguousarray(group_y, dtype=np.float64),
            grand_mean_X=np.ascontiguousarray(grand_X, dtype=np.float64),
            grand_mean_y=float(grand_y),
            n_iterations=1,
        )
    if time is None:
        raise ValueError("within='two_way' requires time")
    time_fit = np.asarray(time).reshape(-1)[fit_mask]
    time_index, t_codes = _factorize(time_fit, label="time")
    X_work = np.array(X_fit, dtype=np.float64, copy=True)
    y_work = np.array(y_fit, dtype=np.float64, copy=True)
    group_X = np.zeros((len(group_index), X_work.shape[1]), dtype=np.float64)
    group_y = np.zeros(len(group_index), dtype=np.float64)
    time_X = np.zeros((len(time_index), X_work.shape[1]), dtype=np.float64)
    time_y = np.zeros(len(time_index), dtype=np.float64)
    for _ in range(TWO_WAY_ITERATIONS):
        gX = _level_means(X_work, g_codes, len(group_index), w_fit)
        gY = _level_means(y_work, g_codes, len(group_index), w_fit)
        X_work -= gX[g_codes]
        y_work -= gY[g_codes]
        group_X += gX
        group_y += gY
        tX = _level_means(X_work, t_codes, len(time_index), w_fit)
        tY = _level_means(y_work, t_codes, len(time_index), w_fit)
        X_work -= tX[t_codes]
        y_work -= tY[t_codes]
        time_X += tX
        time_y += tY
    return WithinTransform(
        mode="two_way",
        group_index=group_index,
        group_effects_X=np.ascontiguousarray(group_X, dtype=np.float64),
        group_effects_y=np.ascontiguousarray(group_y, dtype=np.float64),
        grand_mean_X=np.ascontiguousarray(grand_X, dtype=np.float64),
        grand_mean_y=float(grand_y),
        time_index=time_index,
        time_effects_X=np.ascontiguousarray(time_X, dtype=np.float64),
        time_effects_y=np.ascontiguousarray(time_y, dtype=np.float64),
        n_iterations=TWO_WAY_ITERATIONS,
    )


def fit_transform_within(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    time: np.ndarray | None,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, WithinTransform]:
    """Fit on these rows and return the demeaned training matrices."""
    fitted = fit_within_transform(mode, X, y, groups, time, sample_weight)
    X_out, y_out = fitted.transform(X, y, groups, time)
    return X_out, y_out, fitted


def restore_feature_matrix(template, values: np.ndarray):
    """Re-wrap a demeaned array as the caller's DataFrame or ndarray."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    return values


def as_float_feature_matrix(X) -> tuple[np.ndarray, object]:
    """Return a finite-capable float64 copy plus the original container."""
    reject_datetime_like_features(X)
    if isinstance(X, pd.DataFrame):
        try:
            values = X.to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "within requires a numeric feature matrix after encoding"
            ) from exc
        return values, X
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    return np.array(values, dtype=np.float64, copy=True), X


def group_level_design(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse rows to one weighted-mean observation per entity.

    Empty-mass entities are dropped. This is the between-entity table used
    for ``between_relevance`` under both ``groups`` and ``two_way``.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    w = _positive_weights(sample_weight, X_arr.shape[0])
    mask = w > 0.0
    X_arr = X_arr[mask]
    y_arr = y_arr[mask]
    w = w[mask]
    group_index, codes = _factorize(np.asarray(groups).reshape(-1)[mask], label="groups")
    wsum = np.bincount(codes, weights=w, minlength=len(group_index)).astype(np.float64)
    keep = wsum > 0.0
    X_g = _level_means(X_arr, codes, len(group_index), w)[keep]
    y_g = _level_means(y_arr, codes, len(group_index), w)[keep]
    w_g = wsum[keep]
    return (
        np.ascontiguousarray(X_g, dtype=np.float64),
        np.ascontiguousarray(y_g, dtype=np.float64),
        np.ascontiguousarray(w_g, dtype=np.float64),
    )
