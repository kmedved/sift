"""Weighted panel within-transforms applied before filter ranks.

``within="groups"`` subtracts per-entity weighted means of ``X`` and ``y``.
``within="two_way"`` alternates entity and time demeaning until the largest
weighted entity-mean and time-mean residual, scaled by the column's weighted
standard deviation, drops below ``TWO_WAY_TOLERANCE`` (cap
``TWO_WAY_MAX_ITERATIONS`` passes). Balanced, unweighted panels reproduce the
closed form ``x - mean_i - mean_t + grand`` in very few passes; unbalanced or
weighted panels keep sweeping until the projection converges, so the result
no longer depends on the sweep order. ``n_iterations`` reports the passes
actually used, ``converged`` whether the tolerance was met, and
``max_residual`` the final scaled residual. Unseen entity ids at transform
time fall back to the training grand mean; unseen time ids add no extra time
effect (time effects are residual after entity demeaning). The transform
itself requires finite numeric ``X`` and ``y``; callers using paths with their
own preprocessing should make the path's missing-data policy explicit. In
particular, classic filter entry points may impute feature values before this
helper, whereas the helper itself never imputes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from sift._preprocess import reject_datetime_like_features

WithinMode = Literal["groups", "two_way"]

#: Scaled residual below which the alternating projection is called converged.
TWO_WAY_TOLERANCE = 1e-10
#: Hard cap on alternating passes; hitting it emits a ``UserWarning``.
TWO_WAY_MAX_ITERATIONS = 200
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
    bad_X = not np.isfinite(X).all()
    bad_y = not np.isfinite(y).all()
    if not (bad_X or bad_y):
        return
    if bad_X and bad_y:
        offender = "X and y contain"
    elif bad_X:
        offender = "X contains"
    else:
        offender = "y contains"
    raise ValueError(
        f"within demeaning requires finite X and y, but {offender} NaN or "
        "infinite values; this path never imputes. Impute or drop the "
        "non-finite rows before selecting (classic estimators mean-impute "
        "features automatically before demeaning, so estimator='classic' "
        "accepts missing X; missing y is never imputed)"
    )


#: (k_method, strategy) pairs whose splits can never leave a within level
#: seen in training, keyed by the within mode they break.
_IMPOSSIBLE_WITHIN_SPLITS: dict[str, dict[str, str]] = {
    "groups": {
        "group_cv": (
            "strategy='group_cv' holds out whole entities, so no validation "
            "entity is ever seen in the training fold"
        ),
    },
    "two_way": {
        "group_cv": (
            "strategy='group_cv' holds out whole entities, so no validation "
            "entity is ever seen in the training fold"
        ),
        "time_holdout": (
            "strategy='time_holdout' puts every validation period after the "
            "split, so no validation time level is ever seen in the training "
            "fold"
        ),
    },
}


def within_split_guidance(mode: str) -> str:
    """Name the auto-k combinations that can satisfy the within guard.

    Kept in one place so the up-front rejection, the fold guard and the
    ``AutoKConfig`` validator all quote the same working routes.
    """
    if mode == "two_way":
        return (
            "within='two_way' scores only under k_method='gaussian_cv' or "
            "'xfit_objective' with strategy='kfold', which keeps entity and "
            "time levels on both sides of every split"
        )
    return (
        "within='groups' scores under k_method='gaussian_cv' or "
        "'xfit_objective' with strategy='kfold', or under k_method='evaluate' "
        "with strategy='time_holdout' when entities persist across the "
        "holdout boundary"
    )


def reject_impossible_within_split(
    within: str | None,
    *,
    k_method: str,
    strategy: str,
) -> None:
    """Reject split schemes that can never satisfy the within guard.

    ``group_cv`` holds out whole entities and ``time_holdout`` holds out whole
    periods, so those splits leave the demeaned dimension with no overlap by
    construction.  Raising here keeps the user from paying for a full feature
    path before the fold guard fails.
    """
    if within is None:
        return
    reason = _IMPOSSIBLE_WITHIN_SPLITS.get(str(within), {}).get(str(strategy))
    if reason is None:
        return
    raise ValueError(
        f"within={within!r} cannot be validated with k_method={k_method!r} and "
        f"strategy={strategy!r}: {reason}. {within_split_guidance(str(within))}"
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


@dataclass
class UnseenWithinLevelTally:
    """Running count of validation rows whose within level was unseen.

    One tally spans a whole auto-k call so the partial-overlap warning is
    emitted once, not once per fold.
    """

    mode: str | None = None
    n_rows: int = 0
    entity_unseen: int = 0
    time_unseen: int = 0
    _dimensions: list[str] = field(default_factory=list)

    def add(self, *, mode: str, n_rows: int, entity_unseen: int, time_unseen: int) -> None:
        self.mode = mode
        self.n_rows += int(n_rows)
        self.entity_unseen += int(entity_unseen)
        self.time_unseen += int(time_unseen)


def _unseen_clause(dimension: str, unseen: int, n_rows: int) -> str:
    fraction = float(unseen) / float(n_rows) if n_rows else 0.0
    return f"{unseen} of {n_rows} validation rows ({fraction:.1%}) had an unseen {dimension} level"


def warn_unseen_within_validation_levels(tally: "UnseenWithinLevelTally | None") -> None:
    """Emit one warning per auto-k call for partially unseen validation levels.

    ``require_seen_within_validation_levels`` only rejects folds in which *no*
    level overlaps training. Rows with an unseen entity use the training grand
    mean for that effect; an unseen time level contributes no time effect.
    Effects from a seen level in the other dimension still apply.
    """
    if tally is None or tally.n_rows <= 0:
        return
    clauses = []
    if tally.entity_unseen:
        clauses.append(_unseen_clause("entity", tally.entity_unseen, tally.n_rows))
    if tally.time_unseen:
        clauses.append(_unseen_clause("time", tally.time_unseen, tally.n_rows))
    if not clauses:
        return
    mode = tally.mode or "groups"
    missing_effects = []
    if tally.entity_unseen:
        missing_effects.append("the unavailable entity effect fell back to the training grand mean")
    if tally.time_unseen:
        missing_effects.append("the unavailable time effect was omitted")
    other_effect = (
        "A fitted effect in the other dimension still applies when its level is seen. "
        if mode == "two_way" else ""
    )
    warnings.warn(
        f"within={mode!r} auto-k scoring: {' and '.join(clauses)}; "
        f"{' and '.join(missing_effects)}. {other_effect}The chosen k "
        "uses a mixture of fully and partially demeaned rows. Drop or "
        "otherwise handle late-entering or early-exiting entities, or choose a "
        f"split that keeps levels overlapping: {within_split_guidance(mode)}",
        UserWarning,
        stacklevel=3,
    )


def require_seen_within_validation_levels(
    fitted: "WithinTransform",
    groups: np.ndarray,
    time: np.ndarray | None = None,
    *,
    tally: "UnseenWithinLevelTally | None" = None,
) -> None:
    """Require each scored demeaned dimension to overlap training levels.

    This guard is for fold-based validation only.  ``WithinTransform.transform``
    intentionally keeps its causal fallback for ordinary transforms of rows
    containing unseen entity or time ids.  Partial overlap still scores; pass a
    ``tally`` to collect the fallback row counts and report them once through
    ``warn_unseen_within_validation_levels``.
    """
    group_codes = fitted.group_index.get_indexer(np.asarray(groups).reshape(-1))
    if not np.any(group_codes >= 0):
        raise ValueError(
            "within validation requires at least one entity level seen in the "
            "training fold; no validation entity can be demeaned from training "
            f"effects. {within_split_guidance(fitted.mode)}, or omit within when "
            "validating between-entity effects"
        )
    time_codes = None
    if fitted.mode == "two_way":
        if fitted.time_index is None:
            raise RuntimeError("two-way within transform is missing time effects")
        if time is None:
            raise ValueError("within='two_way' requires validation time")
        time_codes = fitted.time_index.get_indexer(np.asarray(time).reshape(-1))
        if not np.any(time_codes >= 0):
            raise ValueError(
                "within validation requires at least one time level seen in the "
                "training fold; no validation time can be demeaned from training "
                f"effects. {within_split_guidance(fitted.mode)}, or omit within "
                "when validating between-time effects"
            )
    if tally is not None:
        tally.add(
            mode=fitted.mode,
            n_rows=int(group_codes.shape[0]),
            entity_unseen=int(np.count_nonzero(group_codes < 0)),
            time_unseen=0 if time_codes is None else int(np.count_nonzero(time_codes < 0)),
        )


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


def _weighted_sd_columns(
    values: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
) -> np.ndarray:
    """Per-column weighted standard deviation, with zero-sd columns set to 1."""
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.asarray(means, dtype=np.float64).reshape(1, -1)
    w_sum = float(np.asarray(weights, dtype=np.float64).sum())
    variance = (np.asarray(weights, dtype=np.float64) @ (centered * centered)) / w_sum
    sd = np.sqrt(np.maximum(variance, 0.0))
    # A constant column has no scale to measure a residual against; its level
    # means are already exactly zero, so any positive divisor works.
    return np.where(sd > 0.0, sd, 1.0)


def _scaled_max(level_means: np.ndarray, scale: np.ndarray) -> float:
    """Largest absolute level mean measured in column standard deviations."""
    if level_means.size == 0:
        return 0.0
    return float(np.max(np.abs(level_means) / scale.reshape(1, -1)))


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
    converged: bool = True
    max_residual: float = 0.0

    def transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        time: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply fitted effects without estimating validation-level means.

        ``X`` and ``y`` must be finite at this layer.  Unknown levels use the
        causal training fallbacks described in the module documentation.
        """
        # The 1-D reshape below can be a view of a caller-owned array.  The
        # demeaning operations are in-place, so always detach before them.
        X_arr = np.array(X, dtype=np.float64, copy=True)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
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
    """Fit finite-input demeaning parameters on training rows only.

    ``within="two_way"`` alternates entity and time demeaning until the
    largest weighted level-mean residual, scaled by each column's weighted
    standard deviation, falls below ``TWO_WAY_TOLERANCE``, capped at
    ``TWO_WAY_MAX_ITERATIONS`` passes.  Balanced, unweighted panels reach the
    closed form in a couple of passes; unbalanced or weighted panels keep
    sweeping, so the fitted effects no longer depend on the sweep order.
    """
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
    scale_X = _weighted_sd_columns(X_fit, w_fit, grand_X)
    scale_y = _weighted_sd_columns(
        y_fit.reshape(-1, 1), w_fit, np.asarray([grand_y], dtype=np.float64)
    )
    group_X = np.zeros((len(group_index), X_work.shape[1]), dtype=np.float64)
    group_y = np.zeros(len(group_index), dtype=np.float64)
    time_X = np.zeros((len(time_index), X_work.shape[1]), dtype=np.float64)
    time_y = np.zeros(len(time_index), dtype=np.float64)
    residual = np.inf
    n_iterations = 0
    for n_iterations in range(1, TWO_WAY_MAX_ITERATIONS + 1):
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
        residual = max(
            _scaled_max(gX, scale_X),
            _scaled_max(tX, scale_X),
            _scaled_max(gY.reshape(-1, 1), scale_y),
            _scaled_max(tY.reshape(-1, 1), scale_y),
        )
        if residual < TWO_WAY_TOLERANCE:
            break
    converged = bool(residual < TWO_WAY_TOLERANCE)
    if not converged:
        warnings.warn(
            f"within='two_way' demeaning stopped at the {TWO_WAY_MAX_ITERATIONS}-pass "
            f"cap with a scaled level-mean residual of {residual:.3e}, above the "
            f"{TWO_WAY_TOLERANCE:.0e} tolerance; the entity and time effects are not "
            "fully separated. This usually means the panel splits into weakly "
            "connected entity/time components -- check for entities or periods that "
            "barely overlap the rest of the panel, or use within='groups'",
            UserWarning,
            stacklevel=2,
        )
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
        n_iterations=int(n_iterations),
        converged=converged,
        max_residual=float(residual),
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
