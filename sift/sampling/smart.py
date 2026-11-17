"""Pilot-residual importance sampling of training rows.

A cheap pilot model is fitted on a small random slice of the data and scores
every row by how wrong it is: the magnitude ``g`` of the loss gradient at the
pilot prediction.  Half of each row's inclusion probability is proportional to
``g`` and half is spread evenly (a defensive mixture; Hesterberg, 1995).  The
draw runs systematically along a random order of the rows, so it keeps an exact
size and cannot line up with a repeating pattern in the data, and every kept
row carries the inverse of its inclusion probability as a weight.
``sample_rows(method="chronological")`` instead draws with equal probability,
spread along each entity's timeline (``sift.sampling.chronological_sample``).  Weighted sums over the sample are unbiased for the
full-data sums (Horvitz-Thompson).  A model fitted with the weights is
consistent for the full-data fit but not exactly unbiased: the fit divides by
the realized weight total, and nonlinear models add their own bias.

Optimal-subsampling results for observed labels (Wang, Zhu & Ma, 2018; Ai et
al., 2021; Ting & Brochu, 2018) make a row's probability grow with its
residual, usually times a term that depends on its features and the model.
The gradient alone is the model-agnostic part of that rule.  It is an
empirical choice that beat smoothed scores in the benchmark, not a proven
optimum for every model.  The even half keeps every row drawable, bounds every
weight, and keeps the draw from chasing noise on targets that are mostly
noise.  The defaults come from a benchmark on a ~770k-row player panel; see
``docs/ADVANCED.md``.
"""

from __future__ import annotations

import numbers
import warnings
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from sift._logging import logger

DEFAULT_SIZE = 0.2
DEFAULT_PILOT_SIZE = 0.02
DEFAULT_UNIFORM_MIX = 0.5
# Provisional floor so tiny datasets still get a usable pilot; not benchmarked.
MIN_PILOT_ROWS = 1_000
PILOT_MAX_ITER = 100

_TASKS = ("auto", "regression", "classification")
_METHODS = ("residual", "chronological", "uniform")

_REMOVED_FIELDS = {
    "sample_frac": "renamed to 'size' (a fraction in (0, 1] or a row count)",
    "pilot_sample_size": "replaced by 'pilot_size' (default 2% of rows)",
    "min_per_group": "removed; use anchor_fn=first_per_group to guarantee coverage",
    "leverage_batch_size": "removed with leverage scoring",
    "svd_sample_size": "removed with leverage scoring",
    "weight_clip_quantile": "removed; weights are exact inverse probabilities",
    "residual_weight_cap": "removed; use uniform_mix to blend in equal-probability draws",
    "uniform_floor": "replaced by 'uniform_mix'",
    "anchor_max_share": "removed; anchors are always kept and count against the budget",
}


def _is_int_like(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _is_real_like(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _validate_size(value: object, name: str) -> None:
    if _is_int_like(value):
        if value < 1:
            raise ValueError(f"{name} must be a row count >= 1 or a fraction in (0, 1].")
        return
    if not _is_real_like(value):
        raise TypeError(f"{name} must be an int row count or a float fraction.")
    if not np.isfinite(value) or not (0 < value <= 1):
        raise ValueError(
            f"{name} must be a fraction in (0, 1] when given as a float "
            f"(pass an int for a row count); got {value!r}."
        )


def _resolve_count(value, n: int) -> int:
    """Rows requested by a fraction or a count, clipped to [1, n]."""
    if _is_int_like(value):
        return int(min(int(value), n))
    return int(min(n, max(1, np.floor(float(value) * n))))


def _is_precomputed_pilot(pilot: object) -> bool:
    return isinstance(pilot, (np.ndarray, pd.Series, pd.DataFrame, list, tuple))


def _validate_common(size, method, task, pilot, pilot_size, uniform_mix, random_state) -> None:
    _validate_size(size, "size")
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got {method!r}.")
    _validate_size(pilot_size, "pilot_size")
    if task not in _TASKS:
        raise ValueError(f"task must be one of {_TASKS}; got {task!r}.")
    if not (
        pilot is None
        or (isinstance(pilot, str) and pilot == "auto")
        or _is_precomputed_pilot(pilot)
        or hasattr(pilot, "fit")
    ):
        raise TypeError(
            "pilot must be 'auto', None, an estimator with fit/predict, or an "
            "array of precomputed predictions."
        )
    if not _is_real_like(uniform_mix) or not np.isfinite(uniform_mix):
        raise TypeError("uniform_mix must be a finite real number.")
    if not (0 < uniform_mix <= 1):
        raise ValueError(
            "uniform_mix must be in (0, 1]; a positive share keeps every row drawable, "
            "which inverse-probability weighting needs."
        )
    if random_state is not None and not _is_int_like(random_state):
        raise TypeError("random_state must be an integer or None.")


# =============================================================================
# Result
# =============================================================================


@dataclass(frozen=True)
class RowSample:
    """Rows chosen by ``sample_rows`` and their fitting weights.

    Attributes
    ----------
    indices : ndarray of int64
        Sorted positions (not labels) of the kept rows in the input.
    weights : ndarray of float64
        Mean-one fitting weights: ``1 / inclusion_prob`` rescaled to mean 1
        over the kept rows, so the sample has the same total weight as an
        unweighted sample of its size. Pass them to the downstream fit as
        ``sample_weight``. To estimate a full-data total, use
        ``1 / inclusion_prob`` itself.
    inclusion_prob : ndarray of float64
        Probability that each kept row was drawn.
    n_rows : int
        Number of rows in the input.
    task : str or None
        ``"regression"`` or ``"classification"`` as resolved for the pilot;
        None when no pilot was fitted.
    method : str
        ``"residual"``, ``"chronological"`` or ``"uniform"``.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import sample_rows
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(2_000, 3))
    >>> y = X[:, 0] + rng.normal(size=2_000)
    >>> s = sample_rows(X, y, size=0.25, random_state=0)
    >>> len(s), s.n_rows
    (500, 2000)
    >>> X_small, y_small = s.take(X), s.take(y)
    >>> X_small.shape
    (500, 3)
    >>> round(float(s.weights.mean()), 6)
    1.0
    """

    indices: np.ndarray
    weights: np.ndarray
    inclusion_prob: np.ndarray
    n_rows: int
    task: Optional[str]
    method: str

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def fraction(self) -> float:
        """Share of input rows that were kept."""
        return len(self) / self.n_rows if self.n_rows else 0.0

    def take(self, data):
        """Return the kept rows of ``data`` (DataFrame, Series or array)."""
        if hasattr(data, "iloc"):
            return data.iloc[self.indices]
        if hasattr(data, "to_pandas") and not isinstance(data, np.ndarray):
            data = data.to_pandas()
            return data.iloc[self.indices]
        return np.asarray(data)[self.indices]


# =============================================================================
# Core sampler
# =============================================================================


def _as_1d(values, n: int, name: str) -> np.ndarray:
    arr = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    arr = arr.reshape(-1) if arr.ndim > 1 and arr.shape[-1] == 1 else arr
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must be 1-D with one entry per row ({n}); got shape {arr.shape}.")
    return arr


def _n_rows(*candidates) -> int:
    for obj in candidates:
        if obj is not None:
            return int(len(obj))
    raise ValueError(
        "sample_rows needs at least one of X, y, groups, time, strata, always_include or "
        "precomputed pilot predictions to count rows."
    )


def _resolve_task(y: np.ndarray, task: str) -> str:
    if task != "auto":
        return task
    if y.dtype.kind in "OSUbc" or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    values = np.asarray(y, dtype=np.float64)
    if not np.isfinite(values).all():
        return "regression"  # rejected with a clear message by the regression path
    distinct = np.unique(values)
    if distinct.size <= 2 and np.all(distinct == np.round(distinct)):
        return "classification"
    return "regression"


def _capped_pi(score: np.ndarray, k: int) -> np.ndarray:
    """Inclusion probabilities proportional to ``score``, summing to ``k``, capped at 1."""
    pi = np.zeros(score.size, dtype=np.float64)
    free = np.ones(score.size, dtype=bool)
    budget = float(k)
    while True:
        total = score[free].sum()
        if total <= 0:
            pi[free] = budget / max(int(free.sum()), 1)
            return np.minimum(pi, 1.0)
        pi[free] = budget * score[free] / total
        over = free & (pi > 1.0)
        if not over.any():
            return pi
        pi[over] = 1.0
        free &= ~over
        budget = k - float((~free).sum())


def _systematic(order: np.ndarray, pi: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic PPS along ``order``: exactly ``k`` rows with inclusion probability ``pi``.

    The caller passes ``order`` already shuffled.  Along a fixed order, one
    shared start would pick the same position in every repeat of a pattern
    whose probabilities sum to 1, keeping all of one kind of row or none.
    """
    if k <= 0 or order.size == 0:
        return np.array([], dtype=np.int64)
    cum = np.cumsum(pi[order])
    cum *= k / cum[-1]
    u = rng.random()
    hits = np.floor(cum - u) - np.floor(np.r_[0.0, cum[:-1]] - u)
    chosen = order[hits > 0]
    if chosen.size > k:  # floating-point edge
        chosen = rng.choice(chosen, k, replace=False)
    elif chosen.size < k:
        rest = np.setdiff1d(order, chosen)
        chosen = np.r_[chosen, rng.choice(rest, k - chosen.size, replace=False)]
    return np.sort(chosen).astype(np.int64)


def _take_rows(X, idx: np.ndarray):
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _auto_matrix(X) -> np.ndarray:
    from sift._preprocess import to_numpy

    try:
        Xa = to_numpy(X, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "pilot='auto' needs numeric features; encode categorical or datetime "
            "columns first, or pass your own pilot estimator."
        ) from exc
    if Xa.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {Xa.shape}.")
    if np.isinf(Xa).any():
        raise ValueError("X contains infinite values; NaN is allowed, inf is not.")
    return Xa


def _class_gradient(y: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Norm of the softmax log-loss gradient, ||onehot(y) - p||."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape != (y.size, classes.size):
        raise ValueError(
            f"pilot probabilities must have shape ({y.size}, {classes.size}); got {proba.shape}."
        )
    pos = np.searchsorted(classes, y)
    pos = np.clip(pos, 0, classes.size - 1)
    known = classes[pos] == y
    p_true = np.where(known, proba[np.arange(y.size), pos], 0.0)
    sq = (proba ** 2).sum(axis=1) - np.where(known, p_true ** 2, 0.0)
    return np.sqrt(np.maximum((1.0 - p_true) ** 2 + sq, 0.0))


def _pilot_gradient(X, y, task, pilot, pilot_size, rng, random_state, verbose) -> np.ndarray:
    n = y.size
    if task == "classification":
        classes = np.unique(y)
    if _is_precomputed_pilot(pilot):
        pred = np.asarray(pilot, dtype=np.float64)
        if pred.shape[0] != n:
            raise ValueError(f"precomputed pilot has {pred.shape[0]} rows; expected {n}.")
        if not np.isfinite(pred).all():
            raise ValueError("precomputed pilot predictions must be finite.")
        if task == "classification":
            return _class_gradient(y, pred, classes)
        return np.abs(y - pred.reshape(-1))

    if X is None:
        raise ValueError("X is required to fit a pilot model.")
    p_rows = max(_resolve_count(pilot_size, n), min(n, MIN_PILOT_ROWS))
    pidx = np.sort(rng.choice(n, size=p_rows, replace=False))
    if task == "classification" and np.unique(y[pidx]).size == 1:
        # A slice that misses every rare class cannot train a classifier.
        # Predict its one class everywhere, so rows of other classes score highest.
        warnings.warn(
            "pilot slice holds a single class; scoring rows by whether they differ from it.",
            RuntimeWarning,
            stacklevel=4,
        )
        full = np.zeros((n, classes.size))
        full[:, np.searchsorted(classes, y[pidx][0])] = 1.0
        return _class_gradient(y, full, classes)
    if isinstance(pilot, str):  # "auto"
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        Xfit = _auto_matrix(X)
        cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        model = cls(max_iter=PILOT_MAX_ITER, random_state=random_state)
    else:
        from sklearn.base import clone

        Xfit = X
        model = clone(pilot)
    model.fit(_take_rows(Xfit, pidx), y[pidx])
    if task == "classification":
        if not hasattr(model, "predict_proba"):
            raise TypeError("a classification pilot needs predict_proba.")
        proba = np.asarray(model.predict_proba(Xfit), dtype=np.float64)
        full = np.zeros((n, classes.size))
        full[:, np.searchsorted(classes, np.asarray(model.classes_))] = proba
        g = _class_gradient(y, full, classes)
    else:
        pred = np.asarray(model.predict(Xfit), dtype=np.float64).reshape(-1)
        g = np.abs(y - pred)
    if not np.isfinite(g).all():
        raise ValueError("pilot model produced non-finite predictions.")
    if verbose:
        logger.info(f"Pilot fitted on {p_rows:,} of {n:,} rows ({type(model).__name__}).")
    return g


def sample_rows(
    X=None,
    y=None,
    size: float | int = DEFAULT_SIZE,
    *,
    method: str = "residual",
    groups=None,
    time=None,
    strata=None,
    task: str = "auto",
    pilot: Any = "auto",
    pilot_size: float | int = DEFAULT_PILOT_SIZE,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    always_include=None,
    random_state: Optional[int] = 42,
    verbose: bool = False,
) -> RowSample:
    """Draw exactly ``size`` training rows, with mean-one fitting weights.

    Three methods:

    ``"residual"`` (default)
        Fits a cheap pilot model on ``pilot_size`` of the rows and scores
        every row by the magnitude ``g`` of the pilot's loss gradient (the
        absolute residual for regression, ``||onehot(y) - p||`` for
        classification).  A row's inclusion probability is
        ``(1 - uniform_mix)`` times a share proportional to ``g`` (capped at
        1) plus ``uniform_mix`` times the sampling fraction.  Rows are drawn
        systematically along a random order, so a repeating pattern in the
        data cannot line up with the draw.
    ``"chronological"``
        Equal probability, spread over time: each entity in ``groups`` (and
        each stratum, such as a season) gets its proportional share rounded
        up or down, drawn evenly along its timeline.  This is
        ``sift.sampling.chronological_sample``.
    ``"uniform"``
        A simple random sample.

    Fit downstream models with ``sample_weight=weights``.

    Parameters
    ----------
    X : array-like of shape (n_rows, n_features), optional
        Features for the pilot. DataFrame, ndarray or anything your pilot
        estimator accepts. Needed only for ``method="residual"`` with a
        fitted pilot.
    y : array-like of shape (n_rows,), optional
        Target. Needed only for ``method="residual"``.
    size : float or int, default=0.2
        Rows to keep: a float fraction in ``(0, 1]`` or an int row count.
        Draw it from all eligible training rows: sampling a sample (20% of
        a 20% selection sample, say) leaves a much smaller fit.
    method : {"residual", "chronological", "uniform"}, default="residual"
        How rows are chosen; see above.
    groups : array-like of shape (n_rows,), optional
        ``method="chronological"`` only. Entity of each row (player,
        customer, ticker); each entity gets its share of the draw.
    time : array-like of shape (n_rows,), optional
        ``method="chronological"`` only. Sortable time of each row; the draw
        is spread along each entity's timeline. Defaults to input order.
    strata : array-like of shape (n_rows,), optional
        ``method="chronological"`` only. Outer level (for example a season)
        whose shares are allocated before the entities inside it.
    task : {"auto", "regression", "classification"}, default="auto"
        ``method="residual"`` only. ``"auto"`` treats non-numeric, boolean
        and two-valued targets as classification and everything else as
        regression. Pass ``"classification"`` explicitly for integer-coded
        multiclass labels.
    pilot : "auto", estimator or array-like, default="auto"
        ``method="residual"`` only. ``"auto"`` fits a 100-iteration
        scikit-learn HistGradientBoosting model. An estimator (for example a
        100-tree booster from your own library) is cloned and fitted
        instead; classification pilots need ``predict_proba``. An array of
        length ``n_rows`` is used as precomputed predictions (probabilities
        for classification), for example yesterday's production model.
    pilot_size : float or int, default=0.02
        ``method="residual"`` only. Rows used to fit the pilot: a fraction or
        a count. At least 1,000 rows (or all rows, if fewer) are used.
    uniform_mix : float, default=0.5
        ``method="residual"`` only. Share of the inclusion probability spread
        evenly over rows, in ``(0, 1]``. The even share keeps every row
        drawable and caps each raw weight at
        ``1 / (uniform_mix * fraction)``. 0.5 is the benchmarked value.
    always_include : array-like of bool, optional
        Rows that must be kept (probability 1). They count against ``size``
        and must leave room for at least one drawn row.
    random_state : int or None, default=42
        Seed for the pilot slice and the draw.
    verbose : bool, default=False
        Log progress at INFO on the ``sift`` logger.

    Returns
    -------
    RowSample
        ``indices``, mean-one fitting ``weights``, ``inclusion_prob`` and a
        ``take()`` helper.

    Notes
    -----
    Learners whose automatic hyperparameters depend on the weights (for
    example a learning rate or leaf regularisation scaled by the effective
    sample size) must have those pinned to what an unweighted fit of the
    same size would use; otherwise the comparison with a full-data fit is
    confounded. The weights are mean-one, so ``sum(weights) == len(sample)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sift import sample_rows
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(5_000, 4))
    >>> y = X @ [1.0, -2.0, 0.5, 0.0] + rng.standard_t(3, size=5_000)
    >>> s = sample_rows(X, y, size=1_000, random_state=0)
    >>> model = Ridge().fit(s.take(X), s.take(y), sample_weight=s.weights)
    >>> len(s), s.method
    (1000, 'residual')
    >>> players = np.repeat(np.arange(50), 100)
    >>> days = np.tile(np.arange(100), 50)
    >>> len(sample_rows(size=0.1, method="chronological", groups=players, time=days))
    500
    """
    return _sample_rows(
        X, y, size, method=method, groups=groups, time=time, strata=strata, task=task,
        pilot=pilot, pilot_size=pilot_size, uniform_mix=uniform_mix,
        always_include=always_include, random_state=random_state, verbose=verbose,
    )


def _sample_rows(
    X, y, size, *, method, groups, time, strata, task, pilot, pilot_size, uniform_mix,
    always_include, random_state, verbose,
) -> RowSample:
    """Implementation shared by ``sample_rows`` and ``smart_sample``.

    Both public functions call this directly, so ``stacklevel=3`` warnings
    point at the user's call site.
    """
    _validate_common(size, method, task, pilot, pilot_size, uniform_mix, random_state)
    precomputed = pilot if method == "residual" and _is_precomputed_pilot(pilot) else None
    n = _n_rows(X, y, precomputed, groups, time, strata, always_include)
    k = _resolve_count(size, n)
    rng = np.random.default_rng(random_state)

    anchors = np.zeros(n, dtype=bool)
    if always_include is not None:
        anchors = _as_1d(always_include, n, "always_include").astype(bool)
    n_anchor = int(anchors.sum())
    if n_anchor > k:
        raise ValueError(
            f"always_include marks {n_anchor:,} rows but size allows only {k:,}; "
            "raise size or include fewer rows."
        )
    if n_anchor == k < n:
        raise ValueError(
            f"always_include marks {n_anchor:,} rows, the whole budget, so no other row "
            "could be drawn; raise size or include fewer rows."
        )
    groups = None if groups is None else _as_1d(groups, n, "groups")
    time = None if time is None else _as_1d(time, n, "time")
    strata = None if strata is None else _as_1d(strata, n, "strata")
    if method != "chronological" and not (groups is None and time is None and strata is None):
        raise ValueError(
            f"groups, time and strata shape only method='chronological'; "
            f"method={method!r} draws rows in random order."
        )
    if method == "chronological" and groups is None and time is None:
        raise ValueError(
            "method='chronological' needs groups or time; use method='uniform' for a "
            "plain random draw."
        )
    if method == "residual" and pilot is None:
        raise ValueError(
            "method='residual' needs a pilot; for an equal-probability draw use "
            "method='uniform' or method='chronological'."
        )

    if k >= n:
        idx = np.arange(n, dtype=np.int64)
        ones = np.ones(n, dtype=np.float64)
        return RowSample(idx, ones, ones.copy(), n, None, method)

    free = ~anchors
    free_idx = np.flatnonzero(free)
    k_free = k - n_anchor
    pi = np.ones(n, dtype=np.float64)
    resolved = None
    if method == "residual":
        if y is None:
            raise ValueError("y is required for method='residual'.")
        y_raw = _as_1d(y, n, "y")
        resolved = _resolve_task(y_raw, task)
        if resolved == "regression":
            try:
                y_arr = y_raw.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("regression y must be numeric; pass task='classification'.") from exc
            if not np.isfinite(y_arr).all():
                raise ValueError("y must be finite for regression.")
        else:
            if pd.isna(y_raw).any():
                raise ValueError("y must not contain missing labels for classification.")
            y_arr = y_raw
        # The pilot slice is the first draw from ``rng``.
        g = _pilot_gradient(X, y_arr, resolved, pilot, pilot_size, rng, random_state, verbose)
        score = g
        if not g[free].any():
            warnings.warn(
                "pilot fits every row exactly; sampling with equal probability.",
                RuntimeWarning,
                stacklevel=3,
            )
            score = np.ones(n, dtype=np.float64)
        pi_free = _capped_pi(score[free], k_free)
        pi_free = (1.0 - uniform_mix) * pi_free + uniform_mix * k_free / free_idx.size
        pi[free] = pi_free
        drawn = _systematic(rng.permutation(free_idx), pi, k_free, rng)
    elif method == "uniform":
        pi[free] = k_free / free_idx.size
        drawn = rng.choice(free_idx, k_free, replace=False)
    else:
        from sift.sampling.chronological import chronological_sample

        pi[free] = k_free / free_idx.size
        frame = pd.DataFrame({
            "group": groups[free] if groups is not None else np.zeros(free_idx.size, dtype=np.int8),
            "time": time[free] if time is not None else np.arange(free_idx.size),
        })
        if strata is not None:
            frame["stratum"] = strata[free]
        positions = chronological_sample(
            frame, k_free, group_col="group", time_col="time",
            stratum_col="stratum" if strata is not None else None,
            random_state=random_state,
        )
        drawn = free_idx[positions]
    idx = np.sort(np.r_[np.flatnonzero(anchors), drawn]).astype(np.int64)
    inclusion = pi[idx]
    weights = 1.0 / inclusion
    weights /= weights.mean()
    if verbose:
        logger.info(
            f"Sampled {idx.size:,} of {n:,} rows ({idx.size / n:.1%}, method={method}); "
            f"max weight {weights.max():.2f}, Kish fraction "
            f"{weights.sum() ** 2 / (weights ** 2).sum() / idx.size:.2f}."
        )
    return RowSample(idx, weights, inclusion, n, resolved, method)


# =============================================================================
# DataFrame convenience API
# =============================================================================


def _validate_smart_sampler_config(config: "SmartSamplerConfig") -> None:
    _validate_common(
        config.size, config.method, config.task, config.pilot, config.pilot_size,
        config.uniform_mix, config.random_state,
    )
    for name in ("group_col", "time_col", "strata_col"):
        value = getattr(config, name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")
    if config.anchor_fn is not None and not callable(config.anchor_fn):
        raise TypeError("anchor_fn must be callable or None.")
    if config.method != "chronological":
        if config.strata_col is not None:
            raise ValueError("strata_col applies only to method='chronological'.")
        if config.anchor_fn is None and (config.group_col or config.time_col):
            raise ValueError(
                f"group_col and time_col shape only method='chronological' (or feed "
                f"anchor_fn); method={config.method!r} draws rows in random order."
            )
    elif not (config.group_col or config.time_col):
        raise ValueError("method='chronological' needs group_col or time_col.")
    if config.method == "residual" and config.pilot is None:
        raise ValueError(
            "method='residual' needs a pilot; for an equal-probability draw use "
            "method='uniform' or method='chronological'."
        )
    if not isinstance(config.verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a bool.")


@dataclass
class SmartSamplerConfig:
    """Settings for ``smart_sample`` and ``StabilitySelector(use_smart_sampler=True)``.

    Every field mirrors the keyword of the same name in ``sample_rows``;
    ``group_col``, ``time_col``, ``strata_col`` and ``anchor_fn`` name
    DataFrame columns and an anchor rule instead of passing arrays.

    Parameters
    ----------
    size : float or int, default=0.2
        Rows to keep: a fraction in ``(0, 1]`` or a row count.
    method : {"residual", "chronological", "uniform"}, default="residual"
        How rows are chosen; see ``sample_rows``.
    group_col : str, optional
        Entity column. Shapes ``method="chronological"``; other methods use
        it only to call ``anchor_fn``.
    time_col : str, optional
        Time column. Shapes ``method="chronological"``; other methods use it
        only to call ``anchor_fn``.
    strata_col : str, optional
        Outer level (such as a season) for ``method="chronological"``.
    task : {"auto", "regression", "classification"}, default="auto"
        How the pilot's gradient is computed (``method="residual"``).
    pilot : "auto", estimator or array-like, default="auto"
        Pilot model for ``method="residual"``; see ``sample_rows``.
    pilot_size : float or int, default=0.02
        Rows used to fit the pilot.
    uniform_mix : float, default=0.5
        Share of probability spread evenly over rows, in ``(0, 1]``.
    anchor_fn : callable, optional
        ``anchor_fn(df, group_col, time_col) -> bool mask`` of rows that must
        be kept, e.g. ``sift.sampling.first_per_group``.
    random_state : int or None, default=42
        Seed.
    verbose : bool, default=False
        Log progress at INFO on the ``sift`` logger.

    See Also
    --------
    sift.sample_rows : The array-level sampler.
    sift.smart_sample : DataFrame sampler that consumes this configuration.
    sift.panel_config : Preset for entity-over-time data.

    Examples
    --------
    >>> import dataclasses
    >>> from sift import SmartSamplerConfig
    >>> config = SmartSamplerConfig(size=0.25)
    >>> config.size, config.method, config.pilot, config.uniform_mix
    (0.25, 'residual', 'auto', 0.5)
    >>> dataclasses.replace(config, size=10_000).size
    10000
    """

    size: float | int = DEFAULT_SIZE
    method: str = "residual"
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    strata_col: Optional[str] = None
    task: str = "auto"
    pilot: Any = "auto"
    pilot_size: float | int = DEFAULT_PILOT_SIZE
    uniform_mix: float = DEFAULT_UNIFORM_MIX
    anchor_fn: Optional[Callable] = None
    random_state: Optional[int] = 42
    verbose: bool = False

    def __post_init__(self) -> None:
        _validate_smart_sampler_config(self)


def _removed_field_error(names) -> TypeError:
    details = "; ".join(
        f"{name}: {_REMOVED_FIELDS[name]}" for name in sorted(names) if name in _REMOVED_FIELDS
    )
    suffix = f" sift 2.0 changes: {details}." if details else ""
    return TypeError(f"Unknown SmartSamplerConfig field(s): {sorted(names)}.{suffix}")


def _resolve_smart_config(config: Optional[SmartSamplerConfig], overrides: dict) -> SmartSamplerConfig:
    valid = {f.name for f in fields(SmartSamplerConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise _removed_field_error(unknown)
    if config is None:
        return SmartSamplerConfig(**overrides)
    if not isinstance(config, SmartSamplerConfig):
        raise TypeError("config must be a SmartSamplerConfig or None.")
    if overrides:
        return replace(config, **overrides)
    _validate_smart_sampler_config(config)
    return config


def smart_sample(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    config: Optional[SmartSamplerConfig] = None,
    **kwargs,
) -> pd.DataFrame:
    """Draw training rows of a DataFrame, with a ``sample_weight`` column.

    DataFrame front end to ``sample_rows``. With the default
    ``method="residual"`` it fits a pilot on ``feature_cols`` and draws
    exactly ``size`` rows weighted toward the rows the pilot gets wrong;
    ``method="chronological"`` and ``"uniform"`` draw with equal probability.
    It returns the kept rows (original index preserved) with mean-one
    fitting weights.

    Parameters
    ----------
    df : DataFrame
        Input data.
    feature_cols : list of str
        Numeric feature columns for the pilot (NaN allowed).
    y_col : str
        Target column.
    config : SmartSamplerConfig, optional
        Settings. If None, defaults plus keyword overrides are used.
    **kwargs
        Override individual ``SmartSamplerConfig`` fields by name.

    Returns
    -------
    DataFrame
        The kept rows in input order with an added ``sample_weight`` column.

    See Also
    --------
    sift.sample_rows : Array-level sampler returning indices and weights.
    sift.panel_config : Preset for entity-over-time data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sift import smart_sample
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "x0": rng.normal(size=6_000),
    ...     "x1": rng.normal(size=6_000),
    ...     "team": np.repeat([f"t{i}" for i in range(6)], 1_000),
    ... })
    >>> df["y"] = 2.0 * df["x0"] - df["x1"] + rng.standard_t(3, size=6_000)
    >>> sample = smart_sample(df, ["x0", "x1"], "y", size=0.25, random_state=0)
    >>> len(sample), list(sample.columns)
    (1500, ['x0', 'x1', 'team', 'y', 'sample_weight'])
    >>> spread = smart_sample(df, ["x0", "x1"], "y", size=0.25, method="chronological",
    ...                       group_col="team")
    >>> spread["team"].value_counts().tolist()
    [250, 250, 250, 250, 250, 250]
    >>> round(float(sample["sample_weight"].mean()), 6)
    1.0
    """
    config = _resolve_smart_config(config, kwargs)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("smart_sample requires a pandas DataFrame; use sample_rows for arrays.")
    feature_cols = list(feature_cols)
    required = set(feature_cols) | {y_col}
    for col in (config.group_col, config.time_col, config.strata_col):
        if col:
            required.add(col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "sample_weight" in df.columns:
        raise ValueError(
            "smart_sample writes its inverse-probability weights to a "
            "'sample_weight' column, but the input already has one. Rename or "
            "drop that column first so a feature is not silently overwritten."
        )
    residual = config.method == "residual"
    if residual and not feature_cols and not _is_precomputed_pilot(config.pilot):
        raise ValueError("feature_cols must name at least one column to fit the pilot.")
    X = df[feature_cols]
    if residual and isinstance(config.pilot, str):
        bad = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
               or pd.api.types.is_bool_dtype(df[c])]
        if bad:
            raise TypeError(
                f"smart_sample's default pilot needs numeric feature columns; "
                f"non-numeric: {bad[:5]}"
            )
    anchors = None
    if config.anchor_fn is not None:
        anchors = np.asarray(
            config.anchor_fn(df.reset_index(drop=True), config.group_col, config.time_col),
            dtype=bool,
        ).reshape(-1)
        if anchors.size != len(df):
            raise ValueError(f"anchor_fn returned {anchors.size} rows but df has {len(df)} rows")
    chronological = config.method == "chronological"
    result = _sample_rows(
        X,
        df[y_col],
        config.size,
        method=config.method,
        groups=df[config.group_col] if chronological and config.group_col else None,
        time=df[config.time_col] if chronological and config.time_col else None,
        strata=df[config.strata_col] if config.strata_col else None,
        task=config.task,
        pilot=config.pilot,
        pilot_size=config.pilot_size,
        uniform_mix=config.uniform_mix,
        always_include=anchors,
        random_state=config.random_state,
        verbose=bool(config.verbose),
    )
    out = df.iloc[result.indices].copy()
    out["sample_weight"] = result.weights
    return out


__all__ = [
    "RowSample",
    "SmartSamplerConfig",
    "sample_rows",
    "smart_sample",
]
