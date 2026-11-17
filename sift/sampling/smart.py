"""Pilot-residual importance sampling of training rows.

A cheap pilot model is fitted on a small random slice of the data and scores
every row by how wrong it is: the magnitude ``g`` of the loss gradient at the
pilot prediction.  Half of each row's inclusion probability is proportional to
``g`` and half is spread evenly (a defensive mixture; Hesterberg, 1995).  The
draw runs systematically along each group's timeline, so it keeps an exact size
and an even spread, and every kept row carries the inverse of its inclusion
probability as a weight.  Fitting on the weighted sample is an unbiased
estimate of fitting on all rows, with lower variance than a uniform sample of
the same size.

When labels are observed, the realized gradient magnitude is the
variance-optimal sampling score (Wang, Zhu & Ma, 2018; Ai et al., 2021; Ting &
Brochu, 2018); smoothing it, for example with isotonic regression, discards
information.  The even half keeps every row drawable, bounds every weight, and
keeps the draw from chasing noise on targets that are mostly noise.  The
defaults come from a benchmark on a ~770k-row player panel; see
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


def _validate_common(size, task, pilot, pilot_size, uniform_mix, random_state) -> None:
    _validate_size(size, "size")
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
            "which the inverse-probability weights need to stay unbiased."
        )
    if random_state is not None and not _is_int_like(random_state):
        raise TypeError("random_state must be an integer or None.")


# =============================================================================
# Result
# =============================================================================


@dataclass(frozen=True)
class RowSample:
    """Rows chosen by ``sample_rows`` and their inverse-probability weights.

    Attributes
    ----------
    indices : ndarray of int64
        Sorted positions (not labels) of the kept rows in the input.
    weights : ndarray of float64
        ``1 / inclusion_prob`` rescaled to mean 1 over the kept rows, so the
        sample has the same total weight as an unweighted sample of its size.
        Pass them to the downstream fit as ``sample_weight``.
    inclusion_prob : ndarray of float64
        Probability that each kept row was drawn.
    n_rows : int
        Number of rows in the input.
    task : str
        ``"regression"`` or ``"classification"`` as resolved for the pilot,
        or ``"uniform"`` when no pilot was fitted (``pilot=None``, or
        ``size`` covering every row).

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
    task: str

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
        "sample_rows needs at least one of X, y, groups, time, always_include or "
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


def _draw_order(n: int, groups, time, rng: np.random.Generator) -> np.ndarray:
    """Groups in random order, rows by time inside each group."""
    if groups is None and time is None:
        return rng.permutation(n)
    keys = [np.arange(n)]
    if time is not None:
        codes = pd.factorize(pd.Series(time), sort=True)[0]
        keys.append(codes)
    if groups is not None:
        codes = pd.factorize(pd.Series(groups), use_na_sentinel=True)[0].astype(np.int64)
        codes[codes < 0] = codes.max() + 1
        keys.append(rng.permutation(int(codes.max()) + 1)[codes])
    return np.lexsort(tuple(keys))


def _systematic(order: np.ndarray, pi: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic PPS along ``order``: exactly ``k`` rows with inclusion probability ``pi``."""
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
    groups=None,
    time=None,
    task: str = "auto",
    pilot: Any = "auto",
    pilot_size: float | int = DEFAULT_PILOT_SIZE,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    always_include=None,
    random_state: Optional[int] = 42,
    verbose: bool = False,
) -> RowSample:
    """Choose the most informative training rows and weight them for unbiased fits.

    Fits a cheap pilot model on ``pilot_size`` of the rows and scores every
    row by the magnitude ``g`` of the pilot's loss gradient (the absolute
    residual for regression, ``||onehot(y) - p||`` for classification).  It
    then draws exactly ``size`` rows; a row's inclusion probability is
    ``(1 - uniform_mix)`` times a share proportional to ``g`` (capped at 1)
    plus ``uniform_mix`` times the sampling fraction.  The draw is systematic
    along ``groups`` (in random order) and
    ``time`` (in order inside each group), so every entity's timeline is
    covered evenly.  Fit downstream models with ``sample_weight=weights``.

    Parameters
    ----------
    X : array-like of shape (n_rows, n_features), optional
        Features for the pilot. DataFrame, ndarray or anything your pilot
        estimator accepts. Not needed when ``pilot`` is None or precomputed.
    y : array-like of shape (n_rows,), optional
        Target. Required unless ``pilot`` is None.
    size : float or int, default=0.2
        Rows to keep: a float fraction in ``(0, 1]`` or an int row count.
        20-25% matched full-data accuracy to about 0.1-0.15% in the panel
        benchmark; 5-10% is enough for a rough feature screen.
    groups : array-like of shape (n_rows,), optional
        Entity of each row (player, customer, ticker). Spreads the draw across
        entities.
    time : array-like of shape (n_rows,), optional
        Sortable time of each row. Spreads the draw along each timeline.
    task : {"auto", "regression", "classification"}, default="auto"
        ``"auto"`` treats non-numeric, boolean and two-valued targets as
        classification and everything else as regression. Pass
        ``"classification"`` explicitly for integer-coded multiclass labels.
    pilot : "auto", estimator, array-like or None, default="auto"
        ``"auto"`` fits a 100-iteration scikit-learn HistGradientBoosting
        model. An estimator (for example a 100-tree booster from your own
        library) is cloned and fitted instead; classification pilots need
        ``predict_proba``. An array of length ``n_rows`` is used as
        precomputed predictions (probabilities for classification), for
        example yesterday's production model. ``None`` skips the pilot and
        draws with equal probability, still spread along groups and time.
    pilot_size : float or int, default=0.02
        Rows used to fit the pilot: a fraction or a count. At least 1,000
        rows (or all rows, if fewer) are used.
    uniform_mix : float, default=0.5
        Share of the inclusion probability spread evenly over rows, in
        ``(0, 1]``. The even share keeps every row drawable and caps each raw
        weight at ``1 / (uniform_mix * fraction)``; ``1.0`` is an
        equal-probability draw. 0.5 is the benchmarked value.
    always_include : array-like of bool, optional
        Rows that must be kept (probability 1). They count against ``size``.
    random_state : int or None, default=42
        Seed for the pilot slice, the group order and the systematic start.
    verbose : bool, default=False
        Log progress at INFO on the ``sift`` logger.

    Returns
    -------
    RowSample
        ``indices``, ``weights`` (mean 1), ``inclusion_prob`` and a
        ``take()`` helper.

    Notes
    -----
    Learners whose automatic hyperparameters depend on the weights (for
    example a learning rate or leaf regularisation scaled by the effective
    sample size) must have those pinned to what an unweighted fit of the
    same size would use; otherwise the comparison with a full-data fit is
    confounded. The weights are mean-1, so ``sum(weights) == len(sample)``.

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
    >>> len(s)
    1000
    """
    return _sample_rows(
        X, y, size, groups=groups, time=time, task=task, pilot=pilot, pilot_size=pilot_size,
        uniform_mix=uniform_mix, always_include=always_include, random_state=random_state,
        verbose=verbose,
    )


def _sample_rows(
    X, y, size, *, groups, time, task, pilot, pilot_size, uniform_mix, always_include,
    random_state, verbose,
) -> RowSample:
    """Implementation shared by ``sample_rows`` and ``smart_sample``.

    Both public functions call this directly, so ``stacklevel=3`` warnings
    point at the user's call site.
    """
    _validate_common(size, task, pilot, pilot_size, uniform_mix, random_state)
    n = _n_rows(X, y, pilot if _is_precomputed_pilot(pilot) else None, groups, time, always_include)
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
    groups = None if groups is None else _as_1d(groups, n, "groups")
    time = None if time is None else _as_1d(time, n, "time")

    if k >= n:
        idx = np.arange(n, dtype=np.int64)
        ones = np.ones(n, dtype=np.float64)
        return RowSample(idx, ones, ones.copy(), n, "uniform")

    if pilot is None:
        resolved = "uniform"
        y_arr = None
    else:
        if y is None:
            raise ValueError("y is required unless pilot=None.")
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

    k_free = k - n_anchor
    if y_arr is None or k_free == 0:
        score = np.ones(n, dtype=np.float64)
    else:
        g = _pilot_gradient(X, y_arr, resolved, pilot, pilot_size, rng, random_state, verbose)
        if not g[~anchors].any():
            warnings.warn(
                "pilot fits every row exactly; sampling with equal probability.",
                RuntimeWarning,
                stacklevel=3,
            )
            score = np.ones(n, dtype=np.float64)
        else:
            score = g

    free = ~anchors
    pi = np.ones(n, dtype=np.float64)
    if k_free > 0:
        pi_free = _capped_pi(score[free], k_free)
        if uniform_mix > 0:
            pi_free = (1.0 - uniform_mix) * pi_free + uniform_mix * k_free / free.sum()
        pi[free] = pi_free
        order = _draw_order(n, groups, time, rng)
        drawn = _systematic(order[free[order]], pi, k_free, rng)
    else:
        drawn = np.array([], dtype=np.int64)
    idx = np.sort(np.r_[np.flatnonzero(anchors), drawn]).astype(np.int64)
    inclusion = pi[idx]
    weights = 1.0 / inclusion
    weights /= weights.mean()
    if verbose:
        logger.info(
            f"Sampled {idx.size:,} of {n:,} rows ({idx.size / n:.1%}); "
            f"max weight {weights.max():.2f}, Kish fraction "
            f"{weights.sum() ** 2 / (weights ** 2).sum() / idx.size:.2f}."
        )
    return RowSample(idx, weights, inclusion, n, resolved)


# =============================================================================
# DataFrame convenience API
# =============================================================================


def _validate_smart_sampler_config(config: "SmartSamplerConfig") -> None:
    _validate_common(
        config.size, config.task, config.pilot, config.pilot_size,
        config.uniform_mix, config.random_state,
    )
    for name in ("group_col", "time_col"):
        value = getattr(config, name)
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")
    if config.anchor_fn is not None and not callable(config.anchor_fn):
        raise TypeError("anchor_fn must be callable or None.")
    if not isinstance(config.verbose, (bool, np.bool_)):
        raise TypeError("verbose must be a bool.")


@dataclass
class SmartSamplerConfig:
    """Settings for ``smart_sample`` and ``StabilitySelector(use_smart_sampler=True)``.

    Every field mirrors the keyword of the same name in ``sample_rows``;
    ``group_col``, ``time_col`` and ``anchor_fn`` name DataFrame columns and
    an anchor rule instead of passing arrays.

    Parameters
    ----------
    size : float or int, default=0.2
        Rows to keep: a fraction in ``(0, 1]`` or a row count.
    group_col : str, optional
        Entity column; spreads the draw across entities.
    time_col : str, optional
        Time column; spreads the draw along each entity's timeline.
    task : {"auto", "regression", "classification"}, default="auto"
        How the pilot's gradient is computed.
    pilot : "auto", estimator, array-like or None, default="auto"
        Pilot model; see ``sample_rows``.
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
    >>> config = SmartSamplerConfig(size=0.25, group_col="team")
    >>> config.size, config.pilot, config.uniform_mix
    (0.25, 'auto', 0.5)
    >>> dataclasses.replace(config, size=10_000).size
    10000
    """

    size: float | int = DEFAULT_SIZE
    group_col: Optional[str] = None
    time_col: Optional[str] = None
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
    """Keep the most informative rows of a DataFrame, with a ``sample_weight`` column.

    DataFrame front end to ``sample_rows``: fits a pilot on
    ``feature_cols``, draws exactly ``size`` rows weighted toward the rows the
    pilot gets wrong, and returns them (original index preserved) with exact
    inverse-probability weights rescaled to mean 1.

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
    >>> sample = smart_sample(df, ["x0", "x1"], "y", size=0.25, group_col="team",
    ...                       random_state=0)
    >>> len(sample), list(sample.columns)
    (1500, ['x0', 'x1', 'team', 'y', 'sample_weight'])
    >>> round(float(sample["sample_weight"].mean()), 6)
    1.0
    """
    config = _resolve_smart_config(config, kwargs)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("smart_sample requires a pandas DataFrame; use sample_rows for arrays.")
    feature_cols = list(feature_cols)
    required = set(feature_cols) | {y_col}
    for col in (config.group_col, config.time_col):
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
    if not feature_cols and config.pilot is not None and not _is_precomputed_pilot(config.pilot):
        raise ValueError("feature_cols must name at least one column to fit the pilot.")
    X = df[feature_cols]
    if isinstance(config.pilot, str):
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
    result = _sample_rows(
        X,
        df[y_col],
        config.size,
        groups=df[config.group_col] if config.group_col else None,
        time=df[config.time_col] if config.time_col else None,
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
