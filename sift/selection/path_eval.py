"""Feature-path evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, List, Mapping, Optional

import copy
import inspect
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sift._metadata import resolve_row_metadata
from sift._preprocess import best_score_from_dict, ensure_weights
from sift.scoring import (
    UnsupportedScorerSampleWeightError,
    is_sklearn_scorer,
    score_with_sklearn_scorer,
    sklearn_scorer_label,
)


EstimatorFactory = Callable[[], Any]
Scoring = str | Callable[[np.ndarray, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class FeaturePathEvaluationResult:
    """Result of explicit feature-path evaluation.

    Returned by ``evaluate_feature_path``, the explicit counterpart to the
    ``AutoKConfig`` rules: instead of choosing a ``k_method``, the caller
    supplies the ordered path, the exact k grid, the estimator, and the
    scoring, and gets the whole curve back. It is a predictive-sufficiency
    object -- every number in it is an out-of-sample score of a fitted model,
    on a lower-is-better scale -- so it answers "how many of these features
    does this model need", not "which features carry conditional signal".

    The dataclass is frozen: treat every field as read-only, and build a
    different result rather than mutating one. ``diagnostics`` is a plain
    DataFrame and is not copied, so it is the one field that can be edited in
    place; do not.

    Attributes
    ----------
    feature_path : list of str
        The ordered path as resolved against the input columns: string names
        matched to the first column with that label, integer entries resolved
        positionally, and repeats of an already-resolved position dropped
        while keeping first-seen order. It can therefore be shorter than the
        ``feature_path`` argument that produced it.
    k : list of int
        Evaluated prefix sizes, in the order given, deduplicated. Every entry
        is at least 1 and at most ``len(feature_path)``.
    features : list of str
        The first ``best_k`` names of ``feature_path``; empty when
        ``best_k == 0``.
    scores : mapping of int to float
        Mean validation score per evaluated k, lower is better. A k that
        produced a non-finite score on any split is recorded as ``inf`` so it
        cannot win on partial coverage. Estimator-style sklearn scorers are
        negated into this scale.
    best_k : int
        The k with the smallest finite mean score, ties broken toward the
        smaller k; ``0`` when no k scored finitely.
    diagnostics : DataFrame
        One row per evaluated k with ``k``, ``score`` (the mean), ``std``
        (population standard deviation across splits, NaN with a single
        finite split), ``n_finite``, ``n_splits``, ``best_score``, and
        ``scoring`` (the metric name, the sklearn scorer label, or a
        ``legacy:module.qualname`` tag for a plain callable).

    Methods
    -------
    result_view(input_features=None)
        Return an additive ``SelectionView`` over this result without
        changing it. ``input_features`` names the original input columns when
        they cannot be recovered from the result itself.

    See Also
    --------
    evaluate_feature_path : Produces this object.
    select_k_auto : The ``AutoKConfig(k_method='evaluate')`` rule, which picks
        its own k grid and proxy model.
    select_k_gaussian_cv : Model-free predictive-risk curve over every k.

    Notes
    -----
    ``best_k`` is recomputed by ``result_view`` and by
    ``sift.selection.view.as_result``, which validate that it agrees with
    ``scores`` and ``diagnostics``; a hand-assembled instance whose fields
    disagree is rejected there rather than silently normalized.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import evaluate_feature_path
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    >>> y = X["a"] + 0.6 * X["b"] + 0.8 * rng.normal(size=120)
    >>> result = evaluate_feature_path(
    ...     X, y.to_numpy(), ["a", "b", "c"], [1, 2, 3], random_state=0
    ... )
    >>> result.best_k, result.features
    (2, ['a', 'b'])
    >>> result.k, sorted(result.scores)
    ([1, 2, 3], [1, 2, 3])
    >>> result.diagnostics["scoring"].iloc[0]
    'rmse'
    >>> view = result.result_view()
    >>> view.k, view.features
    (2, ['a', 'b'])
    """

    feature_path: List[str]
    k: List[int]
    features: List[str]
    scores: Mapping[int, float]
    best_k: int
    diagnostics: pd.DataFrame

    def result_view(self, input_features=None):
        """Return an additive normalized view without changing this result."""
        from sift.selection.view import as_result

        return as_result(self, input_features=input_features)


def _split_weights(weights: np.ndarray, idx: np.ndarray, *, label: str) -> np.ndarray:
    """Return fold-local mean-one weights."""
    out = np.asarray(weights[idx], dtype=np.float64)
    total = float(out.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{label} split has invalid sample_weight")
    mean = float(out.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"{label} split has invalid sample_weight mean")
    return out / mean


def _resolve_feature_path(
    X: pd.DataFrame,
    feature_path: Iterable[Hashable],
) -> tuple[List[Hashable], List[int]]:
    """Resolve the path against DataFrame columns.

    Returns ``(names, positions)``. Integer entries are treated as positional
    indices (so duplicate column labels cannot redirect them), string entries
    resolve to the first matching column label.
    """
    resolved_names: List[Hashable] = []
    resolved_positions: List[int] = []
    missing: list[str] = []
    columns = list(X.columns)
    first_position: dict[Hashable, int] = {}
    for idx, col in enumerate(columns):
        first_position.setdefault(col, idx)

    for f in feature_path:
        if isinstance(f, str):
            if f in first_position:
                resolved_names.append(f)
                resolved_positions.append(first_position[f])
                continue
            missing.append(f)
            continue
        if isinstance(f, (int, np.integer)):
            idx = int(f)
            if idx < 0 or idx >= len(columns):
                missing.append(str(f))
                continue
            resolved_names.append(columns[idx])
            resolved_positions.append(idx)
            continue
        raise TypeError(
            "feature_path must contain feature names (str) or positional indices (int)"
        )

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"feature_path contains missing features: {preview}{suffix}")

    if not resolved_positions:
        raise ValueError("feature_path resolved to zero features")

    # Keep path order while de-duplicating by position.
    names: list[Hashable] = []
    positions: list[int] = []
    seen: set[int] = set()
    for name, pos in zip(resolved_names, resolved_positions):
        if pos in seen:
            continue
        seen.add(pos)
        names.append(name)
        positions.append(pos)
    return names, positions


def _resolve_k_grid(k_grid: Iterable[int], *, max_k: int) -> List[int]:
    """Validate and deduplicate explicit k grid values."""
    deduped: list[int] = []
    seen = set()

    for k in k_grid:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise ValueError(f"k_grid entries must be positive integers, got {k!r}")
        k_int = int(k)
        if k_int < 1:
            raise ValueError(f"k_grid must contain values >= 1, got {k_int}")
        if k_int > max_k:
            raise ValueError(f"k_grid contains k={k_int} > len(feature_path)={max_k}")
        if k_int in seen:
            continue
        seen.add(k_int)
        deduped.append(k_int)

    if not deduped:
        raise ValueError("k_grid contains no valid values")

    return deduped


def _build_splits(
    n: int,
    splitter: Any,
    *,
    random_state: int,
    val_frac: float,
    groups: Optional[np.ndarray],
    y: Optional[np.ndarray] = None,
) -> List[tuple[np.ndarray, np.ndarray]]:
    """Build train/validation splits from splitter or default holdout."""
    def _coerce_split_pair(pair: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("split entries must be (train_idx, val_idx)")
        train_idx = np.asarray(pair[0], dtype=np.int64)
        val_idx = np.asarray(pair[1], dtype=np.int64)
        if train_idx.ndim != 1 or val_idx.ndim != 1:
            raise ValueError("splitter indices must be 1D")
        if len(np.intersect1d(train_idx, val_idx)):
            raise ValueError("train and validation indices overlap")
        if not (len(train_idx) and len(val_idx)):
            raise ValueError("train and validation splits must be non-empty")
        if train_idx.min() < 0 or val_idx.min() < 0:
            raise ValueError("splitter indices must be non-negative")
        if train_idx.max() >= n or val_idx.max() >= n:
            raise ValueError("splitter indices out of range")
        return train_idx, val_idx

    def _is_single_split_pair(obj: Any) -> bool:
        if not isinstance(obj, (list, tuple)) or len(obj) != 2:
            return False
        try:
            first = np.asarray(obj[0])
            second = np.asarray(obj[1])
        except (TypeError, ValueError):
            return False
        return first.ndim == 1 and second.ndim == 1

    if splitter is None:
        if not (0.0 < val_frac < 1.0):
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
        if n < 2:
            raise ValueError("Need at least 2 rows for holdout splitting")
        order = np.arange(n)
        order = np.random.default_rng(random_state).permutation(order)
        cut = int(np.floor((1.0 - val_frac) * n))
        cut = max(1, min(cut, n - 1))
        return [(order[:cut], order[cut:])]

    if _is_single_split_pair(splitter):
        return [_coerce_split_pair(splitter)]

    if isinstance(splitter, (list, tuple)):
        splits = [_coerce_split_pair(pair) for pair in splitter]
        if not splits:
            raise ValueError("splitter iterable must contain at least one split")
        return splits

    if hasattr(splitter, "split"):
        # Splitter is expected to provide a .split(X, y, groups=None)-style API.
        data = np.empty((n, 1))
        # Pass the real target so stratified splitters actually stratify.
        y_split = np.zeros(n) if y is None else np.asarray(y).ravel()
        groups_arr = None if groups is None else np.asarray(groups).ravel()
        if groups_arr is not None and groups_arr.shape[0] != n:
            raise ValueError(f"groups has {groups_arr.shape[0]} rows but expected {n}")
        if groups_arr is not None and not _accepts_keyword(splitter.split, "groups"):
            raise TypeError(
                "groups were provided, but splitter.split does not accept a groups argument"
            )
        raw_splits = (
            splitter.split(data, y_split, groups=groups_arr)
            if groups_arr is not None
            else splitter.split(data, y_split)
        )
        splits = [_coerce_split_pair(pair) for pair in raw_splits]
        if not splits:
            raise ValueError("splitter object produced no splits")
        return splits

    raise TypeError(
        "splitter must be None, a splitter object with split(...), or (train_idx, val_idx)"
    )


def _impute_train_val(Xtr: np.ndarray, Xva: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute train/val arrays with train fold means."""
    means = np.nanmean(Xtr, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)

    Xtr_out = Xtr.copy()
    Xva_out = Xva.copy()

    tr_mask = ~np.isfinite(Xtr_out)
    if tr_mask.any():
        Xtr_out[tr_mask] = means[np.where(tr_mask)[1]]

    va_mask = ~np.isfinite(Xva_out)
    if va_mask.any():
        Xva_out[va_mask] = means[np.where(va_mask)[1]]

    return Xtr_out, Xva_out


def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    *,
    sample_weight: np.ndarray,
) -> float:
    """Compute lower-is-better metric."""
    if metric == "rmse":
        return float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight)))
    if metric == "mae":
        return float(np.average(np.abs(y_true - y_pred), weights=sample_weight))
    raise ValueError(
        "scoring must be 'rmse', 'mae', or a callable(y_true, y_pred, sample_weight)"
    )


def _to_estimator(
    *,
    estimator: Any,
    estimator_factory: Optional[EstimatorFactory],
) -> Any:
    """Build a fresh estimator instance for one fit/evaluate cycle."""
    if estimator_factory is not None:
        est = estimator_factory()
    elif estimator is not None:
        try:
            est = clone(estimator)
        except Exception:
            est = copy.deepcopy(estimator)
    else:
        est = make_pipeline(StandardScaler(), LinearRegression())

    if not hasattr(est, "fit") or not hasattr(est, "predict"):
        raise TypeError("estimator must have fit and predict methods")

    return est


def _accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    """Return whether a callable explicitly or generically accepts a keyword."""
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Cannot inspect {callable_obj!r} to determine support for {keyword!r}"
        ) from exc
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _fit_estimator(
    estimator: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
) -> None:
    if hasattr(estimator, "steps"):
        final_name, final_estimator = estimator.steps[-1]
        if _accepts_keyword(final_estimator.fit, "sample_weight"):
            estimator.fit(X_tr, y_tr, **{f"{final_name}__sample_weight": w_tr})
            return
        estimator.fit(X_tr, y_tr)
        return

    if _accepts_keyword(estimator.fit, "sample_weight"):
        estimator.fit(X_tr, y_tr, sample_weight=w_tr)
        return
    estimator.fit(X_tr, y_tr)


def _fit_predict_score(
    estimator: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    w_va: np.ndarray,
    scoring: Scoring,
    sample_weight_supplied: bool,
) -> float:
    try:
        _fit_estimator(estimator, X_tr, y_tr, w_tr)
        if is_sklearn_scorer(scoring):
            signed_score = score_with_sklearn_scorer(
                scoring,
                estimator,
                X_va,
                y_va,
                sample_weight=w_va if sample_weight_supplied else None,
            )
            score = -signed_score
            return float(score) if np.isfinite(score) else float("inf")
        y_pred = np.asarray(estimator.predict(X_va), dtype=np.float64).ravel()
        if y_pred.shape[0] != y_va.shape[0]:
            raise ValueError("estimator.predict returned wrong number of predictions")

        if callable(scoring):
            score = float(scoring(y_va, y_pred, w_va))
        else:
            score = _compute_metric(y_va, y_pred, scoring, sample_weight=w_va)

        if not np.isfinite(score):
            return float("inf")
        return float(score)
    except UnsupportedScorerSampleWeightError:
        raise
    except Exception as exc:
        warnings.warn(
            "Feature-path evaluation failed for one estimator fit/score and "
            f"recorded an infinite score: {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        return float("inf")


def evaluate_feature_path(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    feature_path: Iterable[Hashable],
    k_grid: Iterable[int],
    *,
    estimator: Any | None = None,
    estimator_factory: EstimatorFactory | None = None,
    scoring: Scoring = "rmse",
    splitter: Any | None = None,
    val_frac: float = 0.2,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> FeaturePathEvaluationResult:
    """Evaluate an ordered feature path over an explicit k grid.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix.
    y : array-like
        Regression target.
    feature_path : sequence of str or int
        Ordered feature names (DataFrame) or positional indices.
    k_grid : sequence of int
        Explicit candidate feature counts.
    estimator : estimator, optional
        Reusable sklearn-like estimator template to clone per (split, k).
    estimator_factory : callable, optional
        Factory called to create one estimator per (split, k). Mutually
        exclusive with ``estimator``; supplying both raises ``ValueError``.
    scoring : {'rmse', 'mae'}, sklearn scorer object, or callable
        Lower-is-better scoring. Callable signature is
        ``scoring(y_true, y_pred, sample_weight)``.
        Estimator-style sklearn scorers are negated into this result's
        historical lower-is-better score curve.
    splitter : optional
        If None, a simple random holdout split is used. Otherwise a splitter
        object with ``split(...)`` or ``(train_idx, val_idx)`` tuple.
    val_frac : float, default 0.2
        Holdout fraction when ``splitter`` is None.
    random_state : int, default 42
        Holdout RNG seed.
    sample_weight : ndarray, optional
        Row weights used in fits and scoring.
    groups : ndarray, optional
        Optional groups passed through to splitter objects that accept groups.

    Returns
    -------
    FeaturePathEvaluationResult
        Includes tested ks, evaluated scores, best-k, and diagnostics dataframe.

    See Also
    --------
    FeaturePathEvaluationResult : The object returned here.
    sift.select_k_auto : The rule-driven counterpart that picks its own grid.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import evaluate_feature_path
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(150, 4)), columns=list("abcd"))
    >>> y = 2.0 * X["a"] + X["b"] + 0.5 * rng.normal(size=150)
    >>> result = evaluate_feature_path(
    ...     X, y.to_numpy(), ["a", "b", "c", "d"], [1, 2, 3, 4], random_state=0
    ... )
    >>> result.k
    [1, 2, 3, 4]
    >>> result.best_k, result.features
    (2, ['a', 'b'])

    Any sklearn-like estimator and either of the built-in metrics can be
    supplied instead of the default:

    >>> from sklearn.ensemble import RandomForestRegressor
    >>> forest = evaluate_feature_path(
    ...     X, y.to_numpy(), ["a", "b", "c", "d"], [2, 4],
    ...     estimator=RandomForestRegressor(n_estimators=20, random_state=0),
    ...     scoring="mae", random_state=0,
    ... )
    >>> forest.diagnostics["scoring"].iloc[0]
    'mae'
    """
    metadata = resolve_row_metadata(
        X,
        groups=groups,
        sample_weight=sample_weight,
    )
    X = metadata.X
    groups = metadata.groups
    sample_weight = metadata.sample_weight
    if estimator is not None and estimator_factory is not None:
        raise ValueError("Pass either estimator or estimator_factory, not both")
    if not callable(scoring) and scoring not in {"rmse", "mae"}:
        raise ValueError(
            "scoring must be 'rmse', 'mae', an sklearn scorer object, or a "
            "callable(y_true, y_pred, sample_weight)"
        )

    if isinstance(X, pd.DataFrame):
        X_df = X
        path_names, path_positions = _resolve_feature_path(X_df, feature_path)
        X_path = X_df.iloc[:, path_positions].to_numpy(dtype=np.float64)
    else:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        X_df = pd.DataFrame(columns=[f"x{i}" for i in range(X_arr.shape[1])])
        path_names, path_positions = _resolve_feature_path(X_df, feature_path)
        X_path = X_arr[:, path_positions]

    y_arr = np.asarray(y).ravel()
    if y_arr.shape[0] != X_path.shape[0]:
        raise ValueError(f"X has {X_path.shape[0]} rows but y has {y_arr.shape[0]}")
    if not np.isfinite(y_arr).all():
        raise ValueError("y contains non-finite values")

    sample_weight_supplied = sample_weight is not None
    w_arr = ensure_weights(sample_weight, X_path.shape[0], normalize=True)

    k_values = _resolve_k_grid(k_grid, max_k=X_path.shape[1])
    splits = _build_splits(
        X_path.shape[0],
        splitter,
        random_state=random_state,
        val_frac=val_frac,
        groups=groups,
        y=y_arr,
    )

    raw_scores: dict[int, list[float]] = {k: [] for k in k_values}

    for train_idx, val_idx in splits:
        w_train = _split_weights(w_arr, train_idx, label="train")
        w_val = _split_weights(w_arr, val_idx, label="validation")

        X_train_full = X_path[train_idx]
        X_val_full = X_path[val_idx]
        y_train = y_arr[train_idx]
        y_val = y_arr[val_idx]

        X_train_full, X_val_full = _impute_train_val(X_train_full, X_val_full)

        for k in k_values:
            est = _to_estimator(estimator=estimator, estimator_factory=estimator_factory)
            score = _fit_predict_score(
                est,
                X_train_full[:, :k],
                y_train,
                w_train,
                X_val_full[:, :k],
                y_val,
                w_val,
                scoring=scoring,
                sample_weight_supplied=sample_weight_supplied,
            )
            raw_scores[k].append(score)

    means: dict[int, float] = {}
    stds: dict[int, float] = {}
    finite_counts: dict[int, int] = {}
    n_splits = len(splits)

    for k in k_values:
        values = np.asarray(raw_scores[k], dtype=np.float64)
        finite = values[np.isfinite(values)]
        finite_counts[k] = int(finite.size)
        if finite.size != values.size:
            means[k] = float("inf")
            stds[k] = (
                float(np.std(finite, ddof=0)) if finite.size > 1 else float("nan")
            )
            continue
        means[k] = float(np.mean(finite))
        stds[k] = float(np.std(finite, ddof=0)) if finite.size > 1 else 0.0

    best_k, best_score = best_score_from_dict(means, higher_is_better=False)
    if best_k == 0:
        best_features: list[str] = []
    else:
        best_features = path_names[:best_k]

    if is_sklearn_scorer(scoring):
        scoring_label = sklearn_scorer_label(scoring)
    elif isinstance(scoring, str):
        scoring_label = scoring
    else:
        module = getattr(scoring, "__module__", type(scoring).__module__)
        name = getattr(scoring, "__qualname__", type(scoring).__qualname__)
        scoring_label = f"legacy:{module}.{name}"

    diagnostics = pd.DataFrame(
        {
            "k": k_values,
            "score": [means[k] for k in k_values],
            "std": [stds[k] for k in k_values],
            "n_finite": [finite_counts[k] for k in k_values],
            "n_splits": [n_splits] * len(k_values),
            "best_score": [best_score] * len(k_values),
            "scoring": [scoring_label] * len(k_values),
        }
    )

    return FeaturePathEvaluationResult(
        feature_path=path_names,
        k=k_values,
        features=best_features,
        scores=means,
        best_k=best_k,
        diagnostics=diagnostics,
    )


__all__ = ["FeaturePathEvaluationResult", "evaluate_feature_path"]
