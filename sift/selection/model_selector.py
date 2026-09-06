"""Generic sklearn estimator-based feature selector."""

from __future__ import annotations

import inspect
from collections.abc import Hashable, Mapping, Sequence, Set
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import get_scorer, get_scorer_names
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.utils.validation import check_is_fitted

from sift._logging import logger
from sift._metadata import drop_fitted_metadata_columns, resolve_row_metadata
from sift._preprocess import reject_datetime_like_features
from sift._selector_compat import (
    check_fitted_column_identity,
    feature_names_array,
    inverse_selected_matrix,
    ordered_indices,
    reject_sparse,
    selector_tags,
    validate_fit_matrix,
    validate_output_order,
)
from sift.scoring import (
    UnsupportedScorerSampleWeightError,
    get_scoring,
    is_sklearn_scorer,
    score_with_sklearn_scorer,
    sklearn_scorer_label,
)
from sift.selection import orchestration as _selection_orchestration
from sift.selection.orchestration import SelectionBackend
from sift.selection.purged_cv import GroupPurgedTimeSeriesSplit, PurgedTimeSeriesSplit


_METHODS = frozenset({"rfe", "forward", "stability"})
_IMPORTANCE = frozenset({"auto", "coef", "feature_importances", "permutation"})
_GROUP_SPLITTER_TYPES: tuple[type, ...] = (GroupKFold, GroupShuffleSplit)
try:
    from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold

    _GROUP_SPLITTER_TYPES = _GROUP_SPLITTER_TYPES + (
        StratifiedGroupKFold,
        LeaveOneGroupOut,
        LeavePGroupsOut,
    )
except ImportError:  # pragma: no cover - older sklearn without these splitters
    pass


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return out


def _strict_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return out


def _coerce_feature_names(feature_names, *, argument: str = "feature_names") -> list[Hashable]:
    invalid_container = isinstance(
        feature_names,
        (str, bytes, bytearray, memoryview, Mapping, Set),
    )
    ndim = getattr(feature_names, "ndim", None)
    if invalid_container or (ndim is not None and ndim != 1):
        raise ValueError(
            f"{argument} must be an ordered, one-dimensional iterable of names"
        )
    try:
        names = list(feature_names)
    except TypeError as exc:
        raise ValueError(
            f"{argument} must be an ordered, one-dimensional iterable of names"
        ) from exc
    for name in names:
        try:
            hash(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{argument} entries must be hashable column labels") from exc
    return names


def _explicit_kwarg(callable_obj: Any, keyword: str) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    param = parameters.get(keyword)
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _has_var_keyword(callable_obj: Any) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _accepts_sample_weight(estimator: Any) -> bool:
    fit = getattr(estimator, "fit", None)
    if fit is None:
        return False
    if _explicit_kwarg(fit, "sample_weight"):
        return True
    if not _has_var_keyword(fit):
        return False
    nested = getattr(estimator, "estimator", None)
    if nested is not None and _explicit_kwarg(getattr(nested, "fit", None), "sample_weight"):
        return True
    if hasattr(estimator, "named_steps"):
        try:
            last = estimator[-1]
        except Exception:
            return False
        return _explicit_kwarg(getattr(last, "fit", None), "sample_weight")
    return False


def _row_take(values: Any, idx: np.ndarray) -> Any:
    if values is None:
        return None
    positions = np.asarray(idx, dtype=np.int64)
    if isinstance(values, pd.DataFrame):
        return values.iloc[positions]
    if isinstance(values, pd.Series):
        return values.iloc[positions]
    return np.asarray(values)[positions]


def _column_take(X: Any, columns: np.ndarray) -> Any:
    cols = np.asarray(columns, dtype=np.int64)
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, cols]
    return np.asarray(X)[:, cols]


def _n_features(X: Any) -> int:
    return int(X.shape[1])


def _n_rows(X: Any) -> int:
    return int(X.shape[0])


def _validate_1d_aligned(values: Any, n_rows: int, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"{name} has {int(arr.shape[0])} rows but X has {n_rows}")
    return arr


def _validate_sample_weight(sample_weight: Any, n_rows: int) -> np.ndarray:
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != n_rows:
        raise ValueError(
            f"sample_weight has {w.shape[0]} elements but X has {n_rows} rows"
        )
    if not np.isfinite(w).all():
        raise ValueError("sample_weight must be finite")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative")
    if not np.any(w > 0):
        raise ValueError("sample_weight must contain at least one positive value")
    return w


def _as_y(y: Any, n_rows: int) -> np.ndarray:
    if y is None:
        raise ValueError("y is required for ModelSelector")
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("ModelSelector does not accept multi-column y")
        y = y.iloc[:, 0]
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if int(arr.shape[0]) != int(n_rows):
        raise ValueError(f"y has {int(arr.shape[0])} rows but X has {n_rows}")
    return arr


def _unwrap_estimator(estimator: Any) -> Any:
    current = estimator
    seen: set[int] = set()
    while True:
        ident = id(current)
        if ident in seen:
            return current
        seen.add(ident)
        best = getattr(current, "best_estimator_", None)
        if best is not None and best is not current:
            current = best
            continue
        if hasattr(current, "named_steps"):
            try:
                nxt = current[-1]
            except Exception:
                return current
            if nxt is current:
                return current
            current = nxt
            continue
        return current


def _step_mixes_features(step: Any) -> bool:
    if hasattr(step, "named_steps"):
        return any(_step_mixes_features(child) for child in step.named_steps.values())
    transformers = getattr(step, "transformers_", None)
    if transformers is not None:
        for item in transformers:
            trans = item[1] if len(item) > 1 else None
            if trans not in (None, "drop", "passthrough") and _step_mixes_features(trans):
                return True
        remainder = getattr(step, "remainder", None)
        if remainder not in (None, "drop", "passthrough") and _step_mixes_features(
            remainder
        ):
            return True
        return False
    components = getattr(step, "components_", None)
    if components is None:
        return False
    arr = np.asarray(components)
    return arr.ndim == 2


def _output_names_match_raw(input_names: list[str], output_names: list[str]) -> bool:
    if len(input_names) != len(output_names):
        return False
    for raw, out in zip(input_names, output_names):
        if out == raw:
            continue
        if out.endswith("__" + raw):
            continue
        return False
    return True


def _raw_column_names(X: Any) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return [str(name) for name in X.columns]
    return [f"x{i}" for i in range(_n_features(X))]


_UNALIGNED_IMPORTANCE = (
    "ModelSelector cannot assign extracted coefficients or "
    "feature_importances_ to the current raw columns because the fitted "
    "pipeline preprocessing reorders, expands, mixes, or does not report "
    "aligned feature names. Pass importance='permutation' or a callable "
    "that returns one value per current raw column."
)


def _require_preprocess_aligned(preprocess: Any, input_names: list[str]) -> list[str]:
    steps = getattr(preprocess, "steps", None)
    if steps is not None and any(_step_mixes_features(step) for _, step in steps):
        raise ValueError(_UNALIGNED_IMPORTANCE)
    if _step_mixes_features(preprocess):
        raise ValueError(_UNALIGNED_IMPORTANCE)
    getter = getattr(preprocess, "get_feature_names_out", None)
    if not callable(getter):
        raise ValueError(_UNALIGNED_IMPORTANCE)
    try:
        output = getter(input_names)
    except Exception:
        raise ValueError(_UNALIGNED_IMPORTANCE) from None
    output_names = [str(name) for name in np.asarray(output).reshape(-1)]
    if not _output_names_match_raw(input_names, output_names):
        raise ValueError(_UNALIGNED_IMPORTANCE)
    return output_names


def _require_raw_aligned_extraction(estimator: Any, X: Any) -> None:
    names = _raw_column_names(X)
    current = estimator
    seen: set[int] = set()
    while True:
        ident = id(current)
        if ident in seen:
            return
        seen.add(ident)
        best = getattr(current, "best_estimator_", None)
        if best is not None and best is not current:
            current = best
            continue
        if not hasattr(current, "named_steps"):
            return
        try:
            n_steps = len(current.steps)
            nxt = current[-1]
        except Exception:
            raise ValueError(_UNALIGNED_IMPORTANCE) from None
        if n_steps > 1:
            names = _require_preprocess_aligned(current[:-1], names)
        if nxt is current:
            return
        current = nxt


def _is_precomputed_cv(cv: Any) -> bool:
    if cv is None or isinstance(cv, (str, bytes, int, np.integer, bool, np.bool_)):
        return False
    return not hasattr(cv, "split")


def _validate_index_pair(train, val, n_rows: int, *, fold: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train)
    val_idx = np.asarray(val)
    for name, idx in (("train", train_idx), ("validation", val_idx)):
        if idx.ndim != 1:
            raise ValueError(f"precomputed {name} indices for fold {fold} must be 1-D")
        if idx.size == 0:
            raise ValueError(f"precomputed {name} indices for fold {fold} are empty")
        if idx.dtype == np.bool_ or idx.dtype.kind == "b":
            raise ValueError(
                f"precomputed {name} indices for fold {fold} must be integer positions, "
                "not a boolean mask"
            )
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(
                f"precomputed {name} indices for fold {fold} must be integer positions"
            )
        if np.any(idx < 0) or np.any(idx >= n_rows):
            raise ValueError(
                f"precomputed {name} indices for fold {fold} are outside [0, {n_rows})"
            )
        if np.unique(idx).size != idx.size:
            raise ValueError(f"precomputed {name} indices for fold {fold} contain duplicates")
    if np.intersect1d(train_idx, val_idx, assume_unique=True).size:
        raise ValueError(f"precomputed fold {fold} has overlapping train and validation indices")
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def _validate_precomputed_cv(cv: Any, n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        raw_pairs = list(cv)
    except TypeError as exc:
        raise TypeError("cv must be a splitter, fold count, or sequence of index pairs") from exc
    if not raw_pairs:
        raise ValueError("precomputed cv is empty")
    for fold, pair in enumerate(raw_pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each precomputed cv item must be a (train, validation) pair")
        pairs.append(_validate_index_pair(pair[0], pair[1], n_rows, fold=fold))
    return pairs


def _split_requests_groups(splitter: Any) -> bool:
    getter = getattr(splitter, "get_metadata_routing", None)
    if callable(getter):
        try:
            requests = getattr(getattr(getter(), "split", None), "requests", None)
            if isinstance(requests, Mapping) and requests.get("groups") is True:
                return True
        except Exception:
            pass
    request = getattr(splitter, "__metadata_request__split", None)
    return isinstance(request, Mapping) and request.get("groups") is True


def _splitter_uses_groups(splitter: Any) -> bool:
    if isinstance(splitter, GroupPurgedTimeSeriesSplit):
        return True
    if isinstance(splitter, PurgedTimeSeriesSplit):
        return False
    if isinstance(splitter, _GROUP_SPLITTER_TYPES):
        return True
    return _split_requests_groups(splitter)


def _splitter_uses_time(splitter: Any) -> bool:
    return isinstance(splitter, (PurgedTimeSeriesSplit, GroupPurgedTimeSeriesSplit))


def _resolve_counts(
    n_features: int,
    n_features_to_select: Any,
    min_features_to_select: int,
    step: int,
    *,
    method: str,
) -> tuple[list[int] | None, int | None]:
    """Return (search_grid, explicit_or_cap).

    An explicit integer is the RFE/forward count or the stability cap.
    A sequence is the searched count grid (RFE/forward only). ``None``
    searches every admissible count, or keeps every stability passer.
    """
    min_k = min(min_features_to_select, n_features)
    if min_k < 1:
        raise ValueError("min_features_to_select must be >= 1")
    if n_features_to_select is None:
        if method == "stability":
            return None, n_features
        grid = list(range(min_k, n_features + 1, step))
        if grid[-1] != n_features:
            grid.append(n_features)
        return grid, None
    if isinstance(n_features_to_select, (bool, np.bool_)):
        raise ValueError("n_features_to_select must be an integer, a sequence of integers, or None")
    if isinstance(n_features_to_select, (int, np.integer)):
        k = int(n_features_to_select)
        if k < 1:
            raise ValueError("n_features_to_select must be >= 1")
        if k > n_features:
            raise ValueError(
                f"n_features_to_select={k} exceeds n_features={n_features}"
            )
        return None, k
    try:
        raw_values = list(n_features_to_select)
    except TypeError as exc:
        raise ValueError(
            "n_features_to_select must be an integer, a sequence of integers, or None"
        ) from exc
    if method == "stability":
        raise ValueError(
            "method='stability' takes a single integer cap or None, not a searched count grid"
        )
    if not raw_values:
        raise ValueError("n_features_to_select sequence is empty")
    values = [_strict_int(value, name="n_features_to_select") for value in raw_values]
    for k in values:
        if k < 1 or k > n_features:
            raise ValueError(
                "searched feature counts must be integers in "
                f"[1, {n_features}]; got {k}"
            )
    grid = sorted(set(values))
    return grid, None


def _parsimonious_k(
    scores: Mapping[int, float],
    *,
    tolerance: float,
    patience: int,
) -> int:
    finite = {int(k): float(v) for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        raise ValueError("no finite validation scores were produced for any feature count")
    best_k = max(finite, key=lambda k: (finite[k], -k))
    best_score = finite[best_k]
    delta = abs(best_score) * float(tolerance)
    chosen = int(best_k)
    misses = 0
    for k in sorted((k for k in finite if k < best_k), reverse=True):
        if finite[k] >= best_score - delta:
            chosen = int(k)
            misses = 0
        else:
            misses += 1
            if misses >= patience:
                break
    return chosen


class _GenericModelBackend(SelectionBackend):
    """Generic sklearn-ranking backend on the shared F6 selection runner."""

    def __init__(self, selector: Any):
        self.selector = selector

    def prepare(self, X, y, **context: Any) -> dict[str, Any]:
        return {"X": X, "y": y, **context}

    def evaluate(self, prepared: dict[str, Any]) -> dict[str, Any]:
        ranking, frequencies, scores_by_k, scores_se, chosen_k = self.selector._select_on_rows(
            prepared["X"],
            prepared["y"],
            sample_weight=prepared["sample_weight"],
            groups=prepared["groups"],
            time=prepared["time"],
            event_end=prepared["event_end"],
            search_grid=prepared["search_grid"],
            explicit_k=prepared["explicit_k"],
            allow_precomputed=True,
            cv=self.selector.cv,
            precomputed_pairs=prepared["precomputed_pairs"],
        )
        return {
            "ranking": ranking,
            "frequencies": frequencies,
            "scores_by_k": scores_by_k,
            "scores_se": scores_se,
            "chosen_k": chosen_k,
        }

    def choose(self, prepared: dict[str, Any], evaluated: dict[str, Any]) -> dict[str, Any]:
        return {"chosen_k": int(evaluated["chosen_k"])}

    def finalize(self, prepared: dict[str, Any], evaluated: dict[str, Any], chosen: dict[str, Any]):
        selector = self.selector
        ranking = evaluated["ranking"]
        chosen_k = chosen["chosen_k"]
        names = prepared["names"]
        n_features = int(prepared["n_features"])
        selected = np.asarray(ranking[: int(chosen_k)], dtype=np.int64)
        selector.n_features_in_ = n_features
        selector.feature_names_in_ = feature_names_array(names)
        selector.selected_indices_ = selected
        selector.selected_features_ = [names[int(i)] for i in selected]
        selector.n_features_selected_ = int(selected.size)
        selector.n_features_to_select_ = int(chosen_k)
        selector.ranking_ = np.asarray(ranking, dtype=np.int64)
        selector.scores_by_k_ = evaluated["scores_by_k"]
        selector.scores_by_k_se_ = evaluated["scores_se"]
        selector.nested_scores_ = prepared["nested_scores"]
        selector.nested_fold_diagnostics_ = prepared["nested_fold_diagnostics"]
        selector.selection_frequencies_ = evaluated["frequencies"]
        selector._n_rows_original_ = int(prepared["n_rows"])
        selector._fit_used_sample_weight_ = prepared["sample_weight"] is not None
        selector._fit_used_groups_ = prepared["groups"] is not None
        selector._fit_used_time_ = prepared["time"] is not None
        selector._fit_configured_options_ = selector._snapshot_fit_options(
            precomputed_pairs=prepared["precomputed_pairs"]
        )
        if bool(selector.verbose):
            logger.info(
                "ModelSelector method=%s selected %s/%s features",
                selector.method,
                int(chosen_k),
                n_features,
            )
        return selector


class ModelSelector(SelectorMixin, BaseEstimator):
    """Generic RFE, forward, or stability selector around a cloned estimator.

    ``estimator`` is cloned for every fit and fold. The caller's instance is
    not mutated. Feature ranking uses only the current training rows;
    validation rows score candidate *counts*, they do not choose which
    columns enter a step. Opt-in ``nested=True`` reruns that whole search
    inside every outer training fold and reports outer-validation scores as
    independent evidence, not as the inner curve used to pick ``k``.

    Parameters
    ----------
    estimator : estimator
        Cloneable sklearn-style estimator or pipeline. Pass an instance, not
        a class. Pipelines must fit inside each training fold; they are not
        pre-fit on all rows.
    method : {'rfe', 'forward', 'stability'}, default 'rfe'
        ``'rfe'`` recursively drops the lowest training importance.
        ``'forward'`` adds the unused column that most improves the configured
        training-fold criterion when fitted together with the already
        selected columns. ``'stability'`` draws row subsets, runs the
        selector's count-search on each draw, then keeps genuine frequency
        threshold-passers up to ``n_features_to_select`` (a cap, not padding).
    n_features_to_select : int, sequence of int, or None, default None
        Explicit count for RFE/forward, a searched count grid, or ``None`` to
        search every admissible count. For ``method='stability'`` this is a
        cap on threshold-passers, or ``None`` to keep all passers. A sequence
        is rejected for stability.
    min_features_to_select : int, default 1
        Smallest count in an automatic search grid and the RFE floor.
    step : int, default 1
        Features dropped per RFE iteration, and the stride of an automatic
        count grid.
    scoring : str, sklearn scorer, or None, default None
        Higher-is-better. ``None`` uses ``estimator.score``. String names may
        be SIFT scorers (``r2``, ``neg_mse``, ``accuracy``, ...) or sklearn
        scorer names. Sklearn scorer objects keep their signed convention.
        Weights are forwarded or rejected; they are never dropped.
    cv : splitter, int, precomputed pairs, or None, default None
        Count-search and nested outer folds. ``None`` uses
        ``GroupPurgedTimeSeriesSplit`` when both ``groups`` and ``time`` are
        given, ``PurgedTimeSeriesSplit`` when only ``time`` is given,
        ``GroupKFold`` when only ``groups`` is given, and shuffled ``KFold``
        otherwise. An integer is that family's fold count. Precomputed
        ``(train, validation)`` pairs must be integer, disjoint, in-range,
        and nonempty. Nested count search and ``method='stability'`` require
        a reusable splitter or integer fold count; precomputed pairs cannot
        be reused on a resampled or outer-train subset. Explicit-k
        non-stability nested evaluation may still use precomputed outer
        pairs. Full-data precomputed pairs are never reused inside an outer
        training subset.
    nested : bool, default False
        If True, rerun selection and count search on each outer training
        fold, refit the chosen subset on that fold, and score only outer
        validation. Final full-data selection is separate.
    importance : {'auto', 'coef', 'feature_importances', 'permutation'} or callable, default 'auto'
        Training-fold importance. ``'auto'`` uses ``feature_importances_`` or
        mean absolute ``coef_``. ``'permutation'`` is permutation importance
        on the training fold. A callable is ``importance(fitted) -> ndarray``
        aligned to the current raw columns. Direct coefficient extraction
        walks each fitted pipeline segment on the unwrap path, including a
        nested pipeline used as the final estimator. Alignment uses reported
        ``get_feature_names_out`` names and known mixing structure such as
        PCA ``components_``; in-place scalers such as ``StandardScaler``
        qualify. The guard trusts truthful name reporting and does not
        semantically inspect arbitrary user transform bodies. A
        ``FunctionTransformer`` that reorders values while declaring
        ``feature_names_out="one-to-one"`` violates that transform's contract
        and is not detected. Reported reordering, expansion, mixing, or
        unknown naming is rejected rather than guessed; use
        ``importance='permutation'`` or a raw-aligned callable.
    threshold : float, default 0.6
        Stability frequency cutoff in ``[0, 1]``. Ignored for RFE/forward.
    n_resamples : int, default 20
        Stability row draws. Ignored for RFE/forward.
    random_state : int, default 0
        Seed for shuffled KFold, permutation importance, and stability draws.
    parsimony_tolerance : float, default 0.0
        Relative score window below the best count. ``0`` keeps the smallest
        count tied with the best score.
    selection_patience : int, default 1
        Consecutive misses allowed while walking down from the best count.
    output_order : {'legacy', 'original'}, default 'legacy'
        ``'legacy'`` is discovery order (RFE/forward path, or frequency then
        index for stability). ``'original'`` is ascending fitted position.
    verbose : bool, default False
        Log the method and selected count at INFO.

    Attributes
    ----------
    selected_features_ : list
        Selected raw feature names.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted raw matrix, in discovery order.
    n_features_to_select_ : int
        Selected count after search, parsimony, or the stability cap.
    scores_by_k_ : dict or None
        Mean inner validation score per searched count. Absent for explicit
        counts and for stability. This is the feature-count-selection curve,
        not nested outer evidence.
    nested_scores_ : ndarray or None
        Outer-validation scores when ``nested=True``. Never used to choose
        ``k``.
    nested_fold_diagnostics_ : list of dict or None
        Per-outer-fold nested scoring notes. Empty stability selections
        record an intercept-only dummy baseline; nonempty folds keep the
        configured estimator.
    ranking_ : ndarray
        Discovery order. For ``method='forward'`` this is only the evaluated
        prefix, not an unused full path.
    selection_frequencies_ : ndarray or None
        Stability frequencies over the raw width. Not padded.

    See Also
    --------
    PurgedTimeSeriesSplit : Default time-aware count/outer folds.
    GroupPurgedTimeSeriesSplit : Default group+time folds.
    Stabilized : Frequency wrapper around another *selector*.
    catboost_select : CatBoost preset on the same internal ``run_selection``
        contract, with native SHAP/Pool ranking rather than this class.

    Notes
    -----
    ``fit(X, y, sample_weight=..., groups=..., time=..., event_end=...)``.
    Unused metadata is rejected. ``sample_weight`` is sliced onto the same
    rows as ``X`` for estimator fits, scorers, and resamples. Estimators or
    scorers that cannot honor weights raise. Duplicate column labels raise.
    Routed ``groups``/``time``/``event_end``/``sample_weight`` default to
    unrequested (``False``): a grouped outer ``cross_validate`` can keep
    groups on the splitter while a fixed-k fit ignores them. Call
    ``set_fit_request(groups=True)`` (and the same for ``time`` or
    ``sample_weight``) when those arrays should be forwarded into this
    selector's own group/time-aware search. Direct ``fit(..., groups=)``
    still rejects unused arrays. ``importance='auto'`` / ``'coef'`` /
    ``'feature_importances'`` extract from the unwrapped final estimator only
    after each pipeline segment on the unwrap path reports raw-column
    alignment. The check uses reported names and known mixing structure; it
    does not inspect arbitrary user-defined transform bodies.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from sift import ModelSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 4))
    >>> y = X[:, 0] + 0.8 * X[:, 1] + 0.05 * rng.normal(size=40)
    >>> sel = ModelSelector(Ridge(), n_features_to_select=2, cv=2, random_state=0)
    >>> sel.fit(X, y).selected_indices_.tolist()
    [0, 1]
    """

    __metadata_request__fit = {
        "sample_weight": False,
        "groups": False,
        "time": False,
        "event_end": False,
    }

    def __init__(
        self,
        estimator,
        *,
        method: str = "rfe",
        n_features_to_select: int | Sequence[int] | None = None,
        min_features_to_select: int = 1,
        step: int = 1,
        scoring: Any = None,
        cv: Any = None,
        nested: bool = False,
        importance: Any = "auto",
        threshold: float = 0.6,
        n_resamples: int = 20,
        random_state: int = 0,
        parsimony_tolerance: float = 0.0,
        selection_patience: int = 1,
        output_order: str = "legacy",
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.method = method
        self.n_features_to_select = n_features_to_select
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.scoring = scoring
        self.cv = cv
        self.nested = nested
        self.importance = importance
        self.threshold = threshold
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.parsimony_tolerance = parsimony_tolerance
        self.selection_patience = selection_patience
        self.output_order = output_order
        self.verbose = verbose

    def _more_tags(self):
        return selector_tags({}, non_deterministic=False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return selector_tags(tags)

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        event_end=None,
    ):
        """Learn a subset from cloned estimators on training rows only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense feature matrix or DataFrame. Raw column identities are
            preserved. Sparse input is rejected.
        y : array-like of shape (n_samples,)
            Target. Required.
        sample_weight : array-like of shape (n_samples,), optional
            Per-row weights. Sliced with CV/resample indices and passed to
            estimator ``fit`` and scoring, or rejected when unsupported.
        groups : array-like, optional
            Group labels for group-aware CV. Rejected when unused.
        time : array-like, optional
            Timestamps for purged CV. Rejected when unused.
        event_end : array-like, optional
            Information-interval ends for purged CV. Requires ``time``.

        Returns
        -------
        self
            Fitted selector.
        """
        self._clear_fit_state()
        try:
            return self._fit_impl(
                X,
                y,
                sample_weight=sample_weight,
                groups=groups,
                time=time,
                event_end=event_end,
            )
        except Exception:
            self._clear_fit_state()
            raise

    def _validate_estimator(self) -> None:
        estimator = self.estimator
        if isinstance(estimator, type):
            raise TypeError("estimator must be an instance, not a class")
        if not hasattr(estimator, "fit"):
            raise TypeError("estimator must expose fit")
        try:
            clone(estimator)
        except Exception as exc:
            raise TypeError("estimator must be cloneable by sklearn.base.clone") from exc

    def _validate_options(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                "method must be 'rfe', 'forward', or 'stability'; "
                f"got {self.method!r}"
            )
        if not isinstance(self.nested, (bool, np.bool_)):
            raise TypeError("nested must be a boolean")
        if not isinstance(self.verbose, (bool, np.bool_)):
            raise TypeError("verbose must be a boolean")
        _strict_int(self.min_features_to_select, name="min_features_to_select", minimum=1)
        _strict_int(self.step, name="step", minimum=1)
        _strict_int(self.n_resamples, name="n_resamples", minimum=1)
        _strict_int(self.random_state, name="random_state", minimum=0)
        _strict_int(self.selection_patience, name="selection_patience", minimum=1)
        _strict_float(self.parsimony_tolerance, name="parsimony_tolerance", minimum=0.0)
        threshold = _strict_float(self.threshold, name="threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        validate_output_order(self.output_order)
        if not callable(self.importance) and self.importance not in _IMPORTANCE:
            raise ValueError(
                "importance must be 'auto', 'coef', 'feature_importances', "
                "'permutation', or a callable"
            )
        if self.method != "stability":
            if self.threshold != 0.6:
                raise ValueError("threshold is only used with method='stability'")
            if self.n_resamples != 20:
                raise ValueError("n_resamples is only used with method='stability'")

    def _describe_scoring(self) -> Any:
        scoring = self.scoring
        if scoring is None:
            return None
        if isinstance(scoring, str):
            return scoring
        if is_sklearn_scorer(scoring):
            return sklearn_scorer_label(scoring)
        return "callable"

    def _describe_cv(self, *, precomputed_pairs) -> Any:
        from sift.selection.reproducibility import describe_splitter

        if precomputed_pairs is not None:
            return {"type": "precomputed", "n_splits": int(len(precomputed_pairs))}
        cv = self.cv
        if cv is None:
            return None
        if isinstance(cv, (int, np.integer)) and not isinstance(cv, (bool, np.bool_)):
            return int(cv)
        if hasattr(cv, "get_params") or hasattr(cv, "split"):
            return describe_splitter(cv)
        return {"type": type(cv).__qualname__}

    def _snapshot_fit_options(self, *, precomputed_pairs) -> dict[str, Any]:
        from sift.selection.reproducibility import describe_estimator, snapshot_selector_kwargs

        return snapshot_selector_kwargs(
            {
                "estimator": describe_estimator(self.estimator),
                "method": self.method,
                "n_features_to_select": self.n_features_to_select,
                "min_features_to_select": int(self.min_features_to_select),
                "step": int(self.step),
                "scoring": self._describe_scoring(),
                "cv": self._describe_cv(precomputed_pairs=precomputed_pairs),
                "nested": bool(self.nested),
                "importance": self.importance
                if not callable(self.importance)
                else "callable",
                "threshold": float(self.threshold),
                "n_resamples": int(self.n_resamples),
                "random_state": int(self.random_state),
                "parsimony_tolerance": float(self.parsimony_tolerance),
                "selection_patience": int(self.selection_patience),
                "output_order": self.output_order,
            }
        )

    def _fit_impl(self, X, y, *, sample_weight, groups, time, event_end):
        self._validate_estimator()
        self._validate_options()
        metadata = resolve_row_metadata(X, groups=groups, time=time, sample_weight=sample_weight)
        X = metadata.X
        groups = metadata.groups
        time = metadata.time
        if sample_weight is None:
            sample_weight = metadata.sample_weight
        reject_datetime_like_features(X)
        validate_fit_matrix(X)
        was_dataframe = isinstance(X, pd.DataFrame)
        self._fit_input_kind_ = "dataframe" if was_dataframe else "positional"
        self._row_metadata_columns_ = metadata.extracted_columns
        self._fit_feature_names_generated_ = not was_dataframe

        if was_dataframe:
            column_index = pd.Index(list(X.columns), dtype=object, tupleize_cols=False)
            if column_index.duplicated().any():
                duplicates = column_index[column_index.duplicated()].unique().tolist()[:5]
                raise ValueError(
                    "Duplicate DataFrame column labels are not supported: "
                    f"{duplicates}. Rename columns before fitting."
                )
            names = list(X.columns)
        else:
            X = np.asarray(X)
            names = [f"x{i}" for i in range(int(X.shape[1]))]

        n_rows = _n_rows(X)
        n_features = _n_features(X)
        y_arr = _as_y(y, n_rows)
        if sample_weight is not None:
            sample_weight = _validate_sample_weight(sample_weight, n_rows)
            if not _accepts_sample_weight(self.estimator):
                raise TypeError(
                    "sample_weight was supplied but estimator.fit does not accept "
                    "sample_weight; use a weight-aware estimator or omit sample_weight"
                )
        if groups is not None:
            groups = _validate_1d_aligned(groups, n_rows, name="groups")
        if time is not None:
            time = _validate_1d_aligned(time, n_rows, name="time")
        if event_end is not None:
            if time is None:
                raise ValueError("event_end requires time")
            event_end = _validate_1d_aligned(event_end, n_rows, name="event_end")

        precomputed_pairs = None
        if _is_precomputed_cv(self.cv):
            precomputed_pairs = _validate_precomputed_cv(self.cv, n_rows)

        search_grid, explicit_k = _resolve_counts(
            n_features,
            self.n_features_to_select,
            int(self.min_features_to_select),
            int(self.step),
            method=str(self.method),
        )
        uses_cv = bool(self.nested) or str(self.method) == "stability" or search_grid is not None
        needs_reusable_inner = str(self.method) == "stability" or (
            bool(self.nested) and search_grid is not None
        )
        if needs_reusable_inner and precomputed_pairs is not None:
            raise ValueError(
                "nested count search and method='stability' require a reusable "
                "splitter with split() or an integer fold count; precomputed "
                "pairs cannot be reused on a local inner or resampled subset. "
                "method='stability' always runs inner count search, so "
                "precomputed pairs are never valid there. For non-stability "
                "nested evaluation, an explicit n_features_to_select skips "
                "inner CV and may use precomputed outer pairs"
            )
        if not uses_cv:
            unused = [
                name
                for name, value in (
                    ("groups", groups),
                    ("time", time),
                    ("event_end", event_end),
                )
                if value is not None
            ]
            if unused:
                raise ValueError(
                    "unused metadata "
                    + ", ".join(unused)
                    + "; this fit does not split rows. Pass nested=True or a "
                    "searched feature-count grid, or omit the unused arrays"
                )

        nested_scores = None
        nested_fold_diagnostics = None
        if bool(self.nested):
            nested_scores, nested_fold_diagnostics = self._nested_outer_scores(
                X,
                y_arr,
                sample_weight=sample_weight,
                groups=groups,
                time=time,
                event_end=event_end,
                search_grid=search_grid,
                explicit_k=explicit_k,
                precomputed_pairs=precomputed_pairs,
            )

        return _selection_orchestration.run_selection(
            _GenericModelBackend(self),
            X,
            y_arr,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
            event_end=event_end,
            search_grid=search_grid,
            explicit_k=explicit_k,
            precomputed_pairs=precomputed_pairs,
            names=names,
            n_features=n_features,
            n_rows=n_rows,
            nested_scores=nested_scores,
            nested_fold_diagnostics=nested_fold_diagnostics,
        )

    def _nested_outer_scores(
        self,
        X,
        y,
        *,
        sample_weight,
        groups,
        time,
        event_end,
        search_grid,
        explicit_k,
        precomputed_pairs,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        outer_pairs, _consumed = self._iter_splits(
            X,
            y,
            groups=groups,
            time=time,
            event_end=event_end,
            cv=self.cv,
            allow_precomputed=True,
            purpose="nested outer",
            precomputed_pairs=precomputed_pairs,
        )
        scores: list[float] = []
        fold_notes: list[dict[str, Any]] = []
        for fold_i, (train_idx, val_idx) in enumerate(outer_pairs):
            X_tr = _row_take(X, train_idx)
            y_tr = _row_take(y, train_idx)
            w_tr = _row_take(sample_weight, train_idx)
            g_tr = _row_take(groups, train_idx)
            t_tr = _row_take(time, train_idx)
            e_tr = _row_take(event_end, train_idx)
            inner_cv = None if precomputed_pairs is not None else self.cv
            fold_grid = search_grid
            if search_grid is not None:
                fold_grid = [k for k in search_grid if k <= _n_features(X_tr)]
            fold_explicit = explicit_k
            if explicit_k is not None:
                fold_explicit = min(int(explicit_k), _n_features(X_tr))
            ranking, _, _, _, chosen_k = self._select_on_rows(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                groups=g_tr,
                time=t_tr,
                event_end=e_tr,
                search_grid=fold_grid,
                explicit_k=fold_explicit,
                allow_precomputed=False,
                cv=inner_cv,
                precomputed_pairs=None,
            )
            chosen = np.asarray(ranking[: int(chosen_k)], dtype=np.int64)
            y_va = _row_take(y, val_idx)
            w_va = _row_take(sample_weight, val_idx)
            if chosen.size == 0:
                est, note = self._fit_empty_baseline(y_tr, sample_weight=w_tr)
                X_va = np.zeros((_n_rows(_row_take(X, val_idx)), 0), dtype=np.float64)
                note["fold"] = int(fold_i)
                fold_notes.append(note)
            else:
                est = clone(self.estimator)
                self._fit_estimator(
                    est,
                    _column_take(X_tr, chosen),
                    y_tr,
                    sample_weight=w_tr,
                )
                X_va = _column_take(_row_take(X, val_idx), chosen)
                fold_notes.append(
                    {
                        "fold": int(fold_i),
                        "empty_selection": False,
                        "baseline": None,
                    }
                )
            scores.append(
                self._score_estimator(
                    est,
                    X_va,
                    y_va,
                    sample_weight=w_va,
                )
            )
        return np.asarray(scores, dtype=np.float64), fold_notes

    def _select_on_rows(
        self,
        X,
        y,
        *,
        sample_weight,
        groups,
        time,
        event_end,
        search_grid,
        explicit_k,
        allow_precomputed: bool,
        cv: Any,
        precomputed_pairs=None,
        ranking_method: str | None = None,
    ):
        n_features = _n_features(X)
        method = str(ranking_method or self.method)
        if method == "stability":
            cap = n_features if explicit_k is None else int(explicit_k)
            frequencies, selected = self._stability_select(
                X,
                y,
                sample_weight=sample_weight,
                groups=groups,
                time=time,
                event_end=event_end,
                cap=cap,
                cv=cv,
            )
            ranking = list(selected) + [
                i for i in range(n_features) if i not in set(int(x) for x in selected)
            ]
            return ranking, frequencies, None, None, int(len(selected))

        if search_grid is None and explicit_k is not None:
            ranking = self._rank_features(
                X,
                y,
                sample_weight=sample_weight,
                min_keep=int(explicit_k),
                max_keep=int(explicit_k),
                ranking_method=method,
            )
            return ranking, None, None, None, int(explicit_k)

        grid = [k for k in (search_grid or []) if 1 <= k <= n_features]
        if not grid:
            raise ValueError("no valid feature counts remain to search on this fold")
        min_keep = min(grid)
        max_keep = max(grid)
        pairs, _ = self._iter_splits(
            X,
            y,
            groups=groups,
            time=time,
            event_end=event_end,
            cv=cv,
            allow_precomputed=allow_precomputed,
            purpose="feature-count search",
            precomputed_pairs=precomputed_pairs,
        )
        score_lists: dict[int, list[float]] = {k: [] for k in grid}
        for train_idx, val_idx in pairs:
            X_tr = _row_take(X, train_idx)
            y_tr = _row_take(y, train_idx)
            w_tr = _row_take(sample_weight, train_idx)
            ranking = self._rank_features(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                min_keep=min_keep,
                max_keep=max_keep,
                ranking_method=method,
            )
            X_va = _row_take(X, val_idx)
            y_va = _row_take(y, val_idx)
            w_va = _row_take(sample_weight, val_idx)
            for k in grid:
                chosen = np.asarray(ranking[: int(k)], dtype=np.int64)
                est = clone(self.estimator)
                self._fit_estimator(est, _column_take(X_tr, chosen), y_tr, sample_weight=w_tr)
                score_lists[k].append(
                    self._score_estimator(
                        est,
                        _column_take(X_va, chosen),
                        y_va,
                        sample_weight=w_va,
                    )
                )
        scores = {k: float(np.mean(vals)) for k, vals in score_lists.items()}
        se = {
            k: (
                float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                if len(vals) > 1
                else float("nan")
            )
            for k, vals in score_lists.items()
        }
        chosen_k = _parsimonious_k(
            scores,
            tolerance=float(self.parsimony_tolerance),
            patience=int(self.selection_patience),
        )
        ranking = self._rank_features(
            X,
            y,
            sample_weight=sample_weight,
            min_keep=int(chosen_k),
            max_keep=int(chosen_k),
            ranking_method=method,
        )
        return ranking, None, scores, se, int(chosen_k)

    def _iter_splits(
        self,
        X,
        y,
        *,
        groups,
        time,
        event_end,
        cv,
        allow_precomputed: bool,
        purpose: str,
        precomputed_pairs=None,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], set[str]]:
        n_rows = _n_rows(X)
        consumed: set[str] = set()
        if precomputed_pairs is not None:
            if not allow_precomputed:
                raise ValueError(
                    f"{purpose} cannot reuse full-dataset precomputed split indices "
                    "inside an outer training subset; pass a cloneable splitter"
                )
            if groups is not None or time is not None or event_end is not None:
                raise ValueError(
                    "precomputed cv does not consume groups/time/event_end; omit "
                    "unused metadata or pass a splitter that uses it"
                )
            return list(precomputed_pairs), consumed
        if cv is None:
            splitter = self._default_splitter(groups=groups, time=time, n_rows=n_rows)
        elif _is_precomputed_cv(cv):
            if not allow_precomputed:
                raise ValueError(
                    f"{purpose} cannot reuse full-dataset precomputed split indices "
                    "inside an outer training subset; pass a cloneable splitter"
                )
            if groups is not None or time is not None or event_end is not None:
                raise ValueError(
                    "precomputed cv does not consume groups/time/event_end; omit "
                    "unused metadata or pass a splitter that uses it"
                )
            return _validate_precomputed_cv(cv, n_rows), consumed
        elif isinstance(cv, (int, np.integer)) and not isinstance(cv, (bool, np.bool_)):
            n_splits = _strict_int(cv, name="cv", minimum=2)
            splitter = self._default_splitter(
                groups=groups, time=time, n_rows=n_rows, n_splits=n_splits
            )
        elif hasattr(cv, "split"):
            splitter = cv
        else:
            raise TypeError(
                "cv must be None, an integer fold count, a splitter with split(), "
                "or a sequence of (train, validation) index pairs"
            )

        uses_groups = _splitter_uses_groups(splitter)
        uses_time = _splitter_uses_time(splitter)
        if uses_groups:
            if groups is None:
                raise ValueError(f"{type(splitter).__name__} requires groups")
            consumed.add("groups")
        elif groups is not None:
            raise ValueError(
                f"{type(splitter).__name__} does not consume groups; omit groups "
                "or pass a group-aware splitter"
            )
        if uses_time:
            if time is None:
                raise ValueError(f"{type(splitter).__name__} requires time")
            consumed.add("time")
            if event_end is not None:
                consumed.add("event_end")
        else:
            if time is not None:
                raise ValueError(
                    f"{type(splitter).__name__} does not consume time; omit time "
                    "or pass a purged time-series splitter"
                )
            if event_end is not None:
                raise ValueError("event_end requires a purged time-series splitter")

        if isinstance(splitter, GroupPurgedTimeSeriesSplit):
            raw = splitter.split(X, y, groups=groups, time=time, event_end=event_end)
        elif isinstance(splitter, PurgedTimeSeriesSplit):
            raw = splitter.split(X, y, time=time, event_end=event_end)
        elif uses_groups:
            raw = splitter.split(X, y, groups)
        else:
            raw = splitter.split(X, y)
        pairs = [
            _validate_index_pair(train, val, n_rows, fold=fold)
            for fold, (train, val) in enumerate(raw)
        ]
        if not pairs:
            raise ValueError(f"{purpose} produced no splits")
        return pairs, consumed

    def _default_splitter(self, *, groups, time, n_rows: int, n_splits: int | None = None):
        if n_splits is None:
            n_splits = 3
        if time is not None and groups is not None:
            return GroupPurgedTimeSeriesSplit(n_splits=n_splits)
        if time is not None:
            return PurgedTimeSeriesSplit(n_splits=n_splits)
        if groups is not None:
            n_groups = int(np.unique(groups).size)
            splits = min(int(n_splits), n_groups)
            if splits < 2:
                raise ValueError("groups must contain at least two distinct labels for GroupKFold")
            return GroupKFold(n_splits=splits)
        splits = min(int(n_splits), max(2, n_rows // 2))
        if splits < 2:
            raise ValueError("not enough rows to build a default KFold")
        return KFold(
            n_splits=splits,
            shuffle=True,
            random_state=int(self.random_state),
        )

    def _routing_enabled(self) -> bool:
        try:
            from sklearn import get_config

            return bool(get_config().get("enable_metadata_routing"))
        except Exception:
            return False

    def _fit_estimator(self, estimator, X, y, *, sample_weight) -> Any:
        if sample_weight is None:
            estimator.fit(X, y)
            return estimator
        if hasattr(estimator, "steps"):
            if self._routing_enabled():
                estimator.fit(X, y, sample_weight=sample_weight)
                return estimator
            final_name, final_estimator = estimator.steps[-1]
            if _explicit_kwarg(getattr(final_estimator, "fit", None), "sample_weight"):
                estimator.fit(X, y, **{f"{final_name}__sample_weight": sample_weight})
                return estimator
            raise TypeError(
                "sample_weight was supplied but the pipeline's final estimator "
                "does not accept sample_weight"
            )
        estimator.fit(X, y, sample_weight=sample_weight)
        return estimator

    def _fit_empty_baseline(self, y, *, sample_weight) -> tuple[Any, dict[str, Any]]:
        from sklearn.base import is_classifier
        from sklearn.dummy import DummyClassifier, DummyRegressor

        if is_classifier(self.estimator):
            est = DummyClassifier(strategy="prior")
            note = {
                "empty_selection": True,
                "baseline": "DummyClassifier",
                "strategy": "prior",
            }
        else:
            est = DummyRegressor(strategy="mean")
            note = {
                "empty_selection": True,
                "baseline": "DummyRegressor",
                "strategy": "mean",
            }
        n_rows = int(np.asarray(y).reshape(-1).shape[0])
        X_empty = np.zeros((n_rows, 0), dtype=np.float64)
        self._fit_estimator(est, X_empty, np.asarray(y).reshape(-1), sample_weight=sample_weight)
        return est, note

    def _score_estimator(self, estimator, X, y, *, sample_weight) -> float:
        scoring = self.scoring
        if scoring is None:
            if sample_weight is None:
                if not hasattr(estimator, "score"):
                    raise TypeError(
                        "scoring=None requires estimator.score; pass scoring="
                    )
                return float(estimator.score(X, y))
            score_fn = getattr(estimator, "score", None)
            if score_fn is None or not _explicit_kwarg(score_fn, "sample_weight"):
                raise TypeError(
                    "sample_weight was supplied but estimator.score does not accept "
                    "sample_weight; pass a weight-aware scorer or omit sample_weight"
                )
            return float(estimator.score(X, y, sample_weight=sample_weight))
        if isinstance(scoring, str):
            if scoring in get_scorer_names():
                scoring = get_scorer(scoring)
            else:
                spec = get_scoring(scoring)
                y_arr = np.asarray(y)
                if sample_weight is None:
                    w = np.ones(y_arr.shape[0], dtype=np.float64)
                else:
                    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
                return float(spec(estimator, X, y_arr, w))
        if is_sklearn_scorer(scoring):
            y_arr = np.asarray(y)
            w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
            try:
                return float(
                    score_with_sklearn_scorer(
                        scoring, estimator, X, y_arr, sample_weight=w
                    )
                )
            except UnsupportedScorerSampleWeightError:
                raise
        raise TypeError(
            "scoring must be None, a SIFT/sklearn scorer name, or an sklearn scorer object"
        )

    def _rank_features(
        self,
        X,
        y,
        *,
        sample_weight,
        min_keep: int,
        max_keep: int | None = None,
        ranking_method: str | None = None,
    ) -> list[int]:
        n_features = _n_features(X)
        remaining = list(range(n_features))
        method = str(ranking_method or self.method)
        if method == "forward":
            needed = n_features if max_keep is None else min(int(max_keep), n_features)
            selected: list[int] = []
            leftover = list(remaining)
            while leftover and len(selected) < needed:
                best_j = leftover[0]
                best_score = -np.inf
                for j in leftover:
                    cols = np.asarray(selected + [j], dtype=np.int64)
                    est = clone(self.estimator)
                    self._fit_estimator(
                        est, _column_take(X, cols), y, sample_weight=sample_weight
                    )
                    score = self._score_estimator(
                        est, _column_take(X, cols), y, sample_weight=sample_weight
                    )
                    if score > best_score or (score == best_score and j < best_j):
                        best_score = float(score)
                        best_j = int(j)
                leftover.remove(best_j)
                selected.append(best_j)
            return selected
        dropped: list[int] = []
        current = list(remaining)
        min_keep = max(1, min(int(min_keep), n_features))
        step = int(self.step)
        while len(current) > min_keep:
            importances = self._feature_importance(
                X, y, columns=np.asarray(current, dtype=np.int64), sample_weight=sample_weight
            )
            order = np.argsort(importances, kind="mergesort")
            n_drop = min(step, len(current) - min_keep)
            drop_local = order[:n_drop]
            drop_set = {current[int(i)] for i in drop_local}
            for idx in drop_local:
                dropped.append(current[int(idx)])
            current = [c for c in current if c not in drop_set]
        if current:
            survivor_imp = self._feature_importance(
                X, y, columns=np.asarray(current, dtype=np.int64), sample_weight=sample_weight
            )
            current = [
                current[int(i)]
                for i in np.argsort(-survivor_imp, kind="mergesort")
            ]
        ranking = current + list(reversed(dropped))
        return ranking

    def _feature_importance(self, X, y, *, columns: np.ndarray, sample_weight) -> np.ndarray:
        X_sub = _column_take(X, columns)
        est = clone(self.estimator)
        self._fit_estimator(est, X_sub, y, sample_weight=sample_weight)
        kind = self.importance
        if callable(kind):
            values = np.asarray(kind(est), dtype=np.float64).reshape(-1)
            if values.size != columns.size:
                raise ValueError(
                    "importance callable must return one value per current feature"
                )
            return values
        if kind != "permutation":
            _require_raw_aligned_extraction(est, X_sub)
        fitted = _unwrap_estimator(est)
        if kind in {"auto", "feature_importances"} and hasattr(fitted, "feature_importances_"):
            values = np.asarray(fitted.feature_importances_, dtype=np.float64).reshape(-1)
        elif kind in {"auto", "coef"} and hasattr(fitted, "coef_"):
            coef = np.asarray(fitted.coef_, dtype=np.float64)
            if coef.ndim == 1:
                values = np.abs(coef)
            else:
                values = np.mean(np.abs(coef), axis=tuple(range(coef.ndim - 1)))
        elif kind == "permutation" or (
            kind == "auto" and not hasattr(fitted, "coef_") and not hasattr(fitted, "feature_importances_")
        ):
            if kind == "auto":
                raise TypeError(
                    "importance='auto' needs coef_ or feature_importances_ on the "
                    "fitted estimator; pass importance='permutation' or a callable"
                )
            from sklearn.inspection import permutation_importance

            result = permutation_importance(
                est,
                X_sub,
                y,
                n_repeats=5,
                random_state=int(self.random_state),
                scoring=self._permutation_scoring(),
                sample_weight=None
                if sample_weight is None
                else np.asarray(sample_weight, dtype=np.float64),
            )
            values = np.asarray(result.importances_mean, dtype=np.float64).reshape(-1)
        else:
            raise TypeError(
                f"importance={kind!r} is not available on the fitted estimator"
            )
        if values.size != columns.size:
            raise ValueError("extracted importance length does not match the current columns")
        if not np.isfinite(values).all():
            values = np.where(np.isfinite(values), values, -np.inf)
        return values

    def _permutation_scoring(self):
        scoring = self.scoring
        if scoring is None:
            return None
        if isinstance(scoring, str) and scoring in get_scorer_names():
            return scoring
        if is_sklearn_scorer(scoring):
            return scoring
        spec = get_scoring(scoring)

        def scorer(estimator, X, y, sample_weight=None):
            y_arr = np.asarray(y)
            if sample_weight is None:
                weights = np.ones(y_arr.shape[0], dtype=np.float64)
            else:
                weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            return float(spec(estimator, X, y_arr, weights))

        return scorer

    def _stability_select(
        self,
        X,
        y,
        *,
        sample_weight,
        groups,
        time,
        event_end,
        cap: int,
        cv: Any,
    ):
        n_rows = _n_rows(X)
        n_features = _n_features(X)
        cap = max(1, min(int(cap), n_features))
        min_k = min(int(self.min_features_to_select), n_features)
        search_grid = list(range(min_k, n_features + 1, int(self.step)))
        if search_grid[-1] != n_features:
            search_grid.append(n_features)
        rngs = np.random.SeedSequence(int(self.random_state)).spawn(int(self.n_resamples))
        counts = np.zeros(n_features, dtype=np.int64)
        size = max(2, n_rows // 2)
        for child in rngs:
            rng = np.random.default_rng(child)
            idx = rng.choice(n_rows, size=min(size, n_rows), replace=False)
            idx = np.sort(idx)
            ranking, _, _, _, chosen_k = self._select_on_rows(
                _row_take(X, idx),
                _row_take(y, idx),
                sample_weight=_row_take(sample_weight, idx),
                groups=_row_take(groups, idx),
                time=_row_take(time, idx),
                event_end=_row_take(event_end, idx),
                search_grid=search_grid,
                explicit_k=None,
                allow_precomputed=False,
                cv=cv,
                precomputed_pairs=None,
                ranking_method="rfe",
            )
            for col in ranking[: int(chosen_k)]:
                counts[int(col)] += 1
        frequencies = counts.astype(np.float64) / float(self.n_resamples)
        passed = np.flatnonzero(frequencies >= float(self.threshold))
        order = np.argsort(-frequencies[passed], kind="mergesort")
        selected = passed[order][:cap]
        return frequencies, np.asarray(selected, dtype=np.int64)

    def _clear_fit_state(self) -> None:
        for attr in (
            "_fit_configured_options_",
            "_fit_feature_names_generated_",
            "_fit_input_kind_",
            "_fit_used_groups_",
            "_fit_used_sample_weight_",
            "_fit_used_time_",
            "_n_rows_original_",
            "_row_metadata_columns_",
            "feature_names_in_",
            "n_features_in_",
            "n_features_selected_",
            "n_features_to_select_",
            "nested_fold_diagnostics_",
            "nested_scores_",
            "ranking_",
            "scores_by_k_",
            "scores_by_k_se_",
            "selected_features_",
            "selected_indices_",
            "selection_frequencies_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _output_indices(self) -> np.ndarray:
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        return ordered_indices(self.selected_indices_, self.output_order)

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return selected-feature mask (default) or indices (indices=True)."""
        if indices:
            return self._output_indices()
        return self._get_support_mask()

    def transform(self, X):
        """Reduce X to selected raw features."""
        check_is_fitted(
            self, ["selected_indices_", "selected_features_", "feature_names_in_"]
        )
        reject_sparse(X, operation="transform")
        X = drop_fitted_metadata_columns(
            X, getattr(self, "_row_metadata_columns_", ())
        )
        if isinstance(X, pd.DataFrame):
            if getattr(self, "_fit_feature_names_generated_", False):
                raise ValueError(
                    "This ModelSelector was fitted on a positional array with "
                    "generated feature names; pass a positional ndarray to transform, "
                    "or refit on a DataFrame to establish column names."
                )
            check_fitted_column_identity(X, self.feature_names_in_)
            return X.iloc[:, self._output_indices()]
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                "X must be a 2D feature matrix. Reshape your data with "
                "X.reshape(-1, 1) for a single feature."
            )
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but ModelSelector was fitted with "
                f"{self.n_features_in_}"
            )
        return X_arr[:, self._output_indices()]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return names of selected raw columns using sklearn's transformer API."""
        check_is_fitted(self, ["selected_indices_", "feature_names_in_", "n_features_in_"])
        fitted_names = feature_names_array(self.feature_names_in_)
        if input_features is not None:
            supplied = _coerce_feature_names(input_features, argument="input_features")
            if len(supplied) != self.n_features_in_:
                raise ValueError(
                    "input_features must have the same number of features as the fitted data"
                )
            if list(supplied) != list(fitted_names):
                raise ValueError("input_features is not equal to feature_names_in_")
        return fitted_names[self._output_indices()]

    def inverse_transform(self, X):
        """Restore selected values to their fitted raw-column positions."""
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        return inverse_selected_matrix(
            X,
            n_features=self.n_features_in_,
            selected_indices=self._output_indices(),
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def result_view(self):
        """Return a normalized, non-cached view of this fitted selector."""
        from sift.selection.view import as_result

        return as_result(self)

    @property
    def result_view_(self):
        """Return a normalized, non-cached view of this fitted selector."""
        return self.result_view()
