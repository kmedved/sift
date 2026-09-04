"""Time-series-aware permutation importance."""

from __future__ import annotations

import copy
import hashlib
import os
import pickle
from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from sift._deprecate import warn_random_state_none
from sift._metadata import resolve_row_metadata
from sift._permute import (
    PermutationMethod,
    build_group_info,
    permute_array,
    resolve_permutation_method,
)
from sift._preprocess import ensure_weights
from sift.scoring import ScoringSpec, get_scoring


ParallelBackend = Literal["threads", "processes"]
_VALID_PARALLEL_BACKENDS: tuple[ParallelBackend, ...] = ("threads", "processes")


def _importance_labels_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if type(left) is not type(right):
        return False
    values = np.empty(2, dtype=object)
    values[:] = [left, right]
    try:
        return bool(
            pd.Index(values, dtype=object, tupleize_cols=False).duplicated()[1]
        )
    except (TypeError, ValueError):
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return repr(left) == repr(right)


class _ImportanceFeatureNames(list):
    def __init__(self, values: list[Any], source_identity: tuple[int, tuple[int, ...]]):
        super().__init__(values)
        self._sift_source_identity = source_identity
        self._sift_accessor_ids = tuple(id(value) for value in values)
        self._sift_content_digest = _importance_feature_digest(values)


def _importance_feature_digest(values: list[Any]) -> str:
    try:
        payload = pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        fallback = []
        for value in values:
            try:
                state = vars(value)
            except TypeError:
                state = None
            fallback.append(
                (
                    f"{type(value).__module__}.{type(value).__qualname__}",
                    repr(value),
                    repr(state),
                )
            )
        payload = repr(fallback).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, init=False, eq=False)
class ImportanceResult:
    """Rich permutation-importance result with repeat-level diagnostics.

    The default ``permutation_importance`` return remains its historical
    four-column DataFrame.  Request this object with ``return_result=True``
    when repeat-level importance drops or a normalized result view are needed.
    All array, table, and mapping accessors return defensive copies.
    """

    _ranking: pd.DataFrame = field(repr=False)
    _importances: np.ndarray = field(repr=False)
    _feature_names: tuple[Any, ...] = field(repr=False)
    _feature_source_identity: tuple[int, tuple[int, ...]] = field(repr=False)
    _ranking_indices: tuple[int, ...] = field(repr=False)
    baseline_score: float
    _selector_metadata: dict[str, Any] = field(repr=False)
    _diagnostics: dict[str, Any] = field(repr=False)

    def __init__(
        self,
        *,
        ranking: pd.DataFrame,
        importances: np.ndarray,
        feature_names: list[Any],
        ranking_indices: list[int],
        baseline_score: float,
        selector_metadata: dict[str, Any],
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(ranking, pd.DataFrame):
            raise TypeError("ranking must be a pandas DataFrame")
        expected_columns = [
            "feature",
            "importance_mean",
            "importance_std",
            "baseline_score",
        ]
        if list(ranking.columns) != expected_columns:
            raise ValueError(
                f"ranking must have exactly the columns {expected_columns}"
            )
        if isinstance(feature_names, (str, bytes, bytearray, Mapping, Set)):
            raise TypeError("feature_names must be an ordered iterable")
        try:
            names = list(feature_names)
        except TypeError as exc:
            raise TypeError("feature_names must be an ordered iterable") from exc
        if isinstance(ranking_indices, (str, bytes, bytearray, Mapping, Set)):
            raise TypeError("ranking_indices must be an ordered iterable")
        try:
            raw_indices = list(ranking_indices)
        except TypeError as exc:
            raise TypeError("ranking_indices must be an ordered iterable") from exc
        indices: list[int] = []
        for value in raw_indices:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError("ranking_indices must contain integer positions")
            indices.append(int(value))
        if (
            len(ranking) != len(names)
            or len(indices) != len(names)
            or set(indices) != set(range(len(names)))
        ):
            raise ValueError(
                "ranking and ranking_indices must cover every feature position exactly once"
            )
        for row, position in enumerate(indices):
            if not _importance_labels_equal(
                ranking["feature"].iloc[row],
                names[position],
            ):
                raise ValueError(
                    "ranking feature identities must match ranking_indices"
                )

        raw_importances = np.asarray(importances)
        if raw_importances.ndim != 2 or raw_importances.shape[0] != len(names):
            raise ValueError(
                "importances must be a two-dimensional matrix with one row per feature"
            )
        if raw_importances.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                "importances must contain real non-boolean numeric values"
            )
        converted_importances = raw_importances.astype(np.float64, copy=True)
        if isinstance(baseline_score, (bool, np.bool_)) or not isinstance(
            baseline_score,
            (Real, np.integer, np.floating),
        ):
            raise ValueError("baseline_score must be a real non-boolean number")
        if not isinstance(selector_metadata, Mapping):
            raise TypeError("selector_metadata must be a mapping")
        if diagnostics is not None and not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping or None")

        source_identity = (os.getpid(), tuple(id(name) for name in names))
        stored_names = copy.deepcopy(names)
        stored_ranking = ranking.copy(deep=True)
        feature_column = stored_ranking["feature"].copy(deep=True)
        for row, position in enumerate(indices):
            feature_column.iat[row] = stored_names[position]
        stored_ranking["feature"] = feature_column
        object.__setattr__(self, "_ranking", stored_ranking)
        object.__setattr__(
            self,
            "_importances",
            converted_importances,
        )
        object.__setattr__(self, "_feature_names", tuple(stored_names))
        object.__setattr__(self, "_feature_source_identity", source_identity)
        object.__setattr__(
            self,
            "_ranking_indices",
            tuple(indices),
        )
        object.__setattr__(self, "baseline_score", float(baseline_score))
        object.__setattr__(
            self,
            "_selector_metadata",
            copy.deepcopy(dict(selector_metadata)),
        )
        object.__setattr__(
            self,
            "_diagnostics",
            copy.deepcopy(dict(diagnostics or {})),
        )

    @property
    def ranking_(self) -> pd.DataFrame:
        """Legacy-compatible summary table, ranked by mean importance."""
        ranking, _ = self._copy_ranking_and_names()
        return ranking

    @property
    def importances_(self) -> np.ndarray:
        """Importance drops with raw feature positions on rows and repeats on columns."""
        return self._importances.copy()

    @property
    def feature_names(self) -> list[Any]:
        """Feature identities in original input-position order."""
        return _ImportanceFeatureNames(
            copy.deepcopy(list(self._feature_names)),
            self._feature_source_identity,
        )

    @property
    def ranking_indices(self) -> list[int]:
        """Original feature position for each row of ``ranking_``."""
        return list(self._ranking_indices)

    @property
    def selector_metadata(self) -> dict[str, Any]:
        """Configuration and provenance that do not retain caller data."""
        return copy.deepcopy(self._selector_metadata)

    @property
    def diagnostics_(self) -> dict[str, Any]:
        """Repeat-axis and aggregation conventions."""
        return copy.deepcopy(self._diagnostics)

    def get_feature_ranking(self) -> pd.DataFrame:
        """Return the historical permutation-importance summary table."""
        return self.ranking_

    def _copy_ranking_and_names(self) -> tuple[pd.DataFrame, list[Any]]:
        names = copy.deepcopy(list(self._feature_names))
        ranking = self._ranking.copy(deep=True)
        feature_column = ranking["feature"].copy(deep=True)
        for row, position in enumerate(self._ranking_indices):
            feature_column.iat[row] = names[position]
        ranking["feature"] = feature_column
        return ranking, names

    def _adapter_snapshot(self) -> dict[str, Any]:
        ranking, names = self._copy_ranking_and_names()
        return {
            "ranking": ranking,
            "importances": self._importances.copy(),
            "feature_names": names,
            "ranking_indices": list(self._ranking_indices),
            "baseline_score": self.baseline_score,
            "selector_metadata": copy.deepcopy(self._selector_metadata),
            "diagnostics": copy.deepcopy(self._diagnostics),
        }

    def _matches_original_features(self, feature_names: Any) -> bool:
        if (
            getattr(feature_names, "_sift_source_identity", None)
            == self._feature_source_identity
            and getattr(feature_names, "_sift_accessor_ids", None)
            == tuple(id(value) for value in feature_names)
            and getattr(feature_names, "_sift_content_digest", None)
            == _importance_feature_digest(list(feature_names))
        ):
            return True
        try:
            supplied = list(feature_names)
        except TypeError:
            return False
        if len(supplied) != len(self._feature_names):
            return False
        same_process = os.getpid() == self._feature_source_identity[0]
        original_ids = self._feature_source_identity[1]
        for position, (expected, observed) in enumerate(
            zip(self._feature_names, supplied)
        ):
            if same_process and id(observed) == original_ids[position]:
                continue
            if not _importance_labels_equal(expected, observed):
                return False
        return True

    def result_view(self, input_features=None):
        """Return an additive normalized view without changing this result."""
        from sift.selection.view import as_result

        return as_result(self, input_features=input_features)


def permutation_importance(
    model,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    *,
    scoring: str | Callable | ScoringSpec = "neg_mse",
    higher_is_better: bool | None = None,
    n_repeats: int = 10,
    permute_method: PermutationMethod = "auto",
    block_size: int | str = "auto",
    n_jobs: int = -1,
    parallel_backend: ParallelBackend = "threads",
    random_state: int | None = None,
    return_result: bool = False,
) -> pd.DataFrame | ImportanceResult:
    """
    Permutation importance with optional time-series-aware strategies.

    Scores each feature by how much the model's performance drops when that
    column is shuffled, averaged over ``n_repeats`` permutations of a fitted
    model's predictions.  Unlike SIFT's selectors this ranks rather than
    selects: it never chooses a cut-off, and its semantics are recorded as
    ``"ranking_only"``.  The permutation strategy is what makes it usable on
    dependent data -- supply ``groups`` or ``time`` and the shuffle is
    restricted so it cannot destroy the dependence structure it should
    preserve.  By default it uses negative MSE, 10 repeats, all cores, thread
    parallelism, a nondeterministic seed, and returns the historical
    four-column DataFrame sorted by descending mean importance.

    Parameters
    ----------
    model : fitted estimator
        Must have .predict() method.
    X : DataFrame or ndarray
        Features.
    y : array
        Target.
    sample_weight : array, optional
        Weights for scoring. Defaults to uniform.
    groups : array, optional
        Group labels. Enables within_group permutation.
    time : array, optional
        Time values. Enables block/circular_shift. If provided without groups,
        the full dataset is treated as one ordered group.
    scoring : str, callable, or ScoringSpec
        "neg_mse", "neg_rmse", "neg_mae", "r2", "accuracy",
        "balanced_accuracy", "neg_error", "neg_logloss", or
        a scorer object returned by sklearn's ``make_scorer``/``get_scorer``.
        Legacy ``callable(y, y_pred, w) -> float`` callbacks remain supported;
        wrap custom estimator-style scoring in ``ScoringSpec`` to make its
        ``(model, X, y, w)`` contract explicit.
    higher_is_better : bool, optional
        Score direction for legacy callbacks only. Named, ``ScoringSpec``, and
        sklearn scorers already carry direction metadata and reject this
        override. Legacy callbacks default to higher-is-better for
        compatibility; pass ``False`` for loss functions.
    n_repeats : int
        Permutation repeats per feature.
    permute_method : str
        - "auto": circular_shift if time is provided, within_group if groups only, global otherwise
        - "global": standard shuffle
        - "within_group": shuffle within each group (requires groups)
        - "block": shuffle time-ordered blocks (requires time)
        - "circular_shift": rotate within time order (requires time)
    block_size : int or "auto"
        For block method. ``"auto"`` uses ``int(sqrt(n_samples))``. Ignored by
        every other ``permute_method``.
    n_jobs : int
        Number of parallel jobs. The default ``-1`` uses all cores; features
        are split into contiguous chunks, one per worker.
    parallel_backend : {"threads", "processes"}
        Joblib backend preference. "threads" avoids inter-process copies for
        many estimators; "processes" isolates workers when process parallelism
        is preferred.
    random_state : int or None, default None
        Seed for the permutation draws. One seed per (feature, repeat) is
        derived from it, so a given seed reproduces the run exactly, including
        under parallelism. The ``None`` default draws nondeterministic entropy
        and emits a ``FutureWarning``: SIFT 1.0 will resolve it to
        ``random_state=0``. Pass an integer to make the call reproducible and
        silence that warning.
    return_result : bool
        If ``False`` (default), return the historical four-column DataFrame.
        If ``True``, return ``ImportanceResult``, including the
        per-feature, per-repeat importance-drop matrix.

    Returns
    -------
    DataFrame or ImportanceResult
        The default DataFrame has columns ``feature``, ``importance_mean``,
        ``importance_std``, and ``baseline_score``, one row per feature,
        sorted by descending ``importance_mean``. With ``return_result=True``,
        an ``ImportanceResult`` exposing that table as ``ranking_``, the
        ``(n_features, n_repeats)`` matrix of raw score drops as
        ``importances_`` in *raw feature order*, ``feature_names``,
        ``ranking_indices`` mapping ranking rows back to those positions,
        ``baseline_score``, ``selector_metadata`` (including
        ``selection_semantics="ranking_only"``), and ``diagnostics_``.

    Raises
    ------
    ValueError
        If ``n_repeats`` is not a positive integer; if ``return_result`` is
        not a boolean; if ``parallel_backend`` is not ``"threads"`` or
        ``"processes"``; if ``X`` is not 2-D or its row count differs from
        ``y``; if ``permute_method`` is ``"block"`` or ``"circular_shift"``
        without ``time``, or ``"within_group"`` without ``groups``; or if
        ``higher_is_better`` is passed with a named, ``ScoringSpec``, or
        sklearn scorer, all of which already carry a direction.
    TypeError
        If ``scoring`` is neither a scorer name, a ``ScoringSpec``, nor a
        callable.

    Warns
    -----
    FutureWarning
        When ``random_state`` is left at ``None``, because the run is then
        nondeterministic and the default will change in SIFT 1.0.

    See Also
    --------
    ImportanceResult : The richer return type, and its accessors.
    sift.as_result : Normalize an ``ImportanceResult`` into a SelectionView.
    sift.select_fdr : Error-controlled selection rather than a ranking.

    Notes
    -----
    Importance is the *score drop*: the baseline score of the unpermuted data
    minus the score after shuffling one column, oriented so larger always
    means more important whatever the metric's own direction. It is measured
    on the model as given, so it reflects what that fitted model relies on,
    not what an independent model would learn -- correlated features can share
    credit and each look unimportant alone. ``importance_std`` is the
    population standard deviation over repeats (``ddof=0``), so it describes
    permutation noise, not sampling uncertainty. Cost is one model prediction
    per feature and repeat, ``n_features * n_repeats`` in total, plus one
    baseline.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from sift import permutation_importance
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    >>> y = 3.0 * X["a"] - 2.0 * X["b"] + 0.1 * rng.normal(size=200)
    >>> model = LinearRegression().fit(X, y)
    >>> ranking = permutation_importance(model, X, y, n_repeats=5,
    ...                                  random_state=0)
    >>> ranking["feature"].tolist()
    ['a', 'b', 'c', 'd']
    >>> result = permutation_importance(model, X, y, n_repeats=5,
    ...                                 random_state=0, return_result=True)
    >>> result.importances_.shape, result.ranking_indices
    ((4, 5), [0, 1, 2, 3])
    """
    metadata = resolve_row_metadata(
        X,
        groups=groups,
        time=time,
        sample_weight=sample_weight,
    )
    X = metadata.X
    groups = metadata.groups
    time = metadata.time
    sample_weight = metadata.sample_weight
    if isinstance(n_repeats, (bool, np.bool_)) or not isinstance(n_repeats, (int, np.integer)):
        raise ValueError("n_repeats must be a positive integer")
    n_repeats = int(n_repeats)
    if n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")
    if not isinstance(return_result, (bool, np.bool_)):
        raise ValueError("return_result must be a boolean")
    return_result = bool(return_result)
    parallel_backend = _validate_parallel_backend(parallel_backend)
    sample_weight_supplied = sample_weight is not None

    n = len(y)
    X_arr = None
    if isinstance(X, pd.DataFrame):
        if X.shape[0] != n:
            raise ValueError(f"X has {X.shape[0]} rows but y has {n}")
        n_features = X.shape[1]
    else:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or pandas DataFrame")
        if X_arr.shape[0] != n:
            raise ValueError(f"X has {X_arr.shape[0]} rows but y has {n}")
        n_features = X_arr.shape[1]

    w = ensure_weights(sample_weight, n, normalize=True)
    if random_state is None:
        warn_random_state_none("permutation_importance")
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31, size=(n_features, n_repeats))

    requested_permute_method = permute_method
    permute_method = resolve_permutation_method(permute_method, groups=groups, time=time)
    if permute_method in ("block", "circular_shift") and time is None:
        raise ValueError(f"permute_method='{permute_method}' requires time")
    group_info = (
        build_group_info(groups, time, n_samples=n) if permute_method != "global" else None
    )
    score_higher_is_better = _higher_is_better(scoring, higher_is_better)
    selector_metadata = (
        {
            "selector": "permutation_importance",
            "n_features": int(n_features),
            "n_repeats": n_repeats,
            "permute_method_requested": requested_permute_method,
            "permute_method": permute_method,
            "block_size": _metadata_scalar(block_size),
            "scoring": _scoring_label(scoring),
            "higher_is_better": score_higher_is_better,
            "sample_weight_supplied": sample_weight_supplied,
            "groups_supplied": groups is not None,
            "time_supplied": time is not None,
            "n_jobs": n_jobs,
            "parallel_backend": parallel_backend,
            "input_kind": (
                "dataframe" if isinstance(X, pd.DataFrame) else "positional"
            ),
            "selection_semantics": "ranking_only",
        }
        if return_result
        else {}
    )

    if isinstance(X, pd.DataFrame):
        return _permutation_importance_dataframe(
            model,
            X,
            y,
            w,
            scoring,
            n_repeats,
            permute_method,
            block_size,
            group_info,
            seeds,
            baseline=_score(
                model,
                X,
                y,
                w,
                scoring,
                sample_weight_supplied=sample_weight_supplied,
            ),
            higher_is_better=score_higher_is_better,
            sample_weight_supplied=sample_weight_supplied,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            return_result=return_result,
            selector_metadata=selector_metadata,
        )

    assert X_arr is not None
    return _permutation_importance_array(
        model,
        X_arr,
        y,
        w,
        scoring,
        n_repeats,
        permute_method,
        block_size,
        group_info,
        seeds,
        baseline=_score(
            model,
            X_arr,
            y,
            w,
            scoring,
            sample_weight_supplied=sample_weight_supplied,
        ),
        higher_is_better=score_higher_is_better,
        sample_weight_supplied=sample_weight_supplied,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        return_result=return_result,
        selector_metadata=selector_metadata,
    )


def _permutation_importance_dataframe(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable | ScoringSpec,
    n_repeats: int,
    permute_method: PermutationMethod,
    block_size: int | str,
    group_info: dict | None,
    seeds: np.ndarray,
    *,
    baseline: float,
    higher_is_better: bool,
    sample_weight_supplied: bool,
    n_jobs: int,
    parallel_backend: ParallelBackend,
    return_result: bool,
    selector_metadata: dict[str, Any],
) -> pd.DataFrame | ImportanceResult:
    features = list(X.columns)
    row_positions = np.arange(X.shape[0])
    chunks = _feature_chunks(len(features), n_jobs)

    def compute_chunk(feature_indices: np.ndarray) -> list[tuple]:
        X_work = X.copy()
        chunk_results = []

        for feat_idx in feature_indices:
            orig_col = X.iloc[:, feat_idx].copy()
            drops = []

            for rep in range(n_repeats):
                seed = int(seeds[feat_idx, rep])
                source_positions = permute_array(
                    row_positions,
                    method=permute_method,
                    group_info=group_info,
                    block_size=block_size,
                    rng=np.random.default_rng(seed),
                )
                permuted = orig_col.iloc[source_positions].copy()
                permuted.index = X_work.index
                X_work.iloc[:, feat_idx] = permuted
                try:
                    score = _score(
                        model,
                        X_work,
                        y,
                        w,
                        scoring,
                        sample_weight_supplied=sample_weight_supplied,
                    )
                finally:
                    X_work.iloc[:, feat_idx] = orig_col
                drops.append(
                    baseline - score if higher_is_better else score - baseline
                )

            summary = (feat_idx, float(np.mean(drops)), float(np.std(drops)))
            if return_result:
                chunk_results.append((*summary, np.asarray(drops, dtype=np.float64)))
            else:
                chunk_results.append(summary)

        return chunk_results

    result_chunks = Parallel(n_jobs=n_jobs, prefer=parallel_backend)(
        delayed(compute_chunk)(chunk) for chunk in chunks
    )
    results, importances = _flatten_feature_results(
        result_chunks,
        len(features),
        n_repeats=n_repeats,
        return_result=return_result,
    )
    return _build_importance_output(
        features,
        results,
        baseline=baseline,
        importances=importances,
        selector_metadata=selector_metadata,
    )


def _permutation_importance_array(
    model,
    X_arr: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable | ScoringSpec,
    n_repeats: int,
    permute_method: PermutationMethod,
    block_size: int | str,
    group_info: dict | None,
    seeds: np.ndarray,
    *,
    baseline: float,
    higher_is_better: bool,
    sample_weight_supplied: bool,
    n_jobs: int,
    parallel_backend: ParallelBackend,
    return_result: bool,
    selector_metadata: dict[str, Any],
) -> pd.DataFrame | ImportanceResult:
    features = list(range(X_arr.shape[1]))
    chunks = _feature_chunks(len(features), n_jobs)

    def compute_chunk(feature_indices: np.ndarray) -> list[tuple]:
        X_work = X_arr.copy()
        chunk_results = []

        for feat_idx in feature_indices:
            orig_col = X_arr[:, feat_idx]
            drops = []

            for rep in range(n_repeats):
                seed = int(seeds[feat_idx, rep])
                permuted = permute_array(
                    orig_col,
                    method=permute_method,
                    group_info=group_info,
                    block_size=block_size,
                    rng=np.random.default_rng(seed),
                )
                X_work[:, feat_idx] = permuted
                try:
                    score = _score(
                        model,
                        X_work,
                        y,
                        w,
                        scoring,
                        sample_weight_supplied=sample_weight_supplied,
                    )
                finally:
                    X_work[:, feat_idx] = orig_col
                drops.append(
                    baseline - score if higher_is_better else score - baseline
                )

            summary = (feat_idx, float(np.mean(drops)), float(np.std(drops)))
            if return_result:
                chunk_results.append((*summary, np.asarray(drops, dtype=np.float64)))
            else:
                chunk_results.append(summary)

        return chunk_results

    result_chunks = Parallel(n_jobs=n_jobs, prefer=parallel_backend)(
        delayed(compute_chunk)(chunk) for chunk in chunks
    )
    results, importances = _flatten_feature_results(
        result_chunks,
        len(features),
        n_repeats=n_repeats,
        return_result=return_result,
    )
    return _build_importance_output(
        features,
        results,
        baseline=baseline,
        importances=importances,
        selector_metadata=selector_metadata,
    )


def _validate_parallel_backend(parallel_backend: str) -> ParallelBackend:
    if parallel_backend not in _VALID_PARALLEL_BACKENDS:
        raise ValueError(
            f"Unknown parallel_backend: {parallel_backend!r}. "
            f"Expected one of {list(_VALID_PARALLEL_BACKENDS)}."
        )
    return parallel_backend  # type: ignore[return-value]


def _feature_chunks(n_features: int, n_jobs: int) -> list[np.ndarray]:
    if n_features < 1:
        return []
    n_workers = min(n_features, max(1, effective_n_jobs(n_jobs)))
    return [
        chunk.astype(np.intp, copy=False)
        for chunk in np.array_split(np.arange(n_features, dtype=np.intp), n_workers)
        if chunk.size
    ]


def _flatten_feature_results(
    result_chunks: list[list[tuple]],
    n_features: int,
    *,
    n_repeats: int,
    return_result: bool,
) -> tuple[list[tuple[float, float]], np.ndarray | None]:
    results: list[tuple[float, float] | None] = [None] * n_features
    importances = (
        np.empty((n_features, n_repeats), dtype=np.float64)
        if return_result
        else None
    )
    for chunk in result_chunks:
        for feature_result in chunk:
            feat_idx, mean, std = feature_result[:3]
            results[feat_idx] = (mean, std)
            if importances is not None:
                importances[feat_idx] = feature_result[3]

    if any(result is None for result in results):
        raise RuntimeError("Permutation importance did not return every feature result")
    return [result for result in results if result is not None], importances


def _build_importance_output(
    features: list[Any],
    results: list[tuple[float, float]],
    *,
    baseline: float,
    importances: np.ndarray | None,
    selector_metadata: dict[str, Any],
) -> pd.DataFrame | ImportanceResult:
    unsorted = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": [result[0] for result in results],
            "importance_std": [result[1] for result in results],
            "baseline_score": baseline,
        }
    )
    ranked = unsorted.sort_values("importance_mean", ascending=False)
    ranking_indices = ranked.index.to_numpy(dtype=np.intp, copy=True).tolist()
    ranking = ranked.reset_index(drop=True)
    if importances is None:
        return ranking
    return ImportanceResult(
        ranking=ranking,
        importances=importances,
        feature_names=features,
        ranking_indices=ranking_indices,
        baseline_score=baseline,
        selector_metadata=selector_metadata,
        diagnostics={
            "importance_definition": "score_drop",
            "importance_rows": "raw_feature_positions",
            "importance_columns": "permutation_repeats",
            "std_ddof": 0,
        },
    )


def _scoring_label(scoring: str | Callable | ScoringSpec) -> str:
    if isinstance(scoring, str):
        return scoring
    if isinstance(scoring, ScoringSpec):
        return scoring.name
    name = getattr(scoring, "__qualname__", None) or type(scoring).__qualname__
    module = getattr(scoring, "__module__", None) or type(scoring).__module__
    return f"{module}.{name}"


def _metadata_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _metadata_scalar(value.item())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _score(
    model,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scoring: str | Callable | ScoringSpec,
    *,
    sample_weight_supplied: bool,
) -> float:
    if isinstance(scoring, ScoringSpec):
        return scoring(model, X, y, w)
    if isinstance(scoring, str):
        return get_scoring(scoring)(model, X, y, w)
    if _is_sklearn_scorer(scoring):
        if sample_weight_supplied:
            return float(scoring(model, X, y, sample_weight=w))
        return float(scoring(model, X, y))
    if callable(scoring):
        y_pred = model.predict(X)
        return float(scoring(y, y_pred, w))
    raise TypeError("scoring must be a scorer name, ScoringSpec, or callable")


def _is_sklearn_scorer(scoring: object) -> bool:
    # Arbitrary three-argument callables are ambiguous with SIFT's legacy
    # (y, y_pred, w) contract, so only sklearn-created scorer objects are
    # detected here. ScoringSpec is the explicit estimator-style alternative.
    scorer_type = type(scoring)
    return (
        callable(scoring)
        and scorer_type.__module__.startswith("sklearn.metrics._scorer")
        and hasattr(scoring, "_score_func")
        and hasattr(scoring, "_sign")
    )


def _higher_is_better(
    scoring: str | Callable | ScoringSpec,
    override: bool | None = None,
) -> bool:
    if isinstance(scoring, ScoringSpec):
        if override is not None:
            raise ValueError(
                "higher_is_better is only supported for legacy score callbacks"
            )
        return bool(scoring.higher_is_better)
    if isinstance(scoring, str):
        if override is not None:
            raise ValueError(
                "higher_is_better is only supported for legacy score callbacks"
            )
        return bool(get_scoring(scoring).higher_is_better)
    if _is_sklearn_scorer(scoring):
        if override is not None:
            raise ValueError(
                "higher_is_better is only supported for legacy score callbacks"
            )
        return True
    if callable(scoring):
        if override is None:
            return True
        if not isinstance(override, (bool, np.bool_)):
            raise ValueError("higher_is_better must be a boolean or None")
        return bool(override)
    raise TypeError("scoring must be a scorer name, ScoringSpec, or callable")


__all__ = ["ImportanceResult", "permutation_importance"]
