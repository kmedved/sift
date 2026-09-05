"""FDR-calibrated Gaussian-copula knockoff selection."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Hashable, Sequence
from functools import wraps
from typing import Any, Callable, Optional
import warnings

import numpy as np
import pandas as pd

from scipy.linalg import LinAlgError, cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from sift._deprecate import warn_external
from sift._logging import logger
from sift._preprocess import to_numpy
from sift.estimators.copula import (
    FeatureCache,
    build_cache,
    gaussian_mi_from_corr,
    weighted_corr_with_vector,
    weighted_correlation_matrix,
    weighted_rank_gauss_1d,
)
from sift.estimators.knockoffs import (
    GaussianKnockoffModel,
    fit_gaussian_knockoffs,
    gaussian_knockoff_mean,
    sample_gaussian_knockoffs,
)
from sift.selection.conditioning import (
    FDR_COMPATIBLE_PROVENANCE,
    compose_selected,
    conditioning_record,
    named_feature_space,
    require_include_provenance,
    resolve_conditioning,
)


_STATISTIC_NOT_ENABLED = (
    "is reserved for a future tie-safe knockoff statistic and is not yet enabled"
)
_CEFSPLUS_DEFAULT_PATH_DEPTH = 10
_LOW_POWER_S = 0.05
_INTEGER_TARGET_WARNING_EMITTED = False


def _single_threaded_ridge_knockoffs(func):
    """Limit native pools for the complete ridge-knockoff operation."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        if str(kwargs.get("statistic", "relevance")).lower() != "ridge":
            return func(*args, **kwargs)
        with threadpool_limits(limits=1):
            return func(*args, **kwargs)

    return wrapped


class _SubsampleDefaultType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<subsample default: 50,000 rows when X is given>"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_SubsampleDefaultType, ())


_SUBSAMPLE_DEFAULT = _SubsampleDefaultType()


@dataclass(frozen=True)
class KnockoffSelectionResult:
    """Result object for q-calibrated knockoff selection.

    Returned by ``select_fdr``.  Alongside the selected set it carries the
    full per-feature ``W`` table, the threshold that produced the selection,
    and the validity metadata that says how strong the FDR claim actually is
    -- read ``selector_metadata["fdr_control"]`` before quoting one.  The
    object is a frozen dataclass, so it is safe to keep and pass around; call
    ``result_view`` for the normalized ``sift.SelectionView``.

    Parameters
    ----------
    selected_features : list
        Selected feature labels, ordered by descending mean ``W`` with ties
        broken by cache column order.  Empty when nothing cleared the
        threshold, which is a valid answer.
    selected_indices : list of int or None
        Positions of those features in the original feature matrix, in the
        same order.
    selector_metadata : dict
        Run configuration and validity provenance: ``q``, ``offset``,
        ``statistic``, ``s_method``, ``n_draws``, ``eta``, the shrinkage
        diagnostics ``gamma`` and ``lambda_min``, the power diagnostics
        ``s_mean``, ``s_median`` and ``n_low_power_features``, the raw input
        width ``n_features_input`` with ``dropped_feature_positions`` and
        ``dropped_feature_reasons``, feasibility fields ``min_feasible_q``,
        ``n_tested``, ``n_tested_unit``, ``n_tested_per_draw``,
        ``n_eligible``, ``tested_state``, ``n_infeasible_draws``,
        ``tested_sets_vary``, ``n_discoveries_offset_0`` and
        ``n_discoveries_offset_0_per_draw``, and the claim fields
        ``fdr_control``, ``per_draw_fdr_control``, ``q_scope``,
        ``aggregation``, ``aggregation_fdr_control`` and ``validity_model``.
    W : DataFrame
        One row per valid cache feature with columns ``feature``,
        ``selected_index``, ``W`` (the mean statistic over draws),
        ``selected``, ``selection_frequency``, ``relevance``, ``selector``,
        and one ``W_draw_<i>`` column per draw.  Grouped runs add
        ``feature_group``, and ``feature_groups="auto"`` adds
        ``is_representative``.
    threshold : float or None
        The knockoff threshold for a single draw -- ``inf`` when no data-driven
        threshold exists -- and ``None`` for a derandomized ``n_draws > 1``
        run, where selection is by frequency instead.
    selection_frequency : Series or None
        Fraction of draws that selected each feature, indexed by feature
        label, when ``n_draws > 1``; ``None`` for a single draw.
    diagnostics_ : dict or None, default None
        Per-draw detail: ``thresholds``, ``selection_sets`` (original column
        positions), and ``active_valid_positions``.  Grouped runs add
        ``feature_groups``, ``group_W_draws`` and ``group_thresholds``;
        ``feature_groups="auto"`` also adds ``cluster_labels``,
        ``cluster_representatives_valid_positions``, and the nested
        ``representative_result``.

    Attributes
    ----------
    selected_features, selected_indices, selector_metadata, W, threshold,
    selection_frequency, diagnostics_
        The constructor fields above, stored as-is.

    See Also
    --------
    select_fdr : The selector that produces this result.
    sift.as_result : Convert it to a normalized view.
    sift.SelectionView : The normalized view type.

    Notes
    -----
    ``W`` covers only the features the filter actually ran on: constant
    columns never reach the cache, and columns with no weighted variance take
    no part in knockoff construction.  ``selector_metadata`` reconciles that
    against the caller's matrix through ``n_features_input`` and the
    dropped-position lists, so ``len(W)`` is not the raw feature count.  The
    ``W`` statistic is antisymmetric under swapping a feature with its
    knockoff: positive values are evidence for the original, and the threshold
    calibrates how many negatives the selected set is allowed to imply.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_fdr
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(300, 8)),
    ...                  columns=[f"f{i}" for i in range(8)])
    >>> beta = np.zeros(8)
    >>> beta[:5] = 2.5
    >>> y = X.to_numpy() @ beta + 0.5 * rng.normal(size=300)
    >>> result = select_fdr(X, y, q=0.2, random_state=0, verbose=False)
    >>> sorted(result.selected_features), result.selection_frequency is None
    (['f0', 'f1', 'f2', 'f3', 'f4'], True)
    >>> len(result.W), sorted(result.diagnostics_)
    (8, ['active_valid_positions', 'offset_zero_selection_sets', 'selection_sets', 'thresholds'])
    >>> result.get_feature_ranking().loc[0, ["feature", "rank", "selected"]].tolist()
    ['f4', 1, True]
    """

    selected_features: list[Any]
    selected_indices: Optional[list[int]]
    selector_metadata: dict[str, Any]
    W: pd.DataFrame
    threshold: Optional[float]
    selection_frequency: Optional[pd.Series]
    diagnostics_: Optional[dict[str, Any]] = None

    def get_feature_ranking(self) -> pd.DataFrame:
        """Return the ``W`` table sorted into a stable feature ranking.

        Reorders a copy of ``W`` by descending ``W``, breaking ties by the
        original row order so the result is deterministic, and inserts a
        one-based ``rank`` column.  Unselected features are ranked too, which
        makes this the table to read when you want to see how close the
        near-misses came to the threshold.

        Returns
        -------
        DataFrame
            Columns ``feature``, optionally ``feature_group``, then ``W``,
            ``rank``, ``selected``, ``selection_frequency``,
            ``selected_index``, ``relevance``, and ``selector``, with a fresh
            zero-based index.  The per-draw ``W_draw_<i>`` columns of
            ``W`` are not carried over.

        See Also
        --------
        KnockoffSelectionResult.result_view : Normalized view of the same data.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from sift import select_fdr
        >>> rng = np.random.default_rng(0)
        >>> X = pd.DataFrame(rng.normal(size=(300, 8)),
        ...                  columns=[f"f{i}" for i in range(8)])
        >>> beta = np.zeros(8)
        >>> beta[:5] = 2.5
        >>> y = X.to_numpy() @ beta + 0.5 * rng.normal(size=300)
        >>> ranking = select_fdr(X, y, q=0.2, random_state=0,
        ...                      verbose=False).get_feature_ranking()
        >>> ranking["rank"].tolist()
        [1, 2, 3, 4, 5, 6, 7, 8]
        >>> bool(ranking["W"].is_monotonic_decreasing)
        True
        """
        ranking = self.W.copy()
        ranking["_feature_order"] = np.arange(len(ranking), dtype=np.int64)
        ranking = ranking.sort_values(
            ["W", "_feature_order"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranking = ranking.drop(columns=["_feature_order"])
        ranking.insert(2, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
        columns = ["feature"]
        if "feature_group" in ranking.columns:
            columns.append("feature_group")
        columns.extend(
            [
                "W",
                "rank",
                "selected",
                "selection_frequency",
                "selected_index",
                "relevance",
                "selector",
            ]
        )
        return ranking[columns]

    def result_view(self, input_features=None):
        """Return an additive normalized view without changing this result.

        Convenience wrapper around ``sift.as_result``.  This result object
        is left exactly as it is; the view is a separate, normalized copy.

        Parameters
        ----------
        input_features : sequence or None, default None
            Ordered labels of every raw input column, used to establish the
            view's raw feature identity and column hash when the result alone
            cannot prove it.  ``None`` leaves the view with whatever identity
            the result carries.

        Returns
        -------
        SelectionView
            Normalized view exposing ``features``, ``indices``, ``k``,
            ``table``, and ``metadata``.

        See Also
        --------
        sift.as_result : The generic converter this delegates to.
        sift.SelectionView : The returned view type.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from sift import select_fdr
        >>> rng = np.random.default_rng(0)
        >>> X = pd.DataFrame(rng.normal(size=(300, 8)),
        ...                  columns=[f"f{i}" for i in range(8)])
        >>> beta = np.zeros(8)
        >>> beta[:5] = 2.5
        >>> y = X.to_numpy() @ beta + 0.5 * rng.normal(size=300)
        >>> view = select_fdr(X, y, q=0.2, random_state=0,
        ...                   verbose=False).result_view()
        >>> view.k, sorted(view.features)
        (5, ['f0', 'f1', 'f2', 'f3', 'f4'])
        """
        from sift.selection.view import as_result

        return as_result(self, input_features=input_features)


@dataclass(frozen=True)
class KnockoffStatContext:
    """Shared precomputed inputs for knockoff feature statistics."""

    Z: np.ndarray
    Zt: np.ndarray
    zy: np.ndarray
    w: np.ndarray
    model: GaussianKnockoffModel
    r: np.ndarray
    rt: np.ndarray
    kept: np.ndarray
    G: np.ndarray
    r_aug: np.ndarray
    options: dict[str, Any]
    n_jobs: int
    rng: np.random.Generator
    statistic_name: str = ""


@dataclass(frozen=True)
class KnockoffStatSpec:
    name: str
    fn: Callable[[KnockoffStatContext], np.ndarray]
    enabled: bool = True
    needs_screening: bool = True
    allowed_options: frozenset[str] = frozenset()


def _validate_probability(value: float, name: str, *, upper_inclusive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite float in (0, 1)")
    value_float = float(value)
    upper_ok = value_float <= 1.0 if upper_inclusive else value_float < 1.0
    if not np.isfinite(value_float) or value_float <= 0.0 or not upper_ok:
        interval = "(0, 1]" if upper_inclusive else "(0, 1)"
        raise ValueError(f"{name} must be a finite float in {interval}")
    return value_float


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{name} must be a positive integer")
    value_int = int(value)
    if value_int != value:
        raise ValueError(f"{name} must be a positive integer")
    if value_int < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value_int


def _validate_offset(offset: int) -> int:
    if isinstance(offset, (bool, np.bool_)):
        raise ValueError("offset must be 0 or 1")
    if isinstance(offset, (float, np.floating)) and not float(offset).is_integer():
        raise ValueError("offset must be 0 or 1")
    offset_int = int(offset)
    if offset_int != offset:
        raise ValueError("offset must be 0 or 1")
    if offset_int not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    return offset_int


def _tested_unit_ids(
    kept_local: np.ndarray,
    *,
    active_positions: np.ndarray,
    active_group_codes: np.ndarray | None,
) -> tuple[int, ...]:
    kept = np.asarray(kept_local, dtype=np.int64).reshape(-1)
    if kept.size == 0:
        return ()
    if active_group_codes is None:
        return tuple(sorted(int(active_positions[int(i)]) for i in kept))
    return tuple(sorted({int(active_group_codes[int(i)]) for i in kept}))


def _offset_zero_local_selection(W: np.ndarray, q: float) -> np.ndarray:
    W_arr = np.asarray(W, dtype=np.float64).reshape(-1)
    threshold = knockoff_threshold(W_arr, q, offset=0)
    if not np.isfinite(threshold):
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(W_arr >= threshold).astype(np.int64)


def _count_bound_min_feasible_q(n_tested: int) -> float:
    return float("inf") if int(n_tested) <= 0 else float(1.0 / int(n_tested))


def _draw_knockoff_plus_infeasible(n_tested: int, q: float) -> bool:
    return bool(int(n_tested) * float(q) < 1.0)


def _feasibility_metadata(
    *,
    n_tested_per_draw: list[int],
    n_tested_unit: str,
    tested_id_sets: list[tuple[int, ...]],
    n_discoveries_offset_0_per_draw: list[int],
    n_discoveries_offset_0: int,
    n_eligible: int,
    tested_state: str,
    q: float,
    offset: int,
) -> dict[str, Any]:
    counts = [int(v) for v in n_tested_per_draw]
    if tested_state != "post_screening":
        n_tested = 0
        min_feasible_q = float("inf")
        n_infeasible_draws = 0
        infeasible_draws: list[bool] = []
    else:
        n_tested = int(min(counts)) if counts else 0
        min_feasible_q = _count_bound_min_feasible_q(n_tested)
        infeasible_draws = [
            _draw_knockoff_plus_infeasible(m, q) if int(offset) == 1 else False
            for m in counts
        ]
        n_infeasible_draws = int(sum(infeasible_draws))
    return {
        "min_feasible_q": min_feasible_q,
        "n_tested": n_tested,
        "n_tested_unit": n_tested_unit,
        "n_tested_per_draw": counts,
        "n_eligible": int(n_eligible),
        "tested_state": tested_state,
        "n_infeasible_draws": n_infeasible_draws,
        "tested_sets_vary": len(set(tested_id_sets)) > 1,
        "n_discoveries_offset_0": int(n_discoveries_offset_0),
        "n_discoveries_offset_0_per_draw": [
            int(v) for v in n_discoveries_offset_0_per_draw
        ],
    }


def _aggregate_offset_zero_discoveries(
    per_draw_valid: list[list[int]],
    *,
    n_draws: int,
    eta: float,
    p_valid: int,
) -> int:
    if n_draws <= 1:
        return int(len(per_draw_valid[0])) if per_draw_valid else 0
    selected = np.zeros((n_draws, p_valid), dtype=np.float64)
    for draw_idx, chosen in enumerate(per_draw_valid):
        if chosen:
            selected[draw_idx, np.asarray(chosen, dtype=np.int64)] = 1.0
    return int(np.sum(selected.mean(axis=0) >= float(eta)))


def _warn_knockoff_plus_infeasible(metadata: dict[str, Any]) -> None:
    if str(metadata.get("tested_state")) != "post_screening":
        return
    if int(metadata["offset"]) != 1:
        return
    n_infeasible = int(metadata.get("n_infeasible_draws", 0))
    if n_infeasible <= 0:
        return
    counts = list(metadata["n_tested_per_draw"])
    n_draws = len(counts)
    q = float(metadata["q"])
    unit = str(metadata["n_tested_unit"])
    min_q = metadata["min_feasible_q"]
    min_q_text = "inf" if not np.isfinite(min_q) else f"{float(min_q):.3g}"
    bound_note = (
        f"min_feasible_q={min_q_text} is a necessary count-based lower bound "
        "(1/min m over completed draws), not a sufficient condition for discovery"
    )
    if n_draws <= 1:
        message = (
            "knockoff+ (offset=1) cannot select any tested unit at "
            f"q={q:g}: effective m={counts[0] if counts else 0} {unit}(s) so m*q < 1 "
            f"({bound_note}). m is the post-screening, post-conditioning tested "
            f"count at the {unit} level, not raw input width."
        )
    else:
        message = (
            f"knockoff+ (offset=1) cannot select on {n_infeasible} of {n_draws} "
            f"draws at q={q:g}: those draws have effective m*q < 1. {bound_note} "
            "and an infeasible draw does not imply the aggregated selection is "
            f"empty. m is the post-screening tested count at the {unit} level, "
            "not raw input width."
        )
    warn_external(message, UserWarning)


def _validate_screen_pairs(screen_pairs: int | None) -> int | None:
    if screen_pairs is None:
        return None
    return _validate_positive_int(screen_pairs, "screen_pairs")


def _validate_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative float")
    value_float = float(value)
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")
    return value_float


def _warn_if_integer_multiclass_target(y: Any) -> None:
    global _INTEGER_TARGET_WARNING_EMITTED
    if _INTEGER_TARGET_WARNING_EMITTED:
        return
    y_raw = np.asarray(y)
    if y_raw.size == 0 or not np.issubdtype(y_raw.dtype, np.integer):
        return
    n_unique = np.unique(y_raw.ravel()).shape[0]
    if 3 <= n_unique <= 20:
        _INTEGER_TARGET_WARNING_EMITTED = True
        warnings.warn(
            "select_fdr treats y as a continuous target; integer labels with "
            "3-20 unique values look multiclass. For multiclass discovery, run "
            "one-vs-rest targets and combine the selected features.",
            UserWarning,
            stacklevel=4,
        )


def _validate_cache_rxx(Rxx: np.ndarray, p: int) -> np.ndarray:
    R_raw = np.asarray(Rxx)
    if (
        not np.issubdtype(R_raw.dtype, np.number)
        or not np.isrealobj(R_raw)
    ):
        raise ValueError("cache.Rxx must contain only finite real numeric values")
    R = np.asarray(R_raw, dtype=np.float64)
    if R.shape != (p, p):
        raise ValueError(f"cache.Rxx must have shape ({p}, {p})")
    if not np.isfinite(R).all():
        raise ValueError("cache.Rxx must contain only finite values")
    if not np.allclose(R, R.T, atol=1e-6, rtol=1e-6):
        raise ValueError("cache.Rxx must be symmetric")
    if not np.allclose(np.diag(R), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("cache.Rxx must have a unit diagonal")
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _reject_duplicate_feature_names(cache: FeatureCache) -> None:
    if cache.feature_names is None or cache.feature_names_are_synthetic:
        return
    names = pd.Index(cache.feature_names, dtype=object, tupleize_cols=False)
    duplicate_mask = names.duplicated(keep="first")
    if duplicate_mask.any():
        duplicates = names[duplicate_mask].tolist()
        sample = duplicates[:5]
        suffix = "..." if len(duplicates) > 5 else ""
        raise ValueError(f"Duplicate feature names are not supported: {sample}{suffix}")


def _feature_names_for_valid_cols(cache: FeatureCache) -> list[Any]:
    if cache.feature_names is None:
        return [f"x{int(i)}" for i in cache.valid_cols]
    return [cache.feature_names[int(i)] for i in cache.valid_cols]


def _input_width_provenance(
    cache: FeatureCache,
    *,
    inactive_valid_positions: np.ndarray | None = None,
    extra_valid_drops: list[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """Describe the raw input width and the columns knockoffs could not use.

    ``n_features`` counts the post-screening columns the knockoff filter
    actually ran on, so it cannot establish the caller's raw matrix width once
    constant columns are dropped.  ``n_features_input`` is that raw width, and
    the dropped-position lists say which raw columns are missing and why:

    ``"constant"``
        Removed while building the copula cache (zero standard deviation), so
        the column has no row in ``W`` at all.
    ``"zero_weight_variance"``
        Kept in the cache but carrying no weighted variance, so it takes no
        part in knockoff construction.  These columns still have a ``W`` row.
    ``"zero_residual_variance"``
        Usable before conditioning, then residualized to (near) zero variance
        given ``include``.  Never reported as ``zero_weight_variance``.

    The keys are omitted when the cache cannot prove the raw width, which
    happens only for a prebuilt cache that carries no ``feature_names``.
    """
    if cache.feature_names is None:
        return {}
    n_input = len(cache.feature_names)
    valid = np.asarray(cache.valid_cols, dtype=np.int64)
    dropped: list[tuple[int, str]] = [
        (int(position), "constant")
        for position in set(range(n_input)).difference(valid.tolist())
    ]
    if inactive_valid_positions is not None:
        dropped.extend(
            (int(valid[int(position)]), "zero_weight_variance")
            for position in np.asarray(inactive_valid_positions, dtype=np.int64)
        )
    if extra_valid_drops:
        dropped.extend(
            (int(valid[int(position)]), str(reason))
            for position, reason in extra_valid_drops
        )
    dropped.sort(key=lambda item: item[0])
    return {
        "n_features_input": int(n_input),
        "dropped_feature_positions": [position for position, _ in dropped],
        "dropped_feature_reasons": [reason for _, reason in dropped],
    }


def _stable_group_codes(groups: Sequence[Any]) -> tuple[list[Any], np.ndarray]:
    labels: list[Any] = []
    mapping: dict[Any, int] = {}
    codes = np.empty(len(groups), dtype=np.int64)
    for i, group in enumerate(groups):
        missing = pd.isna(group)
        is_missing = bool(np.any(missing)) if isinstance(missing, np.ndarray) else bool(missing)
        if is_missing:
            raise ValueError("feature_groups must not contain missing values")
        if not isinstance(group, Hashable):
            raise ValueError("feature_groups values must be hashable")
        if group not in mapping:
            mapping[group] = len(labels)
            labels.append(group)
        codes[i] = mapping[group]
    return labels, codes


def _resolve_feature_groups(cache: FeatureCache, feature_groups: Sequence[Any] | None) -> tuple[list[Any], np.ndarray] | None:
    if feature_groups is None:
        return None
    groups_list = list(feature_groups)
    p_valid = cache.Z.shape[1]
    n_original = len(cache.feature_names) if cache.feature_names is not None else None
    if len(groups_list) == p_valid:
        valid_groups = groups_list
    elif n_original is not None and len(groups_list) == n_original:
        valid_groups = [groups_list[int(i)] for i in cache.valid_cols]
    else:
        expected = f"{p_valid}" if n_original is None else f"{p_valid} or {n_original}"
        raise ValueError(
            f"feature_groups has length {len(groups_list)}; expected exactly {expected} "
            "(valid cache columns, or the original input columns)"
        )
    return _stable_group_codes(valid_groups)


def _weighted_variance(Z: np.ndarray, w: np.ndarray, *, batch_size: int = 50_000) -> np.ndarray:
    Z_arr = np.asarray(Z)
    was_1d = Z_arr.ndim == 1
    if was_1d:
        Z_arr = Z_arr[:, None]
    if Z_arr.ndim != 2:
        raise ValueError("Z must be 1D or 2D")
    w64 = np.asarray(w, dtype=np.float64).ravel()
    if Z_arr.shape[0] != w64.shape[0]:
        raise ValueError("w length must match Z rows")
    if not np.isfinite(w64).all() or np.any(w64 < 0.0):
        raise ValueError("cache.sample_weight must be finite and non-negative")
    w_sum = float(w64.sum())
    if w_sum <= 0.0:
        raise ValueError("cache.sample_weight must sum to > 0")
    sums = np.zeros(Z_arr.shape[1], dtype=np.float64)
    sq_sums = np.zeros(Z_arr.shape[1], dtype=np.float64)
    batch_size = max(1, int(batch_size))
    for start in range(0, Z_arr.shape[0], batch_size):
        stop = min(Z_arr.shape[0], start + batch_size)
        Zb = np.asarray(Z_arr[start:stop], dtype=np.float64)
        wb = w64[start:stop]
        # ``np.matmul`` can emit spurious floating-point warnings for finite
        # vector-matrix products with some NumPy 2.x/BLAS combinations.  Dot
        # has the same BLAS-backed reduction semantics without that ufunc
        # warning path.
        sums += np.dot(wb, Zb)
        sq_sums += np.dot(wb, Zb * Zb)
    mean = sums / w_sum
    var = sq_sums / w_sum - mean * mean
    np.maximum(var, 0.0, out=var)
    return var[0] if was_1d else var


def _resolve_cache(
    X,
    *,
    cache: FeatureCache | None,
    sample_weight,
    subsample: Any,
    random_state: int,
    n_jobs: int,
) -> FeatureCache:
    if (X is None) == (cache is None):
        raise ValueError("Exactly one of X or cache must be provided")
    if cache is not None:
        if sample_weight is not None:
            raise ValueError("sample_weight cannot be passed with a prebuilt cache")
        if subsample is not _SUBSAMPLE_DEFAULT:
            raise ValueError("subsample cannot be passed with a prebuilt cache")
        # Knockoff consumers validate the active Rxx submatrix after dropping
        # features with zero weighted variance. Inactive cache columns may
        # legitimately have undefined correlations.
        _validate_prebuilt_cache_structure(cache, validate_rxx=False)
        return cache
    resolved_subsample = 50_000 if subsample is _SUBSAMPLE_DEFAULT else subsample
    return build_cache(
        X,
        sample_weight=sample_weight,
        subsample=resolved_subsample,
        random_state=random_state,
        compute_Rxx=True,
        n_jobs=n_jobs,
        rank_backend="threads" if n_jobs != 1 else "serial",
    )


def _validate_prebuilt_cache_structure(
    cache: FeatureCache,
    *,
    original_n_features: int | None = None,
    n_rows: int | None = None,
    validate_rxx: bool = True,
) -> None:
    """Validate the structural contract of a prebuilt knockoff cache."""
    try:
        cache_vars = vars(cache)
    except TypeError:
        cache_vars = None
    has_provenance_marker = (
        "feature_names_are_synthetic" in cache_vars
        if cache_vars is not None
        else hasattr(cache, "feature_names_are_synthetic")
    )
    if not has_provenance_marker:
        raise ValueError(
            "prebuilt cache lacks feature_names_are_synthetic provenance; "
            "rebuild the cache with the current SIFT version"
        )
    provenance = (
        cache_vars["feature_names_are_synthetic"]
        if cache_vars is not None
        else getattr(cache, "feature_names_are_synthetic")
    )
    if not isinstance(provenance, (bool, np.bool_)):
        raise ValueError("cache.feature_names_are_synthetic must be boolean")

    try:
        Z = np.asarray(cache.Z)
        valid_cols_raw = np.asarray(cache.valid_cols)
        row_idx_raw = np.asarray(cache.row_idx)
        sample_weight = np.asarray(cache.sample_weight)
        n_rows_original = cache.n_rows_original
    except AttributeError as exc:
        raise ValueError("prebuilt cache is missing required structural fields") from exc
    if Z.ndim != 2:
        raise ValueError("cache.Z must be a 2-D array")
    if not np.issubdtype(Z.dtype, np.number) or not np.isrealobj(Z) or not np.isfinite(Z).all():
        raise ValueError("cache.Z must contain only finite real numeric values")
    if valid_cols_raw.ndim != 1 or not np.issubdtype(valid_cols_raw.dtype, np.integer):
        raise ValueError("cache.valid_cols must be a 1-D integer array")
    valid_cols = valid_cols_raw.astype(np.int64, copy=False)
    p_valid = int(Z.shape[1])
    if valid_cols.size != p_valid:
        raise ValueError("cache.valid_cols length must match cache.Z columns")

    feature_names = getattr(cache, "feature_names", None)
    if feature_names is not None:
        if isinstance(feature_names, (str, bytes)):
            raise ValueError("cache.feature_names must be a one-dimensional sequence")
        try:
            feature_names_list = list(feature_names)
            n_original = len(feature_names_list)
        except (TypeError, ValueError) as exc:
            raise ValueError("cache.feature_names must be a sized sequence") from exc
        if any(not isinstance(name, Hashable) for name in feature_names_list):
            raise ValueError("cache.feature_names values must be hashable")
        if original_n_features is not None and n_original != int(original_n_features):
            raise ValueError(
                f"X has {int(original_n_features)} columns but the cache was built "
                f"from {n_original}; cache feature names and order must match X"
            )
        original_n_features = n_original
        if provenance:
            expected_names = [f"x{i}" for i in range(n_original)]
            if feature_names_list != expected_names:
                raise ValueError(
                    "synthetic cache.feature_names must use the canonical positional "
                    "labels x0, x1, ...; rebuild the cache"
                )
    elif not provenance:
        raise ValueError(
            "a named prebuilt cache must include feature_names; rebuild the cache"
        )
    elif original_n_features is None:
        raise ValueError(
            "prebuilt cache must include feature_names to validate original column positions"
        )

    if (
        np.any(valid_cols < 0)
        or np.any(valid_cols >= int(original_n_features))
        or np.unique(valid_cols).size != valid_cols.size
    ):
        raise ValueError("cache.valid_cols must be unique and in bounds for the original X")

    if isinstance(n_rows_original, (bool, np.bool_)) or not isinstance(
        n_rows_original, (int, np.integer)
    ) or int(n_rows_original) < 1:
        raise ValueError("cache.n_rows_original must be a positive integer")
    n_rows_original = int(n_rows_original)
    if n_rows is not None and n_rows_original != int(n_rows):
        raise ValueError(
            f"cache was built with {n_rows_original} rows but X has {int(n_rows)} rows"
        )

    if row_idx_raw.ndim != 1 or not np.issubdtype(row_idx_raw.dtype, np.integer):
        raise ValueError("cache.row_idx must be a 1-D integer array")
    row_idx = row_idx_raw.astype(np.int64, copy=False)
    if (
        row_idx.size != Z.shape[0]
        or np.any(row_idx < 0)
        or np.any(row_idx >= n_rows_original)
        or np.unique(row_idx).size != row_idx.size
    ):
        raise ValueError("cache.row_idx is incompatible with cache rows")

    if sample_weight.ndim != 1 or sample_weight.size != Z.shape[0]:
        raise ValueError("cache.sample_weight must be 1-D and match cache rows")
    try:
        sample_weight_float = sample_weight.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache.sample_weight must be numeric") from exc
    sample_weight_sum = float(sample_weight_float.sum())
    if (
        not np.isfinite(sample_weight_float).all()
        or np.any(sample_weight_float < 0.0)
        or not np.isfinite(sample_weight_sum)
        or sample_weight_sum <= 0.0
    ):
        raise ValueError("cache.sample_weight must be finite, non-negative, and sum to > 0")

    Rxx = getattr(cache, "Rxx", None)
    if Rxx is not None and validate_rxx:
        _validate_cache_rxx(Rxx, p_valid)


def _build_active_rxx(cache: FeatureCache, active: np.ndarray, *, verbose: bool) -> np.ndarray:
    p = cache.Z.shape[1]
    if cache.Rxx is not None:
        R_full = np.asarray(cache.Rxx, dtype=np.float64)
        if R_full.shape != (p, p):
            raise ValueError(f"cache.Rxx must have shape ({p}, {p})")
        active_count = int(active.sum())
        R_active = R_full[np.ix_(active, active)]
        return np.ascontiguousarray(_validate_cache_rxx(R_active, active_count), dtype=np.float64)

    if verbose:
        logger.info("cache.Rxx is None; computing a local weighted correlation matrix.")
    Z_active = (
        np.asarray(cache.Z)
        if bool(active.all())
        else np.ascontiguousarray(cache.Z[:, active])
    )
    return weighted_correlation_matrix(
        Z_active,
        np.asarray(cache.sample_weight, dtype=np.float64),
        backend="blas",
    )


def _pair_screen(
    r: np.ndarray,
    rt: np.ndarray,
    screen_pairs: int | None,
) -> np.ndarray:
    p = r.shape[0]
    if screen_pairs is None or screen_pairs >= p:
        return np.arange(p, dtype=np.int64)
    m = min(p, int(screen_pairs))
    pair_score = np.maximum(np.abs(r), np.abs(rt))
    order = np.lexsort((np.arange(p, dtype=np.int64), -pair_score))
    return np.asarray(order[:m], dtype=np.int64)


def _build_augmented_correlation(
    model: GaussianKnockoffModel,
    kept: np.ndarray,
) -> np.ndarray:
    Sigma_m = np.asarray(model.Sigma_g[np.ix_(kept, kept)], dtype=np.float64)
    D_m = np.diag(np.asarray(model.s[kept], dtype=np.float64))
    cross = Sigma_m - D_m
    return np.block([[Sigma_m, cross], [cross, Sigma_m]])


def _build_context(
    Z: np.ndarray,
    Zt: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    model: GaussianKnockoffModel,
    *,
    screen_pairs: int | None,
    options: dict[str, Any],
    n_jobs: int,
    rng: np.random.Generator,
    build_augmented: bool = True,
    statistic_name: str = "",
    r: np.ndarray | None = None,
    fixed_kept: np.ndarray | None = None,
    fixed_G: np.ndarray | None = None,
) -> KnockoffStatContext:
    r = np.asarray(weighted_corr_with_vector(Z, zy, w) if r is None else r, dtype=np.float64).ravel()
    if r.shape[0] != Z.shape[1]:
        raise ValueError("precomputed r length must match Z columns")
    rt = np.asarray(weighted_corr_with_vector(Zt, zy, w), dtype=np.float64).ravel()
    kept = (
        _pair_screen(r, rt, screen_pairs)
        if fixed_kept is None
        else np.asarray(fixed_kept, dtype=np.int64)
    )
    if build_augmented:
        G = (
            _build_augmented_correlation(model, kept)
            if fixed_G is None
            else np.asarray(fixed_G, dtype=np.float64)
        )
        r_aug = np.concatenate([r[kept], rt[kept]]).astype(np.float64, copy=False)
    else:
        G = np.empty((0, 0), dtype=np.float64)
        r_aug = np.empty(0, dtype=np.float64)
    return KnockoffStatContext(
        Z=Z,
        Zt=Zt,
        zy=zy,
        w=w,
        model=model,
        r=r,
        rt=rt,
        kept=kept,
        G=G,
        r_aug=r_aug,
        options=options,
        n_jobs=n_jobs,
        rng=rng,
        statistic_name=statistic_name,
    )


def _stat_relevance(context: KnockoffStatContext) -> np.ndarray:
    return (
        np.asarray(gaussian_mi_from_corr(context.r), dtype=np.float64)
        - np.asarray(gaussian_mi_from_corr(context.rt), dtype=np.float64)
    )


def _center_weighted(A: np.ndarray, w: np.ndarray) -> np.ndarray:
    w64 = np.asarray(w, dtype=np.float64)
    mean = (w64 @ A) / float(w64.sum())
    return A - mean


def _fit_lasso_coefficients(
    Z_aug: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    *,
    options: dict[str, Any],
    n_jobs: int,
    random_state: int,
    alphas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LassoCV

    Z_centered = _center_weighted(np.asarray(Z_aug, dtype=np.float64), w)
    y_centered = _center_weighted(np.asarray(zy, dtype=np.float64)[:, None], w).ravel()
    sqrt_w = np.sqrt(np.asarray(w, dtype=np.float64))
    X_fit = Z_centered * sqrt_w[:, None]
    y_fit = y_centered * sqrt_w

    params: dict[str, Any] = {
        "cv": int(options.get("cv", 5)),
        "fit_intercept": False,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "max_iter": int(options.get("max_iter", 5000)),
        "tol": float(options.get("tol", 1e-4)),
    }
    if "selection" in options:
        params["selection"] = options["selection"]
    if alphas is not None:
        params["alphas"] = alphas
    elif "alphas" in options:
        params["alphas"] = options["alphas"]
    else:
        params["eps"] = float(options.get("eps", 1e-3))
        params["n_alphas"] = int(options.get("n_alphas", 100))

    model = LassoCV(**params)
    model.fit(X_fit, y_fit)
    return np.asarray(model.coef_, dtype=np.float64), np.asarray(model.alphas_, dtype=np.float64)


def _stat_lcd(context: KnockoffStatContext) -> np.ndarray:
    kept = context.kept
    m = kept.shape[0]
    out = np.zeros(context.Z.shape[1], dtype=np.float64)
    if m == 0:
        return out

    Z_m = np.asarray(context.Z[:, kept], dtype=np.float64)
    Zt_m = np.asarray(context.Zt[:, kept], dtype=np.float64)
    seed = int(context.rng.integers(0, np.iinfo(np.int32).max))
    inner_n_jobs = context.n_jobs
    beta1, alphas = _fit_lasso_coefficients(
        np.column_stack([Z_m, Zt_m]),
        context.zy,
        context.w,
        options=context.options,
        n_jobs=inner_n_jobs,
        random_state=seed,
    )
    beta2, _ = _fit_lasso_coefficients(
        np.column_stack([Zt_m, Z_m]),
        context.zy,
        context.w,
        options=context.options,
        n_jobs=inner_n_jobs,
        random_state=seed,
        alphas=alphas,
    )
    W_kept = 0.5 * (np.abs(beta1[:m]) - np.abs(beta1[m:]))
    W_kept += 0.5 * (np.abs(beta2[m:]) - np.abs(beta2[:m]))
    out[kept] = W_kept
    return out


def _validate_path_depth(value: Any, m: int, *, default: int | None = None) -> int:
    if value is None:
        return m if default is None else min(m, default)
    depth = _validate_positive_int(value, "path_depth")
    return min(depth, 2 * m)


def _default_cefsplus_path_depth(q: float, offset: int, m: int) -> int:
    """Choose a bounded q-aware starting depth for the greedy knockoff path."""
    q_aware = int(np.ceil(2.0 * max(1, offset) / q))
    return _validate_path_depth(
        None,
        m,
        default=max(_CEFSPLUS_DEFAULT_PATH_DEPTH, q_aware),
    )


def _cefsplus_incremental_scores(
    G: np.ndarray,
    r: np.ndarray,
    *,
    path_depth: int,
    tie_break: np.ndarray,
    min_gain_ratio: float = 0.0,
    shrink: float = 1e-6,
    eps: float = 1e-10,
    tie_tol: float = 1e-12,
) -> np.ndarray:
    G_arr = np.asarray(G, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64).ravel()
    if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
        raise ValueError("G must be square")
    if G_arr.shape[0] != r_arr.shape[0] or G_arr.shape[0] % 2:
        raise ValueError("G/r dimensions must describe original-knockoff pairs")
    if not np.isfinite(G_arr).all() or not np.isfinite(r_arr).all():
        raise ValueError("G and r must contain only finite values")

    n_aug = r_arr.shape[0]
    n_pairs = n_aug // 2
    if n_aug == 0 or path_depth <= 0 or np.all(np.abs(r_arr) <= tie_tol):
        return np.zeros(n_aug, dtype=np.float64)

    Gs = (1.0 - shrink) * G_arr.copy()
    np.fill_diagonal(Gs, 1.0)
    rs = (1.0 - shrink) * r_arr
    tie_break_arr = np.asarray(tie_break, dtype=np.float64).ravel()
    if tie_break_arr.shape[0] != n_aug:
        tie_break_arr = np.asarray(gaussian_mi_from_corr(rs), dtype=np.float64)

    h = np.zeros(n_aug, dtype=np.float64)
    remaining = np.ones(n_aug, dtype=bool)
    selected = np.empty(0, dtype=np.int64)
    inv_S = np.empty((0, 0), dtype=np.float64)
    inv_yS = np.array([[1.0]], dtype=np.float64)
    logdet_S = 0.0
    logdet_yS = 0.0
    count = 0
    min_gain_abs = 0.0

    while count < path_depth and bool(remaining.any()):
        rem = np.flatnonzero(remaining)
        s = selected.shape[0]
        if s == 0:
            s1 = np.ones(rem.shape[0], dtype=np.float64)
            lf = np.zeros(rem.shape[0], dtype=np.float64)
            B = np.empty((0, rem.shape[0]), dtype=np.float64)
        else:
            B = Gs[np.ix_(selected, rem)]
            tmp = inv_S @ B
            s1 = np.maximum(1.0 - np.einsum("ij,ij->j", B, tmp), eps)
            lf = logdet_S + np.log(s1)

        B2 = np.vstack([rs[rem], B])
        tmp2 = inv_yS @ B2
        s2 = np.maximum(1.0 - np.einsum("ij,ij->j", B2, tmp2), eps)
        lc = logdet_yS + np.log(s2)
        scores = lf - lc
        best_score = float(np.max(scores))
        if not np.isfinite(best_score):
            break
        gain_best = best_score - (logdet_S - logdet_yS)
        if count > 0 and min_gain_ratio > 0.0 and gain_best < min_gain_abs:
            break
        if count == 0:
            first_gain = max(gain_best, eps)
            min_gain_abs = min_gain_ratio * first_gain
        tied = rem[np.abs(scores - best_score) <= tie_tol]

        pair_ids = tied % n_pairs
        sides = tied >= n_pairs
        neutralized = False
        for pair_id in np.unique(pair_ids):
            pair_sides = sides[pair_ids == pair_id]
            if pair_sides.size > 1 and np.any(pair_sides) and np.any(~pair_sides):
                remaining[int(pair_id)] = False
                remaining[int(pair_id) + n_pairs] = False
                neutralized = True
        if neutralized:
            continue

        best_tie_break = float(np.max(tie_break_arr[tied]))
        tied = tied[np.abs(tie_break_arr[tied] - best_tie_break) <= tie_tol]
        pair_ids = tied % n_pairs
        sides = tied >= n_pairs
        neutralized = False
        for pair_id in np.unique(pair_ids):
            pair_sides = sides[pair_ids == pair_id]
            if pair_sides.size > 1 and np.any(pair_sides) and np.any(~pair_sides):
                remaining[int(pair_id)] = False
                remaining[int(pair_id) + n_pairs] = False
                neutralized = True
        if neutralized:
            continue

        pair_order = tied % n_pairs
        j = int(tied[np.argmin(pair_order)])
        rem_pos = int(np.where(rem == j)[0][0])
        s1_best = s1[rem_pos]

        if s == 0:
            inv_S = np.array([[1.0 / s1_best]], dtype=np.float64)
        else:
            b = B[:, rem_pos].reshape(-1, 1)
            v = inv_S @ b
            inv_S_new = np.empty((s + 1, s + 1), dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                inv_S_new[:s, :s] = inv_S + (v @ v.T) / s1_best
                inv_S_new[:s, s] = (-v[:, 0]) / s1_best
                inv_S_new[s, :s] = (-v[:, 0]) / s1_best
                inv_S_new[s, s] = 1.0 / s1_best
            if not np.isfinite(inv_S_new).all():
                break
            inv_S = inv_S_new
        logdet_S += float(np.log(s1[rem_pos]))

        b2 = B2[:, rem_pos].reshape(-1, 1)
        v2 = inv_yS @ b2
        s2_best = s2[rem_pos]
        inv_yS_new = np.empty((s + 2, s + 2), dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            inv_yS_new[: s + 1, : s + 1] = inv_yS + (v2 @ v2.T) / s2_best
            inv_yS_new[: s + 1, s + 1] = (-v2[:, 0]) / s2_best
            inv_yS_new[s + 1, : s + 1] = (-v2[:, 0]) / s2_best
            inv_yS_new[s + 1, s + 1] = 1.0 / s2_best
        if not np.isfinite(inv_yS_new).all():
            break
        inv_yS = inv_yS_new
        logdet_yS += float(np.log(s2_best))

        gain = float(np.log(s1_best) - np.log(s2_best))
        h[j] = max(gain, 0.0)
        selected = np.append(selected, j)
        remaining[j] = False
        count += 1

    return h


def _stat_cefsplus(context: KnockoffStatContext) -> np.ndarray:
    kept = context.kept
    m = kept.shape[0]
    out = np.zeros(context.Z.shape[1], dtype=np.float64)
    if m == 0:
        return out
    path_depth = _validate_path_depth(
        context.options.get("path_depth"),
        m,
        default=_CEFSPLUS_DEFAULT_PATH_DEPTH,
    )
    r_aug = np.asarray(context.r_aug, dtype=np.float64)
    if np.all(np.abs(r_aug) <= 1e-12):
        return out
    tie_break = np.asarray(gaussian_mi_from_corr(r_aug), dtype=np.float64)
    min_gain_ratio = _validate_nonnegative_float(context.options.get("min_gain_ratio", 0.0), "min_gain_ratio")
    adaptive = bool(context.options.get("_adaptive_path_depth", False))
    q = float(context.options.get("_q", 0.1))
    offset = int(context.options.get("_offset", 1))
    while True:
        h = _cefsplus_incremental_scores(
            context.G,
            r_aug,
            path_depth=path_depth,
            tie_break=tie_break,
            min_gain_ratio=min_gain_ratio,
        )
        W_kept = h[:m] - h[m:]
        threshold = knockoff_threshold(W_kept, q, offset=offset)
        n_selected = int(np.sum(W_kept >= threshold)) if np.isfinite(threshold) else 0
        saturated = n_selected >= path_depth and path_depth < 2 * m
        if not adaptive or not saturated:
            break
        path_depth = min(2 * m, 2 * path_depth)

    context.options["_path_depth_used"] = max(
        int(context.options.get("_path_depth_used", 0)),
        int(path_depth),
    )
    context.options["_path_depth_saturated"] = bool(
        context.options.get("_path_depth_saturated", False) or saturated
    )
    out[kept] = h[:m] - h[m:]
    return out


def _stat_ridge(context: KnockoffStatContext) -> np.ndarray:
    """Ridge coefficient difference on the analytic augmented correlation.

    ``beta = (G + lambda I)^{-1} [r; r_tilde]`` where ``G`` is the analytic
    original/knockoff correlation of the screened pairs. ``G`` is invariant
    under swapping a feature with its knockoff, so swapping ``r_j`` and
    ``r_tilde_j`` swaps ``beta_j`` and ``beta_{j+m}`` exactly and
    ``W_j = |beta_j| - |beta_{j+m}|`` flips sign: the statistic is
    antisymmetric by construction, with no path or tie dependence.
    """
    kept = context.kept
    m = kept.shape[0]
    out = np.zeros(context.Z.shape[1], dtype=np.float64)
    if m == 0:
        return out
    lam = _validate_nonnegative_float(context.options.get("ridge_lambda", 0.5), "ridge_lambda")
    if lam <= 0.0:
        raise ValueError("ridge_lambda must be > 0")
    G = np.asarray(context.G, dtype=np.float64)
    r_aug = np.asarray(context.r_aug, dtype=np.float64)
    A = G + lam * np.eye(G.shape[0], dtype=np.float64)
    try:
        cf = cho_factor(A, lower=True, check_finite=False)
        beta = cho_solve(cf, r_aug, check_finite=False)
    except LinAlgError:
        beta = np.linalg.lstsq(A, r_aug, rcond=None)[0]
    out[kept] = np.abs(beta[:m]) - np.abs(beta[m:])
    return out


def _lasso_entry_penalties(
    alphas: np.ndarray,
    coef_path: np.ndarray,
    final_active: Sequence[int],
) -> np.ndarray:
    """Recover first-entry penalties from a LARS coefficient path.

    ``lars_path_gram(..., method="lasso")`` returns the active variables at
    the *end* of the path. Variables can drop and re-enter, so that final list
    cannot be zipped to the path alphas. A variable first becomes nonzero one
    knot after it is admitted; its entry penalty is therefore the preceding
    alpha. A final active variable that was admitted at the truncated last knot
    can still have an all-zero coefficient path, so retain that one boundary
    case at the last returned alpha.
    """
    alpha_arr = np.asarray(alphas, dtype=np.float64).ravel()
    coefs = np.asarray(coef_path, dtype=np.float64)
    if coefs.ndim != 2:
        raise ValueError("coef_path must be a 2D array")
    if coefs.shape[1] != alpha_arr.shape[0]:
        raise ValueError("coef_path columns must match the number of alphas")

    n_features = coefs.shape[0]
    entry = np.zeros(n_features, dtype=np.float64)
    if alpha_arr.size == 0 or n_features == 0:
        return entry

    nonzero = coefs != 0.0
    entered = np.any(nonzero, axis=1)
    first_nonzero = np.argmax(nonzero, axis=1)
    entered_idx = np.flatnonzero(entered)
    if entered_idx.size:
        alpha_idx = np.maximum(first_nonzero[entered_idx] - 1, 0)
        entry[entered_idx] = np.maximum(alpha_arr[alpha_idx], 0.0)

    # When max_iter truncates immediately after admitting a variable, its
    # coefficient has not yet moved away from zero. It is nevertheless in the
    # returned terminal active set and entered at the final knot.
    for col in np.asarray(final_active, dtype=np.int64):
        col_int = int(col)
        if 0 <= col_int < n_features and not entered[col_int]:
            entry[col_int] = float(max(alpha_arr[-1], 0.0))
    return entry


def _stat_lsm(context: KnockoffStatContext) -> np.ndarray:
    """Lasso signed-max statistic from a Gram-form LARS path.

    Runs the lasso path (LARS) on the analytic augmented correlation ``G`` and
    the observed correlations ``[r; r_tilde]`` and records the penalty at which
    each column first enters. ``W_j = max(Z_j, Z_tilde_j) * sign(Z_j -
    Z_tilde_j)``. Because ``G`` is swap-invariant and LARS is permutation
    equivariant, swapping a pair swaps the entry penalties and flips ``W_j``.
    Columns that never enter within ``max_steps`` get ``Z = 0``; a pair with
    both entries zero contributes ``W = 0`` and is ignored by the threshold.
    """
    from sklearn.linear_model import lars_path_gram

    kept = context.kept
    m = kept.shape[0]
    out = np.zeros(context.Z.shape[1], dtype=np.float64)
    if m == 0:
        return out
    n_rows = int(context.Z.shape[0])
    max_steps_opt = context.options.get("max_steps")
    if max_steps_opt is None:
        max_steps = min(2 * m, max(200, 4 * _CEFSPLUS_DEFAULT_PATH_DEPTH * 10))
    else:
        max_steps = min(2 * m, _validate_positive_int(max_steps_opt, "max_steps"))
    G = np.asarray(context.G, dtype=np.float64)
    r_aug = np.asarray(context.r_aug, dtype=np.float64)
    if np.all(np.abs(r_aug) <= 1e-12):
        return out
    # Scale by n so the penalties live on the usual lasso scale; LARS is
    # equivariant to this common scaling of Gram and Xy.
    alphas, active, coefs = lars_path_gram(
        Xy=r_aug * n_rows,
        Gram=G * n_rows,
        n_samples=n_rows,
        method="lasso",
        max_iter=int(max_steps),
        eps=np.finfo(np.float64).eps,
    )
    entry = _lasso_entry_penalties(alphas, coefs, active)
    z_orig = entry[:m]
    z_ko = entry[m:]
    out[kept] = np.maximum(z_orig, z_ko) * np.sign(z_orig - z_ko)
    return out


def _reserved_statistic(context: KnockoffStatContext) -> np.ndarray:
    name = context.statistic_name or "statistic"
    raise ValueError(f"{name} {_STATISTIC_NOT_ENABLED}")


_KNOCKOFF_STAT_REGISTRY: dict[str, KnockoffStatSpec] = {
    "relevance": KnockoffStatSpec(
        "relevance",
        _stat_relevance,
        enabled=True,
        needs_screening=False,
        allowed_options=frozenset(),
    ),
    "lcd": KnockoffStatSpec(
        "lcd",
        _stat_lcd,
        enabled=False,
        needs_screening=True,
        allowed_options=frozenset({"cv", "max_iter", "tol", "selection", "alphas", "eps", "n_alphas"}),
    ),
    "cefsplus": KnockoffStatSpec(
        "cefsplus",
        _stat_cefsplus,
        enabled=True,
        needs_screening=True,
        allowed_options=frozenset({"path_depth", "min_gain_ratio"}),
    ),
    "lsm": KnockoffStatSpec(
        "lsm",
        _stat_lsm,
        enabled=True,
        needs_screening=True,
        allowed_options=frozenset({"max_steps"}),
    ),
    "ridge": KnockoffStatSpec(
        "ridge",
        _stat_ridge,
        enabled=True,
        needs_screening=True,
        allowed_options=frozenset({"ridge_lambda"}),
    ),
    "mrmr_diff": KnockoffStatSpec("mrmr_diff", _reserved_statistic, enabled=False),
    "mrmr_quot": KnockoffStatSpec("mrmr_quot", _reserved_statistic, enabled=False),
    "jmi": KnockoffStatSpec("jmi", _reserved_statistic, enabled=False),
    "jmim": KnockoffStatSpec("jmim", _reserved_statistic, enabled=False),
}
VALID_KNOCKOFF_STATISTICS = tuple(_KNOCKOFF_STAT_REGISTRY)


def _get_statistic(statistic: str) -> KnockoffStatSpec:
    key = str(statistic).lower()
    if key not in _KNOCKOFF_STAT_REGISTRY:
        valid = ", ".join(VALID_KNOCKOFF_STATISTICS)
        raise ValueError(f"Unknown knockoff statistic {statistic!r}; expected one of: {valid}")
    spec = _KNOCKOFF_STAT_REGISTRY[key]
    if not spec.enabled:
        raise ValueError(f"Knockoff statistic {key!r} {_STATISTIC_NOT_ENABLED}")
    return spec


def knockoff_threshold(W: np.ndarray, q: float, *, offset: int = 1) -> float:
    """Return the knockoff/knockoff+ threshold, or ``inf`` if none exists."""

    q_float = _validate_probability(q, "q")
    offset_int = _validate_offset(offset)
    W_arr = np.asarray(W, dtype=np.float64).ravel()
    if not np.isfinite(W_arr).all():
        raise ValueError("W must contain only finite values")
    positive = np.sort(W_arr[W_arr > 0.0])
    negative_magnitudes = np.sort(-W_arr[W_arr < 0.0])
    if positive.size == 0 and negative_magnitudes.size == 0:
        return float(np.inf)
    ts = np.union1d(positive, negative_magnitudes)
    n_positive = positive.size - np.searchsorted(positive, ts, side="left")
    n_negative = negative_magnitudes.size - np.searchsorted(
        negative_magnitudes, ts, side="left"
    )
    fdp = (offset_int + n_negative) / np.maximum(1, n_positive)
    eligible = np.flatnonzero(fdp <= q_float)
    if eligible.size:
        return float(ts[int(eligible[0])])
    return float(np.inf)


def _group_knockoff_statistics(
    W: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    W_arr = np.asarray(W, dtype=np.float64).ravel()
    codes = np.asarray(group_codes, dtype=np.int64).ravel()
    if W_arr.shape[0] != codes.shape[0]:
        raise ValueError("group_codes length must match W")
    out = np.zeros(n_groups, dtype=np.float64)
    for group_idx in range(n_groups):
        values = W_arr[codes == group_idx]
        if values.size == 0:
            continue
        max_abs = float(np.max(np.abs(values)))
        if max_abs <= tol:
            continue
        tied = values[np.abs(np.abs(values) - max_abs) <= tol]
        has_pos = bool(np.any(tied > tol))
        has_neg = bool(np.any(tied < -tol))
        if has_pos and has_neg:
            continue
        out[group_idx] = max_abs if has_pos else -max_abs
    return out


def sample_knockoffs(
    cache: FeatureCache,
    *,
    s_method: str = "equi",
    min_eig: float = 1e-3,
    random_state: int = 0,
) -> np.ndarray:
    """Fit and sample one Gaussian-copula knockoff draw for a cache.

    Fits the second-order Gaussian knockoff operators from the cache's own
    copula correlation matrix and returns one sampled knockoff copy of
    ``cache.Z``, laid out on exactly the same columns.  This is an advanced
    helper for diagnostics and for building custom feature statistics -- for
    ordinary discovery use ``select_fdr``, which does the fitting,
    sampling, statistic, and thresholding in one call.  With defaults it uses
    equicorrelated decorrelation and seed 0, and returns a fresh float32
    array; nothing is cached or mutated on ``cache``.

    Parameters
    ----------
    cache : FeatureCache
        Cache from ``sift.build_cache``.  Its structural contract is
        revalidated here, duplicate non-synthetic feature names are rejected,
        and its weights must be finite, non-negative, and sum above zero.
        ``Rxx`` is used when present and recomputed locally otherwise.
    s_method : {"equi", "mvr", "me"}, default "equi"
        How the knockoff decorrelation vector ``s`` is solved.  ``"equi"`` is
        the cheapest; ``"mvr"`` and ``"me"`` run diagonal coordinate descent
        and can raise power on correlated designs.
    min_eig : float, default 0.001
        Minimum eigenvalue required of the correlation matrix.  A matrix that
        falls short is mixed with the identity and a ``UserWarning`` is
        emitted.
    random_state : int, default 0
        Seed for the knockoff noise draw.  The same seed and cache reproduce
        the same matrix exactly.

    Returns
    -------
    ndarray of shape (n_cached_rows, n_valid_features), float32
        One knockoff copy aligned column-for-column with ``cache.Z``.  Columns
        with no weighted variance take no part in the construction and come
        back as exact zeros.

    Raises
    ------
    ValueError
        If the cache fails its structural or provenance checks, carries
        duplicate feature names, has weights that are non-finite, negative, or
        sum to zero, or retains no non-constant feature.

    Warns
    -----
    UserWarning
        When the copula correlation matrix had to be shrunk toward the
        identity to reach ``min_eig``; the knockoffs are then an approximate
        plug-in model and exact Model-X FDR is not claimed.

    See Also
    --------
    select_fdr : The full q-calibrated knockoff filter.
    sift.build_cache : Build the cache this helper consumes.
    KnockoffSelectionResult : The result object ``select_fdr`` returns.

    Notes
    -----
    The knockoffs are second-order: they match the copula correlation matrix
    ``Sigma`` in the sense that the joint covariance of ``(Z, Z_tilde)`` has
    ``Sigma`` on both diagonal blocks and ``Sigma - diag(s)`` off-diagonal, so
    each column is decorrelated from its own knockoff by ``s_j`` while every
    cross-correlation is preserved.  Larger ``s`` means more power and,
    because ``s`` is bounded by the smallest eigenvalue, near-collinear
    columns drive it toward zero.  Only one draw is returned; independent
    draws come from different ``random_state`` values, and combining them is
    derandomization, which does not preserve a per-draw FDR claim.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import build_cache, sample_knockoffs
    >>> rng = np.random.default_rng(0)
    >>> cache = build_cache(rng.normal(size=(200, 5)), compute_Rxx=True)
    >>> Z_tilde = sample_knockoffs(cache, random_state=123)
    >>> Z_tilde.shape == cache.Z.shape, Z_tilde.dtype
    (True, dtype('float32'))
    >>> repeat = sample_knockoffs(cache, random_state=123)
    >>> bool(np.array_equal(Z_tilde, repeat))  # same seed, same draw
    True
    """

    _validate_prebuilt_cache_structure(cache, validate_rxx=False)
    _reject_duplicate_feature_names(cache)
    w = np.asarray(cache.sample_weight, dtype=np.float64)
    if not np.isfinite(w).all() or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("cache.sample_weight must be finite, non-negative, and sum to > 0")
    variances = _weighted_variance(cache.Z, w)
    active = variances > 1e-12
    if not bool(active.any()):
        raise ValueError("No active non-constant features remain for knockoffs")
    R_active = _build_active_rxx(cache, active, verbose=False)
    model = fit_gaussian_knockoffs(R_active, s_method=s_method, min_eig=min_eig)
    rng = np.random.default_rng(random_state)
    Z_active = (
        np.asarray(cache.Z, dtype=np.float32)
        if bool(active.all())
        else np.ascontiguousarray(cache.Z[:, active], dtype=np.float32)
    )
    Zt_active = sample_gaussian_knockoffs(Z_active, model, rng)
    Zt = np.zeros_like(cache.Z, dtype=np.float32)
    Zt[:, active] = Zt_active
    return Zt


def _unclipped_weighted_gram(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted Gram of already-standardized columns, without off-diagonal clip.

    ``weighted_correlation_matrix`` clips correlations to ``±0.999999``, which
    lifts the smallest eigenvalue of an exact duplicate block to ``1e-6``.
    The include-rank guard needs the raw Gram so exact and numerical
    singularity remain visible.
    """
    Z64 = np.ascontiguousarray(Z, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64).ravel()
    w_sum = float(w64.sum())
    zw = Z64 * np.sqrt(w64)[:, None]
    gram = zw.T @ zw
    gram /= w_sum
    return 0.5 * (gram + gram.T)


def _residualize_discovery_given_include(
    Z: np.ndarray,
    zy: np.ndarray,
    w: np.ndarray,
    *,
    include_valid: np.ndarray,
    discovery_valid: np.ndarray,
    shrink: float = 1e-6,
    min_eig: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partial out include columns in rank-Gaussian space (Schur complement).

    ``min_eig`` is the documented tolerance on the *unregularized* include
    Gram (weighted, unclipped). Exact copies and rank-equivalent monotone
    transforms fall below it and raise. Ridge-style ``shrink`` is applied
    only after that check, and only for the Schur solve.
    """
    from sift.selection.panel import local_standardize

    if include_valid.size == 0:
        active = np.zeros(Z.shape[1], dtype=bool)
        active[np.asarray(discovery_valid, dtype=np.int64)] = True
        return np.asarray(Z, dtype=np.float64), np.asarray(zy, dtype=np.float64), active
    Zs = np.ascontiguousarray(Z[:, include_valid], dtype=np.float64)
    Zd = np.ascontiguousarray(Z[:, discovery_valid], dtype=np.float64)
    n_s = int(Zs.shape[1])
    w_arr = np.asarray(w, dtype=np.float64)
    Z_joint = np.ascontiguousarray(np.concatenate([Zs, Zd], axis=1), dtype=np.float64)
    G_joint = _unclipped_weighted_gram(Z_joint, w_arr)
    R_ss_raw = np.ascontiguousarray(G_joint[:n_s, :n_s])
    eig_min = float(np.min(np.linalg.eigvalsh(R_ss_raw)))
    if not np.isfinite(eig_min) or eig_min < min_eig:
        raise ValueError(
            "include set is numerically singular; cannot condition the knockoff model"
        )
    raw_coef = np.linalg.solve(R_ss_raw, G_joint[:n_s, n_s:])
    residual_var = _weighted_variance(Zd - Zs @ raw_coef, w_arr)
    R = weighted_correlation_matrix(Z_joint, w_arr, backend="blas")
    R_ss = (1.0 - shrink) * np.asarray(R[:n_s, :n_s], dtype=np.float64) + shrink * np.eye(
        n_s, dtype=np.float64
    )
    R_sd = np.asarray(R[:n_s, n_s:], dtype=np.float64)
    coef = np.linalg.solve(R_ss, R_sd)
    Zd_res = Zd - Zs @ coef
    r_ys = np.asarray(weighted_corr_with_vector(Zs, zy, w_arr), dtype=np.float64)
    b_y = np.linalg.solve(R_ss, r_ys)
    zy_res = np.asarray(zy, dtype=np.float64) - Zs @ b_y
    Zd_res = local_standardize(Zd_res, w_arr)
    zy_res = local_standardize(zy_res.reshape(-1, 1), w_arr).ravel()
    Z_out = np.array(Z, dtype=np.float64, copy=True)
    Z_out[:, np.asarray(discovery_valid, dtype=np.int64)] = Zd_res
    active = np.zeros(Z.shape[1], dtype=bool)
    keep = residual_var > 1e-12
    active[np.asarray(discovery_valid, dtype=np.int64)[keep]] = True
    return Z_out, np.asarray(zy_res, dtype=np.float64), active


def _all_zero_result(
    *,
    cache: FeatureCache,
    feature_names: list[Any],
    group_labels: list[Any] | None = None,
    group_codes: np.ndarray | None = None,
    relevance: np.ndarray,
    metadata: dict[str, Any],
    diagnostic_reason: str,
    resolved_sets=None,
    cache_names: list[Any] | None = None,
    include_valid: np.ndarray | None = None,
    provenance: str | None = None,
    discovery_valid_mask: np.ndarray | None = None,
) -> KnockoffSelectionResult:
    n_draws = int(metadata.get("n_draws", 1))
    if n_draws == 1:
        threshold: float | None = float(np.inf)
        selection_frequency_arr = np.full(len(feature_names), np.nan)
        selection_frequency = None
    else:
        threshold = None
        selection_frequency_arr = np.zeros(len(feature_names), dtype=np.float64)
        selection_frequency = pd.Series(
            selection_frequency_arr,
            index=feature_names,
            name="selection_frequency",
        )
    selected_mask = np.zeros(len(feature_names), dtype=bool)
    role = np.array(["ineligible"] * len(feature_names), dtype=object)
    if discovery_valid_mask is not None:
        mask = np.asarray(discovery_valid_mask, dtype=bool).reshape(-1)
        if mask.shape[0] == role.shape[0]:
            role[mask] = "discovery"
    selected_features: list[Any] = []
    selected_indices: list[int] = []
    if resolved_sets is not None and resolved_sets.include:
        names = cache_names if cache_names is not None else feature_names
        selected_features, selected_indices = compose_selected(
            names, resolved_sets.include, []
        )
        if include_valid is not None:
            for valid_i in include_valid:
                selected_mask[int(valid_i)] = True
                role[int(valid_i)] = "include"
                if n_draws != 1:
                    selection_frequency_arr[int(valid_i)] = 1.0
    zero_cols = {
        "feature": feature_names,
        "selected_index": cache.valid_cols.astype(np.int64),
        "W": np.zeros(len(feature_names), dtype=np.float64),
        "selected": selected_mask,
        "selection_frequency": selection_frequency_arr,
        "relevance": relevance,
    }
    if resolved_sets is not None and getattr(resolved_sets, "active", False):
        zero_cols["role"] = role
    zero_cols["selector"] = "knockoff_fdr"
    W_table = pd.DataFrame(zero_cols)
    if group_labels is not None and group_codes is not None:
        W_table["feature_group"] = [group_labels[int(code)] for code in group_codes]
    for draw_idx in range(n_draws):
        W_table[f"W_draw_{draw_idx}"] = np.zeros(len(feature_names), dtype=np.float64)
    diagnostics = {
        "thresholds": [float(np.inf)] * n_draws,
        "selection_sets": [[] for _ in range(n_draws)],
        "reason": diagnostic_reason,
    }
    if group_labels is not None and group_codes is not None:
        diagnostics["feature_groups"] = group_labels
        diagnostics["group_W_draws"] = [
            [0.0] * len(group_labels)
            for _ in range(n_draws)
        ]
        diagnostics["group_thresholds"] = [float(np.inf)] * n_draws
    cond_record = conditioning_record(
        resolved_sets,
        feature_names=cache_names or feature_names,
        discovered_idx=[],
        include_provenance=provenance,
    )
    if cond_record is not None:
        diagnostics["conditioning"] = cond_record
        metadata = dict(metadata)
        metadata["conditioning"] = cond_record
    n_draws = int(metadata.get("n_draws", 1))
    if discovery_valid_mask is not None:
        eligible_local = np.flatnonzero(np.asarray(discovery_valid_mask, dtype=bool))
    else:
        eligible_local = np.arange(len(feature_names), dtype=np.int64)
    if group_labels is not None and group_codes is not None:
        eligible_ids = tuple(sorted({int(group_codes[int(i)]) for i in eligible_local}))
        n_tested_unit = "group"
    else:
        eligible_ids = tuple(int(i) for i in eligible_local)
        n_tested_unit = "feature"
    metadata = dict(metadata)
    metadata.update(
        _feasibility_metadata(
            n_tested_per_draw=[],
            n_tested_unit=n_tested_unit,
            tested_id_sets=[],
            n_discoveries_offset_0_per_draw=[],
            n_discoveries_offset_0=0,
            n_eligible=len(eligible_ids),
            tested_state="not_run",
            q=float(metadata.get("q", 0.1)),
            offset=int(metadata.get("offset", 1)),
        )
    )
    diagnostics["offset_zero_selection_sets"] = []
    diagnostics["tested_state"] = "not_run"
    _warn_knockoff_plus_infeasible(metadata)
    return KnockoffSelectionResult(
        selected_features=selected_features,
        selected_indices=selected_indices,
        selector_metadata=metadata,
        W=W_table,
        threshold=threshold,
        selection_frequency=selection_frequency,
        diagnostics_=diagnostics,
    )


def cluster_feature_groups(
    Rxx: np.ndarray,
    *,
    corr_threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster features by absolute correlation and pick a medoid per cluster.

    Returns ``(labels, representatives)`` where ``labels[j]`` is the 0-based
    cluster id of feature ``j`` and ``representatives[c]`` is the member of
    cluster ``c`` with the largest total absolute correlation to its cluster.
    Uses average linkage on ``1 - |R|`` cut at ``1 - corr_threshold``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    threshold = float(corr_threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("group_corr_threshold must be a finite float in (0, 1)")
    R = np.abs(np.asarray(Rxx, dtype=np.float64))
    p = R.shape[0]
    if R.ndim != 2 or R.shape != (p, p):
        raise ValueError("Rxx must be a square correlation matrix")
    if p == 1:
        return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    D = np.clip(1.0 - R, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z_link = linkage(squareform(D, checks=False), method="average")
    raw = fcluster(Z_link, t=1.0 - threshold, criterion="distance")
    _, labels = np.unique(raw, return_inverse=True)
    labels = labels.astype(np.int64)
    n_clusters = int(labels.max()) + 1
    reps = np.empty(n_clusters, dtype=np.int64)
    for c in range(n_clusters):
        members = np.flatnonzero(labels == c)
        if members.shape[0] == 1:
            reps[c] = members[0]
            continue
        sub = R[np.ix_(members, members)]
        reps[c] = members[int(np.argmax(sub.sum(axis=1)))]
    return labels, reps


def _select_fdr_cluster_representatives(
    cache: FeatureCache,
    y,
    *,
    q: float,
    statistic: str,
    n_draws: int,
    eta: float,
    offset: int,
    s_method: str,
    min_eig: float,
    screen_pairs: int | None,
    statistic_options: dict,
    group_corr_threshold: float,
    random_state: int,
    n_jobs: int,
    verbose: bool,
) -> KnockoffSelectionResult:
    p_valid = cache.Z.shape[1]
    active_all = np.ones(p_valid, dtype=bool)
    R_full = _build_active_rxx(cache, active_all, verbose=verbose)
    labels, reps = cluster_feature_groups(R_full, corr_threshold=group_corr_threshold)
    reps_sorted = np.sort(reps)
    reduced = FeatureCache(
        Z=np.ascontiguousarray(cache.Z[:, reps_sorted]),
        Rxx=np.ascontiguousarray(R_full[np.ix_(reps_sorted, reps_sorted)]).astype(np.float32),
        valid_cols=np.asarray(cache.valid_cols, dtype=np.int64)[reps_sorted],
        row_idx=cache.row_idx,
        sample_weight=cache.sample_weight,
        n_rows_original=cache.n_rows_original,
        feature_names=cache.feature_names,
        feature_names_are_synthetic=cache.feature_names_are_synthetic,
    )
    if verbose:
        logger.info(
            f"feature_groups='auto': {p_valid} features -> {reps_sorted.shape[0]} "
            f"clusters at |corr| >= {group_corr_threshold:g}; running knockoffs on representatives"
        )
    rep_result = select_fdr(
        y=y,
        cache=reduced,
        q=q,
        statistic=statistic,
        n_draws=n_draws,
        eta=eta,
        offset=offset,
        s_method=s_method,
        min_eig=min_eig,
        screen_pairs=screen_pairs,
        statistic_options=statistic_options,
        feature_groups=None,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Expand representative-level results back to every valid feature.
    feature_names = _feature_names_for_valid_cols(cache)
    rep_of_feature = reps[labels]  # valid-column position of each feature's representative
    rep_row = {int(pos): i for i, pos in enumerate(reps_sorted)}
    W_rep = rep_result.W["W"].to_numpy()
    selected_rep = rep_result.W["selected"].to_numpy()
    freq_rep = rep_result.W["selection_frequency"].to_numpy()
    rel_rep = rep_result.W["relevance"].to_numpy()
    idx_in_rep = np.asarray([rep_row[int(pos)] for pos in rep_of_feature], dtype=np.int64)

    W_table = pd.DataFrame(
        {
            "feature": feature_names,
            "selected_index": np.asarray(cache.valid_cols, dtype=np.int64),
            "W": W_rep[idx_in_rep],
            "selected": selected_rep[idx_in_rep],
            "selection_frequency": freq_rep[idx_in_rep],
            "relevance": rel_rep[idx_in_rep],
            "selector": "knockoff_fdr",
            "feature_group": labels.astype(int),
            "is_representative": np.asarray([int(rep_of_feature[j]) == j for j in range(p_valid)], dtype=bool),
        }
    )
    for col in rep_result.W.columns:
        if col.startswith("W_draw_"):
            W_table[col] = rep_result.W[col].to_numpy()[idx_in_rep]

    # Selected features: members of selected clusters, ordered by cluster W
    # (representatives first within a cluster, then valid-column order).
    order = np.lexsort(
        (
            np.arange(p_valid),
            ~W_table["is_representative"].to_numpy(),
            -W_table["W"].to_numpy(),
        )
    )
    selected_positions = [int(i) for i in order if bool(W_table["selected"].iloc[int(i)])]
    selected_features = [feature_names[i] for i in selected_positions]
    selected_indices = [int(cache.valid_cols[i]) for i in selected_positions]

    metadata = dict(rep_result.selector_metadata)
    metadata.update(
        {
            "n_features": int(p_valid),
            "feature_groups": True,
            "n_feature_groups": int(reps_sorted.shape[0]),
            "group_mode": "cluster_representative",
            "discovery_unit": "cluster",
            "q_calibration_unit": "cluster_representative",
            "representative_fdr_control": rep_result.selector_metadata.get(
                "fdr_control", "none"
            ),
            "representative_per_draw_fdr_control": rep_result.selector_metadata.get(
                "per_draw_fdr_control", "none"
            ),
            "group_fdr_control": "none",
            "group_per_draw_fdr_control": "none",
            "feature_level_fdr_control": "none",
            "fdr_control": "none",
            "per_draw_fdr_control": "none",
            "aggregation": (
                "cluster_expansion"
                if int(rep_result.selector_metadata.get("n_draws", 1)) == 1
                else "selection_frequency_then_cluster_expansion"
            ),
            "aggregation_fdr_control": "none",
            "aggregation_preserves_per_draw_fdr": False,
            "group_corr_threshold": float(group_corr_threshold),
            "n_representatives": int(reps_sorted.shape[0]),
        }
    )
    # The representative run only saw one column per cluster, so its dropped
    # positions describe the reduced cache.  This result expands back to every
    # valid column, so recompute the provenance against the full cache.
    metadata.pop("n_features_input", None)
    metadata.pop("dropped_feature_positions", None)
    metadata.pop("dropped_feature_reasons", None)
    metadata.update(_input_width_provenance(cache))
    metadata["n_tested_unit"] = "cluster_representative"
    diagnostics = dict(rep_result.diagnostics_ or {})
    inner_offset0 = list(diagnostics.get("offset_zero_selection_sets") or [])
    cluster_of_orig = {
        int(cache.valid_cols[int(i)]): int(labels[int(i)])
        for i in range(p_valid)
    }
    members_by_cluster: dict[int, list[int]] = {}
    orig_to_valid = {
        int(cache.valid_cols[int(i)]): int(i) for i in range(p_valid)
    }
    for valid_i in range(p_valid):
        members_by_cluster.setdefault(int(labels[int(valid_i)]), []).append(
            int(cache.valid_cols[int(valid_i)])
        )
    expanded_offset0: list[list[int]] = []
    expanded_valid: list[list[int]] = []
    for origs in inner_offset0:
        clusters = {
            cluster_of_orig[int(orig)]
            for orig in origs
            if int(orig) in cluster_of_orig
        }
        expanded: list[int] = []
        for cluster_id in sorted(clusters):
            expanded.extend(members_by_cluster[cluster_id])
        expanded_offset0.append(expanded)
        expanded_valid.append(
            [orig_to_valid[orig] for orig in expanded if orig in orig_to_valid]
        )
    metadata["n_discoveries_offset_0_per_draw"] = [
        len(chosen) for chosen in expanded_offset0
    ]
    metadata["n_discoveries_offset_0"] = _aggregate_offset_zero_discoveries(
        expanded_valid,
        n_draws=int(metadata.get("n_draws", 1)),
        eta=float(metadata.get("eta", 0.5)),
        p_valid=p_valid,
    )
    diagnostics["offset_zero_selection_sets"] = expanded_offset0
    diagnostics.update(
        {
            "cluster_labels": labels.astype(int).tolist(),
            "cluster_representatives_valid_positions": reps.astype(int).tolist(),
            "representative_result": rep_result,
        }
    )
    selection_frequency = None
    if rep_result.selection_frequency is not None:
        selection_frequency = pd.Series(
            W_table["selection_frequency"].to_numpy(),
            index=feature_names,
            name="selection_frequency",
        )
    return KnockoffSelectionResult(
        selected_features=selected_features,
        selected_indices=selected_indices,
        selector_metadata=metadata,
        W=W_table,
        threshold=rep_result.threshold,
        selection_frequency=selection_frequency,
        diagnostics_=diagnostics,
    )


@_single_threaded_ridge_knockoffs
def select_fdr(
    X=None,
    y=None,
    *,
    q: float = 0.1,
    statistic: str = "relevance",
    n_draws: int = 1,
    eta: float = 0.5,
    offset: int = 1,
    s_method: str = "equi",
    min_eig: float = 1e-3,
    screen_pairs: int | None = 2000,
    statistic_options: dict | None = None,
    feature_groups: Sequence[Any] | str | None = None,
    group_corr_threshold: float = 0.7,
    sample_weight=None,
    subsample: Any = _SUBSAMPLE_DEFAULT,
    cache: FeatureCache | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    verbose: bool = True,
    include=None,
    exclude=None,
    candidates=None,
    include_provenance=None,
) -> KnockoffSelectionResult:
    """Select features by a q-calibrated Gaussian-copula knockoff filter.

    Answers "which features survive a target false-discovery level?" rather
    than "which ``k`` features are best": it builds or reuses a copula
    ``sift.FeatureCache``, fits and samples second-order Gaussian
    knockoffs, computes a swap-antisymmetric statistic ``W`` per feature, and
    keeps everything at or above the knockoff+ threshold for ``q``.  Use it
    when you want error control instead of a fixed count; use the filter
    selectors when you want a ranking of fixed size.  With defaults it targets
    ``q=0.1`` with the fast marginal ``"relevance"`` statistic, one knockoff
    draw, the knockoff+ offset, equicorrelated decorrelation, a 50,000-row
    subsample seeded at 0, and returns a
    ``KnockoffSelectionResult``.  An empty selection is a valid answer:
    it means nothing survived the threshold.

    ``feature_groups="auto"`` clusters near-collinear features (average-linkage
    on ``1 - |corr|`` cut at ``1 - group_corr_threshold``), runs the knockoff
    filter on one representative (medoid) per cluster, and reports selected
    clusters. This restores power when tightly correlated blocks would
    otherwise force the knockoff decorrelation ``s`` towards zero; the
    discovery unit becomes the cluster, not the individual feature.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features), default None
        Feature matrix from which to build the copula cache.  Exactly one of
        ``X`` and ``cache`` must be given.  Numeric only; encode categoricals
        beforehand -- this selector has no ``cat_encoding`` argument, because
        target-derived preprocessing would invalidate the Model-X claim.
    y : Series or ndarray of shape (n_samples,), default None
        Continuous numeric target, required despite the ``None`` default.  It
        must be finite and have exactly as many rows as the matrix the cache
        was built from, before any subsampling.
    q : float, default 0.1
        Target false-discovery rate, validated in ``(0, 1)``.
    statistic : {"relevance", "cefsplus", "lsm", "ridge"}, default "relevance"
        Feature statistic.  ``"relevance"`` is the fast marginal Gaussian-MI
        difference between each feature and its knockoff.  ``"ridge"`` is the
        analytic coefficient difference ``|beta_j| - |beta_j_tilde|`` from
        ``(G + lambda I)^-1 [r; r_tilde]``, deterministic and antisymmetric by
        construction.  ``"lsm"`` is the lasso signed-max from a Gram-form LARS
        path.  ``"cefsplus"`` is a redundancy-aware greedy entry-order
        statistic and is markedly slower -- treat it as a second opinion, not
        a better default.  The names ``"lcd"``, ``"mrmr_diff"``,
        ``"mrmr_quot"``, ``"jmi"``, and ``"jmim"`` are reserved and raise.
    n_draws : int, default 1
        Number of independent knockoff draws.  Values above 1 derandomize by
        selecting features chosen in at least a fraction ``eta`` of draws --
        which drops the FDR claim; see Notes.
    eta : float, default 0.5
        Selection-frequency threshold in ``(0, 1]`` applied when
        ``n_draws > 1``.  Ignored for a single draw.
    offset : {0, 1}, default 1
        ``1`` is the knockoff+ threshold, which adds one to the negative count
        in the estimated FDP.  ``0`` is the less conservative plain knockoff
        threshold, best read as modified-FDR control.
    s_method : {"equi", "mvr", "me"}, default "equi"
        How the knockoff decorrelation vector ``s`` is solved.  ``"equi"`` is
        the cheapest; ``"mvr"`` and ``"me"`` run diagonal coordinate descent
        and can raise power on correlated designs.  A lower reported
        ``s_mean`` does not by itself mean a worse solution -- it is a
        diagnostic, not the objective.
    min_eig : float, default 0.001
        Minimum eigenvalue required of the copula correlation matrix.  When
        the empirical matrix falls short it is mixed with the identity by a
        factor ``gamma`` and a ``UserWarning`` is emitted; ``gamma`` and
        ``lambda_min`` are reported in metadata.
    screen_pairs : int or None, default 2000
        Cap on the number of original/knockoff pairs handed to statistics that
        need screening (``"cefsplus"``, ``"lsm"``, ``"ridge"``).  ``None``
        disables the cap.  The default ``"relevance"`` statistic needs no
        screening and ignores this.
    statistic_options : dict or None, default None
        Extra options for the chosen statistic; unknown keys are rejected by
        name.  ``"cefsplus"`` accepts ``path_depth`` (an explicit hard cap on
        the greedy path) and ``min_gain_ratio`` (early stop once the best gain
        is small relative to the first; disabled by default).  ``"lsm"``
        accepts ``max_steps``.  ``"ridge"`` accepts ``ridge_lambda``
        (default ``0.5``).  ``"relevance"`` accepts none.
    feature_groups : sequence, "auto", or None, default None
        Group the discovery unit.  A sequence of labels, of length equal to
        either the valid cache columns or the original columns, thresholds a
        heuristic signed-maximum group statistic and expands selected groups
        back to their members.  ``"auto"`` instead clusters near-collinear
        features and runs the filter on one medoid per cluster.  Both modes
        report ``fdr_control="none"``; see Notes.
    group_corr_threshold : float, default 0.7
        Absolute-correlation cut for ``feature_groups="auto"``, in ``(0, 1)``.
        Clustering is average linkage on ``1 - |corr|`` cut at
        ``1 - group_corr_threshold`` and costs ``O(p**2)`` time and memory, so
        pre-screen very wide matrices.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Finite, non-negative row weights used when building the cache from
        ``X``.  Rejected with a prebuilt ``cache``, whose weights are already
        fixed.  Weighting yields an importance-weighted approximation, not an
        exact weighted Model-X guarantee.
    subsample : int or None, default 50000
        Row cap for cache construction from ``X``.  ``None`` uses every
        positive-weight row.  Rejected with a prebuilt ``cache``.
    cache : FeatureCache or None, default None
        Prebuilt copula cache from ``sift.build_cache``, used instead of
        ``X``.  Build it with ``compute_Rxx=True`` so the correlation matrix
        the knockoff model needs is already there.
    random_state : int, default 0
        Seed for cache subsampling when building from ``X``, and for the
        knockoff draws.  Unlike ``sample_weight`` and ``subsample``, this
        stays meaningful with a prebuilt cache because it seeds a fresh draw.
    n_jobs : int, default 1
        Worker count for cache construction and for statistics that fit
        sklearn models.  Building a cache from ``X`` rejects ``0``; the
        analytic statistics never spawn workers, so it does not reach them.
    verbose : bool, default True
        Log the threshold, selected count, and ``s_mean`` at INFO on the
        ``"sift"`` logger.
    include : sequence of names or positions, optional
        Conditioning set. These features are not tested; they are prepended
        to ``selected_features`` in caller order. Any of ``include``,
        ``exclude``, or ``candidates`` requires ``include_provenance``.
    exclude : sequence of names or positions, optional
        Features removed from the tested discovery universe. Requires
        ``include_provenance``.
    candidates : sequence of names or positions, optional
        Hard allow-list for the tested discovery universe. ``include`` may
        sit outside it. Overlap with ``exclude`` is rejected. Requires
        ``include_provenance``.
    include_provenance : {"prespecified", "sample_split", "data_derived"} or None
        Required when ``include``, ``exclude``, or ``candidates`` is
        provided. FDR-compatible wording is allowed only for
        ``prespecified`` and ``sample_split``. A ``data_derived``
        conditioning set is labeled exploratory and reports
        ``fdr_control="none"``.

    Returns
    -------
    KnockoffSelectionResult
        Result carrying ``selected_features`` ordered by descending ``W``,
        their ``selected_indices`` in ``X``, the ``W`` table with one row per
        valid cache feature, the ``threshold`` (``None`` when derandomized),
        ``selection_frequency`` (``None`` for a single draw), the validity
        ``selector_metadata``, and per-draw ``diagnostics_``.

    Raises
    ------
    ValueError
        If neither or both of ``X`` and ``cache`` are given; if ``y`` is
        ``None``, non-finite, or has the wrong row count; if ``q`` or ``eta``
        is outside its interval, ``n_draws`` is not a positive integer,
        ``offset`` is not 0 or 1, or ``screen_pairs`` is not a positive
        integer or ``None``; if ``statistic`` is unknown or reserved; if
        ``statistic_options`` carries keys the statistic does not accept; if
        ``feature_groups`` is a string other than ``"auto"``, has the wrong
        length, or contains missing or unhashable labels; if
        ``group_corr_threshold`` is outside ``(0, 1)``; if ``sample_weight``
        or ``subsample`` accompanies a prebuilt ``cache``; if the cache fails
        its structural or provenance checks or carries duplicate feature
        names; or if no feature retains positive weighted variance.

    Warns
    -----
    UserWarning
        When the median knockoff decorrelation ``s`` falls below ``0.05``:
        most knockoffs are then nearly identical to their originals and ``W``
        is near zero, which usually means near-collinear columns.  Consider
        ``s_method="mvr"`` or ``"me"``, ``feature_groups``, or pruning
        duplicates.  Also when the copula correlation matrix had to be shrunk
        toward the identity to reach ``min_eig``; when a ``"cefsplus"`` run
        with an explicit ``path_depth`` saturates that cap; when knockoff+
        (``offset=1``) has one or more completed draws with effective
        ``m·q < 1`` (per-draw; an infeasible draw does not imply an empty
        aggregate); and, once per process, when ``y`` holds integer labels with
        3-20 distinct values, which look multiclass -- run one-vs-rest targets
        instead.

    See Also
    --------
    sample_knockoffs : Draw one knockoff copy for custom statistics.
    KnockoffSelectionResult : The returned container.
    sift.build_cache : Build the cache this selector can reuse.
    sift.select_cefsplus : Fixed-size selection instead of error control.

    Notes
    -----
    Validity is *plug-in*: metadata reports
    ``validity_model="gaussian_copula_plugin"`` and, for a single ungrouped
    draw, ``fdr_control="approximate_plugin"``.  Exact finite-sample Model-X
    FDR would require the sampled Gaussian-copula model to be the true feature
    distribution and the statistic to be valid under swaps; with estimated
    correlations, shrinkage, or weights, read the output as an approximate
    practical knockoff filter.  The claim is dropped outright -- metadata
    reports ``fdr_control="none"``, ``q_scope="per_draw"``, and
    ``aggregation_fdr_control="none"`` -- whenever ``n_draws > 1`` or any
    ``feature_groups`` mode is used, because ``q`` then calibrates each draw
    or each representative rather than the reported set.  Note also that
    knockoff+ is discrete: at level ``q`` with ``offset=1`` at least
    ``ceil(1 / q)`` tested units must clear the threshold before the estimated
    FDP can reach ``q``, so a problem with few true signals can legitimately
    return nothing at a small ``q``.  Metadata reports ``min_feasible_q`` as
    ``1/min(m)`` over **completed** draws: a necessary count-based lower bound,
    not a sufficient condition for discovery.  ``n_tested`` is that minimum
    post-screening count; ``n_tested_per_draw`` is the truthful per-draw
    record.  ``n_eligible`` is the pre-screen discovery-unit count.
    ``tested_state="not_run"`` means no knockoff draw or pair-screen ran
    (for example a constant target), so ``n_tested`` is 0 and per-draw lists
    are empty.  ``n_discoveries_offset_0`` counts **reported discovery
    features** from the same ``W`` at ``offset=0`` (group/cluster members
    expanded); it is not the number of tested groups.  ``m`` is
    post-screening and post-conditioning -- group-level when grouped,
    representative-level under ``feature_groups="auto"`` -- not raw input
    width.  Included conditioning features are not discoveries.  When
    ``offset=1`` and a completed draw has ``m·q < 1``, a ``UserWarning`` is
    emitted for that draw scope; it does not claim the aggregated selection
    is empty.  The selection, statistic, and FDR labels are otherwise
    unchanged.  Rerunning with new seeds until something is selected
    destroys the guarantee.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_fdr
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(400, 12)),
    ...                  columns=[f"f{i}" for i in range(12)])
    >>> beta = np.zeros(12)
    >>> beta[:6] = 2.0
    >>> y = X.to_numpy() @ beta + 0.5 * rng.normal(size=400)
    >>> result = select_fdr(X, y, q=0.2, random_state=0, verbose=False)
    >>> sorted(result.selected_features)
    ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    >>> result.selector_metadata["fdr_control"]
    'approximate_plugin'
    >>> bool(np.isfinite(result.threshold)), result.selection_frequency is None
    (True, True)
    """

    q_float = _validate_probability(q, "q")
    n_draws_int = _validate_positive_int(n_draws, "n_draws")
    eta_float = _validate_probability(eta, "eta", upper_inclusive=True)
    offset_int = _validate_offset(offset)
    screen_pairs_int = _validate_screen_pairs(screen_pairs)
    stat_spec = _get_statistic(statistic)
    options = dict(statistic_options or {})
    unknown_options = set(options) - stat_spec.allowed_options
    if unknown_options:
        allowed = sorted(stat_spec.allowed_options) or ["<none>"]
        raise ValueError(
            f"Unknown statistic_options for {stat_spec.name!r}: {sorted(unknown_options)}; "
            f"allowed: {allowed}"
        )

    resolved_cache = _resolve_cache(
        X,
        cache=cache,
        sample_weight=sample_weight,
        subsample=subsample,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    _reject_duplicate_feature_names(resolved_cache)
    cache_names = list(resolved_cache.feature_names) if resolved_cache.feature_names is not None else [
        f"x{i}"
        for i in range(
            int(np.max(resolved_cache.valid_cols)) + 1 if len(resolved_cache.valid_cols) else 0
        )
    ]
    named = named_feature_space(
        resolved_cache.feature_names,
        synthetic=bool(getattr(resolved_cache, "feature_names_are_synthetic", False))
        or resolved_cache.feature_names is None,
    )
    resolved_sets = resolve_conditioning(
        include,
        exclude,
        candidates,
        feature_names=cache_names,
        named=named,
        k=1,
    )
    provenance = require_include_provenance(
        include_provenance,
        conditioning_active=bool(resolved_sets is not None and resolved_sets.active),
    )
    if (
        resolved_sets is not None
        and resolved_sets.active
        and isinstance(feature_groups, str)
        and feature_groups == "auto"
    ):
        raise ValueError(
            "feature_groups='auto' cannot be combined with include/exclude/candidates; "
            "cluster the discovery universe yourself or omit conditioning"
        )
    p_valid = resolved_cache.Z.shape[1]
    if resolved_cache.valid_cols.shape[0] != p_valid:
        raise ValueError("cache.valid_cols length must match cache.Z columns")
    if isinstance(feature_groups, str):
        if feature_groups != "auto":
            raise ValueError("feature_groups must be None, 'auto', or a sequence of group labels")
        return _select_fdr_cluster_representatives(
            resolved_cache,
            y,
            q=q_float,
            statistic=stat_spec.name,
            n_draws=n_draws_int,
            eta=eta_float,
            offset=offset_int,
            s_method=s_method,
            min_eig=min_eig,
            screen_pairs=screen_pairs_int,
            statistic_options=options,
            group_corr_threshold=group_corr_threshold,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
    feature_names = _feature_names_for_valid_cols(resolved_cache)
    group_info = _resolve_feature_groups(resolved_cache, feature_groups)
    if group_info is None:
        group_labels: list[Any] | None = None
        group_codes = None
    else:
        group_labels, group_codes = group_info

    if y is None:
        raise ValueError("y is required")
    # Preserve target ordering before the rank transform. Large offsets can
    # collapse a genuinely varying float64 target to one float32 value.
    y_arr = to_numpy(y, dtype=np.float64).ravel()
    if y_arr.shape[0] != resolved_cache.n_rows_original:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but cache was built from "
            f"{resolved_cache.n_rows_original} rows"
        )
    if not np.isfinite(y_arr).all():
        raise ValueError("y must contain only finite values")
    _warn_if_integer_multiclass_target(y)

    w = np.asarray(resolved_cache.sample_weight, dtype=np.float64).ravel()
    if w.shape[0] != resolved_cache.Z.shape[0]:
        raise ValueError("cache.sample_weight length must match cache.Z rows")
    if not np.isfinite(w).all() or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("cache.sample_weight must be finite, non-negative, and sum to > 0")

    variances = _weighted_variance(resolved_cache.Z, w)
    zero_var = variances <= 1e-12
    n_zero_variance = int(zero_var.sum())
    valid_cols_arr = np.asarray(resolved_cache.valid_cols, dtype=np.int64)
    include_valid = np.empty(0, dtype=np.int64)
    if resolved_sets is not None and resolved_sets.include:
        include_orig = {int(i) for i in resolved_sets.include}
        include_valid = np.array(
            [i for i, orig in enumerate(valid_cols_arr) if int(orig) in include_orig],
            dtype=np.int64,
        )
        if include_valid.size != len(resolved_sets.include):
            raise ValueError(
                "include features are not present in the cache valid columns "
                "(dropped as constant/non-finite or never cached)"
            )
        if np.any(zero_var[include_valid]):
            raise ValueError(
                "include features have no usable variation for knockoff conditioning"
            )
    if resolved_sets is not None and resolved_sets.active:
        discovery_original = set(int(i) for i in resolved_sets.discovery)
        discovery_valid = np.array(
            [i for i, orig in enumerate(valid_cols_arr) if int(orig) in discovery_original],
            dtype=np.int64,
        )
        discovery_mask = np.zeros(p_valid, dtype=bool)
        if discovery_valid.size:
            discovery_mask[discovery_valid] = True
        active = (~zero_var) & discovery_mask
    else:
        active = ~zero_var
    if not bool(active.any()):
        raise ValueError("No active non-constant features remain for knockoffs")

    ys = y_arr[resolved_cache.row_idx]
    zy = np.asarray(weighted_rank_gauss_1d(ys, w), dtype=np.float64)
    zy_var = float(_weighted_variance(zy[:, None], w)[0])
    Z_work = np.asarray(resolved_cache.Z, dtype=np.float64)
    residual_zero_valid = np.empty(0, dtype=np.int64)
    if include_valid.size:
        pre_resid = np.flatnonzero(active).astype(np.int64)
        Z_work, zy, active = _residualize_discovery_given_include(
            Z_work,
            zy,
            w,
            include_valid=include_valid,
            discovery_valid=pre_resid,
        )
        residual_zero_valid = np.array(
            [int(i) for i in pre_resid if not bool(active[int(i)])],
            dtype=np.int64,
        )
        if not bool(active.any()):
            raise ValueError(
                "No active residual features remain after conditioning on include"
            )
        R_active = weighted_correlation_matrix(
            np.ascontiguousarray(Z_work[:, active], dtype=np.float64),
            w,
            backend="blas",
        )
    else:
        R_active = _build_active_rxx(resolved_cache, active, verbose=verbose)
    model = fit_gaussian_knockoffs(R_active, s_method=s_method, min_eig=min_eig)
    s_median = float(np.median(model.s))
    n_low_s = int(np.sum(model.s < _LOW_POWER_S))
    if s_median < _LOW_POWER_S:
        warnings.warn(
            "Knockoff construction has very little power: the median knockoff "
            f"decorrelation s is {s_median:.3g} (s_method={s_method!r}, "
            f"lambda_min={model.lambda_min:.3g}), so most knockoffs are nearly "
            "identical to their originals and W statistics will be close to 0. "
            "This usually means near-collinear features (duplicates, "
            "interactions, one-hot blocks). Consider s_method='mvr' or 'me', "
            "feature_groups for collinear clusters, or pruning near-duplicate "
            "columns before select_fdr.",
            UserWarning,
            stacklevel=3,
        )
    active_positions = np.flatnonzero(active).astype(np.int64)
    path_depth_requested = options.get("path_depth")
    if stat_spec.name == "cefsplus":
        m_pairs = active_positions.shape[0] if screen_pairs_int is None else min(active_positions.shape[0], screen_pairs_int)
        if path_depth_requested is None:
            path_depth_effective = _default_cefsplus_path_depth(
                q_float,
                offset_int,
                int(m_pairs),
            )
        else:
            path_depth_effective = _validate_path_depth(
                path_depth_requested,
                int(m_pairs),
            )
        options["path_depth"] = path_depth_effective
        options["_adaptive_path_depth"] = path_depth_requested is None
        options["_q"] = q_float
        options["_offset"] = offset_int
    else:
        path_depth_effective = None
    Z_active = (
        np.asarray(Z_work, dtype=np.float32)
        if bool(active.all())
        else np.ascontiguousarray(Z_work[:, active], dtype=np.float32)
    )

    relevance = np.zeros(p_valid, dtype=np.float64)
    r_orig_active = None
    if zy_var > 1e-12:
        r_orig = np.asarray(weighted_corr_with_vector(Z_active, zy, w), dtype=np.float64)
        r_orig_active = r_orig
        relevance[active_positions] = np.asarray(gaussian_mi_from_corr(r_orig), dtype=np.float64)

    manual_group_heuristic = group_labels is not None
    per_draw_fdr_control = "none" if manual_group_heuristic else "approximate_plugin"
    metadata: dict[str, Any] = {
        "selector": "knockoff_fdr",
        "n_features": int(p_valid),
        "q": q_float,
        "offset": offset_int,
        "statistic": stat_spec.name,
        "s_method": s_method,
        "n_draws": n_draws_int,
        "eta": eta_float,
        "screen_pairs": screen_pairs_int,
        "path_depth_requested": path_depth_requested,
        "path_depth": path_depth_effective,
        "path_depth_initial": path_depth_effective,
        "path_depth_adaptive": stat_spec.name == "cefsplus" and path_depth_requested is None,
        "gamma": float(model.gamma),
        "lambda_min": float(model.lambda_min),
        "s_mean": float(np.mean(model.s)),
        "s_median": s_median,
        "n_low_power_features": n_low_s,
        "random_state": int(random_state),
        "n_rows_used": int(resolved_cache.Z.shape[0]),
        "fdr_control": (
            "approximate_plugin"
            if n_draws_int == 1 and not manual_group_heuristic
            else "none"
        ),
        "per_draw_fdr_control": per_draw_fdr_control,
        "q_scope": "per_draw",
        "aggregation": "single_draw" if n_draws_int == 1 else "selection_frequency",
        "aggregation_threshold": None if n_draws_int == 1 else eta_float,
        "aggregation_fdr_control": "not_applicable" if n_draws_int == 1 else "none",
        "aggregation_preserves_per_draw_fdr": n_draws_int == 1 and not manual_group_heuristic,
        "validity_model": "gaussian_copula_plugin",
        "weighted_model": bool(np.ptp(w) > 1e-9),
        "n_zero_weight_variance_features": n_zero_variance,
        "feature_groups": group_labels is not None,
        "n_feature_groups": None if group_labels is None else len(group_labels),
        "group_mode": None if group_labels is None else "signed_max_heuristic",
        "group_fdr_control": None if group_labels is None else "none",
    }
    if provenance == "data_derived":
        metadata["fdr_control"] = "none"
        metadata["per_draw_fdr_control"] = "none"
        metadata["aggregation_preserves_per_draw_fdr"] = False
        metadata["exploratory"] = True
        metadata["include_provenance"] = provenance
    elif provenance is not None:
        metadata["include_provenance"] = provenance
        metadata["exploratory"] = False
        if provenance not in FDR_COMPATIBLE_PROVENANCE:
            metadata["aggregation_preserves_per_draw_fdr"] = False
    if resolved_sets is not None and (resolved_sets.exclude or resolved_sets.candidates is not None):
        metadata["discovery_universe_constrained"] = True
        if provenance not in FDR_COMPATIBLE_PROVENANCE:
            metadata["fdr_control"] = "none"
            metadata["exploratory"] = True
            metadata["aggregation_preserves_per_draw_fdr"] = False
    metadata.update(
        _input_width_provenance(
            resolved_cache,
            inactive_valid_positions=np.flatnonzero(zero_var),
            extra_valid_drops=[
                (int(position), "zero_residual_variance")
                for position in residual_zero_valid
            ],
        )
    )

    if zy_var <= 1e-12:
        return _all_zero_result(
            cache=resolved_cache,
            feature_names=feature_names,
            group_labels=group_labels,
            group_codes=group_codes,
            relevance=relevance,
            metadata=metadata,
            diagnostic_reason="zero_target_variance",
            resolved_sets=resolved_sets,
            cache_names=cache_names,
            include_valid=include_valid,
            provenance=provenance,
            discovery_valid_mask=active,
        )

    seed_sequence = np.random.SeedSequence(random_state)
    child_sequences = seed_sequence.spawn(n_draws_int)
    W_draws = np.zeros((n_draws_int, p_valid), dtype=np.float64)
    thresholds: list[float] = []
    group_W_draws: list[list[float]] = []
    group_thresholds: list[float] = []
    selection_sets_valid: list[list[int]] = []
    offset_zero_sets_valid: list[list[int]] = []
    tested_id_sets: list[tuple[int, ...]] = []
    n_tested_per_draw: list[int] = []
    mean_active = gaussian_knockoff_mean(Z_active, model) if n_draws_int > 1 else None
    active_group_codes = None if group_codes is None else group_codes[active_positions]
    fixed_kept = None
    fixed_G = None
    if stat_spec.needs_screening and (
        screen_pairs_int is None or screen_pairs_int >= active_positions.size
    ):
        fixed_kept = np.arange(active_positions.size, dtype=np.int64)
        fixed_G = _build_augmented_correlation(model, fixed_kept)

    for draw_idx, child in enumerate(child_sequences):
        rng = np.random.default_rng(child)
        Zt_active = sample_gaussian_knockoffs(Z_active, model, rng, mean=mean_active)
        context = _build_context(
            Z_active,
            Zt_active,
            zy,
            w,
            model,
            screen_pairs=screen_pairs_int if stat_spec.needs_screening else None,
            options=options,
            n_jobs=n_jobs,
            rng=rng,
            build_augmented=stat_spec.needs_screening,
            statistic_name=stat_spec.name,
            r=r_orig_active,
            fixed_kept=fixed_kept,
            fixed_G=fixed_G,
        )
        W_active = np.asarray(stat_spec.fn(context), dtype=np.float64).ravel()
        if W_active.shape[0] != active_positions.shape[0]:
            raise RuntimeError("Knockoff statistic returned the wrong number of W values")
        if not np.isfinite(W_active).all():
            raise RuntimeError("Knockoff statistic returned non-finite W values")
        W_draws[draw_idx, active_positions] = W_active
        tested_ids = _tested_unit_ids(
            context.kept,
            active_positions=active_positions,
            active_group_codes=active_group_codes,
        )
        tested_id_sets.append(tested_ids)
        n_tested_per_draw.append(len(tested_ids))
        if active_group_codes is None or group_labels is None:
            threshold = knockoff_threshold(W_active, q_float, offset=offset_int)
            if np.isfinite(threshold):
                selected_active = np.where(W_active >= threshold)[0]
            else:
                selected_active = np.empty(0, dtype=np.int64)
            selected_active_offset0 = _offset_zero_local_selection(W_active, q_float)
        else:
            group_W = _group_knockoff_statistics(W_active, active_group_codes, len(group_labels))
            threshold = knockoff_threshold(group_W, q_float, offset=offset_int)
            group_W_draws.append(group_W.astype(float).tolist())
            group_thresholds.append(threshold)
            if np.isfinite(threshold):
                selected_group_codes = np.flatnonzero(group_W >= threshold)
                selected_active = np.where(np.isin(active_group_codes, selected_group_codes))[0]
            else:
                selected_active = np.empty(0, dtype=np.int64)
            selected_groups_offset0 = _offset_zero_local_selection(group_W, q_float)
            if selected_groups_offset0.size:
                selected_active_offset0 = np.where(
                    np.isin(active_group_codes, selected_groups_offset0)
                )[0]
            else:
                selected_active_offset0 = np.empty(0, dtype=np.int64)
        thresholds.append(threshold)
        selected_valid = active_positions[selected_active]
        selection_sets_valid.append(selected_valid.astype(int).tolist())
        offset_zero_sets_valid.append(
            active_positions[selected_active_offset0].astype(int).tolist()
        )

    if stat_spec.name == "cefsplus":
        metadata["path_depth"] = int(
            options.get("_path_depth_used", path_depth_effective)
        )
        metadata["path_depth_saturated"] = bool(
            options.get("_path_depth_saturated", False)
        )
        if metadata["path_depth_saturated"]:
            warnings.warn(
                "The CEFS+ knockoff discovery set reached the effective "
                f"path_depth={metadata['path_depth']}; increasing path_depth may "
                "yield additional discoveries.",
                UserWarning,
                stacklevel=3,
            )

    n_tested_unit = (
        "group"
        if group_labels is not None
        else "feature"
    )
    eligible_ids = _tested_unit_ids(
        np.arange(active_positions.size, dtype=np.int64),
        active_positions=active_positions,
        active_group_codes=active_group_codes,
    )
    n_disc_0_per_draw = [len(chosen) for chosen in offset_zero_sets_valid]
    n_disc_0 = _aggregate_offset_zero_discoveries(
        offset_zero_sets_valid,
        n_draws=n_draws_int,
        eta=eta_float,
        p_valid=p_valid,
    )
    metadata.update(
        _feasibility_metadata(
            n_tested_per_draw=n_tested_per_draw,
            n_tested_unit=n_tested_unit,
            tested_id_sets=tested_id_sets,
            n_discoveries_offset_0_per_draw=n_disc_0_per_draw,
            n_discoveries_offset_0=n_disc_0,
            n_eligible=len(eligible_ids),
            tested_state="post_screening",
            q=q_float,
            offset=offset_int,
        )
    )
    _warn_knockoff_plus_infeasible(metadata)

    mean_W = W_draws.mean(axis=0)
    if n_draws_int == 1:
        selection_frequency_arr = np.full(p_valid, np.nan, dtype=np.float64)
        threshold_out: float | None = thresholds[0]
        selected_mask = np.zeros(p_valid, dtype=bool)
        if selection_sets_valid:
            selected_mask[np.asarray(selection_sets_valid[0], dtype=np.int64)] = True
        selection_frequency = None
    else:
        selected_by_draw = np.zeros((n_draws_int, p_valid), dtype=np.float64)
        for draw_idx, selected_valid in enumerate(selection_sets_valid):
            selected_by_draw[draw_idx, np.asarray(selected_valid, dtype=np.int64)] = 1.0
        selection_frequency_arr = selected_by_draw.mean(axis=0)
        threshold_out = None
        selected_mask = selection_frequency_arr >= eta_float
        selection_frequency = pd.Series(selection_frequency_arr, index=feature_names, name="selection_frequency")

    selected_valid_positions = np.where(selected_mask)[0]
    selected_order = selected_valid_positions[
        np.lexsort((selected_valid_positions, -mean_W[selected_valid_positions]))
    ]
    discovered_original = [int(resolved_cache.valid_cols[int(i)]) for i in selected_order]
    selected_features = [feature_names[int(i)] for i in selected_order]
    selected_indices = list(discovered_original)
    if resolved_sets is not None and resolved_sets.include:
        selected_features, selected_indices = compose_selected(
            cache_names,
            resolved_sets.include,
            discovered_original,
        )
        include_valid_set = set(int(i) for i in include_valid)
        selected_mask = selected_mask.copy()
        for valid_i in include_valid_set:
            selected_mask[int(valid_i)] = True
            mean_W[int(valid_i)] = 0.0
            if n_draws_int == 1:
                selection_frequency_arr[int(valid_i)] = np.nan
            else:
                selection_frequency_arr[int(valid_i)] = 1.0
                if selection_frequency is not None:
                    selection_frequency.iloc[int(valid_i)] = 1.0

    W_cols = {
        "feature": feature_names,
        "selected_index": resolved_cache.valid_cols.astype(np.int64),
        "W": mean_W,
        "selected": selected_mask,
        "selection_frequency": selection_frequency_arr,
        "relevance": relevance,
    }
    if resolved_sets is not None and resolved_sets.active:
        role = np.array(["ineligible"] * p_valid, dtype=object)
        role[active_positions] = "discovery"
        if include_valid.size:
            role[include_valid] = "include"
        W_cols["role"] = role
    W_cols["selector"] = "knockoff_fdr"
    W_table = pd.DataFrame(W_cols)
    if group_codes is not None and group_labels is not None:
        W_table["feature_group"] = [group_labels[int(code)] for code in group_codes]
    for draw_idx in range(n_draws_int):
        W_table[f"W_draw_{draw_idx}"] = W_draws[draw_idx]

    diagnostics = {
        "thresholds": thresholds,
        "selection_sets": [
            [int(resolved_cache.valid_cols[int(i)]) for i in selected_valid]
            for selected_valid in selection_sets_valid
        ],
        "offset_zero_selection_sets": [
            [int(resolved_cache.valid_cols[int(i)]) for i in selected_valid]
            for selected_valid in offset_zero_sets_valid
        ],
        "active_valid_positions": active_positions.astype(int).tolist(),
    }
    cond_record = conditioning_record(
        resolved_sets,
        feature_names=cache_names,
        discovered_idx=[
            int(resolved_cache.valid_cols[int(i)])
            for i in selected_order
        ],
        include_provenance=provenance,
        discovery_universe=None if resolved_sets is None else resolved_sets.discovery,
    )
    if cond_record is not None:
        diagnostics["conditioning"] = cond_record
        metadata["conditioning"] = cond_record
    if group_codes is not None and group_labels is not None:
        diagnostics["feature_groups"] = group_labels
        diagnostics["group_W_draws"] = group_W_draws
        diagnostics["group_thresholds"] = group_thresholds

    if verbose:
        threshold_text = "derandomized" if threshold_out is None else f"threshold={threshold_out:.6g}"
        threshold_name = "knockoff+" if offset_int == 1 else "knockoff"
        logger.info(
            f"{threshold_name} q={q_float:.3g}: selected {len(selected_features)} features "
            f"({threshold_text}, s_mean={metadata['s_mean']:.3g})"
        )

    return KnockoffSelectionResult(
        selected_features=selected_features,
        selected_indices=selected_indices,
        selector_metadata=metadata,
        W=W_table,
        threshold=threshold_out,
        selection_frequency=selection_frequency,
        diagnostics_=diagnostics,
    )


__all__ = [
    "KnockoffSelectionResult",
    "cluster_feature_groups",
    "KnockoffStatContext",
    "KnockoffStatSpec",
    "VALID_KNOCKOFF_STATISTICS",
    "knockoff_threshold",
    "sample_knockoffs",
    "select_fdr",
]
