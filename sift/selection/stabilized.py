"""Generic resampling meta-selector with per-feature selection frequencies."""

from __future__ import annotations

import inspect
from collections.abc import Hashable, Iterable, Mapping, Set
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.metadata_routing import UNUSED
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from sift._logging import logger
from sift._metadata import drop_fitted_metadata_columns, resolve_row_metadata
from sift._preprocess import reject_datetime_like_features
from sift._selector_compat import (
    check_fitted_column_identity,
    feature_names_array,
    inverse_selected_matrix,
    ordered_indices,
    reject_sparse,
    validate_fit_matrix,
    validate_output_order,
)
from sift.sampling.stability import _block_bootstrap_indices


_RESAMPLE_MODES = frozenset({"half", "bootstrap", "blocks"})
_AGGREGATIONS = frozenset({None, "frequency", "evalues"})
_BLOCK_METHODS = frozenset({"moving", "circular", "stationary"})
_EVALUE_DEFAULT_RESAMPLE = "half"
_EVALUE_DEFAULT_THRESHOLD = 0.6
_EVALUE_DEFAULT_RANDOM_STATE = 0
_EVALUE_DEFAULT_BLOCK_SIZE = "auto"
_EVALUE_DEFAULT_BLOCK_METHOD = "moving"


def _coerce_feature_names(feature_names, *, argument: str = "feature_names") -> list[Hashable]:
    invalid_container = isinstance(
        feature_names,
        (str, bytes, bytearray, memoryview, Mapping, Set),
    )
    ndim = getattr(feature_names, "ndim", None)
    if invalid_container or (ndim is not None and ndim != 1):
        raise ValueError(
            f"{argument} must be an ordered, one-dimensional iterable of names; "
            "pass a list, tuple, pandas Index, or one-dimensional NumPy array, "
            "not a string, bytes-like object, mapping, set, scalar, or matrix."
        )
    try:
        names = list(feature_names)
    except TypeError as exc:
        raise ValueError(
            f"{argument} must be an ordered, one-dimensional iterable of names."
        ) from exc
    for name in names:
        try:
            hash(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{argument} entries must be hashable column labels.") from exc
    return names


def _feature_names_index(feature_names) -> pd.Index:
    return pd.Index(
        feature_names_array(feature_names),
        dtype=object,
        tupleize_cols=False,
    )


def _exact_column_positions(columns, required_names) -> np.ndarray:
    available = _feature_names_index(columns)
    required = _feature_names_index(required_names)
    return available.get_indexer(required)


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return out


def _strict_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


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


def _base_accepts_sample_weight(selector: Any) -> bool:
    """Return whether sample_weight can be forwarded without inventing consumption.

    Explicit ``fit(..., sample_weight=)`` is accepted. sklearn meta-selectors
    such as ``RFE`` and ``SelectFromModel`` take ``**fit_params`` and forward
    ``sample_weight`` only when their nested ``estimator.fit`` names it.
    A bare ``**kwargs`` sink is not treated as consumption.
    """
    if _explicit_kwarg(selector.fit, "sample_weight"):
        return True
    if not _has_var_keyword(selector.fit):
        return False
    nested = getattr(selector, "estimator", None)
    return nested is not None and _explicit_kwarg(nested.fit, "sample_weight")


def _base_requires_y(selector: Any) -> bool:
    try:
        parameters = inspect.signature(selector.fit).parameters
    except (TypeError, ValueError):
        return True
    y_param = parameters.get("y")
    if y_param is None:
        return False
    return y_param.default is inspect.Parameter.empty


def _is_sift_fixed_k_filter(selector: Any) -> bool:
    k = getattr(selector, "k", None)
    if k is None or k == "auto":
        return False
    if getattr(selector, "within", None) is not None:
        return False
    return hasattr(selector, "_selector_fn")


def _is_knockoff_selector(selector: Any) -> bool:
    cls = type(selector)
    return cls.__name__ == "KnockoffSelector" and str(cls.__module__).startswith("sift.")


def _base_needs_named_frame(selector: Any) -> bool:
    """SIFT wrappers read raw names from X columns, not a generic ndarray."""
    if _is_knockoff_selector(selector):
        return True
    return hasattr(selector, "_selector_fn") and str(
        type(selector).__module__
    ).startswith("sift.")


def _n_rows_used_from_fitted(fitted: Any) -> int | None:
    meta = getattr(fitted, "selector_metadata_", None)
    if not isinstance(meta, Mapping):
        result = getattr(fitted, "result_", None)
        meta = None if result is None else getattr(result, "selector_metadata", None)
    if not isinstance(meta, Mapping) or "n_rows_used" not in meta:
        return None
    value = meta["n_rows_used"]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _base_consumes_row_context(selector: Any, name: str) -> bool:
    if not _explicit_kwarg(selector.fit, name):
        return False
    if _is_knockoff_selector(selector):
        return False
    if _is_sift_fixed_k_filter(selector):
        return False
    return True


def _row_take(values: Any, idx: np.ndarray) -> Any:
    if values is None:
        return None
    positions = np.asarray(idx, dtype=np.int64)
    if isinstance(values, pd.DataFrame):
        return values.iloc[positions]
    if isinstance(values, pd.Series):
        return values.iloc[positions]
    return np.asarray(values)[positions]


def _raw_evalue_frequencies(result: Any, n_features: int) -> np.ndarray:
    """Reindex knockoff draw frequencies onto the raw column universe.

    ``KnockoffSelectionResult.selection_frequency`` covers cache-valid
    columns only. Dropped constants stay 0. Length mismatches raise instead
    of substituting the final 0/1 e-BH support.
    """
    if result is None:
        raise ValueError("KnockoffSelector e-value fit did not produce a result")
    table = getattr(result, "W", None)
    columns = getattr(table, "columns", ())
    if table is None or "selected_index" not in columns:
        raise ValueError(
            "aggregation='evalues' requires KnockoffSelector W['selected_index'] "
            "to reindex draw frequencies onto the raw universe"
        )
    raw_pos = np.asarray(table["selected_index"], dtype=np.int64).reshape(-1)
    if "selection_frequency" in columns:
        values = np.asarray(table["selection_frequency"], dtype=np.float64).reshape(-1)
    else:
        freq = getattr(result, "selection_frequency", None)
        if freq is None:
            raise ValueError(
                "aggregation='evalues' did not report per-feature selection frequencies"
            )
        values = np.asarray(freq, dtype=np.float64).reshape(-1)
    if raw_pos.size != values.size:
        raise ValueError(
            "KnockoffSelector selection_frequency length does not match "
            "W selected_index; refusing to substitute the final 0/1 support"
        )
    if np.any((raw_pos < 0) | (raw_pos >= int(n_features))):
        raise ValueError(
            "KnockoffSelector selected_index is outside the raw feature universe"
        )
    out = np.zeros(int(n_features), dtype=np.float64)
    finite = np.isfinite(values)
    out[raw_pos[finite]] = values[finite]
    return out


def _support_mask_from_fitted(fitted: Any, n_features: int) -> np.ndarray:
    if hasattr(fitted, "get_support"):
        mask = np.asarray(fitted.get_support(), dtype=bool).reshape(-1)
        if mask.size != n_features:
            raise ValueError(
                "base selector get_support() length "
                f"{mask.size} does not match the {n_features}-column raw universe; "
                "encoded dummy width is not Stabilized's feature identity"
            )
        return mask.astype(bool, copy=False)
    indices = getattr(fitted, "selected_indices_", None)
    if indices is None:
        raise TypeError(
            "selector must expose sklearn get_support() or SIFT selected_indices_ "
            "in the raw feature universe"
        )
    mask = np.zeros(n_features, dtype=bool)
    pos = np.asarray(indices, dtype=np.int64).reshape(-1)
    if pos.size and np.any((pos < 0) | (pos >= n_features)):
        raise ValueError(
            "base selector selected_indices_ is outside the raw feature universe"
        )
    mask[pos] = True
    return mask


def _spawn_resample_rngs(random_state: int, n_resamples: int) -> list[np.random.Generator]:
    sequence = np.random.SeedSequence(int(random_state))
    return [np.random.default_rng(child) for child in sequence.spawn(int(n_resamples))]


class Stabilized(SelectorMixin, BaseEstimator):
    """Resample any cloneable selector and keep frequent raw features.

    ``Stabilized`` is an additive meta-selector: each resample clones ``selector``
    and fits it on a row subset. Features whose selection frequency meets
    ``threshold`` are kept. This is a robustness diagnostic, not Meinshausen
    and Bühlmann error control and not knockoff FDR.

    ``resample="half"`` draws without replacement. ``resample="bootstrap"``
    draws with replacement. ``resample="blocks"`` uses the existing
    group/time block-bootstrap helper, which draws blocks with replacement.
    The similarly named ``sift.sampling.stability._bootstrap_indices`` helper
    is a subsample *without* replacement and is not used for
    ``resample="bootstrap"``.

    Random numbers use ``numpy.random.SeedSequence(random_state)`` to spawn
    one child sequence per resample; each child seeds
    ``numpy.random.default_rng`` for that resample's row index draw. The
    default ``random_state=0`` is a plain integer. This class does not change
    ``StabilitySelector`` or ``KnockoffSelector`` defaults.

    ``aggregation="evalues"`` is valid only for a ``KnockoffSelector`` base.
    It reuses that class's native full-data ``n_draws`` /
    ``aggregation="evalues"`` path. It does not average e-values across
    bootstrap datasets and does not claim FDR for frequency voting.

    Parameters
    ----------
    selector : estimator
        Cloneable sklearn-style selector or SIFT wrapper. Must expose
        ``get_support()`` or raw ``selected_indices_`` after ``fit``. Pass an
        instance, not a class. Row arrays are not constructor arguments.
    n_resamples : int, default=30
        Number of row resamples in frequency mode. In ``aggregation="evalues"``
        this becomes ``KnockoffSelector.n_draws`` when that base has
        ``n_draws <= 1``; otherwise it must equal the base ``n_draws``.
    resample : {"half", "bootstrap", "blocks"}, default="half"
        Row scheme in frequency mode. ``"half"`` is without replacement,
        ``"bootstrap"`` is with replacement, and ``"blocks"`` requires both
        ``groups`` and ``time`` at fit time. Ignored only when it remains
        ``"half"`` under ``aggregation="evalues"``; other values raise.
    threshold : float, default=0.6
        Minimum selection frequency in frequency mode, in ``[0, 1]``. Must
        remain ``0.6`` under ``aggregation="evalues"``.
    sample_frac : float or None, default=None
        Fraction of rows drawn per resample. ``None`` resolves to ``0.5`` for
        ``"half"`` and ``1.0`` for ``"bootstrap"`` and ``"blocks"``. Must stay
        ``None`` under ``aggregation="evalues"``.
    aggregation : {None, "frequency", "evalues"}, default=None
        ``None`` and ``"frequency"`` count resample support. ``"evalues"``
        is the KnockoffSelector full-data e-value path described above.
    random_state : int, default=0
        Seed for ``SeedSequence`` resampling. Unused under
        ``aggregation="evalues"`` and must remain ``0`` unless it already
        equals the KnockoffSelector seed.
    store_proxies : bool, default=False
        If True, retain the rank-Gaussian candidate-by-selected correlation
        block and, in frequency mode, per-resample boolean indicators for
        ``SelectionView`` proxy/cluster reports. Storage is capped; X is not
        retained. Default False.
    output_order : {"legacy", "original"}, default="legacy"
        Transform order. ``"legacy"`` is descending frequency then original
        index in frequency mode, or the base discovery order for e-values.
        ``"original"`` is ascending fitted position.
    n_jobs : int, default=1
        Resamples run serially. Values other than ``1`` raise.
    block_size : int or {"auto"}, default="auto"
        Block length for ``resample="blocks"``. Overrides raise in other modes.
    block_method : {"moving", "circular", "stationary"}, default="moving"
        Block-bootstrap flavor for ``resample="blocks"``.
    verbose : bool, default=True
        Emit the resample scheme and selected count at INFO on the ``sift``
        logger.

    Attributes
    ----------
    selection_frequencies_ : ndarray of shape (n_features,)
        Float64 fraction of resamples that selected each raw feature, or the
        KnockoffSelector frequencies copied in e-value mode. Never-selected
        features are true zeros; the vector is not padded past the fitted
        width.
    selected_features_ : list
        Selected raw feature names, not encoded dummy names and not
        positional ``StabilitySelector.selected_features_``.
    selected_indices_ : ndarray of shape (n_selected,)
        Their positions in the fitted raw matrix.
    n_features_selected_ : int
        Number of selected raw features.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        One-dimensional object array of fitted raw feature names.
    n_features_in_ : int
        Number of raw candidate features seen during ``fit``.

    See Also
    --------
    StabilitySelector : Lasso/logistic stability selection, unchanged in 0.9.
    KnockoffSelector : Native e-value aggregation reused by ``aggregation="evalues"``.
    sift.as_result : Normalize a fitted selector into a ``SelectionView``.

    Notes
    -----
    Fit is ``fit(X, y=None, sample_weight=..., groups=..., time=...)``.
    Constructor options never store row arrays. ``y`` may be omitted when the
    base selector accepts a missing target. Supervised bases raise their own
    missing-``y`` errors. DataFrame ``feature_names`` must
    match the full column order and count; they do not subset or reorder.
    Explicit ndarray names become the raw identity. Compatible SIFT bases
    receive them as a named frame or a ``feature_names`` argument; generic
    ndarray selectors keep the original array container. ``groups``/``time`` are consumed by
    ``resample="blocks"`` (with ``min_oob=0``, because this wrapper never uses
    OOB rows) or forwarded to bases that accept them. Fixed-k SIFT filters and
    ``KnockoffSelector`` reject unused row context. Sample weights are sliced
    onto the same resample indices when the base, or a nested sklearn
    ``estimator`` behind ``**fit_params``, names ``sample_weight``.

    Frequency aggregation has no FDR claim. E-value mode keeps the existing
    ``approximate_plugin`` / exploratory qualifications, antisymmetry,
    conditioning, screening, and group limitations of ``KnockoffSelector``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.feature_selection import SelectKBest, f_regression
    >>> from sift import Stabilized
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(80, 6))
    >>> y = X[:, 0] + 0.8 * X[:, 1] + 0.1 * rng.normal(size=80)
    >>> selector = Stabilized(
    ...     SelectKBest(f_regression, k=2),
    ...     n_resamples=8,
    ...     resample="half",
    ...     threshold=0.6,
    ...     random_state=0,
    ...     verbose=False,
    ... )
    >>> selector.fit(X, y).selected_features_
    ['x0', 'x1']
    """

    __metadata_request__fit = {"feature_names": UNUSED}

    def __init__(
        self,
        selector,
        n_resamples: int = 30,
        resample: str = "half",
        threshold: float = 0.6,
        sample_frac: float | None = None,
        aggregation: str | None = None,
        random_state: int = 0,
        store_proxies: bool = False,
        output_order: str = "legacy",
        n_jobs: int = 1,
        block_size: int | str = "auto",
        block_method: str = "moving",
        verbose: bool = True,
    ):
        self.selector = selector
        self.n_resamples = n_resamples
        self.resample = resample
        self.threshold = threshold
        self.sample_frac = sample_frac
        self.aggregation = aggregation
        self.random_state = random_state
        self.store_proxies = store_proxies
        self.output_order = output_order
        self.n_jobs = n_jobs
        self.block_size = block_size
        self.block_method = block_method
        self.verbose = verbose

    def fit(
        self,
        X,
        y=None,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        feature_names: Iterable[Hashable] | None = None,
    ):
        """Fit resampled clones of ``selector`` and threshold frequencies.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training features. Sparse input is rejected.
        y : array-like of shape (n_samples,) or None, default None
            Target values. Required when the base selector requires ``y``;
            omitted for unsupervised bases such as ``VarianceThreshold``.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. Sliced with the resample indices when the base
            ``fit`` accepts them; unused weights raise.
        groups : array-like, optional
            Group labels supplied at fit/routing time. Required with ``time``
            for ``resample="blocks"``. Forwarded only to bases that consume
            row context.
        time : array-like, optional
            Time values supplied at fit/routing time, under the same rules.
        feature_names : ordered iterable of hashable labels, optional
            Raw feature names. For a DataFrame they must list every column
            exactly once in column order. For an ndarray they must have one
            name per column and become the wrapper's raw identity. Compatible
            SIFT bases receive a named frame; generic ndarray selectors keep
            the original array. Strings, mappings, and sets are rejected.

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
                feature_names=feature_names,
            )
        except Exception:
            self._clear_fit_state()
            raise

    def _fit_impl(self, X, y, *, sample_weight, groups, time, feature_names):
        self._validate_estimator()
        self._validate_runtime_params()
        metadata = resolve_row_metadata(X, groups=groups, time=time)
        X = metadata.X
        groups = metadata.groups
        time = metadata.time
        reject_datetime_like_features(X)
        validate_fit_matrix(X)
        validate_output_order(self.output_order)
        self._row_metadata_columns_ = metadata.extracted_columns
        was_dataframe = isinstance(X, pd.DataFrame)
        self._fit_input_kind_ = "dataframe" if was_dataframe else "positional"

        if was_dataframe:
            column_index = _feature_names_index(X.columns)
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
        n = int(X.shape[0])
        p = int(X.shape[1])
        if feature_names is not None:
            names = _coerce_feature_names(feature_names)
            if len(names) != p:
                raise ValueError("feature_names must contain one name per column of X")
            if was_dataframe:
                positions = _exact_column_positions(X.columns, names)
                missing = [names[i] for i in np.flatnonzero(positions < 0)]
                if missing:
                    raise ValueError(
                        "feature_names must reference existing DataFrame columns; "
                        f"missing: {missing[:5]}"
                    )
                if not np.array_equal(positions, np.arange(p)):
                    raise ValueError(
                        "feature_names must list every DataFrame column exactly once, "
                        "in column order"
                    )
            elif _base_needs_named_frame(self.selector):
                X = pd.DataFrame(np.asarray(X), columns=names)
        name_index = _feature_names_index(names)
        if name_index.duplicated().any():
            raise ValueError("feature_names must be unique")

        if y is None:
            y_arr = None
        else:
            y_arr = y if isinstance(y, pd.Series) else np.asarray(y)
            if np.ndim(y_arr) == 0:
                raise ValueError("y must be a one-dimensional array of length n_samples")
            if int(np.asarray(y_arr).shape[0]) != n:
                raise ValueError(
                    f"y has {int(np.asarray(y_arr).shape[0])} rows but X has {n}"
                )
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if sample_weight.size != n:
                raise ValueError(
                    f"sample_weight has {sample_weight.size} rows but X has {n}"
                )
        if groups is not None:
            groups = np.asarray(groups).reshape(-1)
            if groups.size != n:
                raise ValueError(f"groups has {groups.size} rows but X has {n}")
        if time is not None:
            time = np.asarray(time).reshape(-1)
            if time.size != n:
                raise ValueError(f"time has {time.size} rows but X has {n}")

        self._fit_feature_names_generated_ = feature_names is None and not was_dataframe
        self.feature_names_in_ = feature_names_array(names)
        self.n_features_in_ = p
        self._n_rows_original_ = n

        aggregation = self._resolved_aggregation()
        used_groups, used_time, used_weight = self._validate_row_context(
            sample_weight, groups, time, aggregation
        )
        self._fit_used_sample_weight_ = used_weight
        self._fit_used_groups_ = used_groups
        self._fit_used_time_ = used_time

        if aggregation == "evalues":
            self._fit_evalues(X, y_arr, sample_weight=sample_weight, names=names)
        else:
            self._fit_frequency(
                X,
                y_arr,
                sample_weight=sample_weight,
                groups=groups,
                time=time,
                names=names,
            )

        if self.store_proxies:
            self._store_proxy_payload(X, sample_weight)
        self._fit_configured_options_ = self._snapshot_fit_configuration()
        return self

    def _snapshot_fit_configuration(self) -> dict[str, Any]:
        from sift.selection.reproducibility import describe_estimator, snapshot_selector_kwargs

        return snapshot_selector_kwargs(
            {
                "base_selector": describe_estimator(self.selector),
                "n_resamples": int(self.n_resamples),
                "resample": self.resample,
                "threshold": float(self.threshold),
                "sample_frac": self.sample_frac,
                "aggregation": self.aggregation,
                "random_state": int(self.random_state),
                "store_proxies": bool(self.store_proxies),
                "output_order": self.output_order,
                "n_jobs": self.n_jobs,
                "block_size": self.block_size,
                "block_method": self.block_method,
            }
        )

    def _validate_estimator(self) -> None:
        selector = self.selector
        if selector is None or isinstance(selector, type):
            raise TypeError("selector must be a cloneable estimator instance, not a class")
        if not hasattr(selector, "fit"):
            raise TypeError("selector must be a cloneable sklearn-style estimator with fit")
        try:
            clone(selector)
        except Exception as exc:
            raise TypeError("selector must be cloneable with sklearn.base.clone") from exc

    def _validate_runtime_params(self) -> None:
        _strict_int(self.n_resamples, name="n_resamples", minimum=1)
        if self.resample not in _RESAMPLE_MODES:
            raise ValueError("resample must be 'half', 'bootstrap', or 'blocks'")
        threshold = _strict_float(self.threshold, name="threshold")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a finite value in [0, 1]")
        if self.sample_frac is not None:
            frac = _strict_float(self.sample_frac, name="sample_frac")
            if not 0.0 < frac <= 1.0:
                raise ValueError("sample_frac must be finite and in (0, 1]")
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError("aggregation must be None, 'frequency', or 'evalues'")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise ValueError("random_state must be an integer")
        if self.n_jobs not in (1, None):
            raise ValueError("Stabilized runs resamples serially; n_jobs must be 1")
        if self.block_method not in _BLOCK_METHODS:
            raise ValueError("block_method must be 'moving', 'circular', or 'stationary'")
        if self.block_size != "auto":
            _strict_int(self.block_size, name="block_size", minimum=1)
        if self.resample != "blocks":
            if self.block_size != "auto":
                raise ValueError("block_size is used only when resample='blocks'")
            if self.block_method != "moving":
                raise ValueError("block_method is used only when resample='blocks'")
        validate_output_order(self.output_order)

    def _resolved_aggregation(self) -> str:
        return "evalues" if self.aggregation == "evalues" else "frequency"

    def _resolved_sample_frac(self) -> float:
        if self.sample_frac is not None:
            return float(self.sample_frac)
        return 0.5 if self.resample == "half" else 1.0

    def _validate_row_context(self, sample_weight, groups, time, aggregation: str):
        if aggregation == "evalues":
            self._validate_evalue_overrides()
            if groups is not None:
                raise ValueError(
                    "aggregation='evalues' fits KnockoffSelector on the full sample "
                    "and does not consume groups"
                )
            if time is not None:
                raise ValueError(
                    "aggregation='evalues' fits KnockoffSelector on the full sample "
                    "and does not consume time"
                )
            if sample_weight is not None and not _base_accepts_sample_weight(self.selector):
                raise ValueError(
                    "sample_weight was supplied but the KnockoffSelector base does "
                    "not accept it in this configuration"
                )
            return False, False, sample_weight is not None

        used_groups = False
        used_time = False
        if self.resample == "blocks":
            if groups is None or time is None:
                raise ValueError(
                    "resample='blocks' requires both groups and time at fit time"
                )
            used_groups = True
            used_time = True
        else:
            if groups is not None:
                if not _base_consumes_row_context(self.selector, "groups"):
                    raise ValueError(
                        "groups was supplied but not used: resample "
                        f"{self.resample!r} does not consume groups, and the base "
                        "selector rejects unused row context"
                    )
                used_groups = True
            if time is not None:
                if not _base_consumes_row_context(self.selector, "time"):
                    raise ValueError(
                        "time was supplied but not used: resample "
                        f"{self.resample!r} does not consume time, and the base "
                        "selector rejects unused row context"
                    )
                used_time = True
        if sample_weight is not None and not _base_accepts_sample_weight(self.selector):
            raise ValueError(
                "sample_weight was supplied but the base selector does not accept it"
            )
        return used_groups, used_time, sample_weight is not None

    def _validate_evalue_overrides(self) -> None:
        if not _is_knockoff_selector(self.selector):
            raise TypeError(
                "aggregation='evalues' is only available for KnockoffSelector bases; "
                "it reuses native full-data n_draws/aggregation='evalues' and does "
                "not average e-values across bootstrap datasets"
            )
        if self.resample != _EVALUE_DEFAULT_RESAMPLE:
            raise ValueError(
                "aggregation='evalues' does not resample rows; resample must remain "
                f"{_EVALUE_DEFAULT_RESAMPLE!r}"
            )
        if float(self.threshold) != _EVALUE_DEFAULT_THRESHOLD:
            raise ValueError(
                "aggregation='evalues' does not apply a frequency threshold; "
                f"threshold must remain {_EVALUE_DEFAULT_THRESHOLD}"
            )
        if self.sample_frac is not None:
            raise ValueError(
                "aggregation='evalues' does not resample rows; sample_frac must "
                "remain None"
            )
        if self.block_size != _EVALUE_DEFAULT_BLOCK_SIZE:
            raise ValueError(
                "aggregation='evalues' does not resample rows; block_size must "
                "remain 'auto'"
            )
        if self.block_method != _EVALUE_DEFAULT_BLOCK_METHOD:
            raise ValueError(
                "aggregation='evalues' does not resample rows; block_method must "
                "remain 'moving'"
            )
        base_seed = getattr(self.selector, "random_state", _EVALUE_DEFAULT_RANDOM_STATE)
        if (
            int(self.random_state) != _EVALUE_DEFAULT_RANDOM_STATE
            and int(self.random_state) != int(base_seed)
        ):
            raise ValueError(
                "aggregation='evalues' does not use Stabilized.random_state for "
                "row resampling; leave it at 0 so KnockoffSelector.random_state "
                "seeds the draws"
            )

    def _base_fit_kwargs(
        self,
        selector,
        sample_weight,
        groups,
        time,
        *,
        feature_names=None,
        named_frame: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if sample_weight is not None and _base_accepts_sample_weight(selector):
            kwargs["sample_weight"] = sample_weight
        if groups is not None and _base_consumes_row_context(selector, "groups"):
            kwargs["groups"] = groups
        if time is not None and _base_consumes_row_context(selector, "time"):
            kwargs["time"] = time
        if (
            feature_names is not None
            and not named_frame
            and _explicit_kwarg(selector.fit, "feature_names")
        ):
            kwargs["feature_names"] = list(feature_names)
        return kwargs

    def _fit_one(
        self,
        selector,
        X,
        y,
        sample_weight,
        groups,
        time,
        *,
        feature_names=None,
    ):
        kwargs = self._base_fit_kwargs(
            selector,
            sample_weight,
            groups,
            time,
            feature_names=feature_names,
            named_frame=isinstance(X, pd.DataFrame),
        )
        with threadpool_limits(limits=1):
            if y is None:
                selector.fit(X, **kwargs)
            else:
                selector.fit(X, y, **kwargs)
        return _support_mask_from_fitted(selector, self.n_features_in_)

    def _fit_evalues(self, X, y, *, sample_weight, names) -> None:
        base = clone(self.selector)
        n_draws = int(getattr(base, "n_draws", 1))
        n_resamples = int(self.n_resamples)
        if n_draws <= 1:
            base.n_draws = n_resamples
        elif n_draws != n_resamples:
            raise ValueError(
                "aggregation='evalues' reuses KnockoffSelector.n_draws on the full "
                f"sample; n_resamples={n_resamples} does not match n_draws={n_draws}"
            )
        current = getattr(base, "aggregation", None)
        if current not in {None, "evalues"}:
            raise ValueError(
                "aggregation='evalues' requires the KnockoffSelector aggregation "
                "to be None or 'evalues'"
            )
        base.aggregation = "evalues"
        if self.verbose:
            logger.info(
                "Stabilized e-value mode: full-data KnockoffSelector "
                f"n_draws={int(base.n_draws)}, no row resampling"
            )
        mask = self._fit_one(
            base, X, y, sample_weight, None, None, feature_names=names
        )
        result = getattr(base, "result_", None)
        frequencies = _raw_evalue_frequencies(result, int(self.n_features_in_))
        selected_indices = np.asarray(
            getattr(base, "selected_indices_", np.flatnonzero(mask)),
            dtype=np.int64,
        ).reshape(-1)
        selected_features = [names[int(i)] for i in selected_indices]
        knockoff_meta = {}
        if result is not None:
            knockoff_meta = dict(getattr(result, "selector_metadata", {}) or {})
        n_rows_used = knockoff_meta.get("n_rows_used")
        self._n_rows_used_ = None if n_rows_used is None else int(n_rows_used)
        self._actual_random_state_ = knockoff_meta.get(
            "random_state", getattr(base, "random_state", None)
        )
        self._rng_mechanism_ = "KnockoffSelector.random_state"
        self._finalize_selection(
            frequencies,
            names,
            selected_indices=selected_indices,
            selected_features=selected_features,
            resample_selections=None,
            n_completed=int(getattr(base, "n_draws", self.n_resamples)),
            mode="evalues",
            extra_metadata=knockoff_meta,
        )

    def _draw_indices(self, rng: np.random.Generator, n: int, groups, time) -> np.ndarray:
        frac = self._resolved_sample_frac()
        if self.resample == "half":
            size = max(1, min(n, int(n * frac)))
            return rng.choice(n, size=size, replace=False).astype(np.int64, copy=False)
        if self.resample == "bootstrap":
            size = max(1, int(round(n * frac)))
            return rng.choice(n, size=size, replace=True).astype(np.int64, copy=False)
        train_idx, _ = next(
            _block_bootstrap_indices(
                n=n,
                n_bootstrap=1,
                groups=groups,
                time=time,
                block_size=self.block_size,
                block_method=self.block_method,
                random_state=rng,
                sample_frac=frac,
                min_oob=0,
            )
        )
        return np.asarray(train_idx, dtype=np.int64)

    def _fit_frequency(self, X, y, *, sample_weight, groups, time, names) -> None:
        n = int(X.shape[0])
        p = int(self.n_features_in_)
        n_resamples = int(self.n_resamples)
        rngs = _spawn_resample_rngs(int(self.random_state), n_resamples)
        counts = np.zeros(p, dtype=np.int64)
        resample_selections = None
        if self.store_proxies:
            from sift.selection.proxies import MAX_RESAMPLE_SELECTION_BYTES

            storage_bytes = int(n_resamples) * int(p) * np.dtype(bool).itemsize
            if storage_bytes > MAX_RESAMPLE_SELECTION_BYTES:
                limit_mib = MAX_RESAMPLE_SELECTION_BYTES / 1024**2
                requested_mib = storage_bytes / 1024**2
                raise ValueError(
                    "store_proxies=True would retain "
                    f"{requested_mib:.2f} MiB of resample selection indicators, "
                    f"exceeding the {limit_mib:.0f} MiB limit"
                )
            resample_selections = np.zeros((n_resamples, p), dtype=bool)
        if self.verbose:
            logger.info(
                f"Stabilized({type(self.selector).__name__}): {n_resamples} "
                f"{self.resample} resamples, threshold={float(self.threshold)}"
            )
        completed = 0
        row_counts: list[int] = []
        unique_counts: list[int] = []
        base_used: list[int] = []
        missing_base_used = False
        for i, rng in enumerate(rngs):
            idx = self._draw_indices(rng, n, groups, time)
            row_counts.append(int(np.asarray(idx).size))
            unique_counts.append(int(np.unique(idx).size))
            X_i = _row_take(X, idx)
            y_i = _row_take(y, idx)
            w_i = _row_take(sample_weight, idx)
            g_i = _row_take(groups, idx) if self._fit_used_groups_ else None
            t_i = _row_take(time, idx) if self._fit_used_time_ else None
            if not _base_consumes_row_context(self.selector, "groups"):
                g_i = None
            if not _base_consumes_row_context(self.selector, "time"):
                t_i = None
            fitted = clone(self.selector)
            mask = self._fit_one(
                fitted, X_i, y_i, w_i, g_i, t_i, feature_names=names
            )
            used = _n_rows_used_from_fitted(fitted)
            if used is None:
                missing_base_used = True
            else:
                base_used.append(used)
            counts += mask.astype(np.int64, copy=False)
            if resample_selections is not None:
                resample_selections[i] = mask
            completed += 1
        self._resample_row_counts_ = np.asarray(row_counts, dtype=np.int64)
        self._resample_unique_counts_ = np.asarray(unique_counts, dtype=np.int64)
        if (
            not missing_base_used
            and base_used
            and len(set(base_used)) == 1
        ):
            self._n_rows_used_ = int(base_used[0])
        else:
            self._n_rows_used_ = None
        self._actual_random_state_ = int(self.random_state)
        self._rng_mechanism_ = "numpy.random.SeedSequence.spawn"
        frequencies = (counts / float(completed)).astype(np.float64, copy=False)
        self._finalize_selection(
            frequencies,
            names,
            selected_indices=None,
            selected_features=None,
            resample_selections=resample_selections,
            n_completed=completed,
            mode="frequency",
            extra_metadata=None,
        )

    def _finalize_selection(
        self,
        frequencies: np.ndarray,
        names: list[Hashable],
        *,
        selected_indices,
        selected_features,
        resample_selections,
        n_completed: int,
        mode: str,
        extra_metadata: dict[str, Any] | None,
    ) -> None:
        frequencies = np.asarray(frequencies, dtype=np.float64).reshape(-1)
        if frequencies.size != self.n_features_in_:
            raise ValueError("selection frequencies must have one value per raw feature")
        self.selection_frequencies_ = frequencies
        if selected_indices is None:
            mask = frequencies >= float(self.threshold)
            selected = np.flatnonzero(mask)
            order = np.argsort(-frequencies[selected], kind="mergesort")
            selected_indices = selected[order]
            selected_features = [names[int(i)] for i in selected_indices]
        else:
            selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
            if selected_features is None:
                selected_features = [names[int(i)] for i in selected_indices]
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        self.selected_features_ = list(selected_features)
        self.n_features_selected_ = int(self.selected_indices_.size)
        self._n_completed_resamples_ = int(n_completed)
        self._aggregation_mode_ = mode
        self._extra_result_metadata_ = dict(extra_metadata or {})
        if resample_selections is not None:
            self._resample_selections_ = np.ascontiguousarray(resample_selections)
        if self.verbose:
            logger.info(
                f"Selected {self.n_features_selected_} / {self.n_features_in_} features"
            )

    def _store_proxy_payload(self, X, sample_weight) -> None:
        from sift._impute import mean_impute
        from sift._preprocess import ensure_weights
        from sift.estimators.copula import weighted_rank_gauss_2d
        from sift.selection.proxies import (
            _check_storage_size,
            weighted_correlation_columns,
        )

        if isinstance(X, pd.DataFrame):
            try:
                Xs = np.asarray(X, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "store_proxies=True requires a numeric feature matrix"
                ) from exc
        else:
            Xs = np.asarray(X, dtype=np.float64)
        n = int(Xs.shape[0])
        weights = ensure_weights(sample_weight, n, normalize=True)
        positive = np.flatnonzero(weights > 0.0)
        if positive.size == 0:
            raise ValueError("store_proxies=True requires at least one positive-weight row")
        Xs = mean_impute(Xs[positive], copy=True)
        ws = np.asarray(weights[positive], dtype=np.float64).copy()
        p = int(Xs.shape[1])
        varying = np.array(
            [float(np.ptp(Xs[:, j])) > 0.0 for j in range(p)],
            dtype=bool,
        )
        selected = [int(i) for i in np.asarray(self.selected_indices_)]
        varying_raw = np.flatnonzero(varying).astype(np.int64)
        candidate_raw = sorted(set(varying_raw.tolist()) | set(selected))
        _check_storage_size(len(candidate_raw), len(selected))
        varying_selected = [pos for pos in selected if bool(varying[pos])]
        if varying_raw.size and varying_selected:
            Z = weighted_rank_gauss_2d(Xs[:, varying_raw], ws)
            raw_to_local = {int(raw): local for local, raw in enumerate(varying_raw.tolist())}
            local_selected = [raw_to_local[pos] for pos in varying_selected]
            varying_block = weighted_correlation_columns(Z, ws, local_selected)
        else:
            raw_to_local = {}
            varying_block = np.zeros((int(varying_raw.size), 0), dtype=np.float64)
        varsel_to_col = {pos: j for j, pos in enumerate(varying_selected)}
        block = np.zeros((len(candidate_raw), len(selected)), dtype=np.float64)
        for row, cand in enumerate(candidate_raw):
            for col, sel in enumerate(selected):
                if cand == sel:
                    block[row, col] = 1.0
                elif cand in raw_to_local and sel in varsel_to_col:
                    block[row, col] = varying_block[raw_to_local[cand], varsel_to_col[sel]]
        self._proxy_correlations = pd.DataFrame(
            block.astype(np.float32, copy=False),
            index=pd.Index(candidate_raw, name="selected_index"),
            columns=pd.Index(selected, name="selected_index"),
        )

    def _clear_fit_state(self) -> None:
        for attr in (
            "_aggregation_mode_",
            "_extra_result_metadata_",
            "_fit_configured_options_",
            "_fit_feature_names_generated_",
            "_fit_input_kind_",
            "_fit_used_groups_",
            "_fit_used_sample_weight_",
            "_fit_used_time_",
            "_actual_random_state_",
            "_n_completed_resamples_",
            "_n_rows_original_",
            "_n_rows_used_",
            "_proxy_correlations",
            "_resample_row_counts_",
            "_resample_unique_counts_",
            "_rng_mechanism_",
            "_resample_selections_",
            "_row_metadata_columns_",
            "feature_names_in_",
            "n_features_in_",
            "n_features_selected_",
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
                    "This Stabilized selector was fitted on a positional array with "
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
                f"X has {X_arr.shape[1]} features, but Stabilized was fitted with "
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

    @property
    def result_view_(self):
        """Return a normalized, non-cached view of this fitted selector."""
        from sift.selection.view import as_result

        return as_result(self)

    def get_metadata_routing(self):
        routing = super().get_metadata_routing()
        if self.resample != "blocks" and not (
            _base_consumes_row_context(self.selector, "groups")
            or _base_consumes_row_context(self.selector, "time")
        ):
            unsupported = [
                name
                for name in ("groups", "time")
                if routing.fit.requests.get(name) not in (None, False)
            ]
            if unsupported:
                raise ValueError(
                    "Stabilized can request groups/time metadata only when "
                    "resample='blocks' or the base selector consumes row context"
                )
        if (
            not _base_accepts_sample_weight(self.selector)
            and routing.fit.requests.get("sample_weight") not in (None, False)
        ):
            raise ValueError(
                "Stabilized can request sample_weight metadata only when the base "
                "selector consumes sample weights"
            )
        return routing

    def _base_tag_facts(self) -> tuple[bool, bool, bool]:
        allow_nan = False
        requires_y = _base_requires_y(self.selector)
        non_deterministic = False
        getter = getattr(self.selector, "_get_tags", None)
        if callable(getter):
            try:
                tags = getter()
            except Exception:
                tags = None
            if isinstance(tags, dict):
                allow_nan = bool(tags.get("allow_nan", allow_nan))
                if "requires_y" in tags:
                    requires_y = bool(tags["requires_y"])
                non_deterministic = bool(tags.get("non_deterministic", False))
        more = getattr(self.selector, "_more_tags", None)
        if callable(more):
            try:
                tags = more()
            except Exception:
                tags = None
            if isinstance(tags, dict):
                if "allow_nan" in tags:
                    allow_nan = bool(tags["allow_nan"])
                if "requires_y" in tags:
                    requires_y = bool(tags["requires_y"])
                if tags.get("non_deterministic"):
                    non_deterministic = True
        sklearn_tags = getattr(self.selector, "__sklearn_tags__", None)
        if callable(sklearn_tags):
            try:
                tags = sklearn_tags()
            except Exception:
                tags = None
            if tags is not None:
                input_tags = getattr(tags, "input_tags", None)
                target_tags = getattr(tags, "target_tags", None)
                if input_tags is not None:
                    allow_nan = bool(getattr(input_tags, "allow_nan", allow_nan))
                if target_tags is not None:
                    requires_y = bool(getattr(target_tags, "required", requires_y))
                non_deterministic = bool(
                    getattr(tags, "non_deterministic", non_deterministic)
                )
        return allow_nan, requires_y, non_deterministic

    def _more_tags(self):
        parent = getattr(super(), "_more_tags", None)
        tags = {} if parent is None else dict(parent())
        allow_nan, requires_y, non_deterministic = self._base_tag_facts()
        tags["allow_nan"] = allow_nan
        tags["requires_y"] = requires_y
        if non_deterministic:
            tags["non_deterministic"] = True
        return tags

    def __sklearn_tags__(self):
        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:
            return self._more_tags()
        tags = parent()
        allow_nan, requires_y, non_deterministic = self._base_tag_facts()
        input_tags = getattr(tags, "input_tags", None)
        target_tags = getattr(tags, "target_tags", None)
        if input_tags is not None:
            input_tags.allow_nan = allow_nan
        if target_tags is not None:
            target_tags.required = requires_y
        if non_deterministic:
            tags.non_deterministic = True
        return tags


__all__ = ["Stabilized"]
