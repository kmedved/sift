"""Spec-driven function-style filter selector APIs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sift._metadata import resolve_row_metadata
from sift._progress import ProgressCallback
from sift._preprocess import (
    CatEncoding,
    EstimatorJMI,
    EstimatorMRMR,
    Formula,
    RelevanceMethod,
    Task,
    resolve_jmi_estimator,
    validate_target_cv_encoding_flags,
    validate_task,
    validate_k,
)
from sift.estimators.copula import FeatureCache
from sift.selection.auto_k import AutoKConfig, resolve_auto_k_config
from sift.selection.cefsplus_binary_common import (
    prepare_binary_problem,
    validate_binary_options,
)
from sift.selection.filter_payloads import (
    GAUSSIAN_AUTO,
    GAUSSIAN_CHANGEPOINT,
    GAUSSIAN_CHI2,
    GAUSSIAN_CONSENSUS,
    GAUSSIAN_CV,
    GAUSSIAN_ELBOW,
    GAUSSIAN_EVALUATE,
    GAUSSIAN_FORWARD_STOP,
    GAUSSIAN_KNOCKOFF,
    GAUSSIAN_PENALIZED,
    GAUSSIAN_PERM_GAP,
    GAUSSIAN_POSTERIOR,
    GAUSSIAN_STABILITY,
    GAUSSIAN_XFIT_OBJECTIVE,
    SelectionPayload,
    binary_auto_auto_payload,
    binary_auto_changepoint_payload,
    binary_auto_elbow_payload,
    binary_auto_evaluate_payload,
    binary_auto_penalized_payload,
    binary_auto_posterior_payload,
    binary_fixed_payload,
    make_auto_classic,
    make_auto_gaussian,
    make_fixed_classic,
    make_fixed_gaussian,
    make_jmi_classic_path,
    make_mrmr_classic_path,
    mrmr_gaussian_method,
    no_extra,
    selector_gaussian_method,
    standard_extra,
    validate_cefsplus,
    validate_ksg_no_weight,
    validate_standard,
)
from sift.selection.loops import MrmrBackend, resolve_mrmr_backend
from sift.selection.result import (
    _PROXY_CORRELATIONS_ATTR,
    FilterSelectionResult,
    build_selector_metadata,
)
from sift.selection.conditioning import resolve_conditioning
from sift.selection.knockoff_filter import (
    _SUBSAMPLE_DEFAULT,
    _reject_duplicate_feature_names,
    _validate_prebuilt_cache_structure,
)


class _FilterRandomStateDefaultType:
    """Sentinel distinguishing an omitted cache-construction seed."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<random_state default: 0>"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_FilterRandomStateDefaultType, ())


_RANDOM_STATE_DEFAULT = _FilterRandomStateDefaultType()


def _single_threaded_binary_blas(func):
    """Limit native pools across path construction and auto-k refits."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        with threadpool_limits(limits=1):
            return func(*args, **kwargs)

    return wrapped


XInput = Union[pd.DataFrame, np.ndarray]
YInput = Union[pd.Series, np.ndarray]
KInput = Union[int, Literal["auto"]]


@dataclass(frozen=True)
class FilterRequest:
    X: XInput
    y: YInput
    k: KInput
    task: Task
    cache: Optional[FeatureCache] = None
    groups: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    auto_k_config: Optional[AutoKConfig] = None
    sample_weight: np.ndarray | None = None
    return_result: bool = False
    store_proxies: bool = False
    selector_kwargs: dict[str, Any] | None = None
    callback: ProgressCallback | None = None


@dataclass(frozen=True)
class FilterSpec:
    selector: str
    display_name: str
    estimator: str
    fixed_handler: Callable[["FilterContext"], SelectionPayload]
    auto_k_handlers: dict[str, Callable[["FilterContext"], SelectionPayload]]
    metadata_extra: Callable[["FilterContext"], dict[str, Any]]
    validate: Callable[["FilterContext"], None] = lambda _ctx: None


@dataclass(frozen=True)
class FilterContext:
    spec: FilterSpec
    request: FilterRequest
    selector_kwargs: dict[str, Any]
    k: int | Literal["auto"]
    groups: np.ndarray | None
    time: np.ndarray | None
    auto_k_config: AutoKConfig | None
    n_rows: int
    n_features_input: int
    feature_names: list[str]
    estimator: str
    n_jobs: int
    mrmr_backend: str
    rank_backend: str
    conditioning: object | None = None


_COMMON_REQUEST_LOCAL_NAMES = frozenset(
    {
        "X",
        "y",
        "k",
        "task",
        "cache",
        "groups",
        "time",
        "auto_k_config",
        "sample_weight",
        "callback",
        "return_result",
        "store_proxies",
    }
)

CONDITIONING_SELECTOR_KWARGS = (
    "include",
    "exclude",
    "candidates",
)
MRMR_SELECTOR_KWARGS = (
    "relevance",
    "estimator",
    "formula",
    "top_m",
    "cat_features",
    "cat_encoding",
    "target_cv_n_splits",
    "target_cv_smoothing",
    "target_prior",
    "warmup_policy",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "n_jobs",
    "mrmr_backend",
    "verbose",
) + CONDITIONING_SELECTOR_KWARGS
JMI_SELECTOR_KWARGS = (
    "estimator",
    "relevance",
    "top_m",
    "cat_features",
    "cat_encoding",
    "target_cv_n_splits",
    "target_cv_smoothing",
    "target_prior",
    "warmup_policy",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "verbose",
) + CONDITIONING_SELECTOR_KWARGS
CEFSPLUS_SELECTOR_KWARGS = (
    "top_m",
    "corr_prune",
    "cat_features",
    "cat_encoding",
    "target_cv_n_splits",
    "target_cv_smoothing",
    "target_prior",
    "warmup_policy",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "verbose",
) + CONDITIONING_SELECTOR_KWARGS
CEFSPLUS_BINARY_SELECTOR_KWARGS = (
    "loss",
    "top_m",
    "corr_prune",
    "class_weight",
    "ridge",
    "refit_every",
    "cat_features",
    "cat_encoding",
    "target_cv_n_splits",
    "target_cv_smoothing",
    "target_prior",
    "warmup_policy",
    "loo_smoothing",
    "loo_clip_min",
    "loo_clip_max",
    "allow_full_data_target_encoding",
    "subsample",
    "random_state",
    "verbose",
) + CONDITIONING_SELECTOR_KWARGS


def _request_from_public_locals(
    values: dict[str, Any],
    *,
    task: Task,
    selector_names: tuple[str, ...],
) -> FilterRequest:
    metadata = resolve_row_metadata(
        values["X"],
        groups=values.get("groups"),
        time=values.get("time"),
        sample_weight=values.get("sample_weight"),
    )
    validate_target_cv_encoding_flags(
        values.get("cat_encoding", "none"),
        values.get("allow_full_data_target_encoding", False),
    )
    store_proxies = values.get("store_proxies", False)
    if not isinstance(store_proxies, (bool, np.bool_)):
        raise ValueError("store_proxies must be a boolean")
    if store_proxies and not bool(values.get("return_result", False)):
        raise ValueError("store_proxies=True requires return_result=True")
    return FilterRequest(
        metadata.X,
        values["y"],
        values["k"],
        task,
        cache=values.get("cache"),
        groups=metadata.groups,
        time=metadata.time,
        auto_k_config=values.get("auto_k_config"),
        sample_weight=metadata.sample_weight,
        callback=values.get("callback"),
        return_result=bool(values.get("return_result", False)),
        store_proxies=bool(store_proxies),
        selector_kwargs={name: values[name] for name in selector_names},
    )


def select_mrmr(
    X: XInput, y: YInput, k: KInput, *, task: Task,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    relevance: RelevanceMethod = "f", estimator: EstimatorMRMR = "classic",
    formula: Formula = "quotient", top_m: Optional[int] = None,
    cat_features: Optional[list[str]] = None, cat_encoding: CatEncoding = "none",
    target_cv_n_splits: int = 5, target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = _SUBSAMPLE_DEFAULT, random_state: int = _RANDOM_STATE_DEFAULT, n_jobs: int = 1,
    mrmr_backend: MrmrBackend = "auto",
    verbose: bool = True, return_result: bool = False, store_proxies: bool = False,
    include=None, exclude=None, candidates=None,
    callback: ProgressCallback | None = None,
) -> list[str] | FilterSelectionResult:
    """Minimum Redundancy Maximum Relevance feature selection.

    Greedily grows a feature set that trades relevance to the target against
    redundancy with the features already chosen, and returns the selected
    names in selection order.  Reach for it as the fast relevance/redundancy
    baseline on a wide matrix.  Prefer ``sift.select_jmi``,
    ``sift.select_jmim``, or ``sift.select_cefsplus`` when the
    informative features are mutually redundant: those condition on the whole
    selected set rather than penalizing pairwise redundancy, which is the
    regime in which mRMR can promote a low-relevance, low-redundancy noise
    column.  With defaults it runs the classic estimator with F-test
    relevance, screens to ``max(5 * k, 250)`` candidates, subsamples 50,000
    rows with seed 0, logs progress, and returns a plain ``list[str]``.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Feature matrix.  DataFrame labels are preserved in the output; an
        unlabelled array yields the positional names ``"x0", "x1", ...``.
        Object, category, and string columns must either be listed in
        ``cat_features`` with a ``cat_encoding``, or encoded beforehand.
    y : Series or ndarray of shape (n_samples,)
        Target.  Interpreted according to ``task``: numeric for
        ``"regression"``, class labels for ``"classification"``.
    k : int or "auto"
        Number of features to select, treated as an *upper bound*: fewer can
        come back after constant-column filtering, relevance screening, or an
        exhausted candidate path.  ``"auto"`` hands the count to the auto-k
        machinery -- see ``auto_k_config``.
    task : {"regression", "classification"}
        Required keyword.  Chooses the relevance estimators and the target
        validation applied to ``y``.
    cache : FeatureCache or None, default None
        Prebuilt copula cache from ``sift.build_cache``, accepted only
        with ``estimator="gaussian"``.  A named cache requires ``X`` to be the
        DataFrame whose column labels and order built it; a positional cache
        requires the matching ndarray.  Because a cache freezes its rows and
        weights, ``sample_weight``, ``subsample``, and ``random_state`` cannot
        be passed alongside it.
    groups : ndarray of shape (n_samples,), str, or None, default None
        Group labels defining auto-k validation splits, or the name of a
        DataFrame column to use as such (the column is then removed from the
        features).  Only meaningful with ``k="auto"``; a fixed-``k`` call that
        supplies it is rejected rather than silently ignoring it.
    time : ndarray of shape (n_samples,), str, or None, default None
        Time values ordering auto-k holdout splits, or the name of a DataFrame
        column, under the same rules as ``groups``.
    auto_k_config : AutoKConfig or None, default None
        Auto-k policy used when ``k="auto"``.  ``None`` infers
        ``strategy="time_holdout"`` from ``time`` or ``strategy="group_cv"``
        from ``groups`` and raises if neither is present.  mRMR supports
        ``k_method="evaluate"`` with ``estimator="classic"``, and additionally
        ``"auto"``, ``"elbow"``, ``"gaussian_cv"``, ``"xfit_objective"`` and
        ``"stability"`` with ``estimator="gaussian"``; any other method is
        rejected by name.  Function-style calls stay on
        ``auto_k_mode="prefix_only"``: one path is built and its prefixes are
        scored.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Finite, non-negative row weights with at least one positive entry.
        Used for relevance, redundancy, and auto-k validation scoring.  Not
        supported by ``estimator="ksg"``, and rejected together with ``cache``.
    relevance : {"f", "ks", "rf"}, default "f"
        Marginal relevance score for ``estimator="classic"``.  ``"f"`` is the
        F-test and works for both tasks; ``"rf"`` is a random-forest score;
        ``"ks"`` is classification-only.  Ignored by
        ``estimator="gaussian"``, which always uses copula Gaussian MI.
    estimator : {"classic", "gaussian"}, default "classic"
        ``"classic"`` scores relevance with ``relevance`` and redundancy with
        Pearson-style correlation.  ``"gaussian"`` is the regression-only
        rank-Gaussian copula path; it is much faster, accepts ``cache``, and
        is the only route that supports ``store_proxies`` or non-``evaluate``
        auto-k methods.
    formula : {"quotient", "difference"}, default "quotient"
        How relevance and mean redundancy are combined: ``"quotient"`` scores
        ``relevance / mean_redundancy``, ``"difference"`` scores
        ``relevance - mean_redundancy``.
    top_m : int or None, default None
        Candidate screen applied before the greedy loop.  ``None`` means
        ``max(5 * k, 250)``; the effective value is never below ``k``.
    cat_features : list of str or None, default None
        Categorical columns to encode.  ``None`` with a DataFrame ``X`` means
        every object, category, and string column.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", \
"loo_logit"}, default "none"
        Categorical encoding.  ``"none"`` leaves columns untouched, so
        non-numeric ones raise.  ``"target_cv"`` is SIFT's built-in
        cross-fitted encoder and needs no optional dependency: every emitted
        value is a *centered category effect* -- out-of-fold training rows get
        ``fold_encoding - fold_training_prior`` and inference rows get
        ``full_fit_encoding - full_training_prior`` -- so a level a fold never
        saw emits exactly zero rather than a fold-identifying prior, and an
        ID-like column cannot mark its own fold.  That centering neutralizes
        only unseen-in-fold emissions; a level appearing twice in a fold's
        training rows still transmits its siblings' targets, so drop ID-like
        columns or pass ``groups`` if that must not reach selection.  The
        remaining values are the legacy full-data supervised encoders and
        require ``allow_full_data_target_encoding=True``.
    target_cv_n_splits : int, default 5
        Requested fold count for ``cat_encoding="target_cv"``.  Must be at
        least 2; the encoder reports the count it could actually use in
        result metadata.
    target_cv_smoothing : {"auto"} or float, default "auto"
        Empirical-Bayes shrinkage for ``"target_cv"``.  ``"auto"`` is defined
        by weighted row mass and is therefore available on every fold kind,
        weighted or not; an explicit value must be finite and non-negative.
    target_prior : float or None, default None
        Target-independent prior used to encode the earliest time-fold rows,
        which have no history.  Only meaningful for time-aware ``"target_cv"``
        and mutually exclusive with ``warmup_policy="exclude"``.
    warmup_policy : {"exclude", "zero_weight"}, default "zero_weight"
        Disposition of those warmup rows when no ``target_prior`` is given.
        Both settings remove them from the selection fit through zero
        effective weight.  Only meaningful for time-aware ``"target_cv"``.
    allow_full_data_target_encoding : bool, default False
        Opt in to fitting a legacy supervised encoder on every row, which
        leaks the target into the features.  Required by ``cat_encoding`` in
        ``{"target", "loo", "james_stein", "loo_logit"}`` and rejected
        together with ``"target_cv"``, whose cross-fitted contract it
        contradicts.
    subsample : int or None, default 50000
        Row cap for the selection path, sampled with ``random_state``.
        ``None`` uses every row.  Cannot be passed with ``cache``.
    random_state : int, default 0
        Seed for subsampling and for the ``relevance="rf"`` forest.  Cannot be
        passed with ``cache``; rebuild the cache with the seed you want.
    n_jobs : int, default 1
        Worker count for the redundancy loop and, for ``estimator="gaussian"``
        without a cache, for building the rank-Gaussian transform.  Must not
        be 0.
    mrmr_backend : {"auto", "serial", "blas", "processes"}, default "auto"
        Redundancy-update backend.  ``"auto"`` resolves to ``"blas"``
        regardless of ``n_jobs``, because the BLAS matvec update avoids
        process start-up and pickling costs; pass ``"processes"`` explicitly
        to opt into joblib workers.
    verbose : bool, default True
        Log progress at INFO on the ``"sift"`` logger.  Use
        ``sift.set_verbosity`` for a process-wide default.
    return_result : bool, default False
        Return a ``sift.selection.result.FilterSelectionResult`` instead
        of the bare list.
    store_proxies : bool, default False
        Retain the selection-time copula correlation block so
        ``result_view().proxies()`` can report near-duplicate stand-ins.
        Requires ``return_result=True`` and ``estimator="gaussian"``.
    include : sequence of names or positions, optional
        Conditioning set. Redundancy state is initialized from these features
        before step 1. They appear in the output in caller order but are not
        discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    callback : callable or None, default None
        Progress hook ``callback(step, total, info)`` fired after each
        completed path step with a one-based ``step``.  Exceptions raised
        inside it propagate.

    Returns
    -------
    list of str or FilterSelectionResult
        By default the selected feature names in selection order.  With
        ``return_result=True``, a
        ``sift.selection.result.FilterSelectionResult`` carrying
        ``selected_features``, ``selected_indices`` (positions in ``X``),
        ``selector_metadata``, a ``ranking_`` table, and ``diagnostics_``;
        pass it to ``sift.as_result`` for a normalized view.

    Raises
    ------
    ValueError
        If ``task`` is not one of the two allowed values; if ``k`` is not a
        positive integer or ``"auto"``; if ``X`` is not 2-D or its row count
        differs from ``y``, ``groups``, ``time``, or ``sample_weight``; if
        ``estimator`` is not ``"classic"`` or ``"gaussian"``; if ``relevance``
        is invalid for ``task``; if ``groups`` or ``time`` is supplied for a
        fixed-``k`` call; if ``k="auto"`` has neither split context nor an
        ``auto_k_config``, or names a ``k_method`` this route does not
        support; if ``cache`` is combined with a non-Gaussian estimator,
        ``sample_weight``, ``subsample``, or ``random_state``, or does not
        match ``X``; if ``store_proxies`` is used without ``return_result`` or
        outside the Gaussian route; or if the categorical-encoding flags
        conflict as described above.
    TypeError
        If ``cat_features``/``cat_encoding`` are used with an ndarray ``X``.

    Warns
    -----
    UserWarning
        When ``k="auto"`` with ``k_method="auto"`` selects zero features: the
        routed criterion supported no feature, which is a real answer on
        noise-like data.  Inspect ``diagnostics_["auto_k"]`` with
        ``return_result=True``, or pass an explicit
        ``AutoKConfig(k_method=..., min_k=1)`` for a hard non-empty floor.
        A fixed-``k`` ``estimator="gaussian"`` run additionally warns when
        selected features have marginal Gaussian-MI relevance below the noise
        floor expected from the strongest of the valid columns under
        independence; the auto-k path builders suppress that check because
        they truncate the path afterwards.

    See Also
    --------
    sift.select_jmi : Joint-information selection that avoids mRMR's failure mode.
    sift.select_jmim : Conservative minimum-pair variant of JMI.
    sift.select_cefsplus : Log-determinant conditional-information selection.
    sift.build_cache : Build the cache the Gaussian route can reuse.

    Notes
    -----
    At each step mRMR scores every remaining candidate by its relevance to
    ``y`` against its mean redundancy with the already-selected set, combining
    the two by ``formula``.  Both terms are on one scale, which is the source
    of its known failure mode: once informative features are mutually
    redundant, a pure-noise column with tiny relevance and tiny redundancy can
    win.  The classic route measures redundancy with correlations of the raw
    columns; the Gaussian route measures it as ``-0.5 * log(1 - r**2)`` on
    rank-Gaussian correlations, which makes it monotone-invariant and lets a
    ``sift.FeatureCache`` be reused across targets.  ``k`` is an upper
    bound throughout, and every screening or pruning step can only lower the
    count actually returned.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_mrmr
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)),
    ...                  columns=[f"f{i}" for i in range(6)])
    >>> X["dup"] = X["f0"]  # an exact duplicate of a strong feature
    >>> y = X["f0"] + 0.5 * X["f3"] + 0.1 * rng.normal(size=200)
    >>> select_mrmr(X, y, k=2, task="regression", verbose=False)
    ['f0', 'f3']
    >>> result = select_mrmr(X, y, k=2, task="regression", verbose=False,
    ...                      return_result=True)
    >>> result.selected_features, result.selected_indices
    (['f0', 'f3'], [0, 3])
    """
    request = _request_from_public_locals(
        locals(),
        task=task,
        selector_names=MRMR_SELECTOR_KWARGS,
    )
    return _select_filter(_mrmr_spec(request), request)


def select_jmi(
    X: XInput, y: YInput, k: KInput, *, task: Task,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    estimator: EstimatorJMI = "auto", relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None, cat_features: Optional[list[str]] = None,
    cat_encoding: CatEncoding = "none",
    target_cv_n_splits: int = 5, target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = _SUBSAMPLE_DEFAULT, random_state: int = _RANDOM_STATE_DEFAULT,
    verbose: bool = True, return_result: bool = False, store_proxies: bool = False,
    include=None, exclude=None, candidates=None,
    callback: ProgressCallback | None = None,
) -> list[str] | FilterSelectionResult:
    """Joint Mutual Information feature selection.

    Greedily grows a feature set by scoring each candidate on the joint
    information it carries about the target *together with* each
    already-selected feature, summed over the selected set.  Because the score
    conditions on what is already chosen instead of penalizing pairwise
    redundancy, JMI prefers complementary features and does not share mRMR's
    tendency to promote low-relevance noise columns.  Use it as the default
    filter when features overlap; use ``sift.select_jmim`` for the
    conservative minimum-pair variant.  With defaults it resolves the
    estimator from ``task``, screens to ``max(5 * k, 250)`` candidates,
    subsamples 50,000 rows with seed 0, logs progress, and returns a plain
    ``list[str]``.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Feature matrix.  DataFrame labels are preserved in the output; an
        unlabelled array yields the positional names ``"x0", "x1", ...``.
        Non-numeric columns need ``cat_features``/``cat_encoding`` or
        pre-encoding.
    y : Series or ndarray of shape (n_samples,)
        Target, interpreted according to ``task``.
    k : int or "auto"
        Number of features to select, treated as an *upper bound*.  ``"auto"``
        hands the count to the auto-k machinery -- see ``auto_k_config``.
    task : {"regression", "classification"}
        Required keyword.  Chooses the estimator resolution and the target
        validation applied to ``y``.
    cache : FeatureCache or None, default None
        Prebuilt copula cache from ``sift.build_cache``, accepted only
        with ``estimator="gaussian"``.  A named cache requires the DataFrame
        whose labels and order built it; a positional cache requires the
        matching ndarray.  ``sample_weight``, ``subsample``, and
        ``random_state`` cannot accompany it.
    groups : ndarray of shape (n_samples,), str, or None, default None
        Group labels defining auto-k validation splits, or the name of a
        DataFrame column to use as such (the column is then removed from the
        features).  Rejected for fixed-``k`` calls.
    time : ndarray of shape (n_samples,), str, or None, default None
        Time values ordering auto-k holdout splits, or a DataFrame column
        name, under the same rules as ``groups``.
    auto_k_config : AutoKConfig or None, default None
        Auto-k policy used when ``k="auto"``.  ``None`` infers the strategy
        from ``time`` or ``groups`` and raises if neither is present.  The
        classic estimators support ``k_method="evaluate"`` only;
        ``estimator="gaussian"`` additionally supports ``"auto"``,
        ``"elbow"``, ``"gaussian_cv"``, ``"xfit_objective"`` and
        ``"stability"``.  Function-style calls stay on
        ``auto_k_mode="prefix_only"``.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Finite, non-negative row weights with at least one positive entry.
        With ``estimator="binned"`` they weight both the bin edges and the
        entropy counts.  Rejected by ``estimator="ksg"`` and by ``cache``.
    estimator : {"auto", "binned", "r2", "ksg", "gaussian"}, default "auto"
        Mutual-information estimator.  ``"auto"`` resolves to ``"binned"`` for
        classification and ``"r2"`` for regression.  ``"binned"`` uses
        quantile bins and is the only classification-capable choice;
        ``"r2"``, ``"ksg"``, and ``"gaussian"`` are regression-only.
        ``"gaussian"`` is the cache-compatible rank-Gaussian path and the only
        one that accepts ``cache`` or ``store_proxies``.
    relevance : {"f", "ks", "rf"}, default "f"
        Marginal relevance used to seed the path and break ties for the
        classic estimators.  ``"f"`` and ``"rf"`` serve both tasks; ``"ks"``
        is classification-only.  Ignored by ``estimator="gaussian"``, which
        uses copula Gaussian MI.
    top_m : int or None, default None
        Candidate screen applied before the greedy loop.  ``None`` means
        ``max(5 * k, 250)``; the effective value is never below ``k``.
    cat_features : list of str or None, default None
        Categorical columns to encode.  ``None`` with a DataFrame ``X`` means
        every object, category, and string column.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", \
"loo_logit"}, default "none"
        Categorical encoding.  ``"none"`` leaves columns untouched, so
        non-numeric ones raise.  ``"target_cv"`` is SIFT's built-in
        cross-fitted encoder: every emitted value is a *centered category
        effect* -- out-of-fold training rows get
        ``fold_encoding - fold_training_prior`` and inference rows get
        ``full_fit_encoding - full_training_prior`` -- so a level a fold never
        saw emits exactly zero instead of a fold-identifying prior.  That
        centering neutralizes only unseen-in-fold emissions; a level seen
        twice in a fold's training rows still transmits its siblings' targets,
        so drop ID-like columns or pass ``groups`` if that must not reach
        selection.  The remaining values are legacy full-data supervised
        encoders and require ``allow_full_data_target_encoding=True``.
    target_cv_n_splits : int, default 5
        Requested fold count for ``cat_encoding="target_cv"``; at least 2.
    target_cv_smoothing : {"auto"} or float, default "auto"
        Empirical-Bayes shrinkage for ``"target_cv"``.  ``"auto"`` is defined
        by weighted row mass and works on every fold kind; an explicit value
        must be finite and non-negative.
    target_prior : float or None, default None
        Target-independent prior for the earliest time-fold rows.  Only
        meaningful for time-aware ``"target_cv"``, and mutually exclusive with
        ``warmup_policy="exclude"``.
    warmup_policy : {"exclude", "zero_weight"}, default "zero_weight"
        Disposition of those warmup rows when no ``target_prior`` is given;
        both remove them from the selection fit through zero effective weight.
        Only meaningful for time-aware ``"target_cv"``.
    allow_full_data_target_encoding : bool, default False
        Opt in to fitting a legacy supervised encoder on every row.  Required
        by ``cat_encoding`` in ``{"target", "loo", "james_stein",
        "loo_logit"}`` and rejected together with ``"target_cv"``.
    subsample : int or None, default 50000
        Row cap for the selection path, sampled with ``random_state``.
        ``None`` uses every row.  Cannot be passed with ``cache``.
    random_state : int, default 0
        Seed for subsampling and for the ``relevance="rf"`` forest.  Cannot be
        passed with ``cache``.
    verbose : bool, default True
        Log progress at INFO on the ``"sift"`` logger.
    return_result : bool, default False
        Return a ``sift.selection.result.FilterSelectionResult`` instead
        of the bare list.
    store_proxies : bool, default False
        Retain the selection-time copula correlation block for
        ``result_view().proxies()``.  Requires ``return_result=True`` and
        ``estimator="gaussian"``.
    include : sequence of names or positions, optional
        Conditioning set. Joint-information state is initialized from these
        features before step 1. They appear in the output in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    callback : callable or None, default None
        Progress hook ``callback(step, total, info)`` fired after each
        completed path step with a one-based ``step``; exceptions propagate.

    Returns
    -------
    list of str or FilterSelectionResult
        By default the selected feature names in selection order.  With
        ``return_result=True``, a
        ``sift.selection.result.FilterSelectionResult`` carrying
        ``selected_features``, ``selected_indices``, ``selector_metadata``, a
        ``ranking_`` table, and ``diagnostics_``.

    Raises
    ------
    ValueError
        If ``task`` is invalid; if ``k`` is not a positive integer or
        ``"auto"``; if ``X`` is not 2-D or row counts disagree; if
        ``estimator`` is not one of the five allowed values, or is a
        regression-only estimator used with ``task="classification"``; if
        ``relevance`` is invalid for ``task``; if ``estimator="ksg"`` is
        combined with ``sample_weight``; if ``groups`` or ``time`` is supplied
        for a fixed-``k`` call; if ``k="auto"`` lacks split context and an
        ``auto_k_config``, or names an unsupported ``k_method``; if ``cache``
        is combined with a non-Gaussian estimator, ``sample_weight``,
        ``subsample``, or ``random_state``, or does not match ``X``; if
        ``store_proxies`` is used without ``return_result`` or outside the
        Gaussian route; or if the categorical-encoding flags conflict.
    TypeError
        If ``cat_features``/``cat_encoding`` are used with an ndarray ``X``.

    Warns
    -----
    UserWarning
        When ``k="auto"`` with ``k_method="auto"`` selects zero features: the
        routed criterion supported no feature.  Inspect
        ``diagnostics_["auto_k"]`` with ``return_result=True``, or pass an
        explicit ``AutoKConfig(k_method=..., min_k=1)`` for a hard non-empty
        floor.

    See Also
    --------
    sift.select_jmim : The conservative minimum-pair aggregation of this score.
    sift.select_mrmr : Faster relevance/redundancy baseline.
    sift.select_cefsplus : Log-determinant conditional-information selection.
    sift.build_cache : Build the cache the Gaussian route can reuse.

    Notes
    -----
    The JMI score of a candidate ``f`` given the selected set ``S`` is
    ``sum over s in S of I(f, s; y)``, so a feature is rewarded for
    information it adds *jointly* with each incumbent rather than penalized
    for correlating with it.  Summing keeps the score growing with ``|S|`` and
    makes it comparatively tolerant of one redundant pair;
    ``sift.select_jmim`` takes the minimum instead and is stricter.  The
    Gaussian route evaluates the pair term in closed form from rank-Gaussian
    correlations, costing one correlation-row update per step, and can reuse a
    ``sift.FeatureCache`` across targets.  ``k`` is an upper bound.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_jmi
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)),
    ...                  columns=[f"f{i}" for i in range(6)])
    >>> X["dup"] = X["f0"]
    >>> y = X["f0"] + 0.5 * X["f3"] + 0.1 * rng.normal(size=200)
    >>> select_jmi(X, y, k=2, task="regression", verbose=False)
    ['f0', 'f3']
    >>> select_jmi(X, y, k=2, task="regression", estimator="gaussian",
    ...            verbose=False)
    ['f0', 'f3']
    """
    request = _request_from_public_locals(
        locals(),
        task=task,
        selector_names=JMI_SELECTOR_KWARGS,
    )
    return _select_filter(_jmi_spec(request, JMI_CLASSIC_SPECS, JMI_GAUSSIAN_SPEC), request)


def select_jmim(
    X: XInput, y: YInput, k: KInput, *, task: Task,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    estimator: EstimatorJMI = "auto", relevance: RelevanceMethod = "f",
    top_m: Optional[int] = None, cat_features: Optional[list[str]] = None,
    cat_encoding: CatEncoding = "none",
    target_cv_n_splits: int = 5, target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = _SUBSAMPLE_DEFAULT, random_state: int = _RANDOM_STATE_DEFAULT,
    verbose: bool = True, return_result: bool = False, store_proxies: bool = False,
    include=None, exclude=None, candidates=None,
    callback: ProgressCallback | None = None,
) -> list[str] | FilterSelectionResult:
    """JMI Maximization, using the conservative minimum-pair aggregation.

    Identical to ``sift.select_jmi`` except that a candidate is scored by
    its *worst* pairing with the already-selected set rather than the sum over
    all of them.  A feature therefore has to add joint information alongside
    every incumbent, not merely on average, which makes JMIM the conservative
    choice when one redundant pair must not carry a candidate through.  It
    typically returns a smaller, less overlapping set than JMI at the same
    ``k``.  With defaults it resolves the estimator from ``task``, screens to
    ``max(5 * k, 250)`` candidates, subsamples 50,000 rows with seed 0, logs
    progress, and returns a plain ``list[str]``.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Feature matrix.  DataFrame labels are preserved in the output; an
        unlabelled array yields the positional names ``"x0", "x1", ...``.
        Non-numeric columns need ``cat_features``/``cat_encoding`` or
        pre-encoding.
    y : Series or ndarray of shape (n_samples,)
        Target, interpreted according to ``task``.
    k : int or "auto"
        Number of features to select, treated as an *upper bound*.  ``"auto"``
        hands the count to the auto-k machinery -- see ``auto_k_config``.
    task : {"regression", "classification"}
        Required keyword.  Chooses the estimator resolution and the target
        validation applied to ``y``.
    cache : FeatureCache or None, default None
        Prebuilt copula cache from ``sift.build_cache``, accepted only
        with ``estimator="gaussian"``.  A named cache requires the DataFrame
        whose labels and order built it; a positional cache requires the
        matching ndarray.  ``sample_weight``, ``subsample``, and
        ``random_state`` cannot accompany it.
    groups : ndarray of shape (n_samples,), str, or None, default None
        Group labels defining auto-k validation splits, or the name of a
        DataFrame column to use as such (the column is then removed from the
        features).  Rejected for fixed-``k`` calls.
    time : ndarray of shape (n_samples,), str, or None, default None
        Time values ordering auto-k holdout splits, or a DataFrame column
        name, under the same rules as ``groups``.
    auto_k_config : AutoKConfig or None, default None
        Auto-k policy used when ``k="auto"``.  ``None`` infers the strategy
        from ``time`` or ``groups`` and raises if neither is present.  The
        classic estimators support ``k_method="evaluate"`` only;
        ``estimator="gaussian"`` additionally supports ``"auto"``,
        ``"elbow"``, ``"gaussian_cv"``, ``"xfit_objective"`` and
        ``"stability"``.  Function-style calls stay on
        ``auto_k_mode="prefix_only"``.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Finite, non-negative row weights with at least one positive entry.
        With ``estimator="binned"`` they weight both the bin edges and the
        entropy counts.  Rejected by ``estimator="ksg"`` and by ``cache``.
    estimator : {"auto", "binned", "r2", "ksg", "gaussian"}, default "auto"
        Mutual-information estimator.  ``"auto"`` resolves to ``"binned"`` for
        classification and ``"r2"`` for regression.  ``"binned"`` uses
        quantile bins and is the only classification-capable choice;
        ``"r2"``, ``"ksg"``, and ``"gaussian"`` are regression-only.
        ``"gaussian"`` is the cache-compatible rank-Gaussian path and the only
        one that accepts ``cache`` or ``store_proxies``.
    relevance : {"f", "ks", "rf"}, default "f"
        Marginal relevance used to seed the path and break ties for the
        classic estimators.  ``"f"`` and ``"rf"`` serve both tasks; ``"ks"``
        is classification-only.  Ignored by ``estimator="gaussian"``.
    top_m : int or None, default None
        Candidate screen applied before the greedy loop.  ``None`` means
        ``max(5 * k, 250)``; the effective value is never below ``k``.
    cat_features : list of str or None, default None
        Categorical columns to encode.  ``None`` with a DataFrame ``X`` means
        every object, category, and string column.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", \
"loo_logit"}, default "none"
        Categorical encoding.  ``"none"`` leaves columns untouched, so
        non-numeric ones raise.  ``"target_cv"`` is SIFT's built-in
        cross-fitted encoder: every emitted value is a *centered category
        effect* -- out-of-fold training rows get
        ``fold_encoding - fold_training_prior`` and inference rows get
        ``full_fit_encoding - full_training_prior`` -- so a level a fold never
        saw emits exactly zero instead of a fold-identifying prior.  That
        centering neutralizes only unseen-in-fold emissions; a level seen
        twice in a fold's training rows still transmits its siblings' targets,
        so drop ID-like columns or pass ``groups`` if that must not reach
        selection.  The remaining values are legacy full-data supervised
        encoders and require ``allow_full_data_target_encoding=True``.
    target_cv_n_splits : int, default 5
        Requested fold count for ``cat_encoding="target_cv"``; at least 2.
    target_cv_smoothing : {"auto"} or float, default "auto"
        Empirical-Bayes shrinkage for ``"target_cv"``.  ``"auto"`` is defined
        by weighted row mass and works on every fold kind; an explicit value
        must be finite and non-negative.
    target_prior : float or None, default None
        Target-independent prior for the earliest time-fold rows.  Only
        meaningful for time-aware ``"target_cv"``, and mutually exclusive with
        ``warmup_policy="exclude"``.
    warmup_policy : {"exclude", "zero_weight"}, default "zero_weight"
        Disposition of those warmup rows when no ``target_prior`` is given;
        both remove them from the selection fit through zero effective weight.
        Only meaningful for time-aware ``"target_cv"``.
    allow_full_data_target_encoding : bool, default False
        Opt in to fitting a legacy supervised encoder on every row.  Required
        by ``cat_encoding`` in ``{"target", "loo", "james_stein",
        "loo_logit"}`` and rejected together with ``"target_cv"``.
    subsample : int or None, default 50000
        Row cap for the selection path, sampled with ``random_state``.
        ``None`` uses every row.  Cannot be passed with ``cache``.
    random_state : int, default 0
        Seed for subsampling and for the ``relevance="rf"`` forest.  Cannot be
        passed with ``cache``.
    verbose : bool, default True
        Log progress at INFO on the ``"sift"`` logger.
    return_result : bool, default False
        Return a ``sift.selection.result.FilterSelectionResult`` instead
        of the bare list.
    store_proxies : bool, default False
        Retain the selection-time copula correlation block for
        ``result_view().proxies()``.  Requires ``return_result=True`` and
        ``estimator="gaussian"``.
    include : sequence of names or positions, optional
        Conditioning set. Joint-information state is initialized from these
        features before step 1. They appear in the output in caller order
        but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    callback : callable or None, default None
        Progress hook ``callback(step, total, info)`` fired after each
        completed path step with a one-based ``step``; exceptions propagate.

    Returns
    -------
    list of str or FilterSelectionResult
        By default the selected feature names in selection order.  With
        ``return_result=True``, a
        ``sift.selection.result.FilterSelectionResult`` carrying
        ``selected_features``, ``selected_indices``, ``selector_metadata``, a
        ``ranking_`` table, and ``diagnostics_``.

    Raises
    ------
    ValueError
        If ``task`` is invalid; if ``k`` is not a positive integer or
        ``"auto"``; if ``X`` is not 2-D or row counts disagree; if
        ``estimator`` is not one of the five allowed values, or is a
        regression-only estimator used with ``task="classification"``; if
        ``relevance`` is invalid for ``task``; if ``estimator="ksg"`` is
        combined with ``sample_weight``; if ``groups`` or ``time`` is supplied
        for a fixed-``k`` call; if ``k="auto"`` lacks split context and an
        ``auto_k_config``, or names an unsupported ``k_method``; if ``cache``
        is combined with a non-Gaussian estimator, ``sample_weight``,
        ``subsample``, or ``random_state``, or does not match ``X``; if
        ``store_proxies`` is used without ``return_result`` or outside the
        Gaussian route; or if the categorical-encoding flags conflict.
    TypeError
        If ``cat_features``/``cat_encoding`` are used with an ndarray ``X``.

    Warns
    -----
    UserWarning
        When ``k="auto"`` with ``k_method="auto"`` selects zero features: the
        routed criterion supported no feature.  Inspect
        ``diagnostics_["auto_k"]`` with ``return_result=True``, or pass an
        explicit ``AutoKConfig(k_method=..., min_k=1)`` for a hard non-empty
        floor.

    See Also
    --------
    sift.select_jmi : The sum aggregation of the same joint-information score.
    sift.select_mrmr : Faster relevance/redundancy baseline.
    sift.select_cefsplus : Log-determinant conditional-information selection.
    sift.build_cache : Build the cache the Gaussian route can reuse.

    Notes
    -----
    The JMIM score of a candidate ``f`` given the selected set ``S`` is
    ``min over s in S of I(f, s; y)``.  The minimum is a hard constraint: one
    incumbent with which the candidate is uninformative caps its score,
    whatever the other pairings contribute.  That is exactly why it is more
    conservative than JMI's sum, and why it is the safer default when a single
    strong feature would otherwise pull near-duplicates in behind it.  Both
    aggregations share one path builder, so estimator behavior, screening,
    weighting, and cache reuse are identical; only the aggregation differs.
    ``k`` is an upper bound.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_jmim
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)),
    ...                  columns=[f"f{i}" for i in range(6)])
    >>> X["dup"] = X["f0"]
    >>> y = X["f0"] + 0.5 * X["f3"] + 0.1 * rng.normal(size=200)
    >>> select_jmim(X, y, k=2, task="regression", verbose=False)
    ['f0', 'f3']
    >>> select_jmim(X, y, k=3, task="regression", verbose=False)[0]
    'f0'
    """
    request = _request_from_public_locals(
        locals(),
        task=task,
        selector_names=JMI_SELECTOR_KWARGS,
    )
    return _select_filter(
        _jmi_spec(request, JMIM_CLASSIC_SPECS, JMIM_GAUSSIAN_SPEC),
        request,
    )


def select_cefsplus(
    X: XInput, y: YInput, k: KInput = 75, *,
    cache: Optional[FeatureCache] = None, groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    top_m: Optional[int] = None, corr_prune: float | None = None,
    cat_features: Optional[list[str]] = None, cat_encoding: CatEncoding = "none",
    target_cv_n_splits: int = 5, target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = _SUBSAMPLE_DEFAULT, random_state: int = _RANDOM_STATE_DEFAULT,
    verbose: bool = True, return_result: bool = False, store_proxies: bool = False,
    include=None, exclude=None, candidates=None,
    callback: ProgressCallback | None = None,
) -> list[str] | FilterSelectionResult:
    """CEFS+ feature selection using log-det Gaussian MI proxy.

    Greedily grows a feature set that maximizes a log-determinant conditional
    information objective on the rank-Gaussian (copula) transform of ``X``, so
    each step adds the feature that explains the most target variance *given*
    everything already selected.  This is SIFT's strongest regression filter
    and the recommended default: unlike mRMR it conditions on the full
    selected set rather than penalizing pairwise redundancy, and unlike JMI it
    accounts for the joint covariance rather than pairs.  It is regression-only
    -- ``y`` is always read as numeric.  With defaults it selects up to 75
    features, screens to ``max(5 * k, 250)`` candidates, applies no
    correlation pruning, subsamples 50,000 rows with seed 0, logs progress,
    and returns a plain ``list[str]``.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Feature matrix.  DataFrame labels are preserved in the output; an
        unlabelled array yields the positional names ``"x0", "x1", ...``.
        Non-numeric columns need ``cat_features``/``cat_encoding`` or
        pre-encoding.
    y : Series or ndarray of shape (n_samples,)
        Numeric regression target.  Must be finite; there is no ``task``
        argument, so labels-shaped targets are only warned about, not encoded.
    k : int or "auto", default 75
        Number of features to select, treated as an *upper bound*.  ``"auto"``
        hands the count to the auto-k machinery -- see ``auto_k_config``.
    cache : FeatureCache or None, default None
        Prebuilt copula cache from ``sift.build_cache``, reused instead of
        transforming ``X`` again.  A named cache requires the DataFrame whose
        labels and order built it; a positional cache requires the matching
        ndarray.  Because a cache freezes its rows and weights,
        ``sample_weight``, ``subsample``, and ``random_state`` cannot be
        passed alongside it.
    groups : ndarray of shape (n_samples,), str, or None, default None
        Group labels defining auto-k validation splits, or the name of a
        DataFrame column to use as such (the column is then removed from the
        features).  Rejected for fixed-``k`` calls.
    time : ndarray of shape (n_samples,), str, or None, default None
        Time values ordering auto-k holdout splits, or a DataFrame column
        name, under the same rules as ``groups``.
    auto_k_config : AutoKConfig or None, default None
        Auto-k policy used when ``k="auto"``.  Leaving it ``None`` selects the
        zero-config router, ``AutoKConfig(k_method="auto")``, which needs no
        ``groups`` or ``time`` and records its branch in
        ``diagnostics_["auto_k"]``.  CEFS+ supports the widest set of methods:
        ``"auto"``, ``"evaluate"``, ``"elbow"``, ``"xfit_objective"``,
        ``"gaussian_cv"``, ``"stability"``, ``"penalized_objective"``,
        ``"k_posterior"``, ``"chi2_stop"``, ``"forward_stop"``,
        ``"changepoint"``, ``"perm_gap"``, ``"knockoff_path"``, and
        ``"consensus"``.  Function-style calls stay on
        ``auto_k_mode="prefix_only"``: one path is built and its prefixes are
        scored.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Finite, non-negative row weights with at least one positive entry,
        used for the copula transform, the correlations, and auto-k scoring.
        Rejected together with ``cache``, whose weights are already fixed.
    top_m : int or None, default None
        Candidate screen applied before the greedy loop: only the features
        with the largest absolute copula correlation with ``y`` compete.
        ``None`` means ``max(5 * k, 250)`` (``max_k`` in place of ``k`` for
        auto-k); the effective value is never below ``k``.
    corr_prune : float or None, default None
        Redundancy prefilter on the screened panel.  ``None`` means no
        pruning, which keeps suppressor pairs eligible.  A float in
        ``(0, 1]`` such as ``0.95`` greedily drops any candidate whose
        absolute correlation with a better-scoring survivor reaches the
        threshold -- useful when duplicate suppression matters more than
        recovering suppressors.
    cat_features : list of str or None, default None
        Categorical columns to encode.  ``None`` with a DataFrame ``X`` means
        every object, category, and string column.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", \
"loo_logit"}, default "none"
        Categorical encoding.  ``"none"`` leaves columns untouched, so
        non-numeric ones raise.  ``"target_cv"`` is SIFT's built-in
        cross-fitted encoder: every emitted value is a *centered category
        effect* -- out-of-fold training rows get
        ``fold_encoding - fold_training_prior`` and inference rows get
        ``full_fit_encoding - full_training_prior`` -- so a level a fold never
        saw emits exactly zero instead of a fold-identifying prior, and an
        ID-like column cannot mark its own fold.  That centering neutralizes
        only unseen-in-fold emissions; a level seen twice in a fold's training
        rows still transmits its siblings' targets, so drop ID-like columns or
        pass ``groups`` if that must not reach selection.  The remaining
        values are legacy full-data supervised encoders and require
        ``allow_full_data_target_encoding=True``.  Contextual ``"target_cv"``
        with ``groups`` or ``time`` is accepted only under ``k="auto"`` with
        ``AutoKConfig(k_method="evaluate")``.
    target_cv_n_splits : int, default 5
        Requested fold count for ``cat_encoding="target_cv"``; at least 2.
        The fitted encoder reports the count it could actually use.
    target_cv_smoothing : {"auto"} or float, default "auto"
        Empirical-Bayes shrinkage for ``"target_cv"``.  ``"auto"`` is defined
        by weighted row mass and is therefore available on every fold kind,
        weighted or not; an explicit value must be finite and non-negative.
    target_prior : float or None, default None
        Target-independent prior used to encode the earliest time-fold rows,
        which have no history.  Only meaningful for time-aware ``"target_cv"``
        and mutually exclusive with ``warmup_policy="exclude"``.
    warmup_policy : {"exclude", "zero_weight"}, default "zero_weight"
        Disposition of those warmup rows when no ``target_prior`` is given;
        both remove them from the selection fit through zero effective weight.
        Only meaningful for time-aware ``"target_cv"``.
    allow_full_data_target_encoding : bool, default False
        Opt in to fitting a legacy supervised encoder on every row, which
        leaks the target into the features.  Required by ``cat_encoding`` in
        ``{"target", "loo", "james_stein", "loo_logit"}`` and rejected
        together with ``"target_cv"``.
    subsample : int or None, default 50000
        Row cap for the copula cache built from ``X``, sampled with
        ``random_state``.  ``None`` uses every positive-weight row.  Cannot be
        passed with ``cache``.
    random_state : int, default 0
        Seed for that subsampling draw and for stochastic auto-k methods.
        Cannot be passed with ``cache``; rebuild the cache with the seed you
        want.
    verbose : bool, default True
        Log progress at INFO on the ``"sift"`` logger.  Use
        ``sift.set_verbosity`` for a process-wide default.
    return_result : bool, default False
        Return a ``sift.selection.result.FilterSelectionResult`` instead
        of the bare list.  Required to inspect the objective path and the
        auto-k diagnostics.
    store_proxies : bool, default False
        Retain the selection-time candidate-by-selected copula correlation
        block so ``result_view().proxies()`` can report near-duplicate
        stand-ins for a selected feature.  Requires ``return_result=True``;
        the block never contains ``X`` or a cache.
    include : sequence of names or positions, optional
        Conditioning set. Partial-Cholesky residual state is initialized from
        these features before step 1. They appear in the output in caller
        order but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    callback : callable or None, default None
        Progress hook ``callback(step, total, info)`` fired after each
        completed greedy step with a one-based ``step``.  Exceptions raised
        inside it propagate.

    Returns
    -------
    list of str or FilterSelectionResult
        By default the selected feature names in selection order.  With
        ``return_result=True``, a
        ``sift.selection.result.FilterSelectionResult`` carrying
        ``selected_features``, ``selected_indices`` (positions in ``X``),
        ``selector_metadata``, a ``ranking_`` table, and ``diagnostics_``
        holding the cumulative ``"objective_path"`` and its per-step
        ``"objective_gain"`` for a fixed ``k``, or the ``"auto_k"`` summary,
        diagnostics, and curve when ``k="auto"``.

    Raises
    ------
    ValueError
        If ``k`` is not a positive integer or ``"auto"``; if ``X`` is not 2-D
        or its row count differs from ``y``, ``groups``, ``time``, or
        ``sample_weight``; if ``y`` is non-finite; if ``corr_prune`` is
        outside ``(0, 1]``; if ``groups`` or ``time`` is supplied for a
        fixed-``k`` call; if ``k="auto"`` names a ``k_method`` this route does
        not support; if contextual ``cat_encoding="target_cv"`` is combined
        with ``groups``/``time`` outside ``k_method="evaluate"``; if ``cache``
        is combined with ``sample_weight``, ``subsample``, or
        ``random_state``, or does not match ``X``; if ``store_proxies`` is
        used without ``return_result``; or if the categorical-encoding flags
        conflict as described above.
    TypeError
        If ``cat_features``/``cat_encoding`` are used with an ndarray ``X``.

    Warns
    -----
    UserWarning
        When ``y`` holds only 3-20 distinct integer-valued levels and so looks
        like multiclass labels rather than a numeric target -- use a
        ``task="classification"`` selector or
        ``sift.select_cefsplus_binary`` instead.  Also when ``k="auto"``
        with ``k_method="auto"`` selects zero features: the routed criterion
        supported no feature, which is a real answer on noise-like data.
        Inspect ``diagnostics_["auto_k"]`` with ``return_result=True``, or
        pass an explicit ``AutoKConfig(k_method=..., min_k=1)`` for a hard
        non-empty floor.

    See Also
    --------
    sift.select_cefsplus_binary : Binary-target counterpart with a logistic path.
    sift.select_cached : Same objective, run repeatedly against one cache.
    sift.build_cache : Build the cache this selector can reuse.
    sift.select_fdr : Error-controlled discovery instead of a fixed count.

    Notes
    -----
    Selecting ``j`` given the current set ``S`` maximizes
    ``log|Sigma_{S+j}| - log|Sigma_{y,S+j}|``, whose cumulative value equals
    ``2 * I(y; S)`` under the fitted Gaussian copula and is what
    ``diagnostics_["objective_path"]`` reports; it is non-decreasing by
    construction.  The step is evaluated by a partial Cholesky (residual)
    recursion costing ``O(m * t)`` at step ``t`` for ``m`` screened
    candidates, so ``O(m * k**2)`` overall, and it issues no BLAS calls, so it
    cannot thrash the caller's thread pool.  Because everything before the
    greedy loop is target-independent, passing a prebuilt
    ``sift.FeatureCache`` makes repeated selection across targets cheap;
    ``sift.select_cached`` is the direct form of that loop.  ``k`` is an
    upper bound throughout.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_cefsplus
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(300, 8)),
    ...                  columns=[f"f{i}" for i in range(8)])
    >>> y = X["f1"] + 0.8 * X["f4"] + 0.1 * rng.normal(size=300)
    >>> select_cefsplus(X, y, k=2, verbose=False)
    ['f1', 'f4']
    >>> select_cefsplus(X, y, k="auto", verbose=False)  # zero-config router
    ['f1', 'f4']
    >>> result = select_cefsplus(X, y, k=2, verbose=False, return_result=True)
    >>> path = result.diagnostics_["objective_path"]
    >>> len(path), path[1] >= path[0]
    (2, True)
    """
    request = _request_from_public_locals(
        locals(),
        task="regression",
        selector_names=CEFSPLUS_SELECTOR_KWARGS,
    )
    return _select_filter(CEFSPLUS_SPEC, request)


@_single_threaded_binary_blas
def select_cefsplus_binary(
    X: XInput, y: YInput, k: KInput, *,
    loss: str = "logloss", top_m: Optional[int] = None, corr_prune: float | None = None,
    groups: Optional[np.ndarray] = None, time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: np.ndarray | None = None,
    class_weight=None,
    ridge: float = 1e-4, refit_every: int = 1,
    cat_features: Optional[list[str]] = None, cat_encoding: str = "none",
    target_cv_n_splits: int = 5, target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    loo_smoothing: float = 20.0, loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    allow_full_data_target_encoding: bool = False,
    subsample: Optional[int] = None, random_state: int = 0,
    verbose: bool = True, return_result: bool = False, store_proxies: bool = False,
    include=None, exclude=None, candidates=None,
    callback: ProgressCallback | None = None,
) -> list[str] | FilterSelectionResult:
    """Binary CEFS+ using a greedy conditional Bernoulli deviance proxy.

    Greedily grows a feature set for a two-class target by ranking each
    candidate on the conditional Bernoulli deviance it would remove from a
    weighted logistic fit on the features already selected, scored with a
    Rao/Fisher score-test update.  Use it as the binary counterpart of
    ``sift.select_cefsplus`` when the target really is a label and a
    logistic path is the right proxy; it is a different estimator, not the
    Gaussian log-determinant objective under another name.  With defaults it
    runs the log-loss path over all finite candidates and all rows, refits the
    selected-feature logistic every step, applies no correlation pruning, logs
    progress, and returns a plain ``list[str]``.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Feature matrix.  DataFrame labels are preserved in the output; an
        unlabelled array yields the positional names ``"x0", "x1", ...``.
        Non-numeric columns need ``cat_features``/``cat_encoding`` or
        pre-encoding.
    y : Series or ndarray of shape (n_samples,)
        Binary target with exactly two distinct non-missing values.  Labels of
        any hashable type are accepted and mapped to ``0``/``1``; the mapping
        is reported in the result metadata.
    k : int or "auto"
        Number of features to select, treated as an *upper bound*.  ``"auto"``
        hands the count to the auto-k machinery -- see ``auto_k_config``.
    loss : {"logloss", "brier"}, default "logloss"
        Selection proxy.  ``"logloss"`` runs the greedy logistic score-test
        path described above.  ``"brier"`` instead delegates to
        ``sift.select_cefsplus`` with the 0/1 target cast to float, which
        makes the Gaussian-only options (notably ``store_proxies``) available
        and reports ``delegate_selector="cefsplus"`` in metadata.
    top_m : int or None, default None
        Candidate screen applied before the greedy loop.  Unlike the other
        filters, ``None`` here means *every* finite candidate, so set it
        explicitly for wide binary screens.
    corr_prune : float or None, default None
        Absolute weighted feature-correlation pruning threshold in ``(0, 1]``.
        ``None`` means no pruning, which keeps possible suppressor pairs.
    groups : ndarray of shape (n_samples,), str, or None, default None
        Group labels defining auto-k validation splits, or the name of a
        DataFrame column to use as such (the column is then removed from the
        features).  Rejected for fixed-``k`` calls.
    time : ndarray of shape (n_samples,), str, or None, default None
        Time values ordering auto-k holdout splits, or a DataFrame column
        name, under the same rules as ``groups``.
    auto_k_config : AutoKConfig or None, default None
        Auto-k policy used when ``k="auto"``.  Leaving it ``None`` selects the
        zero-config router, ``AutoKConfig(k_method="auto")``, which needs no
        ``groups`` or ``time``.  Binary CEFS+ supports ``"auto"``,
        ``"evaluate"``, ``"elbow"``, ``"penalized_objective"``,
        ``"k_posterior"``, and ``"changepoint"``; the log-loss path also
        rejects non-default ``auto_dense_*`` fields rather than ignoring them,
        while ``loss="brier"`` inherits the Gaussian CEFS+ contract.  For
        ``k="auto"`` the path is built out to ``AutoKConfig.max_k``.
    sample_weight : ndarray of shape (n_samples,) or None, default None
        Finite, non-negative row weights with at least one positive entry.
        Combined multiplicatively with ``class_weight`` and normalized to mean
        1; each class must retain positive total weight.
    class_weight : None, "balanced", or dict, default None
        Per-class multiplier.  ``"balanced"`` equalizes the two classes' total
        weight; a dict must supply a finite, non-negative value for both *raw*
        class labels.  Weights are resolved on the input rows *before* any
        ``subsample`` draw, which the metadata records as
        ``class_weight_scope="pre_subsample"``.
    ridge : float, default 0.0001
        Positive, finite L2 penalty stabilizing both the selected-feature
        logistic fit and the candidate score-test information; the same term
        enters every candidate denominator.
    refit_every : int, default 1
        Positive step interval at which the selected-feature logistic null is
        refit.  The default refits every step.  Larger values switch to a
        block-Gram accelerator between refits and should be treated as an
        approximate speed mode, not an equivalent computation.
    cat_features : list of str or None, default None
        Categorical columns to encode.  ``None`` with a DataFrame ``X`` means
        every object, category, and string column.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", \
"loo_logit"}, default "none"
        Categorical encoding.  ``"none"`` leaves columns untouched, so
        non-numeric ones raise.  ``"target_cv"`` is SIFT's built-in
        cross-fitted encoder: every emitted value is a *centered category
        effect* -- out-of-fold training rows get
        ``fold_encoding - fold_training_prior`` and inference rows get
        ``full_fit_encoding - full_training_prior`` -- so a level a fold never
        saw emits exactly zero instead of a fold-identifying prior.  That
        centering neutralizes only unseen-in-fold emissions; a level seen
        twice in a fold's training rows still transmits its siblings' targets,
        so drop ID-like columns or pass ``groups`` if that must not reach
        selection.  ``"loo_logit"`` is the binary-specific leave-one-out logit
        encoder tuned by ``loo_smoothing`` and the clip bounds; under
        ``loss="brier"`` it is delegated as plain ``"loo"``.  Every value but
        ``"none"`` and ``"target_cv"`` requires
        ``allow_full_data_target_encoding=True``.
    target_cv_n_splits : int, default 5
        Requested fold count for ``cat_encoding="target_cv"``; at least 2.
    target_cv_smoothing : {"auto"} or float, default "auto"
        Empirical-Bayes shrinkage for ``"target_cv"``.  ``"auto"`` is defined
        by weighted row mass and works on every fold kind; an explicit value
        must be finite and non-negative.
    target_prior : float or None, default None
        Target-independent prior for the earliest time-fold rows; for a binary
        target it must lie in ``[0, 1]``.  Only meaningful for time-aware
        ``"target_cv"``, and mutually exclusive with
        ``warmup_policy="exclude"``.
    warmup_policy : {"exclude", "zero_weight"}, default "zero_weight"
        Disposition of those warmup rows when no ``target_prior`` is given;
        both remove them from the selection fit through zero effective weight.
        Only meaningful for time-aware ``"target_cv"``.
    loo_smoothing : float, default 20.0
        Positive, finite smoothing constant pulling each category's
        leave-one-out rate toward the global prior, for
        ``cat_encoding="loo_logit"``.
    loo_clip_min : float, default 0.0001
        Lower probability clip applied before the logit, for
        ``cat_encoding="loo_logit"``.
    loo_clip_max : float, default 0.9999
        Upper probability clip, which must satisfy
        ``0 < loo_clip_min < loo_clip_max < 1``.
    allow_full_data_target_encoding : bool, default False
        Opt in to fitting a legacy supervised encoder on every row.  Required
        by ``cat_encoding`` in ``{"target", "loo", "james_stein",
        "loo_logit"}`` and rejected together with ``"target_cv"``.
    subsample : int or None, default None
        Row cap for the selection path, sampled with ``random_state``.  Unlike
        the other filters this defaults to ``None``, meaning every row.
    random_state : int, default 0
        Seed for the subsampling draw and for stochastic auto-k methods.
    verbose : bool, default True
        Log progress at INFO on the ``"sift"`` logger.
    return_result : bool, default False
        Return a ``sift.selection.result.FilterSelectionResult`` instead
        of the bare list.
    store_proxies : bool, default False
        Retain the selection-time copula correlation block for
        ``result_view().proxies()``.  Requires ``return_result=True`` and
        ``loss="brier"``; the log-loss path rejects it rather than ignoring it.
    include : sequence of names or positions, optional
        Conditioning set. The logistic score-test state is initialized from
        these features before step 1. They appear in the output in caller
        order but are not discoveries; ``k`` counts additional features.
    exclude : sequence of names or positions, optional
        Features removed from the discovery pool. Cannot overlap ``include``.
    candidates : sequence of names or positions, optional
        Hard allow-list for discovery. ``include`` may sit outside it.
        Overlap with ``exclude`` is rejected. An empty remaining pool raises.
    callback : callable or None, default None
        Progress hook ``callback(step, total, info)`` fired after each
        completed path step with a one-based ``step``; exceptions propagate.

    Returns
    -------
    list of str or FilterSelectionResult
        By default the selected feature names in selection order.  With
        ``return_result=True``, a
        ``sift.selection.result.FilterSelectionResult`` whose
        ``selector_metadata`` records ``selector="cefsplus_binary"``, the
        ``loss``, the ``target_mapping`` from raw labels to ``0``/``1``,
        whether the run was ``weighted``, and the ``class_weight`` scope.

    Raises
    ------
    ValueError
        If ``y`` does not have exactly two distinct non-missing classes, or
        holds missing or non-finite values; if either class ends up with
        non-positive effective weight; if ``k`` is not a positive integer or
        ``"auto"``; if ``X`` is not 2-D or row counts disagree; if ``loss`` is
        not ``"logloss"`` or ``"brier"``; if ``ridge`` is not positive and
        finite; if ``refit_every`` is not a positive integer; if ``top_m`` or
        ``subsample`` is not a positive integer or ``None``; if ``corr_prune``
        is outside ``(0, 1]``; if the ``loo_*`` bounds violate
        ``0 < min < max < 1`` or ``loo_smoothing`` is not positive; if
        ``class_weight`` is neither ``None``, ``"balanced"``, nor a dict
        covering both raw labels; if ``groups`` or ``time`` is supplied for a
        fixed-``k`` call; if ``k="auto"`` names an unsupported ``k_method`` or
        sets ``auto_dense_*`` on the log-loss path; if ``store_proxies`` is
        used without ``return_result`` or on the log-loss path; or if the
        categorical-encoding flags conflict.
    TypeError
        If ``cat_features``/``cat_encoding`` are used with an ndarray ``X``.

    Warns
    -----
    UserWarning
        When ``k="auto"`` with ``k_method="auto"`` selects zero features: the
        routed criterion supported no feature, which is a real answer on
        noise-like data.  Inspect ``diagnostics_["auto_k"]`` with
        ``return_result=True``, or pass an explicit
        ``AutoKConfig(k_method=..., min_k=1)`` for a hard non-empty floor.

    See Also
    --------
    sift.select_cefsplus : Regression counterpart, and the ``"brier"`` delegate.
    sift.select_mrmr : Task-aware relevance/redundancy filter.
    sift.select_jmim : Conservative joint-information filter.

    Notes
    -----
    The log-loss path standardizes the weighted columns, fits an
    intercept-only Bernoulli model, and then repeatedly admits the candidate
    with the largest ridge-regularized score-test statistic against the
    current selected-feature fit, refitting that fit every ``refit_every``
    steps.  Scoring candidates by a score test rather than by ``k`` separate
    logistic refits is what keeps the path affordable; the price is that the
    ranking is a local quadratic approximation around the current fit, not an
    exact deviance comparison.  ``loss="brier"`` sidesteps the logistic path
    entirely by handing the 0/1 target to the Gaussian log-determinant
    objective.  Constant or non-finite columns are dropped before selection
    and reported in the diagnostics, and ``k`` is an upper bound throughout.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import select_cefsplus_binary
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(300, 6)),
    ...                  columns=[f"f{i}" for i in range(6)])
    >>> logit = 1.5 * X["f2"] - 1.5 * X["f5"]
    >>> y = (rng.uniform(size=300) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    >>> select_cefsplus_binary(X, y, k=2, verbose=False)
    ['f5', 'f2']
    >>> result = select_cefsplus_binary(X, y, k=2, verbose=False,
    ...                                 return_result=True)
    >>> result.selector_metadata["selector"], result.selector_metadata["loss"]
    ('cefsplus_binary', 'logloss')
    """
    request = _request_from_public_locals(
        locals(),
        task="classification",
        selector_names=CEFSPLUS_BINARY_SELECTOR_KWARGS,
    )
    if str((request.selector_kwargs or {}).get("loss")).lower() == "brier":
        return _select_brier_delegate(request)
    return _select_filter(CEFSPLUS_BINARY_SPEC, request)


def _select_filter(
    spec: FilterSpec,
    request: FilterRequest,
) -> list[str] | FilterSelectionResult:
    ctx = _build_context(spec, request)
    _require_fixed_filter_metadata(ctx)
    if ctx.k == "auto":
        if request.auto_k_config is None and spec.selector in {"cefsplus", "cefsplus_binary"}:
            resolved_config = AutoKConfig(k_method="auto")
        else:
            resolved_config = resolve_auto_k_config(request.auto_k_config, ctx.time, ctx.groups)
        ctx = replace(
            ctx,
            auto_k_config=resolved_config,
        )
        assert ctx.auto_k_config is not None
        handler = spec.auto_k_handlers.get(ctx.auto_k_config.k_method)
        if handler is None:
            raise ValueError(
                f"{spec.display_name} does not support k_method="
                f"{ctx.auto_k_config.k_method!r}"
            )
        _require_context_auto_k_compatibility(ctx)
        _require_auto_k_eval_context(ctx)
    else:
        handler = spec.fixed_handler

    spec.validate(ctx)
    if request.store_proxies and ctx.estimator != "gaussian":
        raise ValueError(
            "store_proxies=True is currently supported only by Gaussian/cached "
            "filter routes; choose estimator='gaussian' or omit store_proxies"
        )
    payload = handler(ctx)
    if (
        ctx.k == "auto"
        and ctx.auto_k_config is not None
        and ctx.auto_k_config.k_method == "auto"
        and not payload.selected_features
    ):
        warnings.warn(
            "k='auto' selected 0 features: the routed criterion found no "
            "supportable feature, which is a real answer on noise-like data. "
            "Pass return_result=True and inspect diagnostics_['auto_k'], or "
            "pass an explicit AutoKConfig(k_method=..., min_k=1) for a hard "
            "non-empty floor.",
            UserWarning,
            stacklevel=3,
        )
    return _format_payload(ctx, payload)


def _build_context(spec: FilterSpec, request: FilterRequest) -> FilterContext:
    validate_task(request.task)
    x_shape = request.X.shape if hasattr(request.X, "shape") else np.asarray(request.X).shape
    if len(x_shape) != 2:
        raise ValueError("X must be a 2D feature matrix")
    n_rows, n_features = int(x_shape[0]), int(x_shape[1])
    if request.cache is not None and spec.estimator == "gaussian":
        _validate_gaussian_cache_compatibility(request.X, request.cache, n_rows, n_features)
        _validate_gaussian_cache_overrides(request)
    groups, time = _validate_groups_time(request.groups, request.time, n_rows)
    selector_kwargs = dict(request.selector_kwargs or {})
    if selector_kwargs.get("subsample") is _SUBSAMPLE_DEFAULT:
        selector_kwargs["subsample"] = 50_000
    if selector_kwargs.get("random_state") is _RANDOM_STATE_DEFAULT:
        selector_kwargs["random_state"] = 0
    n_jobs = int(selector_kwargs.get("n_jobs", 1))
    mrmr_backend = resolve_mrmr_backend(
        selector_kwargs.get("mrmr_backend", "auto"),
        n_jobs,
    )
    feature_names = (
        list(request.X.columns)
        if isinstance(request.X, pd.DataFrame)
        else [f"x{i}" for i in range(n_features)]
    )
    conditioning = resolve_conditioning(
        selector_kwargs.get("include"),
        selector_kwargs.get("exclude"),
        selector_kwargs.get("candidates"),
        feature_names=feature_names,
        named=isinstance(request.X, pd.DataFrame),
        k=request.k,
    )
    return FilterContext(
        spec=spec,
        request=request,
        selector_kwargs=selector_kwargs,
        k=validate_k(request.k),
        groups=groups,
        time=time,
        auto_k_config=request.auto_k_config,
        n_rows=n_rows,
        n_features_input=n_features,
        feature_names=feature_names,
        estimator=spec.estimator,
        n_jobs=n_jobs,
        mrmr_backend=mrmr_backend,
        rank_backend="threads" if n_jobs != 1 else "serial",
        conditioning=conditioning,
    )


def _validate_gaussian_cache_compatibility(
    X: XInput,
    cache: FeatureCache,
    n_rows: int,
    n_features: int,
) -> None:
    """Validate a cache against the source matrix before any Gaussian work.

    Named caches are tied to a DataFrame's labels and order.  Caches built from
    arrays are positional and therefore only carry a feature-count contract.
    In both cases the cache must describe the same number of source rows and
    valid original feature positions; otherwise it could return features that
    are not present in ``X``.
    """
    # Validate provenance before interpreting generated-looking names. An old
    # array-built pickle can otherwise be mistaken for a cache whose real
    # DataFrame labels happened to be ``x0``, ``x1``, ... .
    _validate_prebuilt_cache_structure(
        cache,
        original_n_features=n_features,
        n_rows=n_rows,
    )
    _reject_duplicate_feature_names(cache)
    cache_names = getattr(cache, "feature_names", None)
    synthetic = cache_names is None or bool(
        getattr(cache, "feature_names_are_synthetic", False)
    )
    if synthetic:
        if isinstance(X, pd.DataFrame):
            raise ValueError(
                "A cache built from unnamed/positional features requires X to be "
                "the compatible positional ndarray; rebuild the cache from this "
                "DataFrame to establish column names and order"
            )
        if cache_names is not None and len(cache_names) != n_features:
            raise ValueError(
                f"X has {n_features} columns but the positional cache was built from "
                f"{len(cache_names)}"
            )
    else:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "A named Gaussian cache requires X to be a DataFrame with the "
                "same column names and order used to build the cache"
            )
        if list(X.columns) != list(cache_names):
            raise ValueError(
                "X columns do not match cache.feature_names (names and order must "
                "be identical); fit the cache from the same matrix"
            )
        if len(cache_names) != n_features:
            raise ValueError(
                f"X has {n_features} columns but the named cache was built from "
                f"{len(cache_names)}"
            )
def _validate_gaussian_cache_overrides(request: FilterRequest) -> None:
    kwargs = request.selector_kwargs or {}
    if kwargs.get("subsample") is not _SUBSAMPLE_DEFAULT:
        raise ValueError(
            "subsample cannot be passed with a prebuilt cache; leave it omitted "
            "when supplying cache"
        )
    if kwargs.get("random_state") is not _RANDOM_STATE_DEFAULT:
        raise ValueError(
            "random_state controls cache construction and cannot be passed with a "
            "prebuilt cache; use the seed used to build the cache"
        )


def _require_fixed_filter_metadata(ctx: FilterContext) -> None:
    if ctx.k != "auto" and (ctx.groups is not None or ctx.time is not None):
        raise ValueError(
            "groups and time are only meaningful for auto-k evaluation; "
            "use k='auto' or omit them for a fixed-k filter call"
        )


def _format_payload(
    ctx: FilterContext,
    payload: SelectionPayload,
) -> list[str] | FilterSelectionResult:
    assert payload.selected_features is not None
    if not ctx.request.return_result:
        return payload.selected_features

    extra = ctx.spec.metadata_extra(ctx)
    if ctx.k == "auto":
        assert ctx.auto_k_config is not None
        extra.update(
            {
                "auto_k_mode": ctx.auto_k_config.auto_k_mode,
                "k_method": ctx.auto_k_config.k_method,
            }
        )
        if ctx.auto_k_config.k_method in {"evaluate", "xfit_objective", "gaussian_cv"}:
            extra.update(
                {
                    "auto_k_strategy": ctx.auto_k_config.strategy,
                    "selection_rule": ctx.auto_k_config.selection_rule,
                }
            )
    if payload.metadata_extra:
        extra.update(payload.metadata_extra)
    if ctx.conditioning is not None and getattr(ctx.conditioning, "active", False):
        from sift.selection.conditioning import conditioning_record

        discovered_idx = payload.selected_indices
        if discovered_idx is not None and ctx.conditioning.include:
            include_set = set(ctx.conditioning.include)
            discovered_idx = [i for i in discovered_idx if int(i) not in include_set]
        extra["conditioning"] = conditioning_record(
            ctx.conditioning,
            feature_names=ctx.feature_names,
            discovered_idx=discovered_idx,
        )

    metadata = build_selector_metadata(
        ctx.spec.selector,
        k=len(payload.selected_features),
        k_requested="auto" if ctx.k == "auto" else int(ctx.k),
        top_m=payload.top_m,
        n_features=payload.n_features or ctx.n_features_input,
        auto_k=ctx.k == "auto",
        extra=extra,
    )
    result = FilterSelectionResult(
        selected_features=payload.selected_features,
        selected_indices=payload.selected_indices,
        selector_metadata=metadata,
        ranking_=payload.ranking,
        diagnostics_=payload.diagnostics,
    )
    if payload.proxy_correlations is not None:
        object.__setattr__(
            result,
            _PROXY_CORRELATIONS_ATTR,
            payload.proxy_correlations.copy(deep=True),
        )
    return result


def _require_auto_k_eval_context(ctx: FilterContext) -> None:
    config = ctx.auto_k_config
    if ctx.k != "auto" or config is None:
        return
    _require_evaluate_context(config, ctx.groups, ctx.time)
    _require_unique_evaluate_feature_names(config, ctx.request.X)


def _require_context_auto_k_compatibility(ctx: FilterContext) -> None:
    """Contextual target encoding is safe only in split-based evaluation."""
    if (
        (ctx.groups is None and ctx.time is None)
        or (ctx.selector_kwargs or {}).get("cat_encoding") != "target_cv"
    ):
        return
    assert ctx.auto_k_config is not None
    if ctx.auto_k_config.k_method != "evaluate":
        raise ValueError(
            "groups and time are supported only with k='auto' and "
            "AutoKConfig(k_method='evaluate')"
        )


def _require_evaluate_context(
    config: AutoKConfig,
    groups: np.ndarray | None,
    time: np.ndarray | None,
) -> None:
    if config.k_method != "evaluate":
        return
    if config.strategy == "time_holdout" and time is None:
        raise ValueError("auto-k evaluate with strategy='time_holdout' requires time parameter")
    if config.strategy == "group_cv" and groups is None:
        raise ValueError("auto-k evaluate with strategy='group_cv' requires groups parameter")


def _require_unique_evaluate_feature_names(config: AutoKConfig, X: XInput) -> None:
    if config.k_method != "evaluate" or not isinstance(X, pd.DataFrame):
        return
    if X.columns.is_unique:
        return
    duplicates = pd.Index(X.columns[X.columns.duplicated()]).unique().astype(str).tolist()
    sample = duplicates[:5]
    suffix = "..." if len(duplicates) > 5 else ""
    raise ValueError(
        "function-style k='auto' with k_method='evaluate' requires unique "
        "DataFrame column labels because prefix evaluation is name-based. "
        f"Duplicate labels: {sample}{suffix}"
    )


def _select_brier_delegate(request: FilterRequest) -> list[str] | FilterSelectionResult:
    x_shape = request.X.shape if hasattr(request.X, "shape") else np.asarray(request.X).shape
    if len(x_shape) != 2:
        raise ValueError("X must be a 2D feature matrix")
    groups, time = _validate_groups_time(request.groups, request.time, int(x_shape[0]))
    kw = (request.selector_kwargs or {}).get
    k_value = validate_k(request.k)
    if k_value != "auto" and (groups is not None or time is not None):
        raise ValueError(
            "groups and time are only meaningful for auto-k evaluation; "
            "use k='auto' or omit them for a fixed-k filter call"
        )
    if k_value == "auto" and request.auto_k_config is not None:
        # With no explicit config, the delegated select_cefsplus call routes
        # k='auto' through the Auto-K router just like the logloss path.
        auto_k_config = resolve_auto_k_config(request.auto_k_config, time, groups)
        _require_evaluate_context(auto_k_config, groups, time)
        _require_unique_evaluate_feature_names(auto_k_config, request.X)
    options = validate_binary_options(
        request.k,
        loss=kw("loss"),
        top_m=kw("top_m"),
        corr_prune=kw("corr_prune"),
        subsample=kw("subsample"),
        ridge=kw("ridge"),
        refit_every=kw("refit_every"),
        cat_encoding=kw("cat_encoding"),
        loo_smoothing=kw("loo_smoothing"),
        loo_clip_min=kw("loo_clip_min"),
        loo_clip_max=kw("loo_clip_max"),
        sample_weight=request.sample_weight,
        class_weight=kw("class_weight"),
    )
    problem = prepare_binary_problem(
        request.X,
        request.y,
        groups=groups,
        time=time,
        sample_weight=request.sample_weight,
        class_weight=kw("class_weight"),
    )
    cat_encoding_eff = "loo" if kw("cat_encoding") == "loo_logit" else kw("cat_encoding")
    result = select_cefsplus(
        request.X,
        problem.y01.astype(float),
        k=options.k_value,
        groups=problem.groups,
        time=problem.time,
        auto_k_config=request.auto_k_config,
        sample_weight=problem.weights if problem.weighted else None,
        top_m=options.top_m,
        corr_prune=options.corr_prune,
        cat_features=kw("cat_features"),
        cat_encoding=cat_encoding_eff,
        target_cv_n_splits=kw("target_cv_n_splits"),
        target_cv_smoothing=kw("target_cv_smoothing"),
        target_prior=kw("target_prior"),
        warmup_policy=kw("warmup_policy"),
        allow_full_data_target_encoding=kw("allow_full_data_target_encoding"),
        subsample=options.subsample,
        random_state=kw("random_state"),
        verbose=kw("verbose"),
        include=kw("include"),
        exclude=kw("exclude"),
        candidates=kw("candidates"),
        callback=request.callback,
        return_result=request.return_result,
        store_proxies=request.store_proxies,
    )
    if not request.return_result:
        return result

    assert isinstance(result, FilterSelectionResult)
    metadata = dict(result.selector_metadata)
    metadata.update(
        {
            "selector": "cefsplus_binary",
            "loss": "brier",
            "delegate_selector": "cefsplus",
            "weighted": problem.weighted,
            "class_weight": kw("class_weight"),
            "class_weight_scope": "pre_subsample"
            if kw("class_weight") is not None
            else None,
            "target_mapping": problem.target_mapping,
            "cat_encoding": cat_encoding_eff,
        }
    )
    delegated = FilterSelectionResult(
        selected_features=result.selected_features,
        selected_indices=result.selected_indices,
        selector_metadata=metadata,
        ranking_=result.ranking_,
        diagnostics_=result.diagnostics_,
    )
    proxy_correlations = getattr(result, _PROXY_CORRELATIONS_ATTR, None)
    if proxy_correlations is not None:
        object.__setattr__(
            delegated,
            _PROXY_CORRELATIONS_ATTR,
            proxy_correlations.copy(deep=True),
        )
    return delegated


def _validate_groups_time(
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    n_rows: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        if len(groups) != n_rows:
            raise ValueError(f"groups has {len(groups)} elements but X has {n_rows} rows")
    if time is not None:
        time = np.asarray(time).reshape(-1)
        if len(time) != n_rows:
            raise ValueError(f"time has {len(time)} elements but X has {n_rows} rows")
    return groups, time


def _mrmr_spec(request: FilterRequest) -> FilterSpec:
    estimator = str((request.selector_kwargs or {}).get("estimator", "classic"))
    if estimator == "classic":
        return MRMR_CLASSIC_SPEC
    if estimator == "gaussian":
        return MRMR_GAUSSIAN_SPEC
    raise ValueError("estimator must be one of 'classic' or 'gaussian'")


def _jmi_spec(
    request: FilterRequest,
    classic_specs: dict[str, FilterSpec],
    gaussian_spec: FilterSpec,
) -> FilterSpec:
    estimator = resolve_jmi_estimator(
        str((request.selector_kwargs or {}).get("estimator", "auto")),
        request.task,
    )
    if estimator == "gaussian":
        return gaussian_spec
    if estimator in classic_specs:
        return classic_specs[estimator]
    raise ValueError("estimator must be one of 'auto', 'binned', 'r2', 'ksg', or 'gaussian'")


def _classic_spec(
    selector: str,
    display_name: str,
    estimator: str,
    aggregation: str | None,
    path_func,
    validate,
) -> FilterSpec:
    return FilterSpec(
        selector=selector,
        display_name=display_name,
        estimator=estimator,
        fixed_handler=make_fixed_classic(path_func),
        auto_k_handlers={"evaluate": make_auto_classic(path_func)},
        metadata_extra=standard_extra(aggregation),
        validate=validate,
    )


def _gaussian_spec(
    selector: str,
    display_name: str,
    method_func,
    *,
    cefsplus: bool = False,
) -> FilterSpec:
    auto_handlers = {
        "auto": make_auto_gaussian(
            method_func,
            GAUSSIAN_AUTO,
            include_diagnostics=True,
        ),
        "evaluate": make_auto_gaussian(
            method_func,
            GAUSSIAN_EVALUATE,
            include_diagnostics=cefsplus,
        ),
        "elbow": make_auto_gaussian(
            method_func,
            GAUSSIAN_ELBOW,
            include_diagnostics=cefsplus,
        ),
        "xfit_objective": make_auto_gaussian(
            method_func,
            GAUSSIAN_XFIT_OBJECTIVE,
            include_diagnostics=True,
        ),
        "gaussian_cv": make_auto_gaussian(
            method_func,
            GAUSSIAN_CV,
            include_diagnostics=True,
        ),
        "stability": make_auto_gaussian(
            method_func,
            GAUSSIAN_STABILITY,
            include_diagnostics=True,
        ),
    }
    if cefsplus:
        auto_handlers["penalized_objective"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_PENALIZED,
            include_diagnostics=True,
            include_objective_penalty=True,
        )
        auto_handlers["k_posterior"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_POSTERIOR,
            include_diagnostics=True,
        )
        auto_handlers["chi2_stop"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_CHI2,
            include_diagnostics=True,
        )
        auto_handlers["forward_stop"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_FORWARD_STOP,
            include_diagnostics=True,
        )
        auto_handlers["changepoint"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_CHANGEPOINT,
            include_diagnostics=True,
        )
        auto_handlers["perm_gap"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_PERM_GAP,
            include_diagnostics=True,
        )
        auto_handlers["knockoff_path"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_KNOCKOFF,
            include_diagnostics=True,
        )
        auto_handlers["consensus"] = make_auto_gaussian(
            method_func,
            GAUSSIAN_CONSENSUS,
            include_diagnostics=True,
        )
    return FilterSpec(
        selector=selector,
        display_name=display_name,
        estimator="gaussian",
        fixed_handler=make_fixed_gaussian(method_func),
        auto_k_handlers=auto_handlers,
        metadata_extra=no_extra if cefsplus else standard_extra(),
        validate=validate_cefsplus if cefsplus else validate_standard,
    )


_MRMR_CLASSIC_PATH = make_mrmr_classic_path()
MRMR_CLASSIC_SPEC = _classic_spec(
    "mrmr",
    "mRMR",
    "classic",
    None,
    _MRMR_CLASSIC_PATH,
    validate_standard,
)
MRMR_GAUSSIAN_SPEC = _gaussian_spec(
    "mrmr",
    "mRMR",
    mrmr_gaussian_method,
)

JMI_CLASSIC_SPECS = {
    estimator: _classic_spec(
        "jmi",
        "JMI",
        estimator,
        "sum",
        make_jmi_classic_path(
            aggregation="sum",
            pass_sample_weight=estimator != "ksg",
        ),
        validate_ksg_no_weight if estimator == "ksg" else validate_standard,
    )
    for estimator in ("r2", "binned", "ksg")
}
JMI_GAUSSIAN_SPEC = _gaussian_spec(
    "jmi",
    "JMI",
    selector_gaussian_method("jmi"),
)

JMIM_CLASSIC_SPECS = {
    estimator: _classic_spec(
        "jmim",
        "JMIM",
        estimator,
        "min",
        make_jmi_classic_path(
            aggregation="min",
            pass_sample_weight=estimator != "ksg",
        ),
        validate_ksg_no_weight if estimator == "ksg" else validate_standard,
    )
    for estimator in ("r2", "binned", "ksg")
}
JMIM_GAUSSIAN_SPEC = _gaussian_spec(
    "jmim",
    "JMIM",
    selector_gaussian_method("jmim"),
)

CEFSPLUS_SPEC = _gaussian_spec(
    "cefsplus",
    "CEFS+",
    selector_gaussian_method("cefsplus"),
    cefsplus=True,
)

CEFSPLUS_BINARY_SPEC = FilterSpec(
    selector="cefsplus_binary",
    display_name="CEFS+ binary",
    estimator="binary",
    fixed_handler=binary_fixed_payload,
    auto_k_handlers={
        "auto": binary_auto_auto_payload,
        "evaluate": binary_auto_evaluate_payload,
        "elbow": binary_auto_elbow_payload,
        "penalized_objective": binary_auto_penalized_payload,
        "k_posterior": binary_auto_posterior_payload,
        "changepoint": binary_auto_changepoint_payload,
    },
    metadata_extra=no_extra,
)


__all__ = [
    "select_mrmr",
    "select_jmi",
    "select_jmim",
    "select_cefsplus",
    "select_cefsplus_binary",
]
