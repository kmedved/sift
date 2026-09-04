"""Automatic k selection for filter methods."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields as dataclass_fields, replace
import importlib.util
from typing import Any, Iterator, TYPE_CHECKING, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from sklearn.model_selection import GroupKFold

from sift._deprecate import warn_external
from sift._metadata import resolve_row_metadata
from sift._preprocess import (
    LeaveOneOutLogitEncoder,
    TargetCVEncoder,
    ensure_weights,
    suppress_category_encoder_pandas_warnings,
)
from sift.selection.auto_k_core import (
    build_k_grid,
    build_score_curve_diagnostics,
    evaluate_numeric_prefixes,
    resolve_metric,
    split_weights,
    time_holdout_split,
)
from sift.scoring import is_sklearn_scorer, sklearn_scorer_label
from sift.selection.auto_k_options import (
    AUTO_K_OPTION_GROUP_TYPES,
    AutoKCVOptions,
    AutoKExperimentalOptions,
    AutoKKnockoffOptions,
    AutoKObjectiveOptions,
    AutoKPermutationOptions,
    AutoKStabilityOptions,
    AutoKTestOptions,
)

if TYPE_CHECKING:
    from sift.estimators.copula import FeatureCache


@dataclass
class AutoKConfig:
    """Configuration for automatic k selection.

    ``auto_k_mode="prefix_only"`` is the current public behavior: build one
    supervised feature path, then evaluate prefixes of that fixed path. It is
    fast, but it is not an unbiased estimate of a nested selector procedure.
    ``auto_k_mode="nested"`` is implemented by sklearn-style selector classes,
    where each validation split fits its own train-only selector path. The
    function-style selectors still reject nested mode and keep this helper on
    the prefix-only contract.

    Every option is a flat dataclass field; ``k_method`` decides which of them
    is actually read. Setting a field that the chosen ``k_method`` ignores
    emits a ``UserWarning`` naming that field, so a config cannot quietly
    pretend to control something. ``AutoKConfig.from_groups(...)`` accepts the
    frozen option groups in `sift.selection.auto_k_options` and flattens
    them into these same fields; the ``objective``, ``test``, ``perm``,
    ``knockoff``, ``cv``, ``stability``, and ``experimental`` properties are
    read-only snapshots of the corresponding subsets.

    Attributes
    ----------
    k_method : str, default 'evaluate'
        Rule that turns a feature path into a k. One of ``'evaluate'``,
        ``'elbow'``, ``'penalized_objective'``, ``'chi2_stop'``,
        ``'forward_stop'``, ``'perm_gap'``, ``'knockoff_path'``,
        ``'xfit_objective'``, ``'gaussian_cv'``, ``'k_posterior'``,
        ``'stability'``, ``'changepoint'``, ``'consensus'``, or ``'auto'``
        (the measured router, which never warns about unused fields).
    strategy : str, default 'time_holdout'
        Split scheme: ``'time_holdout'``, ``'group_cv'``, or ``'kfold'``.
        Read by ``'evaluate'``, ``'gaussian_cv'``, and ``'xfit_objective'``.
        ``'kfold'`` is rejected for ``k_method='evaluate'``.
    metric : str or sklearn scorer, default 'auto'
        Prefix score for ``k_method='evaluate'``. ``'auto'`` resolves to
        ``'rmse'`` (regression) or ``'logloss'`` (classification); the other
        SIFT names are ``'mae'`` and ``'error'``. Estimator-style sklearn
        scorer objects are negated into the lower-is-better curve.
    max_k : int, default 100
        Largest k any rule may return; positive integer. Every rule clamps it
        to the realized path or curve length.
    min_k : int, default 5
        Smallest k any rule may return; non-negative integer and ``<= max_k``.
        Discovery-flavored rules need ``min_k=0`` to be able to answer "no
        features at all".
    val_frac : float, default 0.2
        Time-holdout validation fraction, in (0, 1). Read by ``'evaluate'``,
        ``'gaussian_cv'``, and ``'xfit_objective'`` under
        ``strategy='time_holdout'``.
    n_splits : int, default 5
        Fold count for ``k_method='evaluate'`` with ``strategy='group_cv'``.
        Deliberately distinct from ``xfit_folds``.
    random_state : int, default 42
        Seed for the resampling rules (``'perm_gap'``, ``'knockoff_path'``,
        ``'stability'``) and for the shuffled ``strategy='kfold'`` splits used
        by the cross-fitted rules.
    elbow_min_rel_gain : float, default 0.02
        Relative-gain threshold for ``k_method='elbow'``; finite and >= 0.
    elbow_patience : int, default 3
        Consecutive small-gain steps ``k_method='elbow'`` requires before it
        stops; positive integer.
    auto_k_mode : str, default 'prefix_only'
        ``'prefix_only'`` everywhere, or ``'nested'`` on selector classes with
        ``k_method='evaluate'``. Function-style selectors raise
        ``NotImplementedError`` for ``'nested'``.
    selection_rule : str, default 'best'
        Score-curve rule: ``'best'``, ``'one_se'``, ``'plateau'``, or
        ``'tolerance'``. Read by ``'evaluate'``, ``'gaussian_cv'``, and
        ``'xfit_objective'``.
    one_se_multiplier : float, default 1.0
        Standard-error multiple used by ``selection_rule='one_se'``; positive
        and finite.
    score_abs_tol : float or None, default None
        Absolute score slack for ``selection_rule`` in
        ``{'plateau', 'tolerance'}``; None, or finite and >= 0.
    score_rel_tol : float or None, default None
        Relative score slack for the same two rules. ``k_method='evaluate'``
        requires at least one of the two tolerances for those rules.
    plateau_prefer : str, default 'smallest'
        Which plateau member ``selection_rule='plateau'`` returns:
        ``'smallest'``, ``'center'``, ``'best'``, or ``'largest'``.
    plateau_min_points : int, default 2
        Minimum plateau width before ``selection_rule='plateau'`` prefers a
        plateau member over the raw best k; positive integer.
    objective_penalty : str, default 'bic'
        Penalty family for ``k_method='penalized_objective'``: ``'bic'``,
        ``'mdl'``, ``'aic'``, ``'hqc'``, ``'custom'``, ``'ebic'``, or
        ``'ric'``. ``'ebic'`` and ``'ric'`` additionally require
        ``n_candidates`` at call time.
    objective_penalty_weight : float or None, default None
        Per-degree-of-freedom penalty weight. Required by, and valid only
        with, ``objective_penalty='custom'``; finite and >= 0.
    objective_n_eff : float or None, default None
        Explicit effective sample size that overrides ``n_eff_mode`` for the
        objective, posterior, gain-test, and changepoint rules. None, or
        finite and > 1 (> e for ``objective_penalty='hqc'``).
    binary_objective_mode : str, default 'refit'
        Binary CEFS+ objective source: ``'refit'`` log-likelihood gains or
        ``'score_test'``. Read by direct ``'penalized_objective'`` and
        ``'k_posterior'`` runs.
    n_eff_mode : str or float, default 'auto'
        Effective sample size rule: ``'auto'``, ``'kish'``, ``'weight_sum'``,
        or a finite float > 1. ``'auto'`` means Kish ``(sum w)^2 / sum w^2``
        for the Auto-K v2 rules and for EBIC/RIC, and the weight sum
        otherwise.
    alpha : float, default 0.05
        Level in (0, 1) for ``'chi2_stop'``, for ``'forward_stop'`` (read as
        a ForwardStop FDR level), and for ``'perm_gap'`` with
        ``gap_rule='gain_envelope'``.
    m_mode : str, default 'all'
        Multiplicity count for the path gain tests: ``'all'`` (``p - t + 1``,
        conservative), ``'panel'``, or ``'li_ji'``. Read by ``'chi2_stop'``
        and ``'forward_stop'``.
    stop_patience : int, default 2
        Consecutive non-significant steps required before stopping in
        ``'chi2_stop'`` and in ``'perm_gap'`` with
        ``gap_rule='gain_envelope'``; reused by ``'changepoint'`` as the
        median-smoothing width when > 2.
    perm_B : int, default 20
        Number of permutation null paths for ``k_method='perm_gap'``.
    perm_null : str, default 'auto'
        Permutation null: ``'auto'`` (time -> circular shift, else groups ->
        within-group, else plain permutation), ``'permute'``,
        ``'circular_shift'`` (requires ``time``), or ``'within_group'``
        (requires ``groups``).
    gap_rule : str, default 'tibshirani'
        Gap-curve rule for ``'perm_gap'``: ``'tibshirani'``, ``'argmax'``, or
        ``'gain_envelope'``.
    knockoff_q : float, default 0.2
        Target FDR in (0, 1) for ``k_method='knockoff_path'``.
    knockoff_draws : int, default 1
        Number of knockoff draws. Values > 1 aggregate draws by selection
        frequency >= 0.5, which is not itself FDR-controlled.
    knockoff_s_method : str, default 'equi'
        Knockoff s-vector construction: ``'equi'``, ``'mvr'``, or ``'me'``.
    knockoff_return : str, default 'set'
        ``'set'`` keeps the selected originals; ``'prefix'`` asks the
        orchestrator for a plain CEFS+ prefix of the same length, which does
        not inherit the q guarantee.
    xfit_folds : int, default 5
        Fold count for ``'gaussian_cv'`` and ``'xfit_objective'``.
    xfit_mode : str, default 'shared_z'
        ``'shared_z'`` re-standardizes the shared marginal ranks inside each
        fold; ``'exact'`` requires fold-local cache rebuilding and is
        rejected by the function-style cache orchestration.
    xfit_ridge : float, default 1e-3
        Ridge added to the fold-train correlation matrix in
        ``k_method='gaussian_cv'``; finite and >= 0.
    ebic_gamma : str or float, default 'auto'
        EBIC multiplicity weight, read by ``objective_penalty='ebic'`` and by
        ``k_method='k_posterior'``. ``'auto'`` uses the Chen-Chen threshold
        ``min(1, max(0, 1 - log(n_eff) / (2 log n_candidates)))``; otherwise a
        float in [0, 1].
    posterior_level : float, default 0.9
        Highest-posterior-density mass in (0, 1) for ``'k_posterior'``.
    posterior_pick : str, default 'map'
        ``'map'`` for the posterior mode, or ``'smallest_in_hpd'`` for the
        parsimonious end of the credible set.
    boot_B : int, default 30
        Bootstrap replicates for ``k_method='stability'``.
    boot_mode : str, default 'bayes'
        ``'bayes'`` (Exp(1) reweighting of every row) or ``'half'`` (uniform
        half-sampling without replacement).
    stability_rule : str, default 'max_one_se'
        ``'max_one_se'`` (largest k within one jackknife SE of peak
        stability) or ``'pi_threshold'`` (count features above
        ``stability_pi``).
    stability_pi : float, default 0.6
        Selection-frequency threshold in (0.5, 1], read only by
        ``stability_rule='pi_threshold'``.
    floor_z : float, default 2.5
        Noise-floor sigma multiple for ``k_method='changepoint'``; positive
        and finite.
    floor_window : float or int, default 0.2
        Tail window used to estimate the changepoint noise floor: a fraction
        in (0, 0.5] of the evaluated path, or an integer count >= 5.
    consensus_methods : tuple of str, default four-rule tuple
        Member rules for ``k_method='consensus'``; defaults to
        ``('ebic', 'chi2_stop', 'perm_gap', 'gaussian_cv')``. Non-empty tuple
        drawn from ``'ebic'``, ``'ric'``, ``'posterior'``, ``'k_posterior'``,
        ``'chi2_stop'``, ``'forward_stop'``, ``'changepoint'``,
        ``'perm_gap'``, ``'gaussian_cv'``, ``'xfit_objective'``, and
        ``'stability'``.
    auto_dense_check : bool, default False
        Opt-in dense-regime cross-check for ``k_method='auto'`` on Gaussian
        CEFS+. Non-default ``auto_dense_*`` values are rejected by binary
        log-loss CEFS+.
    auto_dense_min_k : int, default 100
        Selected-k count above which the dense check runs; non-negative.
    auto_dense_min_frac : float, default 0.25
        Fraction of the effective max k above which the dense check runs; in
        [0, 1].
    auto_dense_disagreement_ratio : float, default 2.0
        Ratio between the EBIC pick and the Gaussian-CV cross-check pick that
        triggers the dense-regime warning; finite and > 1.

    See Also
    --------
    validate_auto_k_config : Runtime validation applied by every rule.
    select_k_auto : Prefix evaluation for ``k_method='evaluate'``.
    select_k_penalized_objective : Penalized-objective rule.
    select_k_gaussian_cv : Closed-form cross-validated risk rule.

    Notes
    -----
    The intent presets construct validated configs directly:
    ``AutoKConfig.default()`` (router), ``AutoKConfig.predictive(...)``
    (``gaussian_cv``), ``AutoKConfig.discovery(alpha)`` (``chi2_stop`` with
    ``min_k=0``), and ``AutoKConfig.downstream(strategy, metric, rule)``
    (``evaluate``). ``predictive(n_folds=...)`` maps to ``xfit_folds`` only.
    Router branches override method-specific floors (``min_k=0`` for EBIC and
    permutation-gap stops, at least 1 for Gaussian CV curves), so pass an
    explicit ``k_method`` when a hard ``min_k`` matters.

    Examples
    --------
    >>> from sift import AutoKConfig
    >>> config = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=50)
    >>> config.k_method, config.alpha, config.m_mode
    ('chi2_stop', 0.05, 'all')
    >>> AutoKConfig.discovery(alpha=0.01).min_k
    0
    >>> AutoKConfig.predictive(strategy="kfold", n_folds=3).xfit_folds
    3
    """

    k_method: Literal[
        "evaluate",
        "elbow",
        "penalized_objective",
        "chi2_stop",
        "forward_stop",
        "perm_gap",
        "knockoff_path",
        "xfit_objective",
        "gaussian_cv",
        "k_posterior",
        "stability",
        "changepoint",
        "consensus",
        "auto",
    ] = "evaluate"
    strategy: Literal["time_holdout", "group_cv", "kfold"] = "time_holdout"
    metric: Any = "auto"
    max_k: int = 100
    min_k: int = 5
    val_frac: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    elbow_min_rel_gain: float = 0.02
    elbow_patience: int = 3
    auto_k_mode: Literal["prefix_only", "nested"] = "prefix_only"
    selection_rule: Literal["best", "one_se", "plateau", "tolerance"] = "best"
    one_se_multiplier: float = 1.0
    score_abs_tol: float | None = None
    score_rel_tol: float | None = None
    plateau_prefer: Literal["smallest", "center", "best", "largest"] = "smallest"
    plateau_min_points: int = 2
    objective_penalty: Literal["bic", "mdl", "aic", "hqc", "custom", "ebic", "ric"] = "bic"
    objective_penalty_weight: float | None = None
    objective_n_eff: float | None = None
    binary_objective_mode: Literal["refit", "score_test"] = "refit"
    n_eff_mode: Literal["auto", "kish", "weight_sum"] | float = "auto"
    alpha: float = 0.05
    m_mode: Literal["all", "panel", "li_ji"] = "all"
    stop_patience: int = 2
    perm_B: int = 20
    perm_null: Literal["auto", "permute", "circular_shift", "within_group"] = "auto"
    gap_rule: Literal["tibshirani", "argmax", "gain_envelope"] = "tibshirani"
    knockoff_q: float = 0.2
    knockoff_draws: int = 1
    knockoff_s_method: Literal["equi", "mvr", "me"] = "equi"
    knockoff_return: Literal["set", "prefix"] = "set"
    xfit_folds: int = 5
    xfit_mode: Literal["shared_z", "exact"] = "shared_z"
    xfit_ridge: float = 1e-3
    ebic_gamma: Literal["auto"] | float = "auto"
    posterior_level: float = 0.9
    posterior_pick: Literal["map", "smallest_in_hpd"] = "map"
    boot_B: int = 30
    boot_mode: Literal["bayes", "half"] = "bayes"
    stability_rule: Literal["max_one_se", "pi_threshold"] = "max_one_se"
    stability_pi: float = 0.6
    floor_z: float = 2.5
    floor_window: float | int = 0.2
    consensus_methods: tuple[str, ...] = ("ebic", "chi2_stop", "perm_gap", "gaussian_cv")
    auto_dense_check: bool = False
    auto_dense_min_k: int = 100
    auto_dense_min_frac: float = 0.25
    auto_dense_disagreement_ratio: float = 2.0

    @classmethod
    def default(cls) -> AutoKConfig:
        """Return the measured automatic router preset."""
        return cls._validated(k_method="auto")

    @classmethod
    def predictive(
        cls,
        strategy: Literal["time_holdout", "group_cv", "kfold"] = "kfold",
        rule: Literal["best", "one_se", "plateau", "tolerance"] = "best",
        n_folds: int = 5,
    ) -> AutoKConfig:
        """Return the closed-form Gaussian predictive-risk preset."""
        return cls._validated(
            k_method="gaussian_cv",
            strategy=strategy,
            selection_rule=rule,
            xfit_folds=n_folds,
        )

    @classmethod
    def discovery(cls, alpha: float = 0.05) -> AutoKConfig:
        """Return the calibrated no-signal discovery-stop preset."""
        return cls._validated(k_method="chi2_stop", min_k=0, alpha=alpha)

    @classmethod
    def downstream(
        cls,
        strategy: Literal["time_holdout", "group_cv"],
        metric: Any,
        rule: Literal["best", "one_se", "plateau", "tolerance"],
    ) -> AutoKConfig:
        """Return the downstream-model prefix-evaluation preset."""
        return cls._validated(
            k_method="evaluate",
            strategy=strategy,
            metric=metric,
            selection_rule=rule,
        )

    @classmethod
    def from_groups(
        cls,
        *,
        objective: AutoKObjectiveOptions | None = None,
        test: AutoKTestOptions | None = None,
        perm: AutoKPermutationOptions | None = None,
        knockoff: AutoKKnockoffOptions | None = None,
        cv: AutoKCVOptions | None = None,
        stability: AutoKStabilityOptions | None = None,
        experimental: AutoKExperimentalOptions | None = None,
        **flat_fields: Any,
    ) -> AutoKConfig:
        """Flatten immutable option groups into the canonical flat config."""
        known_fields = {field.name for field in dataclass_fields(cls)}
        unknown = sorted(set(flat_fields) - known_fields)
        if unknown:
            raise TypeError(f"Unknown AutoKConfig field(s): {unknown}")

        groups = {
            "objective": objective,
            "test": test,
            "perm": perm,
            "knockoff": knockoff,
            "cv": cv,
            "stability": stability,
            "experimental": experimental,
        }
        flattened: dict[str, Any] = {}
        for group_name, group in groups.items():
            if group is None:
                continue
            expected_type = AUTO_K_OPTION_GROUP_TYPES[group_name]
            if not isinstance(group, expected_type):
                raise TypeError(
                    f"{group_name} must be {expected_type.__name__}, "
                    f"got {type(group).__name__}"
                )
            for field in dataclass_fields(group):
                if field.name in flat_fields:
                    raise ValueError(
                        f"AutoKConfig field {field.name!r} was supplied both "
                        f"through {group_name} and as a flat keyword"
                    )
                flattened[field.name] = getattr(group, field.name)

        return cls._validated(**flat_fields, **flattened)

    @classmethod
    def _validated(cls, **values: Any) -> AutoKConfig:
        config = cls(**values)
        validate_auto_k_config(config)
        return config

    @property
    def objective(self) -> AutoKObjectiveOptions:
        return self._group_view(AutoKObjectiveOptions)

    @property
    def test(self) -> AutoKTestOptions:
        return self._group_view(AutoKTestOptions)

    @property
    def perm(self) -> AutoKPermutationOptions:
        return self._group_view(AutoKPermutationOptions)

    @property
    def knockoff(self) -> AutoKKnockoffOptions:
        return self._group_view(AutoKKnockoffOptions)

    @property
    def cv(self) -> AutoKCVOptions:
        return self._group_view(AutoKCVOptions)

    @property
    def stability(self) -> AutoKStabilityOptions:
        return self._group_view(AutoKStabilityOptions)

    @property
    def experimental(self) -> AutoKExperimentalOptions:
        return self._group_view(AutoKExperimentalOptions)

    def _group_view(self, group_type):
        return group_type(
            **{
                field.name: getattr(self, field.name)
                for field in dataclass_fields(group_type)
            }
        )


_VALID_K_METHODS = frozenset(
    {
        "evaluate",
        "elbow",
        "penalized_objective",
        "chi2_stop",
        "forward_stop",
        "perm_gap",
        "knockoff_path",
        "xfit_objective",
        "gaussian_cv",
        "k_posterior",
        "stability",
        "changepoint",
        "consensus",
        "auto",
    }
)
_VALID_STRATEGIES = frozenset({"time_holdout", "group_cv", "kfold"})
_VALID_SELECTION_RULES = frozenset({"best", "one_se", "plateau", "tolerance"})
_VALID_PLATEAU_PREFERS = frozenset({"smallest", "center", "best", "largest"})
_VALID_OBJECTIVE_PENALTIES = frozenset({"bic", "mdl", "aic", "hqc", "custom", "ebic", "ric"})
_VALID_BINARY_OBJECTIVE_MODES = frozenset({"refit", "score_test"})
_POSITIVE_INT_FIELDS = (
    "max_k",
    "n_splits",
    "elbow_patience",
    "plateau_min_points",
    "stop_patience",
    "perm_B",
    "knockoff_draws",
    "xfit_folds",
    "boot_B",
)
_NONNEGATIVE_INT_FIELDS = ("min_k",)
_VALID_N_EFF_MODES = frozenset({"auto", "kish", "weight_sum"})
_VALID_M_MODES = frozenset({"all", "panel", "li_ji"})
_VALID_PERM_NULLS = frozenset({"auto", "permute", "circular_shift", "within_group"})
_VALID_GAP_RULES = frozenset({"tibshirani", "argmax", "gain_envelope"})
_VALID_KNOCKOFF_S_METHODS = frozenset({"equi", "mvr", "me"})
_VALID_KNOCKOFF_RETURNS = frozenset({"set", "prefix"})
_VALID_XFIT_MODES = frozenset({"shared_z", "exact"})
_VALID_POSTERIOR_PICKS = frozenset({"map", "smallest_in_hpd"})
_VALID_BOOT_MODES = frozenset({"bayes", "half"})
_VALID_STABILITY_RULES = frozenset({"max_one_se", "pi_threshold"})
_VALID_CONSENSUS_METHODS = frozenset(
    {
        "ebic",
        "ric",
        "posterior",
        "k_posterior",
        "chi2_stop",
        "forward_stop",
        "changepoint",
        "perm_gap",
        "gaussian_cv",
        "xfit_objective",
        "stability",
    }
)
_DEFAULT_AUTOK_CONFIG = None
_REAL_TYPES = (int, float, np.integer, np.floating)
_WARN_UNUSED_METHOD_FIELDS = ContextVar(
    "sift_warn_unused_auto_k_method_fields",
    default=True,
)


@contextmanager
def _suppress_auto_k_unused_field_warnings() -> Iterator[None]:
    """Suppress warnings while an already-validated config is routed internally."""
    token = _WARN_UNUSED_METHOD_FIELDS.set(False)
    try:
        yield
    finally:
        _WARN_UNUSED_METHOD_FIELDS.reset(token)


def _is_real_number(value) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, _REAL_TYPES)


def validate_auto_k_config(
    config: AutoKConfig,
    *,
    warn_unused: bool = True,
) -> None:
    """Validate runtime values on an AutoKConfig instance."""
    if config.k_method not in _VALID_K_METHODS:
        raise ValueError(
            "AutoKConfig.k_method must be one of "
            f"{sorted(_VALID_K_METHODS)}; got {config.k_method!r}"
        )

    if config.strategy not in _VALID_STRATEGIES:
        raise ValueError(
            "AutoKConfig.strategy must be one of "
            f"{sorted(_VALID_STRATEGIES)}; got {config.strategy!r}"
        )
    if config.k_method == "evaluate" and config.strategy == "kfold":
        raise ValueError(
            "AutoKConfig.strategy='kfold' is only supported by gaussian_cv and "
            "xfit_objective; use time_holdout or group_cv for k_method='evaluate'"
        )

    for name in _POSITIVE_INT_FIELDS:
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"AutoKConfig.{name} must be a positive integer")
    for name in _NONNEGATIVE_INT_FIELDS:
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ValueError(f"AutoKConfig.{name} must be a non-negative integer")

    if int(config.min_k) > int(config.max_k):
        raise ValueError("AutoKConfig.min_k must be <= AutoKConfig.max_k")
    if not isinstance(config.auto_dense_check, (bool, np.bool_)):
        raise ValueError("AutoKConfig.auto_dense_check must be boolean")
    if (
        isinstance(config.auto_dense_min_k, (bool, np.bool_))
        or not isinstance(config.auto_dense_min_k, (int, np.integer))
        or int(config.auto_dense_min_k) < 0
    ):
        raise ValueError("AutoKConfig.auto_dense_min_k must be a non-negative integer")

    if (
        not _is_real_number(config.val_frac)
        or not np.isfinite(config.val_frac)
        or not 0.0 < float(config.val_frac) < 1.0
    ):
        raise ValueError("AutoKConfig.val_frac must be finite and between 0 and 1")

    if (
        not _is_real_number(config.elbow_min_rel_gain)
        or not np.isfinite(config.elbow_min_rel_gain)
        or float(config.elbow_min_rel_gain) < 0.0
    ):
        raise ValueError("AutoKConfig.elbow_min_rel_gain must be finite and non-negative")

    if config.selection_rule not in _VALID_SELECTION_RULES:
        raise ValueError(
            "AutoKConfig.selection_rule must be one of "
            f"{sorted(_VALID_SELECTION_RULES)}; got {config.selection_rule!r}"
        )
    if config.plateau_prefer not in _VALID_PLATEAU_PREFERS:
        raise ValueError(
            "AutoKConfig.plateau_prefer must be one of "
            f"{sorted(_VALID_PLATEAU_PREFERS)}; got {config.plateau_prefer!r}"
        )
    if (
        not _is_real_number(config.one_se_multiplier)
        or not np.isfinite(config.one_se_multiplier)
        or float(config.one_se_multiplier) <= 0.0
    ):
        raise ValueError("AutoKConfig.one_se_multiplier must be positive and finite")
    for name in ("score_abs_tol", "score_rel_tol"):
        value = getattr(config, name)
        if value is not None and (
            not _is_real_number(value) or not np.isfinite(value) or float(value) < 0.0
        ):
            raise ValueError(f"AutoKConfig.{name} must be None or finite and non-negative")
    if (
        config.k_method == "evaluate"
        and config.selection_rule in {"plateau", "tolerance"}
        and config.score_abs_tol is None
        and config.score_rel_tol is None
    ):
        raise ValueError(
            "selection_rule='plateau' or 'tolerance' requires score_abs_tol or score_rel_tol"
        )

    if config.n_eff_mode not in _VALID_N_EFF_MODES and (
        not _is_real_number(config.n_eff_mode)
        or not np.isfinite(config.n_eff_mode)
        or float(config.n_eff_mode) <= 1.0
    ):
        raise ValueError(
            "AutoKConfig.n_eff_mode must be 'auto', 'kish', 'weight_sum', or a finite float > 1"
        )

    if config.m_mode not in _VALID_M_MODES:
        raise ValueError(
            "AutoKConfig.m_mode must be one of "
            f"{sorted(_VALID_M_MODES)}; got {config.m_mode!r}"
        )
    if config.perm_null not in _VALID_PERM_NULLS:
        raise ValueError(
            "AutoKConfig.perm_null must be one of "
            f"{sorted(_VALID_PERM_NULLS)}; got {config.perm_null!r}"
        )
    if config.gap_rule not in _VALID_GAP_RULES:
        raise ValueError(
            "AutoKConfig.gap_rule must be one of "
            f"{sorted(_VALID_GAP_RULES)}; got {config.gap_rule!r}"
        )
    if config.knockoff_s_method not in _VALID_KNOCKOFF_S_METHODS:
        raise ValueError(
            "AutoKConfig.knockoff_s_method must be one of "
            f"{sorted(_VALID_KNOCKOFF_S_METHODS)}; got {config.knockoff_s_method!r}"
        )
    if config.knockoff_return not in _VALID_KNOCKOFF_RETURNS:
        raise ValueError(
            "AutoKConfig.knockoff_return must be one of "
            f"{sorted(_VALID_KNOCKOFF_RETURNS)}; got {config.knockoff_return!r}"
        )
    if config.xfit_mode not in _VALID_XFIT_MODES:
        raise ValueError(
            "AutoKConfig.xfit_mode must be one of "
            f"{sorted(_VALID_XFIT_MODES)}; got {config.xfit_mode!r}"
        )
    if config.posterior_pick not in _VALID_POSTERIOR_PICKS:
        raise ValueError(
            "AutoKConfig.posterior_pick must be one of "
            f"{sorted(_VALID_POSTERIOR_PICKS)}; got {config.posterior_pick!r}"
        )
    if config.boot_mode not in _VALID_BOOT_MODES:
        raise ValueError(
            "AutoKConfig.boot_mode must be one of "
            f"{sorted(_VALID_BOOT_MODES)}; got {config.boot_mode!r}"
        )
    if config.stability_rule not in _VALID_STABILITY_RULES:
        raise ValueError(
            "AutoKConfig.stability_rule must be one of "
            f"{sorted(_VALID_STABILITY_RULES)}; got {config.stability_rule!r}"
        )

    for name in ("alpha", "knockoff_q", "posterior_level"):
        value = getattr(config, name)
        if (
            not _is_real_number(value)
            or not np.isfinite(value)
            or not 0.0 < float(value) < 1.0
        ):
            raise ValueError(f"AutoKConfig.{name} must be finite and between 0 and 1")
    if (
        not _is_real_number(config.stability_pi)
        or not np.isfinite(config.stability_pi)
        or not 0.5 < float(config.stability_pi) <= 1.0
    ):
        raise ValueError("AutoKConfig.stability_pi must be finite and in (0.5, 1]")
    if (
        not _is_real_number(config.xfit_ridge)
        or not np.isfinite(config.xfit_ridge)
        or float(config.xfit_ridge) < 0.0
    ):
        raise ValueError("AutoKConfig.xfit_ridge must be finite and non-negative")
    if (
        not _is_real_number(config.floor_z)
        or not np.isfinite(config.floor_z)
        or float(config.floor_z) <= 0.0
    ):
        raise ValueError("AutoKConfig.floor_z must be positive and finite")
    if not _is_real_number(config.floor_window) or not np.isfinite(config.floor_window):
        raise ValueError("AutoKConfig.floor_window must be finite")
    if isinstance(config.floor_window, (int, np.integer)):
        if int(config.floor_window) < 5:
            raise ValueError("AutoKConfig.floor_window as an integer must be >= 5")
    elif not 0.0 < float(config.floor_window) <= 0.5:
        raise ValueError("AutoKConfig.floor_window as a fraction must be in (0, 0.5]")
    if config.ebic_gamma != "auto" and (
        not _is_real_number(config.ebic_gamma)
        or not np.isfinite(config.ebic_gamma)
        or not 0.0 <= float(config.ebic_gamma) <= 1.0
    ):
        raise ValueError("AutoKConfig.ebic_gamma must be 'auto' or finite in [0, 1]")
    if not isinstance(config.consensus_methods, tuple) or not config.consensus_methods:
        raise ValueError("AutoKConfig.consensus_methods must be a non-empty tuple")
    if not all(isinstance(method, str) and method for method in config.consensus_methods):
        raise ValueError("AutoKConfig.consensus_methods must contain non-empty strings")
    unknown_consensus = [
        method
        for method in config.consensus_methods
        if method.lower() not in _VALID_CONSENSUS_METHODS
    ]
    if unknown_consensus:
        raise ValueError(
            "AutoKConfig.consensus_methods contains unsupported method(s): "
            f"{unknown_consensus}; supported methods are {sorted(_VALID_CONSENSUS_METHODS)}"
        )

    if config.objective_penalty not in _VALID_OBJECTIVE_PENALTIES:
        raise ValueError(
            "AutoKConfig.objective_penalty must be one of "
            f"{sorted(_VALID_OBJECTIVE_PENALTIES)}; got {config.objective_penalty!r}"
        )
    if config.objective_penalty == "custom":
        if config.objective_penalty_weight is None:
            raise ValueError(
                "AutoKConfig.objective_penalty_weight is required when "
                "objective_penalty='custom'"
            )
        if (
            not _is_real_number(config.objective_penalty_weight)
            or not np.isfinite(config.objective_penalty_weight)
            or float(config.objective_penalty_weight) < 0.0
        ):
            raise ValueError(
                "AutoKConfig.objective_penalty_weight must be finite and non-negative"
            )
    elif config.objective_penalty_weight is not None:
        raise ValueError(
            "AutoKConfig.objective_penalty_weight is only valid when "
            "objective_penalty='custom'"
        )

    if config.objective_n_eff is not None and (
        not _is_real_number(config.objective_n_eff)
        or not np.isfinite(config.objective_n_eff)
        or float(config.objective_n_eff) <= 1.0
    ):
        raise ValueError("AutoKConfig.objective_n_eff must be None or finite and > 1")
    if (
        not _is_real_number(config.auto_dense_min_frac)
        or not np.isfinite(config.auto_dense_min_frac)
        or not 0.0 <= float(config.auto_dense_min_frac) <= 1.0
    ):
        raise ValueError("AutoKConfig.auto_dense_min_frac must be finite and between 0 and 1")
    if (
        not _is_real_number(config.auto_dense_disagreement_ratio)
        or not np.isfinite(config.auto_dense_disagreement_ratio)
        or float(config.auto_dense_disagreement_ratio) <= 1.0
    ):
        raise ValueError("AutoKConfig.auto_dense_disagreement_ratio must be finite and > 1")
    if config.objective_penalty == "hqc" and (
        config.objective_n_eff is not None and float(config.objective_n_eff) <= np.e
    ):
        raise ValueError("AutoKConfig.objective_n_eff must be > e for HQC")

    if config.binary_objective_mode not in _VALID_BINARY_OBJECTIVE_MODES:
        raise ValueError(
            "AutoKConfig.binary_objective_mode must be one of "
            f"{sorted(_VALID_BINARY_OBJECTIVE_MODES)}; got {config.binary_objective_mode!r}"
        )

    if warn_unused and _WARN_UNUSED_METHOD_FIELDS.get():
        _warn_unused_method_fields(config)


def _warn_unused_method_fields(config: AutoKConfig) -> None:
    if config.k_method == "auto":
        return
    global _DEFAULT_AUTOK_CONFIG
    if _DEFAULT_AUTOK_CONFIG is None:
        _DEFAULT_AUTOK_CONFIG = AutoKConfig()
    defaults = _DEFAULT_AUTOK_CONFIG
    used_by = {
        "strategy": {"evaluate", "gaussian_cv", "xfit_objective"},
        "metric": {"evaluate"},
        "val_frac": {"evaluate", "gaussian_cv", "xfit_objective"},
        "n_splits": {"evaluate"},
        "random_state": {
            "perm_gap",
            "knockoff_path",
            "xfit_kfold_split",
            "stability",
        },
        "elbow_min_rel_gain": {"elbow"},
        "elbow_patience": {"elbow"},
        "selection_rule": {"evaluate", "gaussian_cv", "xfit_objective"},
        "one_se_multiplier": {"score_rule_one_se"},
        "score_abs_tol": {"score_rule_plateau", "score_rule_tolerance"},
        "score_rel_tol": {"score_rule_plateau", "score_rule_tolerance"},
        "plateau_prefer": {"score_rule_plateau"},
        "plateau_min_points": {"score_rule_plateau"},
        "objective_penalty": {"direct_penalized_objective"},
        "objective_penalty_weight": {"penalized_custom"},
        "objective_n_eff": {
            "penalized_objective",
            "k_posterior",
            "chi2_stop",
            "forward_stop",
            "changepoint",
        },
        "binary_objective_mode": {
            "direct_penalized_objective",
            "direct_k_posterior",
        },
        "n_eff_mode": {
            "penalized_objective",
            "k_posterior",
            "chi2_stop",
            "forward_stop",
            "changepoint",
        },
        "alpha": {"chi2_stop", "forward_stop", "perm_gain_envelope"},
        "m_mode": {"chi2_stop", "forward_stop"},
        "stop_patience": {
            "chi2_stop",
            "changepoint",
            "perm_gain_envelope",
        },
        "perm_B": {"perm_gap"},
        "perm_null": {"perm_gap"},
        "gap_rule": {"perm_gap"},
        "knockoff_q": {"knockoff_path"},
        "knockoff_draws": {"knockoff_path"},
        "knockoff_s_method": {"knockoff_path"},
        "knockoff_return": {"knockoff_path"},
        "xfit_folds": {"xfit_objective", "gaussian_cv"},
        "xfit_mode": {"xfit_objective", "gaussian_cv"},
        "xfit_ridge": {"gaussian_cv"},
        "ebic_gamma": {"penalized_ebic", "k_posterior"},
        "posterior_level": {"k_posterior"},
        "posterior_pick": {"k_posterior"},
        "boot_B": {"stability"},
        "boot_mode": {"stability"},
        "stability_rule": {"stability"},
        "stability_pi": {"stability_pi_threshold"},
        "floor_z": {"changepoint"},
        "floor_window": {"changepoint"},
        "consensus_methods": {"consensus"},
        "auto_dense_check": {"auto"},
        "auto_dense_min_k": {"auto"},
        "auto_dense_min_frac": {"auto"},
        "auto_dense_disagreement_ratio": {"auto"},
    }
    method_tags = _auto_k_method_tags(config)
    for field_name, methods in used_by.items():
        if method_tags & methods:
            continue
        if getattr(config, field_name) != getattr(defaults, field_name):
            warn_external(
                f"AutoKConfig.{field_name} is set but k_method={config.k_method!r} "
                "does not use it.",
                UserWarning,
            )


def _auto_k_method_tags(config: AutoKConfig) -> set[str]:
    """Return semantic tags used to classify flat fields for warnings."""
    method = config.k_method
    tags = {method}
    if method == "consensus":
        tags = {"consensus"}
        for raw_method in config.consensus_methods:
            lower = raw_method.lower()
            if lower == "ebic":
                tags.update({"penalized_objective", "penalized_ebic"})
            elif lower == "ric":
                tags.update({"penalized_objective", "penalized_ric"})
            elif lower == "posterior":
                tags.add("k_posterior")
            else:
                tags.add(lower)
    elif method == "penalized_objective":
        tags.add("direct_penalized_objective")
        tags.add(f"penalized_{config.objective_penalty}")
    elif method == "k_posterior":
        tags.add("direct_k_posterior")

    score_methods = {"evaluate", "gaussian_cv", "xfit_objective"}
    if tags & score_methods:
        tags.add(f"score_rule_{config.selection_rule}")
    if tags & {"gaussian_cv", "xfit_objective"}:
        tags.add(f"xfit_{config.strategy}_split")
    if "perm_gap" in tags and config.gap_rule == "gain_envelope":
        tags.add("perm_gain_envelope")
    if "stability" in tags and config.stability_rule == "pi_threshold":
        tags.add("stability_pi_threshold")
    return tags


def _ensure_supported_auto_k_mode(
    config: AutoKConfig,
    *,
    allow_nested: bool = False,
    warn_unused: bool = True,
) -> None:
    """Validate path-selection semantics for the current implementation."""
    validate_auto_k_config(config, warn_unused=warn_unused)
    if config.auto_k_mode == "prefix_only":
        return
    if config.auto_k_mode == "nested":
        if allow_nested:
            return
        raise NotImplementedError(
            "AutoKConfig(auto_k_mode='nested') is not implemented yet. "
            "Use auto_k_mode='prefix_only' for the current behavior: build one "
            "supervised feature path on the rows available to the selector, "
            "then evaluate prefixes. This is fast but is not an unbiased "
            "estimate of the full nested selector-plus-k-selection procedure."
        )
    raise ValueError(
        "auto_k_mode must be 'prefix_only' or 'nested'; "
        f"got {config.auto_k_mode!r}"
    )


def with_effective_k_bounds(config: AutoKConfig, *, min_k: int, max_k: int) -> AutoKConfig:
    """Return a config copy with k bounds clamped to an actual feature path."""
    return replace(config, min_k=int(min_k), max_k=int(max_k))


def resolve_auto_k_config(
    auto_k_config: Optional[AutoKConfig],
    time: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    *,
    allow_nested: bool = False,
) -> AutoKConfig:
    """Resolve auto-k config, inferring strategy from supplied split context."""
    if auto_k_config is not None:
        _ensure_supported_auto_k_mode(
            auto_k_config,
            allow_nested=allow_nested,
            warn_unused=False,
        )
        return auto_k_config
    if time is not None:
        config = AutoKConfig(strategy="time_holdout")
        _ensure_supported_auto_k_mode(
            config,
            allow_nested=allow_nested,
            warn_unused=False,
        )
        return config
    if groups is not None:
        config = AutoKConfig(strategy="group_cv")
        _ensure_supported_auto_k_mode(
            config,
            allow_nested=allow_nested,
            warn_unused=False,
        )
        return config
    raise ValueError(
        "k='auto' requires time, groups, or auto_k_config with an explicit "
        "AutoKConfig for a non-evaluate k_method such as 'elbow', "
        "'penalized_objective', 'gaussian_cv', or 'perm_gap'"
    )


def _score_curve_tolerance(best_score: float, config: AutoKConfig) -> float:
    tol = 0.0
    if config.score_abs_tol is not None:
        tol = max(tol, float(config.score_abs_tol))
    if config.score_rel_tol is not None:
        tol = max(tol, abs(best_score) * float(config.score_rel_tol))
    return tol


def _choose_best_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    del best_row, best_score, config, lower_is_better
    diag["within_tolerance"] = diag["k"] == best_k
    return best_k, "best", False


def _choose_one_se_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    best_se = float(best_row.get("score_se", np.nan))
    if not np.isfinite(best_se):
        warnings.warn(
            "selection_rule='one_se' requires at least two finite split scores; "
            "falling back to selection_rule='best'.",
            UserWarning,
            stacklevel=3,
        )
        diag["within_tolerance"] = diag["k"] == best_k
        return best_k, "best", True

    tol = float(config.one_se_multiplier) * best_se
    if lower_is_better:
        diag["within_tolerance"] = diag["score_mean"] <= best_score + tol
    else:
        diag["within_tolerance"] = diag["score_mean"] >= best_score - tol
    eligible = diag[diag["within_tolerance"] & np.isfinite(diag["score_mean"])]
    selected_k = int(eligible.sort_values("k", kind="mergesort").iloc[0]["k"])
    return selected_k, "one_se", False


def _mark_tolerance(
    diag: pd.DataFrame,
    best_score: float,
    config: AutoKConfig,
    *,
    lower_is_better: bool,
) -> None:
    tol = _score_curve_tolerance(best_score, config)
    if lower_is_better:
        diag["within_tolerance"] = diag["score_mean"] <= best_score + tol
    else:
        diag["within_tolerance"] = diag["score_mean"] >= best_score - tol
    diag.loc[~np.isfinite(diag["score_mean"]), "within_tolerance"] = False


def _choose_tolerance_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    del best_row, best_k
    _mark_tolerance(diag, best_score, config, lower_is_better=lower_is_better)
    eligible = diag[diag["within_tolerance"]]
    selected_k = int(eligible.sort_values("k", kind="mergesort").iloc[0]["k"])
    return selected_k, "tolerance", False


def _selected_plateau_ks(diag: pd.DataFrame, best_k: int) -> list[int]:
    eligible_mask = diag["within_tolerance"].to_numpy(dtype=bool)
    best_positions = np.flatnonzero(diag["k"].to_numpy(dtype=int) == best_k)
    if not best_positions.size:
        return [best_k]
    pos = int(best_positions[0])
    start = pos
    while start > 0 and eligible_mask[start - 1]:
        start -= 1
    end = pos
    while end + 1 < len(eligible_mask) and eligible_mask[end + 1]:
        end += 1
    diag.iloc[start : end + 1, diag.columns.get_loc("in_selected_plateau")] = True
    return diag.iloc[start : end + 1]["k"].astype(int).tolist()


def _choose_plateau_rule(diag, best_row, best_k, best_score, config, *, lower_is_better):
    del best_row
    _mark_tolerance(diag, best_score, config, lower_is_better=lower_is_better)
    plateau_ks = _selected_plateau_ks(diag, best_k)
    if len(plateau_ks) < int(config.plateau_min_points):
        selected_k = best_k
    elif config.plateau_prefer == "smallest":
        selected_k = int(plateau_ks[0])
    elif config.plateau_prefer == "largest":
        selected_k = int(plateau_ks[-1])
    elif config.plateau_prefer == "center":
        selected_k = int(plateau_ks[len(plateau_ks) // 2])
    else:
        selected_k = best_k
    return selected_k, "plateau", False


_RULE_SELECTORS = {
    "best": _choose_best_rule,
    "one_se": _choose_one_se_rule,
    "tolerance": _choose_tolerance_rule,
    "plateau": _choose_plateau_rule,
}


def choose_k_from_score_curve(
    diagnostics: pd.DataFrame,
    config: AutoKConfig,
    *,
    lower_is_better: bool = True,
) -> Tuple[int, pd.DataFrame]:
    """Choose k from an evaluated score curve according to AutoKConfig."""
    validate_auto_k_config(config)
    diag = diagnostics.copy()
    if "k" not in diag.columns:
        raise ValueError("score-curve diagnostics must include a 'k' column")
    diag["k"] = diag["k"].astype(int)
    diag = diag[
        (diag["k"] >= int(config.min_k)) & (diag["k"] <= int(config.max_k))
    ].copy()
    diag = diag.sort_values("k", kind="mergesort").reset_index(drop=True)
    if diag.empty:
        return 0, diag
    if "score_mean" not in diag.columns:
        diag["score_mean"] = diag["score"]
    diag["score"] = diag["score_mean"]

    finite = diag[np.isfinite(diag["score_mean"])].copy()
    if finite.empty:
        fallback_k = max(0, int(config.min_k))
        warnings.warn(
            "All candidate score-curve values are non-finite; falling back to "
            f"the method floor k={fallback_k}.",
            UserWarning,
            stacklevel=2,
        )
        diag["best_k"] = fallback_k
        diag["best_score"] = np.inf if lower_is_better else -np.inf
        diag["within_tolerance"] = False
        diag["in_selected_plateau"] = False
        diag["selected"] = diag["k"] == fallback_k
        diag["selection_rule"] = config.selection_rule
        diag["selection_rule_effective"] = config.selection_rule
        diag["one_se_unavailable"] = config.selection_rule == "one_se"
        return fallback_k, diag

    ascending = [lower_is_better, True]
    best_rows = finite.sort_values(["score_mean", "k"], ascending=ascending, kind="mergesort")
    best_row = best_rows.iloc[0]
    best_k = int(best_row["k"])
    best_score = float(best_row["score_mean"])
    rule = config.selection_rule
    effective_rule = rule
    one_se_unavailable = False

    diag["best_k"] = best_k
    diag["best_score"] = best_score
    diag["within_tolerance"] = False
    diag["in_selected_plateau"] = False
    diag["selection_rule"] = rule

    selector = _RULE_SELECTORS.get(rule)
    if selector is None:
        raise ValueError(f"Unknown selection_rule: {rule!r}")
    selected_k, effective_rule, one_se_unavailable = selector(
        diag,
        best_row,
        best_k,
        best_score,
        config,
        lower_is_better=lower_is_better,
    )

    diag["selection_rule_effective"] = effective_rule
    diag["one_se_unavailable"] = one_se_unavailable
    diag["selected"] = diag["k"] == selected_k
    return int(selected_k), diag


def _evaluate_prefix_split(
    *,
    X_path_df: pd.DataFrame,
    valid_features: List[str],
    y_arr: np.ndarray,
    w_arr: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    task: Literal["regression", "classification"],
    metric: object,
    k_grid: list[int],
    sample_weight_supplied: bool,
    cat_features: Optional[List[str]],
    cat_encoding: Literal[
        "none",
        "target_cv",
        "target",
        "loo",
        "james_stein",
        "loo_logit",
    ],
    loo_smoothing: float,
    loo_clip_min: float,
    loo_clip_max: float,
    target_cv_n_splits: int,
    target_cv_smoothing: Literal["auto"] | float,
    target_prior: float | None,
    warmup_policy: Literal["exclude", "zero_weight"],
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    encoding_weight_arr: Optional[np.ndarray],
) -> dict:
    """Evaluate all k values for one train/validation split."""
    Xtr_df = X_path_df.iloc[train_idx]
    Xva_df = X_path_df.iloc[val_idx]
    ytr = y_arr[train_idx]
    yva = y_arr[val_idx]
    wtr = split_weights(w_arr, train_idx, "train")
    wva = split_weights(w_arr, val_idx, "validation")

    if cat_features is None:
        fold_cat = (
            Xtr_df.select_dtypes(include=["object", "category", "string"])
            .columns.intersection(valid_features)
            .tolist()
        )
    else:
        fold_cat = [col for col in cat_features if col in Xtr_df.columns]

    if cat_encoding == "target_cv" and fold_cat:
        enc = TargetCVEncoder(
            fold_cat,
            target_type="binary" if task == "classification" else "continuous",
            smooth=target_cv_smoothing,
            cv=target_cv_n_splits,
            target_prior=target_prior,
            warmup_policy=warmup_policy,
        )
        encoder_kwargs = {}
        if sample_weight_supplied:
            assert encoding_weight_arr is not None
            encoder_kwargs["sample_weight"] = encoding_weight_arr[train_idx]
        if groups is not None:
            encoder_kwargs["groups"] = groups[train_idx]
        if time is not None:
            encoder_kwargs["time"] = time[train_idx]
        Xtr_df = enc.fit_transform(Xtr_df, ytr, **encoder_kwargs)
        Xva_df = enc.transform(Xva_df)
        if enc.effective_sample_weight_ is not None:
            wtr = ensure_weights(
                enc.effective_sample_weight_,
                len(train_idx),
                normalize=True,
            )
    elif cat_encoding == "loo_logit" and fold_cat:
        if task != "classification":
            raise ValueError("cat_encoding='loo_logit' requires task='classification'")
        enc = LeaveOneOutLogitEncoder(
            cols=fold_cat,
            smoothing=loo_smoothing,
            clip_min=loo_clip_min,
            clip_max=loo_clip_max,
        )
        Xtr_df = enc.fit_transform(Xtr_df, ytr, sample_weight=wtr)
        Xva_df = enc.transform(Xva_df)
    elif cat_encoding != "none" and fold_cat:
        if importlib.util.find_spec("category_encoders") is None:
            raise ImportError(
                "cat_encoding requires category_encoders. Install with: pip install category_encoders"
            )
        import category_encoders as ce

        enc_map = {
            "loo": ce.LeaveOneOutEncoder,
            "target": ce.TargetEncoder,
            "james_stein": ce.JamesSteinEncoder,
        }
        Encoder = enc_map[cat_encoding]
        try:
            enc = Encoder(
                cols=fold_cat,
                handle_missing="return_nan",
                handle_unknown="value",
            )
        except TypeError:
            enc = Encoder(cols=fold_cat, handle_missing="return_nan")
        with suppress_category_encoder_pandas_warnings():
            Xtr_df = enc.fit_transform(Xtr_df, ytr)
            Xva_df = enc.transform(Xva_df)

    return evaluate_numeric_prefixes(
        Xtr_df,
        Xva_df,
        ytr,
        yva,
        wtr,
        wva,
        task=task,
        metric=metric,
        k_grid=k_grid,
        ridge_alpha_strategy="full_path",
        sample_weight_supplied=sample_weight_supplied,
    )


def select_k_auto(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_path: List[str],
    config: AutoKConfig,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    task: Literal["regression", "classification"] = "regression",
    cat_encoding: Literal[
        "none",
        "target_cv",
        "target",
        "loo",
        "james_stein",
        "loo_logit",
    ] = "none",
    cat_features: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
) -> Tuple[int, List[str], pd.DataFrame]:
    """Select optimal k by evaluating prefixes of feature_path.

    This is the rule behind ``AutoKConfig(k_method="evaluate")``: build one
    supervised feature path elsewhere, then score its prefixes with a cheap
    proxy model (RidgeCV/Ridge for regression, logistic regression for
    classification) on held-out rows and apply ``config.selection_rule`` to
    the resulting curve. It targets *predictive sufficiency* against a
    concrete downstream metric, so it is the right choice when the question is
    "how many features does my model actually need"; it is the wrong choice
    for discovery, where the curve is flat near the optimum and a support
    recovery rule such as ``chi2_stop`` or a penalized objective is better
    calibrated.

    Parameters
    ----------
    X : DataFrame
        Feature matrix. Must be a pandas DataFrame with unique column labels,
        because ``feature_path`` entries are resolved by name.
    y : ndarray of shape (n_samples,)
        Target, raveled before use.
    feature_path : list of str
        Ordered feature names. Entries missing from ``X.columns`` are dropped;
        the path is truncated to the effective ``max_k``.
    config : AutoKConfig
        Must have ``k_method='evaluate'``. Reads ``strategy``, ``metric``,
        ``min_k``, ``max_k``, ``val_frac`` (time holdout), ``n_splits``
        (group CV), ``selection_rule`` and its tolerance fields, and
        ``auto_k_mode``.
    groups : ndarray of shape (n_samples,) or str, optional
        Group labels, or the name of an ``X`` column holding them (that
        column is then removed from the feature matrix). Required for
        ``strategy='group_cv'`` and forwarded to the
        ``cat_encoding='target_cv'`` encoder when supplied.
    time : ndarray of shape (n_samples,) or str, optional
        Row timestamps, or the name of an ``X`` column holding them (removed
        from the feature matrix as above). Required for
        ``strategy='time_holdout'`` and forwarded to the ``'target_cv'``
        encoder when supplied.
    task : {'regression', 'classification'}, default 'regression'
        Proxy-model family and default metric resolution.
    cat_encoding : str, default 'none'
        Fold-local categorical encoding: ``'none'``, ``'target_cv'``,
        ``'target'``, ``'loo'``, ``'james_stein'``, or ``'loo_logit'``.
        ``'target'``, ``'loo'``, and
        ``'james_stein'`` require the optional ``category_encoders``
        dependency; ``'loo_logit'`` requires ``task='classification'``.
    cat_features : list of str, optional
        Columns to treat as categorical. When None, object/category/string
        columns of the fold-train frame are detected automatically.
    sample_weight : ndarray of shape (n_samples,), optional
        Row weights. Normalized to mean one per split for fitting and scoring;
        the unnormalized copy is handed to the supervised encoders.
    loo_smoothing : float, default 20.0
        Smoothing for ``cat_encoding='loo_logit'``.
    loo_clip_min : float, default 1e-4
        Lower probability clip for ``cat_encoding='loo_logit'``.
    loo_clip_max : float, default 1 - 1e-4
        Upper probability clip for ``cat_encoding='loo_logit'``.
    target_cv_n_splits : int, default 5
        Inner CV folds for ``cat_encoding='target_cv'``.
    target_cv_smoothing : {'auto'} or float, default 'auto'
        Smoothing for ``cat_encoding='target_cv'``.
    target_prior : float or None, default None
        Explicit prior for ``cat_encoding='target_cv'``; None estimates it.
    warmup_policy : {'exclude', 'zero_weight'}, default 'zero_weight'
        How ``cat_encoding='target_cv'`` treats warm-up rows.

    Returns
    -------
    best_k : int
        Selected prefix length, ``0`` when the path resolves to no usable
        feature.
    features : list of str
        The first ``best_k`` names of the resolved path.
    diagnostics : DataFrame
        One row per evaluated k with ``k``, ``score``, ``score_mean``,
        ``score_std``, ``score_se``, ``n_splits``, ``n_finite``,
        ``split_scores``, ``best_k``, ``best_score``, ``within_tolerance``,
        ``in_selected_plateau``, ``selection_rule``,
        ``selection_rule_effective``, ``one_se_unavailable``, and
        ``selected``; plus ``metric`` when an sklearn scorer was used. Empty
        when the path is empty.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'evaluate'``, if ``X`` has duplicate
        column labels, if ``strategy='time_holdout'`` without ``time`` or
        ``strategy='group_cv'`` without ``groups``, if fewer than two groups
        are available, or if ``strategy`` is unknown.
    NotImplementedError
        If ``config.auto_k_mode='nested'``; function-style selectors are
        prefix-only.
    ImportError
        If ``cat_encoding`` needs ``category_encoders`` and it is not
        installed.

    Warns
    -----
    UserWarning
        When every candidate score is non-finite (the method floor is
        returned), and when ``selection_rule='one_se'`` has no usable split
        standard error and falls back to ``'best'``.

    See Also
    --------
    AutoKConfig : Field-by-field description of the options read here.
    choose_k_from_score_curve : Rule engine shared by every curve method.
    select_k_gaussian_cv : Closed-form cross-validated risk, same intent.
    evaluate_feature_path : Explicit k grid with a user-supplied estimator.

    Notes
    -----
    Prefix scores are mildly optimistic: the path is built on all rows,
    including the validation rows, so this is not an unbiased estimate of a
    nested selector. The k grid is dense for small k and sparse afterwards
    (see ``build_k_grid``), and a prefix that fails on some folds is recorded
    with ``score=inf`` so it cannot win on partial coverage. Cost is one
    proxy-model fit per (split, k) -- one split under ``'time_holdout'``, up
    to ``n_splits`` under ``'group_cv'`` -- which makes this the most
    expensive auto-k rule in the library. ``k_method='auto'`` is *not*
    handled here: the router
    lives in `sift.selection.filter_auto_k` and dispatches to a concrete
    rule (EBIC by default for CEFS+), so this function rejects any
    ``k_method`` other than ``'evaluate'``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import AutoKConfig, select_k_auto
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    >>> y = X["a"] + 0.6 * X["b"] + 0.5 * rng.normal(size=80)
    >>> config = AutoKConfig(
    ...     k_method="evaluate", strategy="time_holdout", min_k=1, max_k=5
    ... )
    >>> best_k, features, diag = select_k_auto(
    ...     X, y.to_numpy(), list("abcde"), config, time=np.arange(80)
    ... )
    >>> best_k, features
    (3, ['a', 'b', 'c'])
    >>> print(diag[["k", "selected"]].to_string(index=False))
     k  selected
     1     False
     3      True
     5     False
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
    _ensure_supported_auto_k_mode(config)
    if config.k_method != "evaluate":
        raise ValueError(
            "select_k_auto supports only AutoKConfig(k_method='evaluate'). "
            "Use select_k_elbow(...) or a selector path that explicitly supports "
            "objective-path auto-k."
        )

    if not feature_path:
        return 0, [], pd.DataFrame()
    if isinstance(X, pd.DataFrame) and not X.columns.is_unique:
        duplicates = pd.Index(X.columns[X.columns.duplicated()]).unique().astype(str).tolist()
        sample = duplicates[:5]
        suffix = "..." if len(duplicates) > 5 else ""
        raise ValueError(
            "select_k_auto requires unique DataFrame column labels because "
            "feature_path entries are name-based. "
            f"Duplicate labels: {sample}{suffix}"
        )

    y_arr = np.asarray(y).ravel()
    sample_weight_supplied = sample_weight is not None
    encoding_weight_arr = (
        ensure_weights(sample_weight, len(y_arr), normalize=False)
        if sample_weight_supplied
        else None
    )
    w_arr = ensure_weights(sample_weight, len(y_arr), normalize=True)
    max_k = min(config.max_k, len(feature_path))
    min_k = max(1, min(config.min_k, max_k))

    valid_features = [f for f in feature_path if f in X.columns]
    if not valid_features:
        return 0, [], pd.DataFrame()

    max_k = min(max_k, len(valid_features))
    min_k = max(1, min(config.min_k, max_k))
    valid_features = valid_features[:max_k]
    k_grid = build_k_grid(min_k, max_k)

    X_path_df = X[valid_features]

    metric = resolve_metric(config.metric, task)
    eval_kwargs = {
        "X_path_df": X_path_df,
        "valid_features": valid_features,
        "y_arr": y_arr,
        "w_arr": w_arr,
        "task": task,
        "metric": metric,
        "k_grid": k_grid,
        "sample_weight_supplied": sample_weight_supplied,
        "cat_features": cat_features,
        "cat_encoding": cat_encoding,
        "loo_smoothing": loo_smoothing,
        "loo_clip_min": loo_clip_min,
        "loo_clip_max": loo_clip_max,
        "target_cv_n_splits": target_cv_n_splits,
        "target_cv_smoothing": target_cv_smoothing,
        "target_prior": target_prior,
        "warmup_policy": warmup_policy,
        "groups": groups,
        "time": time,
        "encoding_weight_arr": encoding_weight_arr,
    }

    if config.strategy == "time_holdout":
        if time is None:
            raise ValueError("time_holdout strategy requires time parameter")

        train_idx, val_idx = time_holdout_split(time, config.val_frac)
        scores = _evaluate_prefix_split(
            train_idx=train_idx,
            val_idx=val_idx,
            **eval_kwargs,
        )
        split_scores = {k: [score] for k, score in scores.items()}
        diag = build_score_curve_diagnostics(k_grid, split_scores)

    elif config.strategy == "group_cv":
        if groups is None:
            raise ValueError("group_cv strategy requires groups parameter")

        n_unique = len(np.unique(groups))
        n_splits = min(config.n_splits, n_unique)
        if n_splits < 2:
            raise ValueError(f"group_cv requires at least 2 groups, got {n_unique}")

        gkf = GroupKFold(n_splits=n_splits)

        all_scores = {k: [] for k in k_grid}
        for train_idx, val_idx in gkf.split(X_path_df, y_arr, groups):
            fold_scores = _evaluate_prefix_split(
                train_idx=train_idx,
                val_idx=val_idx,
                **eval_kwargs,
            )
            for k, score in fold_scores.items():
                all_scores[k].append(score)

        diag = build_score_curve_diagnostics(k_grid, all_scores)

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    if diag.empty:
        return max_k, valid_features[:max_k], diag

    curve_config = with_effective_k_bounds(config, min_k=min_k, max_k=max_k)
    best_k, diag = choose_k_from_score_curve(diag, curve_config, lower_is_better=True)
    if is_sklearn_scorer(metric):
        diag["metric"] = sklearn_scorer_label(metric)

    return best_k, valid_features[:best_k], diag


def select_k_elbow(
    objective_path: np.ndarray,
    min_k: int = 5,
    max_k: int = 100,
    min_rel_gain: float = 0.02,
    patience: int = 3,
) -> Tuple[int, pd.DataFrame]:
    """Select the prefix before a patience-confirmed run of small gains.

    ``k`` is the number of retained features. Consequently, the first feature
    in a confirmed low-gain run is excluded from the selected prefix unless
    retaining it is required by the ``min_k`` floor.

    This is the rule behind ``AutoKConfig(k_method="elbow")``, which forwards
    ``elbow_min_rel_gain`` and ``elbow_patience`` here. It reads only the
    in-sample objective curve, so it costs nothing beyond the path itself, but
    its threshold is uncalibrated: gains shrink like ``1/n_eff`` under the
    null while the denominator tracks accumulated signal, so a fixed
    ``min_rel_gain`` means different things at different ``n`` and different
    signal strengths. Treat it as a fast heuristic for a first look and prefer
    a rule that cleared the Auto-K v2 campaign -- ``select_k_chi2_stop``, or
    ``select_k_penalized_objective`` with ``objective_penalty="ebic"`` -- for
    anything load-bearing.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative, non-decreasing objective after each path step, typically
        the CEFS+ objective ``obj[t] = -log(1 - R^2_t) = 2 I(y; S_t)``. Must
        be one-dimensional, numeric, and entirely finite.
    min_k : int, default 5
        Floor on the returned k; non-negative integer, clamped to the
        effective ``max_k``.
    max_k : int, default 100
        Ceiling on the returned k; positive integer, clamped to
        ``len(objective_path)``.
    min_rel_gain : float, default 0.02
        Relative-gain threshold. Step ``k`` counts as small when
        ``(obj[k-1] - obj[k-2]) / max(|obj[k-2]|, 1) < min_rel_gain``. Finite
        and non-negative.
    patience : int, default 3
        Consecutive small-gain steps required to stop; positive integer. On a
        confirmed run starting at step ``k``, the selection is
        ``max(min_k, k - patience)``.

    Returns
    -------
    best_k : int
        Selected prefix length, or ``0`` when the effective ``max_k`` is
        non-positive. Falls back to the effective ``max_k`` when no run of
        ``patience`` small gains is confirmed.
    diagnostics : DataFrame
        One row per evaluated k with ``k`` (1..effective max), ``objective``,
        ``delta`` (step gain, with ``delta[0] = obj[0]``), and ``rel_gain``
        (``inf`` at ``k=1``). Empty when the effective ``max_k`` is
        non-positive.

    Raises
    ------
    ValueError
        If ``objective_path`` is not a one-dimensional numeric array or holds
        a non-finite value; if ``min_k`` is not a non-negative integer, if
        ``max_k`` or ``patience`` is not a positive integer, if
        ``min_k > max_k``, or if ``min_rel_gain`` is not finite and
        non-negative.

    See Also
    --------
    select_k_changepoint : Same shape with an empirical noise floor
        (experimental).
    select_k_chi2_stop : Calibrated sequential test on the same gain path.
    select_k_penalized_objective : Information-criterion stop on the same
        objective path.
    AutoKConfig : ``elbow_min_rel_gain`` and ``elbow_patience`` fields.

    Notes
    -----
    Scanning is ``O(L)`` on top of the path, with no resampling and no model
    fits. Because the scan starts at ``max(min_k, 2)`` and stops at the first
    confirmed run, a single large interior gain resets the counter, which is
    what keeps a masked-then-revealed signal from truncating the path early.
    ``select_k_changepoint`` was written as the calibrated replacement for
    this rule, but it did not clear the campaign's null-calibration gate
    either; both stay available as diagnostics rather than as defaults.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import select_k_elbow
    >>> objective = np.array([1.0, 1.8, 2.4, 2.42, 2.43, 2.44])
    >>> best_k, diag = select_k_elbow(
    ...     objective, min_k=1, max_k=6, min_rel_gain=0.05, patience=2
    ... )
    >>> best_k
    3
    >>> print(diag[["k", "delta", "rel_gain"]].round(3).to_string(index=False))
     k  delta  rel_gain
     1   1.00       inf
     2   0.80     0.800
     3   0.60     0.333
     4   0.02     0.008
     5   0.01     0.004
     6   0.01     0.004
    """
    raw_obj = np.asarray(objective_path)
    if raw_obj.ndim != 1:
        raise ValueError("objective_path must be a one-dimensional numeric array")
    try:
        obj = raw_obj.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("objective_path must be a one-dimensional numeric array") from exc

    for name, value, allow_zero in (
        ("min_k", min_k, True),
        ("max_k", max_k, False),
        ("patience", patience, False),
    ):
        lower = 0 if allow_zero else 1
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < lower
        ):
            qualifier = "a non-negative" if allow_zero else "a positive"
            raise ValueError(f"{name} must be {qualifier} integer")
    if int(min_k) > int(max_k):
        raise ValueError("min_k must be <= max_k")
    if (
        isinstance(min_rel_gain, (bool, np.bool_))
        or not isinstance(min_rel_gain, (int, float, np.integer, np.floating))
        or not np.isfinite(float(min_rel_gain))
        or float(min_rel_gain) < 0.0
    ):
        raise ValueError("min_rel_gain must be finite and non-negative")
    if obj.size and not np.isfinite(obj).all():
        raise ValueError("objective_path must contain only finite values")

    max_k = min(int(max_k), len(obj))

    if max_k <= 0:
        return 0, pd.DataFrame()

    min_k_eff = min(int(min_k), max_k)

    delta = np.zeros_like(obj, dtype=np.float64)
    delta[0] = obj[0]
    delta[1:] = obj[1:] - obj[:-1]

    rel_gain = np.zeros_like(obj, dtype=np.float64)
    rel_gain[0] = np.inf
    denom = np.maximum(np.abs(obj[:-1]), 1.0)
    rel_gain[1:] = delta[1:] / denom

    best_k = max_k
    run = 0

    for k in range(max(min_k_eff, 2), max_k + 1):
        if rel_gain[k - 1] < min_rel_gain:
            run += 1
            if run >= patience:
                best_k = max(min_k_eff, k - int(patience))
                break
        else:
            run = 0

    diag = pd.DataFrame(
        {
            "k": np.arange(1, max_k + 1),
            "objective": obj[:max_k],
            "delta": delta[:max_k],
            "rel_gain": rel_gain[:max_k],
        }
    )

    return best_k, diag


def _resolve_n_eff_mode(config: AutoKConfig) -> str | float:
    mode = config.n_eff_mode
    if mode == "auto":
        v2_methods = {
            "chi2_stop",
            "forward_stop",
            "perm_gap",
            "knockoff_path",
            "xfit_objective",
            "gaussian_cv",
            "k_posterior",
            "stability",
            "changepoint",
            "consensus",
            "auto",
        }
        if config.k_method in v2_methods or config.objective_penalty in {"ebic", "ric"}:
            return "kish"
        return "weight_sum"
    return mode


def _penalty_weight(config: AutoKConfig, n_eff: float) -> float:
    if config.objective_penalty in {"bic", "mdl", "ebic"}:
        return float(np.log(n_eff))
    if config.objective_penalty == "aic":
        return 2.0
    if config.objective_penalty == "hqc":
        if n_eff <= np.e:
            raise ValueError("n_eff must be > e for objective_penalty='hqc'")
        return float(2.0 * np.log(np.log(n_eff)))
    if config.objective_penalty == "custom":
        assert config.objective_penalty_weight is not None
        return float(config.objective_penalty_weight)
    if config.objective_penalty == "ric":
        return 0.0
    raise ValueError(f"Unknown objective_penalty: {config.objective_penalty!r}")


def _log_comb(n: int, k: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k, dtype=np.float64)
    out = gammaln(float(n) + 1.0) - gammaln(k_arr + 1.0) - gammaln(float(n) - k_arr + 1.0)
    out[(k_arr < 0) | (k_arr > n)] = np.inf
    return out


def _resolve_ebic_gamma(config: AutoKConfig, *, n_eff: float, n_candidates: int) -> float:
    if config.ebic_gamma == "auto":
        if n_candidates <= 1:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - np.log(n_eff) / (2.0 * np.log(n_candidates)))))
    return float(config.ebic_gamma)


def _penalty_array(
    config: AutoKConfig,
    ks: np.ndarray,
    *,
    n_eff: float,
    n_candidates: int | None,
) -> tuple[np.ndarray, float, float | None, int | None]:
    penalty_kind = config.objective_penalty
    if penalty_kind in {"ebic", "ric"}:
        if n_candidates is None:
            raise ValueError("n_candidates is required for EBIC/RIC objective penalties")
        n_candidates_int = int(n_candidates)
        if n_candidates_int < 1:
            raise ValueError("n_candidates must be a positive integer")
        if np.max(ks, initial=0) > n_candidates_int:
            raise ValueError("n_candidates must be >= the largest evaluated k")
    else:
        n_candidates_int = None

    if penalty_kind == "ebic":
        gamma = _resolve_ebic_gamma(config, n_eff=n_eff, n_candidates=n_candidates_int)
        penalty = ks.astype(np.float64) * np.log(n_eff) + 2.0 * gamma * _log_comb(n_candidates_int, ks)
        return penalty, float(np.log(n_eff)), gamma, n_candidates_int
    if penalty_kind == "ric":
        gamma = None
        penalty = 2.0 * ks.astype(np.float64) * np.log(float(n_candidates_int))
        return penalty, 2.0 * float(np.log(float(n_candidates_int))), gamma, n_candidates_int

    penalty_weight = _penalty_weight(config, n_eff)
    return penalty_weight * ks.astype(np.float64), penalty_weight, None, n_candidates_int


def _objective_weight_diagnostics(
    sample_weight: Optional[np.ndarray],
    n_samples: int,
    config: AutoKConfig,
) -> tuple[np.ndarray, float, float, float, str]:
    w = ensure_weights(sample_weight, n_samples, normalize=True)
    weight_sum = float(np.sum(w))
    sum_sq = float(np.sum(w * w))
    kish_n_eff = float(weight_sum * weight_sum / sum_sq) if sum_sq > 0.0 else float("nan")
    if config.objective_n_eff is not None:
        n_eff = float(config.objective_n_eff)
        n_eff_source = "objective_n_eff"
    else:
        mode = _resolve_n_eff_mode(config)
        if mode == "kish":
            n_eff = kish_n_eff
            n_eff_source = "kish"
        elif mode == "weight_sum":
            n_eff = weight_sum
            n_eff_source = "selector_weight_sum"
        else:
            n_eff = float(mode)
            n_eff_source = "n_eff_mode"
    if n_eff <= 1.0 or not np.isfinite(n_eff):
        raise ValueError("objective effective sample size must be finite and > 1")
    if config.objective_penalty == "hqc" and n_eff <= np.e:
        raise ValueError("n_eff must be > e for objective_penalty='hqc'")
    return w, weight_sum, kish_n_eff, n_eff, n_eff_source


def select_k_penalized_objective(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale: float | Literal["n_eff"],
    n_samples: int,
    sample_weight: Optional[np.ndarray] = None,
    n_candidates: int | None = None,
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
    df_path: Optional[np.ndarray] = None,
) -> Tuple[int, pd.DataFrame]:
    """Select k by maximizing a penalized CEFS+ proxy objective path.

    This is the rule behind ``AutoKConfig(k_method="penalized_objective")``
    and the router's measured default for CEFS+ (with
    ``objective_penalty="ebic"``). It maximizes
    ``objective_scale * obj(k) - penalty(k)`` over the prefix grid, treating
    the objective as a scaled log-likelihood gain and charging an information
    criterion for model size. Its target is *support recovery*, so it suits
    discovery work; for predictive sizing prefer a risk curve
    (``select_k_gaussian_cv``). The classical BIC/AIC/HQC penalties are
    structurally too weak here because the greedy step takes a maximum over
    the remaining candidates: use ``'ebic'`` or ``'ric'``, which charge for
    that multiplicity.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative objective after each path step, indexed from ``k=1``.
        Reshaped to one dimension and cast to float.
    config : AutoKConfig
        Must have ``k_method='penalized_objective'``. Reads
        ``objective_penalty``, ``objective_penalty_weight`` (custom only),
        ``ebic_gamma`` (EBIC only), ``objective_n_eff``, ``n_eff_mode``,
        ``min_k``, and ``max_k``.
    objective_scale : float or {'n_eff'}
        Multiplier turning the objective into a log-likelihood scale.
        ``'n_eff'`` uses the resolved effective sample size (Gaussian CEFS+);
        binary log-likelihood/score-test gains pass ``2.0`` by Wilks. Must be
        finite.
    n_samples : int
        Row count used to normalize ``sample_weight`` and to derive the
        effective sample size.
    sample_weight : ndarray of shape (n_samples,), optional
        Row weights, normalized to mean one before the Kish and weight-sum
        effective sizes are computed. None means uniform weights.
    n_candidates : int or None, default None
        Number of candidate features *before* screening or pruning. Required
        by ``objective_penalty`` in ``{'ebic', 'ric'}`` and must be at least
        the largest evaluated k; ignored by the other penalties.
    min_k : int or None, default None
        Floor on the returned k; falls back to ``config.min_k``. Clamped into
        ``[0, effective_max_k]``. A floor of 0 adds a ``k=0`` row with
        objective 0, letting the rule answer "no features".
    max_k : int or None, default None
        Ceiling on the returned k; falls back to ``config.max_k``. Clamped to
        ``len(objective_path)``.
    df_path : ndarray, optional
        Per-step degrees of freedom replacing the default ``df = k``. Must be
        at least as long as the effective ``max_k``. Honored by the
        ``k``-proportional penalties (BIC, MDL, AIC, HQC, custom); the EBIC
        and RIC penalties are defined on k itself and ignore it.

    Returns
    -------
    best_k : int
        Argmax of the penalized score over ``k >= effective min_k``, ties
        broken toward the smaller k. ``0`` when the effective ``max_k`` is
        non-positive.
    diagnostics : DataFrame
        One row per evaluated k with ``k``, ``objective``,
        ``delta_objective``, ``df``, ``penalty_weight``, ``penalty``,
        ``penalty_kind``, ``ebic_gamma``, ``n_candidates``,
        ``penalized_score``, ``selected``, ``n_eff``, ``n_eff_source``,
        ``weight_sum``, ``kish_n_eff``, ``objective_scale``,
        ``objective_scale_source``, ``objective_nonmonotone_steps``,
        ``n_finite_objective``, ``n_finite_penalized_score``,
        ``all_penalized_scores_invalid``, ``effective_min_k``,
        ``effective_max_k``, ``path_length``, and the saturation flags
        ``selected_at_effective_max_k``, ``selected_at_config_max_k``,
        ``path_exhausted_before_max_k``,
        ``evaluation_limited_before_path_end``, and ``selected_at_min_k``.
        Empty when the effective ``max_k`` is non-positive.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'penalized_objective'``; if
        ``objective_scale`` is not finite; if ``df_path`` is shorter than the
        effective path; if ``n_candidates`` is missing, non-positive, or
        smaller than the largest evaluated k under EBIC/RIC; if the resolved
        effective sample size is not finite and > 1; or if it is not greater
        than ``e`` under ``objective_penalty='hqc'``.

    Warns
    -----
    UserWarning
        When every candidate penalized score is non-finite; the effective
        minimum k is returned and ``all_penalized_scores_invalid`` is True.

    See Also
    --------
    select_k_posterior : Same criterion, exponentiated into a distribution.
    select_k_chi2_stop : Sequential test on the same gain path.
    select_k_gaussian_cv : Predictive-risk sizing instead of support
        recovery.
    AutoKConfig : ``objective_penalty``, ``ebic_gamma``, ``n_eff_mode``.

    Notes
    -----
    With ``d`` the degrees of freedom at k, the penalties are ``log(n_eff)*d``
    (BIC, MDL), ``2*d`` (AIC), ``2*log(log(n_eff))*d`` (HQC),
    ``objective_penalty_weight*d`` (custom),
    ``k*log(n_eff) + 2*gamma*log C(p, k)`` (EBIC), and ``2*k*log(p)`` (RIC),
    where ``p`` is ``n_candidates`` and ``log C`` is the exact log binomial
    coefficient. ``ebic_gamma='auto'`` resolves to the Chen-Chen threshold
    ``min(1, max(0, 1 - log(n_eff)/(2 log p)))``, degrading to plain BIC when
    ``n_eff >= p^2``. ``n_eff_mode='auto'`` selects the Kish size
    ``(sum w)^2 / sum w^2`` for EBIC and RIC and the weight sum otherwise;
    since weights are normalized to mean one, the weight sum equals
    ``n_samples`` regardless of weight skew. Evaluation is ``O(L)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import AutoKConfig, select_k_penalized_objective
    >>> objective = np.array([1.0, 1.8, 2.4, 2.42, 2.43, 2.44])
    >>> config = AutoKConfig(
    ...     k_method="penalized_objective",
    ...     objective_penalty="ebic",
    ...     min_k=0,
    ...     max_k=6,
    ... )
    >>> best_k, diag = select_k_penalized_objective(
    ...     objective,
    ...     config,
    ...     objective_scale="n_eff",
    ...     n_samples=200,
    ...     n_candidates=50,
    ... )
    >>> best_k
    3
    >>> diag["penalty_kind"].iloc[0], diag["n_eff_source"].iloc[0]
    ('ebic', 'kish')
    >>> bool(0.0 < diag["ebic_gamma"].iloc[0] <= 1.0)
    True
    """
    validate_auto_k_config(config)
    if config.k_method != "penalized_objective":
        raise ValueError(
            "select_k_penalized_objective requires "
            "AutoKConfig(k_method='penalized_objective')"
        )

    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)
    path_length = int(len(obj))
    effective_max_k = min(int(max_k if max_k is not None else config.max_k), path_length)
    if effective_max_k <= 0:
        return 0, pd.DataFrame()
    min_k_raw = int(min_k if min_k is not None else config.min_k)
    min_k_eff = max(0, min(min_k_raw, effective_max_k))

    _, weight_sum, kish_n_eff, n_eff, n_eff_source = _objective_weight_diagnostics(
        sample_weight,
        int(n_samples),
        config,
    )
    if objective_scale == "n_eff":
        scale_value = n_eff
        scale_label = "n_eff"
    else:
        scale_value = float(objective_scale)
        scale_label = str(float(objective_scale))
    if not np.isfinite(scale_value):
        raise ValueError("objective_scale must be finite")

    k_start = 0 if min_k_eff == 0 else 1
    ks = np.arange(k_start, effective_max_k + 1, dtype=np.int64)
    if df_path is None:
        df = ks.astype(np.float64)
    else:
        df_arr = np.asarray(df_path, dtype=np.float64).reshape(-1)
        if len(df_arr) < effective_max_k:
            raise ValueError("df_path must be at least as long as the effective objective path")
        if k_start == 0:
            df = np.concatenate(([0.0], df_arr[:effective_max_k]))
        else:
            df = df_arr[:effective_max_k]
    penalty, penalty_weight, ebic_gamma, n_candidates_used = _penalty_array(
        config,
        ks,
        n_eff=n_eff,
        n_candidates=n_candidates,
    )
    if config.objective_penalty not in {"ebic", "ric"}:
        penalty = penalty_weight * df
    objective_used = obj[ks - 1].astype(np.float64, copy=True)
    objective_used[ks == 0] = 0.0
    penalized_score = scale_value * objective_used - penalty
    n_finite_objective = int(np.sum(np.isfinite(objective_used)))
    n_finite_penalized_score = int(np.sum(np.isfinite(penalized_score)))
    valid = (ks >= min_k_eff) & np.isfinite(penalized_score)
    all_penalized_scores_invalid = not bool(valid.any())
    if valid.any():
        order = np.lexsort((ks[valid], -penalized_score[valid]))
        best_pos = np.flatnonzero(valid)[int(order[0])]
        best_k = int(ks[best_pos])
    else:
        warnings.warn(
            "All candidate penalized objective scores are non-finite; "
            "falling back to the effective minimum k.",
            UserWarning,
            stacklevel=2,
        )
        best_k = int(min_k_eff)

    full_objective = np.concatenate(([0.0], obj[:effective_max_k]))
    full_delta = np.diff(full_objective)
    delta_map = dict(zip(np.arange(1, effective_max_k + 1, dtype=np.int64), full_delta))
    delta = np.array([0.0 if k == 0 else delta_map[int(k)] for k in ks], dtype=np.float64)
    objective_nonmonotone_steps = int(np.sum(full_delta[1:] < -1e-12))
    path_exhausted_before_max_k = bool(path_length < int(config.max_k))
    evaluation_limited_before_path_end = bool(
        effective_max_k < min(path_length, int(config.max_k))
    )
    selected_at_effective_max_k = bool(best_k == effective_max_k)
    selected_at_config_max_k = bool(best_k == int(config.max_k))
    selected_at_min_k = bool(best_k == min_k_eff)

    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": objective_used,
            "delta_objective": delta,
            "df": df,
            "penalty_weight": penalty_weight,
            "penalty": penalty,
            "penalty_kind": config.objective_penalty,
            "ebic_gamma": ebic_gamma,
            "n_candidates": n_candidates_used,
            "penalized_score": penalized_score,
            "selected": ks == best_k,
            "n_eff": n_eff,
            "n_eff_source": n_eff_source,
            "weight_sum": weight_sum,
            "kish_n_eff": kish_n_eff,
            "objective_scale": scale_value,
            "objective_scale_source": scale_label,
            "objective_nonmonotone_steps": objective_nonmonotone_steps,
            "n_finite_objective": n_finite_objective,
            "n_finite_penalized_score": n_finite_penalized_score,
            "all_penalized_scores_invalid": all_penalized_scores_invalid,
            "effective_min_k": min_k_eff,
            "effective_max_k": effective_max_k,
            "path_length": path_length,
            "selected_at_effective_max_k": selected_at_effective_max_k,
            "selected_at_config_max_k": selected_at_config_max_k,
            "path_exhausted_before_max_k": path_exhausted_before_max_k,
            "evaluation_limited_before_path_end": evaluation_limited_before_path_end,
            "selected_at_min_k": selected_at_min_k,
        }
    )
    return best_k, diag


def select_k_posterior(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale: float | Literal["n_eff"],
    n_samples: int,
    n_candidates: int,
    sample_weight: Optional[np.ndarray] = None,
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
) -> Tuple[int, pd.DataFrame]:
    """Select k from a pseudo-posterior over prefixes on one greedy path.

    HPD intervals are computed over selectable k values. If ``min_k > 0``, the
    zero-feature posterior mass is still reported as ``p_zero`` but is excluded
    from MAP/HPD selection.

    This is the rule behind ``AutoKConfig(k_method="k_posterior")``. It
    exponentiates the EBIC criterion into a normalized distribution over
    prefix lengths, so besides a point estimate it reports a credible set,
    ``P(k = 0)``, and the entropy of the size distribution. Use it for
    discovery work where the *sharpness* of k matters: a wide HPD is the
    signal that the data do not pin k down and that parsimony rules (or the
    consensus combiner) should decide. It is not a predictive-sizing rule.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative objective after each path step, indexed from ``k=1``.
    config : AutoKConfig
        Must have ``k_method='k_posterior'``. Reads ``ebic_gamma``,
        ``posterior_level``, ``posterior_pick``, ``objective_n_eff``,
        ``n_eff_mode``, ``min_k``, and ``max_k``.
    objective_scale : float or {'n_eff'}
        Multiplier turning the objective into a log-likelihood scale.
        ``'n_eff'`` uses the resolved effective sample size; binary CEFS+
        gains pass ``2.0``. Must be finite.
    n_samples : int
        Row count used to normalize ``sample_weight`` and derive the
        effective sample size.
    n_candidates : int
        Number of candidate features before screening or pruning. Required:
        it drives the binomial size prior. Must be a positive integer at
        least as large as the largest evaluated k.
    sample_weight : ndarray of shape (n_samples,), optional
        Row weights, normalized to mean one. None means uniform weights.
    min_k : int or None, default None
        Floor on the selectable k; falls back to ``config.min_k`` and is
        clamped into ``[0, effective_max_k]``. ``k=0`` always appears in the
        grid so ``p_zero`` stays reportable, but is selectable only when the
        effective floor is 0.
    max_k : int or None, default None
        Ceiling on the evaluated k; falls back to ``config.max_k`` and is
        clamped to ``len(objective_path)``.

    Returns
    -------
    best_k : int
        The posterior mode when ``posterior_pick='map'``, or the smallest k
        inside the HPD set when ``posterior_pick='smallest_in_hpd'``. ``0``
        when the effective ``max_k`` is non-positive.
    diagnostics : DataFrame
        One row per grid k with ``k``, ``objective``, ``delta_objective``,
        ``log_post``, ``post``, ``in_hpd``, ``selected``, ``n_eff``,
        ``n_eff_source``, ``weight_sum``, ``kish_n_eff``, ``objective_scale``,
        ``objective_scale_source``, ``ebic_gamma``, ``n_candidates``,
        ``posterior_level``, ``hpd_lo``, ``hpd_hi``, ``p_zero``, ``entropy``,
        ``effective_min_k``, ``effective_max_k``, and ``path_length``. Empty
        when the effective ``max_k`` is non-positive.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'k_posterior'``; if
        ``objective_scale`` is not finite; if ``n_candidates`` is not a
        positive integer at least as large as the largest evaluated k; or if
        the resolved effective sample size is not finite and > 1.

    Warns
    -----
    UserWarning
        When every posterior log-weight is non-finite, and when no
        *selectable* log-weight is finite. Both fall back to the effective
        minimum k with an all-zero posterior and an empty HPD set.

    See Also
    --------
    select_k_penalized_objective : The EBIC point estimate this normalizes.
    select_k_stability : Reliability-flavored alternative when k is fuzzy.
    AutoKConfig : ``posterior_level``, ``posterior_pick``, ``ebic_gamma``.

    Notes
    -----
    The grid weight is
    ``log pi(k) = 0.5 * (objective_scale * obj(k) - k log n_eff)
    - gamma * log C(n_candidates, k)``, normalized with ``logsumexp``: the
    unit-information Gaussian prior gives the half-BIC Laplace core and the
    gamma-weighted binomial term is the multiplicity correction, so the MAP is
    the EBIC argmax by construction. The HPD set sorts k by descending mass
    and accumulates to ``posterior_level``; ``hpd_lo``/``hpd_hi`` report its
    envelope, not a contiguous interval. This is a *pseudo*-posterior computed
    along one greedy path: it does not integrate over model space and it
    inherits the greedy's path-dependence, so read it as calibrated relative
    evidence rather than as a coverage guarantee. Evaluation is ``O(L)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import AutoKConfig, select_k_posterior
    >>> objective = np.array([1.0, 1.8, 2.4, 2.42, 2.43, 2.44])
    >>> config = AutoKConfig(k_method="k_posterior", min_k=0, max_k=6)
    >>> best_k, diag = select_k_posterior(
    ...     objective,
    ...     config,
    ...     objective_scale="n_eff",
    ...     n_samples=200,
    ...     n_candidates=50,
    ... )
    >>> best_k
    3
    >>> int(diag["hpd_lo"].iloc[0]), int(diag["hpd_hi"].iloc[0])
    (3, 4)
    >>> float(round(diag["post"].sum(), 6))
    1.0
    """
    validate_auto_k_config(config)
    if config.k_method != "k_posterior":
        raise ValueError("select_k_posterior requires AutoKConfig(k_method='k_posterior')")

    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)
    path_length = int(len(obj))
    effective_max_k = min(int(max_k if max_k is not None else config.max_k), path_length)
    if effective_max_k <= 0:
        return 0, pd.DataFrame()
    min_k_raw = int(min_k if min_k is not None else config.min_k)
    min_k_eff = max(0, min(min_k_raw, effective_max_k))

    _, weight_sum, kish_n_eff, n_eff, n_eff_source = _objective_weight_diagnostics(
        sample_weight,
        int(n_samples),
        config,
    )
    if objective_scale == "n_eff":
        scale_value = n_eff
        scale_label = "n_eff"
    else:
        scale_value = float(objective_scale)
        scale_label = str(float(objective_scale))
    if not np.isfinite(scale_value):
        raise ValueError("objective_scale must be finite")

    if min_k_eff == 0:
        ks = np.arange(0, effective_max_k + 1, dtype=np.int64)
    else:
        ks = np.concatenate(
            (
                np.array([0], dtype=np.int64),
                np.arange(min_k_eff, effective_max_k + 1, dtype=np.int64),
            )
        )
    if int(n_candidates) < 1 or int(n_candidates) < int(np.max(ks, initial=0)):
        raise ValueError("n_candidates must be a positive integer >= the largest evaluated k")
    objective_used = obj[ks - 1].astype(np.float64, copy=True)
    objective_used[ks == 0] = 0.0
    gamma = _resolve_ebic_gamma(config, n_eff=n_eff, n_candidates=int(n_candidates))
    log_comb = _log_comb(int(n_candidates), ks)
    log_post = 0.5 * (scale_value * objective_used - ks.astype(np.float64) * np.log(n_eff))
    log_post -= gamma * log_comb
    finite = np.isfinite(log_post)
    if not bool(finite.any()):
        warnings.warn(
            "All posterior log-weights are non-finite; falling back to effective minimum k.",
            UserWarning,
            stacklevel=2,
        )
        best_k = int(min_k_eff)
        post = np.zeros_like(log_post)
        in_hpd = np.zeros_like(finite, dtype=bool)
    else:
        log_norm = float(logsumexp(log_post[finite]))
        post = np.zeros_like(log_post, dtype=np.float64)
        post[finite] = np.exp(log_post[finite] - log_norm)
        selectable = finite.copy()
        if min_k_eff > 0:
            selectable &= ks >= min_k_eff
        if not bool(selectable.any()):
            warnings.warn(
                "No selectable posterior log-weights are finite; falling back to effective minimum k.",
                UserWarning,
                stacklevel=2,
            )
            best_k = int(min_k_eff)
            in_hpd = np.zeros_like(finite, dtype=bool)
        else:
            selectable_pos = np.flatnonzero(selectable)
            selectable_log_norm = float(logsumexp(log_post[selectable]))
            selectable_post = np.exp(log_post[selectable_pos] - selectable_log_norm)
            map_pos = int(np.lexsort((ks[selectable_pos], -selectable_post))[0])
            map_k = int(ks[selectable_pos][map_pos])
            order = np.argsort(-selectable_post, kind="mergesort")
            cumsum = np.cumsum(selectable_post[order])
            cutoff = int(np.searchsorted(cumsum, float(config.posterior_level), side="left"))
            cutoff = min(cutoff, len(order) - 1)
            hpd_positions = selectable_pos[order[: cutoff + 1]]
            in_hpd = np.zeros_like(finite, dtype=bool)
            in_hpd[hpd_positions] = True
            if config.posterior_pick == "smallest_in_hpd":
                best_k = int(np.min(ks[in_hpd]))
            else:
                best_k = map_k

    hpd_ks = ks[in_hpd]
    hpd_lo = int(np.min(hpd_ks)) if hpd_ks.size else int(min_k_eff)
    hpd_hi = int(np.max(hpd_ks)) if hpd_ks.size else int(min_k_eff)
    p_zero = float(post[ks == 0][0]) if np.any(ks == 0) else 0.0
    entropy = float(-np.sum(post[post > 0.0] * np.log(post[post > 0.0])))
    delta = np.zeros_like(objective_used)
    nonzero = ks > 0
    delta[nonzero] = np.diff(np.concatenate(([0.0], obj[:effective_max_k])))[ks[nonzero] - 1]

    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": objective_used,
            "delta_objective": delta,
            "log_post": log_post,
            "post": post,
            "in_hpd": in_hpd,
            "selected": ks == best_k,
            "n_eff": n_eff,
            "n_eff_source": n_eff_source,
            "weight_sum": weight_sum,
            "kish_n_eff": kish_n_eff,
            "objective_scale": scale_value,
            "objective_scale_source": scale_label,
            "ebic_gamma": gamma,
            "n_candidates": int(n_candidates),
            "posterior_level": float(config.posterior_level),
            "hpd_lo": hpd_lo,
            "hpd_hi": hpd_hi,
            "p_zero": p_zero,
            "entropy": entropy,
            "effective_min_k": min_k_eff,
            "effective_max_k": effective_max_k,
            "path_length": path_length,
        }
    )
    return int(best_k), diag


def compute_objective_for_path(
    cache: "FeatureCache",
    y: np.ndarray,
    feature_path: List[str],
    *,
    shrink: float = 1e-6,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute objective path for an arbitrary ordered feature_path.

    Objective at step t:
        obj[t] = log|Σ_S| - log|Σ_{y,S}|
               = 2 * I(y; S)   (Gaussian MI proxy)

    This is the shared objective primitive, not a k rule: no
    ``AutoKConfig.k_method`` routes to it. The path-only rules
    (``elbow``, ``penalized_objective``, ``k_posterior``, ``chi2_stop``,
    ``forward_stop``, ``changepoint``, ``perm_gap``) consume exactly this
    curve as their ``objective_path`` argument, and the orchestrators build it
    while running the greedy. Call it directly to re-score an ordering you
    already have -- a hand-picked feature list, a path from another selector,
    or a saved path checked against a new target -- against one full cache.
    It is a discovery-flavored quantity (conditional information carried by
    the prefix), not a predictive score, and it is computed in sample on every
    cache row: it is not the cross-fitting primitive. Fold and bootstrap
    methods must build fold-local correlations and call
    ``objective_from_corr_path`` instead.

    Parameters
    ----------
    cache : FeatureCache
        Prebuilt Gaussian-copula cache from ``build_cache``. Supplies the
        rank-Gauss matrix ``Z``, the row subsample, the sample weights, and
        (when ``compute_Rxx=True``) the cached feature correlation matrix,
        which is sliced instead of recomputed. Duplicate non-synthetic feature
        names are rejected.
    y : ndarray of shape (n_rows_original,)
        Target aligned to the *original* rows the cache was built from, not to
        the cached subsample. It is raveled, sliced by ``cache.row_idx``, and
        rank-Gauss transformed under the cache weights.
    feature_path : list of str or int
        Ordered features. Strings resolve through ``cache.feature_names``,
        integers are original column indices. Entries that are unknown or that
        fell out of ``cache.valid_cols`` are skipped silently, so the result
        can be shorter than the input.
    shrink : float, default 1e-6
        Shrinkage of the correlation matrix toward the identity for numerical
        stability, forwarded to ``objective_from_corr_path``.
    eps : float, default 1e-12
        Floor on the Schur-complement determinants, forwarded to
        ``objective_from_corr_path``.

    Returns
    -------
    objective : ndarray of shape (n_resolved,)
        Cumulative, monotonically non-decreasing objective after each resolved
        step, indexed from ``k=1``. Empty when ``feature_path`` is empty or
        nothing resolves to a valid cache column.

    Raises
    ------
    ValueError
        If ``y`` does not have ``cache.n_rows_original`` rows, if the cache
        fails its structural contract (missing provenance marker, non-finite
        ``Z``, inconsistent ``valid_cols``/``row_idx``/weights), or if the
        cache carries duplicate feature names.

    See Also
    --------
    select_k_elbow : Consumes this curve.
    select_k_penalized_objective : Consumes this curve.
    select_k_chi2_stop : Consumes this curve.
    build_cache : Builds the ``FeatureCache`` this expects.

    Notes
    -----
    ``obj[t] = -log(1 - R^2_t)`` where ``R^2_t = r_S' R_S^-1 r_S`` is the
    squared multiple correlation of the copula-space target on the first ``t``
    path features, so the per-step gain is ``-log(1 - rho^2_t)`` for the
    sample partial correlation of the entering feature. Target correlations
    are clipped to ``+/-0.999999`` before the Schur-complement recursion,
    which costs ``O(k^2)`` in total; extracting ``R_path`` is an ``O(k^2)``
    slice with a cached ``Rxx`` and one weighted correlation otherwise.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import build_cache, compute_objective_for_path
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> objective = compute_objective_for_path(cache, y.to_numpy(), ["a", "b", "c"])
    >>> objective.shape
    (3,)
    >>> bool(np.all(np.diff(objective) >= 0.0))
    True
    >>> compute_objective_for_path(cache, y.to_numpy(), ["missing"]).size
    0
    """
    from sift.estimators.copula import (
        weighted_corr_with_vector,
        weighted_correlation_matrix,
        weighted_rank_gauss_1d,
    )
    from sift.selection.objective import objective_from_corr_path
    from sift.selection.knockoff_filter import (
        _reject_duplicate_feature_names,
        _validate_prebuilt_cache_structure,
    )

    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)

    if not feature_path:
        return np.empty(0, dtype=np.float64)

    valid_cols = np.asarray(cache.valid_cols)
    orig_to_valid = {int(orig): int(pos) for pos, orig in enumerate(valid_cols)}

    name_to_orig = {}
    if cache.feature_names:
        name_to_orig = {name: i for i, name in enumerate(cache.feature_names)}

    path_valid_pos = []
    for f in feature_path:
        if isinstance(f, str):
            orig_idx = name_to_orig.get(f, None)
            if orig_idx is None:
                continue
        else:
            orig_idx = int(f)

        vpos = orig_to_valid.get(int(orig_idx), None)
        if vpos is None:
            continue
        path_valid_pos.append(vpos)

    if not path_valid_pos:
        return np.empty(0, dtype=np.float64)

    path_valid_pos = np.asarray(path_valid_pos, dtype=np.int64)

    y_arr = np.asarray(y).ravel()
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but cache was built from "
            f"{cache.n_rows_original} rows"
        )
    ys = y_arr[np.asarray(cache.row_idx)]
    zy = weighted_rank_gauss_1d(ys, cache.sample_weight)
    r_y_full = weighted_corr_with_vector(cache.Z, zy, cache.sample_weight).astype(np.float64)

    r_path = r_y_full[path_valid_pos].copy()
    np.clip(r_path, -0.999999, 0.999999, out=r_path)

    if cache.Rxx is not None:
        R_full = np.asarray(cache.Rxx, dtype=np.float64)
        R_path = np.ascontiguousarray(R_full[np.ix_(path_valid_pos, path_valid_pos)], dtype=np.float64)
    else:
        Z_path = np.ascontiguousarray(cache.Z[:, path_valid_pos], dtype=np.float64)
        R_path = weighted_correlation_matrix(
            Z_path,
            np.asarray(cache.sample_weight, dtype=np.float64),
            backend="blas",
        )

    return objective_from_corr_path(R_path, r_path, shrink=shrink, eps=eps)
