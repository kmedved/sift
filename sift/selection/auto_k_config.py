"""Auto-k configuration, validation, and unused-field warnings."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import Any, Iterator, Literal, Optional

import numpy as np

from sift._deprecate import warn_external
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


AutoKConfig.__module__ = "sift.selection.auto_k"
validate_auto_k_config.__module__ = "sift.selection.auto_k"
resolve_auto_k_config.__module__ = "sift.selection.auto_k"
with_effective_k_bounds.__module__ = "sift.selection.auto_k"
