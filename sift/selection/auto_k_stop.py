"""Path-only automatic-k stopping rules for CEFS+ objective curves."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, f

from sift.selection.auto_k import AutoKConfig, validate_auto_k_config


def _objective_gains(objective_path: np.ndarray) -> np.ndarray:
    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)
    if obj.size == 0:
        return np.empty(0, dtype=np.float64)
    gains = np.empty_like(obj)
    gains[0] = obj[0]
    if obj.size > 1:
        gains[1:] = np.diff(obj)
    np.maximum(gains, 0.0, out=gains)
    return gains


def _sidak_max_pvalue(p_single: np.ndarray, m_eff: np.ndarray) -> np.ndarray:
    p1 = np.clip(np.asarray(p_single, dtype=np.float64), 0.0, 1.0)
    m = np.maximum(np.asarray(m_eff, dtype=np.float64), 1.0)
    m = np.broadcast_to(m, p1.shape).astype(np.float64, copy=False)
    out = np.empty_like(p1, dtype=np.float64)
    small = p1 < 1e-12
    certain = p1 >= 1.0
    mid = ~(small | certain)
    out[certain] = 1.0
    out[small] = np.minimum(1.0, m[small] * p1[small])
    out[mid] = -np.expm1(m[mid] * np.log1p(-p1[mid]))
    return np.clip(out, 0.0, 1.0)


def _li_ji_effective_tests(panel_eigs: np.ndarray | None) -> float | None:
    if panel_eigs is None:
        return None
    eigs = np.asarray(panel_eigs, dtype=np.float64).reshape(-1)
    if eigs.size == 0 or not np.isfinite(eigs).all():
        return None
    parts = np.where(eigs >= 1.0, 1.0 + (eigs - np.floor(eigs)), eigs)
    return float(max(1.0, np.sum(parts)))


def _gain_test_frame(
    objective_path: np.ndarray,
    *,
    n_eff: float,
    p_candidates: int,
    m_mode: str = "all",
    panel_eigs: np.ndarray | None = None,
) -> pd.DataFrame:
    gains = _objective_gains(objective_path)
    if gains.size == 0:
        return pd.DataFrame()
    if not np.isfinite(n_eff) or n_eff <= 2.0:
        raise ValueError("n_eff must be finite and > 2 for path gain tests")
    p = int(p_candidates)
    if p < 1:
        raise ValueError("p_candidates must be positive")

    stat_max_k = min(gains.size, max(0, int(np.floor(n_eff)) - 2))
    if stat_max_k <= 0:
        return pd.DataFrame()
    ks = np.arange(1, stat_max_k + 1, dtype=np.int64)
    gains = gains[:stat_max_k]
    nu = n_eff - ks.astype(np.float64) - 1.0
    valid_nu = nu > 0.0

    if m_mode == "panel":
        m_eff = np.maximum(1.0, p - ks + 1.0)
    elif m_mode == "li_ji":
        m_base = _li_ji_effective_tests(panel_eigs)
        if m_base is None:
            m_eff = np.maximum(1.0, p - ks + 1.0)
        else:
            panel_width = max(1, len(panel_eigs))
            m_eff = np.maximum(1.0, m_base * (p - ks + 1.0) / panel_width)
    else:
        m_eff = np.maximum(1.0, p - ks + 1.0)

    F_stat = np.full_like(gains, np.nan, dtype=np.float64)
    p_single = np.ones_like(gains, dtype=np.float64)
    F_stat[valid_nu] = nu[valid_nu] * np.expm1(gains[valid_nu])
    p_single[valid_nu] = f.sf(F_stat[valid_nu], 1.0, nu[valid_nu])
    p_max = _sidak_max_pvalue(p_single, m_eff)
    objective = np.asarray(objective_path, dtype=np.float64).reshape(-1)[:stat_max_k]
    return pd.DataFrame(
        {
            "k": ks,
            "objective": objective,
            "gain": gains,
            "F_stat": F_stat,
            "nu": nu,
            "m_eff": m_eff,
            "p_single": p_single,
            "p_max": p_max,
            "stat_max_k": stat_max_k,
        }
    )


def path_gain_pvalues(
    objective_path: np.ndarray,
    *,
    n_eff: float,
    p_candidates: int,
    m_mode: str = "all",
    panel_eigs: np.ndarray | None = None,
) -> np.ndarray:
    """Return Sidak-corrected max-gain p-values for each evaluated path step.

    This is the shared inference primitive behind
    ``AutoKConfig(k_method="chi2_stop")`` and
    ``AutoKConfig(k_method="forward_stop")``: both rules threshold exactly
    this p-value sequence, differing only in how they turn it into a stop.
    Call it directly when you want the evidence curve itself -- to plot where
    the path stops carrying conditional signal, or to feed a custom stopping
    rule. It is a discovery quantity (is the next step distinguishable from
    noise?), not a predictive one, and it costs nothing beyond the path.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative CEFS+ objective after each path step. Step gains are the
        first difference, floored at 0, with ``gain[0] = obj[0]``.
    n_eff : float
        Effective sample size behind the objective, typically the Kish size
        ``(sum w)^2 / sum w^2``. Must be finite and > 2; it also caps the
        evaluated path at ``floor(n_eff) - 2`` steps, past which the partial
        correlation has no residual degrees of freedom.
    p_candidates : int
        Number of candidate features before screening or pruning; positive.
        Drives the multiplicity correction.
    m_mode : {'all', 'panel', 'li_ji'}, default 'all'
        Effective test count per step. ``'all'`` uses ``p - t + 1``
        (conservative, and correct at step 1 because the top-m screen already
        maximized over all p). ``'panel'`` and ``'li_ji'`` fall back to that
        same count unless usable ``panel_eigs`` are supplied.
    panel_eigs : ndarray, optional
        Eigenvalues of the candidate panel correlation matrix, used only by
        ``m_mode='li_ji'``. Ignored when None, empty, or non-finite, in which
        case ``'li_ji'`` degrades to the ``'all'`` count.

    Returns
    -------
    p_max : ndarray of shape (n_steps,)
        Sidak-corrected p-value for each evaluated step ``k = 1..n_steps``,
        where ``n_steps = min(L, floor(n_eff) - 2)``. Empty when the path is
        empty or the degrees-of-freedom cap leaves no evaluable step.

    Raises
    ------
    ValueError
        If ``n_eff`` is not finite and > 2, or if ``p_candidates`` < 1.

    See Also
    --------
    select_k_chi2_stop : First-failure stop on these p-values.
    select_k_forward_stop : FDR-flavored accumulation stop on these p-values.
    select_k_perm_gap : Empirical permutation null when the analytic null is
        untrustworthy.

    Notes
    -----
    The single-candidate p-value at step ``t`` is
    ``SF_{F(1, nu_t)}(nu_t * (exp(gain_t) - 1))`` with
    ``nu_t = n_eff - t - 1``, the exact Gaussian test for the partial
    correlation the greedy step just added. The greedy took a maximum over
    ``m_eff`` remaining candidates, so the reported value is the Sidak
    correction ``1 - (1 - p)^m_eff``, evaluated in log space for tiny p. Note
    that ``corr_prune`` removes near-duplicates before that maximum; their
    statistics are almost perfectly correlated with the retained features, so
    the effective count is essentially unchanged and no adjustment is made.
    Cost is ``O(L)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import path_gain_pvalues
    >>> gains = np.concatenate([np.full(4, 0.15), np.full(26, 0.002)])
    >>> objective = np.cumsum(gains)
    >>> pvalues = path_gain_pvalues(objective, n_eff=200.0, p_candidates=50)
    >>> pvalues.shape
    (30,)
    >>> np.round(pvalues[:4], 3)
    array([0., 0., 0., 0.])
    >>> np.round(pvalues[4:8], 3)
    array([1., 1., 1., 1.])
    """
    frame = _gain_test_frame(
        objective_path,
        n_eff=n_eff,
        p_candidates=p_candidates,
        m_mode=m_mode,
        panel_eigs=panel_eigs,
    )
    return frame["p_max"].to_numpy(dtype=np.float64) if not frame.empty else np.empty(0)


def select_k_chi2_stop(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    n_eff: float,
    p_candidates: int,
    panel_eigs: np.ndarray | None = None,
) -> tuple[int, pd.DataFrame]:
    """Stop at the first patience-smoothed non-significant max-gain run.

    This is the rule behind ``AutoKConfig(k_method="chi2_stop")`` and the
    ``AutoKConfig.discovery(alpha)`` preset. It is the calibrated elbow: each
    path step is tested as a maximum over the remaining candidates, and the
    path stops just before the first run of ``stop_patience`` consecutive
    non-significant steps. Its target is support recovery with an
    interpretable level, so it fits discovery work; it under-selects in dense
    weak-signal regimes where every effect is individually tiny, and it is not
    a predictive-sizing rule. For an alpha to mean anything, pair it with
    ``min_k=0`` so a no-signal path can return nothing.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative CEFS+ objective after each path step. Truncated to
        ``config.max_k`` before testing.
    config : AutoKConfig
        Must have ``k_method='chi2_stop'``. Reads ``alpha``, ``m_mode``,
        ``stop_patience``, ``min_k``, and ``max_k``.
    n_eff : float
        Effective sample size behind the objective; finite and > 2. Also caps
        the evaluated path at ``floor(n_eff) - 2`` steps.
    p_candidates : int
        Number of candidate features before screening or pruning; positive.
    panel_eigs : ndarray, optional
        Candidate-panel correlation eigenvalues, used only when
        ``config.m_mode='li_ji'``.

    Returns
    -------
    selected_k : int
        Prefix length. The step before the first confirmed non-significant
        run, raised to the effective floor ``min(min_k, max evaluated k)``;
        the largest evaluated k when no such run occurs; ``0`` when no step is
        evaluable.
    diagnostics : DataFrame
        One row per evaluated step with ``k``, ``objective``, ``gain``,
        ``F_stat``, ``nu``, ``m_eff``, ``p_single``, ``p_max``,
        ``stat_max_k``, ``significant``, ``selected``, ``alpha``, ``m_mode``,
        ``n_eff``, and ``stopped_by`` (``'test'``, ``'floored'`` when the
        floor overrode the test, or ``'max_k'``). Empty when no step is
        evaluable.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'chi2_stop'``, if ``n_eff`` is not
        finite and > 2, or if ``p_candidates`` < 1.

    See Also
    --------
    path_gain_pvalues : The p-value sequence this thresholds.
    select_k_forward_stop : Averages the same p-values instead of stopping at
        the first failure.
    select_k_changepoint : Empirical noise floor instead of an analytic null.
    select_k_perm_gap : Permutation null for weighted or copula-awkward data.

    Notes
    -----
    Patience exists because greedy ordering is not monotone in signal: a
    masked signal can enter after a null feature, and ``stop_patience=2``
    recovers those at bounded cost. The analytic null assumes the Gaussian
    partial-correlation F identity; when weights or the copula transform make
    that suspect, cross-check against ``select_k_perm_gap``, which calibrates
    the same curve empirically. Cost is ``O(L)`` on top of the path.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import AutoKConfig, select_k_chi2_stop
    >>> gains = np.concatenate([np.full(4, 0.15), np.full(26, 0.002)])
    >>> objective = np.cumsum(gains)
    >>> config = AutoKConfig(k_method="chi2_stop", min_k=0, max_k=30)
    >>> selected_k, diag = select_k_chi2_stop(
    ...     objective, config, n_eff=200.0, p_candidates=50
    ... )
    >>> selected_k
    4
    >>> diag["stopped_by"].iloc[0]
    'test'
    >>> print(diag[["k", "p_max", "significant"]].head(6).round(3).to_string(index=False))
     k  p_max  significant
     1    0.0         True
     2    0.0         True
     3    0.0         True
     4    0.0         True
     5    1.0        False
     6    1.0        False
    """
    validate_auto_k_config(config)
    if config.k_method != "chi2_stop":
        raise ValueError("select_k_chi2_stop requires AutoKConfig(k_method='chi2_stop')")
    objective_eval = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    diag = _gain_test_frame(
        objective_eval,
        n_eff=n_eff,
        p_candidates=p_candidates,
        m_mode=config.m_mode,
        panel_eigs=panel_eigs,
    )
    if diag.empty:
        return 0, diag

    floor = max(0, min(int(config.min_k), int(diag["k"].max())))
    patience = int(config.stop_patience)
    significant = diag["p_max"].to_numpy(dtype=np.float64) <= float(config.alpha)
    bad = ~significant
    selected_k = int(diag["k"].max())
    stopped_by = "max_k"
    run = 0
    for pos, is_bad in enumerate(bad):
        if is_bad:
            run += 1
        else:
            run = 0
        if run >= patience:
            start_pos = pos - patience + 1
            candidate_k = int(diag["k"].iloc[start_pos] - 1)
            selected_k = max(candidate_k, floor)
            stopped_by = "floored" if candidate_k < floor else "test"
            break

    diag = diag.copy()
    diag["significant"] = significant
    diag["selected"] = diag["k"] == selected_k
    diag["alpha"] = float(config.alpha)
    diag["m_mode"] = config.m_mode
    diag["n_eff"] = float(n_eff)
    diag["stopped_by"] = stopped_by
    return selected_k, diag


def select_k_forward_stop(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    n_eff: float,
    p_candidates: int,
    panel_eigs: np.ndarray | None = None,
) -> tuple[int, pd.DataFrame]:
    """Select the largest prefix accepted by the ForwardStop accumulation rule.

    This is the rule behind ``AutoKConfig(k_method="forward_stop")``. It reads
    the same Sidak-corrected step p-values as ``select_k_chi2_stop`` but
    interprets ``alpha`` as a false-discovery-rate level over the selected
    prefix (G'Sell et al. 2016) rather than as a per-step threshold, so a
    single large interior p-value cannot terminate the path -- the running
    average has to degrade. Use it for discovery when you want an FDR reading
    of the prefix and a rule that tolerates masked-then-revealed signals; it
    is not a predictive-sizing rule.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative CEFS+ objective after each path step. Truncated to
        ``config.max_k`` before testing.
    config : AutoKConfig
        Must have ``k_method='forward_stop'``. Reads ``alpha`` (as the FDR
        level), ``m_mode``, ``min_k``, and ``max_k``. ``stop_patience`` is not
        used by this rule.
    n_eff : float
        Effective sample size behind the objective; finite and > 2. Also caps
        the evaluated path at ``floor(n_eff) - 2`` steps.
    p_candidates : int
        Number of candidate features before screening or pruning; positive.
    panel_eigs : ndarray, optional
        Candidate-panel correlation eigenvalues, used only when
        ``config.m_mode='li_ji'``.

    Returns
    -------
    selected_k : int
        Largest evaluated k at or above the effective floor
        ``max(1, min(min_k, max evaluated k))`` whose running mean of
        ``Y = -log1p(-p_max)`` stays at or below ``alpha``; ``0`` when no k
        qualifies or no step is evaluable.
    diagnostics : DataFrame
        One row per evaluated step with ``k``, ``objective``, ``gain``,
        ``F_stat``, ``nu``, ``m_eff``, ``p_single``, ``p_max``,
        ``stat_max_k``, ``Y``, ``Y_running_mean``, ``eligible``,
        ``selected``, ``alpha``, ``m_mode``, ``n_eff``, and ``stopped_by``
        (``'forward_stop'`` or ``'empty'``). Empty when no step is evaluable.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'forward_stop'``, if ``n_eff`` is not
        finite and > 2, or if ``p_candidates`` < 1.

    See Also
    --------
    path_gain_pvalues : The p-value sequence this accumulates.
    select_k_chi2_stop : First-failure stop on the same p-values.
    select_k_penalized_objective : Information-criterion alternative.

    Notes
    -----
    ForwardStop transforms each p-value into ``Y_t = -log1p(-p_t)``, which is
    near 0 for a decisive step and Exp(1) for a null step, then keeps the
    largest ``k`` with ``mean(Y_1..Y_k) <= alpha``. Under independent uniform
    nulls that controls the FDR of the selected prefix at ``alpha``; the path
    p-values here are sequentially dependent, so the guarantee is approximate
    and was measured rather than assumed. ``selected_k`` is non-decreasing in
    ``alpha``. Cost is ``O(L)`` on top of the path.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import AutoKConfig, select_k_forward_stop
    >>> gains = np.concatenate([np.full(4, 0.15), np.full(26, 0.002)])
    >>> objective = np.cumsum(gains)
    >>> config = AutoKConfig(
    ...     k_method="forward_stop", min_k=0, max_k=30, alpha=0.1
    ... )
    >>> selected_k, diag = select_k_forward_stop(
    ...     objective, config, n_eff=200.0, p_candidates=50
    ... )
    >>> selected_k
    4
    >>> diag["stopped_by"].iloc[0]
    'forward_stop'
    >>> diag.loc[diag["k"] <= 5, "eligible"].tolist()
    [True, True, True, True, False]
    """
    validate_auto_k_config(config)
    if config.k_method != "forward_stop":
        raise ValueError("select_k_forward_stop requires AutoKConfig(k_method='forward_stop')")
    objective_eval = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    diag = _gain_test_frame(
        objective_eval,
        n_eff=n_eff,
        p_candidates=p_candidates,
        m_mode=config.m_mode,
        panel_eigs=panel_eigs,
    )
    if diag.empty:
        return 0, diag

    eps = np.finfo(np.float64).eps
    pvals = np.clip(diag["p_max"].to_numpy(dtype=np.float64), 0.0, 1.0 - eps)
    Y = -np.log1p(-pvals)
    ks = diag["k"].to_numpy(dtype=np.int64)
    running = np.cumsum(Y) / ks
    floor = max(1, min(int(config.min_k), int(ks.max())))
    eligible = (ks >= floor) & (running <= float(config.alpha))
    selected_k = int(ks[eligible][-1]) if np.any(eligible) else 0

    diag = diag.copy()
    diag["Y"] = Y
    diag["Y_running_mean"] = running
    diag["eligible"] = eligible
    diag["selected"] = diag["k"] == selected_k
    diag["alpha"] = float(config.alpha)
    diag["m_mode"] = config.m_mode
    diag["n_eff"] = float(n_eff)
    diag["stopped_by"] = "forward_stop" if selected_k else "empty"
    return selected_k, diag


def _median_smooth(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width % 2 == 0:
        width += 1
    if width <= 1 or x.size < width:
        return x.copy()
    half = width // 2
    out = x.copy()
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def select_k_changepoint(
    objective_path: np.ndarray,
    config: AutoKConfig,
    *,
    objective_scale: float,
    n_eff: float,
    p_candidates: int,
) -> tuple[int, pd.DataFrame]:
    """Select k from a noise-floor changepoint on scaled objective gains.

    This is the rule behind ``AutoKConfig(k_method="changepoint")``: the elbow
    with its arbitrary relative-gain threshold replaced by a floor estimated
    from the path's own noise tail and cross-checked against the analytic
    max-of-chi-square prediction. It costs nothing beyond the path and needs
    no folds, groups, or permutations, which makes it the cheapest
    discovery-flavored option when no split context is available. It is
    experimental: in the Auto-K v2 campaign it over-selected on the deep-null
    design and did not clear the gate, so read it as a diagnostic on the gain
    curve rather than as an automatic sizing default.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Cumulative CEFS+ objective after each path step. Truncated to
        ``config.max_k``. The tail must actually contain noise for the floor
        estimate to mean anything, so run the path to a generous ``max_k``.
    config : AutoKConfig
        Must have ``k_method='changepoint'``. Reads ``floor_z``,
        ``floor_window``, ``min_k``, ``max_k``, and ``stop_patience`` (reused
        as the median-smoothing width when > 2, else 3).
    objective_scale : float
        Multiplier turning a gain into a chi-square-scale statistic. Gaussian
        CEFS+ passes the effective sample size; binary CEFS+ log-likelihood or
        score-test gains pass ``2.0`` by Wilks. Must be positive and finite.
    n_eff : float
        Effective sample size. Caps the evaluated path at
        ``floor(n_eff) - 2`` steps; a non-finite value disables that cap.
    p_candidates : int
        Number of candidate features before screening or pruning. Sets the
        survivor count ``max(1, p_candidates - L_eff + 1)`` used by the
        analytic floor cross-check.

    Returns
    -------
    selected_k : int
        Position of the last pre-tail gain above the noise threshold; the
        effective floor ``min(min_k, L_eff)`` when none exceeds it; the
        effective maximum when the tail is not noise or leaves no pre-tail
        range; ``0`` when no step is evaluable.
    diagnostics : DataFrame
        One row per evaluated step with ``k``, ``objective``, ``gain``,
        ``log_scaled_gain``, ``objective_scale``, ``n_eff``, ``floor_mu``,
        ``floor_sigma``, ``analytic_floor_median``, ``threshold``,
        ``tail_width``, ``floor_not_reached``, ``exceeds``, and ``selected``.
        The floor columns are NaN on the degenerate branches. Empty when no
        step is evaluable.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'changepoint'``, or if
        ``objective_scale`` is not positive and finite.

    Warns
    -----
    UserWarning
        When fewer than three gains are evaluable (falls back to the method
        floor); when the tail window leaves no pre-tail range (falls back to
        the effective maximum); and when the estimated floor sits far above
        the analytic null median, meaning signal extends past ``max_k``
        (``floor_not_reached=True``, returns the effective maximum -- treat
        that k as censored).

    See Also
    --------
    select_k_elbow : The uncalibrated rule this was written to replace.
    select_k_chi2_stop : Analytic sequential test on the same gains.
    select_k_perm_gap : Empirical permutation envelope on the same curve.
    AutoKConfig : ``floor_z`` and ``floor_window`` fields.

    Notes
    -----
    Work happens on ``x_t = log(objective_scale * gain_t + 1e-12)``. The tail
    window is ``W = min(L_eff - 1, max(10, ceil(floor_window * L_eff)))`` when
    ``floor_window`` is a fraction, or that many steps when it is an integer;
    the floor is ``mu = median(x_W)`` and ``sigma = 1.4826 * MAD(x_W)``, and
    the threshold is ``mu + floor_z * sigma``. Before thresholding, ``x`` is
    median-smoothed so one heavy-tailed null spike cannot extend k. The
    analytic cross-check compares ``mu`` against
    ``log(chi2.isf(-expm1(log(0.5) / m_tail), df=1))``, the null median of a
    maximum over ``m_tail`` survivors. Cost is ``O(L * width)`` for the
    smoothing pass, negligible beside the path itself.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import AutoKConfig, select_k_changepoint
    >>> noise = np.tile([0.0015, 0.0025, 0.002, 0.003, 0.001], 7)
    >>> objective = np.cumsum(np.concatenate([np.full(5, 0.5), noise]))
    >>> config = AutoKConfig(k_method="changepoint", min_k=0, max_k=40)
    >>> selected_k, diag = select_k_changepoint(
    ...     objective,
    ...     config,
    ...     objective_scale=200.0,
    ...     n_eff=200.0,
    ...     p_candidates=50,
    ... )
    >>> selected_k
    5
    >>> bool(diag["floor_not_reached"].iloc[0]), int(diag["tail_width"].iloc[0])
    (False, 10)
    >>> diag.loc[diag["k"] <= 6, "exceeds"].tolist()
    [True, True, True, True, True, False]
    """
    validate_auto_k_config(config)
    if config.k_method != "changepoint":
        raise ValueError("select_k_changepoint requires AutoKConfig(k_method='changepoint')")
    objective_eval = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    gains = _objective_gains(objective_eval)
    L = int(gains.size)
    if L == 0:
        return 0, pd.DataFrame()
    effective_max = min(L, max(0, int(np.floor(n_eff)) - 2)) if np.isfinite(n_eff) else L
    effective_max = max(0, effective_max)
    if effective_max <= 0:
        return 0, pd.DataFrame()
    gains = gains[:effective_max]
    ks = np.arange(1, effective_max + 1, dtype=np.int64)
    scale = float(objective_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("objective_scale must be positive and finite")
    x = np.log(scale * gains + 1e-12)

    if effective_max < 3:
        warnings.warn(
            "changepoint requires at least three objective gains; falling back to the method floor.",
            UserWarning,
            stacklevel=2,
        )
        selected_k = max(0, min(int(config.min_k), effective_max))
        floor_not_reached = False
        tail_width = 0
        floor_mu = floor_sigma = analytic = threshold = np.nan
        exceeds = np.zeros(effective_max, dtype=bool)
    else:
        if isinstance(config.floor_window, (int, np.integer)):
            requested = int(config.floor_window)
        else:
            requested = int(np.ceil(float(config.floor_window) * effective_max))
        tail_width = min(effective_max - 1, max(10, requested))
        pre_tail = effective_max - tail_width
        if pre_tail <= 0:
            warnings.warn(
                "changepoint tail window leaves no pre-tail range; falling back to effective max.",
                UserWarning,
                stacklevel=2,
            )
            selected_k = effective_max
            floor_not_reached = False
            floor_mu = floor_sigma = analytic = threshold = np.nan
            exceeds = np.zeros(effective_max, dtype=bool)
        else:
            smooth_width = int(config.stop_patience) if int(config.stop_patience) > 2 else 3
            x_used = _median_smooth(x, smooth_width)
            tail = x_used[-tail_width:]
            floor_mu = float(np.median(tail))
            floor_sigma = float(1.4826 * np.median(np.abs(tail - floor_mu)))
            m_tail = max(1, int(p_candidates) - effective_max + 1)
            analytic = float(np.log(chi2.isf(-np.expm1(np.log(0.5) / m_tail), df=1)))
            threshold = floor_mu + float(config.floor_z) * floor_sigma
            floor_not_reached = bool(floor_mu > analytic + 3.0 * floor_sigma)
            if floor_not_reached:
                warnings.warn(
                    "changepoint noise floor was not reached before max_k; returning effective max.",
                    UserWarning,
                    stacklevel=2,
                )
                selected_k = effective_max
                exceeds = np.zeros(effective_max, dtype=bool)
            else:
                exceeds = x_used > threshold
                pre_exceeds = np.flatnonzero(exceeds[:pre_tail])
                floor = max(0, min(int(config.min_k), effective_max))
                selected_k = int(pre_exceeds[-1] + 1) if pre_exceeds.size else floor

    objective = objective_eval[:effective_max]
    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": objective,
            "gain": gains,
            "log_scaled_gain": x,
            "objective_scale": scale,
            "n_eff": float(n_eff),
            "floor_mu": floor_mu,
            "floor_sigma": floor_sigma,
            "analytic_floor_median": analytic,
            "threshold": threshold,
            "tail_width": tail_width,
            "floor_not_reached": floor_not_reached,
            "exceeds": exceeds,
            "selected": ks == selected_k,
        }
    )
    return int(selected_k), diag
