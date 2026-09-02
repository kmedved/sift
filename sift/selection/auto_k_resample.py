"""Resampling-based automatic-k selectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from sift._permute import build_group_info, permute_array, resolve_permutation_method
from sift.estimators.copula import weighted_rank_gauss_1d
from sift.selection.auto_k import AutoKConfig, validate_auto_k_config
from sift.selection.cefsplus import (
    _gaussian_jmi_select,
    _gaussian_mrmr_select,
    cefsplus_loop_with_objective,
)
from sift.selection.knockoff_filter import (
    _reject_duplicate_feature_names,
    _validate_prebuilt_cache_structure,
)
from sift.selection.panel import build_candidate_panel, local_corr_panel

_STABILITY_PHI_FLOOR = 0.5


def _resolve_null(null: str, *, groups, time) -> str:
    if null == "permute":
        return "global"
    if null == "auto":
        return resolve_permutation_method("auto", groups=groups, time=time)
    if null in {"within_group", "circular_shift"}:
        return null
    raise ValueError("perm_null must be 'auto', 'permute', 'circular_shift', or 'within_group'")


def _run_panel_path(panel, max_k: int, *, method: str = "cefsplus") -> tuple[np.ndarray, np.ndarray]:
    k_actual = min(int(max_k), len(panel.cand))
    if k_actual <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if method == "cefsplus":
        local_path, objective = cefsplus_loop_with_objective(panel.R, panel.r, k_actual, panel.rel)
    elif method in {"mrmr_quot", "mrmr_diff"}:
        local_path = _gaussian_mrmr_select(
            panel.R,
            panel.rel,
            k_actual,
            use_quotient=method == "mrmr_quot",
        )
        objective = np.empty(0, dtype=np.float64)
    elif method in {"jmi", "jmim"}:
        local_path = _gaussian_jmi_select(
            panel.R,
            panel.r,
            panel.rel,
            k_actual,
            use_min=method == "jmim",
        )
        objective = np.empty(0, dtype=np.float64)
    else:
        raise ValueError(f"Unknown Gaussian selector method: {method!r}")
    return panel.cand[local_path].astype(np.int64), np.asarray(objective, dtype=np.float64)


def null_objective_paths(
    cache,
    y,
    *,
    B: int,
    max_k: int,
    null: str,
    time=None,
    groups=None,
    top_m: int,
    corr_prune,
    random_state: int,
) -> np.ndarray:
    """Build permutation-null CEFS+ objective paths, extended flat to max_k.

    This is the null-calibration half of ``AutoKConfig(k_method="perm_gap")``:
    it reruns the whole CEFS+ greedy against ``B`` permuted targets so the
    real objective curve can be compared with the curve the same pipeline
    produces when there is nothing to find. Use it whenever the analytic
    Gaussian null behind ``chi2_stop`` is suspect -- skewed weights, awkward
    copulas, grouped or autocorrelated rows -- or to plot a null envelope on
    its own. It is a discovery quantity; it says nothing about predictive
    risk.

    Parameters
    ----------
    cache : FeatureCache
        Prebuilt Gaussian-copula cache from ``build_cache``. Duplicate
        non-synthetic feature names are rejected. A cached ``Rxx`` makes each
        permutation a cheap slice; without one every permutation pays a
        ``top_m``-wide weighted correlation, so prefer ``compute_Rxx=True``
        for moderate p.
    y : array-like of shape (n_rows_original,)
        Target aligned to the original rows, not to the cached subsample.
    B : int
        Number of null replicates; one row of the result per replicate.
    max_k : int
        Path depth for every null run and the column count of the result.
    null : {'auto', 'permute', 'circular_shift', 'within_group'}
        Null construction. ``'permute'`` is an iid permutation of ``y``;
        ``'circular_shift'`` requires ``time`` and preserves autocorrelation;
        ``'within_group'`` requires ``groups`` and preserves group effects;
        ``'auto'`` resolves to circular shift with ``time``, within-group with
        ``groups``, and plain permutation otherwise.
    time : array-like of shape (n_rows_original,), optional
        Row timestamps, required by ``null='circular_shift'``.
    groups : array-like of shape (n_rows_original,), optional
        Group labels, required by ``null='within_group'``.
    top_m : int
        Screening width for the per-permutation candidate panel. The screen is
        y-dependent, so it is redone for every null target.
    corr_prune : float, None, or {'auto'}
        Correlation-pruning threshold forwarded to the panel builder, matching
        the real run's setting.
    random_state : int
        Seed for the permutation streams; replicate ``b`` uses the ``b``-th
        child of ``SeedSequence(random_state)``, so results are reproducible
        and independent across replicates.

    Returns
    -------
    null_paths : ndarray of shape (B, max_k)
        Cumulative null objective per replicate. A run that exhausts its
        candidates before ``max_k`` is extended flat with its last value; a
        replicate that produced no objective at all stays all zeros.

    Raises
    ------
    ValueError
        If ``y``, ``groups``, or ``time`` do not match the cache's original
        row count; if ``null`` is not one of the four accepted values; if
        ``'within_group'`` is requested without ``groups`` or
        ``'circular_shift'`` without ``time``; or if the cache fails its
        structural contract or carries duplicate feature names.

    See Also
    --------
    select_k_perm_gap : Consumes these paths.
    bootstrap_paths : Same replicate scaffolding for stability selection.
    path_gain_pvalues : Analytic alternative to this empirical null.

    Notes
    -----
    The raw target is permuted *before* the cache row subsample is taken and
    before the rank-Gauss transform, because ranks are weight-dependent and a
    shift applied after subsampling would destroy the time structure the null
    is supposed to preserve. Each replicate also re-screens and re-prunes its
    own panel: reusing the real-y panel would understate the null maximum and
    inflate the gap. Cost is roughly ``B`` times one rank transform, one
    correlation-with-target, a panel slice, and the greedy loop.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import build_cache, null_objective_paths
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> nulls = null_objective_paths(
    ...     cache,
    ...     y.to_numpy(),
    ...     B=5,
    ...     max_k=3,
    ...     null="permute",
    ...     top_m=6,
    ...     corr_prune=None,
    ...     random_state=0,
    ... )
    >>> nulls.shape
    (5, 3)
    >>> bool(np.all(np.diff(nulls, axis=1) >= 0.0))
    True
    """
    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError("y length must match the cache's original row count")
    groups_arr = None if groups is None else np.asarray(groups).reshape(-1)
    time_arr = None if time is None else np.asarray(time).reshape(-1)
    if groups_arr is not None and groups_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("groups length must match y")
    if time_arr is not None and time_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("time length must match y")

    method = _resolve_null(null, groups=groups_arr, time=time_arr)
    if method == "within_group" and groups_arr is None:
        raise ValueError("perm_null='within_group' requires groups")
    if method == "circular_shift" and time_arr is None:
        raise ValueError("perm_null='circular_shift' requires time")
    group_info = None
    if method != "global":
        group_info = build_group_info(groups_arr, time_arr, n_samples=y_arr.shape[0])

    seeds = np.random.SeedSequence(random_state).spawn(int(B))
    out = np.zeros((int(B), int(max_k)), dtype=np.float64)
    for b, child in enumerate(seeds):
        rng = np.random.default_rng(child)
        y_b_full = permute_array(
            y_arr,
            method=method,
            group_info=group_info,
            block_size="auto",
            rng=rng,
        )
        y_b_cache = y_b_full[np.asarray(cache.row_idx, dtype=np.int64)]
        zy_b = weighted_rank_gauss_1d(y_b_cache, cache.sample_weight)
        panel = build_candidate_panel(
            cache,
            None,
            int(max_k),
            top_m=top_m,
            corr_prune=corr_prune,
            method="cefsplus",
            zy=zy_b,
        )
        _path, objective = _run_panel_path(panel, int(max_k), method="cefsplus")
        if objective.size:
            out[b, : objective.size] = objective
            if objective.size < max_k:
                out[b, objective.size :] = objective[-1]
    return out


def select_k_perm_gap(
    objective_path: np.ndarray,
    null_paths: np.ndarray,
    config: AutoKConfig,
) -> tuple[int, pd.DataFrame]:
    """Select k by comparing the real objective curve to permutation null paths.

    This is the rule behind ``AutoKConfig(k_method="perm_gap")`` and the
    router's choice under heavy weight skew. It is Tibshirani's gap statistic
    moved from clustering onto a selection path: the gap
    ``Gap(k) = obj(k) - mean_b obj_b(k)`` rises while signal remains and goes
    flat once the real greedy is picking the same kind of maxima a null run
    would. Because the null is generated by the same pipeline, it stays
    calibrated where the analytic F/chi-square null of ``chi2_stop`` breaks --
    skewed weights, awkward copulas, grouped or autocorrelated rows. Its
    target is support recovery, not predictive sizing.

    Parameters
    ----------
    objective_path : ndarray of shape (L,)
        Real cumulative CEFS+ objective, indexed from ``k=1``. Truncated to
        ``config.max_k``.
    null_paths : ndarray of shape (B, L_null)
        Permutation-null objective curves, one row per replicate, as returned
        by ``null_objective_paths``. The comparison length is
        ``min(L, L_null, config.max_k)``.
    config : AutoKConfig
        Must have ``k_method='perm_gap'``. Reads ``gap_rule``, ``min_k``,
        ``max_k``, and -- only under ``gap_rule='gain_envelope'`` -- ``alpha``
        and ``stop_patience``. ``perm_B`` and ``perm_null`` are consumed
        upstream by ``null_objective_paths``, not here.

    Returns
    -------
    selected_k : int
        Prefix length in ``[floor, L]`` where ``floor = min(min_k, L)``. Zero
        is reachable when ``min_k=0``.
    diagnostics : DataFrame
        One row per k in ``0..L`` (the ``k=0`` row anchors both curves at 0)
        with ``k``, ``objective``, ``null_mean``, ``null_sd``, ``gap``,
        ``gap_se``, ``selected``, ``perm_B``, and ``gap_rule``. Empty when
        either curve is empty.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'perm_gap'``.

    See Also
    --------
    null_objective_paths : Builds the ``null_paths`` argument.
    select_k_chi2_stop : Analytic null on the same gain path.
    select_k_changepoint : Empirical noise floor without permutations.
    AutoKConfig : ``gap_rule``, ``perm_B``, ``perm_null`` fields.

    Notes
    -----
    ``gap_se(k) = sd_b(obj_b(k)) * sqrt(1 + 1/B)`` uses the sample standard
    deviation over replicates (zero when ``B < 2``). The three rules are:
    ``'tibshirani'`` (default), the smallest ``k >= floor`` with
    ``Gap(k) >= Gap(k+1) - gap_se(k+1)``, falling back to the argmax;
    ``'argmax'``, the largest gap with ties broken toward the smaller k; and
    ``'gain_envelope'``, which stops at the first run of ``stop_patience``
    steps whose real gain fails to clear
    ``mean_b gain_b + z_alpha * sd_b(gain_b)``. Cost here is ``O(B * L)``; the
    expense of the method lives in building ``null_paths``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import (
    ...     AutoKConfig,
    ...     build_cache,
    ...     compute_objective_for_path,
    ...     null_objective_paths,
    ...     select_k_perm_gap,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> objective = compute_objective_for_path(cache, y.to_numpy(), ["a", "b", "c"])
    >>> nulls = null_objective_paths(
    ...     cache,
    ...     y.to_numpy(),
    ...     B=5,
    ...     max_k=3,
    ...     null="permute",
    ...     top_m=6,
    ...     corr_prune=None,
    ...     random_state=0,
    ... )
    >>> config = AutoKConfig(k_method="perm_gap", min_k=0, max_k=3, perm_B=5)
    >>> selected_k, diag = select_k_perm_gap(objective, nulls, config)
    >>> selected_k
    2
    >>> diag["gap_rule"].iloc[0], int(diag["perm_B"].iloc[0])
    ('tibshirani', 5)
    """
    validate_auto_k_config(config)
    if config.k_method != "perm_gap":
        raise ValueError("select_k_perm_gap requires AutoKConfig(k_method='perm_gap')")
    obj = np.asarray(objective_path, dtype=np.float64).reshape(-1)[: int(config.max_k)]
    nulls = np.asarray(null_paths, dtype=np.float64)
    if obj.size == 0 or nulls.size == 0:
        return 0, pd.DataFrame()
    L = min(obj.size, nulls.shape[1], int(config.max_k))
    obj = obj[:L]
    nulls = nulls[:, :L]
    nulls_full = np.concatenate([np.zeros((nulls.shape[0], 1)), nulls], axis=1)
    obj_full = np.concatenate(([0.0], obj))
    ks = np.arange(0, L + 1, dtype=np.int64)
    null_mean = np.mean(nulls_full, axis=0)
    null_sd = (
        np.std(nulls_full, axis=0, ddof=1)
        if nulls_full.shape[0] >= 2
        else np.zeros(L + 1)
    )
    gap = obj_full - null_mean
    gap_se = null_sd * np.sqrt(1.0 + 1.0 / max(1, nulls.shape[0]))
    floor = max(0, min(int(config.min_k), L))

    if config.gap_rule == "argmax":
        valid = ks >= floor
        selected_k = int(ks[valid][np.argmax(gap[valid])]) if np.any(valid) else 0
    elif config.gap_rule == "gain_envelope":
        step_ks = np.arange(1, L + 1, dtype=np.int64)
        real_gain = np.diff(obj_full)
        null_gain = np.diff(nulls_full, axis=1)
        z = float(norm.ppf(1.0 - float(config.alpha)))
        null_gain_sd = (
            np.std(null_gain, axis=0, ddof=1)
            if null_gain.shape[0] >= 2
            else np.zeros(null_gain.shape[1], dtype=np.float64)
        )
        envelope = np.mean(null_gain, axis=0) + z * null_gain_sd
        bad = real_gain <= envelope
        run = 0
        selected_k = L
        for pos, is_bad in enumerate(bad):
            run = run + 1 if is_bad else 0
            if run >= int(config.stop_patience):
                candidate = int(step_ks[pos - int(config.stop_patience) + 1] - 1)
                if candidate >= floor:
                    selected_k = candidate
                    break
    else:
        valid = ks >= floor
        selected_k = int(ks[valid][np.argmax(gap[valid])]) if np.any(valid) else 0
        for i in range(floor, L):
            if gap[i] >= gap[i + 1] - gap_se[i + 1]:
                selected_k = int(ks[i])
                break

    diag = pd.DataFrame(
        {
            "k": ks,
            "objective": obj_full,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "gap": gap,
            "gap_se": gap_se,
            "selected": ks == selected_k,
            "perm_B": int(nulls.shape[0]),
            "gap_rule": config.gap_rule,
        }
    )
    return selected_k, diag


def bootstrap_paths(
    cache,
    y,
    *,
    B: int,
    max_k: int,
    boot_mode: str,
    top_m: int,
    corr_prune,
    random_state: int,
    method: str = "cefsplus",
) -> list[np.ndarray]:
    """Return bootstrap CEFS+ paths in cache-valid feature coordinates.

    This is the resampling half of ``AutoKConfig(k_method="stability")``: it
    reruns the greedy under ``B`` reweightings of the same rows so the
    *reproducibility* of the selected set can be measured as a function of k.
    Signals keep entering the path across replicates; the noise tail
    reshuffles. Call it directly when you want per-feature selection
    frequencies rather than a k. It is a reliability diagnostic, not a
    predictive or level-controlled discovery rule.

    Parameters
    ----------
    cache : FeatureCache
        Prebuilt Gaussian-copula cache from ``build_cache``. Duplicate
        non-synthetic feature names are rejected. A cached ``Rxx`` does not
        help here: replicate weights change, so each replicate recomputes its
        own panel correlations.
    y : array-like of shape (n_rows_original,)
        Target aligned to the original rows, not to the cached subsample.
    B : int
        Number of bootstrap replicates; one path per replicate.
    max_k : int
        Path depth per replicate, capped by the candidate count.
    boot_mode : {'bayes', 'half'}
        ``'bayes'`` multiplies the cache weights by iid ``Exp(1)`` draws,
        keeping every row; ``'half'`` zeroes a uniformly chosen half of the
        rows without replacement.
    top_m : int
        Screening width for each replicate-local candidate panel.
    corr_prune : float, None, or {'auto'}
        Correlation-pruning threshold forwarded to the panel builder.
    random_state : int
        Seed for the replicate streams; replicate ``b`` uses the ``b``-th
        child of ``SeedSequence(random_state)``.
    method : str, default 'cefsplus'
        Greedy selector rerun on each replicate: ``'cefsplus'``,
        ``'mrmr_quot'``, ``'mrmr_diff'``, ``'jmi'``, or ``'jmim'``.
        Only ``'cefsplus'`` returns
        an objective internally; the others return the path alone, which is
        all this function needs.

    Returns
    -------
    paths : list of ndarray
        ``B`` integer arrays of feature positions in cache-valid coordinates
        (columns of ``cache.Z``, so indices into ``cache.valid_cols``), each
        at most ``max_k`` long. A replicate whose weights sum to zero
        contributes an empty array.

    Raises
    ------
    ValueError
        If ``y`` does not have ``cache.n_rows_original`` rows, if
        ``boot_mode`` is neither ``'bayes'`` nor ``'half'``, if ``method`` is
        unknown, or if the cache fails its structural contract or carries
        duplicate feature names.

    See Also
    --------
    select_k_stability : Consumes these paths.
    null_objective_paths : Same replicate scaffolding for the permutation
        null.
    build_cache : Builds the ``FeatureCache`` this expects.

    Notes
    -----
    The marginal rank transform in ``cache.Z`` is held fixed and only
    re-standardized under the replicate weights, the same approximation class
    as ``xfit_mode='shared_z'``. The dominant cost is one
    ``top_m``-wide weighted correlation per replicate, so runtime scales as
    ``B * (panel correlation + greedy)`` and grows with rows rather than with
    the path depth; keep the cache row subsample in mind for large data.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import bootstrap_paths, build_cache
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> paths = bootstrap_paths(
    ...     cache,
    ...     y.to_numpy(),
    ...     B=6,
    ...     max_k=3,
    ...     boot_mode="bayes",
    ...     top_m=6,
    ...     corr_prune=None,
    ...     random_state=0,
    ... )
    >>> len(paths), paths[0].shape
    (6, (3,))
    >>> sorted({int(path[0]) for path in paths})
    [0]
    """
    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError(
            f"y has {y_arr.shape[0]} rows but cache was built from "
            f"{cache.n_rows_original} rows"
        )
    y_cache = y_arr[np.asarray(cache.row_idx, dtype=np.int64)]
    seeds = np.random.SeedSequence(random_state).spawn(int(B))
    paths: list[np.ndarray] = []
    for child in seeds:
        rng = np.random.default_rng(child)
        base_w = np.asarray(cache.sample_weight, dtype=np.float64)
        if boot_mode == "half":
            mask = np.zeros(base_w.shape[0], dtype=np.float64)
            chosen = rng.choice(base_w.shape[0], size=max(1, base_w.shape[0] // 2), replace=False)
            mask[chosen] = 1.0
            w_b = base_w * mask
        elif boot_mode == "bayes":
            w_b = base_w * rng.exponential(scale=1.0, size=base_w.shape[0])
        else:
            raise ValueError("boot_mode must be 'bayes' or 'half'")
        if float(np.sum(w_b)) <= 0.0:
            paths.append(np.empty(0, dtype=np.int64))
            continue
        zy_b = weighted_rank_gauss_1d(y_cache, w_b)
        panel = local_corr_panel(
            cache.Z,
            zy_b,
            w_b,
            top_m=top_m,
            corr_prune=corr_prune,
            method=method,
            local_standardize=True,
        )
        path, _objective = _run_panel_path(panel, int(max_k), method=method)
        paths.append(path)
    return paths


def _stability_phi_from_counts(counts: np.ndarray, *, B: int, k: int, p: int) -> float:
    denom = (k / p) * (1.0 - k / p) if 0 < k < p else 0.0
    if denom <= 0.0:
        return np.nan
    pi = np.asarray(counts, dtype=np.float64) / max(1, int(B))
    instability = np.mean((B / max(1, B - 1)) * pi * (1.0 - pi))
    return float(1.0 - instability / denom)


def _stability_phi_jackknife_se(indicators: np.ndarray, counts: np.ndarray, *, k: int, p: int) -> float:
    B = int(indicators.shape[0])
    if B < 2:
        return float("nan")
    loo = np.empty(B, dtype=np.float64)
    for b in range(B):
        loo[b] = _stability_phi_from_counts(
            counts - indicators[b],
            B=B - 1,
            k=k,
            p=p,
        )
    finite = loo[np.isfinite(loo)]
    if finite.size < 2:
        return float("nan")
    center = float(np.mean(finite))
    return float(np.sqrt((finite.size - 1) / finite.size * np.sum((finite - center) ** 2)))


def select_k_stability(
    paths: list[np.ndarray],
    p_valid: int,
    config: AutoKConfig,
) -> tuple[int, pd.DataFrame]:
    """Select k from chance-corrected bootstrap path stability.

    This is the rule behind ``AutoKConfig(k_method="stability")``. It is the
    only auto-k rule whose target is the *reliability of the feature list
    itself*: k is chosen where the selected set stops being reproducible under
    data perturbation, using the chance-corrected agreement of Nogueira,
    Sechidis and Brown (2018). Reach for it when the deliverable is a stable
    feature list rather than a level-controlled discovery set or a predictive
    size. It is experimental: it did not clear the Auto-K v2 accuracy gate for
    automatic sizing, so prefer it as a diagnostic (the per-k stability curve
    and Jaccard overlap are the useful output) over a default.

    Parameters
    ----------
    paths : list of ndarray
        Bootstrap paths in cache-valid feature coordinates, as returned by
        ``bootstrap_paths``. The evaluated depth is the shortest path length,
        capped by ``config.max_k``.
    p_valid : int
        Number of valid candidate features, i.e. the width of ``cache.Z``
        (``len(cache.valid_cols)``). It is the ``p`` in the chance correction,
        and it caps the identifiable depth at ``p_valid - 1`` because the
        correction is undefined at ``k = p``.
    config : AutoKConfig
        Must have ``k_method='stability'``. Reads ``stability_rule``,
        ``stability_pi`` (threshold rule only), ``min_k``, and ``max_k``.
        ``boot_B`` and ``boot_mode`` are consumed upstream by
        ``bootstrap_paths`` and only echoed into the diagnostics here.

    Returns
    -------
    selected_k : int
        Prefix length. Under ``stability_rule='max_one_se'`` the largest k
        within one jackknife standard error of peak stability; under
        ``'pi_threshold'`` the count of features whose selection frequency
        reaches ``stability_pi``, clamped by the stability plateau and the
        identifiable depth. ``0`` when no depth is identifiable, or when peak
        agreement is below the 0.5 floor and ``min_k <= 0``.
    diagnostics : DataFrame
        One row per k from 1 to the identifiable depth with ``k``, ``phi``
        (chance-corrected stability), ``phi_se`` (jackknife over replicates,
        NaN with fewer than two usable replicates), ``mean_jaccard`` (mean
        pairwise overlap of the prefix sets), ``selected``, ``boot_B``,
        ``boot_mode``, ``max_phi``, ``stability_floor_threshold``, and
        ``stopped_by`` (the rule name, ``'stability_floor'``, or
        ``'degenerate'``). ``max_phi``, ``stability_floor_threshold``, and
        ``stopped_by`` are also mirrored into ``diagnostics.attrs``. Empty
        when ``paths`` is empty or nothing is identifiable.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'stability'``.

    See Also
    --------
    bootstrap_paths : Builds the ``paths`` argument.
    select_k_posterior : Reports uncertainty about k instead of about the set.
    select_k_perm_gap : Permutation-calibrated discovery alternative.
    AutoKConfig : ``boot_B``, ``boot_mode``, ``stability_rule``,
        ``stability_pi``.

    Notes
    -----
    With ``pi_j(k)`` the fraction of replicates whose length-k prefix contains
    feature ``j``, stability is
    ``phi(k) = 1 - mean_j[B/(B-1) * pi_j (1 - pi_j)] / [(k/p)(1 - k/p)]``:
    1.0 when every replicate picks the same set, near 0 for independent
    picks. Peak agreement below ``0.5`` is treated as chance level and returns
    the zero-capable floor with ``stopped_by='stability_floor'``. Redundant
    feature blocks depress ``phi`` even when the choice is stable at block
    level, because within-block members swap between replicates; ``corr_prune``
    upstream is the mitigation. The scan is ``O(max_k * (B + B^2))`` for the
    pairwise overlap bookkeeping, plus a jackknife over replicates per k.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import AutoKConfig, bootstrap_paths, build_cache, select_k_stability
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> paths = bootstrap_paths(
    ...     cache,
    ...     y.to_numpy(),
    ...     B=6,
    ...     max_k=3,
    ...     boot_mode="bayes",
    ...     top_m=6,
    ...     corr_prune=None,
    ...     random_state=0,
    ... )
    >>> config = AutoKConfig(
    ...     k_method="stability", min_k=1, max_k=3, boot_B=6
    ... )
    >>> selected_k, diag = select_k_stability(paths, len(cache.valid_cols), config)
    >>> selected_k
    2
    >>> diag["stopped_by"].iloc[0]
    'max_one_se'
    >>> [round(float(value), 2) for value in diag["phi"]]
    [1.0, 1.0, 0.64]
    """
    validate_auto_k_config(config)
    if config.k_method != "stability":
        raise ValueError("select_k_stability requires AutoKConfig(k_method='stability')")
    if not paths:
        return 0, pd.DataFrame()
    max_len = min(int(config.max_k), min((len(path) for path in paths), default=0))
    max_identifiable = min(max_len, max(0, int(p_valid) - 1))
    if max_identifiable <= 0:
        return 0, pd.DataFrame()
    effective_min = max(1, min(int(config.min_k), max_identifiable))
    B = len(paths)
    p = int(p_valid)
    rows = []
    indicators = np.zeros((B, p), dtype=bool)
    counts = np.zeros(p, dtype=np.float64)
    set_sizes = np.zeros(B, dtype=np.int64)
    intersections = np.zeros((B, B), dtype=np.int64)
    normalized_paths = [np.asarray(path, dtype=np.int64) for path in paths]
    upper_i, upper_j = np.triu_indices(B, k=1)
    for k in range(1, max_identifiable + 1):
        for b, path in enumerate(normalized_paths):
            feature = int(path[k - 1])
            if feature < 0 or feature >= p or indicators[b, feature]:
                continue
            peers = np.flatnonzero(indicators[:, feature])
            if peers.size:
                intersections[b, peers] += 1
                intersections[peers, b] += 1
            indicators[b, feature] = True
            counts[feature] += 1.0
            set_sizes[b] += 1
        phi = _stability_phi_from_counts(counts, B=B, k=k, p=p)
        phi_se = _stability_phi_jackknife_se(indicators, counts, k=k, p=p)
        union_sizes = (
            set_sizes[upper_i]
            + set_sizes[upper_j]
            - intersections[upper_i, upper_j]
        )
        jaccards = np.ones(union_sizes.size, dtype=np.float64)
        nonempty = union_sizes > 0
        jaccards[nonempty] = (
            intersections[upper_i[nonempty], upper_j[nonempty]]
            / union_sizes[nonempty]
        )
        rows.append(
            {
                "k": k,
                "phi": float(phi),
                "phi_se": phi_se,
                "mean_jaccard": float(np.mean(jaccards)) if jaccards.size else 1.0,
            }
        )
    diag = pd.DataFrame(rows)
    finite_all = diag[np.isfinite(diag["phi"])]
    max_phi = float(finite_all["phi"].max()) if not finite_all.empty else float("nan")
    stopped_by = str(config.stability_rule)
    if config.stability_rule == "pi_threshold":
        raw_selected = int(np.sum(counts / max(1, B) >= float(config.stability_pi)))
        threshold_floor = 0 if int(config.min_k) <= 0 else effective_min
        finite = finite_all
        if finite.empty:
            selected_k = threshold_floor
            stopped_by = "degenerate"
        else:
            best = finite.sort_values(["phi", "k"], ascending=[False, False], kind="mergesort").iloc[0]
            best_phi = float(best["phi"])
            if best_phi < _STABILITY_PHI_FLOOR:
                selected_k = threshold_floor
                stopped_by = "stability_floor"
            else:
                tol = float(best.get("phi_se", np.nan))
                tol = 0.0 if not np.isfinite(tol) else tol
                plateau = finite[finite["phi"] >= best_phi - tol]
                plateau_cap = int(plateau["k"].max()) if not plateau.empty else max_identifiable
                selected_k = min(
                    max_identifiable,
                    max(threshold_floor, min(raw_selected, plateau_cap)),
                )
    else:
        finite = diag[np.isfinite(diag["phi"]) & (diag["k"] >= effective_min)]
        if finite.empty:
            selected_k = effective_min
            stopped_by = "degenerate"
        else:
            best = finite.sort_values(["phi", "k"], ascending=[False, False], kind="mergesort").iloc[0]
            if float(best["phi"]) < _STABILITY_PHI_FLOOR and int(config.min_k) <= 0:
                selected_k = 0
                stopped_by = "stability_floor"
            elif float(best["phi"]) < _STABILITY_PHI_FLOOR:
                selected_k = effective_min
                stopped_by = "stability_floor"
            else:
                tol = float(best.get("phi_se", np.nan))
                tol = 0.0 if not np.isfinite(tol) else tol
                eligible = finite[finite["phi"] >= float(best["phi"]) - tol]
                selected_k = int(eligible.sort_values("k", ascending=False, kind="mergesort").iloc[0]["k"])
    diag["selected"] = diag["k"] == selected_k
    diag["boot_B"] = B
    diag["boot_mode"] = config.boot_mode
    diag["max_phi"] = max_phi
    diag["stability_floor_threshold"] = _STABILITY_PHI_FLOOR
    diag["stopped_by"] = stopped_by
    diag.attrs["max_phi"] = max_phi
    diag.attrs["stability_floor_threshold"] = _STABILITY_PHI_FLOOR
    diag.attrs["stopped_by"] = stopped_by
    return selected_k, diag
