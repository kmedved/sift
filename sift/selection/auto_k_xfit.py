"""Cross-fitted automatic-k selectors in Gaussian-copula space."""

from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import digamma
from sklearn.model_selection import GroupKFold, KFold

from sift.estimators.copula import (
    gaussian_mi_from_corr,
    weighted_corr_with_vector,
    weighted_correlation_matrix,
    weighted_rank_gauss_1d,
)
from sift.selection.auto_k import (
    AutoKConfig,
    choose_k_from_score_curve,
    validate_auto_k_config,
    with_effective_k_bounds,
)
from sift.selection.auto_k_core import (
    build_score_curve_diagnostics,
    split_weights,
    time_holdout_split,
)
from sift.selection.cefsplus import (
    _gaussian_jmi_select,
    _gaussian_mrmr_select,
    cefsplus_loop_with_objective,
)
from sift.selection.panel import local_corr_panel, local_standardize, score_path_from_corr


def _cache_target_ranks(cache, y) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.shape[0] != cache.n_rows_original:
        raise ValueError("y length must match the cache's original row count")
    y_cache = y_arr[np.asarray(cache.row_idx, dtype=np.int64)]
    return weighted_rank_gauss_1d(y_cache, cache.sample_weight)


def _cache_aligned_metadata(cache, values, name: str):
    if values is None:
        return None
    arr = np.asarray(values).reshape(-1)
    n_cache = int(cache.Z.shape[0])
    if arr.shape[0] == n_cache:
        return arr
    if arr.shape[0] == int(cache.n_rows_original):
        return arr[np.asarray(cache.row_idx, dtype=np.int64)]
    raise ValueError(
        f"{name} has {arr.shape[0]} rows; expected cache rows ({n_cache}) "
        f"or original rows ({cache.n_rows_original})"
    )


def _kish_n_eff(w: np.ndarray) -> float:
    w_arr = np.asarray(w, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w_arr * w_arr))
    return float(np.sum(w_arr) ** 2 / denom) if denom > 0.0 else float("nan")


def _fold_splits(
    n_rows: int,
    config: AutoKConfig,
    *,
    groups=None,
    time=None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if config.strategy == "time_holdout":
        if time is None:
            raise ValueError(f"{config.k_method} with strategy='time_holdout' requires time")
        train_idx, val_idx = time_holdout_split(np.asarray(time).reshape(-1), config.val_frac)
        return [(train_idx.astype(np.int64), val_idx.astype(np.int64))]

    if config.strategy == "group_cv":
        if groups is None:
            raise ValueError(f"{config.k_method} with strategy='group_cv' requires groups")
        groups_arr = np.asarray(groups).reshape(-1)
        n_unique = len(np.unique(groups_arr))
        n_splits = min(int(config.xfit_folds), n_unique)
        if n_splits < 2:
            raise ValueError(f"group_cv requires at least 2 groups, got {n_unique}")
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (train.astype(np.int64), val.astype(np.int64))
            for train, val in splitter.split(np.arange(n_rows), groups=groups_arr)
        ]

    if config.strategy == "kfold":
        n_splits = min(int(config.xfit_folds), int(n_rows))
        if n_splits < 2:
            raise ValueError(f"kfold requires at least 2 rows, got {n_rows}")
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(config.random_state))
        return [
            (train.astype(np.int64), val.astype(np.int64))
            for train, val in splitter.split(np.arange(n_rows))
        ]

    raise ValueError(f"Unknown strategy: {config.strategy!r}")


def _select_local_path(panel, method: str, max_k: int) -> np.ndarray:
    k_actual = min(int(max_k), len(panel.cand))
    if k_actual <= 0:
        return np.empty(0, dtype=np.int64)
    if method == "cefsplus":
        local_path, _objective = cefsplus_loop_with_objective(
            panel.R,
            panel.r,
            k_actual,
            panel.rel,
        )
        return np.asarray(local_path, dtype=np.int64)
    if method in {"mrmr_quot", "mrmr_diff"}:
        return _gaussian_mrmr_select(
            panel.R,
            panel.rel,
            k_actual,
            use_quotient=method == "mrmr_quot",
        ).astype(np.int64)
    if method in {"jmi", "jmim"}:
        return _gaussian_jmi_select(
            panel.R,
            panel.r,
            panel.rel,
            k_actual,
            use_min=method == "jmim",
        ).astype(np.int64)
    raise ValueError(f"Unknown Gaussian selector method: {method!r}")


def _validation_corr_for_path(
    Z_val: np.ndarray,
    zy_val: np.ndarray,
    w_val: np.ndarray,
    path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if path.size == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64)
    Z_path = local_standardize(Z_val[:, path], w_val)
    zy_path = local_standardize(zy_val, w_val).reshape(-1)
    R_val = weighted_correlation_matrix(Z_path, w_val, backend="blas")
    r_val = weighted_corr_with_vector(Z_path, zy_path, w_val)
    return np.ascontiguousarray(R_val, dtype=np.float64), np.asarray(r_val, dtype=np.float64)


def _xfit_scores(
    objective_val: np.ndarray,
    *,
    n_eff_val: float,
) -> np.ndarray:
    ks = np.arange(1, len(objective_val) + 1, dtype=np.float64)
    nu = n_eff_val - ks - 1.0
    out = np.full(len(objective_val), np.nan, dtype=np.float64)
    valid = nu > 0.0
    if not bool(np.any(valid)):
        return out
    drift = digamma((nu[valid] + 1.0) / 2.0) - digamma(nu[valid] / 2.0)
    drift_cum = np.cumsum(drift)
    valid_positions = np.flatnonzero(valid)
    out[valid_positions] = objective_val[valid_positions] - drift_cum
    return out


def _solve_beta(R: np.ndarray, r: np.ndarray, ridge: float) -> np.ndarray:
    k = len(r)
    if k == 0:
        return np.empty(0, dtype=np.float64)
    base = np.asarray(R, dtype=np.float64)
    lam = float(ridge)
    for attempt in range(5):
        try:
            return np.linalg.solve(base + lam * np.eye(k), r)
        except np.linalg.LinAlgError:
            lam = max(1e-8, 10.0 * (lam if lam > 0.0 else 1e-8))
            if attempt == 0:
                warnings.warn(
                    "gaussian_cv train correlation solve was singular; increasing ridge.",
                    UserWarning,
                    stacklevel=5,
                )
    return np.linalg.pinv(base + lam * np.eye(k)) @ r


def _gaussian_cv_scores(
    R_train: np.ndarray,
    r_train: np.ndarray,
    R_val: np.ndarray,
    r_val: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    L = len(r_train)
    out = np.full(L, np.nan, dtype=np.float64)
    if L == 0:
        return out
    A = np.asarray(R_train, dtype=np.float64) + float(ridge) * np.eye(L)
    lam_extra = 0.0
    chol = None
    for attempt in range(5):
        try:
            chol = np.linalg.cholesky(A + lam_extra * np.eye(L))
            break
        except np.linalg.LinAlgError:
            lam_extra = max(1e-8, 10.0 * (lam_extra if lam_extra > 0.0 else 1e-8))
            if attempt == 0:
                warnings.warn(
                    "gaussian_cv train correlation Cholesky was singular; increasing ridge.",
                    UserWarning,
                    stacklevel=4,
                )
    if chol is None:
        for k in range(1, L + 1):
            beta = _solve_beta(R_train[:k, :k], r_train[:k], float(ridge) + lam_extra)
            out[k - 1] = float(
                1.0
                - 2.0 * float(beta @ r_val[:k])
                + float(beta @ R_val[:k, :k] @ beta)
            )
        return out

    for k in range(1, L + 1):
        Lk = chol[:k, :k]
        z = solve_triangular(Lk, r_train[:k], lower=True, check_finite=False)
        beta = solve_triangular(Lk.T, z, lower=False, check_finite=False)
        out[k - 1] = float(
            1.0
            - 2.0 * float(beta @ r_val[:k])
            + float(beta @ R_val[:k, :k] @ beta)
        )
    return out


def _fold_score_arrays(
    cache,
    y,
    *,
    config: AutoKConfig,
    groups=None,
    time=None,
    top_m: int,
    corr_prune,
    method: str,
    score_kind: str,
) -> tuple[list[np.ndarray], dict]:
    from sift.selection.knockoff_filter import (
        _reject_duplicate_feature_names,
        _validate_prebuilt_cache_structure,
    )

    validate_auto_k_config(config)
    _validate_prebuilt_cache_structure(cache)
    _reject_duplicate_feature_names(cache)
    if config.xfit_mode != "shared_z":
        raise ValueError(
            f"{config.k_method} xfit_mode='exact' requires fold-local cache rebuilding; "
            "function-style cache orchestration currently supports xfit_mode='shared_z'."
        )
    if score_kind not in {"xfit_objective", "gaussian_cv"}:
        raise ValueError("score_kind must be 'xfit_objective' or 'gaussian_cv'")

    Z = np.asarray(cache.Z)
    base_w = np.asarray(cache.sample_weight, dtype=np.float64).reshape(-1)
    if Z.shape[0] != base_w.shape[0]:
        raise ValueError("cache sample weights must match cache rows")
    zy = _cache_target_ranks(cache, y)
    groups_cache = _cache_aligned_metadata(cache, groups, "groups")
    time_cache = _cache_aligned_metadata(cache, time, "time")
    splits = _fold_splits(Z.shape[0], config, groups=groups_cache, time=time_cache)

    fold_scores: list[np.ndarray] = []
    fold_limits: list[int] = []
    fold_n_eff: list[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        w_train = split_weights(base_w, train_idx, f"{score_kind} train fold {fold_idx}")
        w_val = split_weights(base_w, val_idx, f"{score_kind} validation fold {fold_idx}")
        n_eff_val = _kish_n_eff(w_val)
        stat_limit = max(0, int(np.floor(n_eff_val)) - 2)
        if stat_limit <= 0:
            fold_scores.append(np.empty(0, dtype=np.float64))
            fold_limits.append(0)
            fold_n_eff.append(n_eff_val)
            continue

        panel = local_corr_panel(
            Z[train_idx],
            zy[train_idx],
            w_train,
            top_m=top_m,
            corr_prune=corr_prune,
            method=method,
            local_standardize=True,
        )
        local_path = _select_local_path(panel, method, min(int(config.max_k), stat_limit))
        if local_path.size == 0:
            fold_scores.append(np.empty(0, dtype=np.float64))
            fold_limits.append(0)
            fold_n_eff.append(n_eff_val)
            continue

        path = np.asarray(panel.cand[local_path], dtype=np.int64)
        L = min(path.size, int(config.max_k), stat_limit)
        path = path[:L]
        local_path = local_path[:L]
        R_train = np.ascontiguousarray(panel.R[np.ix_(local_path, local_path)], dtype=np.float64)
        r_train = np.asarray(panel.r[local_path], dtype=np.float64)
        R_val, r_val = _validation_corr_for_path(Z[val_idx], zy[val_idx], w_val, path)
        objective_val = score_path_from_corr(R_val, r_val)
        if score_kind == "xfit_objective":
            scores = _xfit_scores(objective_val, n_eff_val=n_eff_val)
        else:
            scores = _gaussian_cv_scores(
                R_train,
                r_train,
                R_val,
                r_val,
                ridge=float(config.xfit_ridge),
            )
        fold_scores.append(scores)
        fold_limits.append(int(len(scores)))
        fold_n_eff.append(n_eff_val)

    extra = {
        "xfit_mode": config.xfit_mode,
        "xfit_folds": len(splits),
        "fold_max_k": tuple(fold_limits),
        "fold_n_eff": tuple(float(v) for v in fold_n_eff),
    }
    return fold_scores, extra


def _curve_from_fold_scores(
    fold_scores: list[np.ndarray],
    config: AutoKConfig,
    *,
    extra: dict,
    score_kind: str,
) -> pd.DataFrame:
    healthy_scores = [
        np.asarray(scores, dtype=np.float64)
        for scores in fold_scores
        if len(scores) > 0 and bool(np.isfinite(scores).any())
    ]
    dropped = len(fold_scores) - len(healthy_scores)
    if dropped and len(healthy_scores) >= 2:
        warnings.warn(
            f"{score_kind} dropped {dropped} degenerate fold(s) with no finite scores.",
            UserWarning,
            stacklevel=3,
        )
    elif dropped and len(healthy_scores) < 2:
        warnings.warn(
            f"{score_kind} has only {len(healthy_scores)} healthy fold(s) after dropping "
            "degenerate folds; falling back to the method floor.",
            UserWarning,
            stacklevel=3,
        )
        diag = pd.DataFrame()
        diag.attrs["stopped_by"] = "degenerate_folds"
        diag.attrs["healthy_folds"] = len(healthy_scores)
        diag.attrs["dropped_folds"] = dropped
        return diag
    if not healthy_scores:
        diag = pd.DataFrame()
        diag.attrs["stopped_by"] = "degenerate_folds"
        diag.attrs["healthy_folds"] = 0
        diag.attrs["dropped_folds"] = dropped
        return diag
    max_common = min((len(scores) for scores in healthy_scores), default=0)
    if max_common <= 0:
        diag = pd.DataFrame()
        diag.attrs["stopped_by"] = "degenerate_folds"
        diag.attrs["healthy_folds"] = len(healthy_scores)
        diag.attrs["dropped_folds"] = dropped
        return diag
    min_k = max(1, min(int(config.min_k), max_common))
    split_scores = {
        k: [float(scores[k - 1]) for scores in healthy_scores]
        for k in range(min_k, max_common + 1)
    }
    diag = build_score_curve_diagnostics(list(range(min_k, max_common + 1)), split_scores)
    diag["score_kind"] = score_kind
    diag["xfit_mode"] = extra["xfit_mode"]
    diag["xfit_folds"] = len(healthy_scores)
    diag["fold_max_k"] = (extra["fold_max_k"],) * len(diag)
    diag["fold_n_eff"] = (extra["fold_n_eff"],) * len(diag)
    diag["dropped_folds"] = dropped
    return diag


def xfit_objective_curves(
    cache,
    y,
    *,
    config: AutoKConfig,
    groups=None,
    time=None,
    top_m: int,
    corr_prune,
    method: str,
) -> pd.DataFrame:
    """Return a cross-fitted, drift-debiased objective score curve.

    This is the curve half of ``AutoKConfig(k_method="xfit_objective")``: the
    honest version of the objective path, at correlation-math cost. Each fold
    builds its own greedy path on fold-train rows, scores that path's
    objective on fold-validation rows, and subtracts the exact Gaussian null
    drift so that a null step contributes about zero instead of a positive
    bias. It is a discovery-flavored curve -- it measures conditional
    *information*, with no downstream model, metric, or hyperparameter -- and
    it is dense in k, so every prefix length is scored. It is experimental:
    the Auto-K v2 campaign failed it for automatic sizing.

    Parameters
    ----------
    cache : FeatureCache
        Prebuilt Gaussian-copula cache from ``build_cache``. Duplicate
        non-synthetic feature names are rejected and the cache weights must
        match the cache rows.
    y : array-like of shape (n_rows_original,)
        Target aligned to the original rows, not to the cached subsample.
    config : AutoKConfig
        Must have ``k_method='xfit_objective'`` and ``xfit_mode='shared_z'``.
        Reads ``strategy``, ``xfit_folds``, ``val_frac`` (time holdout),
        ``random_state`` (shuffled k-fold), ``min_k``, and ``max_k``.
    groups : array-like, optional
        Group labels, accepted either at cache-row length or at original-row
        length. Required for ``strategy='group_cv'``.
    time : array-like, optional
        Row timestamps, accepted at either length. Required for
        ``strategy='time_holdout'``.
    top_m : int
        Screening width for each fold-local candidate panel.
    corr_prune : float, None, or {'auto'}
        Correlation-pruning threshold forwarded to the fold panel builder.
    method : {'cefsplus', 'mrmr_quot', 'mrmr_diff', 'jmi', 'jmim'}
        Greedy selector used to build each fold-train path.

    Returns
    -------
    curves : DataFrame
        Higher-is-better score curve with one row per k from the effective
        floor to the deepest depth common to all healthy folds:
        ``k``, ``score``, ``score_mean``, ``score_std``, ``score_se``,
        ``n_splits``, ``n_finite``, ``split_scores``, ``score_kind``
        (``'xfit_objective'``), ``xfit_mode``, ``xfit_folds``,
        ``fold_max_k``, ``fold_n_eff``, ``dropped_folds``, and ``debias``
        (True). On a degenerate run the frame is empty and
        ``curves.attrs`` carries ``stopped_by='degenerate_folds'``,
        ``healthy_folds``, and ``dropped_folds``.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'xfit_objective'``; if
        ``xfit_mode='exact'`` (fold-local cache rebuilding is not available
        from cache-only orchestration); if the requested strategy lacks its
        ``time`` or ``groups`` context, has fewer than two groups or rows, or
        is unknown; if ``groups``/``time`` match neither row count; or if the
        cache fails its structural contract.

    Warns
    -----
    UserWarning
        When folds with no finite score are dropped, and when fewer than two
        healthy folds remain (an empty curve is returned so the caller falls
        back to its method floor).

    See Also
    --------
    select_k_xfit_objective : Turns this curve into a k.
    gaussian_cv_curves : Same folds and paths, predictive risk instead.
    compute_objective_for_path : In-sample objective on one full cache.

    Notes
    -----
    For fold ``f`` and step ``t`` the debias term is
    ``digamma((nu + 1)/2) - digamma(nu/2)`` with
    ``nu = n_eff_val - t - 1``, the exact expectation of the null
    log-gain (about ``1/nu`` for large ``nu``); the fold score is the
    validation objective minus the cumulative drift. Each fold is capped at
    ``floor(n_eff_val) - 2`` steps, where ``n_eff_val`` is the Kish size of
    that fold's validation weights. ``xfit_mode='shared_z'`` keeps the
    full-cache marginal ranks and re-standardizes them inside each fold, so
    leakage is limited to the marginal transform having seen every row. Cost
    is ``folds x (panel + greedy + O(K^2) objective)`` with no model fits.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import AutoKConfig, build_cache, xfit_objective_curves
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> config = AutoKConfig(
    ...     k_method="xfit_objective",
    ...     strategy="kfold",
    ...     xfit_folds=3,
    ...     min_k=1,
    ...     max_k=4,
    ... )
    >>> curves = xfit_objective_curves(
    ...     cache,
    ...     y.to_numpy(),
    ...     config=config,
    ...     top_m=6,
    ...     corr_prune=None,
    ...     method="cefsplus",
    ... )
    >>> curves["k"].tolist()
    [1, 2, 3, 4]
    >>> curves["score_kind"].iloc[0], bool(curves["debias"].iloc[0])
    ('xfit_objective', True)
    >>> bool(curves["score_mean"].iloc[1] > curves["score_mean"].iloc[0])
    True
    """
    if config.k_method != "xfit_objective":
        raise ValueError("xfit_objective_curves requires AutoKConfig(k_method='xfit_objective')")
    fold_scores, extra = _fold_score_arrays(
        cache,
        y,
        config=config,
        groups=groups,
        time=time,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
        score_kind="xfit_objective",
    )
    diag = _curve_from_fold_scores(
        fold_scores,
        config,
        extra=extra,
        score_kind="xfit_objective",
    )
    if not diag.empty:
        diag["debias"] = True
    return diag


def gaussian_cv_curves(
    cache,
    y,
    *,
    config: AutoKConfig,
    groups=None,
    time=None,
    top_m: int,
    corr_prune,
    method: str,
) -> pd.DataFrame:
    """Return a closed-form cross-validated Gaussian linear risk curve.

    This is the curve half of ``AutoKConfig(k_method="gaussian_cv")`` and of
    the ``AutoKConfig.predictive(...)`` preset. It reproduces the ``evaluate``
    semantics -- out-of-sample predictive risk against k -- without fitting a
    single sklearn model: in copula space the prefix model is linear, its
    coefficients come in closed form from the fold-train correlations, and its
    validation risk is a quadratic form in the fold-validation correlations.
    That makes it the predictive-sufficiency rule to reach for: honest
    fold-train paths, dense in k, no alpha noise, and a standard error worth
    trusting, which is why the parsimony rules (``one_se``, ``plateau``,
    ``tolerance``) are usable here.

    Parameters
    ----------
    cache : FeatureCache
        Prebuilt Gaussian-copula cache from ``build_cache``. Duplicate
        non-synthetic feature names are rejected and the cache weights must
        match the cache rows.
    y : array-like of shape (n_rows_original,)
        Target aligned to the original rows, not to the cached subsample.
    config : AutoKConfig
        Must have ``k_method='gaussian_cv'`` and ``xfit_mode='shared_z'``.
        Reads ``strategy``, ``xfit_folds``, ``xfit_ridge``, ``val_frac``
        (time holdout), ``random_state`` (shuffled k-fold), ``min_k``, and
        ``max_k``.
    groups : array-like, optional
        Group labels, accepted either at cache-row length or at original-row
        length. Required for ``strategy='group_cv'``.
    time : array-like, optional
        Row timestamps, accepted at either length. Required for
        ``strategy='time_holdout'``.
    top_m : int
        Screening width for each fold-local candidate panel.
    corr_prune : float, None, or {'auto'}
        Correlation-pruning threshold forwarded to the fold panel builder.
    method : {'cefsplus', 'mrmr_quot', 'mrmr_diff', 'jmi', 'jmim'}
        Greedy selector used to build each fold-train path.

    Returns
    -------
    curves : DataFrame
        Lower-is-better risk curve with one row per k from the effective floor
        to the deepest depth common to all healthy folds: ``k``, ``score``,
        ``score_mean``, ``score_std``, ``score_se``, ``n_splits``,
        ``n_finite``, ``split_scores``, ``score_kind`` (``'gaussian_cv'``),
        ``xfit_mode``, ``xfit_folds``, ``fold_max_k``, ``fold_n_eff``,
        ``dropped_folds``, ``proxy`` (``'gaussian_linear_copula'``), and
        ``xfit_ridge``. On a degenerate run the frame is empty and
        ``curves.attrs`` carries ``stopped_by='degenerate_folds'``,
        ``healthy_folds``, and ``dropped_folds``.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'gaussian_cv'``; if
        ``xfit_mode='exact'``; if the requested strategy lacks its ``time`` or
        ``groups`` context, has fewer than two groups or rows, or is unknown;
        if ``groups``/``time`` match neither row count; or if the cache fails
        its structural contract.

    Warns
    -----
    UserWarning
        When the fold-train correlation matrix is singular and the ridge is
        increased (Cholesky first, then the per-k solve), when folds with no
        finite score are dropped, and when fewer than two healthy folds remain
        (an empty curve is returned).

    See Also
    --------
    select_k_gaussian_cv : Turns this curve into a k.
    xfit_objective_curves : Same folds and paths, information instead of risk.
    select_k_auto : The model-fitting ``evaluate`` rule this replaces.

    Notes
    -----
    Per fold, ``beta_k = (R_train[:k, :k] + lambda I)^-1 r_train[:k]`` and
    ``risk(k) = 1 - 2 beta_k' r_val[:k] + beta_k' R_val[:k, :k] beta_k``, the
    exact out-of-sample normalized MSE of the Gaussian-model predictor -- the
    population quantity a RidgeCV/RMSE curve estimates noisily. All k are
    obtained from one incremental Cholesky of the fold-train matrix, so the
    whole curve costs ``O(K^3/3)`` flops per fold; a singular matrix falls
    back to per-k solves with an inflated ridge. Each fold is capped at
    ``floor(n_eff_val) - 2`` steps. The proxy is copula-linear, so treat the
    curve as a well-behaved stand-in for, not a measurement of, a nonlinear
    downstream model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import AutoKConfig, build_cache, gaussian_cv_curves
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> config = AutoKConfig(
    ...     k_method="gaussian_cv",
    ...     strategy="kfold",
    ...     xfit_folds=3,
    ...     min_k=1,
    ...     max_k=4,
    ... )
    >>> curves = gaussian_cv_curves(
    ...     cache,
    ...     y.to_numpy(),
    ...     config=config,
    ...     top_m=6,
    ...     corr_prune=None,
    ...     method="cefsplus",
    ... )
    >>> curves["k"].tolist()
    [1, 2, 3, 4]
    >>> curves["proxy"].iloc[0]
    'gaussian_linear_copula'
    >>> bool(curves["score_mean"].iloc[1] < curves["score_mean"].iloc[0])
    True
    """
    if config.k_method != "gaussian_cv":
        raise ValueError("gaussian_cv_curves requires AutoKConfig(k_method='gaussian_cv')")
    fold_scores, extra = _fold_score_arrays(
        cache,
        y,
        config=config,
        groups=groups,
        time=time,
        top_m=top_m,
        corr_prune=corr_prune,
        method=method,
        score_kind="gaussian_cv",
    )
    diag = _curve_from_fold_scores(
        fold_scores,
        config,
        extra=extra,
        score_kind="gaussian_cv",
    )
    if not diag.empty:
        diag["proxy"] = "gaussian_linear_copula"
        diag["xfit_ridge"] = float(config.xfit_ridge)
    return diag


def select_k_xfit_objective(
    curves: pd.DataFrame,
    config: AutoKConfig,
) -> tuple[int, pd.DataFrame]:
    """Select k from a cross-fitted objective curve.

    This is the selection half of ``AutoKConfig(k_method="xfit_objective")``,
    consuming the higher-is-better curve from ``xfit_objective_curves``.
    Because the curve measures debiased conditional information rather than
    predictive risk, this is a discovery-flavored sizing rule; a null-guard
    check refuses to name a k at all when the best fold-mean score is not
    distinguishable from zero. It is experimental for automatic sizing.

    Parameters
    ----------
    curves : DataFrame
        Score curve from ``xfit_objective_curves``, with at least ``k``,
        ``score_mean``, and ``score_se``. Higher is better.
    config : AutoKConfig
        Must have ``k_method='xfit_objective'``. Reads ``selection_rule`` and
        its tolerance fields, ``min_k``, and ``max_k``. ``selection_rule
        ='best'`` is deliberately upgraded to ``'one_se'`` here, since
        fold-level curves give a usable standard error and the raw argmax of a
        flat curve is close to a coin flip; the request is preserved in
        ``selection_rule_requested``.

    Returns
    -------
    selected_k : int
        Selected prefix length. ``max(0, min_k)`` when ``curves`` is empty;
        ``0`` (or the floor when ``min_k > 0``) when the null guard fires.
    diagnostics : DataFrame
        A copy of ``curves`` plus ``best_k``, ``best_score``,
        ``selection_rule``, ``selection_rule_effective``,
        ``selection_rule_requested``, ``one_se_unavailable``,
        ``within_tolerance``, ``in_selected_plateau``, and ``selected``. When
        the null guard fires it also carries ``stopped_by='null_guard'``,
        ``null_guard_z``, and ``null_guard_threshold``.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'xfit_objective'``, or if the curve
        lacks a ``k`` column.

    Warns
    -----
    UserWarning
        When every candidate score is non-finite (the method floor is
        returned), and when ``'one_se'`` has no usable standard error and
        falls back to ``'best'``.

    See Also
    --------
    xfit_objective_curves : Builds the ``curves`` argument.
    select_k_gaussian_cv : Predictive-risk sibling on the same folds.
    choose_k_from_score_curve : Shared rule engine.

    Notes
    -----
    The null guard compares the best fold-mean score with
    ``2.5 * score_se`` at that k: a debiased objective whose peak is inside
    2.5 standard errors of zero is evidence that no prefix carries
    information, so the rule returns the zero-capable floor rather than the
    argmax of noise. Otherwise selection is delegated to
    ``choose_k_from_score_curve`` with ``lower_is_better=False`` and the k
    bounds clamped to ``[max(1, min(min_k, max evaluated k)), max evaluated
    k]``. Cost is ``O(len(curves))``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import (
    ...     AutoKConfig,
    ...     build_cache,
    ...     select_k_xfit_objective,
    ...     xfit_objective_curves,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> config = AutoKConfig(
    ...     k_method="xfit_objective",
    ...     strategy="kfold",
    ...     xfit_folds=3,
    ...     min_k=1,
    ...     max_k=4,
    ... )
    >>> curves = xfit_objective_curves(
    ...     cache,
    ...     y.to_numpy(),
    ...     config=config,
    ...     top_m=6,
    ...     corr_prune=None,
    ...     method="cefsplus",
    ... )
    >>> selected_k, diag = select_k_xfit_objective(curves, config)
    >>> selected_k
    2
    >>> diag["selection_rule_requested"].iloc[0], diag["selection_rule_effective"].iloc[0]
    ('best', 'one_se')
    """
    validate_auto_k_config(config)
    if config.k_method != "xfit_objective":
        raise ValueError("select_k_xfit_objective requires AutoKConfig(k_method='xfit_objective')")
    if curves.empty:
        diag = curves.copy()
        return max(0, int(config.min_k)), diag
    max_k = int(curves["k"].max())
    selectable_floor = max(1, min(int(config.min_k), max_k))
    finite = curves[np.isfinite(curves["score_mean"])].copy()
    if not finite.empty:
        best = finite.sort_values(["score_mean", "k"], ascending=[False, True], kind="mergesort").iloc[0]
        best_score = float(best["score_mean"])
        best_se = float(best.get("score_se", np.nan))
        null_guard_z = 2.5
        if np.isfinite(best_se) and best_score <= null_guard_z * best_se:
            selected_k = 0 if int(config.min_k) <= 0 else selectable_floor
            diag = curves.copy()
            diag["best_k"] = int(best["k"])
            diag["best_score"] = best_score
            diag["selection_rule"] = config.selection_rule
            diag["selection_rule_effective"] = "null_guard"
            diag["one_se_unavailable"] = False
            diag["within_tolerance"] = False
            diag["in_selected_plateau"] = False
            diag["selected"] = diag["k"] == selected_k
            diag["stopped_by"] = "null_guard"
            diag["null_guard_z"] = null_guard_z
            diag["null_guard_threshold"] = null_guard_z * best_se
            return selected_k, diag

    rule_config = config
    if config.selection_rule == "best":
        rule_config = replace(config, selection_rule="one_se")
    bounded = with_effective_k_bounds(rule_config, min_k=selectable_floor, max_k=max_k)
    selected_k, diag = choose_k_from_score_curve(curves, bounded, lower_is_better=False)
    diag["selection_rule_requested"] = config.selection_rule
    return selected_k, diag


def select_k_gaussian_cv(
    curves: pd.DataFrame,
    config: AutoKConfig,
) -> tuple[int, pd.DataFrame]:
    """Select k from a closed-form Gaussian CV risk curve.

    This is the selection half of ``AutoKConfig(k_method="gaussian_cv")`` and
    of the ``AutoKConfig.predictive(...)`` preset, consuming the
    lower-is-better risk curve from ``gaussian_cv_curves``. Because the curve
    is out-of-sample predictive risk, this is the predictive-sufficiency rule:
    ``selection_rule`` is honored exactly as requested, and the standard error
    behind ``'one_se'`` is trustworthy here because it comes from fold curves
    with no model-fitting noise inside them.

    Parameters
    ----------
    curves : DataFrame
        Risk curve from ``gaussian_cv_curves``, with at least ``k`` and
        ``score_mean``. Lower is better.
    config : AutoKConfig
        Must have ``k_method='gaussian_cv'``. Reads ``selection_rule``,
        ``one_se_multiplier``, ``score_abs_tol``, ``score_rel_tol``,
        ``plateau_prefer``, ``plateau_min_points``, ``min_k``, and ``max_k``.

    Returns
    -------
    selected_k : int
        Selected prefix length, at least ``max(1, min(min_k, max evaluated
        k))``. ``max(0, config.min_k)`` when ``curves`` is empty.
    diagnostics : DataFrame
        A copy of ``curves`` plus ``best_k``, ``best_score``,
        ``selection_rule``, ``selection_rule_effective``,
        ``one_se_unavailable``, ``within_tolerance``,
        ``in_selected_plateau``, and ``selected``. Returned unchanged when
        ``curves`` is empty.

    Raises
    ------
    ValueError
        If ``config.k_method`` is not ``'gaussian_cv'``, if the curve lacks a
        ``k`` column, or if ``selection_rule`` is unknown.

    Warns
    -----
    UserWarning
        When every candidate score is non-finite (the method floor is
        returned), and when ``'one_se'`` has no usable standard error and
        falls back to ``'best'`` -- which is what a single time-holdout split
        always does.

    See Also
    --------
    gaussian_cv_curves : Builds the ``curves`` argument.
    select_k_xfit_objective : Information-flavored sibling on the same folds.
    choose_k_from_score_curve : Shared rule engine.
    select_k_auto : The model-fitting ``evaluate`` rule this replaces.

    Notes
    -----
    Selection is delegated to ``choose_k_from_score_curve`` with
    ``lower_is_better=True`` and the k bounds clamped to the evaluated range,
    so ``'best'``, ``'one_se'``, ``'plateau'``, and ``'tolerance'`` behave
    exactly as they do for ``k_method='evaluate'``. Unlike
    ``select_k_xfit_objective`` there is no null guard and no implicit rule
    upgrade: a flat risk curve returns its argmin under ``'best'``, so choose
    ``'one_se'`` when parsimony matters. Cost is ``O(len(curves))``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from sift import (
    ...     AutoKConfig,
    ...     build_cache,
    ...     gaussian_cv_curves,
    ...     select_k_gaussian_cv,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))
    >>> y = X["a"] + 0.7 * X["b"] + 0.2 * rng.normal(size=200)
    >>> cache = build_cache(X, compute_Rxx=True)
    >>> config = AutoKConfig(
    ...     k_method="gaussian_cv",
    ...     strategy="kfold",
    ...     xfit_folds=3,
    ...     min_k=1,
    ...     max_k=4,
    ... )
    >>> curves = gaussian_cv_curves(
    ...     cache,
    ...     y.to_numpy(),
    ...     config=config,
    ...     top_m=6,
    ...     corr_prune=None,
    ...     method="cefsplus",
    ... )
    >>> selected_k, diag = select_k_gaussian_cv(curves, config)
    >>> selected_k
    3
    >>> diag["selection_rule_effective"].iloc[0]
    'best'
    """
    validate_auto_k_config(config)
    if config.k_method != "gaussian_cv":
        raise ValueError("select_k_gaussian_cv requires AutoKConfig(k_method='gaussian_cv')")
    if curves.empty:
        return max(0, int(config.min_k)), curves.copy()
    max_k = int(curves["k"].max())
    floor = max(1, min(int(config.min_k), max_k))
    bounded = with_effective_k_bounds(config, min_k=floor, max_k=max_k)
    return choose_k_from_score_curve(curves, bounded, lower_is_better=True)


__all__ = [
    "gaussian_cv_curves",
    "select_k_gaussian_cv",
    "select_k_xfit_objective",
    "xfit_objective_curves",
]
