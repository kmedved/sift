"""Automatic k selection for filter methods."""

from __future__ import annotations

import importlib.util
from typing import Any as Any, List, Literal, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold

from sift._metadata import resolve_row_metadata
from sift._preprocess import (
    LeaveOneOutLogitEncoder,
    TargetCVEncoder,
    ensure_weights,
    reject_datetime_like_features,
    suppress_category_encoder_pandas_warnings,
)
from sift.selection.within import require_within_context
from sift.selection.auto_k_config import (
    AutoKConfig as AutoKConfig,
    _NONNEGATIVE_INT_FIELDS as _NONNEGATIVE_INT_FIELDS,
    _POSITIVE_INT_FIELDS as _POSITIVE_INT_FIELDS,
    _REAL_TYPES as _REAL_TYPES,
    _VALID_BINARY_OBJECTIVE_MODES as _VALID_BINARY_OBJECTIVE_MODES,
    _VALID_BOOT_MODES as _VALID_BOOT_MODES,
    _VALID_CONSENSUS_METHODS as _VALID_CONSENSUS_METHODS,
    _VALID_GAP_RULES as _VALID_GAP_RULES,
    _VALID_K_METHODS as _VALID_K_METHODS,
    _VALID_KNOCKOFF_RETURNS as _VALID_KNOCKOFF_RETURNS,
    _VALID_KNOCKOFF_S_METHODS as _VALID_KNOCKOFF_S_METHODS,
    _VALID_M_MODES as _VALID_M_MODES,
    _VALID_N_EFF_MODES as _VALID_N_EFF_MODES,
    _VALID_OBJECTIVE_PENALTIES as _VALID_OBJECTIVE_PENALTIES,
    _VALID_PERM_NULLS as _VALID_PERM_NULLS,
    _VALID_PLATEAU_PREFERS as _VALID_PLATEAU_PREFERS,
    _VALID_POSTERIOR_PICKS as _VALID_POSTERIOR_PICKS,
    _VALID_SELECTION_RULES as _VALID_SELECTION_RULES,
    _VALID_STABILITY_RULES as _VALID_STABILITY_RULES,
    _VALID_STRATEGIES as _VALID_STRATEGIES,
    _VALID_XFIT_MODES as _VALID_XFIT_MODES,
    _WARN_UNUSED_METHOD_FIELDS as _WARN_UNUSED_METHOD_FIELDS,
    _auto_k_method_tags as _auto_k_method_tags,
    _ensure_supported_auto_k_mode as _ensure_supported_auto_k_mode,
    _is_real_number as _is_real_number,
    _suppress_auto_k_unused_field_warnings as _suppress_auto_k_unused_field_warnings,
    _warn_unused_method_fields as _warn_unused_method_fields,
    resolve_auto_k_config as resolve_auto_k_config,
    validate_auto_k_config as validate_auto_k_config,
    with_effective_k_bounds as with_effective_k_bounds,
)
# `_DEFAULT_AUTOK_CONFIG` stays in auto_k_config only. It is lazily rebound
# there, has no consumers, and a facade alias would be silently stale.
from sift.selection.auto_k_core import (
    build_k_grid,
    build_score_curve_diagnostics,
    evaluate_numeric_prefixes,
    resolve_metric,
    split_weights,
    time_holdout_split,
)
from sift.selection.auto_k_elbow import select_k_elbow as select_k_elbow
from sift.selection.auto_k_objective import (
    _log_comb as _log_comb,
    _objective_weight_diagnostics as _objective_weight_diagnostics,
    _penalty_array as _penalty_array,
    _penalty_weight as _penalty_weight,
    _resolve_ebic_gamma as _resolve_ebic_gamma,
    _resolve_n_eff_mode as _resolve_n_eff_mode,
)
from sift.selection.auto_k_path import compute_objective_for_path as compute_objective_for_path
from sift.selection.auto_k_score import (
    _RULE_SELECTORS as _RULE_SELECTORS,
    _choose_best_rule as _choose_best_rule,
    _choose_one_se_rule as _choose_one_se_rule,
    _choose_plateau_rule as _choose_plateau_rule,
    _choose_tolerance_rule as _choose_tolerance_rule,
    _mark_tolerance as _mark_tolerance,
    _score_curve_tolerance as _score_curve_tolerance,
    _selected_plateau_ks as _selected_plateau_ks,
)
from sift.scoring import is_sklearn_scorer, sklearn_scorer_label


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
    within: str | None = None,
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
            .columns.tolist()
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

    if within is not None:
        from sift.selection.within import (
            as_float_feature_matrix,
            fit_within_transform,
            restore_feature_matrix,
        )

        Xtr_num, tr_template = as_float_feature_matrix(Xtr_df)
        Xva_num, va_template = as_float_feature_matrix(Xva_df)
        g_tr = None if groups is None else groups[train_idx]
        g_va = None if groups is None else groups[val_idx]
        t_tr = None if time is None else time[train_idx]
        t_va = None if time is None else time[val_idx]
        fitted = fit_within_transform(within, Xtr_num, ytr, g_tr, t_tr, wtr)
        Xtr_num, ytr = fitted.transform(Xtr_num, ytr, g_tr, t_tr)
        Xva_num, yva = fitted.transform(Xva_num, yva, g_va, t_va)
        Xtr_df = restore_feature_matrix(tr_template, Xtr_num)
        Xva_df = restore_feature_matrix(va_template, Xva_num)

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
    base_features: Optional[List] = None,
    within: str | None = None,
    prefix_sizes: Optional[Sequence[int]] = None,
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
    base_features : list, optional
        Features always present in every evaluated model, in caller order.
        Prefix length ``k`` then counts *additional* ``feature_path``
        entries: the fitted columns are ``base_features + feature_path[:k]``.
        ``min_k``/``max_k`` and the returned ``best_k`` stay in that
        additional-discovery unit. When omitted, behavior is unchanged.
    prefix_sizes : sequence of int, optional
        Cumulative raw widths of ``feature_path`` after 1, 2, ... additional
        blocks. When omitted, each path entry is one step (legacy column
        prefixes). When provided, ``k``/``min_k``/``max_k``/diagnostics count
        additional blocks; each evaluated model uses
        ``base_features + feature_path[:prefix_sizes[k-1]]``. Scores are
        mapped back to those block steps; do not slice a raw-column path
        at a block ``k``.
    within : {'groups', 'two_way'} or None, default None
        Fold-local panel demeaning applied after encoding and before the
        prefix proxy model. Regression only. Means are fit on training rows
        only; unseen entities fall back to the training grand mean.
        Datetime/timedelta path columns are rejected before conversion.

    Returns
    -------
    best_k : int
        Selected prefix length in additional-discovery units (columns, or
        additional blocks when ``prefix_sizes`` is set), ``0`` when the path
        resolves to no usable feature.
    features : list of str
        The resolved discovery prefix: ``feature_path[:best_k]`` without
        ``prefix_sizes``, or ``feature_path[:prefix_sizes[best_k-1]]`` when
        block widths are supplied. ``base_features`` are not included here.
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
        are available, if ``strategy`` is unknown, or if ``within`` is set
        with a non-regression ``task``, missing ``groups``/``time``, or
        datetime/timedelta features.
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
    resolved_within = require_within_context(
        within,
        task=task,
        groups=groups,
        time=time,
    )

    base_valid: List = []
    base_seen: set = set()
    for name in list(base_features or []):
        if name not in X.columns:
            continue
        if name in base_seen:
            continue
        base_valid.append(name)
        base_seen.add(name)
    n_base = len(base_valid)

    if not feature_path and n_base == 0:
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
    valid_features = [f for f in feature_path if f in X.columns and f not in base_seen]
    if not valid_features and n_base == 0:
        return 0, [], pd.DataFrame()

    step_widths: list[int] | None = None
    if prefix_sizes is not None:
        step_widths = [int(w) for w in prefix_sizes if int(w) > 0]
        if not step_widths:
            if n_base == 0:
                return 0, [], pd.DataFrame()
            max_k = 0
        else:
            clipped: list[int] = []
            last = 0
            for width in step_widths:
                raw = min(int(width), len(valid_features))
                if raw <= last:
                    continue
                clipped.append(raw)
                last = raw
                if last >= len(valid_features):
                    break
            step_widths = clipped
            max_k = min(int(config.max_k), len(step_widths))
    else:
        max_k = min(int(config.max_k), len(valid_features))
    if n_base:
        min_k = max(0, min(int(config.min_k), max_k))
    else:
        min_k = max(1, min(int(config.min_k), max_k)) if max_k else 0
        if not valid_features:
            return 0, [], pd.DataFrame()
    if step_widths is None:
        valid_features = valid_features[:max_k]
        k_grid = build_k_grid(min_k, max_k)
        k_grid_eval = [int(k) + n_base for k in k_grid]
    else:
        k_grid = build_k_grid(min_k, max_k)
        k_grid_eval = [
            n_base + int(step_widths[int(k) - 1]) if int(k) > 0 else n_base
            for k in k_grid
        ]
    eval_to_step = {int(width): int(step) for step, width in zip(k_grid, k_grid_eval)}

    X_path_df = X[base_valid + valid_features]
    if resolved_within is not None:
        reject_datetime_like_features(X_path_df)

    metric = resolve_metric(config.metric, task)
    eval_kwargs = {
        "X_path_df": X_path_df,
        "valid_features": valid_features,
        "y_arr": y_arr,
        "w_arr": w_arr,
        "task": task,
        "metric": metric,
        "k_grid": k_grid_eval,
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
        "within": within,
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
        scores = {
            eval_to_step[int(width)]: score
            for width, score in scores.items()
            if int(width) in eval_to_step
        }
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
            for width, score in fold_scores.items():
                step = eval_to_step.get(int(width))
                if step is None:
                    continue
                all_scores[step].append(score)

        diag = build_score_curve_diagnostics(k_grid, all_scores)

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")

    def _features_for_k(step: int) -> List[str]:
        if int(step) <= 0:
            return []
        if step_widths is None:
            return valid_features[: int(step)]
        return valid_features[: int(step_widths[min(int(step), len(step_widths)) - 1])]

    if diag.empty:
        return max_k, _features_for_k(max_k), diag

    curve_config = with_effective_k_bounds(config, min_k=min_k, max_k=max_k)
    best_k, diag = choose_k_from_score_curve(diag, curve_config, lower_is_better=True)
    if is_sklearn_scorer(metric):
        diag["metric"] = sklearn_scorer_label(metric)

    return best_k, _features_for_k(best_k), diag


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
    ic_dimension: Literal["k", "df"] = "k",
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
        ``k``-proportional penalties (BIC, MDL, AIC, HQC, custom). EBIC and
        RIC ignore it unless ``ic_dimension='df'``.
    ic_dimension : {'k', 'df'}, default 'k'
        Likelihood dimension for EBIC/RIC. ``'k'`` is the legacy no-block
        contract (``k log n`` / ``2 k log p``). ``'df'`` uses ``df_path``
        as model dimension while ``k`` remains the search-multiplicity
        argument of ``log C(p, k)`` (EBIC) or the selected-step index
        (diagnostics). Block-aware Gaussian callers pass ``'df'``. The RIC
        block adaptation is ``2 df log(B)`` with ``B`` the eligible
        discovery-block count; it is not a new FDR/calibration claim.

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
    ``d*log(n_eff) + 2*gamma*log C(p, k)`` (EBIC), and ``2*d*log(p)`` (RIC),
    where ``p`` is ``n_candidates`` and ``log C`` is the exact log binomial
    coefficient. Legacy no-block EBIC/RIC keep ``d = k``. Block-aware
    callers set ``ic_dimension='df'`` so ``d`` is usable copula rank and
    ``k`` is the additional-block count in ``log C(B, k)``.
    ``ebic_gamma='auto'`` resolves to the Chen-Chen threshold
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
    if ic_dimension not in {"k", "df"}:
        raise ValueError("ic_dimension must be 'k' or 'df'")
    ic_dim = df if ic_dimension == "df" else None
    penalty, penalty_weight, ebic_gamma, n_candidates_used = _penalty_array(
        config,
        ks,
        n_eff=n_eff,
        n_candidates=n_candidates,
        dimension=ic_dim,
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
