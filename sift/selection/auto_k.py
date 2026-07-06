"""Automatic k selection for filter methods."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from sift._preprocess import (
    LeaveOneOutLogitEncoder,
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
    """

    k_method: Literal["evaluate", "elbow", "penalized_objective"] = "evaluate"
    strategy: Literal["time_holdout", "group_cv"] = "time_holdout"
    metric: Literal["rmse", "mae", "logloss", "error", "auto"] = "auto"
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
    objective_penalty: Literal["bic", "mdl", "aic", "hqc", "custom"] = "bic"
    objective_penalty_weight: float | None = None
    objective_n_eff: float | None = None
    binary_objective_mode: Literal["refit", "score_test"] = "refit"


_VALID_K_METHODS = frozenset({"evaluate", "elbow", "penalized_objective"})
_VALID_STRATEGIES = frozenset({"time_holdout", "group_cv"})
_VALID_SELECTION_RULES = frozenset({"best", "one_se", "plateau", "tolerance"})
_VALID_PLATEAU_PREFERS = frozenset({"smallest", "center", "best", "largest"})
_VALID_OBJECTIVE_PENALTIES = frozenset({"bic", "mdl", "aic", "hqc", "custom"})
_VALID_BINARY_OBJECTIVE_MODES = frozenset({"refit", "score_test"})
_POSITIVE_INT_FIELDS = (
    "min_k",
    "max_k",
    "n_splits",
    "elbow_patience",
    "plateau_min_points",
)
_REAL_TYPES = (int, float, np.integer, np.floating)


def _is_real_number(value) -> bool:
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, _REAL_TYPES)


def validate_auto_k_config(config: AutoKConfig) -> None:
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

    for name in _POSITIVE_INT_FIELDS:
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"AutoKConfig.{name} must be a positive integer")

    if int(config.min_k) > int(config.max_k):
        raise ValueError("AutoKConfig.min_k must be <= AutoKConfig.max_k")

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

    if (
        config.objective_penalty not in _VALID_OBJECTIVE_PENALTIES
    ):
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
    if config.objective_penalty == "hqc" and (
        config.objective_n_eff is not None and float(config.objective_n_eff) <= np.e
    ):
        raise ValueError("AutoKConfig.objective_n_eff must be > e for HQC")

    if config.binary_objective_mode not in _VALID_BINARY_OBJECTIVE_MODES:
        raise ValueError(
            "AutoKConfig.binary_objective_mode must be one of "
            f"{sorted(_VALID_BINARY_OBJECTIVE_MODES)}; got {config.binary_objective_mode!r}"
        )


def _ensure_supported_auto_k_mode(
    config: AutoKConfig,
    *,
    allow_nested: bool = False,
) -> None:
    """Validate path-selection semantics for the current implementation."""
    validate_auto_k_config(config)
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
        _ensure_supported_auto_k_mode(auto_k_config, allow_nested=allow_nested)
        return auto_k_config
    if time is not None:
        config = AutoKConfig(strategy="time_holdout")
        _ensure_supported_auto_k_mode(config, allow_nested=allow_nested)
        return config
    if groups is not None:
        config = AutoKConfig(strategy="group_cv")
        _ensure_supported_auto_k_mode(config, allow_nested=allow_nested)
        return config
    raise ValueError(
        "k='auto' requires time, groups, or auto_k_config with "
        "k_method='elbow' or k_method='penalized_objective'"
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
    fallback_k = int(diag["k"].max())
    if finite.empty:
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
    metric: str,
    k_grid: list[int],
    cat_features: Optional[List[str]],
    cat_encoding: Literal["none", "target", "loo", "james_stein", "loo_logit"],
    loo_smoothing: float,
    loo_clip_min: float,
    loo_clip_max: float,
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

    if cat_encoding == "loo_logit" and fold_cat:
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
    )


def select_k_auto(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_path: List[str],
    config: AutoKConfig,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    task: Literal["regression", "classification"] = "regression",
    cat_encoding: Literal["none", "target", "loo", "james_stein", "loo_logit"] = "none",
    cat_features: Optional[List[str]] = None,
    sample_weight: Optional[np.ndarray] = None,
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
) -> Tuple[int, List[str], pd.DataFrame]:
    """Select optimal k by evaluating prefixes of feature_path."""
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
        "cat_features": cat_features,
        "cat_encoding": cat_encoding,
        "loo_smoothing": loo_smoothing,
        "loo_clip_min": loo_clip_min,
        "loo_clip_max": loo_clip_max,
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

    return best_k, valid_features[:best_k], diag


def select_k_elbow(
    objective_path: np.ndarray,
    min_k: int = 5,
    max_k: int = 100,
    min_rel_gain: float = 0.02,
    patience: int = 3,
) -> Tuple[int, pd.DataFrame]:
    """Select k via elbow detection on an objective path."""
    obj = np.asarray(objective_path).ravel()
    max_k = min(max_k, len(obj))

    if max_k <= 0:
        return 0, pd.DataFrame()

    delta = np.zeros_like(obj)
    delta[0] = obj[0]
    delta[1:] = obj[1:] - obj[:-1]

    rel_gain = np.zeros_like(obj)
    rel_gain[0] = np.inf
    denom = np.maximum(np.abs(obj[:-1]), 1.0)
    rel_gain[1:] = delta[1:] / denom

    best_k = max_k
    run = 0

    for k in range(max(min_k, 2), max_k + 1):
        if rel_gain[k - 1] < min_rel_gain:
            run += 1
            if run >= patience:
                best_k = k - patience + 1
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


def _penalty_weight(config: AutoKConfig, n_eff: float) -> float:
    if config.objective_penalty in {"bic", "mdl"}:
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
    raise ValueError(f"Unknown objective_penalty: {config.objective_penalty!r}")


def _objective_weight_diagnostics(
    sample_weight: Optional[np.ndarray],
    n_samples: int,
    config: AutoKConfig,
) -> tuple[np.ndarray, float, float, float, str]:
    w = ensure_weights(sample_weight, n_samples, normalize=True)
    weight_sum = float(np.sum(w))
    sum_sq = float(np.sum(w * w))
    kish_n_eff = float(weight_sum * weight_sum / sum_sq) if sum_sq > 0.0 else float("nan")
    if config.objective_n_eff is None:
        n_eff = weight_sum
        n_eff_source = "selector_weight_sum"
    else:
        n_eff = float(config.objective_n_eff)
        n_eff_source = "objective_n_eff"
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
    min_k: Optional[int] = None,
    max_k: Optional[int] = None,
    df_path: Optional[np.ndarray] = None,
) -> Tuple[int, pd.DataFrame]:
    """Select k by maximizing a penalized CEFS+ proxy objective path."""
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
    min_k_eff = max(1, min(int(min_k if min_k is not None else config.min_k), effective_max_k))

    _, weight_sum, kish_n_eff, n_eff, n_eff_source = _objective_weight_diagnostics(
        sample_weight,
        int(n_samples),
        config,
    )
    penalty_weight = _penalty_weight(config, n_eff)
    if objective_scale == "n_eff":
        scale_value = n_eff
        scale_label = "n_eff"
    else:
        scale_value = float(objective_scale)
        scale_label = str(float(objective_scale))
    if not np.isfinite(scale_value):
        raise ValueError("objective_scale must be finite")

    ks = np.arange(1, effective_max_k + 1, dtype=np.int64)
    if df_path is None:
        df = ks.astype(np.float64)
    else:
        df_arr = np.asarray(df_path, dtype=np.float64).reshape(-1)
        if len(df_arr) < effective_max_k:
            raise ValueError("df_path must be at least as long as the effective objective path")
        df = df_arr[:effective_max_k]
    penalty = penalty_weight * df
    objective_used = obj[:effective_max_k]
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

    delta = np.zeros(effective_max_k, dtype=np.float64)
    delta[0] = objective_used[0]
    if effective_max_k > 1:
        delta[1:] = np.diff(objective_used)
    objective_nonmonotone_steps = int(np.sum(delta[1:] < -1e-12))
    path_exhausted_before_max_k = bool(effective_max_k < int(config.max_k))
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
            "selected_at_min_k": selected_at_min_k,
        }
    )
    return best_k, diag


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
    """
    from sift.estimators.copula import (
        weighted_corr_with_vector,
        weighted_correlation_matrix,
        weighted_rank_gauss_1d,
    )
    from sift.selection.objective import objective_from_corr_path

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
