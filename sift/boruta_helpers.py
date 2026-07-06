"""Helper utilities for Boruta selection."""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone

from sift._permute import PermutationAxis, PermutationMethod

ImportanceBackend = Literal["native", "shap"]
Task = Literal["regression", "classification"]
_VALID_TASKS = ("regression", "classification")
_VALID_IMPORTANCE = ("native", "shap")
_VALID_IMPORTANCE_DATA = ("train", "test")
_VALID_SHADOW_METHODS = ("auto", "global", "within_group", "block", "circular_shift")
_VALID_SHADOW_MODES = ("columns", "rows")


def _format_valid_choices(valid_choices) -> str:
    return ", ".join(repr(choice) for choice in valid_choices)


def _validate_enum_option(name: str, value, valid_choices) -> None:
    if value not in valid_choices:
        raise ValueError(
            f"{name} must be one of {_format_valid_choices(valid_choices)}; got {value!r}"
        )


def _validate_block_size(block_size) -> None:
    if block_size == "auto":
        return
    if isinstance(block_size, (int, np.integer)) and not isinstance(block_size, bool):
        if int(block_size) >= 1:
            return
    raise ValueError(
        f"block_size must be a positive integer or 'auto'; got {block_size!r}"
    )


def _validate_boruta_options(
    *,
    task: Task,
    importance: ImportanceBackend,
    importance_data: Literal["train", "test"],
    shadow_method: PermutationMethod,
    shadow_mode: PermutationAxis,
    block_size: int | str,
) -> None:
    _validate_enum_option("task", task, _VALID_TASKS)
    _validate_enum_option("importance", importance, _VALID_IMPORTANCE)
    _validate_enum_option("importance_data", importance_data, _VALID_IMPORTANCE_DATA)
    _validate_enum_option("shadow_method", shadow_method, _VALID_SHADOW_METHODS)
    _validate_enum_option("shadow_mode", shadow_mode, _VALID_SHADOW_MODES)
    _validate_block_size(block_size)


# =============================================================================
# Helper Functions
# =============================================================================


def _clone_estimator(estimator, seed: int):
    """Clone estimator and set random seed exactly once (avoid synonym conflicts)."""
    try:
        est = clone(estimator)
    except Exception:
        est = copy.deepcopy(estimator)

    if hasattr(est, "get_params") and hasattr(est, "set_params"):
        try:
            params = est.get_params(deep=False)
        except TypeError:
            params = est.get_params()
        keys = ("random_seed", "random_state", "seed")
        for key in keys:
            if key in params and params[key] is not None:
                est.set_params(**{key: seed})
                return est
        for key in keys:
            if key in params:
                est.set_params(**{key: seed})
                return est

    if hasattr(est, "set_params"):
        for key in ("random_seed", "random_state", "seed"):
            try:
                est.set_params(**{key: seed})
                return est
            except (ValueError, TypeError):
                pass

    for key in ("random_seed", "random_state", "seed"):
        if hasattr(est, key):
            try:
                setattr(est, key, seed)
                break
            except Exception:
                pass
    return est


def _fit_estimator(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    *,
    require_sample_weight: bool = True,
):
    """
    Fit estimator with sample_weight.

    Raises TypeError if sample_weight is provided but estimator doesn't accept it
    (when require_sample_weight=True).
    """
    kwargs = {}

    if w is not None:
        kwargs["sample_weight"] = w

    try:
        estimator.fit(X, y, **kwargs)
        return
    except TypeError as exc:
        if "sample_weight" in str(exc) and "sample_weight" in kwargs:
            if require_sample_weight:
                raise TypeError(
                    "Estimator.fit() does not accept sample_weight, "
                    "but sample_weight was provided. Use an estimator that "
                    "supports sample_weight or set sample_weight=None."
                ) from exc
            kwargs.pop("sample_weight", None)
            estimator.fit(X, y, **kwargs)
            return
        raise


# Fast-by-default auto tree heuristic
_AUTO_N_EST_MULT = 50.0
_AUTO_N_EST_MIN = 50
_AUTO_N_EST_MAX = 500


def _get_estimator_depth(estimator) -> int:
    """
    Extract max_depth from estimator, handling different libraries.

    Returns depth or 10 as default. Treats None, -1, 0 as unbounded (returns 10).
    """
    for attr in ("max_depth", "depth"):
        val = None
        if hasattr(estimator, "get_params"):
            try:
                params = estimator.get_params()
                if attr in params:
                    val = params[attr]
            except Exception:
                pass

        if val is None and hasattr(estimator, attr):
            val = getattr(estimator, attr)

        if val is None:
            continue

        try:
            d = int(val)
        except (ValueError, TypeError):
            continue

        if d <= 0:
            continue
        return d

    return 10


def _compute_auto_n_estimators(n_features: int, depth: int) -> int:
    """
    Fast heuristic for automatic n_estimators.

    - Scales ~sqrt(#features) and inverse with depth.
    - Doubles features because Boruta concatenates shadow features.
    - Clamped for speed and to avoid invalid (0) trees.
    """
    n_total = max(int(n_features), 1) * 2
    depth_i = max(int(depth), 1)

    n_est = int(_AUTO_N_EST_MULT * np.sqrt(n_total) / depth_i)
    if n_est < _AUTO_N_EST_MIN:
        return _AUTO_N_EST_MIN
    if n_est > _AUTO_N_EST_MAX:
        return _AUTO_N_EST_MAX
    return n_est


def _set_n_estimators(estimator, n_estimators: int) -> None:
    """
    Set n_estimators/iterations on estimator without introducing synonym conflicts.
    """
    # Prefer CatBoost's native params first to avoid synonym collisions.
    param_names = [
        "iterations",
        "num_boost_round",
        "num_trees",
        "n_estimators",
        "num_iterations",
        "n_iter",
    ]

    if hasattr(estimator, "get_params") and hasattr(estimator, "set_params"):
        try:
            params = estimator.get_params(deep=False)
        except TypeError:
            params = estimator.get_params()
        for param in param_names:
            if param in params and params[param] is not None:
                estimator.set_params(**{param: n_estimators})
                return
        for param in param_names:
            if param in params:
                estimator.set_params(**{param: n_estimators})
                return

    if hasattr(estimator, "set_params"):
        for param in param_names:
            try:
                estimator.set_params(**{param: n_estimators})
                return
            except (ValueError, TypeError):
                continue

    for param in param_names:
        if hasattr(estimator, param):
            try:
                setattr(estimator, param, n_estimators)
                return
            except Exception:
                continue


def _get_native_importance(estimator) -> np.ndarray:
    """Get feature importance from fitted estimator."""
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=np.float64)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.max(np.abs(coef), axis=0)

    raise TypeError(
        "Estimator must have feature_importances_ or coef_. "
        "For Boruta, use tree-based models."
    )


def _weighted_mean_abs(
    values: np.ndarray,
    w: np.ndarray,
    *,
    n_features: int | None = None,
    feature_axis: int | None = None,
) -> np.ndarray:
    """Weighted mean of absolute values, handling 2D and 3D SHAP arrays."""
    if values.ndim == 2:
        abs_vals = np.abs(values)
        return (abs_vals * w[:, None]).sum(axis=0) / w.sum()
    if values.ndim == 3:
        if feature_axis is not None:
            if feature_axis not in (1, 2):
                raise ValueError("feature_axis must be 1 or 2 for 3D SHAP arrays")
            if n_features is not None and values.shape[feature_axis] != n_features:
                raise ValueError(
                    f"Unexpected SHAP array shape {values.shape} for {n_features} features"
                )
            output_axis = 2 if feature_axis == 1 else 1
            abs_vals = np.abs(values).mean(axis=output_axis)
        elif n_features is not None:
            if values.shape[1] == n_features:
                abs_vals = np.abs(values).mean(axis=2)
            elif values.shape[2] == n_features:
                abs_vals = np.abs(values).mean(axis=1)
            else:
                raise ValueError(
                    f"Unexpected SHAP array shape {values.shape} for {n_features} features"
                )
        else:
            abs_vals = np.abs(values).mean(axis=1)
        return (abs_vals * w[:, None]).sum(axis=0) / w.sum()
    raise ValueError(f"Unexpected SHAP array shape: {values.shape}")


def _catboost_shap_importance(
    model,
    X: np.ndarray,
    y: np.ndarray | None,
    w: np.ndarray,
) -> np.ndarray:
    """Get SHAP importance using CatBoost's native implementation."""
    from catboost import Pool

    pool = Pool(X, label=y, weight=w)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = np.asarray(shap_vals)

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[:, :-1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, :-1]
    else:
        raise ValueError(f"Unexpected CatBoost SHAP shape: {shap_vals.shape}")

    return _weighted_mean_abs(shap_vals, w, n_features=X.shape[1], feature_axis=2)


def _shap_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray | None,
    w: np.ndarray,
    *,
    shap_sample_size: int | None,
    random_state: int,
) -> np.ndarray:
    """
    Compute SHAP-based feature importance.

    Uses CatBoost native SHAP if available, otherwise falls back to shap package.
    """
    if "catboost" in str(type(estimator)).lower():
        if shap_sample_size is not None and shap_sample_size < X.shape[0]:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X.shape[0], size=shap_sample_size, replace=False)
            return _catboost_shap_importance(
                estimator,
                X[idx],
                y[idx] if y is not None else None,
                w[idx],
            )
        return _catboost_shap_importance(estimator, X, y, w)

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP backend requires either:\n"
            "  - catboost (for native SHAP), OR\n"
            "  - shap package (pip install shap)"
        ) from exc

    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    if shap_sample_size is not None and shap_sample_size < n:
        idx = rng.choice(n, size=shap_sample_size, replace=False)
        X_eval = X[idx]
        w_eval = w[idx]
    else:
        X_eval = X
        w_eval = w

    explainer = shap.TreeExplainer(estimator)
    shap_vals = explainer.shap_values(X_eval)

    if isinstance(shap_vals, list):
        arr = np.stack(shap_vals, axis=1)
        feature_axis = 2
    else:
        arr = np.asarray(shap_vals)
        feature_axis = None

    return _weighted_mean_abs(
        arr,
        w_eval,
        n_features=X_eval.shape[1],
        feature_axis=feature_axis,
    )


def _impute_nonfinite_inplace(X: np.ndarray) -> None:
    """Replace non-finite values (NaN, inf, -inf) with column means."""
    mask = ~np.isfinite(X)
    if not mask.any():
        return
    X[mask] = np.nan
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = col_means[np.where(nan_mask)[1]]


def _group_time_holdout_split(
    groups: np.ndarray,
    time: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split data by taking the last `test_size` fraction of each group's timeline.

    Returns (train_indices, test_indices).
    """
    train_idx = []
    test_idx = []

    groups = np.asarray(groups).reshape(-1)
    time = np.asarray(time).reshape(-1)
    missing = np.asarray(pd.isna(groups), dtype=bool)
    group_values = list(pd.unique(groups[~missing]))
    if missing.any():
        group_values.append(None)

    for g in group_values:
        if g is None:
            idx = np.flatnonzero(missing)
        else:
            idx = np.flatnonzero((groups == g) & ~missing)
        idx = idx[np.argsort(time[idx], kind="mergesort")]
        n = len(idx)
        if n <= 1:
            train_idx.append(idx)
            continue
        n_test = max(1, int(np.ceil(n * test_size)))
        n_test = min(n_test, n - 1)
        train_idx.append(idx[:-n_test])
        test_idx.append(idx[-n_test:])

    train = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    test = np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64)
    return train, test


def _poisson_binom_pmf(ps: np.ndarray) -> np.ndarray:
    """
    Poisson-binomial PMF for sum of independent Bernoullis with probabilities ps.

    Returns pmf[k] = P(S = k) for k=0..len(ps)
    """
    ps = np.asarray(ps, dtype=np.float64).reshape(-1)
    pmf = np.zeros(ps.size + 1, dtype=np.float64)
    pmf[0] = 1.0
    for p in ps:
        prev = pmf.copy()
        pmf = prev * (1.0 - p)
        pmf[1:] += prev[:-1] * p
    return pmf


def _tail_pvals_from_pmf(pmf: np.ndarray, h: int) -> tuple[float, float]:
    """
    Returns (p_hi, p_lo) where:
      p_hi = P(S >= h)
      p_lo = P(S <= h)
    """
    if h < 0:
        return 1.0, 0.0
    if h >= pmf.size:
        return 0.0, 1.0
    cdf = np.cumsum(pmf)
    p_lo = float(cdf[h])
    p_hi = 1.0 if h <= 0 else float(1.0 - cdf[h - 1])
    return p_hi, p_lo


def _time_holdout_indices(
    time: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Global forward holdout when groups not provided."""
    time = np.asarray(time).reshape(-1)
    n = time.shape[0]
    order = np.argsort(time, kind="mergesort")
    n_eval = max(1, min(int(np.ceil(n * test_size)), n - 1)) if n > 1 else 0
    if n_eval > 0:
        return order[:-n_eval].astype(np.int64), order[-n_eval:].astype(np.int64)
    return order.astype(np.int64), np.array([], dtype=np.int64)
