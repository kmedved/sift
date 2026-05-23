"""Shared validation and preparation helpers for binary CEFS+."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from sift._preprocess import encode_categoricals, ensure_weights, validate_k
from sift.selection.api_helpers import validate_groups_time
from sift.selection.cefsplus_binary import (
    fit_logistic_ridge,
    predict_logistic,
    validate_corr_prune,
    weighted_standardize,
)


@dataclass(frozen=True)
class BinaryOptions:
    k_value: int | Literal["auto"]
    loss: str
    top_m: int | None
    corr_prune: float | None
    subsample: int | None
    ridge: float
    refit_every: int
    loo_smoothing: float
    loo_clip_min: float
    loo_clip_max: float


@dataclass(frozen=True)
class BinaryProblem:
    n_rows: int
    n_features_input: int
    groups: np.ndarray | None
    time: np.ndarray | None
    y01: np.ndarray
    raw_y: np.ndarray
    target_mapping: dict
    weights: np.ndarray
    weighted: bool


@dataclass(frozen=True)
class BinaryPathRun:
    path: object
    feature_names: list[str]
    X_sub: np.ndarray
    y_sub: np.ndarray
    w_sub: np.ndarray
    row_idx: np.ndarray
    top_m_eff: int | None
    cat_features: list[str] | None


@dataclass(frozen=True)
class BinarySelection:
    selected_features: list[str]
    selected_original: list[int]
    selected_scores: list[float]
    auto_diag: pd.DataFrame | None = None
    auto_objective: np.ndarray | None = None
    auto_summary: dict | None = None


def resolve_cat_features(
    X: Union[pd.DataFrame, np.ndarray],
    cat_features: Optional[List[str]],
) -> Optional[List[str]]:
    if cat_features is None and isinstance(X, pd.DataFrame):
        return X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return cat_features


def validate_binary_options(
    k,
    *,
    loss: str,
    top_m: Optional[int],
    corr_prune: float | None,
    subsample: Optional[int],
    ridge: float,
    refit_every: int,
    cat_encoding: str,
    loo_smoothing: float,
    loo_clip_min: float,
    loo_clip_max: float,
    sample_weight: np.ndarray | None,
    class_weight,
) -> BinaryOptions:
    k_value = validate_k(k)
    try:
        ridge_float = float(ridge)
    except (TypeError, ValueError) as exc:
        raise ValueError("ridge must be positive and finite") from exc
    if not np.isfinite(ridge_float) or ridge_float <= 0.0:
        raise ValueError("ridge must be positive and finite")
    if (
        isinstance(refit_every, (bool, np.bool_))
        or not isinstance(refit_every, (int, np.integer))
        or int(refit_every) < 1
    ):
        raise ValueError("refit_every must be a positive integer")
    if cat_encoding not in {"none", "target", "loo", "james_stein", "loo_logit"}:
        raise ValueError(
            "cat_encoding must be one of 'none', 'target', 'loo', "
            "'james_stein', or 'loo_logit'."
        )

    try:
        loo_smoothing_float = float(loo_smoothing)
        loo_clip_min_float = float(loo_clip_min)
        loo_clip_max_float = float(loo_clip_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "loo_smoothing and LOO-logit clip bounds must be finite numeric values"
        ) from exc
    if loo_smoothing_float <= 0.0 or not np.isfinite(loo_smoothing_float):
        raise ValueError("loo_smoothing must be positive and finite")
    if (
        not np.isfinite(loo_clip_min_float)
        or not np.isfinite(loo_clip_max_float)
        or not 0.0 < loo_clip_min_float < loo_clip_max_float < 1.0
    ):
        raise ValueError("loo_clip_min and loo_clip_max must satisfy 0 < min < max < 1")

    loss_eff = str(loss).lower()
    if loss_eff not in {"logloss", "brier"}:
        raise ValueError("loss must be one of 'logloss' or 'brier'")

    return BinaryOptions(
        k_value=k_value,
        loss=loss_eff,
        top_m=validate_optional_positive_int(top_m, "top_m"),
        corr_prune=validate_corr_prune(corr_prune),
        subsample=validate_optional_positive_int(subsample, "subsample"),
        ridge=ridge_float,
        refit_every=int(refit_every),
        loo_smoothing=loo_smoothing_float,
        loo_clip_min=loo_clip_min_float,
        loo_clip_max=loo_clip_max_float,
    )


def prepare_binary_problem(
    X,
    y,
    *,
    groups: Optional[np.ndarray],
    time: Optional[np.ndarray],
    sample_weight: np.ndarray | None,
    class_weight,
) -> BinaryProblem:
    x_shape = X.shape if hasattr(X, "shape") else np.asarray(X).shape
    if len(x_shape) != 2:
        raise ValueError("X must be a 2D feature matrix")
    n_rows = int(x_shape[0])
    n_features_input = int(x_shape[1])
    groups, time = validate_groups_time(groups, time, n_rows)
    y01, raw_y, target_mapping = validate_binary_target(y)
    if len(y01) != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {len(y01)}")
    weights, weighted = resolve_binary_weights(
        y01,
        raw_y,
        sample_weight=sample_weight,
        class_weight=class_weight,
    )
    check_binary_effective_weights(y01, weights)
    return BinaryProblem(
        n_rows=n_rows,
        n_features_input=n_features_input,
        groups=groups,
        time=time,
        y01=y01,
        raw_y=raw_y,
        target_mapping=target_mapping,
        weights=weights,
        weighted=weighted,
    )


def validate_binary_target(y) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.asarray(y).ravel()
    if pd.isna(raw).any():
        raise ValueError("Missing values in y are not allowed for binary CEFS+.")
    if raw.size == 0:
        raise ValueError("y must contain at least one row")

    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and not np.isfinite(numeric).all():
        raise ValueError("Non-finite values in y are not allowed for binary CEFS+.")

    unique = pd.unique(raw)
    if len(unique) != 2:
        raise ValueError("binary CEFS+ requires exactly two target classes")

    if numeric is not None and set(np.unique(numeric).tolist()) == {0.0, 1.0}:
        y01 = numeric.astype(np.float64)
        classes = np.array([0.0, 1.0], dtype=np.float64)
    elif set(unique.tolist()) == {False, True}:
        y01 = raw.astype(bool).astype(np.float64)
        classes = np.array([False, True], dtype=object)
    else:
        classes = unique
        mapping = {classes[0]: 0.0, classes[1]: 1.0}
        y01 = np.array([mapping[value] for value in raw], dtype=np.float64)

    return y01, raw, {repr(classes[0]): 0, repr(classes[1]): 1}


def resolve_binary_weights(
    y01: np.ndarray,
    raw_y: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    class_weight,
) -> tuple[np.ndarray, bool]:
    n = len(y01)
    w = ensure_weights(sample_weight, n, normalize=False)
    weighted = sample_weight is not None
    if class_weight is None:
        return ensure_weights(w, n, normalize=True), weighted

    weighted = True
    multipliers = np.ones(n, dtype=np.float64)
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        total = float(np.sum(w))
        for cls in (0.0, 1.0):
            mask = y01 == cls
            cls_total = float(np.sum(w[mask]))
            if cls_total <= 0.0:
                raise ValueError("Each binary class must have positive effective weight")
            multipliers[mask] = total / (2.0 * cls_total)
    elif isinstance(class_weight, dict):
        for code in (0.0, 1.0):
            mask = y01 == code
            raw_key = pd.unique(raw_y[mask])[0]
            multipliers[mask] = class_weight_value(class_weight, raw_key)
    else:
        raise ValueError("class_weight must be None, 'balanced', or a dict")

    return ensure_weights(w * multipliers, n, normalize=True), weighted


def class_weight_value(class_weight: dict, raw_key) -> float:
    if raw_key not in class_weight:
        raise ValueError(
            "class_weight dict must provide weights for both raw binary class labels"
        )
    try:
        value = float(class_weight[raw_key])
    except (TypeError, ValueError) as exc:
        raise ValueError("class_weight values must be finite and non-negative") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("class_weight values must be finite and non-negative")
    return value


def check_binary_effective_weights(y01: np.ndarray, w: np.ndarray) -> None:
    for cls in (0.0, 1.0):
        if float(np.sum(w[y01 == cls])) <= 0.0:
            raise ValueError("Each binary class must have positive effective weight")


def validate_optional_positive_int(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer or None")
    value_int = int(value)
    if value_int < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return value_int


def encode_categoricals_for_binary_selector(
    X: Union[pd.DataFrame, np.ndarray],
    y01: np.ndarray,
    cat_features: Optional[List[str]],
    cat_encoding: str,
    *,
    allow_full_data_target_encoding: bool,
    loo_smoothing: float,
    loo_clip_min: float,
    loo_clip_max: float,
    sample_weight: np.ndarray | None,
) -> Union[pd.DataFrame, np.ndarray]:
    if not cat_features or cat_encoding == "none":
        return X
    if not isinstance(X, pd.DataFrame):
        raise TypeError("cat_features/cat_encoding require X to be a pandas DataFrame.")
    present_cat_features = [col for col in cat_features if col in X.columns]
    if not present_cat_features:
        return X
    if cat_encoding in {"target", "loo", "james_stein", "loo_logit"} and not allow_full_data_target_encoding:
        raise ValueError(
            f"cat_encoding='{cat_encoding}' fits a supervised categorical encoder "
            "on the full dataset in function-style selectors. Pass "
            "allow_full_data_target_encoding=True to opt into this leakage-prone "
            "behavior, or set cat_encoding='none' and pre-encode categoricals in a "
            "leakage-safe pipeline."
        )
    return encode_categoricals(
        X,
        y01,
        present_cat_features,
        cat_encoding,
        loo_smoothing=loo_smoothing,
        loo_clip_min=loo_clip_min,
        loo_clip_max=loo_clip_max,
        sample_weight=sample_weight,
    )


def binary_refit_loglik_gains(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    w_sub: np.ndarray,
    selected_original: list[int],
    *,
    ridge: float,
) -> tuple[np.ndarray, int]:
    """Compute unpenalized weighted log-likelihood gains along a binary path."""
    if not selected_original:
        return np.empty(0, dtype=np.float64), 0
    X_selected = np.asarray(X_sub[:, selected_original], dtype=np.float64)
    Z_selected, valid_mask, _, _ = weighted_standardize(X_selected, w_sub)
    gains = np.full(len(selected_original), -np.inf, dtype=np.float64)
    failures = 0
    if not bool(np.all(valid_mask)):
        failures += int(np.sum(~valid_mask))

    p0 = np.clip(float(np.sum(w_sub * y_sub) / np.sum(w_sub)), 1e-12, 1.0 - 1e-12)
    ll0 = binary_loglik_from_prob(
        y_sub,
        w_sub,
        np.full(len(y_sub), p0, dtype=np.float64),
    )
    max_prefix = min(Z_selected.shape[1], len(selected_original))
    for k in range(1, max_prefix + 1):
        try:
            beta = fit_logistic_ridge(Z_selected[:, :k], y_sub, w_sub, ridge=ridge)
            p = predict_logistic(Z_selected[:, :k], beta)
            gains[k - 1] = binary_loglik_from_prob(y_sub, w_sub, p) - ll0
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            failures += 1
    return gains, failures


def binary_loglik_from_prob(y: np.ndarray, w: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(np.sum(w * (y * np.log(p) + (1.0 - y) * np.log1p(-p))))
