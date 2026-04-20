"""Shared scoring utilities for ranking and model evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss


ScoringFn = Callable[[object, np.ndarray | None, np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class ScoringSpec:
    """Container for a named scoring function and metadata."""

    name: str
    fn: ScoringFn
    higher_is_better: bool = True
    requires_proba: bool = False

    def __call__(self, model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
        return float(self.fn(model, X, y, w))


def _to_2d_probability_matrix(y_proba: np.ndarray) -> np.ndarray:
    if y_proba.ndim != 2:
        raise ValueError("model.predict_proba() must return 2D probability arrays")
    return y_proba


def _as_arrays(y: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_arr = np.asarray(y)
    y_pred_arr = np.asarray(y_pred)
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr.ravel()
    if y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == 1:
        y_pred_arr = y_pred_arr.ravel()
    return y_arr, y_pred_arr, np.asarray(w)


def _neg_mse(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    return -float(np.average((y_true - y_pred) ** 2, weights=sample_weight))


def _neg_rmse(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    return -float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=sample_weight)))


def _neg_mae(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    return -float(np.average(np.abs(y_true - y_pred), weights=sample_weight))


def _r2(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    y_mean = np.average(y_true, weights=sample_weight)
    ss_res = np.average((y_true - y_pred) ** 2, weights=sample_weight)
    ss_tot = np.average((y_true - y_mean) ** 2, weights=sample_weight)
    return float(1 - ss_res / (ss_tot + 1e-10))


def _accuracy(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    return float(np.average(y_true == y_pred, weights=sample_weight))


def _balanced_accuracy(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    return float(balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight))


def _neg_error(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    y_true, y_pred, sample_weight = _as_arrays(y, model.predict(X), w)
    return -float(np.average(y_true != y_pred, weights=sample_weight))


def _neg_logloss(model, X: np.ndarray | None, y: np.ndarray, w: np.ndarray) -> float:
    if not hasattr(model, "predict_proba"):
        raise ValueError("scoring='neg_logloss' requires model.predict_proba")
    y_proba = _to_2d_probability_matrix(np.asarray(model.predict_proba(X)))
    labels: Any = getattr(model, "classes_", None)
    y_true = np.asarray(y)
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    return -float(log_loss(y_true, y_proba, labels=labels, sample_weight=w))


_SCORING_REGISTRY: dict[str, ScoringSpec] = {
    "neg_mse": ScoringSpec("neg_mse", _neg_mse, higher_is_better=True),
    "neg_rmse": ScoringSpec("neg_rmse", _neg_rmse, higher_is_better=True),
    "neg_mae": ScoringSpec("neg_mae", _neg_mae, higher_is_better=True),
    "r2": ScoringSpec("r2", _r2, higher_is_better=True),
    "accuracy": ScoringSpec("accuracy", _accuracy, higher_is_better=True),
    "balanced_accuracy": ScoringSpec("balanced_accuracy", _balanced_accuracy, higher_is_better=True),
    "neg_error": ScoringSpec("neg_error", _neg_error, higher_is_better=True),
    "neg_logloss": ScoringSpec(
        "neg_logloss",
        _neg_logloss,
        higher_is_better=True,
        requires_proba=True,
    ),
}

VALID_SCORERS = tuple(_SCORING_REGISTRY.keys())


def get_scoring(scoring: str) -> ScoringSpec:
    """Return a shared scoring spec for a known scorer name."""
    if scoring not in _SCORING_REGISTRY:
        raise ValueError(f"Unknown scoring: {scoring}. Valid string scorers: {VALID_SCORERS}")
    return _SCORING_REGISTRY[scoring]


__all__ = ["ScoringSpec", "VALID_SCORERS", "get_scoring"]
