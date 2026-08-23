"""Shared CatBoost feature-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numbers
import warnings

import numpy as np
import pandas as pd

from sift._preprocess import infer_higher_is_better


# =============================================================================
# Score direction handling
# =============================================================================

_VALID_TASKS = {"regression", "classification"}
_VALID_ALGORITHMS = {"shap", "permutation", "prediction", "forward", "forward_greedy"}
_VALID_PREFILTER_METHODS = {"catboost", "cefsplus", "mrmr", "none"}
_VALID_IMPORTANCE_TYPES = {
    "PredictionValuesChange",
    "LossFunctionChange",
    "FeatureImportance",
    "ShapValues",
}


def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        allowed_str = ", ".join(sorted(repr(v) for v in allowed))
        raise ValueError(f"{name}={value!r} is invalid; expected one of: {allowed_str}")


def _validate_step_function(step_function: float) -> None:
    if (
        isinstance(step_function, (bool, np.bool_))
        or not isinstance(step_function, numbers.Real)
        or not np.isfinite(float(step_function))
        or not (0 < float(step_function) < 1)
    ):
        raise ValueError("step_function must be a finite float in the open interval (0, 1)")


def _validate_stability_params(n_bootstrap: int, stability_threshold: float) -> None:
    if (
        isinstance(n_bootstrap, (bool, np.bool_))
        or not isinstance(n_bootstrap, numbers.Integral)
        or n_bootstrap <= 0
    ):
        raise ValueError("n_bootstrap must be a positive integer")
    if (
        isinstance(stability_threshold, (bool, np.bool_))
        or not isinstance(stability_threshold, numbers.Real)
        or not np.isfinite(float(stability_threshold))
        or not (0 <= float(stability_threshold) <= 1)
    ):
        raise ValueError("stability_threshold must be a finite float in the closed interval [0, 1]")


def _validate_group_splitter_groups(cv: Any, groups: Optional[np.ndarray]) -> None:
    if cv is None or groups is not None:
        return
    splitter_name = type(cv).__name__.lower()
    if "group" in splitter_name:
        raise ValueError(
            f"{type(cv).__name__} requires group_col or an explicit groups array"
        )

try:
    from catboost import (  # type: ignore[import-not-found]
        CatBoostClassifier,
        CatBoostRegressor,
        EFeaturesSelectionAlgorithm,
        EShapCalcType,
        Pool,
    )
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None
    CatBoostClassifier = None
    Pool = None
    EFeaturesSelectionAlgorithm = None
    EShapCalcType = None


def _resolve_metric_and_direction(
    task: str,
    y: pd.Series,
    eval_metric: Optional[str],
    higher_is_better: Optional[bool],
) -> Tuple[str, bool]:
    """
    Resolve (eval_metric, higher_is_better) with multiclass detection.

    Defaults:
      - regression: RMSE
      - binary classification: Logloss
      - multiclass classification: MultiClass
    """
    if eval_metric is None:
        if task == 'regression':
            eval_metric = 'RMSE'
        else:
            n_classes = int(pd.Series(y).nunique(dropna=True))
            eval_metric = 'MultiClass' if n_classes > 2 else 'Logloss'

    if higher_is_better is None:
        higher_is_better = infer_higher_is_better(eval_metric)

    return eval_metric, higher_is_better


def _resolve_loss_function(
    task: str,
    y: pd.Series,
    loss_function: Optional[str],
) -> str:
    """Resolve loss function with multiclass detection."""
    if loss_function is not None:
        return loss_function
    if task == 'regression':
        return 'RMSE'
    n_classes = int(pd.Series(y).nunique(dropna=True))
    return 'MultiClass' if n_classes > 2 else 'Logloss'


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class CatBoostSelectionResult:
    """
    Result of CatBoost feature selection.

    Attributes
    ----------
    selected_features : list of str
        Final selected feature names.
    best_k : int
        Number of selected features. For automatic selection this is the
        parsimonious count chosen from the score curve and can differ from the
        raw best-scoring count; inspect ``scores_by_k`` for the full curve.
    scores_by_k : dict
        Mean validation score for each k tried.
    scores_std_by_k : dict
        Standard deviation of scores across splits (if n_splits > 1).
    feature_importances : pd.Series
        SHAP or loss-function-change importances from final model.
    features_by_k : dict
        Feature lists for each k (from final run or first split).
    stability_scores : pd.Series, optional
        Selection frequency across resampled splits (if stability selection used).
    prefilter_features : list of str, optional
        Features selected by pre-filter stage (per-split, so from first split).
    metric : str
        Evaluation metric used.
    higher_is_better : bool
        Whether higher metric values are better.
    all_scores : dict, optional
        Raw scores per split per k: {k: [score1, score2, ...]}.
    """
    selected_features: List[str]
    best_k: int
    scores_by_k: Dict[int, float]
    scores_std_by_k: Dict[int, float]
    feature_importances: pd.Series
    features_by_k: Dict[int, List[str]] = field(default_factory=dict)
    stability_scores: Optional[pd.Series] = None
    prefilter_features: Optional[List[str]] = None
    metric: str = "RMSE"
    higher_is_better: bool = False
    all_scores: Optional[Dict[int, List[float]]] = None

    def score_at_k(self, k: int) -> Tuple[float, float]:
        """Return (mean, std) score at given k."""
        return self.scores_by_k.get(k, np.nan), self.scores_std_by_k.get(k, np.nan)

    def features_within_tolerance(self, tolerance: float = 0.01) -> List[str]:
        """
        Get smallest feature set within tolerance of best score.

        Uses stored features_by_k when available (exact), falls back to
        top-k by importance otherwise.
        """
        best_score = (max if self.higher_is_better else min)(self.scores_by_k.values())
        delta = abs(best_score) * float(tolerance)

        if self.higher_is_better:
            threshold = best_score - delta
            valid_ks = [k for k, v in self.scores_by_k.items() if v >= threshold]
        else:
            threshold = best_score + delta
            valid_ks = [k for k, v in self.scores_by_k.items() if v <= threshold]

        if not valid_ks:
            return self.selected_features

        min_k = min(valid_ks)

        # Use stored features if available
        if min_k in self.features_by_k:
            return self.features_by_k[min_k]

        # Fallback: top min_k by importance
        if len(self.feature_importances) >= min_k:
            return (
                self.feature_importances.sort_values(ascending=False, kind="mergesort")
                .head(min_k)
                .index.tolist()
            )

        return self.selected_features

    def plot_scores_vs_k(self, figsize: Tuple[float, float] = (10, 6)):
        """Plot validation scores vs number of features."""
        import matplotlib.pyplot as plt

        ks = sorted(self.scores_by_k.keys())
        means = [self.scores_by_k[k] for k in ks]
        stds = [self.scores_std_by_k.get(k, 0) for k in ks]

        fig, ax = plt.subplots(figsize=figsize)
        ax.errorbar(ks, means, yerr=stds, marker='o', capsize=3)
        ax.axvline(self.best_k, color='red', linestyle='--', alpha=0.7,
                   label=f'Selected k={self.best_k}')
        ax.set_xlabel('Number of Features (k)')
        ax.set_ylabel(f'{self.metric} ({"↑" if self.higher_is_better else "↓"} better)')
        ax.set_title('Feature Selection: Score vs k')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax


# =============================================================================
# Pre-filtering (called inside CV to avoid leakage)
# =============================================================================

def _prefilter_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    task: str,
    method: str = 'cefsplus',
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
    n_jobs: int = -1,
) -> List[str]:
    """
    Fast pre-filtering to reduce feature set before expensive CatBoost RFE.

    NOTE: This should be called on train data only (inside CV) to avoid leakage.

    Parameters
    ----------
    X_train : DataFrame
        Training features only.
    y_train : Series
        Training target only.
    k : int
        Number of features to keep.
    task : str
        'regression' or 'classification'.
    method : str
        Pre-filter method:
        - 'cefsplus': Gaussian-copula MI (fast, good for regression)
        - 'mrmr': Minimum redundancy maximum relevance
        - 'catboost': Shallow CatBoost model importance (handles categoricals)
        - 'none': Keep all features
    cat_features : list of str, optional
        Categorical feature names (for catboost prefilter).
    random_state : int, optional
        Random seed.
    verbose : bool
        Print progress.
    n_jobs : int
        Parallel jobs for catboost prefilter.

    Returns
    -------
    list of str
        Pre-filtered feature names.
    """
    _validate_choice("prefilter_method", method, _VALID_PREFILTER_METHODS)

    all_features = list(X_train.columns)

    if method == 'none' or k >= len(all_features):
        return all_features

    if method == 'catboost':
        # CatBoost-native prefilter: fast shallow model
        return _catboost_importance_prefilter(
            X_train, y_train, k, task,
            cat_features=cat_features,
            text_features=text_features,
            sample_weight=sample_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # For sift methods, only use numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['category', 'object', 'string']).columns.tolist()

    if len(numeric_cols) == 0:
        if verbose:
            print(f"No numeric columns for pre-filtering, keeping all {len(all_features)} features")
        return all_features

    k_numeric = min(k, len(numeric_cols))

    if verbose:
        print(f"  Pre-filter: {len(numeric_cols)} numeric → {k_numeric} using {method}")

    if method == 'cefsplus':
        if task == 'classification':
            from sift.api import select_mrmr

            selected = select_mrmr(
                X_train[numeric_cols],
                y_train,
                k=k_numeric,
                task="classification",
                sample_weight=sample_weight,
                verbose=False,
                subsample=30_000,
                random_state=random_state,
            )
        else:
            from sift.api import select_cefsplus

            selected = select_cefsplus(
                X_train[numeric_cols],
                y_train,
                k=k_numeric,
                sample_weight=sample_weight,
                verbose=False,
                subsample=30_000,
                random_state=random_state,
            )
    elif method == 'mrmr':
        from sift.api import select_mrmr

        selected = select_mrmr(
            X_train[numeric_cols],
            y_train,
            k=k_numeric,
            task=task,
            sample_weight=sample_weight,
            verbose=False,
            subsample=30_000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown prefilter method: {method}")

    # Always keep categorical columns (CatBoost handles them natively)
    final = list(selected) + cat_cols
    return final


def _catboost_importance_prefilter(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int,
    task: str,
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    sample_weight: Optional[pd.Series] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> List[str]:
    """
    Prefilter using a shallow CatBoost model's feature importance.

    Fast and handles categoricals natively. Multiclass-safe.
    """
    if cat_features is None:
        cat_features = []
    if text_features is None:
        text_features = []

    # Multiclass-safe loss function
    loss_fn = _resolve_loss_function(task=task, y=y_train, loss_function=None)

    pool = Pool(
        X_train,
        label=y_train,
        weight=sample_weight,
        cat_features=cat_features or None,
        text_features=text_features or None,
    )

    params = {
        'iterations': 100,
        'depth': 4,  # CatBoost uses 'depth', not 'max_depth' in native API
        'learning_rate': 0.1,
        'verbose': False,
        'allow_writing_files': False,
        'loss_function': loss_fn,
    }
    if random_state is not None:
        params['random_seed'] = int(random_state)
    if n_jobs > 0:
        params['thread_count'] = n_jobs

    if task == 'classification':
        model = CatBoostClassifier(**params)
    else:
        model = CatBoostRegressor(**params)

    try:
        model.fit(pool)
    except Exception as e:
        warnings.warn(f"CatBoost prefilter failed ({e}); keeping all features.")
        return list(X_train.columns)

    # Explicitly specify importance type for deterministic behavior
    importance = model.get_feature_importance(pool, type='PredictionValuesChange')
    feature_names = list(X_train.columns)

    importance_series = pd.Series(importance, index=feature_names)
    ranked = importance_series.sort_values(ascending=False, kind="mergesort")
    top_k = ranked.head(k).index.tolist()
    protected = [
        col
        for col in [*cat_features, *text_features]
        if col in X_train.columns and col not in top_k
    ]

    return top_k + protected


# =============================================================================
# Core CatBoost feature selection
# =============================================================================

def _get_feature_types(
    X: pd.DataFrame,
    features: List[str],
    text_features: Optional[List[str]] = None,
    treat_object_as_categorical: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Extract categorical and text feature names.

    By default, object/string columns are treated as CATEGORICAL (not text).
    Set treat_object_as_categorical=False to exclude them from categorical.
    Use text_features parameter to explicitly mark text columns.
    """
    df = X[features]

    if text_features is None:
        text_features = []
    text_set = set(text_features)

    # Category dtype is always categorical
    cat_candidates = set(df.select_dtypes(include=['category']).columns.tolist())

    # Object/string: categorical by default unless in text_features
    if treat_object_as_categorical:
        cat_candidates |= set(df.select_dtypes(include=['object', 'string']).columns.tolist())

    # Remove text features from categorical
    cat_features = sorted([f for f in cat_candidates if f in features and f not in text_set])
    text_features = [f for f in text_features if f in features]

    return cat_features, text_features


def _create_pool(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    weight: Optional[pd.Series] = None,
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
) -> "Pool":
    """Create CatBoost Pool with proper feature types."""
    if cat_features is None:
        cat_features = []
    if text_features is None:
        text_features = []

    # Filter to features that exist
    cat_features = [f for f in cat_features if f in features]
    text_features = [f for f in text_features if f in features]

    return Pool(
        X[features],
        label=y,
        weight=weight,
        cat_features=cat_features if cat_features else None,
        text_features=text_features if text_features else None,
    )


def _extract_score(model, eval_metric: str) -> float:
    """Robustly extract validation score from fitted model."""
    best_scores = model.get_best_score()

    # Find validation key (could be 'validation', 'validation_0', etc.)
    val_key = None
    for key in best_scores:
        if 'validation' in key.lower() or 'test' in key.lower():
            val_key = key
            break

    if val_key is None:
        # Fallback: use last key that isn't 'learn'
        for key in best_scores:
            if key != 'learn':
                val_key = key
                break

    if val_key is None:
        raise ValueError(f"Could not find validation scores in {best_scores.keys()}")

    # Get score for our metric
    val_scores = best_scores[val_key]

    if eval_metric in val_scores:
        return float(val_scores[eval_metric])

    # Try case-insensitive match
    for metric_name, score in val_scores.items():
        if metric_name.lower() == eval_metric.lower():
            return float(score)

    # Fallback: first metric
    return float(list(val_scores.values())[0])


def _compute_feature_importance(
    model: Union["CatBoostRegressor", "CatBoostClassifier"],
    pool: "Pool",
    method: str = 'shap',
) -> pd.Series:
    """
    Compute feature importances with robust handling of multi-class.

    Methods:
    - 'shap': SHAP values (most interpretable, slow)
    - 'loss': LossFunctionChange (aligns with RecursiveByLossFunctionChange)
    - 'prediction': PredictionValuesChange (fast)
    - 'gain': FeatureImportance based on gain
    """
    feature_names = model.feature_names_

    if method == 'shap':
        importance = model.get_feature_importance(pool, type='ShapValues')
        # ShapValues shape can be:
        # - (n_samples, n_features + 1) for regression/binary
        # - (n_samples, n_classes, n_features + 1) for multi-class
        if importance.ndim == 3:
            # Multi-class: average over classes, then samples, drop base value
            importance = np.abs(importance[:, :, :-1]).mean(axis=(0, 1))
        elif importance.ndim == 2:
            # Binary/regression: average over samples, drop base value
            importance = np.abs(importance[:, :-1]).mean(axis=0)
        else:
            importance = np.abs(importance)
    elif method == 'loss':
        # LossFunctionChange - aligns with RecursiveByLossFunctionChange algorithm
        importance = model.get_feature_importance(pool, type='LossFunctionChange')
    elif method in ('permutation', 'prediction'):
        # PredictionValuesChange - fast approximation
        importance = model.get_feature_importance(pool, type='PredictionValuesChange')
    elif method == 'gain':
        importance = model.get_feature_importance(type='FeatureImportance')
    else:
        raise ValueError(f"Unknown importance method: {method}")

    return pd.Series(importance, index=feature_names).sort_values(
        ascending=False,
        kind="mergesort",
    )
