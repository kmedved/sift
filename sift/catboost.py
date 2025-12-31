"""
CatBoost-based feature selection with SHAP/loss-function-change importance.

Wrapper-based feature selection using CatBoost's gradient boosting with:
- Recursive feature elimination via SHAP or loss-function-change importance
- Forward selection via iterative importance (fast)
- One-shot RFE with ranking reconstruction (efficient)
- Multi-split evaluation for robust scoring
- Group-resampled stability selection (group-aware)
- Two-stage pipeline with fast pre-filtering (inside CV to avoid leakage)
- Custom splitter support (time series, grouped, etc.)

Usage:
    from sift import catboost_regression, catboost_classif

    # Simple API (like mrmr_regression)
    selected = catboost_regression(X, y, K=20)

    # With custom time series splitter
    from sklearn.model_selection import TimeSeriesSplit
    result = catboost_select(
        X, y, K=20,
        cv=TimeSeriesSplit(n_splits=5),
    )

    # Forward selection (faster for small K)
    selected = catboost_regression(X, y, K=10, algorithm='forward')
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Literal, Tuple, Any, Iterator
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd

from sift._preprocess import best_score_from_dict, infer_higher_is_better

from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit


# =============================================================================
# Score direction handling
# =============================================================================

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


# Keep old name for backward compatibility
def _resolve_higher_is_better(
    metric: Optional[str],
    higher_is_better: Optional[bool],
    task: str,
) -> Tuple[str, bool]:
    """Resolve metric name and direction. DEPRECATED: use _resolve_metric_and_direction."""
    if metric is None:
        metric = 'RMSE' if task == 'regression' else 'Logloss'
    if higher_is_better is None:
        higher_is_better = infer_higher_is_better(metric)
    return metric, higher_is_better


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
        Number of features in best configuration.
    scores_by_k : dict
        Mean validation score for each K tried.
    scores_std_by_k : dict
        Standard deviation of scores across splits (if n_splits > 1).
    feature_importances : pd.Series
        SHAP or loss-function-change importances from final model.
    features_by_k : dict
        Feature lists for each K (from final run or first split).
    stability_scores : pd.Series, optional
        Selection frequency across resampled splits (if stability selection used).
    prefilter_features : list of str, optional
        Features selected by pre-filter stage (per-split, so from first split).
    metric : str
        Evaluation metric used.
    higher_is_better : bool
        Whether higher metric values are better.
    all_scores : dict, optional
        Raw scores per split per K: {k: [score1, score2, ...]}.
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
        """Return (mean, std) score at given K."""
        return self.scores_by_k.get(k, np.nan), self.scores_std_by_k.get(k, np.nan)

    def features_within_tolerance(self, tolerance: float = 0.01) -> List[str]:
        """
        Get smallest feature set within tolerance of best score.

        Uses stored features_by_k when available (exact), falls back to
        top-K by importance otherwise.
        """
        best_score = (max if self.higher_is_better else min)(self.scores_by_k.values())

        if self.higher_is_better:
            threshold = best_score * (1 - tolerance)
            valid_ks = [k for k, v in self.scores_by_k.items() if v >= threshold]
        else:
            threshold = best_score * (1 + tolerance)
            valid_ks = [k for k, v in self.scores_by_k.items() if v <= threshold]

        if not valid_ks:
            return self.selected_features

        min_k = min(valid_ks)

        # Use stored features if available
        if min_k in self.features_by_k:
            return self.features_by_k[min_k]

        # Fallback: top min_k by importance
        if len(self.feature_importances) >= min_k:
            return self.feature_importances.nlargest(min_k).index.tolist()

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
                   label=f'Best K={self.best_k}')
        ax.set_xlabel('Number of Features (K)')
        ax.set_ylabel(f'{self.metric} ({"↑" if self.higher_is_better else "↓"} better)')
        ax.set_title('Feature Selection: Score vs K')
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
    all_features = list(X_train.columns)

    if method == 'none' or k >= len(all_features):
        return all_features

    if method == 'catboost':
        # CatBoost-native prefilter: fast shallow model
        return _catboost_importance_prefilter(
            X_train, y_train, k, task,
            cat_features=cat_features,
            text_features=text_features,
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
        cat_features=cat_features or None,
        text_features=text_features or None,
    )

    params = {
        'iterations': 100,
        'depth': 4,  # CatBoost uses 'depth', not 'max_depth' in native API
        'learning_rate': 0.1,
        'verbose': False,
        'random_seed': random_state,
        'allow_writing_files': False,
        'loss_function': loss_fn,
    }
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
    top_k = importance_series.nlargest(k).index.tolist()

    return top_k


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

    return pd.Series(importance, index=feature_names).sort_values(ascending=False)


def _select_features_single_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: List[str],
    feature_counts: List[int],
    task: str,
    model_params: Dict[str, Any],
    cat_features: List[str],
    text_features: List[str],
    eval_metric: str,
    higher_is_better: bool,
    w_train: Optional[pd.Series] = None,
    w_val: Optional[pd.Series] = None,
    algorithm: str = 'shap',
    steps: int = 6,
    train_early_stopping_rounds: int = 20,
) -> Tuple[Dict[int, float], Dict[int, List[str]]]:
    """
    Run ONE-SHOT feature selection for a single train/val split.

    Key optimization: Run select_features once at min_k with train_final_model=True,
    then reconstruct feature sets for larger K using elimination order.

    IMPORTANT: Survivors are reordered by importance from the trained model,
    since CatBoost doesn't guarantee ordering in selected_features_names.

    Returns (scores_by_k, features_by_k).
    """
    train_pool = _create_pool(X_train, y_train, features, w_train, cat_features, text_features)
    val_pool = _create_pool(X_val, y_val, features, w_val, cat_features, text_features)

    # Create model
    ModelClass = CatBoostClassifier if task == 'classification' else CatBoostRegressor
    model = ModelClass(**model_params)

    # Map algorithm string to enum
    algo_map = {
        'shap': EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        'permutation': EFeaturesSelectionAlgorithm.RecursiveByLossFunctionChange,
        'prediction': EFeaturesSelectionAlgorithm.RecursiveByPredictionValuesChange,
    }
    algo_enum = algo_map.get(algorithm, EFeaturesSelectionAlgorithm.RecursiveByShapValues)

    min_k = min(feature_counts)
    n_features = len(features)

    # ONE-SHOT: Run RFE only once down to min_k
    # Use train_final_model=True to avoid separate retrain for min_k
    try:
        # Build select_features kwargs - shap_calc_type only for SHAP/loss-change modes
        select_kwargs = dict(
            eval_set=val_pool,
            features_for_select=features,
            num_features_to_select=min(min_k, n_features),
            steps=steps,
            train_final_model=True,
            logging_level='Silent',
            algorithm=algo_enum,
            plot=False,
        )
        # shap_calc_type is used for SHAP- and loss-change selection, not prediction-change
        if algorithm in ('shap', 'permutation'):
            select_kwargs['shap_calc_type'] = EShapCalcType.Regular

        summary = model.select_features(train_pool, **select_kwargs)
    except Exception as e:
        warnings.warn(f"select_features failed: {e}. Falling back to importance ranking.")
        # Fallback: train on all features, rank by importance
        model.fit(train_pool, eval_set=val_pool,
                  early_stopping_rounds=train_early_stopping_rounds, verbose=False)
        imp = model.get_feature_importance()
        order = np.argsort(-imp)
        ranked_features = [features[i] for i in order]
        summary = None
    else:
        # Reconstruct full ranking from elimination order
        survivors = list(summary['selected_features_names'])
        eliminated = list(summary['eliminated_features_names'])

        # IMPORTANT: CatBoost doesn't guarantee ordering of survivors!
        # Reorder survivors by importance from the trained model
        if len(survivors) > 1:
            try:
                imp = model.get_feature_importance(train_pool, type='PredictionValuesChange')
                imp_series = pd.Series(imp, index=features)
                survivors = imp_series.loc[survivors].sort_values(ascending=False).index.tolist()
            except Exception:
                pass  # Keep original order if importance fails

        # Full ranking: survivors (sorted by importance) + eliminated in reverse (best losers first)
        ranked_features = list(survivors) + list(reversed(eliminated))

    scores = {}
    features_selected = {}

    # Evaluate each K
    for k in feature_counts:
        if k >= n_features:
            current_feats = features
        else:
            current_feats = ranked_features[:k]

        features_selected[k] = current_feats

        # For min_k, we can extract score from the trained model (if select_features succeeded)
        if k == min_k and summary is not None:
            try:
                scores[k] = _extract_score(model, eval_metric)
                continue
            except Exception:
                pass  # Fall through to retrain

        # Retrain for exact score at this K
        sel_cat = [f for f in cat_features if f in current_feats]
        sel_text = [f for f in text_features if f in current_feats]

        train_pool_k = _create_pool(X_train, y_train, current_feats, w_train, sel_cat, sel_text)
        val_pool_k = _create_pool(X_val, y_val, current_feats, w_val, sel_cat, sel_text)

        eval_model = ModelClass(**model_params)
        try:
            eval_model.fit(train_pool_k, eval_set=val_pool_k,
                          early_stopping_rounds=train_early_stopping_rounds, verbose=False)
            scores[k] = _extract_score(eval_model, eval_metric)
        except Exception as e:
            warnings.warn(f"Training failed for k={k}: {e}")
            continue

    return scores, features_selected


def _generate_feature_counts(
    n_features: int,
    min_features: int,
    step_function: float,
    max_counts: int = 20,
) -> List[int]:
    """Generate geometric sequence of feature counts to try."""
    counts = [n_features]  # Always include baseline
    k = n_features

    while k > min_features / step_function and len(counts) < max_counts:
        k = int(k * step_function)
        if k >= min_features:
            counts.append(k)

    # Always include min_features
    if min_features not in counts and min_features < n_features:
        counts.append(min_features)

    return sorted(set(counts), reverse=True)


def _aggregate_feature_lists(
    feature_lists: List[List[str]],
    k: Optional[int] = None,
) -> Tuple[List[str], pd.Series]:
    """
    Aggregate multiple selected-feature lists into a single ordered list.

    Sort by:
        1) frequency (descending) - features selected more often rank higher
        2) mean rank position (ascending) - among ties, prefer earlier positions
        3) feature name (stable tie-breaker)

    Returns (ordered_features, stability_scores).
    """
    if not feature_lists:
        return [], pd.Series(dtype=float)

    freq: Dict[str, int] = defaultdict(int)
    rank_sum: Dict[str, float] = defaultdict(float)
    rank_cnt: Dict[str, int] = defaultdict(int)

    for fl in feature_lists:
        for pos, f in enumerate(fl):
            freq[f] += 1
            rank_sum[f] += float(pos)
            rank_cnt[f] += 1

    n_runs = len(feature_lists)
    stability = pd.Series({f: c / n_runs for f, c in freq.items()}).sort_values(ascending=False)

    def sort_key(f: str) -> Tuple[int, float, str]:
        c = freq[f]
        mean_rank = rank_sum[f] / max(rank_cnt[f], 1)
        return (-c, mean_rank, f)  # -c for descending frequency

    ordered = sorted(freq.keys(), key=sort_key)
    if k is not None and k < len(ordered):
        ordered = ordered[:k]

    return ordered, stability


# =============================================================================
# Bootstrap sampler for stability selection (group-aware, O(n))
# =============================================================================

def _bootstrap_indices(
    n: int,
    n_bootstrap: int,
    groups: Optional[np.ndarray] = None,
    y: Optional[pd.Series] = None,
    task: str = 'regression',
    random_state: Optional[int] = None,
    min_oob: int = 10,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate bootstrap-style train/val splits (train=in-bag, val=out-of-bag).

    If groups is provided, samples GROUPS with replacement (group-resampled stability selection).
    Uses O(n) algorithm for group index building.
    Dedupes train groups to avoid row duplication (use sample_weight for weighting).

    For classification, ensures both train and val have all classes (skips bad splits).

    Parameters
    ----------
    n : int
        Number of samples.
    n_bootstrap : int
        Number of bootstrap iterations.
    groups : array-like, optional
        Group labels for group-resampled stability selection.
    y : Series, optional
        Target for classification class checking.
    task : str
        'regression' or 'classification' (for class checking).
    random_state : int, optional
        Random seed.
    min_oob : int
        Minimum OOB samples required.
    """
    rng = np.random.default_rng(random_state)

    # For classification, get unique classes for checking
    classes = None
    if task == 'classification' and y is not None:
        classes = set(pd.Series(y).dropna().unique())

    if groups is not None:
        groups = np.asarray(groups)
        # O(n) group index building using return_inverse
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        # Build group_to_idx in O(n) using inverse indices
        group_idx_lists: List[List[int]] = [[] for _ in range(n_groups)]
        for i, gi in enumerate(inverse):
            group_idx_lists[gi].append(i)
        group_idx_arrays = [np.array(lst, dtype=np.int64) for lst in group_idx_lists]

        valid_count = 0
        attempts = 0
        max_attempts = n_bootstrap * 10

        while valid_count < n_bootstrap and attempts < max_attempts:
            attempts += 1

            # Sample groups with replacement, then DEDUPE for train indices
            sampled_gi = rng.integers(0, n_groups, size=n_groups)

            # Dedupe: each group appears once in train (no row duplication)
            in_bag_gi = np.unique(sampled_gi)
            oob_gi = np.setdiff1d(np.arange(n_groups), in_bag_gi)

            if len(oob_gi) < 1:
                # All groups in bag - use group holdout fallback
                perm = rng.permutation(n_groups)
                split = max(1, int(round(0.25 * n_groups)))
                oob_gi = perm[:split]
                in_bag_gi = perm[split:]

            train_idx = np.concatenate([group_idx_arrays[gi] for gi in in_bag_gi])
            val_idx = np.concatenate([group_idx_arrays[gi] for gi in oob_gi])

            if len(val_idx) < min_oob:
                continue

            # Classification: check class presence
            # Strict for train (all classes), relaxed for val (at least 2)
            if classes is not None and y is not None:
                train_classes = set(y.iloc[train_idx].dropna().unique())
                val_classes = set(y.iloc[val_idx].dropna().unique())
                # Train must have all classes (required for model fitting)
                if train_classes != classes:
                    continue
                # Val: require at least 2 classes (for meaningful eval)
                # For binary, this means both classes; for multiclass, relaxed
                if len(val_classes) < 2:
                    continue

            valid_count += 1
            yield train_idx.astype(np.int64), val_idx.astype(np.int64)

        # If we couldn't get enough valid splits, warn
        if valid_count < n_bootstrap:
            warnings.warn(
                f"Only generated {valid_count}/{n_bootstrap} valid bootstrap splits. "
                "Consider more data or fewer classes."
            )
    else:
        # Standard row-level bootstrap
        valid_count = 0
        attempts = 0
        max_attempts = n_bootstrap * 10

        while valid_count < n_bootstrap and attempts < max_attempts:
            attempts += 1

            train_idx = rng.integers(0, n, size=n)

            # O(n) OOB computation using boolean mask
            in_bag = np.zeros(n, dtype=bool)
            in_bag[train_idx] = True
            val_idx = np.flatnonzero(~in_bag)

            if len(val_idx) < min_oob:
                # OOB too small, use random holdout
                perm = rng.permutation(n)
                split = int(n * 0.75)
                train_idx = perm[:split]
                val_idx = perm[split:]

            # Classification: check class presence
            # Strict for train (all classes), relaxed for val (at least 2)
            if classes is not None and y is not None:
                train_classes = set(y.iloc[train_idx].dropna().unique())
                val_classes = set(y.iloc[val_idx].dropna().unique())
                if train_classes != classes:
                    continue
                if len(val_classes) < 2:
                    continue

            valid_count += 1
            yield train_idx.astype(np.int64), val_idx.astype(np.int64)

        if valid_count < n_bootstrap:
            warnings.warn(
                f"Only generated {valid_count}/{n_bootstrap} valid bootstrap splits."
            )


# =============================================================================
# Forward selection (iterative importance-based)
# =============================================================================

def _forward_select_single_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: List[str],
    feature_counts: List[int],
    task: str,
    model_params: Dict[str, Any],
    cat_features: List[str],
    text_features: List[str],
    eval_metric: str,
    higher_is_better: bool,
    w_train: Optional[pd.Series] = None,
    w_val: Optional[pd.Series] = None,
    importance_type: str = 'PredictionValuesChange',
    early_stopping_rounds: int = 20,
) -> Tuple[Dict[int, float], List[str]]:
    """
    Forward selection by importance ranking (fast heuristic).

    Algorithm:
    1. Train model on all features, get importance ranking
    2. Evaluate ONLY at requested K values (not every k from 1..max)

    This is O(len(feature_counts)) model fits, not O(max_k).
    Returns ranked features for subset reconstruction.

    NOTE: This is "prefix" selection, not true iterative forward selection.
    The ranking is computed once from the full model, not recomputed at each step.
    """
    ModelClass = CatBoostClassifier if task == 'classification' else CatBoostRegressor

    # Step 1: Get importance ranking from full model
    full_cat = [f for f in cat_features if f in features]
    full_text = [f for f in text_features if f in features]

    train_pool = _create_pool(X_train, y_train, features, w_train, full_cat, full_text)
    val_pool = _create_pool(X_val, y_val, features, w_val, full_cat, full_text)

    model = ModelClass(**model_params)
    model.fit(train_pool, eval_set=val_pool,
              early_stopping_rounds=early_stopping_rounds, verbose=False)

    # Get importance (PredictionValuesChange is fast and reliable)
    importance = model.get_feature_importance(train_pool, type=importance_type)
    importance_series = pd.Series(importance, index=features)

    # Rank features by importance (descending)
    ranked_features = importance_series.sort_values(ascending=False).index.tolist()

    # Step 2: Evaluate ONLY at requested K values
    scores = {}
    max_k = max(feature_counts) if feature_counts else len(features)
    max_k = min(max_k, len(features))

    for k in feature_counts:
        if k > len(features):
            k = len(features)
        current_feats = ranked_features[:k]

        sel_cat = [f for f in cat_features if f in current_feats]
        sel_text = [f for f in text_features if f in current_feats]

        train_pool_k = _create_pool(X_train, y_train, current_feats, w_train, sel_cat, sel_text)
        val_pool_k = _create_pool(X_val, y_val, current_feats, w_val, sel_cat, sel_text)

        eval_model = ModelClass(**model_params)
        try:
            eval_model.fit(train_pool_k, eval_set=val_pool_k,
                          early_stopping_rounds=early_stopping_rounds, verbose=False)
            scores[k] = _extract_score(eval_model, eval_metric)
        except Exception as e:
            warnings.warn(f"Forward selection scoring failed at k={k}: {e}")
            continue

    return scores, ranked_features[:max_k]


def _forward_select_greedy_single_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    features: List[str],
    max_k: int,
    task: str,
    model_params: Dict[str, Any],
    cat_features: List[str],
    text_features: List[str],
    eval_metric: str,
    higher_is_better: bool,
    w_train: Optional[pd.Series] = None,
    w_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 20,
) -> Tuple[Dict[int, float], List[str]]:
    """
    True greedy forward selection: at each step, try all remaining features
    and pick the one that improves score the most.

    This is O(K * n_remaining) model fits - expensive but principled.
    Use for small K or final refinement.
    """
    ModelClass = CatBoostClassifier if task == 'classification' else CatBoostRegressor

    selected = []
    remaining = list(features)
    scores = {}

    for step in range(min(max_k, len(features))):
        best_candidate = None
        best_score = float('-inf') if higher_is_better else float('inf')

        for candidate in remaining:
            current_feats = selected + [candidate]

            sel_cat = [f for f in cat_features if f in current_feats]
            sel_text = [f for f in text_features if f in current_feats]

            train_pool = _create_pool(X_train, y_train, current_feats, w_train, sel_cat, sel_text)
            val_pool = _create_pool(X_val, y_val, current_feats, w_val, sel_cat, sel_text)

            model = ModelClass(**model_params)
            try:
                model.fit(train_pool, eval_set=val_pool,
                         early_stopping_rounds=early_stopping_rounds, verbose=False)
                score = _extract_score(model, eval_metric)

                is_better = (score > best_score) if higher_is_better else (score < best_score)
                if is_better:
                    best_score = score
                    best_candidate = candidate
            except Exception:
                continue

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        scores[len(selected)] = best_score

    return scores, selected


# =============================================================================
# Main API
# =============================================================================

def catboost_select(
    X: pd.DataFrame,
    y: pd.Series,
    K: Optional[int] = None,
    task: Literal['regression', 'classification'] = 'regression',
    # Search parameters
    min_features: int = 5,
    step_function: float = 0.67,
    feature_counts: Optional[List[int]] = None,
    selection_patience: int = 3,
    tolerance: float = 0.01,
    # Evaluation parameters - CUSTOM SPLITTER SUPPORT
    cv: Optional[Any] = None,  # Any sklearn-compatible splitter (TimeSeriesSplit, GroupKFold, etc.)
    n_splits: int = 3,
    test_size: float = 0.25,
    group_col: Optional[str] = None,
    sample_weight_col: Optional[str] = None,
    # Pre-filtering (applied inside CV to avoid leakage)
    prefilter_k: Optional[int] = 200,
    prefilter_method: str = 'catboost',
    # Stability selection (group-resampled, group-aware)
    use_stability: bool = False,
    n_bootstrap: int = 20,
    stability_threshold: float = 0.6,
    # CatBoost parameters
    n_estimators: int = 500,
    learning_rate: Optional[float] = None,
    max_depth: Optional[int] = 6,
    eval_metric: Optional[str] = None,
    loss_function: Optional[str] = None,
    catboost_params: Optional[Dict[str, Any]] = None,
    algorithm: Literal['shap', 'permutation', 'prediction', 'forward', 'forward_greedy'] = 'shap',
    steps: int = 6,
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    treat_object_as_categorical: bool = True,
    train_early_stopping_rounds: int = 20,
    gpu: bool = False,
    n_jobs: int = -1,
    # Meta parameters
    higher_is_better: Optional[bool] = None,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> CatBoostSelectionResult:
    """
    CatBoost-based feature selection with SHAP/loss-change/forward importance.

    Combines recursive feature elimination with multi-split evaluation,
    optional pre-filtering, and group-resampled stability selection.

    Supports custom CV splitters for time series and grouped data.

    Parameters
    ----------
    X : DataFrame
        Feature matrix. Can include numeric, categorical columns.
        Object/string columns are treated as categorical by default.
    y : Series
        Target variable.
    K : int, optional
        Exact number of features to select. If None, searches for optimal K.
    task : str, default='regression'
        Either 'regression' or 'classification'.

    Search Parameters
    -----------------
    min_features : int, default=5
        Minimum number of features to consider.
    step_function : float, default=0.67
        Geometric step for feature count search (K → K*step → K*step² ...).
    feature_counts : list of int, optional
        Explicit list of feature counts to try. Overrides step_function.
    selection_patience : int, default=3
        Stop K selection search if score doesn't improve for this many consecutive
        steps. Note: this only affects which K is chosen, not compute time (all Ks
        are still evaluated). For compute savings, use prefilter_k or reduce n_splits.
    tolerance : float, default=0.01
        Score tolerance for considering improvement.

    Evaluation Parameters
    ---------------------
    cv : sklearn splitter, optional
        Custom cross-validation splitter. If provided, overrides n_splits/test_size.
        Examples: TimeSeriesSplit(n_splits=5), GroupKFold(n_splits=5),
        BlockTimeSeriesSplit, custom splitters.
        The splitter's .split(X, y, groups) method will be called.
    n_splits : int, default=3
        Number of train/validation splits (ignored if cv is provided).
    test_size : float, default=0.25
        Fraction of data for validation (ignored if cv is provided).
    group_col : str, optional
        Column name for group-aware splitting (prevents data leakage).
        Used with GroupShuffleSplit or passed to custom cv.split().
    sample_weight_col : str, optional
        Column name for sample weights.

    Pre-filtering Parameters
    ------------------------
    prefilter_k : int, optional, default=200
        Number of features to keep after pre-filtering. Set to None to disable.
        Applied INSIDE each CV fold to avoid data leakage.
    prefilter_method : str, default='catboost'
        Pre-filter method:
        - 'catboost': Shallow CatBoost importance (fast, handles categoricals)
        - 'cefsplus': Gaussian-copula MI (for regression, numeric only)
        - 'mrmr': mRMR (numeric only)
        - 'none': Disable pre-filtering

    Stability Selection Parameters
    ------------------------------
    use_stability : bool, default=False
        If True, run group-resampled stability selection: refit on n_bootstrap
        resampled splits and keep features selected in >= stability_threshold.
        If group_col is provided, bootstrap is group-aware (samples groups).
    n_bootstrap : int, default=20
        Number of bootstrap samples for stability selection.
    stability_threshold : float, default=0.6
        Minimum selection frequency to keep a feature.

    CatBoost Parameters
    -------------------
    n_estimators : int, default=500
        Number of boosting iterations.
    learning_rate : float, optional
        Learning rate. If None, CatBoost auto-selects.
    max_depth : int, default=6
        Maximum tree depth.
    eval_metric : str, optional
        Evaluation metric (e.g., 'RMSE', 'AUC'). Auto-selected if None.
    loss_function : str, optional
        Training objective (e.g., 'RMSE', 'Logloss'). Auto-selected if None.
    catboost_params : dict, optional
        Additional CatBoost parameters (merged with above).
    algorithm : str, default='shap'
        Feature selection algorithm:
        - 'shap': RFE with SHAP importance (most accurate, slowest)
        - 'permutation': RFE with loss-function-change importance (good balance)
        - 'prediction': RFE with prediction change (fastest RFE)
        - 'forward': Forward selection by importance ranking (fast, O(K) fits)
        - 'forward_greedy': True greedy forward selection (O(K*n_features) fits)
    steps : int, default=6
        Number of elimination steps in CatBoost's select_features (RFE modes).
    cat_features : list of str, optional
        Explicit list of categorical feature names. Use this for integer-encoded
        categoricals that wouldn't be auto-detected. Merged with auto-detected
        categorical features from dtype.
    text_features : list of str, optional
        Columns to treat as text (default: object/string are categorical).
    treat_object_as_categorical : bool, default=True
        If True, object/string columns are treated as categorical.
        If False and object columns exist (not in text_features), a warning is raised.
    train_early_stopping_rounds : int, default=20
        Early stopping rounds for model training within selection.
    gpu : bool, default=False
        Use GPU acceleration.
    n_jobs : int, default=-1
        Number of CPU threads (-1 = all).

    Meta Parameters
    ---------------
    higher_is_better : bool, optional
        If True, higher scores are better. Auto-detected from eval_metric if not set.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    CatBoostSelectionResult
        Result object with selected features, scores, and importances.

    Examples
    --------
    # Time series data with TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    result = catboost_select(
        X, y, K=20,
        cv=TimeSeriesSplit(n_splits=5),
    )

    # Grouped data (e.g., NBA players over seasons)
    result = catboost_select(
        X, y, K=20,
        group_col='player_id',
        cv=GroupKFold(n_splits=5),
    )

    # Fast forward selection
    result = catboost_select(
        X, y, K=20,
        algorithm='forward',
    )
    """
    if CatBoostRegressor is None:
        raise ImportError(
            "CatBoost is required for this function. "
            "Install with: pip install catboost"
        )

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    n_samples, n_features_orig = X.shape

    # Resolve metric and direction with MULTICLASS DETECTION
    resolved_metric, resolved_hib = _resolve_metric_and_direction(
        task=task, y=y, eval_metric=eval_metric, higher_is_better=higher_is_better
    )
    resolved_loss = _resolve_loss_function(task=task, y=y, loss_function=loss_function)

    if verbose:
        direction = "↑" if resolved_hib else "↓"
        print(f"CatBoost feature selection: {n_samples:,} samples × {n_features_orig} features")
        print(f"  Metric: {resolved_metric} ({direction} better)")

    # Extract weight and group columns
    X_work = X.copy()
    sample_weights = None
    groups = None

    if sample_weight_col is not None:
        if sample_weight_col in X_work.columns:
            sample_weights = X_work[sample_weight_col]
            X_work = X_work.drop(columns=[sample_weight_col])

    if group_col is not None:
        if group_col in X_work.columns:
            groups = X_work[group_col]
            X_work = X_work.drop(columns=[group_col])

    all_features = list(X_work.columns)

    # Detect categorical features from dtype
    detected_cat, text_feat = _get_feature_types(
        X_work, all_features, text_features, treat_object_as_categorical
    )

    # Merge explicit cat_features with auto-detected (for integer-encoded categoricals)
    if cat_features is not None:
        # Validate explicit cat_features exist in X
        missing = [f for f in cat_features if f not in all_features]
        if missing:
            warnings.warn(f"cat_features not found in X (ignoring): {missing[:5]}")
        explicit_cat = [f for f in cat_features if f in all_features]
        # Merge: explicit + auto-detected, remove duplicates, keep order
        combined = list(explicit_cat)
        for f in detected_cat:
            if f not in combined:
                combined.append(f)
        cat_features_final = combined
    else:
        cat_features_final = detected_cat

    # Warn if treat_object_as_categorical=False and there are orphan object columns
    if not treat_object_as_categorical:
        obj_cols = X_work.select_dtypes(include=['object', 'string']).columns.tolist()
        text_set = set(text_features or [])
        cat_set = set(cat_features_final)
        orphan_obj = [c for c in obj_cols if c not in text_set and c not in cat_set]
        if orphan_obj:
            warnings.warn(
                f"treat_object_as_categorical=False but {len(orphan_obj)} object column(s) "
                f"are not in text_features or cat_features: {orphan_obj[:5]}. "
                "Auto-treating them as categorical to avoid CatBoost errors. "
                "To exclude them, drop from X before calling."
            )
            cat_features_final = list(cat_features_final)
            for c in orphan_obj:
                if c not in cat_features_final:
                    cat_features_final.append(c)

    if verbose and cat_features_final:
        print(f"  Categorical features: {len(cat_features_final)}")

    # Build model parameters - ALWAYS set eval_metric and loss_function
    model_params = {
        'iterations': n_estimators,
        'verbose': False,
        'random_seed': random_state,
        'od_type': 'Iter',
        'od_wait': 30,
        'allow_writing_files': False,
        'eval_metric': resolved_metric,
        'loss_function': resolved_loss,
    }

    if max_depth is not None:
        model_params['depth'] = max_depth
    if learning_rate is not None:
        model_params['learning_rate'] = learning_rate

    if gpu:
        model_params['task_type'] = 'GPU'
        model_params['devices'] = '0'
    elif n_jobs > 0:
        model_params['thread_count'] = n_jobs

    # Merge user catboost_params and RE-READ eval_metric if overridden
    if catboost_params:
        model_params.update(catboost_params)
        # User may have overridden eval_metric - update our resolved values
        if 'eval_metric' in catboost_params:
            resolved_metric = str(catboost_params['eval_metric'])
            if higher_is_better is None:
                resolved_hib = infer_higher_is_better(resolved_metric)

    # Generate feature counts to try
    if K is not None:
        counts = [K]
    elif feature_counts is not None:
        counts = sorted(set(feature_counts), reverse=True)
    else:
        counts = _generate_feature_counts(
            len(all_features), min_features, step_function
        )

    # Guard against expensive forward_greedy runs
    if algorithm == 'forward_greedy':
        max_k = max(counts) if counts else K or len(all_features)
        n_feats = len(all_features)

        # Hard limits unless explicitly allowed
        MAX_FORWARD_GREEDY_K = 30
        MAX_FORWARD_GREEDY_FEATURES = 200

        if max_k > MAX_FORWARD_GREEDY_K or n_feats > MAX_FORWARD_GREEDY_FEATURES:
            raise ValueError(
                f"forward_greedy is O(K × n_features) and would require ~{max_k * n_feats} "
                f"model fits per split. Limits: K≤{MAX_FORWARD_GREEDY_K}, n_features≤{MAX_FORWARD_GREEDY_FEATURES}. "
                f"Use algorithm='forward' (fast heuristic) or 'permutation' (loss-change RFE) instead, "
                f"or reduce K/prefilter to fewer features."
            )

    if verbose:
        print(f"  K values to try: {counts[:5]}{'...' if len(counts) > 5 else ''}")
        print(f"  Algorithm: {algorithm}")

    # Setup cross-validation splitter
    # Priority: use_stability > cv > auto-select based on groups/task
    groups_array = groups.values if groups is not None else None

    if use_stability:
        # Group-resampled stability selection (group-aware if groups provided)
        # Pass y and task for classification class checking
        splits = list(_bootstrap_indices(
            n_samples, n_bootstrap,
            groups=groups_array,
            y=y,
            task=task,
            random_state=random_state
        ))
        if verbose:
            group_msg = " (group-aware)" if groups is not None else ""
            print(f"  Stability selection: {n_bootstrap} resampled splits{group_msg}")
    elif cv is not None:
        # Custom splitter provided (TimeSeriesSplit, GroupKFold, etc.)
        try:
            # Try with groups first (for GroupKFold, etc.)
            splits = list(cv.split(X_work, y, groups))
        except TypeError:
            # Splitter doesn't accept groups (TimeSeriesSplit, KFold, etc.)
            splits = list(cv.split(X_work, y))
        if verbose:
            print(f"  Custom CV: {type(cv).__name__} ({len(splits)} splits)")
    elif groups is not None:
        cv = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        splits = list(cv.split(X_work, y, groups))
        if verbose:
            print(f"  Group-aware splits: {n_splits}")
    elif task == 'classification':
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        splits = list(cv.split(X_work, y))
    else:
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        splits = list(cv.split(X_work, y))

    # Multi-split evaluation
    all_scores: Dict[int, List[float]] = defaultdict(list)
    all_features_by_k: Dict[int, List[List[str]]] = defaultdict(list)
    prefilter_features_first = None

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            print(f"  Split {fold_idx + 1}/{len(splits)}...", end=" ", flush=True)

        X_train, X_val = X_work.iloc[train_idx], X_work.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if sample_weights is not None:
            w_train = sample_weights.iloc[train_idx]
            w_val = sample_weights.iloc[val_idx]
        else:
            w_train, w_val = None, None

        # Pre-filter INSIDE the fold (avoids leakage)
        if prefilter_k is not None and prefilter_k < len(all_features):
            features = _prefilter_features(
                X_train, y_train, k=prefilter_k, task=task,
                method=prefilter_method, cat_features=cat_features_final,
                text_features=text_feat,
                random_state=random_state, verbose=False, n_jobs=n_jobs
            )
            if fold_idx == 0:
                prefilter_features_first = features
        else:
            features = all_features

        # Filter cat/text features to those in prefiltered set
        fold_cat = [f for f in cat_features_final if f in features]
        fold_text = [f for f in text_feat if f in features]

        # Adjust counts to this fold's feature count
        fold_counts = [min(k, len(features)) for k in counts]
        fold_counts = sorted(set(fold_counts), reverse=True)

        # Select algorithm
        if algorithm == 'forward':
            # Fast forward selection: importance ranking + score at requested Ks only
            scores, selected_feats = _forward_select_single_split(
                X_train, y_train, X_val, y_val, features, fold_counts,
                task=task, model_params=model_params,
                cat_features=fold_cat, text_features=fold_text,
                eval_metric=resolved_metric, higher_is_better=resolved_hib,
                w_train=w_train, w_val=w_val,
                importance_type='PredictionValuesChange',
                early_stopping_rounds=train_early_stopping_rounds,
            )
            # Build features_by_k from ranked list
            feats = {k: selected_feats[:k] for k in fold_counts if k <= len(selected_feats)}

        elif algorithm == 'forward_greedy':
            # True greedy forward selection (expensive)
            max_k = max(fold_counts) if fold_counts else K or len(features)
            scores, selected_feats = _forward_select_greedy_single_split(
                X_train, y_train, X_val, y_val, features, max_k,
                task=task, model_params=model_params,
                cat_features=fold_cat, text_features=fold_text,
                eval_metric=resolved_metric, higher_is_better=resolved_hib,
                w_train=w_train, w_val=w_val,
                early_stopping_rounds=train_early_stopping_rounds,
            )
            # Build features_by_k from greedy selection order
            feats = {k: selected_feats[:k] for k in fold_counts if k <= len(selected_feats)}

        else:
            # RFE-based selection (shap, loss-change, prediction)
            scores, feats = _select_features_single_split(
                X_train, y_train, X_val, y_val, features, fold_counts,
                task=task, model_params=model_params,
                cat_features=fold_cat, text_features=fold_text,
                eval_metric=resolved_metric, higher_is_better=resolved_hib,
                w_train=w_train, w_val=w_val,
                algorithm=algorithm, steps=steps,
                train_early_stopping_rounds=train_early_stopping_rounds,
            )

        for k, score in scores.items():
            all_scores[k].append(score)
        for k, feat_list in feats.items():
            all_features_by_k[k].append(feat_list)

        if verbose:
            if scores:
                best_k_fold, best_score_fold = best_score_from_dict(scores, resolved_hib)
                print(f"best K={best_k_fold}, score={best_score_fold:.4f}")
            else:
                print("no valid scores")

    # Aggregate scores across splits
    scores_mean = {k: np.mean(v) for k, v in all_scores.items()}
    scores_std = {k: np.std(v) for k, v in all_scores.items()}

    if not scores_mean:
        raise RuntimeError("No valid scores computed. Check your data and parameters.")

    # Find best K with proper early stopping
    # Start from FIRST K (largest), update as we find improvements
    sorted_counts = sorted(all_scores.keys(), reverse=True)  # Largest to smallest

    # Initialize with first K, not global best (fixes broken early stopping)
    best_k = sorted_counts[0]
    best_score = scores_mean[best_k]
    no_improve_count = 0

    for k in sorted_counts[1:]:  # Skip first, we already used it
        score = scores_mean[k]

        # Check if this is better than current best
        if resolved_hib:
            is_better = score > best_score
            rel_improvement = (score - best_score) / (abs(best_score) + 1e-10)
        else:
            is_better = score < best_score
            rel_improvement = (best_score - score) / (abs(best_score) + 1e-10)

        if is_better and rel_improvement > tolerance:
            best_score = score
            best_k = k
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= selection_patience:
            if verbose:
                print(f"  Early stopping at K={k}")
            break

    # Determine target_k from EVALUATED keys (handles prefilter_k < K case)
    max_eval_k = max(all_scores.keys()) if all_scores else 0

    if K is not None:
        if K > max_eval_k:
            warnings.warn(
                f"K={K} exceeds max evaluated feature count ({max_eval_k}) after "
                f"prefiltering/fit failures; using K={max_eval_k} instead."
            )
            target_k = max_eval_k
        else:
            # Choose closest evaluated K <= requested K
            valid_ks = [k for k in all_scores.keys() if k <= K]
            target_k = max(valid_ks) if valid_ks else max_eval_k
    else:
        target_k = best_k

    # Get final features using aggregation with rank tie-breaking
    # GUARANTEE: When K is specified, ALWAYS return exactly target_k features
    if use_stability and target_k in all_features_by_k:
        # Stability selection: aggregate with frequency + rank
        ordered_all, stability_scores = _aggregate_feature_lists(
            all_features_by_k[target_k], k=None  # Get all ordered features first
        )

        # Apply threshold, but ensure we return exactly target_k if specified
        stable_set = set(stability_scores[stability_scores >= stability_threshold].index)

        if K is not None:
            # GUARANTEE exactly target_k features when K is specified
            # Take stable features first, then fill from ordered list
            selected_features = [f for f in ordered_all if f in stable_set]
            if len(selected_features) < target_k:
                # Fill with remaining features in order
                for f in ordered_all:
                    if f not in selected_features:
                        selected_features.append(f)
                    if len(selected_features) >= target_k:
                        break
            selected_features = selected_features[:target_k]
        else:
            # No fixed K: return stable features or fallback to top by frequency
            stable_features = [f for f in ordered_all if f in stable_set]
            selected_features = stable_features if stable_features else ordered_all[:target_k]
    else:
        # Use aggregation across splits (frequency + rank)
        stability_scores = None
        if target_k in all_features_by_k and all_features_by_k[target_k]:
            ordered_all, stability_scores = _aggregate_feature_lists(
                all_features_by_k[target_k], k=None
            )
            selected_features = ordered_all[:target_k]
        else:
            selected_features = all_features[:target_k]

    # Final fallback: if selected_features < target_k (edge case from failed splits)
    if K is not None and len(selected_features) < target_k:
        fill_from = prefilter_features_first or all_features
        for f in fill_from:
            if f not in selected_features:
                selected_features.append(f)
            if len(selected_features) >= target_k:
                break

    # Build features_by_k using aggregation for each K
    features_by_k = {}
    for k, feat_lists in all_features_by_k.items():
        if feat_lists:
            agg_feats, _ = _aggregate_feature_lists(feat_lists, k=k)
            features_by_k[k] = agg_feats

    # Train final model on FULL DATA for importance computation
    # This gives more stable importances than aggregating per-split importances
    final_cat = [f for f in cat_features_final if f in selected_features]
    final_text = [f for f in text_feat if f in selected_features]

    ModelClass = CatBoostClassifier if task == 'classification' else CatBoostRegressor
    final_model = ModelClass(**model_params)
    final_pool = _create_pool(X_work, y, selected_features, sample_weights, final_cat, final_text)

    try:
        final_model.fit(final_pool, verbose=False)
        # Use LossFunctionChange for 'permutation' (matches RecursiveByLossFunctionChange)
        if algorithm == 'shap':
            importance_method = 'shap'
        elif algorithm == 'permutation':
            importance_method = 'loss'  # LossFunctionChange, not PredictionValuesChange
        else:
            importance_method = 'prediction'
        feature_importances = _compute_feature_importance(final_model, final_pool, method=importance_method)
    except Exception as e:
        warnings.warn(f"Failed to compute final importances: {e}")
        feature_importances = pd.Series(dtype=float)

    if verbose:
        print(f"Selected {len(selected_features)} features (best K={target_k}, score={scores_mean.get(target_k, best_score):.4f})")

    return CatBoostSelectionResult(
        selected_features=selected_features,
        best_k=target_k,
        scores_by_k=scores_mean,
        scores_std_by_k=scores_std,
        feature_importances=feature_importances,
        features_by_k=features_by_k,
        stability_scores=stability_scores,
        prefilter_features=prefilter_features_first,
        metric=resolved_metric,
        higher_is_better=resolved_hib,
        all_scores=dict(all_scores),
    )


# =============================================================================
# Convenience wrappers (match mrmr_regression/mrmr_classif API)
# =============================================================================

def catboost_regression(
    X: pd.DataFrame,
    y: pd.Series,
    K: int,
    cv: Optional[Any] = None,
    n_splits: int = 3,
    prefilter_k: Optional[int] = 200,
    prefilter_method: str = 'catboost',
    n_estimators: int = 500,
    algorithm: Literal['shap', 'permutation', 'prediction', 'forward', 'forward_greedy'] = 'shap',
    eval_metric: Optional[str] = None,
    group_col: Optional[str] = None,
    sample_weight_col: Optional[str] = None,
    text_features: Optional[List[str]] = None,
    gpu: bool = False,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    CatBoost feature selection for regression.

    Simple API matching mrmr_regression. For full control, use catboost_select().

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Continuous target.
    K : int
        Number of features to select.
    cv : sklearn splitter, optional
        Custom CV splitter (e.g., TimeSeriesSplit, GroupKFold).
    n_splits : int, default=3
        Number of CV splits (ignored if cv provided).
    prefilter_k : int, optional, default=200
        Pre-filter to top K features first (inside CV). Set None to disable.
    prefilter_method : str, default='catboost'
        Pre-filter method: 'catboost' (fast), 'cefsplus', 'mrmr', 'none'.
    n_estimators : int, default=500
        Number of boosting iterations.
    algorithm : str, default='shap'
        - 'shap': RFE with SHAP (accurate)
        - 'permutation': RFE with loss-function-change (balanced)
        - 'prediction': RFE with prediction change (fastest RFE)
        - 'forward': Forward selection by importance (fast)
        - 'forward_greedy': True greedy forward selection (expensive)
    eval_metric : str, optional
        Evaluation metric (default: 'RMSE').
    group_col : str, optional
        Column for group-aware splitting.
    sample_weight_col : str, optional
        Column for sample weights.
    text_features : list of str, optional
        Columns to treat as text (object/string are categorical by default).
    gpu : bool, default=False
        Use GPU.
    n_jobs : int, default=-1
        CPU threads.
    random_state : int, optional
        Random seed.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    list of str
        Selected feature names.

    Examples
    --------
    # Time series data
    from sklearn.model_selection import TimeSeriesSplit
    selected = catboost_regression(X, y, K=20, cv=TimeSeriesSplit(n_splits=5))

    # Fast forward selection
    selected = catboost_regression(X, y, K=20, algorithm='forward')
    """
    result = catboost_select(
        X, y, K=K, task='regression',
        cv=cv,
        n_splits=n_splits,
        prefilter_k=prefilter_k,
        prefilter_method=prefilter_method,
        n_estimators=n_estimators,
        algorithm=algorithm,
        eval_metric=eval_metric,
        group_col=group_col,
        sample_weight_col=sample_weight_col,
        text_features=text_features,
        gpu=gpu,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    return result.selected_features


def catboost_classif(
    X: pd.DataFrame,
    y: pd.Series,
    K: int,
    cv: Optional[Any] = None,
    n_splits: int = 3,
    prefilter_k: Optional[int] = 200,
    prefilter_method: str = 'catboost',
    n_estimators: int = 500,
    algorithm: Literal['shap', 'permutation', 'prediction', 'forward', 'forward_greedy'] = 'shap',
    eval_metric: Optional[str] = None,
    group_col: Optional[str] = None,
    sample_weight_col: Optional[str] = None,
    text_features: Optional[List[str]] = None,
    gpu: bool = False,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    CatBoost feature selection for classification.

    Simple API matching mrmr_classif. For full control, use catboost_select().

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : Series
        Categorical target.
    K : int
        Number of features to select.
    cv : sklearn splitter, optional
        Custom CV splitter (e.g., TimeSeriesSplit, GroupKFold).
    n_splits : int, default=3
        Number of CV splits (ignored if cv provided).
    prefilter_k : int, optional, default=200
        Pre-filter to top K features first (inside CV). Set None to disable.
    prefilter_method : str, default='catboost'
        Pre-filter method: 'catboost' (fast), 'mrmr', 'none'.
    n_estimators : int, default=500
        Number of boosting iterations.
    algorithm : str, default='shap'
        - 'shap': RFE with SHAP (accurate)
        - 'permutation': RFE with loss-function-change (balanced)
        - 'prediction': RFE with prediction change (fastest RFE)
        - 'forward': Forward selection by importance (fast)
        - 'forward_greedy': True greedy forward selection (expensive)
    eval_metric : str, optional
        Evaluation metric (default: 'Logloss').
    group_col : str, optional
        Column for group-aware splitting.
    sample_weight_col : str, optional
        Column for sample weights.
    text_features : list of str, optional
        Columns to treat as text (object/string are categorical by default).
    gpu : bool, default=False
        Use GPU.
    n_jobs : int, default=-1
        CPU threads.
    random_state : int, optional
        Random seed.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    list of str
        Selected feature names.
    """
    result = catboost_select(
        X, y, K=K, task='classification',
        cv=cv,
        n_splits=n_splits,
        prefilter_k=prefilter_k,
        prefilter_method=prefilter_method,
        n_estimators=n_estimators,
        algorithm=algorithm,
        eval_metric=eval_metric,
        group_col=group_col,
        sample_weight_col=sample_weight_col,
        text_features=text_features,
        gpu=gpu,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    return result.selected_features


__all__ = [
    'catboost_select',
    'catboost_regression',
    'catboost_classif',
    'CatBoostSelectionResult',
]
