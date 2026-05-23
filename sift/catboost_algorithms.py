"""Per-split CatBoost feature-selection algorithms."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from sift._preprocess import best_score_from_dict
from sift.catboost_common import (
    CatBoostClassifier,
    CatBoostRegressor,
    EFeaturesSelectionAlgorithm,
    EShapCalcType,
    _VALID_IMPORTANCE_TYPES,
    _compute_feature_importance,
    _create_pool,
    _extract_score,
    _validate_choice,
    _validate_step_function,
)

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
    then reconstruct feature sets for larger k using elimination order.

    IMPORTANT: Survivors are reordered by importance from the trained model,
    since CatBoost doesn't guarantee ordering in selected_features_names.

    Returns (scores_by_k, features_by_k).
    """
    _validate_choice("algorithm", algorithm, {"shap", "permutation", "prediction"})

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
    algo_enum = algo_map[algorithm]

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
            model_feats = list(getattr(model, "feature_names_", []))
            if model_feats:
                sel_cat = [f for f in cat_features if f in model_feats]
                sel_text = [f for f in text_features if f in model_feats]
                train_pool_sel = _create_pool(
                    X_train,
                    y_train,
                    model_feats,
                    w_train,
                    sel_cat,
                    sel_text,
                )
                try:
                    imp = model.get_feature_importance(
                        train_pool_sel, type='PredictionValuesChange'
                    )
                except Exception:
                    pass  # Keep original order if importance fails
                else:
                    if len(imp) != len(model_feats):
                        raise RuntimeError(
                            "CatBoost importance length mismatch with model feature names"
                        )
                    imp_series = pd.Series(imp, index=model_feats)
                    survivors = [f for f in survivors if f in imp_series.index]
                    survivors = (
                        imp_series.loc[survivors].sort_values(ascending=False).index.tolist()
                    )

        # Full ranking: survivors (sorted by importance) + eliminated in reverse (best losers first)
        ranked_features = list(survivors) + list(reversed(eliminated))

    scores = {}
    features_selected = {}

    # Evaluate each k
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

        # Retrain for exact score at this k
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
    _validate_step_function(step_function)

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
    2. Evaluate ONLY at requested k values (not every k from 1..max)

    This is O(len(feature_counts)) model fits, not O(max_k).
    Returns ranked features for subset reconstruction.

    NOTE: This is "prefix" selection, not true iterative forward selection.
    The ranking is computed once from the full model, not recomputed at each step.
    """
    _validate_choice("importance_type", importance_type, _VALID_IMPORTANCE_TYPES)

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

    # Step 2: Evaluate ONLY at requested k values
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

    This is O(k * n_remaining) model fits - expensive but principled.
    Use for small k or final refinement.
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
