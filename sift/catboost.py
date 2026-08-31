"""CatBoost-based feature selection orchestration and public wrappers."""

from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional
import inspect
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit

from sift._logging import logger
from sift._preprocess import best_score_from_dict, infer_higher_is_better
from sift.catboost_common import (
    CatBoostClassifier,
    CatBoostRegressor,
    CatBoostSelectionResult,
    _VALID_ALGORITHMS,
    _VALID_PREFILTER_METHODS,
    _VALID_TASKS,
    _compute_feature_importance,
    _create_pool,
    _get_feature_types,
    _prefilter_features,
    _resolve_loss_function,
    _resolve_metric_and_direction,
    _select_parsimonious_k,
    _validate_choice,
    _validate_group_splitter_groups,
    _validate_parsimony_params,
    _validate_stability_params,
    _validate_step_function,
)
from sift.catboost_algorithms import (
    _aggregate_feature_lists,
    _bootstrap_indices,
    _forward_select_greedy_single_split,
    _forward_select_single_split,
    _generate_feature_counts,
    _select_features_single_split,
)


def _normalize_catboost_target(y, index: pd.Index) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=index)
    return y


def _extract_weight_and_group_columns(
    X: pd.DataFrame,
    *,
    sample_weight_col: Optional[str],
    group_col: Optional[str],
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    X_work = X.copy()
    sample_weights = None
    groups = None

    if sample_weight_col is not None:
        if sample_weight_col not in X_work.columns:
            raise ValueError(f"sample_weight_col={sample_weight_col!r} not found in X")
        sample_weights = X_work[sample_weight_col]
        X_work = X_work.drop(columns=[sample_weight_col])

    if group_col is not None:
        if group_col not in X_work.columns:
            raise ValueError(f"group_col={group_col!r} not found in X")
        groups = X_work[group_col]
        X_work = X_work.drop(columns=[group_col])

    return X_work, sample_weights, groups


def _resolve_catboost_feature_types(
    X_work: pd.DataFrame,
    all_features: List[str],
    *,
    cat_features: Optional[List[str]],
    text_features: Optional[List[str]],
    treat_object_as_categorical: bool,
    verbose: bool,
) -> tuple[List[str], List[str]]:
    detected_cat, text_feat = _get_feature_types(
        X_work,
        all_features,
        text_features,
        treat_object_as_categorical,
    )

    if cat_features is not None:
        missing = [f for f in cat_features if f not in all_features]
        if missing:
            warnings.warn(
                f"cat_features not found in X (ignoring): {missing[:5]}",
                UserWarning,
                stacklevel=3,
            )
        cat_features_final = [f for f in cat_features if f in all_features]
        for f in detected_cat:
            if f not in cat_features_final:
                cat_features_final.append(f)
    else:
        cat_features_final = detected_cat

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
                "To exclude them, drop from X before calling.",
                UserWarning,
                stacklevel=3,
            )
            cat_features_final = list(cat_features_final)
            for c in orphan_obj:
                if c not in cat_features_final:
                    cat_features_final.append(c)

    if verbose and cat_features_final:
        logger.info(f"  Categorical features: {len(cat_features_final)}")

    return cat_features_final, text_feat


def _build_catboost_model_params(
    *,
    task: str,
    y: pd.Series,
    n_estimators: int,
    learning_rate: Optional[float],
    max_depth: Optional[int],
    eval_metric: Optional[str],
    loss_function: Optional[str],
    catboost_params: Optional[Dict[str, Any]],
    higher_is_better: Optional[bool],
    random_state: Optional[int],
    gpu: bool,
    n_jobs: int,
) -> tuple[Dict[str, Any], str, bool]:
    resolved_metric, resolved_hib = _resolve_metric_and_direction(
        task=task,
        y=y,
        eval_metric=eval_metric,
        higher_is_better=higher_is_better,
    )
    resolved_loss = _resolve_loss_function(task=task, y=y, loss_function=loss_function)

    model_params = {
        'iterations': n_estimators,
        'verbose': False,
        'allow_writing_files': False,
        'eval_metric': resolved_metric,
        'loss_function': resolved_loss,
    }
    if random_state is not None:
        model_params['random_seed'] = int(random_state)
    if max_depth is not None:
        model_params['depth'] = max_depth
    if learning_rate is not None:
        model_params['learning_rate'] = learning_rate
    if gpu:
        model_params['task_type'] = 'GPU'
        model_params['devices'] = '0'
    elif n_jobs > 0:
        model_params['thread_count'] = n_jobs

    if catboost_params:
        model_params.update(catboost_params)
        if 'eval_metric' in catboost_params:
            resolved_metric = str(catboost_params['eval_metric'])
            if higher_is_better is None:
                resolved_hib = infer_higher_is_better(resolved_metric)

    return model_params, resolved_metric, resolved_hib


def _resolve_catboost_counts(
    *,
    k_req: Optional[int],
    feature_counts: Optional[List[int]],
    n_features: int,
    min_features: int,
    step_function: float,
    algorithm: str,
) -> List[int]:
    if k_req is not None:
        counts = [k_req]
    elif feature_counts is not None:
        counts = sorted(set(feature_counts), reverse=True)
    else:
        counts = _generate_feature_counts(n_features, min_features, step_function)

    if algorithm == 'forward_greedy':
        max_k = max(counts) if counts else k_req or n_features
        max_forward_greedy_k = 30
        max_forward_greedy_features = 200
        if max_k > max_forward_greedy_k or n_features > max_forward_greedy_features:
            raise ValueError(
                f"forward_greedy is O(k x n_features) and would require ~{max_k * n_features} "
                f"model fits per split. Limits: k<={max_forward_greedy_k}, "
                f"n_features<={max_forward_greedy_features}. "
                "Use algorithm='forward' (fast heuristic) or 'permutation' "
                "(loss-change RFE) instead, or reduce k/prefilter to fewer features."
            )

    return counts


def _build_catboost_splits(
    *,
    X_work: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    cv: Optional[Any],
    use_stability: bool,
    n_samples: int,
    n_bootstrap: int,
    task: str,
    random_state: Optional[int],
    n_splits: int,
    test_size: float,
    verbose: bool,
):
    groups_array = groups.values if groups is not None else None
    if use_stability and cv is not None:
        raise ValueError("cv and use_stability=True are mutually exclusive")
    _validate_group_splitter_groups(cv, groups_array)

    if use_stability:
        splits = list(
            _bootstrap_indices(
                n_samples,
                n_bootstrap,
                groups=groups_array,
                y=y,
                task=task,
                random_state=random_state,
            )
        )
        if verbose:
            group_msg = " (group-aware)" if groups is not None else ""
            logger.info(f"  Stability selection: {n_bootstrap} resampled splits{group_msg}")
        return splits

    if cv is not None:
        try:
            split_parameters = inspect.signature(cv.split).parameters.values()
        except (TypeError, ValueError) as exc:
            raise TypeError("Cannot inspect custom cv.split signature") from exc
        accepts_groups = any(
            parameter.name == "groups"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in split_parameters
        )
        if groups_array is not None and not accepts_groups:
            raise TypeError(
                "group_col was provided, but custom cv.split does not accept groups"
            )
        splits = list(
            cv.split(X_work, y, groups=groups_array)
            if groups_array is not None
            else cv.split(X_work, y)
        )
        if verbose:
            logger.info(f"  Custom CV: {type(cv).__name__} ({len(splits)} splits)")
        return splits

    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
        )
        splits = list(splitter.split(X_work, y, groups))
        if verbose:
            logger.info(f"  Group-aware splits: {n_splits}")
        return splits

    if task == 'classification':
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=random_state,
        )
        return list(splitter.split(X_work, y))

    splitter = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    return list(splitter.split(X_work, y))


def _run_catboost_split_evaluation(
    *,
    X_work: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    splits,
    all_features: List[str],
    counts: List[int],
    task: str,
    model_params: Dict[str, Any],
    cat_features_final: List[str],
    text_feat: List[str],
    prefilter_k: Optional[int],
    prefilter_method: str,
    random_state: Optional[int],
    n_jobs: int,
    algorithm: str,
    resolved_metric: str,
    resolved_hib: bool,
    train_early_stopping_rounds: int,
    steps: int,
    k_req: Optional[int],
    verbose: bool,
) -> tuple[Dict[int, List[float]], Dict[int, List[List[str]]], Optional[List[str]]]:
    all_scores: Dict[int, List[float]] = defaultdict(list)
    all_features_by_k: Dict[int, List[List[str]]] = defaultdict(list)
    prefilter_features_first = None
    model_params = dict(model_params)
    user_controls_overfitting = any(
        key in model_params for key in ("od_type", "od_wait", "od_pval")
    )
    if not user_controls_overfitting:
        model_params["od_type"] = "Iter"
        model_params["od_wait"] = int(train_early_stopping_rounds)
    fit_early_stopping_rounds = (
        None if user_controls_overfitting else int(train_early_stopping_rounds)
    )

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if verbose:
            logger.info(f"  Split {fold_idx + 1}/{len(splits)}...")

        X_train, X_val = X_work.iloc[train_idx], X_work.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if sample_weights is not None:
            w_train = sample_weights.iloc[train_idx]
            w_val = sample_weights.iloc[val_idx]
        else:
            w_train, w_val = None, None

        if prefilter_k is not None and prefilter_k < len(all_features):
            features = _prefilter_features(
                X_train,
                y_train,
                k=prefilter_k,
                task=task,
                method=prefilter_method,
                cat_features=cat_features_final,
                text_features=text_feat,
                sample_weight=w_train,
                random_state=random_state,
                verbose=False,
                n_jobs=n_jobs,
            )
            if fold_idx == 0:
                prefilter_features_first = features
        else:
            features = all_features

        fold_cat = [f for f in cat_features_final if f in features]
        fold_text = [f for f in text_feat if f in features]
        fold_counts = sorted({min(kk, len(features)) for kk in counts}, reverse=True)

        if algorithm == 'forward':
            scores, selected_feats = _forward_select_single_split(
                X_train,
                y_train,
                X_val,
                y_val,
                features,
                fold_counts,
                task=task,
                model_params=model_params,
                cat_features=fold_cat,
                text_features=fold_text,
                eval_metric=resolved_metric,
                higher_is_better=resolved_hib,
                w_train=w_train,
                w_val=w_val,
                importance_type='PredictionValuesChange',
                early_stopping_rounds=fit_early_stopping_rounds,
            )
            feats = {kk: selected_feats[:kk] for kk in fold_counts if kk <= len(selected_feats)}
        elif algorithm == 'forward_greedy':
            max_k = max(fold_counts) if fold_counts else k_req or len(features)
            scores, selected_feats = _forward_select_greedy_single_split(
                X_train,
                y_train,
                X_val,
                y_val,
                features,
                max_k,
                task=task,
                model_params=model_params,
                cat_features=fold_cat,
                text_features=fold_text,
                eval_metric=resolved_metric,
                higher_is_better=resolved_hib,
                w_train=w_train,
                w_val=w_val,
                early_stopping_rounds=fit_early_stopping_rounds,
            )
            feats = {kk: selected_feats[:kk] for kk in scores if kk <= len(selected_feats)}
        else:
            scores, feats = _select_features_single_split(
                X_train,
                y_train,
                X_val,
                y_val,
                features,
                fold_counts,
                task=task,
                model_params=model_params,
                cat_features=fold_cat,
                text_features=fold_text,
                eval_metric=resolved_metric,
                higher_is_better=resolved_hib,
                w_train=w_train,
                w_val=w_val,
                algorithm=algorithm,
                steps=steps,
                train_early_stopping_rounds=fit_early_stopping_rounds,
            )

        for kk, score in scores.items():
            all_scores[kk].append(score)
        for kk, feat_list in feats.items():
            all_features_by_k[kk].append(feat_list)

        if verbose:
            if scores:
                best_k_fold, best_score_fold = best_score_from_dict(scores, resolved_hib)
                logger.info(
                    f"  Split {fold_idx + 1}/{len(splits)}: "
                    f"best k={best_k_fold}, score={best_score_fold:.4f}"
                )
            else:
                logger.info(f"  Split {fold_idx + 1}/{len(splits)}: no valid scores")

    return all_scores, all_features_by_k, prefilter_features_first


def _validate_selection_params(
    tolerance: float,
    selection_patience: int,
) -> tuple[float, int]:
    return _validate_parsimony_params(tolerance, selection_patience)


def _choose_catboost_target_k(
    all_scores: Dict[int, List[float]],
    *,
    k_req: Optional[int],
    resolved_hib: bool,
    tolerance: float,
    selection_patience: int,
    verbose: bool,
) -> tuple[int, int, float, Dict[int, float], Dict[int, float]]:
    """Pick the feature count from the per-k validation scores.

    ``best_k``/``best_score`` are the global arg-best over every evaluated
    count (ties go to the smaller count). When ``k`` was not requested, the
    returned ``target_k`` is the parsimonious choice: walking down from
    ``best_k``, the smallest count whose mean score stays within ``tolerance``
    (relative to ``|best_score|``) of the best, giving up after
    ``selection_patience`` consecutive counts outside that band so an isolated
    lucky tiny prefix far below the plateau is not taken. ``tolerance=0``
    therefore selects ``best_k`` unless smaller counts tie it exactly.
    """
    tolerance_float, patience = _validate_selection_params(
        tolerance,
        selection_patience,
    )
    scores_mean = {}
    scores_std = {}
    for kk, values in all_scores.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        scores_mean[kk] = float(np.mean(finite))
        scores_std[kk] = float(np.std(finite))
    if not scores_mean:
        raise RuntimeError("No valid scores computed. Check your data and parameters.")

    # Global arg-best over all evaluated counts; exact ties prefer fewer features.
    best_k, best_score = best_score_from_dict(scores_mean, resolved_hib)

    parsimonious_k = _select_parsimonious_k(
        scores_mean,
        best_k=best_k,
        best_score=best_score,
        higher_is_better=resolved_hib,
        tolerance=tolerance_float,
        selection_patience=patience,
    )

    if verbose:
        logger.info(
            f"  Best score {best_score:.4f} at k={best_k}; "
            f"parsimony rule (tolerance={tolerance_float:g}) selected "
            f"k={parsimonious_k}"
        )

    max_eval_k = max(scores_mean)
    if k_req is not None:
        if k_req > max_eval_k:
            warnings.warn(
                f"k={k_req} exceeds max evaluated feature count ({max_eval_k}) after "
                f"prefiltering/fit failures; using k={max_eval_k} instead.",
                UserWarning,
                stacklevel=3,
            )
            target_k = max_eval_k
        else:
            valid_ks = [kk for kk in scores_mean.keys() if kk <= k_req]
            target_k = max(valid_ks) if valid_ks else max_eval_k
    else:
        target_k = parsimonious_k

    return target_k, best_k, best_score, scores_mean, scores_std


def _select_final_catboost_features(
    *,
    target_k: int,
    k_req: Optional[int],
    all_features_by_k: Dict[int, List[List[str]]],
    all_features: List[str],
    prefilter_features_first: Optional[List[str]],
    use_stability: bool,
    stability_threshold: float,
) -> tuple[List[str], Optional[pd.Series]]:
    if use_stability and target_k in all_features_by_k:
        ordered_all, stability_scores = _aggregate_feature_lists(
            all_features_by_k[target_k],
            k=None,
        )
        stable_set = set(stability_scores[stability_scores >= stability_threshold].index)
        if k_req is not None:
            selected_features = [f for f in ordered_all if f in stable_set]
            if len(selected_features) < target_k:
                for f in ordered_all:
                    if f not in selected_features:
                        selected_features.append(f)
                    if len(selected_features) >= target_k:
                        break
            selected_features = selected_features[:target_k]
        else:
            stable_features = [f for f in ordered_all if f in stable_set]
            selected_features = (
                stable_features if stable_features else ordered_all[:target_k]
            )[:target_k]
    else:
        stability_scores = None
        if target_k in all_features_by_k and all_features_by_k[target_k]:
            ordered_all, _ = _aggregate_feature_lists(
                all_features_by_k[target_k],
                k=None,
            )
            selected_features = ordered_all[:target_k]
        else:
            raise RuntimeError(
                f"No feature list was recorded for selected k={target_k}; "
                "cannot safely choose final CatBoost features."
            )

    return selected_features, stability_scores


def _aggregate_catboost_features_by_k(
    all_features_by_k: Dict[int, List[List[str]]],
) -> Dict[int, List[str]]:
    features_by_k = {}
    for kk, feat_lists in all_features_by_k.items():
        if feat_lists:
            agg_feats, _ = _aggregate_feature_lists(feat_lists, k=kk)
            features_by_k[kk] = agg_feats
    return features_by_k


def _compute_final_catboost_importances(
    *,
    X_work: pd.DataFrame,
    y: pd.Series,
    sample_weights: Optional[pd.Series],
    selected_features: List[str],
    cat_features_final: List[str],
    text_feat: List[str],
    task: str,
    model_params: Dict[str, Any],
    algorithm: str,
) -> pd.Series:
    final_cat = [f for f in cat_features_final if f in selected_features]
    final_text = [f for f in text_feat if f in selected_features]
    model_cls = CatBoostClassifier if task == 'classification' else CatBoostRegressor
    final_params = dict(model_params)
    final_params.pop("od_type", None)
    final_params.pop("od_wait", None)
    final_model = model_cls(**final_params)
    final_pool = _create_pool(X_work, y, selected_features, sample_weights, final_cat, final_text)

    try:
        final_model.fit(final_pool, verbose=False)
        if algorithm == 'shap':
            importance_method = 'shap'
        elif algorithm == 'permutation':
            importance_method = 'loss'
        else:
            importance_method = 'prediction'
        return _compute_feature_importance(
            final_model,
            final_pool,
            method=importance_method,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to compute final importances: {exc}",
            UserWarning,
            stacklevel=3,
        )
        return pd.Series(dtype=float)




# =============================================================================
# Main API
# =============================================================================

def catboost_select(
    X: pd.DataFrame,
    y: pd.Series,
    k: Optional[int] = None,
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
    """Run CatBoost feature selection and return scores, paths, and final importances."""
    k_req = k
    if CatBoostRegressor is None:
        raise ImportError(
            "CatBoost is required for this function. "
            "Install with: pip install catboost"
        )

    _validate_choice("task", task, _VALID_TASKS)
    _validate_choice("algorithm", algorithm, _VALID_ALGORITHMS)
    _validate_choice("prefilter_method", prefilter_method, _VALID_PREFILTER_METHODS)
    _validate_step_function(step_function)
    _validate_stability_params(n_bootstrap, stability_threshold)
    _validate_selection_params(tolerance, selection_patience)

    y = _normalize_catboost_target(y, X.index)
    n_samples, n_features_orig = X.shape
    X_work, sample_weights, groups = _extract_weight_and_group_columns(
        X,
        sample_weight_col=sample_weight_col,
        group_col=group_col,
    )
    all_features = list(X_work.columns)

    model_params, resolved_metric, resolved_hib = _build_catboost_model_params(
        task=task,
        y=y,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric=eval_metric,
        loss_function=loss_function,
        catboost_params=catboost_params,
        higher_is_better=higher_is_better,
        random_state=random_state,
        gpu=gpu,
        n_jobs=n_jobs,
    )

    if verbose:
        direction = "up" if resolved_hib else "down"
        logger.info(
            f"CatBoost feature selection: {n_samples:,} samples x {n_features_orig} features"
        )
        logger.info(f"  Metric: {resolved_metric} ({direction} better)")

    cat_features_final, text_feat = _resolve_catboost_feature_types(
        X_work,
        all_features,
        cat_features=cat_features,
        text_features=text_features,
        treat_object_as_categorical=treat_object_as_categorical,
        verbose=verbose,
    )
    counts = _resolve_catboost_counts(
        k_req=k_req,
        feature_counts=feature_counts,
        n_features=len(all_features),
        min_features=min_features,
        step_function=step_function,
        algorithm=algorithm,
    )

    if verbose:
        logger.info(f"  k values to try: {counts[:5]}{'...' if len(counts) > 5 else ''}")
        logger.info(f"  Algorithm: {algorithm}")

    splits = _build_catboost_splits(
        X_work=X_work,
        y=y,
        groups=groups,
        cv=cv,
        use_stability=use_stability,
        n_samples=n_samples,
        n_bootstrap=n_bootstrap,
        task=task,
        random_state=random_state,
        n_splits=n_splits,
        test_size=test_size,
        verbose=verbose,
    )
    all_scores, all_features_by_k, prefilter_features_first = _run_catboost_split_evaluation(
        X_work=X_work,
        y=y,
        sample_weights=sample_weights,
        splits=splits,
        all_features=all_features,
        counts=counts,
        task=task,
        model_params=model_params,
        cat_features_final=cat_features_final,
        text_feat=text_feat,
        prefilter_k=prefilter_k,
        prefilter_method=prefilter_method,
        random_state=random_state,
        n_jobs=n_jobs,
        algorithm=algorithm,
        resolved_metric=resolved_metric,
        resolved_hib=resolved_hib,
        train_early_stopping_rounds=train_early_stopping_rounds,
        steps=steps,
        k_req=k_req,
        verbose=verbose,
    )
    target_k, best_k, best_score, scores_mean, scores_std = _choose_catboost_target_k(
        all_scores,
        k_req=k_req,
        resolved_hib=resolved_hib,
        tolerance=tolerance,
        selection_patience=selection_patience,
        verbose=verbose,
    )
    selected_features, stability_scores = _select_final_catboost_features(
        target_k=target_k,
        k_req=k_req,
        all_features_by_k=all_features_by_k,
        all_features=all_features,
        prefilter_features_first=prefilter_features_first,
        use_stability=use_stability,
        stability_threshold=stability_threshold,
    )
    features_by_k = _aggregate_catboost_features_by_k(all_features_by_k)
    feature_importances = _compute_final_catboost_importances(
        X_work=X_work,
        y=y,
        sample_weights=sample_weights,
        selected_features=selected_features,
        cat_features_final=cat_features_final,
        text_feat=text_feat,
        task=task,
        model_params=model_params,
        algorithm=algorithm,
    )

    if verbose:
        score = scores_mean.get(target_k, best_score)
        logger.info(
            f"Selected {len(selected_features)} features (k={target_k}, score={score:.4f}; "
            f"best-scoring k={best_k}, score={best_score:.4f})"
        )

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
        selection_patience=selection_patience,
    )


def _catboost_task_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    *,
    task: Literal['regression', 'classification'],
    **kwargs,
) -> List[str]:
    """Return just the selected feature names for a fixed CatBoost task."""
    return catboost_select(X, y, k=k, task=task, **kwargs).selected_features


def catboost_regression(X: pd.DataFrame, y: pd.Series, k: int, **kwargs) -> List[str]:
    """CatBoost feature selection for regression."""
    return _catboost_task_features(X, y, k, task='regression', **kwargs)


def catboost_classif(X: pd.DataFrame, y: pd.Series, k: int, **kwargs) -> List[str]:
    """CatBoost feature selection for classification."""
    return _catboost_task_features(X, y, k, task='classification', **kwargs)


__all__ = [
    'catboost_select',
    'catboost_regression',
    'catboost_classif',
    'CatBoostSelectionResult',
]
