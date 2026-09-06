"""CatBoost-based feature selection orchestration and public wrappers."""

from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional
import inspect

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit

from sift._deprecate import warn_external, warn_random_state_none
from sift._logging import logger
from sift._metadata import resolve_row_metadata
from sift._progress import ProgressCallback, report_progress
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
from sift.selection import orchestration as _selection_orchestration
from sift.selection.orchestration import SelectionBackend
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


def _catboost_row_series(
    values: Any,
    index: pd.Index,
    *,
    argument: str,
) -> Optional[pd.Series]:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{argument} must be a one-dimensional row array")
    if array.shape[0] != len(index):
        raise ValueError(
            f"{argument} has {array.shape[0]} rows but X has {len(index)}"
        )
    return pd.Series(array, index=index, copy=True)


def _sort_catboost_rows_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series],
    groups: Optional[pd.Series],
    time: Optional[pd.Series],
) -> tuple[
    pd.DataFrame,
    pd.Series,
    Optional[pd.Series],
    Optional[pd.Series],
]:
    if time is None:
        return X, y, sample_weight, groups
    if pd.isna(time).any():
        raise ValueError("time must not contain missing values")
    try:
        order = np.argsort(time.to_numpy(), kind="mergesort")
    except (TypeError, ValueError) as exc:
        raise ValueError("time values must be mutually orderable") from exc
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    weight_sorted = (
        None
        if sample_weight is None
        else sample_weight.iloc[order].reset_index(drop=True)
    )
    groups_sorted = (
        None if groups is None else groups.iloc[order].reset_index(drop=True)
    )
    return X_sorted, y_sorted, weight_sorted, groups_sorted


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
            warn_external(
                f"cat_features not found in X (ignoring): {missing[:5]}",
                UserWarning,
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
            warn_external(
                f"treat_object_as_categorical=False but {len(orphan_obj)} object column(s) "
                f"are not in text_features or cat_features: {orphan_obj[:5]}. "
                "Auto-treating them as categorical to avoid CatBoost errors. "
                "To exclude them, drop from X before calling.",
                UserWarning,
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
        collisions = sorted(set(model_params).intersection(catboost_params))
        if collisions:
            warn_external(
                "catboost_params overrides translated SIFT arguments for: "
                f"{collisions}. The catboost_params values continue to win in "
                "SIFT 0.9; conflicting values will be rejected in SIFT 1.0.",
                UserWarning,
            )
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
                "groups were provided, but custom cv.split does not accept groups"
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
    callback: ProgressCallback | None = None,
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

        if callback is not None:
            if scores:
                callback_best_k, callback_best_score = best_score_from_dict(
                    scores, resolved_hib
                )
                callback_best_k = int(callback_best_k)
                callback_best_score = float(callback_best_score)
            else:
                callback_best_k, callback_best_score = None, None
            report_progress(
                callback,
                fold_idx + 1,
                len(splits),
                stage="split",
                train_rows=int(len(train_idx)),
                validation_rows=int(len(val_idx)),
                candidate_features=int(len(features)),
                evaluated_counts=int(len(scores)),
                best_k=callback_best_k,
                best_score=callback_best_score,
            )

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
            warn_external(
                f"k={k_req} exceeds max evaluated feature count ({max_eval_k}) after "
                f"prefiltering/fit failures; using k={max_eval_k} instead.",
                UserWarning,
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
        warn_external(
            f"Failed to compute final importances: {exc}",
            UserWarning,
        )
        return pd.Series(dtype=float)


class _CatBoostNativePreset(SelectionBackend):
    """Native CatBoost backend on the shared F6 selection runner.

    ``catboost_select`` and ``ModelSelector.fit`` both call
    ``sift.selection.orchestration.run_selection``. This backend keeps
    CatBoost SHAP/Pool, fold-local prefilter, fold voting, and explicit-k
    stability padding. It does not use generic coefficient ranking.
    """

    def prepare(self, X, y, **options) -> Dict[str, Any]:
        k_req = options["k"]
        if CatBoostRegressor is None:
            raise ImportError(
                "CatBoost is required for this function. "
                "Install with: pip install catboost"
            )

        _validate_choice("task", options["task"], _VALID_TASKS)
        _validate_choice("algorithm", options["algorithm"], _VALID_ALGORITHMS)
        _validate_choice(
            "prefilter_method", options["prefilter_method"], _VALID_PREFILTER_METHODS
        )
        _validate_step_function(options["step_function"])
        _validate_stability_params(
            options["n_bootstrap"], options["stability_threshold"]
        )
        _validate_selection_params(options["tolerance"], options["selection_patience"])

        y = _normalize_catboost_target(y, X.index)
        n_samples, n_features_orig = X.shape
        metadata = resolve_row_metadata(
            X,
            groups=options["groups"],
            time=options["time"],
            sample_weight=options["sample_weight"],
            group_col=options["group_col"],
            sample_weight_col=options["sample_weight_col"],
        )
        X_work = metadata.X
        sample_weights = _catboost_row_series(
            metadata.sample_weight,
            X.index,
            argument="sample_weight",
        )
        groups = _catboost_row_series(
            metadata.groups,
            X.index,
            argument="groups",
        )
        time_values = _catboost_row_series(
            metadata.time,
            X.index,
            argument="time",
        )
        X_work, y, sample_weights, groups = _sort_catboost_rows_by_time(
            X_work,
            y,
            sample_weights,
            groups,
            time_values,
        )
        if options["random_state"] is None:
            warn_random_state_none("catboost_select")
        all_features = list(X_work.columns)

        model_params, resolved_metric, resolved_hib = _build_catboost_model_params(
            task=options["task"],
            y=y,
            n_estimators=options["n_estimators"],
            learning_rate=options["learning_rate"],
            max_depth=options["max_depth"],
            eval_metric=options["eval_metric"],
            loss_function=options["loss_function"],
            catboost_params=options["catboost_params"],
            higher_is_better=options["higher_is_better"],
            random_state=options["random_state"],
            gpu=options["gpu"],
            n_jobs=options["n_jobs"],
        )

        verbose = options["verbose"]
        if verbose:
            direction = "up" if resolved_hib else "down"
            logger.info(
                f"CatBoost feature selection: {n_samples:,} samples x {n_features_orig} features"
            )
            logger.info(f"  Metric: {resolved_metric} ({direction} better)")

        cat_features_final, text_feat = _resolve_catboost_feature_types(
            X_work,
            all_features,
            cat_features=options["cat_features"],
            text_features=options["text_features"],
            treat_object_as_categorical=options["treat_object_as_categorical"],
            verbose=verbose,
        )
        counts = _resolve_catboost_counts(
            k_req=k_req,
            feature_counts=options["feature_counts"],
            n_features=len(all_features),
            min_features=options["min_features"],
            step_function=options["step_function"],
            algorithm=options["algorithm"],
        )

        if verbose:
            logger.info(
                f"  k values to try: {counts[:5]}{'...' if len(counts) > 5 else ''}"
            )
            logger.info(f"  Algorithm: {options['algorithm']}")

        splits = _build_catboost_splits(
            X_work=X_work,
            y=y,
            groups=groups,
            cv=options["cv"],
            use_stability=options["use_stability"],
            n_samples=n_samples,
            n_bootstrap=options["n_bootstrap"],
            task=options["task"],
            random_state=options["random_state"],
            n_splits=options["n_splits"],
            test_size=options["test_size"],
            verbose=verbose,
        )
        return {
            "k_req": k_req,
            "X_work": X_work,
            "y": y,
            "sample_weights": sample_weights,
            "all_features": all_features,
            "counts": counts,
            "splits": splits,
            "model_params": model_params,
            "resolved_metric": resolved_metric,
            "resolved_hib": resolved_hib,
            "cat_features_final": cat_features_final,
            "text_feat": text_feat,
            "options": options,
        }

    def evaluate_folds(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        options = prepared["options"]
        all_scores, all_features_by_k, prefilter_features_first = (
            _run_catboost_split_evaluation(
                X_work=prepared["X_work"],
                y=prepared["y"],
                sample_weights=prepared["sample_weights"],
                splits=prepared["splits"],
                all_features=prepared["all_features"],
                counts=prepared["counts"],
                task=options["task"],
                model_params=prepared["model_params"],
                cat_features_final=prepared["cat_features_final"],
                text_feat=prepared["text_feat"],
                prefilter_k=options["prefilter_k"],
                prefilter_method=options["prefilter_method"],
                random_state=options["random_state"],
                n_jobs=options["n_jobs"],
                algorithm=options["algorithm"],
                resolved_metric=prepared["resolved_metric"],
                resolved_hib=prepared["resolved_hib"],
                train_early_stopping_rounds=options["train_early_stopping_rounds"],
                steps=options["steps"],
                k_req=prepared["k_req"],
                verbose=options["verbose"],
                callback=options["callback"],
            )
        )
        return {
            "all_scores": all_scores,
            "all_features_by_k": all_features_by_k,
            "prefilter_features_first": prefilter_features_first,
        }

    def choose_count(self, prepared: Dict[str, Any], evaluated: Dict[str, Any]):
        options = prepared["options"]
        target_k, best_k, best_score, scores_mean, scores_std = (
            _choose_catboost_target_k(
                evaluated["all_scores"],
                k_req=prepared["k_req"],
                resolved_hib=prepared["resolved_hib"],
                tolerance=options["tolerance"],
                selection_patience=options["selection_patience"],
                verbose=options["verbose"],
            )
        )
        return {
            "target_k": target_k,
            "best_k": best_k,
            "best_score": best_score,
            "scores_mean": scores_mean,
            "scores_std": scores_std,
        }

    def evaluate(self, prepared: Dict[str, Any]) -> Dict[str, Any]:
        return self.evaluate_folds(prepared)

    def choose(self, prepared: Dict[str, Any], evaluated: Dict[str, Any]):
        return self.choose_count(prepared, evaluated)

    def finalize(
        self,
        prepared: Dict[str, Any],
        evaluated: Dict[str, Any],
        chosen: Dict[str, Any],
    ) -> CatBoostSelectionResult:
        options = prepared["options"]
        selected_features, stability_scores = _select_final_catboost_features(
            target_k=chosen["target_k"],
            k_req=prepared["k_req"],
            all_features_by_k=evaluated["all_features_by_k"],
            all_features=prepared["all_features"],
            prefilter_features_first=evaluated["prefilter_features_first"],
            use_stability=options["use_stability"],
            stability_threshold=options["stability_threshold"],
        )
        features_by_k = _aggregate_catboost_features_by_k(
            evaluated["all_features_by_k"]
        )
        feature_importances = _compute_final_catboost_importances(
            X_work=prepared["X_work"],
            y=prepared["y"],
            sample_weights=prepared["sample_weights"],
            selected_features=selected_features,
            cat_features_final=prepared["cat_features_final"],
            text_feat=prepared["text_feat"],
            task=options["task"],
            model_params=prepared["model_params"],
            algorithm=options["algorithm"],
        )

        if options["verbose"]:
            score = chosen["scores_mean"].get(
                chosen["target_k"], chosen["best_score"]
            )
            logger.info(
                f"Selected {len(selected_features)} features "
                f"(k={chosen['target_k']}, score={score:.4f}; "
                f"best-scoring k={chosen['best_k']}, "
                f"score={chosen['best_score']:.4f})"
            )

        return CatBoostSelectionResult(
            selected_features=selected_features,
            best_k=chosen["target_k"],
            scores_by_k=chosen["scores_mean"],
            scores_std_by_k=chosen["scores_std"],
            feature_importances=feature_importances,
            features_by_k=features_by_k,
            stability_scores=stability_scores,
            prefilter_features=evaluated["prefilter_features_first"],
            metric=prepared["resolved_metric"],
            higher_is_better=prepared["resolved_hib"],
            all_scores=dict(evaluated["all_scores"]),
            selection_patience=options["selection_patience"],
        )


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
    callback: ProgressCallback | None = None,
    groups: Any = None,
    time: Any = None,
    sample_weight: Any = None,
) -> CatBoostSelectionResult:
    """Run CatBoost feature selection and return scores, paths, and final importances.

    Gradient-boosted wrapper selection: for every candidate feature count the
    function runs CatBoost's own recursive elimination (or a forward search) on
    each train/validation split, averages the validation metric per count, then
    refits one final model on the retained features to report importances. Use
    it when the downstream model is CatBoost and a wrapper score is worth its
    cost; prefer the filter selectors when speed or a model-free ranking
    matters more. Pre-filtering, encoders and importance are computed inside
    each split, so validation rows never inform the candidate set.

    ``callback(step, total, info)`` is called after each completed split.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Candidate features. Categorical, text and numeric columns are allowed.
    y : Series of shape (n_samples,)
        Target. A DataFrame is narrowed to its first column, and array-likes
        are wrapped on ``X.index``.
    k : int or None, default=None
        Requested feature count. When given, only that count is evaluated and
        the largest evaluated count at or below it is returned. ``None``
        searches the whole ``feature_counts`` curve and applies the parsimony
        rule: starting from the global arg-best count (exact ties prefer the
        smaller one), walk towards smaller counts and keep the smallest whose
        mean score stays within ``tolerance`` of ``|best_score|``, giving up
        after ``selection_patience`` consecutive counts outside that band.
        ``tolerance=0`` therefore returns the arg-best unless a smaller count
        ties it exactly.
    task : {"regression", "classification"}, default="regression"
        Problem type; it selects ``CatBoostRegressor`` or
        ``CatBoostClassifier`` and the default metric and loss.
    min_features : int, default=5
        Smallest count in the generated search grid. Ignored when ``k`` or
        ``feature_counts`` is given.
    step_function : float, default=0.67
        Geometric ratio between successive generated counts, a float in
        ``(0, 1)``. Ignored when ``k`` or ``feature_counts`` is given.
    feature_counts : list of int or None, default=None
        Explicit search grid, deduplicated and evaluated largest first.
    selection_patience : int, default=3
        Positive number of consecutive out-of-band counts the parsimony walk
        tolerates before it stops.
    tolerance : float, default=0.01
        Non-negative relative score band for that walk.
    cv : splitter or None, default=None
        Any sklearn-compatible splitter (``TimeSeriesSplit``, ``GroupKFold``,
        ...). Its ``split`` must accept ``groups`` when ``groups`` is supplied.
        Mutually exclusive with ``use_stability=True``.
    n_splits : int, default=3
        Split count for the built-in shuffle splitters used when ``cv`` is
        ``None``.
    test_size : float, default=0.25
        Validation fraction for those built-in splitters.
    group_col : str or None, default=None
        Permanent alias for ``groups`` naming a column of ``X``; the column is
        copied out as row metadata and dropped from the candidate features.
        Passing both ``groups`` and ``group_col`` raises.
    sample_weight_col : str or None, default=None
        Permanent alias for ``sample_weight`` naming a column of ``X``, with
        the same extraction and the same conflict rule.
    prefilter_k : int or None, default=200
        Cap on candidates entering the CatBoost search, applied per split on
        training rows only. ``None`` disables pre-filtering, as does a value at
        or above the feature count.
    prefilter_method : str, default="catboost"
        Ranking used for that pre-filter: ``"catboost"``, ``"cefsplus"``,
        ``"mrmr"`` or ``"none"``.
    use_stability : bool, default=False
        Replace the CV splits with group-aware bootstrap resamples and keep
        features by selection frequency.
    n_bootstrap : int, default=20
        Positive number of resamples for ``use_stability=True``.
    stability_threshold : float, default=0.6
        Selection frequency in ``[0, 1]`` a feature must reach in that mode.
    n_estimators : int, default=500
        CatBoost ``iterations`` for every fitted model.
    learning_rate : float or None, default=None
        CatBoost ``learning_rate``; ``None`` leaves CatBoost's own default.
    max_depth : int or None, default=6
        CatBoost ``depth``; ``None`` leaves CatBoost's own default.
    eval_metric : str or None, default=None
        Validation metric. ``None`` resolves to ``"RMSE"`` for regression,
        ``"Logloss"`` for two-class and ``"MultiClass"`` for multiclass
        targets.
    loss_function : str or None, default=None
        Training loss, resolved from ``task`` and the class count the same way.
    catboost_params : dict or None, default=None
        Raw CatBoost parameters merged last. Keys that collide with the
        translated SIFT arguments emit one ``UserWarning`` and keep the
        historical ``catboost_params``-wins precedence; SIFT 1.0 will reject
        conflicting values instead. An ``eval_metric`` supplied here also
        redefines the reported metric and, unless ``higher_is_better`` is
        explicit, its direction.
    algorithm : str, default="shap"
        Search strategy: ``"shap"``, ``"permutation"``, ``"prediction"``,
        ``"forward"`` or ``"forward_greedy"``. The first three drive
        CatBoost's recursive ``select_features`` by SHAP values, loss-function
        change or prediction-values change; ``"forward"`` is an
        importance-ordered
        forward sweep and ``"forward_greedy"`` an exhaustive forward search
        capped at ``k <= 30`` and ``n_features <= 200``.
    steps : int, default=6
        Elimination steps handed to CatBoost's ``select_features`` for the
        three recursive algorithms.
    cat_features : list of str or None, default=None
        Extra categorical columns, merged with the detected ones. Names absent
        from ``X`` are ignored with a ``UserWarning``.
    text_features : list of str or None, default=None
        Columns to pass to CatBoost as text features.
    treat_object_as_categorical : bool, default=True
        Treat ``object``/``string`` columns as categorical. With ``False``, any
        orphan object column is still auto-treated as categorical with a
        ``UserWarning``, because CatBoost would otherwise fail.
    train_early_stopping_rounds : int, default=20
        Overfitting-detector patience for the per-split fits. It is skipped
        when ``catboost_params`` already sets ``od_type``, ``od_wait`` or
        ``od_pval``, and it is never applied to the final full-data refit.
    gpu : bool, default=False
        Run CatBoost with ``task_type="GPU"`` on device 0. GPU execution
        ignores ``n_jobs``.
    n_jobs : int, default=-1
        Thread count for CPU CatBoost fits and for the pre-filter. Positive
        values become CatBoost's ``thread_count``.
    higher_is_better : bool or None, default=None
        Metric direction. ``None`` infers it from the resolved metric name.
    random_state : int or None, default=None
        Seed for splits, bootstraps, pre-filtering and CatBoost's
        ``random_seed``. Leaving it ``None`` emits a ``FutureWarning``: 0.9
        stays nondeterministic while SIFT 1.0 will default to seed 0.
    verbose : bool, default=True
        Emit progress at INFO on the ``sift`` logger.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed split.
    groups : array-like, str or None, default=None
        Row group labels, or the name of an ``X`` column to extract and drop.
        Groups switch the default splitter to ``GroupShuffleSplit`` and make
        ``use_stability`` resampling group-aware.
    time : array-like, str or None, default=None
        Row time values, or the name of an ``X`` column to extract and drop.
        SIFT stably orders every aligned row by ``time`` before splitting.
        Missing or mutually unorderable values raise. Ordering alone does not
        make the default random splitter chronological; pass a time-aware
        ``cv`` when that is required.
    sample_weight : array-like or None, default=None
        Non-negative row weights forwarded to the CatBoost pools.

    Returns
    -------
    CatBoostSelectionResult
        Dataclass carrying ``selected_features``, the chosen ``best_k``,
        ``scores_by_k``/``scores_std_by_k``/``all_scores``, ``features_by_k``,
        the final-model ``feature_importances``, ``prefilter_features``,
        ``stability_scores``, ``metric``, ``higher_is_better`` and
        ``selection_patience``. Call ``result.result_view()`` for the
        normalized ``SelectionView``.

    Raises
    ------
    ImportError
        If the optional ``catboost`` package is not installed.
    ValueError
        For an invalid ``task``, ``algorithm``, ``prefilter_method``,
        ``step_function``, stability or parsimony option; when a direct
        argument and its ``*_col`` alias are both given; when a ``*_col`` name
        is missing, ambiguous or used with a non-DataFrame ``X``; when ``cv``
        is combined with ``use_stability=True``; when ``time`` holds missing or
        unorderable values; or when ``algorithm="forward_greedy"`` exceeds its
        size limits.
    TypeError
        If a custom ``cv.split`` cannot be inspected, or does not accept
        ``groups`` while ``groups`` is supplied.
    RuntimeError
        If no split produced a finite score, or no feature list was recorded
        for the selected count.

    Warns
    -----
    UserWarning
        For ``catboost_params`` collisions, unknown ``cat_features`` names,
        auto-treated orphan object columns, a requested ``k`` above the largest
        evaluated count, a failed ``select_features`` call that falls back to
        importance ranking, and a failed final importance computation.
    FutureWarning
        When ``random_state`` is left at ``None``.

    See Also
    --------
    catboost_regression : Regression wrapper returning only the feature names.
    catboost_classif : Classification wrapper returning only the names.
    ModelSelector : Generic sklearn-estimator F6 selector; not this native path.
    sift.select_boruta : All-relevant tree-based alternative.

    Notes
    -----
    CatBoost is an optional dependency; install it with
    ``python -m pip install -e ".[catboost]"``. ``sift.catboost_select`` is a
    lazy export, so importing ``sift`` never requires the extra, and this
    function raises ``ImportError`` only when actually called without it.
    This is the F6 CatBoost preset: it runs the shared internal
    ``run_selection`` contract (prepare, evaluate, choose, finalize) with a
    native SHAP/Pool backend. It is not ``ModelSelector`` ranking and not
    nested scoring. Reported CV scores stay the historical per-count
    validation curve. ``best_k`` on the returned
    result is the count that was selected, which for ``k=None`` is the
    parsimonious pick and can differ from the raw best-scoring count; read
    ``scores_by_k`` for the full curve.

    Examples
    --------
    These examples require the optional ``catboost`` extra. They are shown as
    code rather than executed doctests because SIFT's own test environment
    does not install CatBoost::

        import numpy as np
        import pandas as pd
        import sift

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(400, 8)),
                         columns=[f"f{i}" for i in range(8)])
        y = pd.Series(2.0 * X["f0"] + X["f1"] + 0.1 * rng.normal(size=400))

        result = sift.catboost_select(
            X, y, k=3, task="regression", algorithm="forward",
            random_state=0, verbose=False,
        )
        result.selected_features        # three retained column names
        result.scores_by_k              # {k: mean validation score}

        # Panel data: name the metadata columns instead of passing arrays.
        panel_result = sift.catboost_select(
            panel_df, panel_df["target"], groups="entity_id", time="date",
            random_state=0,
        )
    """
    return _selection_orchestration.run_selection(
        _CatBoostNativePreset(),
        X,
        y,
        k=k,
        task=task,
        min_features=min_features,
        step_function=step_function,
        feature_counts=feature_counts,
        selection_patience=selection_patience,
        tolerance=tolerance,
        cv=cv,
        n_splits=n_splits,
        test_size=test_size,
        group_col=group_col,
        sample_weight_col=sample_weight_col,
        prefilter_k=prefilter_k,
        prefilter_method=prefilter_method,
        use_stability=use_stability,
        n_bootstrap=n_bootstrap,
        stability_threshold=stability_threshold,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        eval_metric=eval_metric,
        loss_function=loss_function,
        catboost_params=catboost_params,
        algorithm=algorithm,
        steps=steps,
        cat_features=cat_features,
        text_features=text_features,
        treat_object_as_categorical=treat_object_as_categorical,
        train_early_stopping_rounds=train_early_stopping_rounds,
        gpu=gpu,
        n_jobs=n_jobs,
        higher_is_better=higher_is_better,
        random_state=random_state,
        verbose=verbose,
        callback=callback,
        groups=groups,
        time=time,
        sample_weight=sample_weight,
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


def catboost_regression(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    *,
    callback: ProgressCallback | None = None,
    **kwargs,
) -> List[str]:
    """CatBoost feature selection for regression.

    One-call wrapper that runs ``catboost_select`` with
    ``task="regression"`` and returns only the retained column names. Use it
    when the score curve, importances and diagnostics are not needed; call
    ``catboost_select`` directly for the full
    ``sift.catboost_common.CatBoostSelectionResult``.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Candidate features.
    y : Series of shape (n_samples,)
        Continuous target.
    k : int
        Requested feature count, forwarded as ``catboost_select(k=...)``. Pass
        ``k=None`` to search the whole curve and let the parsimony rule choose.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed split.
    **kwargs
        Every other ``catboost_select`` keyword, including ``algorithm``,
        ``cv``, ``prefilter_k``, ``catboost_params``, ``random_state``, the
        ``groups``/``time``/``sample_weight`` row arrays and their permanent
        ``group_col``/``sample_weight_col`` aliases (a direct value and its
        alias together raise).

    Returns
    -------
    list of str
        The selected feature names.

    Raises
    ------
    ImportError
        If the optional ``catboost`` package is not installed.
    ValueError
        Propagated from ``catboost_select`` for invalid options, alias
        conflicts or unorderable ``time`` values.

    Warns
    -----
    UserWarning
        Propagated from ``catboost_select``, notably the
        ``catboost_params``-wins collision notice.
    FutureWarning
        When ``random_state`` is left at ``None``.

    See Also
    --------
    catboost_select : The full-result entry point and option reference.
    catboost_classif : The classification counterpart.

    Notes
    -----
    CatBoost is an optional dependency; install it with
    ``python -m pip install -e ".[catboost]"``. The name is a lazy export from
    ``sift``, so importing the package never requires the extra.

    Examples
    --------
    This example requires the optional ``catboost`` extra and is shown as code
    rather than an executed doctest, because SIFT's own test environment does
    not install CatBoost::

        import sift

        features = sift.catboost_regression(
            X, y, k=20, algorithm="forward", random_state=0, verbose=False,
        )
    """
    return _catboost_task_features(
        X, y, k, task='regression', callback=callback, **kwargs
    )


def catboost_classif(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    *,
    callback: ProgressCallback | None = None,
    **kwargs,
) -> List[str]:
    """CatBoost feature selection for classification.

    One-call wrapper that runs ``catboost_select`` with
    ``task="classification"`` and returns only the retained column names. Use
    it when the score curve, importances and diagnostics are not needed; call
    ``catboost_select`` directly for the full
    ``sift.catboost_common.CatBoostSelectionResult``.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Candidate features.
    y : Series of shape (n_samples,)
        Class labels. Two-class targets default to the ``Logloss`` metric and
        loss; three or more classes default to ``MultiClass``.
    k : int
        Requested feature count, forwarded as ``catboost_select(k=...)``. Pass
        ``k=None`` to search the whole curve and let the parsimony rule choose.
    callback : ProgressCallback or None, default=None
        ``callback(step, total, info)`` called after each completed split.
    **kwargs
        Every other ``catboost_select`` keyword, including ``algorithm``,
        ``cv``, ``prefilter_k``, ``catboost_params``, ``random_state``, the
        ``groups``/``time``/``sample_weight`` row arrays and their permanent
        ``group_col``/``sample_weight_col`` aliases (a direct value and its
        alias together raise).

    Returns
    -------
    list of str
        The selected feature names.

    Raises
    ------
    ImportError
        If the optional ``catboost`` package is not installed.
    ValueError
        Propagated from ``catboost_select`` for invalid options, alias
        conflicts or unorderable ``time`` values.

    Warns
    -----
    UserWarning
        Propagated from ``catboost_select``, notably the
        ``catboost_params``-wins collision notice.
    FutureWarning
        When ``random_state`` is left at ``None``.

    See Also
    --------
    catboost_select : The full-result entry point and option reference.
    catboost_regression : The regression counterpart.

    Notes
    -----
    CatBoost is an optional dependency; install it with
    ``python -m pip install -e ".[catboost]"``. The name is a lazy export from
    ``sift``, so importing the package never requires the extra. Without an
    explicitly time-aware ``cv``, splits stay stratified-random even when
    ``time`` is supplied; ``time`` only fixes the row order.

    Examples
    --------
    This example requires the optional ``catboost`` extra and is shown as code
    rather than an executed doctest, because SIFT's own test environment does
    not install CatBoost::

        import sift

        features = sift.catboost_classif(
            X, y_binary, k=20, algorithm="forward", random_state=0,
            verbose=False,
        )
    """
    return _catboost_task_features(
        X, y, k, task='classification', callback=callback, **kwargs
    )


__all__ = [
    'catboost_select',
    'catboost_regression',
    'catboost_classif',
    'CatBoostSelectionResult',
]
