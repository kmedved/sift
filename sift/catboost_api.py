"""Convenience wrappers around CatBoost feature selection."""

from __future__ import annotations

from typing import Any, List, Literal, Optional

import pandas as pd

from sift.catboost import catboost_select


def catboost_regression(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
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
    """CatBoost feature selection for regression."""
    result = catboost_select(
        X,
        y,
        k=k,
        task='regression',
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
    k: int,
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
    """CatBoost feature selection for classification."""
    result = catboost_select(
        X,
        y,
        k=k,
        task='classification',
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
    'catboost_regression',
    'catboost_classif',
]
