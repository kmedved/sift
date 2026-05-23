"""Convenience APIs for stability selection."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sift.sampling.smart import SmartSamplerConfig
from sift.stability import StabilitySelector

def stability_select(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick stability selection.

    Returns
    -------
    selected_indices : ndarray
    frequencies : ndarray
    """
    selector = StabilitySelector(
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        **kwargs
    )
    selector.fit(X, y)
    return selector.selected_features_, selector.selection_frequencies_


def stability_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    k: int,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    block_size: int | str = "auto",
    block_method: str = "moving",
    alpha: Optional[float] = None,
    l1_ratio: float = 1.0,
    sample_frac: float = 0.5,
    use_smart_sampler: bool = False,
    sampler_config: Optional[SmartSamplerConfig] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
    return_indices: Optional[bool] = None,
) -> Union[List[str], List[int]]:
    """
    Stability selection for regression.

    Fits Lasso/ElasticNet on bootstrap subsamples and returns features
    selected consistently across runs.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Continuous target variable.
    k : int
        Maximum number of features to select.
    threshold : float, default=0.6
        Minimum selection frequency to keep a feature.
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    alpha : float, optional
        Regularization strength. If None, estimated via CV.
    l1_ratio : float, default=1.0
        ElasticNet mixing (1.0 = Lasso, <1.0 = ElasticNet).
    sample_frac : float, default=0.5
        Fraction of data per bootstrap sample.
    use_smart_sampler : bool, default=False
        Whether to apply leverage-based smart sampling.
    sampler_config : SmartSamplerConfig, optional
        Configuration for smart sampler.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : bool, default=True
        Print progress information.
    return_indices : bool, optional
        If True, return feature indices. If False, return feature names.
        If None, returns names for DataFrame inputs and indices for ndarray inputs.

    Returns
    -------
    selected_features : list of str or list of int
        Names or indices of selected features, depending on return_indices.
    """
    selector = StabilitySelector(
        task='regression',
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        l1_ratio=l1_ratio,
        sample_frac=sample_frac,
        max_features=k,
        use_smart_sampler=use_smart_sampler,
        sampler_config=sampler_config,
        block_size=block_size,
        block_method=block_method,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    selector.fit(X, y, sample_weight=sample_weight, groups=groups, time=time)
    if return_indices is None:
        return_indices = not isinstance(X, pd.DataFrame)
    if return_indices:
        return selector.selected_features_.tolist()
    return selector.selected_feature_names_


def stability_classif(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    k: int,
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    block_size: int | str = "auto",
    block_method: str = "moving",
    alpha: Optional[float] = None,
    sample_frac: float = 0.5,
    use_smart_sampler: bool = False,
    sampler_config: Optional[SmartSamplerConfig] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True,
    return_indices: Optional[bool] = None,
) -> Union[List[str], List[int]]:
    """
    Stability selection for classification.

    Fits L1-regularized LogisticRegression on bootstrap subsamples and
    returns features selected consistently across runs.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Categorical target variable.
    k : int
        Maximum number of features to select.
    threshold : float, default=0.6
        Minimum selection frequency to keep a feature.
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    alpha : float, optional
        Regularization strength. If None, estimated via CV.
    sample_frac : float, default=0.5
        Fraction of data per bootstrap sample.
    use_smart_sampler : bool, default=False
        Whether to apply leverage-based smart sampling.
    sampler_config : SmartSamplerConfig, optional
        Configuration for smart sampler.
    random_state : int, optional
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs.
    verbose : bool, default=True
        Print progress information.
    return_indices : bool, optional
        If True, return feature indices. If False, return feature names.
        If None, returns names for DataFrame inputs and indices for ndarray inputs.

    Returns
    -------
    selected_features : list of str or list of int
        Names or indices of selected features, depending on return_indices.
    """
    selector = StabilitySelector(
        task='classification',
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        sample_frac=sample_frac,
        max_features=k,
        use_smart_sampler=use_smart_sampler,
        sampler_config=sampler_config,
        block_size=block_size,
        block_method=block_method,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    selector.fit(X, y, sample_weight=sample_weight, groups=groups, time=time)
    if return_indices is None:
        return_indices = not isinstance(X, pd.DataFrame)
    if return_indices:
        return selector.selected_features_.tolist()
    return selector.selected_feature_names_
