"""Sklearn-style selector wrappers around top-level function selectors."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sift._preprocess import extract_feature_names
from sift.api import select_cefsplus, select_jmim, select_jmi, select_mrmr


def _coerce_selection_indices(
    feature_names: list[str], selected_features: list[str]
) -> np.ndarray:
    """Map selected feature names back to integer positions.

    Keep the first unmatched index for duplicate names so output remains aligned
    with a stable source-order selection path.
    """

    pools: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        pools.setdefault(name, []).append(i)

    used: dict[str, int] = {name: 0 for name in pools}
    indices: list[int] = []
    for name in selected_features:
        if name not in pools:
            raise ValueError(f"Selected feature '{name}' not found in fitted data.")
        pos = used[name]
        choices = pools[name]
        if pos >= len(choices):
            raise ValueError(f"Could not map selected feature '{name}' to a unique index.")
        indices.append(choices[pos])
        used[name] = pos + 1

    return np.asarray(indices, dtype=np.int64)


class _BaseSelector(BaseEstimator, TransformerMixin):
    """Sklearn-style compatibility layer for function-based selectors."""

    _selector_fn: Callable

    def _selector_params(self) -> dict:
        raise NotImplementedError

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        **fit_params,
    ):
        call_params = dict(self._selector_params())

        if cache is not None:
            call_params["cache"] = cache
        elif getattr(self, "cache", None) is not None:
            call_params["cache"] = self.cache

        if auto_k_config is not None:
            call_params["auto_k_config"] = auto_k_config
        elif getattr(self, "auto_k_config", None) is not None:
            call_params["auto_k_config"] = self.auto_k_config

        if groups is not None:
            call_params["groups"] = groups
        if time is not None:
            call_params["time"] = time

        call_params["sample_weight"] = sample_weight
        call_params.update(fit_params)

        selected_features = self._selector_fn(
            X,
            y,
            k=self.k,
            **call_params,
        )

        feature_names = extract_feature_names(X)
        if feature_names is None:
            n_features = np.asarray(X).shape[1]
            feature_names = [f"x{i}" for i in range(n_features)]

        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = list(selected_features)
        self.selected_indices_ = _coerce_selection_indices(
            feature_names,
            self.selected_features_,
        )
        return self

    def transform(self, X):
        check_is_fitted(self, ["selected_indices_", "selected_features_", "feature_names_in_"])
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("DataFrame columns must match fitted columns and order")
            return X.iloc[:, self.selected_indices_]
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but selector was fitted with "
                f"{self.n_features_in_}"
            )
        return X_arr[:, self.selected_indices_]

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return selected-feature mask (default) or indices (indices=True)."""
        check_is_fitted(self, ["selected_indices_", "n_features_in_"])
        if indices:
            return self.selected_indices_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask


class MRMRSelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_mrmr`."""

    def __init__(
        self,
        k: int | str = 10,
        *,
        task: str = "regression",
        relevance: str = "f",
        estimator: str = "classic",
        formula: str = "quotient",
        top_m: int | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "loo",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        n_jobs: int = 1,
        mrmr_backend: str = "auto",
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self.k = k
        self.task = task
        self.relevance = relevance
        self.estimator = estimator
        self.formula = formula
        self.top_m = top_m
        self.cat_features = cat_features
        self.cat_encoding = cat_encoding
        self.allow_full_data_target_encoding = allow_full_data_target_encoding
        self.subsample = subsample
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.mrmr_backend = mrmr_backend
        self.verbose = verbose
        self.cache = cache
        self.auto_k_config = auto_k_config

        self._selector_fn = select_mrmr

    def _selector_params(self) -> dict:
        return dict(
            task=self.task,
            relevance=self.relevance,
            estimator=self.estimator,
            formula=self.formula,
            top_m=self.top_m,
            cat_features=self.cat_features,
            cat_encoding=self.cat_encoding,
            allow_full_data_target_encoding=self.allow_full_data_target_encoding,
            subsample=self.subsample,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            mrmr_backend=self.mrmr_backend,
            verbose=self.verbose,
        )


class JMISelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_jmi`."""

    def __init__(
        self,
        k: int | str = 10,
        *,
        task: str = "regression",
        estimator: str = "auto",
        relevance: str = "f",
        top_m: int | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "loo",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self.k = k
        self.task = task
        self.estimator = estimator
        self.relevance = relevance
        self.top_m = top_m
        self.cat_features = cat_features
        self.cat_encoding = cat_encoding
        self.allow_full_data_target_encoding = allow_full_data_target_encoding
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        self.cache = cache
        self.auto_k_config = auto_k_config

        self._selector_fn = select_jmi

    def _selector_params(self) -> dict:
        return dict(
            task=self.task,
            estimator=self.estimator,
            relevance=self.relevance,
            top_m=self.top_m,
            cat_features=self.cat_features,
            cat_encoding=self.cat_encoding,
            allow_full_data_target_encoding=self.allow_full_data_target_encoding,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose,
        )


class JMIMSelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_jmim`."""

    def __init__(
        self,
        k: int | str = 10,
        *,
        task: str = "regression",
        estimator: str = "auto",
        relevance: str = "f",
        top_m: int | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "loo",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self.k = k
        self.task = task
        self.estimator = estimator
        self.relevance = relevance
        self.top_m = top_m
        self.cat_features = cat_features
        self.cat_encoding = cat_encoding
        self.allow_full_data_target_encoding = allow_full_data_target_encoding
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        self.cache = cache
        self.auto_k_config = auto_k_config

        self._selector_fn = select_jmim

    def _selector_params(self) -> dict:
        return dict(
            task=self.task,
            estimator=self.estimator,
            relevance=self.relevance,
            top_m=self.top_m,
            cat_features=self.cat_features,
            cat_encoding=self.cat_encoding,
            allow_full_data_target_encoding=self.allow_full_data_target_encoding,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose,
        )


class CEFSPlusSelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_cefsplus`."""

    def __init__(
        self,
        k: int | str = 75,
        *,
        top_m: int | None = None,
        corr_prune: float | None = 0.95,
        cat_features: list[str] | None = None,
        cat_encoding: str = "loo",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self.k = k
        self.top_m = top_m
        self.corr_prune = corr_prune
        self.cat_features = cat_features
        self.cat_encoding = cat_encoding
        self.allow_full_data_target_encoding = allow_full_data_target_encoding
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        self.cache = cache
        self.auto_k_config = auto_k_config

        self._selector_fn = select_cefsplus

    def _selector_params(self) -> dict:
        return dict(
            top_m=self.top_m,
            corr_prune=self.corr_prune,
            cat_features=self.cat_features,
            cat_encoding=self.cat_encoding,
            allow_full_data_target_encoding=self.allow_full_data_target_encoding,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose,
        )


__all__ = [
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
]
