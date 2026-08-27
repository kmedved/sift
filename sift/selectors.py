"""Sklearn-style selector wrappers around top-level function selectors."""

from __future__ import annotations

import inspect
import importlib.util
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sift._preprocess import (
    LeaveOneOutLogitEncoder,
    ensure_weights,
    extract_feature_names,
    suppress_category_encoder_pandas_warnings,
)
from sift.api import (
    select_fdr,
    select_cefsplus_binary,
    select_cefsplus,
    select_jmim,
    select_jmi,
    select_mrmr,
)
from sift.selection.cefsplus_binary_common import (
    resolve_binary_weights,
    validate_binary_target,
)
from sift.selection.auto_k import AutoKConfig, resolve_auto_k_config
from sift.selection.auto_k_nested import NestedAutoKFold, select_k_nested
from sift.selection.knockoff_filter import _SUBSAMPLE_DEFAULT

_SUPERVISED_CLASS_ENCODINGS = frozenset({"loo", "target", "james_stein", "loo_logit"})
_BINARY_PREPROCESSING_FIT_PARAM_OVERRIDES = frozenset(
    {
        "loss",
        "cat_features",
        "cat_encoding",
        "class_weight",
        "loo_smoothing",
        "loo_clip_min",
        "loo_clip_max",
    }
)
_SELECTOR_FORWARD_SKIP_PARAMS = frozenset(
    {
        "X",
        "y",
        "k",
        "cache",
        "groups",
        "time",
        "auto_k_config",
        "sample_weight",
        "return_result",
    }
)
_BLOCKED_FIT_PARAM_OVERRIDES = frozenset(
    {
        "return_result",
        "cat_features",
        "cat_encoding",
        "allow_full_data_target_encoding",
    }
)


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


def _feature_names_or_default(X) -> list[str]:
    feature_names = extract_feature_names(X)
    if feature_names is not None:
        return list(feature_names)
    n_features = np.asarray(X).shape[1]
    return [f"x{i}" for i in range(n_features)]


def _slice_rows(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _categorical_columns(X: pd.DataFrame, cat_features: list[str] | None) -> list[str]:
    if cat_features is None:
        return X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return [col for col in cat_features if col in X.columns]


def _selected_training_output(X_fit, selected_indices: np.ndarray):
    """Return selected columns from the matrix used during selector fitting."""
    if isinstance(X_fit, pd.DataFrame):
        return X_fit.iloc[:, selected_indices].copy()
    return np.asarray(X_fit)[:, selected_indices].copy()


def _make_category_encoder(
    method: str,
    columns: list[str],
    *,
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
):
    if method == "none" or not columns:
        return None
    if method == "loo_logit":
        return LeaveOneOutLogitEncoder(
            columns,
            smoothing=loo_smoothing,
            clip_min=loo_clip_min,
            clip_max=loo_clip_max,
        )
    if method not in {"loo", "target", "james_stein"}:
        raise ValueError(
            "cat_encoding must be one of 'none', 'target', 'loo', 'james_stein', "
            "or 'loo_logit'. "
            f"Got {method!r}."
        )
    if importlib.util.find_spec("category_encoders") is None:
        raise ImportError(
            "cat_encoding requires category_encoders. Install with: pip install category_encoders"
        )

    import category_encoders as ce

    encoders = {
        "loo": ce.LeaveOneOutEncoder,
        "target": ce.TargetEncoder,
        "james_stein": ce.JamesSteinEncoder,
    }
    Encoder = encoders[method]
    try:
        return Encoder(
            cols=columns,
            handle_missing="return_nan",
            handle_unknown="value",
        )
    except TypeError:
        return Encoder(cols=columns, handle_missing="return_nan")


class _BaseSelector(BaseEstimator, TransformerMixin):
    """Sklearn-style compatibility layer for function-based selectors."""

    _selector_fn: Callable

    def _init_selector(self, selector_fn: Callable, params: dict) -> None:
        for name, value in params.items():
            if name != "self":
                setattr(self, name, value)
        self._selector_fn = selector_fn

    def _selector_params(self) -> dict:
        return {
            name: getattr(self, name)
            for name in inspect.signature(self._selector_fn).parameters
            if name not in _SELECTOR_FORWARD_SKIP_PARAMS and hasattr(self, name)
        }

    def _clear_fit_state(self) -> None:
        for attr in (
            "_fit_transform_output_",
            "categorical_encoder_",
            "categorical_features_",
            "_categorical_encoding_applied_",
            "feature_names_in_",
            "n_features_in_",
            "selected_features_",
            "selected_indices_",
            "k_",
            "nested_auto_k_diagnostics_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _task(self) -> str:
        return getattr(self, "task", "regression")

    def _supports_auto_k(self) -> bool:
        return True

    def _routes_no_config_auto_k(self) -> bool:
        """Whether k='auto' without a config should use the Auto-K router."""
        return False

    def _categorical_target(self, y):
        return y

    def _categorical_sample_weight(self, y, sample_weight):
        return sample_weight

    def _nested_eval_sample_weight(self, y, sample_weight):
        y_arr = np.asarray(y).reshape(-1)
        return ensure_weights(sample_weight, len(y_arr), normalize=True)

    def _would_fit_supervised_categoricals(self, X) -> bool:
        cat_encoding = getattr(self, "cat_encoding", "none")
        if cat_encoding not in _SUPERVISED_CLASS_ENCODINGS or not isinstance(X, pd.DataFrame):
            return False
        return bool(_categorical_columns(X, getattr(self, "cat_features", None)))

    def _fit_transform_categoricals(self, X, y, sample_weight=None):
        self.categorical_encoder_ = None
        self.categorical_features_ = []
        self._categorical_encoding_applied_ = False

        cat_encoding = getattr(self, "cat_encoding", "none")
        if cat_encoding == "none" or not isinstance(X, pd.DataFrame):
            return X

        cat_features = _categorical_columns(X, getattr(self, "cat_features", None))
        self.categorical_features_ = list(cat_features)
        if not cat_features:
            return X

        encoder = _make_category_encoder(
            cat_encoding,
            cat_features,
            loo_smoothing=getattr(self, "loo_smoothing", 20.0),
            loo_clip_min=getattr(self, "loo_clip_min", 1e-4),
            loo_clip_max=getattr(self, "loo_clip_max", 1.0 - 1e-4),
        )
        if sample_weight is not None and not isinstance(encoder, LeaveOneOutLogitEncoder):
            raise ValueError(
                "sample_weight with selector-class categorical encoding is only "
                "supported for cat_encoding='loo_logit'. category_encoders-backed "
                "methods ('loo', 'target', 'james_stein') do not consume sample weights."
            )
        y_enc = self._categorical_target(y)
        with suppress_category_encoder_pandas_warnings():
            if isinstance(encoder, LeaveOneOutLogitEncoder):
                X_encoded = encoder.fit_transform(
                    X,
                    y_enc,
                    sample_weight=self._categorical_sample_weight(y, sample_weight),
                )
            else:
                X_encoded = encoder.fit_transform(X, y_enc)

        self.categorical_encoder_ = encoder
        self._categorical_encoding_applied_ = True
        return X_encoded

    def _transform_categoricals(self, X):
        if not getattr(self, "_categorical_encoding_applied_", False):
            return X
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "This selector was fitted with DataFrame categorical encoding; "
                "transform also requires a DataFrame with matching columns."
            )
        with suppress_category_encoder_pandas_warnings():
            return self.categorical_encoder_.transform(X)

    def _fit_selector(
        self,
        X,
        y,
        *,
        k,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        fit_params=None,
        capture_training_output: bool = False,
    ):
        call_params = dict(self._selector_params())

        if cache is not None:
            call_params["cache"] = cache
        if auto_k_config is not None:
            call_params["auto_k_config"] = auto_k_config

        if groups is not None:
            call_params["groups"] = groups
        if time is not None:
            call_params["time"] = time

        call_params["sample_weight"] = sample_weight
        if fit_params:
            blocked = sorted(_BLOCKED_FIT_PARAM_OVERRIDES.intersection(fit_params))
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "selector fit-time overrides cannot change return shape or "
                    "preprocessing-affecting parameters: "
                    f"{blocked_text}"
                )
            call_params.update(fit_params)

        feature_names = _feature_names_or_default(X)
        X_fit = self._fit_transform_categoricals(X, y, sample_weight=sample_weight)
        if getattr(self, "_categorical_encoding_applied_", False):
            call_params["cat_features"] = None
            call_params["cat_encoding"] = "none"
            call_params["allow_full_data_target_encoding"] = False

        result = self._selector_fn(
            X_fit,
            y,
            k=k,
            return_result=True,
            **call_params,
        )
        if hasattr(result, "selected_indices"):
            selected_features = list(result.selected_features)
            selected_indices = result.selected_indices
        else:
            selected_features = list(result)
            selected_indices = None
        if selected_indices is None:
            selected_indices = _coerce_selection_indices(
                feature_names,
                selected_features,
            ).tolist()

        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = selected_features
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        if capture_training_output:
            self._fit_transform_output_ = _selected_training_output(
                X_fit,
                self.selected_indices_,
            )
        return self

    def _fit_impl(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        capture_training_output: bool = False,
        **fit_params,
    ):
        resolved_cache = cache if cache is not None else getattr(self, "cache", None)
        resolved_auto_k = auto_k_config
        if resolved_auto_k is None:
            resolved_auto_k = getattr(self, "auto_k_config", None)

        self._clear_fit_state()
        has_supervised_categoricals = self._would_fit_supervised_categoricals(X)

        if resolved_cache is not None and has_supervised_categoricals:
            raise ValueError(
                "selector-class supervised categorical encoding does not support "
                "prebuilt caches. Use cat_encoding='none' with a cache, or omit the "
                "cache so the selector can fit encoders on the training rows."
            )

        if self.k == "auto":
            if not self._supports_auto_k():
                raise ValueError(
                    f"{self.__class__.__name__} requires a fixed positive integer k; "
                    "k='auto' is not supported."
                )
            if resolved_auto_k is None and self._routes_no_config_auto_k():
                # Mirror the function API: CEFS+ selectors without an explicit
                # config use the measured Auto-K router instead of the legacy
                # evaluate/time_holdout inference.
                resolved_auto_k = AutoKConfig(k_method="auto")
            effective_auto_k = resolve_auto_k_config(
                resolved_auto_k,
                time,
                groups,
                allow_nested=True,
            )
            if effective_auto_k.auto_k_mode == "nested":
                if effective_auto_k.k_method != "evaluate":
                    raise ValueError(
                        "auto_k_mode='nested' currently supports only "
                        "k_method='evaluate'"
                    )
                return self._fit_nested_auto_k(
                    X,
                    y,
                    sample_weight=sample_weight,
                    groups=groups,
                    time=time,
                    cache=resolved_cache,
                    auto_k_config=effective_auto_k,
                    fit_params=fit_params,
                    capture_training_output=capture_training_output,
                )

            if (
                effective_auto_k.auto_k_mode == "prefix_only"
                and effective_auto_k.k_method == "evaluate"
                and has_supervised_categoricals
            ):
                raise ValueError(
                    "prefix_only auto-k with supervised selector-class categorical "
                    "encoding would evaluate target-encoded validation rows. Use "
                    "auto_k_mode='nested' or pre-encode/cross-fit categoricals "
                    "outside the selector."
                )

            resolved_auto_k = effective_auto_k

        return self._fit_selector(
            X,
            y,
            k=self.k,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
            cache=resolved_cache,
            auto_k_config=resolved_auto_k,
            fit_params=fit_params,
            capture_training_output=capture_training_output,
        )

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
        try:
            return self._fit_impl(
                X,
                y,
                sample_weight=sample_weight,
                groups=groups,
                time=time,
                cache=cache,
                auto_k_config=auto_k_config,
                capture_training_output=False,
                **fit_params,
            )
        except Exception:
            self._clear_fit_state()
            raise

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the selector and return the training matrix used for fitting.

        For supervised categorical encoders this avoids sklearn's default
        ``fit(X, y).transform(X)`` behavior, which would call a target-blind
        transform on the training rows and could differ from the y-aware encoded
        matrix used during feature selection.
        """
        try:
            self._fit_impl(X, y, capture_training_output=True, **fit_params)
            captured = getattr(self, "_fit_transform_output_", None)
            if captured is not None:
                return captured
            return self.transform(X)
        except Exception:
            self._clear_fit_state()
            raise
        finally:
            if hasattr(self, "_fit_transform_output_"):
                delattr(self, "_fit_transform_output_")

    def _clone_for_nested_path(self, k: int):
        params = self.get_params(deep=False)
        params["k"] = k
        if "auto_k_config" in params:
            params["auto_k_config"] = None
        if "cache" in params:
            params["cache"] = None
        if "verbose" in params:
            params["verbose"] = False
        return self.__class__(**params)

    def _fit_nested_auto_k(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        fit_params=None,
        capture_training_output: bool = False,
    ):
        if cache is not None:
            raise ValueError("auto_k_mode='nested' does not support prebuilt caches")

        y_arr = np.asarray(y).reshape(-1)
        n_features = len(_feature_names_or_default(X))
        config = auto_k_config
        fit_w_arr = (
            ensure_weights(sample_weight, len(y_arr), normalize=True)
            if sample_weight is not None
            else None
        )
        eval_w_arr = self._nested_eval_sample_weight(y, sample_weight)

        def build_fold_path(train_idx: np.ndarray, val_idx: np.ndarray, max_k: int):
            fold_selector = self._clone_for_nested_path(max_k)
            train_X = _slice_rows(X, train_idx)
            X_train_path = fold_selector.fit_transform(
                train_X,
                y_arr[train_idx],
                sample_weight=fit_w_arr[train_idx] if fit_w_arr is not None else None,
                **(fit_params or {}),
            )
            X_val_path = fold_selector.transform(_slice_rows(X, val_idx))
            return NestedAutoKFold(
                train_path=X_train_path,
                val_path=X_val_path,
                feature_path=list(fold_selector.selected_features_),
            )

        nested = select_k_nested(
            X,
            y_arr,
            n_features=n_features,
            config=config,
            build_fold_path=build_fold_path,
            groups=groups,
            time=time,
            sample_weight=eval_w_arr,
            task=self._task(),
        )
        self.nested_auto_k_diagnostics_ = nested.diagnostics
        self.k_ = nested.selected_k

        return self._fit_selector(
            X,
            y,
            k=nested.selected_k,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
            cache=None,
            auto_k_config=None,
            fit_params=fit_params,
            capture_training_output=capture_training_output,
        )

    def transform(self, X):
        check_is_fitted(
            self,
            ["selected_indices_", "selected_features_", "feature_names_in_"],
        )
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("DataFrame columns must match fitted columns and order")
            X = self._transform_categoricals(X)
            return X.iloc[:, self.selected_indices_]
        X_arr = np.asarray(X)
        if getattr(self, "_categorical_encoding_applied_", False):
            raise ValueError(
                "This selector was fitted with categorical DataFrame encoding; "
                "transform also requires a DataFrame."
            )
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
        cat_encoding: str = "none",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        n_jobs: int = 1,
        mrmr_backend: str = "auto",
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self._init_selector(select_mrmr, locals())


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
        cat_encoding: str = "none",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self._init_selector(select_jmi, locals())


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
        cat_encoding: str = "none",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self._init_selector(select_jmim, locals())


class CEFSPlusSelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_cefsplus`."""

    def __init__(
        self,
        k: int | str = 75,
        *,
        top_m: int | None = None,
        corr_prune: float | None = None,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = 50_000,
        random_state: int = 0,
        verbose: bool = True,
        cache=None,
        auto_k_config=None,
    ):
        self._init_selector(select_cefsplus, locals())

    def _routes_no_config_auto_k(self) -> bool:
        return True


class CEFSPlusBinarySelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_cefsplus_binary`."""

    def __init__(
        self,
        k: int | str = 75,
        *,
        loss: str = "logloss",
        top_m: int | None = None,
        corr_prune: float | None = None,
        class_weight=None,
        ridge: float = 1e-4,
        refit_every: int = 1,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        loo_smoothing: float = 20.0,
        loo_clip_min: float = 1e-4,
        loo_clip_max: float = 1.0 - 1e-4,
        allow_full_data_target_encoding: bool = False,
        subsample: int | None = None,
        random_state: int = 0,
        verbose: bool = True,
        auto_k_config=None,
    ):
        self._init_selector(select_cefsplus_binary, locals())

    def _routes_no_config_auto_k(self) -> bool:
        return True

    def _task(self) -> str:
        return "classification"

    def _categorical_target(self, y):
        y01, _, _ = validate_binary_target(y)
        return y01

    def _categorical_sample_weight(self, y, sample_weight):
        y01, raw_y, _ = validate_binary_target(y)
        weights, _ = resolve_binary_weights(
            y01,
            raw_y,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        return weights

    def _nested_eval_sample_weight(self, y, sample_weight):
        y_arr = np.asarray(y).reshape(-1)
        return ensure_weights(sample_weight, len(y_arr), normalize=True)

    def _fit_selector(
        self,
        X,
        y,
        *,
        k,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        fit_params=None,
        capture_training_output: bool = False,
    ):
        if cache is not None:
            raise ValueError("CEFSPlusBinarySelector does not support prebuilt caches.")

        call_params = dict(self._selector_params())
        call_params["sample_weight"] = sample_weight
        if groups is not None:
            call_params["groups"] = groups
        if time is not None:
            call_params["time"] = time
        if auto_k_config is not None:
            call_params["auto_k_config"] = auto_k_config
        if fit_params:
            blocked = sorted(
                _BLOCKED_FIT_PARAM_OVERRIDES.union(
                    _BINARY_PREPROCESSING_FIT_PARAM_OVERRIDES
                ).intersection(fit_params)
            )
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "CEFSPlusBinarySelector return-shape or preprocessing-affecting "
                    f"parameters must be set on the estimator before fit, not as fit-time "
                    f"overrides: {blocked_text}"
                )
            call_params.update(fit_params)

        loss_eff = str(self.loss).lower()
        if (
            loss_eff == "brier"
            and self.cat_encoding == "loo_logit"
            and isinstance(X, pd.DataFrame)
            and _categorical_columns(X, self.cat_features)
        ):
            raise ValueError(
                "CEFSPlusBinarySelector(loss='brier', cat_encoding='loo_logit') "
                "has no selector-class parity with the function API. Use "
                "cat_encoding='loo' for Brier proxy mode or loss='logloss' "
                "for logistic loo_logit encoding."
            )

        feature_names = _feature_names_or_default(X)
        X_fit = self._fit_transform_categoricals(X, y, sample_weight=sample_weight)
        if getattr(self, "_categorical_encoding_applied_", False):
            call_params["cat_features"] = None
            call_params["cat_encoding"] = "none"
            call_params["allow_full_data_target_encoding"] = False

        result = self._selector_fn(
            X_fit,
            y,
            k=k,
            return_result=True,
            **call_params,
        )
        selected_indices = result.selected_indices
        if selected_indices is None:
            selected_indices = _coerce_selection_indices(
                feature_names,
                list(result.selected_features),
            ).tolist()

        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = list(result.selected_features)
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        if capture_training_output:
            self._fit_transform_output_ = _selected_training_output(
                X_fit,
                self.selected_indices_,
            )
        return self


class KnockoffSelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_fdr`.

    ``subsample`` defaults to 50,000 rows when fitting from X. It is not valid
    with a prebuilt cache unless left at the untouched default.
    """

    def __init__(
        self,
        q: float = 0.1,
        *,
        statistic: str = "relevance",
        n_draws: int = 1,
        eta: float = 0.5,
        offset: int = 1,
        s_method: str = "equi",
        min_eig: float = 1e-3,
        screen_pairs: int | None = 2000,
        statistic_options: dict | None = None,
        feature_groups=None,
        group_corr_threshold: float = 0.7,
        cat_features: list[str] | None = None,
        cat_encoding: str = "none",
        allow_full_data_target_encoding: bool = False,
        loo_smoothing: float = 20.0,
        loo_clip_min: float = 1e-4,
        loo_clip_max: float = 1.0 - 1e-4,
        subsample=_SUBSAMPLE_DEFAULT,
        random_state: int = 0,
        n_jobs: int = 1,
        verbose: bool = True,
        cache=None,
    ):
        self._init_selector(select_fdr, locals())

    def _supports_auto_k(self) -> bool:
        return False

    def _clear_fit_state(self) -> None:
        super()._clear_fit_state()
        if hasattr(self, "result_"):
            delattr(self, "result_")

    def _fit_impl(
        self,
        X,
        y,
        *,
        sample_weight=None,
        groups=None,
        time=None,
        cache=None,
        auto_k_config=None,
        capture_training_output: bool = False,
        **fit_params,
    ):
        if groups is not None:
            raise ValueError(
                "KnockoffSelector does not support row groups. Use feature_groups "
                "on the estimator for grouped feature discoveries."
            )
        if time is not None:
            raise ValueError("KnockoffSelector does not support time-aware fitting.")
        if auto_k_config is not None:
            raise ValueError("KnockoffSelector is q-based and does not support auto_k_config.")

        resolved_cache = cache if cache is not None else getattr(self, "cache", None)
        if resolved_cache is not None and sample_weight is not None:
            raise ValueError(
                "sample_weight cannot be passed with a prebuilt cache; the cache "
                "already stores row weights"
            )

        self._clear_fit_state()
        has_supervised_categoricals = self._would_fit_supervised_categoricals(X)
        if resolved_cache is not None and has_supervised_categoricals:
            raise ValueError(
                "KnockoffSelector supervised categorical encoding does not support "
                "prebuilt caches. Use cat_encoding='none' with a cache, or omit the "
                "cache so the selector can fit encoders on the training rows."
            )

        call_params = dict(self._selector_params())
        if fit_params:
            blocked = sorted(_BLOCKED_FIT_PARAM_OVERRIDES.intersection(fit_params))
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "KnockoffSelector return-shape or preprocessing-affecting "
                    f"parameters must be set on the estimator before fit, not as "
                    f"fit-time overrides: {blocked_text}"
                )
            call_params.update(fit_params)

        feature_names = _feature_names_or_default(X)
        if resolved_cache is None:
            X_fit = self._fit_transform_categoricals(X, y, sample_weight=sample_weight)
            result = self._selector_fn(
                X_fit,
                y,
                sample_weight=sample_weight,
                **call_params,
            )
        else:
            if sample_weight is not None:
                raise ValueError("sample_weight cannot be passed with a prebuilt cache")
            cache_names = resolved_cache.feature_names
            if cache_names is not None and not resolved_cache.feature_names_are_synthetic:
                if list(feature_names) != list(cache_names):
                    raise ValueError(
                        "X columns do not match cache.feature_names (names and order must "
                        "be identical); fit the cache from the same matrix"
                    )
            elif cache_names is not None and len(feature_names) != len(cache_names):
                raise ValueError(
                    f"X has {len(feature_names)} columns but the cache was built from "
                    f"{len(cache_names)}"
                )
            X_fit = X
            result = self._selector_fn(
                None,
                y,
                cache=resolved_cache,
                sample_weight=None,
                **call_params,
            )

        selected_indices = result.selected_indices
        if selected_indices is None:
            selected_indices = _coerce_selection_indices(
                feature_names,
                list(result.selected_features),
            ).tolist()

        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = list(result.selected_features)
        self.selected_indices_ = np.asarray(selected_indices, dtype=np.int64)
        self.result_ = result
        if capture_training_output:
            self._fit_transform_output_ = _selected_training_output(
                X_fit,
                self.selected_indices_,
            )
        return self


__all__ = [
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
    "CEFSPlusBinarySelector",
    "KnockoffSelector",
]
