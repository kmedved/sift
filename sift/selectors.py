"""Sklearn-style selector wrappers around top-level function selectors."""

from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from sift._preprocess import (
    LeaveOneOutLogitEncoder,
    ensure_weights,
    extract_feature_names,
    suppress_category_encoder_pandas_warnings,
)
from sift.api import (
    _resolve_binary_weights,
    _validate_binary_target,
    select_cefsplus,
    select_cefsplus_binary,
    select_jmim,
    select_jmi,
    select_mrmr,
)
from sift.selection.auto_k import (
    _build_k_grid,
    _build_score_curve_diagnostics,
    _compute_metric,
    _resolve_metric,
    _split_weights,
    _time_holdout_split,
    choose_k_from_score_curve,
    resolve_auto_k_config,
)

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


def _numeric_train_val(
    X_train,
    X_val,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(X_train, pd.DataFrame):
        Xtr = X_train.to_numpy(dtype=np.float64, copy=False)
    else:
        Xtr = np.asarray(X_train, dtype=np.float64)
    if isinstance(X_val, pd.DataFrame):
        Xva = X_val.to_numpy(dtype=np.float64, copy=False)
    else:
        Xva = np.asarray(X_val, dtype=np.float64)

    with np.errstate(all="ignore"):
        col_means = np.nanmean(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)

    mask_tr = ~np.isfinite(Xtr)
    if mask_tr.any():
        Xtr = Xtr.copy()
        Xtr[mask_tr] = col_means[np.where(mask_tr)[1]]

    mask_va = ~np.isfinite(Xva)
    if mask_va.any():
        Xva = Xva.copy()
        Xva[mask_va] = col_means[np.where(mask_va)[1]]

    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xva)


class _BaseSelector(BaseEstimator, TransformerMixin):
    """Sklearn-style compatibility layer for function-based selectors."""

    _selector_fn: Callable

    def _selector_params(self) -> dict:
        raise NotImplementedError

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
            call_params.update(fit_params)

        feature_names = _feature_names_or_default(X)
        X_fit = self._fit_transform_categoricals(X, y, sample_weight=sample_weight)
        if getattr(self, "_categorical_encoding_applied_", False):
            call_params["cat_features"] = None
            call_params["cat_encoding"] = "none"
            call_params["allow_full_data_target_encoding"] = False

        selected_features = self._selector_fn(
            X_fit,
            y,
            k=k,
            **call_params,
        )

        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)
        self.selected_features_ = list(selected_features)
        self.selected_indices_ = _coerce_selection_indices(
            feature_names,
            self.selected_features_,
        )
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

    def _nested_splits(self, X, y_arr, groups, time, config):
        n = len(y_arr)
        if config.strategy == "time_holdout":
            if time is None:
                raise ValueError("auto_k_mode='nested' with time_holdout requires time")
            time_arr = np.asarray(time).reshape(-1)
            if len(time_arr) != n:
                raise ValueError(f"time has {len(time_arr)} rows but X/y have {n}")
            return [_time_holdout_split(time_arr, config.val_frac)]

        if config.strategy == "group_cv":
            if groups is None:
                raise ValueError("auto_k_mode='nested' with group_cv requires groups")
            group_arr = np.asarray(groups).reshape(-1)
            if len(group_arr) != n:
                raise ValueError(f"groups has {len(group_arr)} rows but X/y have {n}")
            n_unique = len(np.unique(group_arr))
            n_splits = min(config.n_splits, n_unique)
            if n_splits < 2:
                raise ValueError(f"group_cv requires at least 2 groups, got {n_unique}")
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(X, y_arr, group_arr))

        raise ValueError(f"Unknown auto_k strategy: {config.strategy}")

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

    def _evaluate_nested_prefixes(
        self,
        X_train_path,
        X_val_path,
        y_train,
        y_val,
        w_train,
        w_val,
        *,
        task: str,
        metric: str,
        k_grid: list[int],
    ) -> dict[int, float]:
        if X_train_path.shape[1] == 0:
            return {k: np.inf for k in k_grid}
        Xtr_s, Xva_s = _numeric_train_val(X_train_path, X_val_path)
        scores: dict[int, float] = {}
        alphas = np.logspace(-3, 3, 10)

        for k in k_grid:
            if k > Xtr_s.shape[1]:
                scores[k] = np.inf
                continue
            try:
                if task == "classification" and len(np.unique(y_train)) < 2:
                    scores[k] = np.inf
                    continue

                if task == "regression":
                    ridgecv = RidgeCV(alphas=alphas).fit(
                        Xtr_s[:, :k],
                        y_train,
                        sample_weight=w_train,
                    )
                    model = Ridge(alpha=float(ridgecv.alpha_))
                else:
                    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

                model.fit(Xtr_s[:, :k], y_train, sample_weight=w_train)

                if task == "classification" and metric == "logloss":
                    if not np.isin(np.unique(y_val), model.classes_).all():
                        scores[k] = np.inf
                    else:
                        proba = model.predict_proba(Xva_s[:, :k])
                        scores[k] = float(
                            log_loss(
                                y_val,
                                proba,
                                labels=model.classes_,
                                sample_weight=w_val,
                            )
                        )
                else:
                    pred = model.predict(Xva_s[:, :k])
                    scores[k] = _compute_metric(
                        y_val,
                        pred,
                        metric,
                        sample_weight=w_val,
                    )
            except Exception:
                scores[k] = np.inf
        return scores

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
        if n_features < 1:
            raise ValueError("X must contain at least one feature")

        config = auto_k_config
        max_k = min(int(config.max_k), n_features)
        min_k = max(1, min(int(config.min_k), max_k))
        k_grid = _build_k_grid(min_k, max_k)
        task = self._task()
        metric = _resolve_metric(config.metric, task)
        fit_w_arr = ensure_weights(sample_weight, len(y_arr), normalize=True)
        eval_w_arr = self._nested_eval_sample_weight(y, sample_weight)
        splits = self._nested_splits(X, y_arr, groups, time, config)

        all_scores = {k: [] for k in k_grid}
        fold_rows = []

        for split_id, (train_idx, val_idx) in enumerate(splits):
            train_idx = np.asarray(train_idx, dtype=np.int64)
            val_idx = np.asarray(val_idx, dtype=np.int64)
            fold_selector = self._clone_for_nested_path(max_k)

            train_X = _slice_rows(X, train_idx)
            X_train_path = fold_selector.fit_transform(
                train_X,
                y_arr[train_idx],
                sample_weight=fit_w_arr[train_idx],
                **(fit_params or {}),
            )

            X_val_path = fold_selector.transform(_slice_rows(X, val_idx))
            w_train = _split_weights(eval_w_arr, train_idx, "train")
            w_val = _split_weights(eval_w_arr, val_idx, "validation")
            split_scores = self._evaluate_nested_prefixes(
                X_train_path,
                X_val_path,
                y_arr[train_idx],
                y_arr[val_idx],
                w_train,
                w_val,
                task=task,
                metric=metric,
                k_grid=k_grid,
            )

            for k, score in split_scores.items():
                all_scores[k].append(score)
                fold_rows.append(
                    {
                        "split": split_id,
                        "k": k,
                        "score": score,
                        "path": tuple(
                            fold_selector.selected_features_[
                                : min(k, len(fold_selector.selected_features_))
                            ]
                        ),
                    }
                )

        score_df = _build_score_curve_diagnostics(k_grid, all_scores)
        if score_df.empty:
            selected_k = max_k
            score_best_k = None
        else:
            selected_k, score_df = choose_k_from_score_curve(
                score_df,
                config,
                lower_is_better=True,
            )
            score_best_k = (
                None if score_df.empty else int(score_df["best_k"].iloc[0])
            )

        self.nested_auto_k_diagnostics_ = {
            "mode": "nested",
            "strategy": config.strategy,
            "metric": metric,
            "selection_rule": config.selection_rule,
            "selection_rule_effective": (
                None
                if score_df.empty
                else str(score_df["selection_rule_effective"].iloc[0])
            ),
            "best_k": score_best_k,
            "selected_k": selected_k,
            "scores": score_df,
            "folds": pd.DataFrame(fold_rows),
        }
        self.k_ = selected_k

        return self._fit_selector(
            X,
            y,
            k=selected_k,
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
        cat_encoding: str = "none",
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
        cat_encoding: str = "none",
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
        cat_encoding: str = "none",
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


class CEFSPlusBinarySelector(_BaseSelector):
    """Sklearn-style wrapper for :func:`sift.select_cefsplus_binary`."""

    def __init__(
        self,
        k: int | str = 75,
        *,
        loss: str = "logloss",
        top_m: int | None = None,
        corr_prune: float | None = 0.95,
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
        self.k = k
        self.loss = loss
        self.top_m = top_m
        self.corr_prune = corr_prune
        self.class_weight = class_weight
        self.ridge = ridge
        self.refit_every = refit_every
        self.cat_features = cat_features
        self.cat_encoding = cat_encoding
        self.loo_smoothing = loo_smoothing
        self.loo_clip_min = loo_clip_min
        self.loo_clip_max = loo_clip_max
        self.allow_full_data_target_encoding = allow_full_data_target_encoding
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        self.auto_k_config = auto_k_config

        self._selector_fn = select_cefsplus_binary

    def _task(self) -> str:
        return "classification"

    def _categorical_target(self, y):
        y01, _, _ = _validate_binary_target(y)
        return y01

    def _categorical_sample_weight(self, y, sample_weight):
        y01, raw_y, _ = _validate_binary_target(y)
        weights, _ = _resolve_binary_weights(
            y01,
            raw_y,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        return weights

    def _nested_eval_sample_weight(self, y, sample_weight):
        y01, raw_y, _ = _validate_binary_target(y)
        weights, _ = _resolve_binary_weights(
            y01,
            raw_y,
            sample_weight=sample_weight,
            class_weight=self.class_weight,
        )
        return weights

    def _selector_params(self) -> dict:
        return dict(
            loss=self.loss,
            top_m=self.top_m,
            corr_prune=self.corr_prune,
            class_weight=self.class_weight,
            ridge=self.ridge,
            refit_every=self.refit_every,
            cat_features=self.cat_features,
            cat_encoding=self.cat_encoding,
            loo_smoothing=self.loo_smoothing,
            loo_clip_min=self.loo_clip_min,
            loo_clip_max=self.loo_clip_max,
            allow_full_data_target_encoding=self.allow_full_data_target_encoding,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=self.verbose,
        )

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
            blocked = sorted(_BINARY_PREPROCESSING_FIT_PARAM_OVERRIDES.intersection(fit_params))
            if blocked:
                blocked_text = ", ".join(blocked)
                raise ValueError(
                    "CEFSPlusBinarySelector preprocessing-affecting parameters "
                    f"must be set on the estimator before fit, not as fit-time "
                    f"overrides: {blocked_text}"
                )
            call_params.update(fit_params)
        call_params.pop("return_result", None)

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
                "cat_encoding='loo' for brier compatibility or loss='logloss' "
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


__all__ = [
    "MRMRSelector",
    "JMISelector",
    "JMIMSelector",
    "CEFSPlusSelector",
    "CEFSPlusBinarySelector",
]
