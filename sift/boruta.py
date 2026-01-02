"""
Boruta and Boruta-Shap feature selection.

Design:
- Single Boruta loop with pluggable importance backend
- Time-series aware shadow permutations
- Sample weight support throughout
- Optional train/test split for importance computation
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from sift._permute import (
    PermutationAxis,
    PermutationMethod,
    build_group_info,
    permute_matrix,
    resolve_permutation_method,
)
from sift._preprocess import CatEncoding, encode_categoricals, ensure_weights, extract_feature_names, to_numpy

ImportanceBackend = Literal["native", "shap"]
Task = Literal["regression", "classification"]


# =============================================================================
# Helper Functions
# =============================================================================


def _clone_estimator(estimator, seed: int):
    """Clone estimator and set random state."""
    try:
        est = clone(estimator)
    except Exception:
        est = copy.deepcopy(estimator)

    for param in ("random_state", "random_seed", "seed"):
        if hasattr(est, "set_params"):
            try:
                est.set_params(**{param: seed})
            except (ValueError, TypeError):
                pass
        elif hasattr(est, param):
            try:
                setattr(est, param, seed)
            except Exception:
                pass
    return est


def _fit_estimator(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    *,
    require_sample_weight: bool = True,
):
    """
    Fit estimator with sample_weight.

    Raises TypeError if sample_weight is provided but estimator doesn't accept it
    (when require_sample_weight=True).
    """
    kwargs = {}

    if w is not None:
        kwargs["sample_weight"] = w

    try:
        estimator.fit(X, y, **kwargs)
        return
    except TypeError as exc:
        if "sample_weight" in str(exc) and "sample_weight" in kwargs:
            if require_sample_weight:
                raise TypeError(
                    "Estimator.fit() does not accept sample_weight, "
                    "but sample_weight was provided. Use an estimator that "
                    "supports sample_weight or set sample_weight=None."
                ) from exc
            kwargs.pop("sample_weight", None)
            estimator.fit(X, y, **kwargs)
            return
        raise


def _get_native_importance(estimator) -> np.ndarray:
    """Get feature importance from fitted estimator."""
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=np.float64)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.max(np.abs(coef), axis=0)

    raise TypeError(
        "Estimator must have feature_importances_ or coef_. "
        "For Boruta, use tree-based models."
    )


def _weighted_mean_abs(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean of absolute values, handling 2D and 3D SHAP arrays."""
    if values.ndim == 2:
        abs_vals = np.abs(values)
        return (abs_vals * w[:, None]).sum(axis=0) / w.sum()
    if values.ndim == 3:
        abs_vals = np.abs(values).mean(axis=1)
        return (abs_vals * w[:, None]).sum(axis=0) / w.sum()
    raise ValueError(f"Unexpected SHAP array shape: {values.shape}")


def _catboost_shap_importance(
    model,
    X: np.ndarray,
    y: np.ndarray | None,
    w: np.ndarray,
) -> np.ndarray:
    """Get SHAP importance using CatBoost's native implementation."""
    from catboost import Pool

    pool = Pool(X, label=y, weight=w)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    shap_vals = np.asarray(shap_vals)

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[:, :-1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, :-1]
    else:
        raise ValueError(f"Unexpected CatBoost SHAP shape: {shap_vals.shape}")

    return _weighted_mean_abs(shap_vals, w)


def _shap_importance(
    estimator,
    X: np.ndarray,
    y: np.ndarray | None,
    w: np.ndarray,
    *,
    shap_sample_size: int | None,
    random_state: int,
) -> np.ndarray:
    """
    Compute SHAP-based feature importance.

    Uses CatBoost native SHAP if available, otherwise falls back to shap package.
    """
    if "catboost" in str(type(estimator)).lower():
        if shap_sample_size is not None and shap_sample_size < X.shape[0]:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(X.shape[0], size=shap_sample_size, replace=False)
            return _catboost_shap_importance(
                estimator,
                X[idx],
                y[idx] if y is not None else None,
                w[idx],
            )
        return _catboost_shap_importance(estimator, X, y, w)

    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP backend requires either:\n"
            "  - catboost (for native SHAP), OR\n"
            "  - shap package (pip install shap)"
        ) from exc

    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    if shap_sample_size is not None and shap_sample_size < n:
        idx = rng.choice(n, size=shap_sample_size, replace=False)
        X_eval = X[idx]
        w_eval = w[idx]
    else:
        X_eval = X
        w_eval = w

    explainer = shap.TreeExplainer(estimator)
    shap_vals = explainer.shap_values(X_eval)

    if isinstance(shap_vals, list):
        arr = np.stack(shap_vals, axis=1)
    else:
        arr = np.asarray(shap_vals)

    return _weighted_mean_abs(arr, w_eval)


def _impute_nonfinite_inplace(X: np.ndarray) -> None:
    """Replace non-finite values (NaN, inf, -inf) with column means."""
    mask = ~np.isfinite(X)
    if not mask.any():
        return
    X[mask] = np.nan
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(X)
    X[nan_mask] = col_means[np.where(nan_mask)[1]]


def _group_time_holdout_split(
    groups: np.ndarray,
    time: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split data by taking the last `test_size` fraction of each group's timeline.

    Returns (train_indices, test_indices).
    """
    train_idx = []
    test_idx = []

    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(time[idx], kind="mergesort")]
        n = len(idx)
        if n <= 1:
            train_idx.append(idx)
            continue
        n_test = max(1, int(np.ceil(n * test_size)))
        n_test = min(n_test, n - 1)
        train_idx.append(idx[:-n_test])
        test_idx.append(idx[-n_test:])

    train = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    test = np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64)
    return train, test


def _poisson_binom_pmf(ps: np.ndarray) -> np.ndarray:
    """
    Poisson-binomial PMF for sum of independent Bernoullis with probabilities ps.

    Returns pmf[k] = P(S = k) for k=0..len(ps)
    """
    ps = np.asarray(ps, dtype=np.float64).reshape(-1)
    pmf = np.zeros(ps.size + 1, dtype=np.float64)
    pmf[0] = 1.0
    for p in ps:
        prev = pmf.copy()
        pmf = prev * (1.0 - p)
        pmf[1:] += prev[:-1] * p
    return pmf


def _tail_pvals_from_pmf(pmf: np.ndarray, h: int) -> tuple[float, float]:
    """
    Returns (p_hi, p_lo) where:
      p_hi = P(S >= h)
      p_lo = P(S <= h)
    """
    if h < 0:
        return 1.0, 0.0
    if h >= pmf.size:
        return 0.0, 1.0
    cdf = np.cumsum(pmf)
    p_lo = float(cdf[h])
    p_hi = 1.0 if h <= 0 else float(1.0 - cdf[h - 1])
    return p_hi, p_lo


def _time_holdout_indices(
    time: np.ndarray,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Global forward holdout when groups not provided."""
    time = np.asarray(time).reshape(-1)
    n = time.shape[0]
    order = np.argsort(time, kind="mergesort")
    n_eval = max(1, min(int(np.ceil(n * test_size)), n - 1)) if n > 1 else 0
    if n_eval > 0:
        return order[:-n_eval].astype(np.int64), order[-n_eval:].astype(np.int64)
    return order.astype(np.int64), np.array([], dtype=np.int64)


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class BorutaResult:
    """Results from Boruta feature selection."""

    feature_names: list[str]
    status: np.ndarray
    hits: np.ndarray
    n_iter: int
    shadow_thresholds: np.ndarray
    mean_importance: np.ndarray

    @property
    def accepted_mask(self) -> np.ndarray:
        return self.status == 1

    @property
    def rejected_mask(self) -> np.ndarray:
        return self.status == -1

    @property
    def tentative_mask(self) -> np.ndarray:
        return self.status == 0

    def selected_features(self) -> list[str]:
        return [self.feature_names[i] for i in np.where(self.accepted_mask)[0]]

    def get_feature_ranking(self) -> pd.DataFrame:
        """Return features ranked by mean importance with status."""
        status_map = {-1: "rejected", 0: "tentative", 1: "accepted"}
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_importance": self.mean_importance,
                "hits": self.hits,
                "status": [status_map[int(s)] for s in self.status],
            }
        )
        return df.sort_values("mean_importance", ascending=False, na_position="last")


# =============================================================================
# Main Selector Class
# =============================================================================


class BorutaSelector(BaseEstimator, TransformerMixin):
    """
    Boruta / Boruta-Shap feature selector.

    Parameters
    ----------
    estimator : estimator object, optional
        Base estimator. If None, uses RandomForest for native importance
        or CatBoost for SHAP importance.
    task : {"regression", "classification"}
        Problem type.
    importance : {"native", "shap"}
        Importance backend. "native" uses feature_importances_,
        "shap" uses SHAP values.
    max_iter : int
        Maximum Boruta iterations.
    alpha : float
        Significance level for accept/reject decisions.
    perc : int
        Percentile for shadow threshold (100 = max shadow).
    resolve_tentative : bool
        If True, resolve tentative features at end using median comparison.
    max_features : int, optional
        Cap number of selected features.
    shadow_method : {"auto", "global", "within_group", "block", "circular_shift"}
        Shadow feature permutation method. "auto" selects based on
        groups/time availability.
    block_size : int or "auto"
        Block size for block permutation.
    importance_data : {"train", "test"}
        Compute importance on training data or held-out test split.
    test_size : float
        Test split size when importance_data="test".
    shap_sample_size : int, optional
        Subsample size for SHAP computation (faster for large datasets).
    early_stop_rounds : int
        Stop if no decisions made for this many consecutive rounds.
    random_state : int
        Random seed.
    verbose : bool
        Print progress.

    Attributes
    ----------
    feature_names_in_ : list[str]
        Feature names from fit.
    status_ : ndarray
        Feature status: -1=rejected, 0=tentative, 1=accepted.
    selected_features_ : list[str]
        Names of accepted features.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        estimator=None,
        *,
        task: Task = "regression",
        importance: ImportanceBackend = "native",
        max_iter: int = 50,
        alpha: float = 0.05,
        perc: int = 100,
        resolve_tentative: bool = True,
        max_features: int | None = None,
        shadow_method: PermutationMethod = "auto",
        shadow_mode: PermutationAxis = "columns",
        block_size: int | str = "auto",
        cat_features: list[str] | None = None,
        cat_encoding: CatEncoding = "loo",
        importance_data: Literal["train", "test"] = "train",
        test_size: float = 0.3,
        shap_sample_size: int | None = 2000,
        early_stop_rounds: int = 5,
        random_state: int = 0,
        verbose: bool = True,
    ):
        self.estimator = estimator
        self.task = task
        self.importance = importance
        self.max_iter = max_iter
        self.alpha = alpha
        self.perc = perc
        self.resolve_tentative = resolve_tentative
        self.max_features = max_features
        self.shadow_method = shadow_method
        self.shadow_mode = shadow_mode
        self.block_size = block_size
        self.cat_features = cat_features
        self.cat_encoding = cat_encoding
        self.importance_data = importance_data
        self.test_size = test_size
        self.shap_sample_size = shap_sample_size
        self.early_stop_rounds = early_stop_rounds
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X,
        y,
        *,
        sample_weight: np.ndarray | None = None,
        groups: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ):
        """
        Fit Boruta selector.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional
        groups : array-like of shape (n_samples,), optional
            Group labels for time-series shadow permutation.
        time : array-like of shape (n_samples,), optional
            Time values for ordering within groups.
        """
        feature_names = extract_feature_names(X)
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            cat_features = self.cat_features
            if cat_features is None:
                cat_features = X.select_dtypes(
                    include=["object", "category", "string"]
                ).columns.tolist()

            if cat_features and self.cat_encoding != "none":
                if self.task == "classification":
                    y_for_encoder = pd.Series(
                        np.unique(y, return_inverse=True)[1], index=X.index
                    )
                else:
                    y_for_encoder = y
                X = encode_categoricals(X, y_for_encoder, cat_features, self.cat_encoding)
                feature_names = extract_feature_names(X)

            non_numeric = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
            if non_numeric:
                sample = non_numeric[:5]
                suffix = "..." if len(non_numeric) > 5 else ""
                raise ValueError(
                    f"Non-numeric columns found: {sample}{suffix}. "
                    "Encode categorical columns before using Boruta, or use "
                    "cat_encoding in other sift methods."
                )
        X_arr = to_numpy(X, dtype=np.float64)
        y_arr = np.asarray(y).reshape(-1)

        n, p = X_arr.shape
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X has {n} rows but y has {y_arr.shape[0]}")

        w_score = ensure_weights(sample_weight, n, normalize=True)
        w_fit = w_score if sample_weight is not None else None

        if groups is not None:
            groups = np.asarray(groups).reshape(-1)
            if groups.shape[0] != n:
                raise ValueError(
                    f"groups has {groups.shape[0]} elements but X has {n} rows"
                )
        if time is not None:
            time = np.asarray(time).reshape(-1)
            if time.shape[0] != n:
                raise ValueError(f"time has {time.shape[0]} elements but X has {n} rows")

        shadow_method = resolve_permutation_method(
            self.shadow_method, groups=groups, time=time
        )
        if shadow_method in ("within_group", "block", "circular_shift") and groups is None:
            raise ValueError(f"shadow_method='{shadow_method}' requires groups")
        if shadow_method in ("block", "circular_shift") and time is None:
            raise ValueError(f"shadow_method='{shadow_method}' requires time for ordering")

        X_arr = X_arr.copy()
        _impute_nonfinite_inplace(X_arr)

        base_est = self._get_default_estimator()

        rng = np.random.default_rng(self.random_state)

        status = np.zeros(p, dtype=np.int8)
        hits = np.zeros(p, dtype=np.int32)
        n_trials = 0
        no_progress_count = 0

        imp_sum = np.zeros(p, dtype=np.float64)
        imp_count = np.zeros(p, dtype=np.int32)
        shadow_thresholds = []
        p_trials: list[float] = []

        if self.verbose:
            print(
                "Boruta: p={} importance={} shadow={} mode={} max_iter={}".format(
                    p, self.importance, shadow_method, self.shadow_mode, self.max_iter
                )
            )

        group_info = None
        if shadow_method in ("within_group", "block", "circular_shift"):
            group_info = build_group_info(groups, time)

        for it in range(self.max_iter):
            tentative_idx = np.where(status == 0)[0]
            if tentative_idx.size == 0:
                if self.verbose:
                    print(f"  iter={it + 1}: all features decided, stopping")
                break

            active_idx = np.where(status != -1)[0]
            X_active = X_arr[:, active_idx]
            n_active = X_active.shape[1]

            seed = int(rng.integers(0, 2**31 - 1))
            est = _clone_estimator(base_est, seed=seed)

            shadow = permute_matrix(
                X_active,
                method=shadow_method,
                group_info=group_info,
                block_size=self.block_size,
                seed=seed,
                axis=self.shadow_mode,
            )
            X_ext = np.concatenate([X_active, shadow], axis=1)

            imp = self._compute_importance(
                est,
                X_ext,
                y_arr,
                w_score,
                w_fit=w_fit,
                groups=groups,
                time=time,
                seed=seed,
            )

            if imp.shape[0] != X_ext.shape[1]:
                raise RuntimeError(
                    f"Importance length {imp.shape[0]} != expected {X_ext.shape[1]}"
                )

            imp_active = imp[:n_active]
            imp_shadow = imp[n_active:]

            thr = float(np.percentile(imp_shadow, self.perc))
            shadow_thresholds.append(thr)
            if self.perc < 100:
                p_null = float(np.mean(imp_shadow > thr))
                p_null = max(min(p_null, 1.0 - 1e-12), 1e-12)
            else:
                p_null = 1.0 / (len(imp_shadow) + 1.0)
            p_trials.append(p_null)

            for i_local in range(n_active):
                j = active_idx[i_local]
                if status[j] == 0:
                    if imp_active[i_local] > thr:
                        hits[j] += 1
                imp_sum[j] += float(imp_active[i_local])
                imp_count[j] += 1

            n_trials += 1

            pmf = _poisson_binom_pmf(np.asarray(p_trials, dtype=np.float64))

            tent = np.where(status == 0)[0]
            m = max(1, tent.size)
            alpha_adj = self.alpha / m

            decided_this_round = 0
            for j in tent:
                h = int(hits[j])
                p_hi, p_lo = _tail_pvals_from_pmf(pmf, h)
                if p_hi < alpha_adj:
                    status[j] = 1
                    decided_this_round += 1
                elif p_lo < alpha_adj:
                    status[j] = -1
                    decided_this_round += 1

            if self.verbose:
                n_acc = int((status == 1).sum())
                n_rej = int((status == -1).sum())
                n_ten = int((status == 0).sum())
                print(
                    "  iter={:02d} thr={:.4f} acc={} rej={} tent={}".format(
                        it + 1, thr, n_acc, n_rej, n_ten
                    )
                )

            if decided_this_round == 0:
                no_progress_count += 1
                if no_progress_count >= self.early_stop_rounds:
                    if self.verbose:
                        print(
                            "  Early stop: no decisions for {} rounds".format(
                                no_progress_count
                            )
                        )
                    break
            else:
                no_progress_count = 0

        shadow_thresholds_arr = np.asarray(shadow_thresholds, dtype=np.float64)
        mean_importance = np.full(p, np.nan, dtype=np.float64)
        ok = imp_count > 0
        mean_importance[ok] = imp_sum[ok] / imp_count[ok]

        if self.resolve_tentative and (status == 0).any() and shadow_thresholds_arr.size > 0:
            med_thr = float(np.median(shadow_thresholds_arr))
            for j in np.where(status == 0)[0]:
                if not np.isfinite(mean_importance[j]):
                    status[j] = -1
                else:
                    status[j] = 1 if mean_importance[j] > med_thr else -1

        if self.max_features is not None:
            acc = np.where(status == 1)[0]
            if acc.size > self.max_features:
                order = acc[np.argsort(-mean_importance[acc])]
                keep = set(order[: self.max_features].tolist())
                for j in acc:
                    if int(j) not in keep:
                        status[j] = -1

        self.feature_names_in_ = feature_names
        self.status_ = status
        self.hits_ = hits
        self.n_iter_ = int(n_trials)
        self.shadow_thresholds_ = shadow_thresholds_arr
        self.mean_importance_ = mean_importance
        self.selected_features_ = [feature_names[i] for i in np.where(status == 1)[0]]

        return self

    def _get_default_estimator(self):
        """Get default estimator based on importance backend and task."""
        if self.estimator is not None:
            return self.estimator

        if self.importance == "native":
            if self.task == "regression":
                return RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=self.random_state,
                )
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                random_state=self.random_state,
            )

        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            if self.task == "regression":
                return CatBoostRegressor(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=self.random_state,
                )
            return CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=False,
                random_seed=self.random_state,
            )
        except ImportError as exc:
            raise ValueError(
                "importance='shap' requires catboost or an explicit estimator"
            ) from exc

    def _compute_importance(
        self,
        est,
        X: np.ndarray,
        y: np.ndarray,
        w_score: np.ndarray,
        *,
        w_fit: np.ndarray | None,
        groups: np.ndarray | None,
        time: np.ndarray | None,
        seed: int,
    ) -> np.ndarray:
        """Fit estimator and compute importance."""
        if self.importance_data == "test":
            if groups is not None and time is not None:
                train_idx, test_idx = _group_time_holdout_split(
                    groups, time, self.test_size
                )
            elif time is not None:
                train_idx, test_idx = _time_holdout_indices(time, self.test_size)
            else:
                stratify = y if self.task == "classification" else None
                train_idx, test_idx = train_test_split(
                    np.arange(len(y)),
                    test_size=self.test_size,
                    random_state=seed,
                    stratify=stratify,
                )
                train_idx = np.asarray(train_idx)
                test_idx = np.asarray(test_idx)

            if len(test_idx) == 0:
                train_idx = np.arange(len(y))
                test_idx = train_idx

            X_train, X_eval = X[train_idx], X[test_idx]
            y_train, y_eval = y[train_idx], y[test_idx]
            w_fit_train = w_fit[train_idx] if w_fit is not None else None
            _fit_estimator(
                est, X_train, y_train, w_fit_train, require_sample_weight=True
            )
            X_imp, y_imp, w_imp = X_eval, y_eval, w_score[test_idx]
        else:
            _fit_estimator(est, X, y, w_fit, require_sample_weight=True)
            X_imp, y_imp, w_imp = X, y, w_score

        if self.importance == "native":
            return _get_native_importance(est)

        return _shap_importance(
            est,
            X_imp,
            y_imp,
            w_imp,
            shap_sample_size=self.shap_sample_size,
            random_state=seed,
        )

    def get_support(self, indices: bool = False) -> np.ndarray:
        check_is_fitted(self, ["status_"])
        mask = self.status_ == 1
        return np.where(mask)[0] if indices else mask

    def transform(self, X):
        check_is_fitted(self, ["status_"])
        keep_idx = self.get_support(indices=True)
        if isinstance(X, pd.DataFrame):
            cols = [self.feature_names_in_[i] for i in keep_idx]
            return X.loc[:, cols]
        return np.asarray(X)[:, keep_idx]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def result_(self) -> BorutaResult:
        check_is_fitted(self, ["status_"])
        return BorutaResult(
            feature_names=list(self.feature_names_in_),
            status=self.status_.copy(),
            hits=self.hits_.copy(),
            n_iter=int(self.n_iter_),
            shadow_thresholds=self.shadow_thresholds_.copy(),
            mean_importance=self.mean_importance_.copy(),
        )


# =============================================================================
# Functional API
# =============================================================================


def select_boruta(
    X,
    y,
    *,
    task: Task = "regression",
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    estimator=None,
    importance: ImportanceBackend = "native",
    max_iter: int = 50,
    alpha: float = 0.05,
    perc: int = 100,
    resolve_tentative: bool = True,
    max_features: int | None = None,
    shadow_method: PermutationMethod = "auto",
    shadow_mode: PermutationAxis = "columns",
    block_size: int | str = "auto",
    cat_features: list[str] | None = None,
    cat_encoding: CatEncoding = "loo",
    importance_data: Literal["train", "test"] = "train",
    test_size: float = 0.3,
    shap_sample_size: int | None = 2000,
    early_stop_rounds: int = 5,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> list[str] | BorutaResult:
    """
    Boruta feature selection.

    Parameters
    ----------
    X : DataFrame or ndarray
    y : array-like
    task : {"regression", "classification"}
    sample_weight : array-like, optional
    groups : array-like, optional
        Group labels for shadow permutation.
    time : array-like, optional
        Time values for ordering.
    group_col : str, optional
        Column name in X to use as groups (extracted and dropped from X).
    time_col : str, optional
        Column name in X to use as time (extracted and dropped from X).
    estimator : estimator object, optional
    importance : {"native", "shap"}
    max_iter : int
    alpha : float
    perc : int
    resolve_tentative : bool
    max_features : int, optional
    shadow_method : str
    block_size : int or "auto"
    importance_data : {"train", "test"}
    test_size : float
    shap_sample_size : int, optional
    early_stop_rounds : int
    random_state : int
    verbose : bool
    return_result : bool
        If True, return BorutaResult instead of feature list.

    Returns
    -------
    list[str] or BorutaResult
    """
    if isinstance(X, pd.DataFrame):
        X = X.copy()
        if group_col is not None:
            if groups is not None:
                raise ValueError("Cannot specify both groups and group_col")
            groups = X[group_col].values
            X = X.drop(columns=[group_col])
        if time_col is not None:
            if time is not None:
                raise ValueError("Cannot specify both time and time_col")
            time = X[time_col].values
            X = X.drop(columns=[time_col])

    sel = BorutaSelector(
        estimator=estimator,
        task=task,
        importance=importance,
        max_iter=max_iter,
        alpha=alpha,
        perc=perc,
        resolve_tentative=resolve_tentative,
        max_features=max_features,
        shadow_method=shadow_method,
        shadow_mode=shadow_mode,
        block_size=block_size,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        importance_data=importance_data,
        test_size=test_size,
        shap_sample_size=shap_sample_size,
        early_stop_rounds=early_stop_rounds,
        random_state=random_state,
        verbose=verbose,
    )
    sel.fit(X, y, sample_weight=sample_weight, groups=groups, time=time)

    if return_result:
        return sel.result_()
    return sel.selected_features_


def select_boruta_shap(
    X,
    y,
    *,
    task: Task = "regression",
    sample_weight: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    time: np.ndarray | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    estimator=None,
    max_iter: int = 50,
    alpha: float = 0.05,
    perc: int = 100,
    resolve_tentative: bool = True,
    max_features: int | None = None,
    shadow_method: PermutationMethod = "auto",
    shadow_mode: PermutationAxis = "columns",
    block_size: int | str = "auto",
    cat_features: list[str] | None = None,
    cat_encoding: CatEncoding = "loo",
    importance_data: Literal["train", "test"] = "train",
    test_size: float = 0.3,
    shap_sample_size: int | None = 2000,
    early_stop_rounds: int = 5,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
) -> list[str] | BorutaResult:
    """Boruta-Shap feature selection (convenience wrapper for importance='shap')."""
    return select_boruta(
        X,
        y,
        task=task,
        sample_weight=sample_weight,
        groups=groups,
        time=time,
        group_col=group_col,
        time_col=time_col,
        estimator=estimator,
        importance="shap",
        max_iter=max_iter,
        alpha=alpha,
        perc=perc,
        resolve_tentative=resolve_tentative,
        max_features=max_features,
        shadow_method=shadow_method,
        shadow_mode=shadow_mode,
        block_size=block_size,
        cat_features=cat_features,
        cat_encoding=cat_encoding,
        importance_data=importance_data,
        test_size=test_size,
        shap_sample_size=shap_sample_size,
        early_stop_rounds=early_stop_rounds,
        random_state=random_state,
        verbose=verbose,
        return_result=return_result,
    )


__all__ = [
    "BorutaSelector",
    "BorutaResult",
    "select_boruta",
    "select_boruta_shap",
]
