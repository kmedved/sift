"""
Boruta and Boruta-Shap feature selection.

Design:
- Single Boruta loop with pluggable importance backend
- Time-series aware shadow permutations
- Sample weight support throughout
- Optional train/test split for importance computation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from sift._logging import logger
from sift._metadata import drop_fitted_metadata_columns, resolve_row_metadata
from sift._progress import ProgressCallback, report_progress
from sift._selector_compat import (
    feature_names_array,
    inverse_selected_matrix,
    ordered_indices,
    reject_sparse,
    selector_tags,
    validate_fit_matrix,
    validate_output_order,
)
from sift._permute import (
    PermutationAxis,
    PermutationMethod,
    build_group_info,
    permute_matrix,
    resolve_permutation_method,
)
from sift._preprocess import (
    CatEncoding,
    LeaveOneOutLogitEncoder,
    TargetCVEncoder,
    ensure_weights,
    extract_feature_names,
    reject_datetime_like_features,
    suppress_category_encoder_pandas_warnings,
    to_numpy,
    validate_target_cv_encoding_flags,
)

from sift.boruta_helpers import (
    ImportanceBackend,
    Task,
    _clone_estimator,
    _compute_auto_n_estimators,
    _fit_estimator,
    _get_estimator_depth,
    _get_native_importance,
    _group_time_holdout_split,
    _impute_nonfinite_inplace,
    _poisson_binom_pmf,
    _set_n_estimators,
    _shap_importance,
    _tail_pvals_from_pmf,
    _time_holdout_indices,
    _validate_boruta_options,
)


# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class BorutaResult:
    """Results from Boruta feature selection.

    A plain snapshot of one finished Boruta run: the accept/reject/tentative
    verdict for every candidate feature, the hit counts and shadow thresholds
    the verdicts came from, and the mean importance used to rank them. It is
    returned by :func:`select_boruta` and :func:`select_boruta_shap` under
    ``return_result=True``, and by ``BorutaSelector.result_()`` after ``fit``,
    which copies its arrays so later refits cannot mutate an earlier result.
    Every array is positional and aligned with ``feature_names``, so ordinary
    NumPy masking works directly; the convenience masks and
    :meth:`get_feature_ranking` are built on exactly that alignment.

    Unlike the fixed-``k`` filters, Boruta is an all-relevant selector: it
    keeps every feature that beats its shadow, so the accepted count is a
    property of the data rather than a request.

    Attributes
    ----------
    feature_names : list of str
        Candidate feature labels in fitted column order. A positional ndarray
        fit stores the generated ``x0...`` names here.
    status : ndarray of shape (n_features,)
        Per-feature verdict: ``1`` accepted, ``0`` tentative, ``-1`` rejected.
        With ``resolve_tentative=True`` the surviving tentatives are forced to
        ``1`` or ``-1`` against the median shadow threshold, and any
        ``max_features`` cap demotes the weakest accepted features to ``-1``.
    hits : ndarray of shape (n_features,)
        Number of iterations in which a feature beat that iteration's shadow
        threshold while it was still undecided. Counting stops once a feature
        leaves tentative status, so ``hits`` is not comparable across features
        decided in different rounds.
    n_iter : int
        Number of Boruta iterations actually run, which may be below
        ``max_iter`` after an early stop.
    shadow_thresholds : ndarray of shape (n_iter,)
        The ``perc``-percentile shadow importance recorded once per iteration.
    mean_importance : ndarray of shape (n_features,)
        Mean observed importance per feature over the iterations in which it
        was still active; ``NaN`` for a feature that never received one.

    See Also
    --------
    select_boruta : Function-style Boruta returning this result on request.
    BorutaSelector : The estimator whose ``result_()`` builds it.
    sift.as_result : Normalize this result into a ``SelectionView``; also
        reachable as ``result.result_view()``.

    Notes
    -----
    ``accepted_mask``, ``rejected_mask`` and ``tentative_mask`` are boolean
    views of ``status``; ``selected_features()`` returns the accepted labels in
    input-column order, while ``get_feature_ranking()`` returns a DataFrame of
    ``feature``, ``mean_importance``, ``hits`` and a textual ``status``, sorted
    by descending mean importance with ``NaN`` last and stable ties. The
    dataclass is mutable and holds the arrays it was given, so treat a result
    obtained from ``BorutaSelector.result_()`` as the owner of its copies.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import select_boruta
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 5))
    >>> y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * rng.normal(size=120)
    >>> result = select_boruta(X, y, max_iter=10, n_estimators=50,
    ...                        random_state=0, verbose=False,
    ...                        return_result=True)
    >>> result.selected_features()
    ['x0', 'x1']
    >>> sorted(set(result.status.tolist()))
    [-1, 1]
    >>> list(result.get_feature_ranking().columns)
    ['feature', 'mean_importance', 'hits', 'status']
    """

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
        return df.sort_values(
            "mean_importance",
            ascending=False,
            na_position="last",
            kind="mergesort",
        )

    def result_view(self, input_features=None):
        """Return an additive normalized view without changing this result."""
        from sift.selection.view import as_result

        return as_result(self, input_features=input_features)


@dataclass(frozen=True)
class BorutaFitData:
    X_arr: np.ndarray
    y_arr: np.ndarray
    w_score: np.ndarray
    w_fit: np.ndarray | None
    groups: np.ndarray | None
    time: np.ndarray | None
    shadow_method: PermutationMethod
    base_estimator: object
    base_depth: int | None
    feature_names: list[str]


@dataclass(frozen=True)
class BorutaLoopResult:
    status: np.ndarray
    hits: np.ndarray
    n_trials: int
    shadow_thresholds: np.ndarray
    mean_importance: np.ndarray


# =============================================================================
# Main Selector Class
# =============================================================================


class BorutaSelector(SelectorMixin, BaseEstimator):
    """
    Boruta / Boruta-Shap feature selector.

    Parameters
    ----------
    estimator : estimator object, optional
        Base estimator. If None, uses RandomForest for native importance
        or CatBoost for SHAP importance.
    n_estimators : int or "auto"
        Number of trees/iterations for the estimator. When "auto", compute
        a fast bounded heuristic based on active features and depth. Auto
        only applies when estimator is None.
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
    shadow_mode : {"columns", "rows"}, default="columns"
        Permutation axis for the shadows. "columns" permutes each column
        independently (classic Boruta); "rows" applies one row permutation to
        every column, preserving cross-feature covariance in the null.
    block_size : int or "auto"
        Block size for block permutation.
    cat_features : list of str, optional
        Categorical column names to encode. None auto-detects object,
        category and string DataFrame columns; names absent from X are
        dropped. Only used when cat_encoding is not "none".
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", "loo_logit"}
        Categorical encoding fitted before the Boruta loop. "target_cv" is the
        leakage-safe cross-fitted encoder and is the only supervised value
        accepted without an explicit opt-in; the four legacy encodings fit on
        the full dataset and therefore require
        allow_full_data_target_encoding=True, because tree learners can read a
        row's own target back out of them. Any supervised value is rejected
        with importance_data="test". sample_weight is consumed only by
        "target_cv" and "loo_logit".
    target_cv_n_splits : int, default=5
        Fold count for cat_encoding="target_cv".
    target_cv_smoothing : "auto" or float, default="auto"
        Empirical-Bayes shrinkage for "target_cv"; an explicit non-negative
        float is always accepted.
    target_prior : float, optional
        Target-independent prior for time-aware "target_cv" fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        "target_cv" fit when no target_prior is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to the leakage-prone full-dataset supervised encodings. It is
        rejected together with cat_encoding="target_cv", whose cross-fitted
        contract it contradicts.
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
        Emit per-iteration progress at INFO on the ``sift`` logger.
    callback : callable, optional
        Called after each completed Boruta iteration as
        ``callback(step, total, info)``.
    output_order : {'legacy', 'original'}, default='legacy'
        Order used by transform, selected support indices, feature names, and
        inverse transform. Boruta's legacy order is already input order.

    Attributes
    ----------
    feature_names_in_ : ndarray of shape (n_features,)
        One-dimensional object array of fitted feature names. Positional fits
        keep the generated ``x0...`` names in this attribute.
    status_ : ndarray
        Feature status: -1=rejected, 0=tentative, 1=accepted.
    selected_features_ : list[str]
        Names of accepted features.
    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    select_boruta : One-call wrapper around this estimator.
    select_boruta_shap : The same search with the SHAP importance backend.
    BorutaResult : Result object returned by ``select_boruta(return_result=True)``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sift import BorutaSelector
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.normal(size=(120, 5)),
    ...                  columns=[f"f{i}" for i in range(5)])
    >>> y = 3.0 * X["f0"] + 2.0 * X["f1"] + 0.1 * rng.normal(size=120)
    >>> selector = BorutaSelector(
    ...     estimator=RandomForestRegressor(n_estimators=20, random_state=0),
    ...     max_iter=10, random_state=0, verbose=False,
    ... )
    >>> selector.fit(X, y).selected_features_
    ['f0', 'f1']
    >>> selector.transform(X).shape
    (120, 2)
    """

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators: int | str = "auto",
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
        cat_encoding: CatEncoding = "none",
        target_cv_n_splits: int = 5,
        target_cv_smoothing: Literal["auto"] | float = "auto",
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
        allow_full_data_target_encoding: bool = False,
        importance_data: Literal["train", "test"] = "train",
        test_size: float = 0.3,
        shap_sample_size: int | None = 2000,
        early_stop_rounds: int = 5,
        random_state: int = 0,
        verbose: bool = True,
        callback: ProgressCallback | None = None,
        output_order: str = "legacy",
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
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
        self.target_cv_n_splits = target_cv_n_splits
        self.target_cv_smoothing = target_cv_smoothing
        self.target_prior = target_prior
        self.warmup_policy = warmup_policy
        self.allow_full_data_target_encoding = allow_full_data_target_encoding
        self.importance_data = importance_data
        self.test_size = test_size
        self.shap_sample_size = shap_sample_size
        self.early_stop_rounds = early_stop_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.output_order = output_order
        self.callback = callback

    def _get_default_estimator(self, y: np.ndarray | None = None):
        """Get default estimator based on importance backend and task."""
        if self.estimator is not None:
            return self.estimator

        if self.importance == "native":
            if self.task == "regression":
                return RandomForestRegressor(
                    n_estimators=500,
                    max_depth=5,
                    n_jobs=-1,
                    random_state=self.random_state,
                )
            return RandomForestClassifier(
                n_estimators=500,
                max_depth=5,
                n_jobs=-1,
                random_state=self.random_state,
            )

        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            if self.task == "regression":
                return CatBoostRegressor(
                    iterations=500,
                    depth=5,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    verbose=False,
                    random_seed=self.random_state,
                    allow_writing_files=False,
                )
            loss_function = "Logloss"
            if y is not None:
                n_classes = np.unique(np.asarray(y).reshape(-1)).size
                if n_classes >= 3:
                    loss_function = "MultiClass"
            return CatBoostClassifier(
                iterations=500,
                depth=5,
                learning_rate=0.05,
                loss_function=loss_function,
                verbose=False,
                random_seed=self.random_state,
                allow_writing_files=False,
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
        shadow_method: PermutationMethod,
        shadow_mode: PermutationAxis,
        block_size: int | str,
        group_info: dict | None = None,
    ) -> np.ndarray:
        """Fit estimator and compute importance."""
        def make_shadow_group_info(
            X_part: np.ndarray,
            *,
            groups_part: np.ndarray | None,
            time_part: np.ndarray | None,
            group_info_part: dict | None,
        ):
            if shadow_method not in ("within_group", "block", "circular_shift"):
                return None
            if group_info_part is not None:
                return group_info_part
            return build_group_info(groups_part, time_part, n_samples=X_part.shape[0])

        def make_shadow_matrix(
            X_part: np.ndarray,
            *,
            groups_part: np.ndarray | None,
            time_part: np.ndarray | None,
            seed_part: int,
            group_info_part: dict | None = None,
        ) -> np.ndarray:
            group_info = make_shadow_group_info(
                X_part,
                groups_part=groups_part,
                time_part=time_part,
                group_info_part=group_info_part,
            )
            shadow = permute_matrix(
                X_part,
                method=shadow_method,
                groups=groups_part,
                time=time_part,
                group_info=group_info,
                block_size=block_size,
                seed=seed_part,
                axis=shadow_mode,
            )
            return np.concatenate([X_part, shadow], axis=1)

        if self.importance_data == "test":
            if groups is not None and time is not None:
                train_idx, test_idx = _group_time_holdout_split(
                    groups, time, self.test_size
                )
            elif groups is not None:
                if len(pd.unique(groups)) < 2:
                    raise ValueError(
                        "BorutaSelector(importance_data='test') requires at least "
                        "2 groups to create a held-out group split."
                    )
                else:
                    from sklearn.model_selection import GroupShuffleSplit

                    gss = GroupShuffleSplit(
                        n_splits=1,
                        test_size=self.test_size,
                        random_state=seed,
                    )
                    train_idx, test_idx = next(gss.split(X, y, groups))
                    train_idx = np.asarray(train_idx)
                    test_idx = np.asarray(test_idx)
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
                raise ValueError(
                    "BorutaSelector(importance_data='test') could not create a "
                    "non-empty held-out split. Use importance_data='train' or "
                    "provide more rows/groups."
                )
            if np.intersect1d(train_idx, test_idx).size:
                raise ValueError(
                    "BorutaSelector(importance_data='test') requires disjoint "
                    "train and held-out rows."
                )

            X_train = make_shadow_matrix(
                X[train_idx],
                groups_part=groups[train_idx] if groups is not None else None,
                time_part=time[train_idx] if time is not None else None,
                seed_part=seed,
            )
            X_eval = make_shadow_matrix(
                X[test_idx],
                groups_part=groups[test_idx] if groups is not None else None,
                time_part=time[test_idx] if time is not None else None,
                seed_part=seed + 1,
            )
            y_train, y_eval = y[train_idx], y[test_idx]
            w_fit_train = w_fit[train_idx] if w_fit is not None else None
            _fit_estimator(
                est, X_train, y_train, w_fit_train, require_sample_weight=True
            )
            X_imp, y_imp, w_imp = X_eval, y_eval, w_score[test_idx]
        else:
            X_ext = make_shadow_matrix(
                X,
                groups_part=groups,
                time_part=time,
                seed_part=seed,
                group_info_part=group_info,
            )
            _fit_estimator(est, X_ext, y, w_fit, require_sample_weight=True)
            X_imp, y_imp, w_imp = X_ext, y, w_score

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
        if indices:
            return ordered_indices(np.flatnonzero(self.status_ == 1), self.output_order)
        return self._get_support_mask()

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, ["status_"])
        return self.status_ == 1

    def _more_tags(self):
        parent = getattr(super(), "_more_tags", None)
        return selector_tags({} if parent is None else parent())

    def __sklearn_tags__(self):
        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:
            return self._more_tags()
        return selector_tags(parent())

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return accepted feature names following sklearn's transformer API."""
        check_is_fitted(self, ["status_", "feature_names_in_", "n_features_in_"])
        fitted_names = feature_names_array(self.feature_names_in_)
        if input_features is not None:
            input_names = np.asarray(input_features, dtype=object)
            if input_names.ndim != 1 or input_names.shape[0] != self.n_features_in_:
                raise ValueError(
                    "input_features must have the same number of features as the fitted data"
                )
            if not np.array_equal(input_names, fitted_names):
                raise ValueError("input_features is not equal to feature_names_in_")
        return fitted_names[self.get_support(indices=True)]

    def transform(self, X):
        check_is_fitted(self, ["status_"])
        reject_sparse(X, operation="transform")
        X = drop_fitted_metadata_columns(
            X,
            getattr(self, "_row_metadata_columns_", ()),
        )
        if getattr(self, "_categorical_encoding_applied_", False):
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    "This BorutaSelector was fitted with DataFrame categorical "
                    "encoding; transform also requires a DataFrame."
                )
            with suppress_category_encoder_pandas_warnings():
                X = self.categorical_encoder_.transform(X)
        keep_idx = self.get_support(indices=True)
        if isinstance(X, pd.DataFrame):
            cols = [self.feature_names_in_[i] for i in keep_idx]
            return X.loc[:, cols]
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                "X must be a 2D feature matrix. Reshape your data with "
                "X.reshape(-1, 1) for a single feature."
            )
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but BorutaSelector was fitted "
                f"with {self.n_features_in_} features"
            )
        return X_arr[:, keep_idx]

    def inverse_transform(self, X):
        """Restore accepted values to their fitted raw-column positions."""
        check_is_fitted(self, ["status_", "n_features_in_"])
        if getattr(self, "_categorical_encoding_applied_", False):
            raise NotImplementedError(
                "inverse_transform is unavailable after supervised categorical "
                "encoding because the fitted encoder is not invertible"
            )
        return inverse_selected_matrix(
            X,
            n_features=self.n_features_in_,
            selected_indices=self.get_support(indices=True),
        )

    def fit_transform(self, X, y=None, **fit_params):
        if self.cat_encoding != "target_cv":
            return self.fit(X, y, **fit_params).transform(X)
        self._capture_target_cv_training_output_ = True
        try:
            self.fit(X, y, **fit_params)
            captured = getattr(self, "_target_cv_training_output_", None)
            if captured is None:
                return self.transform(X)
            keep_idx = self.get_support(indices=True)
            if isinstance(captured, pd.DataFrame):
                return captured.iloc[:, keep_idx].copy()
            return np.asarray(captured)[:, keep_idx].copy()
        finally:
            for attr in (
                "_capture_target_cv_training_output_",
                "_target_cv_training_output_",
            ):
                if hasattr(self, attr):
                    delattr(self, attr)

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
        self._clear_fit_state()
        try:
            metadata = resolve_row_metadata(X, groups=groups, time=time)
            X = metadata.X
            validate_fit_matrix(X)
            validate_output_order(self.output_order)
            groups = metadata.groups
            time = metadata.time
            self._row_metadata_columns_ = metadata.extracted_columns
            fit_data = self._prepare_boruta_fit(X, y, sample_weight, groups, time)
            loop_result = self._run_boruta_iterations(fit_data)
            status = self._resolve_boruta_final_status(
                loop_result.status,
                loop_result.mean_importance,
                loop_result.shadow_thresholds,
            )
            self._store_boruta_attributes(
                fit_data.feature_names,
                status,
                loop_result.hits,
                loop_result.n_trials,
                loop_result.shadow_thresholds,
                loop_result.mean_importance,
            )
        except Exception:
            self._clear_fit_state()
            raise
        return self

    def _clear_fit_state(self) -> None:
        for attr in (
            "categorical_encoder_",
            "categorical_encoding_metadata_",
            "_target_cv_training_output_",
            "categorical_features_",
            "_categorical_encoding_applied_",
            "_fit_feature_names_generated_",
            "feature_names_in_",
            "n_features_in_",
            "status_",
            "hits_",
            "n_iter_",
            "shadow_thresholds_",
            "mean_importance_",
            "selected_features_",
            "_row_metadata_columns_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _prepare_boruta_fit(self, X, y, sample_weight, groups, time):
        validate_target_cv_encoding_flags(
            self.cat_encoding, self.allow_full_data_target_encoding
        )
        if self.importance_data == "test" and self.importance == "native":
            raise ValueError(
                "BorutaSelector(importance_data='test') is not supported with "
                "importance='native' because native importances are read from "
                "the fitted model and do not evaluate held-out rows. Use "
                "importance='shap' or another held-out-compatible importance "
                "backend if available."
            )

        y_arr = np.asarray(y).reshape(-1)
        _validate_boruta_options(
            task=self.task,
            importance=self.importance,
            importance_data=self.importance_data,
            shadow_method=self.shadow_method,
            shadow_mode=self.shadow_mode,
            block_size=self.block_size,
        )

        feature_names = extract_feature_names(X)
        reject_datetime_like_features(X)
        self.categorical_encoder_ = None
        self.categorical_features_ = []
        self._categorical_encoding_applied_ = False
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            cat_features = self.cat_features
            if cat_features is None:
                cat_features = X.select_dtypes(
                    include=["object", "category", "string"]
                ).columns.tolist()
            elif cat_features:
                cat_features = [c for c in cat_features if c in X.columns]

            if cat_features and self.cat_encoding != "none":
                if self.importance_data == "test":
                    raise ValueError(
                        "BorutaSelector(importance_data='test') cannot use supervised "
                        "cat_encoding on the full dataset. Pre-encode categoricals "
                        "leakage-safely or use importance_data='train'."
                    )
                if (
                    self.cat_encoding != "target_cv"
                    and not self.allow_full_data_target_encoding
                ):
                    raise ValueError(
                        f"cat_encoding={self.cat_encoding!r} fits a supervised categorical "
                        "encoder on the full dataset before Boruta. Tree learners can read "
                        "the row's own target back out of leave-one-out/target encodings, "
                        "so pure-noise high-cardinality categoricals get accepted. Pass "
                        "allow_full_data_target_encoding=True to opt into this "
                        "leakage-prone behavior, or set cat_encoding='none' and "
                        "pre-encode categoricals in a leakage-safe (out-of-fold) pipeline."
                    )
                if self.task == "classification":
                    y_for_encoder = pd.Series(
                        np.unique(y, return_inverse=True)[1], index=X.index
                    )
                else:
                    y_for_encoder = y
                if (
                    sample_weight is not None
                    and self.cat_encoding not in {"loo_logit", "target_cv"}
                ):
                    raise ValueError(
                        "sample_weight with Boruta categorical encoding is only "
                        "supported for cat_encoding='loo_logit'. category_encoders-backed "
                        "methods ('loo', 'target', 'james_stein') do not consume sample weights."
                    )
                if self.cat_encoding == "target_cv":
                    encoder = TargetCVEncoder(
                        cat_features,
                        target_type="binary"
                        if self.task == "classification"
                        else "continuous",
                        smooth=self.target_cv_smoothing,
                        cv=self.target_cv_n_splits,
                        target_prior=self.target_prior,
                        warmup_policy=self.warmup_policy,
                    )
                    X = encoder.fit_transform(
                        X,
                        y_for_encoder,
                        sample_weight=sample_weight,
                        groups=groups,
                        time=time,
                    )
                elif self.cat_encoding == "loo_logit":
                    encoder = LeaveOneOutLogitEncoder(cat_features)
                    X = encoder.fit_transform(X, y_for_encoder, sample_weight=sample_weight)
                else:
                    import category_encoders as ce

                    encoders = {
                        "loo": ce.LeaveOneOutEncoder,
                        "target": ce.TargetEncoder,
                        "james_stein": ce.JamesSteinEncoder,
                    }
                    if self.cat_encoding not in encoders:
                        raise ValueError(
                            "cat_encoding must be one of 'none', 'target_cv', "
                            "'target', 'loo', 'james_stein', or 'loo_logit'. "
                            f"Got {self.cat_encoding!r}."
                        )
                    Encoder = encoders[self.cat_encoding]
                    try:
                        encoder = Encoder(
                            cols=cat_features,
                            handle_missing="return_nan",
                            handle_unknown="value",
                        )
                    except TypeError:
                        encoder = Encoder(cols=cat_features, handle_missing="return_nan")
                    with suppress_category_encoder_pandas_warnings():
                        X = encoder.fit_transform(X, y_for_encoder)
                self.categorical_encoder_ = encoder
                if hasattr(encoder, "encoding_cv_"):
                    self.categorical_encoding_metadata_ = dict(encoder.encoding_cv_)
                self.categorical_features_ = list(cat_features)
                self._categorical_encoding_applied_ = True
                feature_names = extract_feature_names(X)
                if getattr(self, "_capture_target_cv_training_output_", False):
                    self._target_cv_training_output_ = X.copy()

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
        elif self.cat_features:
            raise ValueError("cat_features requires X to be a pandas DataFrame")
        X_arr = to_numpy(X, dtype=np.float64)

        n, p = X_arr.shape
        # Generated names stay in the public ``feature_names_in_`` attribute;
        # the private marker records that the fit was positional.
        self._fit_feature_names_generated_ = feature_names is None
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"X has {n} rows but y has {y_arr.shape[0]}")

        effective_sample_weight = getattr(
            self.categorical_encoder_, "effective_sample_weight_", None
        )
        if effective_sample_weight is not None:
            effective_sample_weight = np.asarray(effective_sample_weight, dtype=float)
        else:
            effective_sample_weight = sample_weight
        w_score = ensure_weights(effective_sample_weight, n, normalize=True)
        w_fit = w_score if effective_sample_weight is not None else None

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
        if shadow_method in ("block", "circular_shift") and time is None:
            raise ValueError(f"shadow_method='{shadow_method}' requires time")

        X_arr = X_arr.copy()
        _impute_nonfinite_inplace(X_arr)

        base_est = self._get_default_estimator(y_arr)
        if isinstance(self.n_estimators, str):
            if self.n_estimators != "auto":
                raise ValueError("n_estimators must be an int or 'auto'")
        else:
            try:
                n_est_int = int(self.n_estimators)
            except Exception as exc:
                raise ValueError("n_estimators must be an int or 'auto'") from exc
            if n_est_int < 1:
                raise ValueError("n_estimators must be >= 1")
        base_depth = _get_estimator_depth(base_est)
        return BorutaFitData(
            X_arr=X_arr,
            y_arr=y_arr,
            w_score=w_score,
            w_fit=w_fit,
            groups=groups,
            time=time,
            shadow_method=shadow_method,
            base_estimator=base_est,
            base_depth=base_depth,
            feature_names=feature_names,
        )

    def _run_boruta_iterations(self, fit_data: BorutaFitData) -> BorutaLoopResult:
        X_arr = fit_data.X_arr
        y_arr = fit_data.y_arr
        w_score = fit_data.w_score
        w_fit = fit_data.w_fit
        groups = fit_data.groups
        time = fit_data.time
        shadow_method = fit_data.shadow_method
        base_est = fit_data.base_estimator
        base_depth = fit_data.base_depth
        n, p = X_arr.shape
        rng = np.random.default_rng(self.random_state)

        status = np.zeros(p, dtype=np.int8)
        hits = np.zeros(p, dtype=np.int32)
        n_trials = 0
        no_progress_count = 0

        imp_sum = np.zeros(p, dtype=np.float64)
        imp_count = np.zeros(p, dtype=np.int32)
        shadow_thresholds = []
        group_info = None
        if (
            self.importance_data != "test"
            and shadow_method in ("within_group", "block", "circular_shift")
        ):
            group_info = build_group_info(groups, time, n_samples=n)

        if self.verbose:
            logger.info(
                "Boruta: p={} importance={} shadow={} mode={} max_iter={}".format(
                    p, self.importance, shadow_method, self.shadow_mode, self.max_iter
                )
            )

        for it in range(self.max_iter):
            tentative_idx = np.where(status == 0)[0]
            if tentative_idx.size == 0:
                if self.verbose:
                    logger.info(f"  iter={it + 1}: all features decided, stopping")
                break

            active_idx = np.where(status != -1)[0]
            n_active = active_idx.size
            X_active = X_arr[:, active_idx]
            seed = int(rng.integers(0, 2**31 - 1))
            est = _clone_estimator(base_est, seed=seed)

            if self.n_estimators == "auto":
                if self.estimator is None:
                    iter_n_estimators = _compute_auto_n_estimators(
                        n_active, base_depth
                    )
                    _set_n_estimators(est, iter_n_estimators)
                else:
                    iter_n_estimators = None
            else:
                iter_n_estimators = int(self.n_estimators)
                _set_n_estimators(est, iter_n_estimators)

            imp = self._compute_importance(
                est,
                X_active,
                y_arr,
                w_score,
                w_fit=w_fit,
                groups=groups,
                time=time,
                seed=seed,
                shadow_method=shadow_method,
                shadow_mode=self.shadow_mode,
                block_size=self.block_size,
                group_info=group_info,
            )

            expected_importance_len = 2 * n_active
            if imp.shape[0] != expected_importance_len:
                raise RuntimeError(
                    f"Importance length {imp.shape[0]} != expected {expected_importance_len}"
                )

            imp_active = imp[:n_active]
            imp_shadow = imp[n_active:]

            thr = float(np.percentile(imp_shadow, self.perc))
            shadow_thresholds.append(thr)

            for i_local in range(n_active):
                j = active_idx[i_local]
                if status[j] == 0:
                    if imp_active[i_local] > thr:
                        hits[j] += 1
                imp_sum[j] += float(imp_active[i_local])
                imp_count[j] += 1

            n_trials += 1

            pmf = _poisson_binom_pmf(
                np.full(n_trials, 0.5, dtype=np.float64)
            )

            tent = np.where(status == 0)[0]
            m = max(1, tent.size)
            alpha_adj = self.alpha / m
            decision_horizon = self._earliest_decidable_trial(
                alpha_adj,
                max_trials=self.max_iter,
            )

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

            if self.verbose or self.callback is not None:
                n_acc = int((status == 1).sum())
                n_rej = int((status == -1).sum())
                n_ten = int((status == 0).sum())
            if self.verbose:
                if iter_n_estimators is not None:
                    logger.info(
                        "  iter={:02d} n_est={} thr={:.4f} acc={} rej={} tent={}".format(
                            it + 1, iter_n_estimators, thr, n_acc, n_rej, n_ten
                        )
                    )
                else:
                    logger.info(
                        "  iter={:02d} thr={:.4f} acc={} rej={} tent={}".format(
                            it + 1, thr, n_acc, n_rej, n_ten
                        )
                    )

            if self.callback is not None:
                report_progress(
                    self.callback,
                    n_trials,
                    self.max_iter,
                    stage="iteration",
                    accepted=n_acc,
                    rejected=n_rej,
                    tentative=n_ten,
                    shadow_threshold=thr,
                    n_estimators=iter_n_estimators,
                )

            if decided_this_round == 0 and n_trials >= decision_horizon:
                no_progress_count += 1
                if no_progress_count >= self.early_stop_rounds:
                    if self.verbose:
                        logger.info(
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
        return BorutaLoopResult(
            status=status,
            hits=hits,
            n_trials=int(n_trials),
            shadow_thresholds=shadow_thresholds_arr,
            mean_importance=mean_importance,
        )

    @staticmethod
    def _earliest_decidable_trial(alpha_adj: float, *, max_trials: int) -> int:
        """Smallest trial count where an all-hit/all-miss binomial tail can pass."""
        if not np.isfinite(alpha_adj) or alpha_adj <= 0.0:
            return int(max_trials) + 1
        trial = 0
        tail = 1.0
        while trial <= int(max_trials) and tail >= float(alpha_adj):
            trial += 1
            tail *= 0.5
        return trial

    def _resolve_boruta_final_status(
        self,
        status: np.ndarray,
        mean_importance: np.ndarray,
        shadow_thresholds_arr: np.ndarray,
    ) -> np.ndarray:
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
                order = acc[np.argsort(-mean_importance[acc], kind="mergesort")]
                keep = set(order[: self.max_features].tolist())
                for j in acc:
                    if int(j) not in keep:
                        status[j] = -1
        return status

    def _store_boruta_attributes(
        self,
        feature_names,
        status,
        hits,
        n_trials,
        shadow_thresholds_arr,
        mean_importance,
    ) -> None:
        self.feature_names_in_ = feature_names_array(feature_names)
        self.n_features_in_ = len(feature_names)
        self.status_ = status
        self.hits_ = hits
        self.n_iter_ = int(n_trials)
        self.shadow_thresholds_ = shadow_thresholds_arr
        self.mean_importance_ = mean_importance
        self.selected_features_ = [feature_names[i] for i in np.where(status == 1)[0]]


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
    n_estimators: int | str = "auto",
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
    cat_encoding: CatEncoding = "none",
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    allow_full_data_target_encoding: bool = False,
    importance_data: Literal["train", "test"] = "train",
    test_size: float = 0.3,
    shap_sample_size: int | None = 2000,
    early_stop_rounds: int = 5,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
    callback: ProgressCallback | None = None,
) -> list[str] | BorutaResult:
    """
    Boruta feature selection.

    Function-style entry point that builds a :class:`BorutaSelector`, fits it,
    and returns the accepted features. Boruta is an all-relevant selector: at
    each iteration it compares every still-undecided feature against a
    permuted "shadow" copy of the matrix and accepts or rejects it by a
    Bonferroni-adjusted binomial tail test on the hit counts, so the answer is
    "everything that carries signal", not a minimal subset of size ``k``. Use
    it for understanding and for all-relevant screening; use the fixed-``k``
    filters when a budgeted, low-redundancy block is what a downstream model
    needs.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Candidate features. ``cat_features`` requires a DataFrame.
    y : array-like of shape (n_samples,)
        Target values.
    task : {"regression", "classification"}, default="regression"
        Problem type; it also picks the default estimator.
    n_estimators : int or "auto", default="auto"
        Number of trees/iterations for the estimator. When "auto", use a fast
        bounded heuristic based on active features and depth. Auto only applies
        when estimator is None.
    sample_weight : array-like, optional
        Non-negative row weights used for importance scoring, and for fitting
        where the estimator supports it.
    groups : array-like, optional
        Group labels for shadow permutation.
    time : array-like, optional
        Time values for ordering.
    group_col : str, optional
        Column name in X to use as groups (extracted and dropped from X).
        Passing both ``groups`` and ``group_col`` raises.
    time_col : str, optional
        Column name in X to use as time (extracted and dropped from X).
        Passing both ``time`` and ``time_col`` raises.
    estimator : estimator object, optional
        Base learner. None uses a RandomForest for native importance or
        CatBoost for SHAP importance.
    importance : {"native", "shap"}, default="native"
        Importance backend: ``feature_importances_`` or SHAP values.
    max_iter : int, default=50
        Maximum Boruta iterations.
    alpha : float, default=0.05
        Significance level for the accept/reject tail tests, Bonferroni
        adjusted by the number of still-tentative features.
    perc : int, default=100
        Percentile of the shadow importances used as the per-iteration
        threshold; 100 is the classic maximum-shadow rule.
    resolve_tentative : bool, default=True
        Force any feature still tentative at the end to accepted or rejected
        against the median shadow threshold.
    max_features : int, optional
        Cap on accepted features; the weakest accepted ones are demoted.
    shadow_method : str, default="auto"
        Shadow permutation method: ``"auto"``, ``"global"``,
        ``"within_group"``, ``"block"`` or ``"circular_shift"``. ``"auto"``
        chooses from the supplied groups/time.
    shadow_mode : {"columns", "rows"}, default="columns"
        Permutation axis. ``"columns"`` permutes each column independently
        (classic Boruta); ``"rows"`` applies one row permutation to every
        column, preserving cross-feature covariance in the null.
    block_size : int or "auto", default="auto"
        Block size for the block permutation methods.
    cat_features : list of str, optional
        Categorical column names to encode; requires a DataFrame ``X``. None
        auto-detects object, category and string columns, and names absent
        from ``X`` are dropped.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", "loo_logit"}
        Categorical encoding fitted before the Boruta loop, default "none".
        "target_cv" is the leakage-safe cross-fitted encoder and is the only
        supervised value accepted without an explicit opt-in; the four legacy
        encodings fit on the full dataset and therefore require
        ``allow_full_data_target_encoding=True``, because tree learners can
        read a row's own target back out of them. Any supervised value is
        rejected with ``importance_data="test"``, and ``sample_weight`` is
        consumed only by "target_cv" and "loo_logit".
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : "auto" or float, default="auto"
        Empirical-Bayes shrinkage for "target_cv"; an explicit non-negative
        float is always accepted.
    target_prior : float, optional
        Target-independent prior for time-aware "target_cv" fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        "target_cv" fit when no ``target_prior`` is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to the leakage-prone full-dataset supervised encodings. It is
        rejected together with ``cat_encoding="target_cv"``.
    importance_data : {"train", "test"}, default="train"
        Compute importance on the training rows or on a held-out split.
    test_size : float, default=0.3
        Held-out fraction when ``importance_data="test"``.
    shap_sample_size : int, optional
        Row subsample for SHAP importance; None uses every row.
    early_stop_rounds : int, default=5
        Stop after this many consecutive iterations that decide nothing.
    random_state : int, default=0
        Seed for the estimator clones and the shadow permutations.
    verbose : bool, default=True
        Emit per-iteration progress at INFO on the ``sift`` logger.
    return_result : bool, default=False
        If True, return BorutaResult instead of feature list.
    callback : callable, optional
        Called after each completed iteration as
        ``callback(step, total, info)``.

    Returns
    -------
    list[str] or BorutaResult
        Accepted feature labels in input-column order, or the full
        :class:`BorutaResult` when ``return_result=True``.

    Raises
    ------
    ValueError
        If ``cat_features`` is given for a non-DataFrame ``X``, if a direct
        row argument and its ``*_col`` alias are both supplied, if a
        supervised ``cat_encoding`` is used without its opt-in or together
        with ``importance_data="test"``, or if an option is out of range.

    See Also
    --------
    select_boruta_shap : The same search with SHAP importance.
    BorutaSelector : The estimator behind this function.
    BorutaResult : The result object returned by ``return_result=True``.

    Notes
    -----
    Because every feature that beats its shadow is kept, the accepted set can
    be far larger than a filter's ``k`` and can include correlated duplicates.
    ``importance="shap"`` needs the optional ``catboost`` extra unless an
    explicit ``estimator`` is supplied.

    Examples
    --------
    >>> import numpy as np
    >>> from sift import select_boruta
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 5))
    >>> y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * rng.normal(size=120)
    >>> select_boruta(X, y, max_iter=10, n_estimators=50, random_state=0,
    ...               verbose=False)
    ['x0', 'x1']
    """
    metadata = resolve_row_metadata(
        X,
        groups=groups,
        time=time,
        sample_weight=sample_weight,
        group_col=group_col,
        time_col=time_col,
    )
    X = metadata.X
    groups = metadata.groups
    time = metadata.time
    sample_weight = metadata.sample_weight
    if isinstance(X, pd.DataFrame):
        if cat_features is not None:
            cat_features = [c for c in cat_features if c in X.columns]
    elif cat_features:
        raise ValueError("cat_features requires X to be a pandas DataFrame")

    sel = BorutaSelector(
        estimator=estimator,
        n_estimators=n_estimators,
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
        target_cv_n_splits=target_cv_n_splits,
        target_cv_smoothing=target_cv_smoothing,
        target_prior=target_prior,
        warmup_policy=warmup_policy,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        importance_data=importance_data,
        test_size=test_size,
        shap_sample_size=shap_sample_size,
        early_stop_rounds=early_stop_rounds,
        random_state=random_state,
        verbose=verbose,
        callback=callback,
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
    n_estimators: int | str = "auto",
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
    cat_encoding: CatEncoding = "none",
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    allow_full_data_target_encoding: bool = False,
    importance_data: Literal["train", "test"] = "train",
    test_size: float = 0.3,
    shap_sample_size: int | None = 2000,
    early_stop_rounds: int = 5,
    random_state: int = 0,
    verbose: bool = True,
    return_result: bool = False,
    callback: ProgressCallback | None = None,
) -> list[str] | BorutaResult:
    """
    Boruta-Shap feature selection (convenience wrapper for importance='shap').

    Identical to :func:`select_boruta` except that the importance backend is
    fixed to SHAP values, which score a feature by its attributed contribution
    to individual predictions rather than by the impurity or split statistics a
    tree ensemble happens to expose. Prefer it when native importances are
    known to be biased (high-cardinality or highly correlated features);
    prefer :func:`select_boruta` when runtime matters, because SHAP is
    markedly more expensive per iteration.

    Parameters
    ----------
    X : DataFrame or ndarray of shape (n_samples, n_features)
        Candidate features. ``cat_features`` requires a DataFrame.
    y : array-like of shape (n_samples,)
        Target values.
    task : {"regression", "classification"}, default="regression"
        Problem type; it also picks the default estimator.
    n_estimators : int or "auto"
        Number of trees/iterations for the estimator. When "auto", use a fast
        bounded heuristic based on active features and depth. Auto only applies
        when estimator is None.
    sample_weight : array-like, optional
        Non-negative row weights used for importance scoring, and for fitting
        where the estimator supports it.
    groups : array-like, optional
        Group labels for shadow permutation.
    time : array-like, optional
        Time values for ordering.
    group_col : str, optional
        Column name in X to use as groups (extracted and dropped from X).
        Passing both ``groups`` and ``group_col`` raises.
    time_col : str, optional
        Column name in X to use as time (extracted and dropped from X).
        Passing both ``time`` and ``time_col`` raises.
    estimator : estimator object, optional
        Base learner. None uses CatBoost, which supplies native SHAP values.
    max_iter : int, default=50
        Maximum Boruta iterations.
    alpha : float, default=0.05
        Significance level for the accept/reject tail tests, Bonferroni
        adjusted by the number of still-tentative features.
    perc : int, default=100
        Percentile of the shadow importances used as the per-iteration
        threshold; 100 is the classic maximum-shadow rule.
    resolve_tentative : bool, default=True
        Force any feature still tentative at the end to accepted or rejected
        against the median shadow threshold.
    max_features : int, optional
        Cap on accepted features; the weakest accepted ones are demoted.
    shadow_method : str, default="auto"
        Shadow permutation method: ``"auto"``, ``"global"``,
        ``"within_group"``, ``"block"`` or ``"circular_shift"``. ``"auto"``
        chooses from the supplied groups/time.
    shadow_mode : {"columns", "rows"}, default="columns"
        Permutation axis. ``"columns"`` permutes each column independently
        (classic Boruta); ``"rows"`` applies one row permutation to every
        column, preserving cross-feature covariance in the null.
    block_size : int or "auto", default="auto"
        Block size for the block permutation methods.
    cat_features : list of str, optional
        Categorical column names to encode; requires a DataFrame ``X``. None
        auto-detects object, category and string columns, and names absent
        from ``X`` are dropped.
    cat_encoding : {"none", "target_cv", "target", "loo", "james_stein", "loo_logit"}
        Categorical encoding fitted before the Boruta loop, default "none".
        "target_cv" is the leakage-safe cross-fitted encoder and is the only
        supervised value accepted without an explicit opt-in; the four legacy
        encodings fit on the full dataset and therefore require
        ``allow_full_data_target_encoding=True``. Any supervised value is
        rejected with ``importance_data="test"``, and ``sample_weight`` is
        consumed only by "target_cv" and "loo_logit".
    target_cv_n_splits : int, default=5
        Fold count for ``cat_encoding="target_cv"``.
    target_cv_smoothing : "auto" or float, default="auto"
        Empirical-Bayes shrinkage for "target_cv"; an explicit non-negative
        float is always accepted.
    target_prior : float, optional
        Target-independent prior for time-aware "target_cv" fits, so the
        earliest block emits a centered neutral zero and stays in the fit.
    warmup_policy : {"exclude", "zero_weight"}, default="zero_weight"
        How to treat the earliest no-history block of a time-aware
        "target_cv" fit when no ``target_prior`` is given.
    allow_full_data_target_encoding : bool, default=False
        Opt in to the leakage-prone full-dataset supervised encodings. It is
        rejected together with ``cat_encoding="target_cv"``.
    importance_data : {"train", "test"}, default="train"
        Compute importance on the training rows or on a held-out split.
    test_size : float, default=0.3
        Held-out fraction when ``importance_data="test"``.
    shap_sample_size : int, optional
        Row subsample for the SHAP computation, default 2000. None uses every
        row and is markedly slower on large matrices.
    early_stop_rounds : int, default=5
        Stop after this many consecutive iterations that decide nothing.
    random_state : int, default=0
        Seed for the estimator clones, the shadow permutations and the SHAP
        row subsample.
    verbose : bool, default=True
        Emit per-iteration progress at INFO on the ``sift`` logger.
    return_result : bool, default=False
        If True, return BorutaResult instead of feature list.
    callback : callable, optional
        Called after each completed iteration as
        ``callback(step, total, info)``.

    Returns
    -------
    list[str] or BorutaResult
        Accepted feature labels in input-column order, or the full
        :class:`BorutaResult` when ``return_result=True``.

    Raises
    ------
    ImportError
        If neither ``catboost`` (native SHAP) nor the ``shap`` package is
        installed for the estimator in use.
    ValueError
        If ``cat_features`` is given for a non-DataFrame ``X``, if a direct
        row argument and its ``*_col`` alias are both supplied, if a
        supervised ``cat_encoding`` is used without its opt-in or together
        with ``importance_data="test"``, or if an option is out of range.

    See Also
    --------
    select_boruta : The same search with a selectable importance backend.
    BorutaSelector : The estimator behind both functions.
    BorutaResult : The result object returned by ``return_result=True``.

    Notes
    -----
    SHAP values come from CatBoost natively when the estimator is a CatBoost
    model, and from the optional ``shap`` package otherwise, so one of the two
    must be installed. Everything else, including the all-relevant accept
    semantics, matches :func:`select_boruta`.

    Examples
    --------
    This example needs the optional ``catboost`` (or ``shap``) dependency. It
    is shown as code rather than an executed doctest, because SIFT's own test
    environment installs neither::

        import numpy as np
        import sift

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 6))
        y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.1 * rng.normal(size=200)

        features = sift.select_boruta_shap(
            X, y, task="regression", max_iter=10, random_state=0,
            verbose=False,
        )
    """
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
        n_estimators=n_estimators,
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
        target_cv_n_splits=target_cv_n_splits,
        target_cv_smoothing=target_cv_smoothing,
        target_prior=target_prior,
        warmup_policy=warmup_policy,
        allow_full_data_target_encoding=allow_full_data_target_encoding,
        importance_data=importance_data,
        test_size=test_size,
        shap_sample_size=shap_sample_size,
        early_stop_rounds=early_stop_rounds,
        random_state=random_state,
        verbose=verbose,
        callback=callback,
        return_result=return_result,
    )


__all__ = [
    "BorutaSelector",
    "BorutaResult",
    "select_boruta",
    "select_boruta_shap",
]
