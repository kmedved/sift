from collections.abc import Hashable, Iterable, Mapping, Set
from dataclasses import replace
from functools import wraps
import inspect
import numbers
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import warnings

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso, LassoCV, ElasticNet, ElasticNetCV,
    LogisticRegression, LogisticRegressionCV, Ridge,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from sift._logging import logger
from sift._progress import ProgressCallback, report_progress
from sift._preprocess import ensure_weights, reject_datetime_like_features
from sift.sampling.smart import SmartSamplerConfig, smart_sample


def _coerce_feature_names(feature_names, *, argument: str = "feature_names") -> list[Hashable]:
    """Normalize an ordered one-dimensional collection of hashable column labels."""
    invalid_container = isinstance(
        feature_names,
        (str, bytes, bytearray, memoryview, Mapping, Set),
    )
    ndim = getattr(feature_names, "ndim", None)
    if invalid_container or (ndim is not None and ndim != 1):
        raise ValueError(
            f"{argument} must be an ordered, one-dimensional iterable of names; "
            "pass a list, tuple, pandas Index, or one-dimensional NumPy array, "
            "not a string, bytes-like object, mapping, set, scalar, or matrix."
        )
    try:
        names = list(feature_names)
    except TypeError as exc:
        raise ValueError(
            f"{argument} must be an ordered, one-dimensional iterable of names."
        ) from exc
    for name in names:
        try:
            hash(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{argument} entries must be hashable column labels."
            ) from exc
    return names


def _feature_names_object_array(feature_names) -> np.ndarray:
    """Build a one-dimensional object array without expanding tuple labels."""
    names = list(feature_names)
    result = np.empty(len(names), dtype=object)
    result[:] = names
    return result


def _feature_names_index(feature_names) -> pd.Index:
    """Build a flat object Index with exact tuple and missing-label semantics."""
    return pd.Index(
        _feature_names_object_array(feature_names),
        dtype=object,
        tupleize_cols=False,
    )


def _exact_column_positions(columns, required_names) -> np.ndarray:
    """Resolve required labels exactly, without pandas MultiIndex partial keys."""
    available = _feature_names_index(columns)
    required = _feature_names_index(required_names)
    return available.get_indexer(required)


# =============================================================================
# Stability Selector
# =============================================================================

from sift.sampling.stability import (
    _block_bootstrap_indices,
    _bootstrap_indices,
)


def _single_threaded_blas(func):
    """Keep bootstrap model fits from multiplying native BLAS threads."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        with threadpool_limits(limits=1):
            return func(*args, **kwargs)

    return wrapped


def _cv_alpha_grid_kwargs(model_cls, n_alphas: int) -> dict[str, int]:
    """Return sklearn-version-compatible alpha-grid kwargs for CV estimators."""
    params = inspect.signature(model_cls).parameters
    if "n_alphas" in params:
        return {"n_alphas": int(n_alphas)}
    return {"alphas": int(n_alphas)}


class StabilitySelector(BaseEstimator, TransformerMixin):
    """
    Stability selection for linear models with optional smart sampling.

    Fits Lasso/ElasticNet (regression) or LogisticRegression (classification)
    on bootstrap subsamples and keeps features selected consistently across runs.
    Handles correlated features by revealing which ones are robustly predictive
    vs. interchangeable proxies.

    Note: This is a practical stability selection implementation inspired by
    Meinshausen & Bühlmann (2010), but does not provide formal false-positive
    control. Use it as a robust heuristic for pre-filtering features.

    Parameters
    ----------
    n_bootstrap : int, default=50
        Number of bootstrap iterations.
    sample_frac : float, default=0.5
        Fraction of data to use in each bootstrap sample.
    threshold : float, default=0.6
        Minimum selection frequency to keep a feature.
    alpha : float, optional
        Regularization strength. If None, estimated via CV.
    alpha_rule : {'one_se', 'best'}, default='one_se'
        Rule used when ``alpha`` is estimated. ``one_se`` chooses the strongest
        regularization whose CV score is within one standard error of the best;
        ``best`` chooses the prediction-optimal grid point.
    l1_ratio : float, default=1.0
        ElasticNet mixing (1.0 = Lasso, <1.0 = ElasticNet). Only for regression.
    task : str, default='regression'
        Either 'regression' or 'classification'.
    max_features : int, optional
        Hard cap on number of selected features.
    use_smart_sampler : bool, default=False
        Whether to apply smart sampling before stability selection.
    sampler_config : SmartSamplerConfig, optional
        Configuration for smart sampler.
    store_coefs : bool, default=True
        Whether to store full coefficient matrix from all bootstraps.
        Set False to save memory (disables get_coef_stability and plot_coef_distributions).
    coef_threshold : float, default=1e-8
        Threshold for considering a coefficient as non-zero.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).
    parallel_backend : str, default='threads'
        Joblib backend preference. 'threads' has lower memory overhead,
        'processes' is more isolated. Set to None for joblib default.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress information.
    callback : callable, optional
        Called after each completed bootstrap as
        ``callback(step, total, info)``.

    Attributes
    ----------
    selection_frequencies_ : ndarray of shape (n_features,)
        Fraction of bootstrap runs in which each feature was selected.
    selected_features_ : ndarray
        Indices of selected features.
    selected_feature_names_ : list of hashable labels
        Names of selected features.
    n_features_selected_ : int
        Number of selected features.
    alpha_ : float
        Regularization alpha used.
    coef_bootstrap_ : ndarray of shape (n_bootstrap, n_features), optional
        Coefficients from each bootstrap run. Only available if store_coefs=True.
    """

    def __init__(
        self,
        n_bootstrap: int = 50,
        sample_frac: float = 0.5,
        threshold: float = 0.6,
        alpha: Optional[float] = None,
        alpha_rule: str = "one_se",
        l1_ratio: float = 1.0,
        task: str = 'regression',
        max_features: Optional[int] = None,
        block_size: int | str = "auto",
        block_method: str = "moving",
        use_smart_sampler: bool = False,
        sampler_config: Optional[SmartSamplerConfig] = None,
        store_coefs: bool = True,
        coef_threshold: float = 1e-8,
        n_jobs: int = -1,
        parallel_backend: str = 'threads',
        random_state: Optional[int] = None,
        verbose: bool = True,
        callback: ProgressCallback | None = None,
    ):
        self.n_bootstrap = n_bootstrap
        self.sample_frac = sample_frac
        self.threshold = threshold
        self.alpha = alpha
        self.alpha_rule = alpha_rule
        self.l1_ratio = l1_ratio
        self.task = task
        self.max_features = max_features
        self.block_size = block_size
        self.block_method = block_method
        self.use_smart_sampler = use_smart_sampler
        self.sampler_config = sampler_config
        self.store_coefs = store_coefs
        self.coef_threshold = coef_threshold
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.random_state = random_state
        self.verbose = verbose
        self.callback = callback

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        groups: np.ndarray | None = None,
        time: np.ndarray | None = None,
        feature_names: Optional[Iterable[Hashable]] = None
    ) -> 'StabilitySelector':
        """
        Run stability selection.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        groups : array, optional
            Group labels. If provided with time, uses block bootstrap.
        time : array, optional
            Time values. If provided with groups, uses block bootstrap.
        feature_names : ordered iterable of hashable labels, optional
            Feature names. Strings, bytes-like objects, mappings, sets, scalar
            arrays, and matrix-like containers are rejected.

        Returns
        -------
        self
        """
        self._clear_fit_state()
        try:
            self._fit_input_kind_ = (
                "dataframe" if isinstance(X, pd.DataFrame) else "positional"
            )
            self._fit_used_sample_weight_ = sample_weight is not None
            self._fit_used_groups_ = groups is not None
            self._fit_used_time_ = time is not None
            self._fit_feature_names_generated_ = (
                feature_names is None and not isinstance(X, pd.DataFrame)
            )
            if isinstance(X, pd.DataFrame):
                column_index = _feature_names_index(X.columns)
                duplicate_mask = column_index.duplicated()
            else:
                duplicate_mask = np.zeros(0, dtype=bool)
            if duplicate_mask.any():
                duplicates = column_index[duplicate_mask].unique().tolist()[:5]
                raise ValueError(
                    "Duplicate DataFrame column labels are not supported: "
                    f"{duplicates}. Rename columns before fitting."
                )
            if feature_names is not None:
                feature_names = _coerce_feature_names(feature_names)
                if len(feature_names) == 0:
                    raise ValueError(
                        "feature_names must be a non-empty list of unique names "
                        "when provided; pass None to derive names from X."
                    )
                feature_index = _feature_names_index(feature_names)
                if feature_index.duplicated().any():
                    raise ValueError(
                        "feature_names must be unique; duplicate names (including "
                        "repeated NaN labels) make name-based selection and "
                        "transform ambiguous."
                    )
                if isinstance(X, pd.DataFrame):
                    positions = _exact_column_positions(X.columns, feature_names)
                    missing = [
                        feature_names[i]
                        for i in np.flatnonzero(positions < 0)
                    ]
                    if missing:
                        sample = missing[:5]
                        suffix = "..." if len(missing) > 5 else ""
                        raise ValueError(
                            "feature_names must reference existing DataFrame "
                            f"columns; missing: {sample}{suffix}"
                        )
            X_scaled, y, sample_weight, feature_names, groups, time = self._prepare_stability_fit(
                X, y, sample_weight, groups, time, feature_names
            )
            split_iter = self._make_stability_split_iterator(len(y), y, groups, time)
            sel_count, sum_abs_coef, n_runs = self._run_stability_chunks(
                X_scaled,
                y,
                sample_weight,
                split_iter,
            )
            self._finalize_stability_selection(sel_count, sum_abs_coef, n_runs, feature_names)
        except Exception:
            self._clear_fit_state()
            raise
        return self

    def _clear_fit_state(self) -> None:
        for attr in (
            "_impute_means_",
            "_label_encoder",
            "_scaler",
            "_alpha_ref_weight_",
            "_target_center_",
            "_fit_feature_names_generated_",
            "_fit_input_kind_",
            "_fit_used_groups_",
            "_fit_used_sample_weight_",
            "_fit_used_time_",
            "alpha_",
            "alpha_rule_effective_",
            "classes_",
            "coef_bootstrap_",
            "feature_names_in_",
            "mean_abs_coef_",
            "n_features_in_",
            "n_features_selected_",
            "sampled_n_",
            "selected_feature_names_",
            "selected_features_",
            "selection_frequencies_",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def _prepare_stability_fit(self, X, y, sample_weight, groups, time, feature_names):
        # Input validation
        self._validate_runtime_params()
        if self.use_smart_sampler and (groups is not None or time is not None):
            raise ValueError("groups/time are not supported when use_smart_sampler=True.")
        if self.use_smart_sampler:
            X, y, sample_weight, feature_names = self._apply_smart_sampler(
                X, y, sample_weight, feature_names
            )
        else:
            X, y, sample_weight, feature_names = self._prep_arrays(
                X, y, sample_weight, feature_names
            )

        if self.task == "regression":
            self._target_center_ = float(
                np.average(
                    np.asarray(y, dtype=np.float64),
                    weights=np.asarray(sample_weight, dtype=np.float64),
                )
            )
            y = np.asarray(y, dtype=np.float64) - self._target_center_

        n, p = X.shape
        self.feature_names_in_ = feature_names
        self.n_features_in_ = p

        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != n:
                raise ValueError(f"groups has {len(groups)} rows but X has {n}")
        if time is not None:
            time = np.asarray(time)
            if len(time) != n:
                raise ValueError(f"time has {len(time)} rows but X has {n}")

        # Tune alpha on raw rows so every CV fold owns its imputation and scale
        # statistics. Full-data preprocessing below is only for final fits.
        self._alpha_ref_weight_ = None
        if self.alpha is None:
            self.alpha_ = self._find_alpha(
                X,
                y,
                sample_weight,
                groups=groups,
                time=time,
            )
            self.alpha_rule_effective_ = self.alpha_rule
        else:
            self.alpha_ = self.alpha
            self.alpha_rule_effective_ = "fixed"

        # Impute and standardize the final fit data after alpha is chosen.
        X = self._impute_with_fit_stats(X, fit=True)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.verbose:
            task_str = 'classification' if self.task == 'classification' else 'regression'
            logger.info(
                f"Stability selection ({task_str}): {self.n_bootstrap} bootstraps, "
                f"α={self.alpha_:.4f}, threshold={self.threshold}"
            )

        return X_scaled, y, sample_weight, feature_names, groups, time

    def _make_stability_split_iterator(self, n: int, y, groups, time):
        use_block = groups is not None and time is not None
        if use_block:
            if self.verbose:
                logger.info(
                    f"Using block bootstrap (method={self.block_method}, size={self.block_size})"
                )
            split_iter = _block_bootstrap_indices(
                n=n,
                n_bootstrap=self.n_bootstrap,
                groups=groups,
                time=time,
                block_size=self.block_size,
                block_method=self.block_method,
                y=y if self.task == "classification" else None,
                task=self.task,
                random_state=self.random_state,
                sample_frac=self.sample_frac,
            )
        else:
            if self.verbose:
                logger.info("Using i.i.d. bootstrap")
            split_iter = _bootstrap_indices(
                n=n,
                n_bootstrap=self.n_bootstrap,
                sample_frac=self.sample_frac,
                y=y if self.task == "classification" else None,
                task=self.task,
                random_state=self.random_state,
            )

        return split_iter

    def _fit_single_stability_run(self, X_scaled, y, sample_weight, train_idx, seed):
        train_idx, train_weight = self._dedupe_train_weights(train_idx, sample_weight)
        if self.task == 'classification':
            model = LogisticRegression(
                penalty='l1', solver='saga', C=self._classification_C(train_weight),
                max_iter=3000, random_state=seed, n_jobs=1,
            )
        elif self.l1_ratio >= 1.0:
            model = Lasso(alpha=self.alpha_, max_iter=3000)
        else:
            model = ElasticNet(alpha=self.alpha_, l1_ratio=self.l1_ratio, max_iter=3000)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_scaled[train_idx], y[train_idx], sample_weight=train_weight)

        coef = model.coef_
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef_summary = coef[0]
            selected = np.abs(coef_summary) > self.coef_threshold
        elif coef.ndim == 2:
            selected = np.any(np.abs(coef) > self.coef_threshold, axis=0)
            coef_summary = np.max(np.abs(coef), axis=0)
        else:
            coef_summary = coef.ravel()
            selected = np.abs(coef_summary) > self.coef_threshold
        return selected.astype(np.int8), coef_summary.astype(np.float32)

    @staticmethod
    def _dedupe_train_weights(train_idx, sample_weight):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        unique_idx, inverse = np.unique(train_idx, return_inverse=True)
        if len(unique_idx) == len(train_idx):
            return train_idx, sample_weight[train_idx]
        summed_weight = np.bincount(
            inverse,
            weights=np.asarray(sample_weight, dtype=np.float64)[train_idx],
            minlength=len(unique_idx),
        )
        return unique_idx, summed_weight.astype(np.float32, copy=False)

    @_single_threaded_blas
    def _run_stability_chunks(self, X_scaled, y, sample_weight, split_iter):
        p = X_scaled.shape[1]
        rng = np.random.default_rng(self.random_state)

        # Chunked execution to reduce peak memory. Splits are streamed instead
        # of materialized up front, which matters for large block bootstraps.
        chunk_size = min(20, self.n_bootstrap)

        sel_count = np.zeros(p, dtype=np.int32)
        sum_abs_coef = np.zeros(p, dtype=np.float64)

        if self.store_coefs:
            self.coef_bootstrap_ = np.empty((self.n_bootstrap, p), dtype=np.float32)

        bootstrap_idx = 0
        split_iterator = iter(split_iter)
        while True:
            chunk_splits = []
            chunk_seeds = []
            for _ in range(chunk_size):
                try:
                    chunk_splits.append(next(split_iterator))
                except StopIteration:
                    break
                chunk_seeds.append(int(rng.integers(0, 2**31)))

            if not chunk_splits:
                break

            chunk_seeds_arr = np.asarray(chunk_seeds, dtype=np.int64)

            chunk_results = Parallel(n_jobs=self.n_jobs, prefer=self.parallel_backend)(
                delayed(self._fit_single_stability_run)(
                    X_scaled,
                    y,
                    sample_weight,
                    train_idx,
                    seed,
                )
                for (train_idx, _), seed in zip(chunk_splits, chunk_seeds_arr)
            )

            # Aggregate this chunk immediately, then discard
            for selected, coef_summary in chunk_results:
                sel_count += selected.astype(np.int32)
                sum_abs_coef += np.abs(coef_summary)

                if self.store_coefs:
                    self.coef_bootstrap_[bootstrap_idx] = coef_summary
                bootstrap_idx += 1
                if self.callback is not None:
                    report_progress(
                        self.callback,
                        bootstrap_idx,
                        self.n_bootstrap,
                        stage="bootstrap",
                        task=self.task,
                        selected_features=int(np.count_nonzero(selected)),
                    )

            # chunk_results goes out of scope here, memory freed

        if bootstrap_idx == 0:
            raise ValueError("No valid bootstrap splits could be generated.")

        if self.store_coefs:
            self.coef_bootstrap_ = self.coef_bootstrap_[:bootstrap_idx]

        return sel_count, sum_abs_coef, bootstrap_idx

    def _finalize_stability_selection(self, sel_count, sum_abs_coef, bootstrap_idx: int, feature_names) -> None:
        p = len(feature_names)
        self.selection_frequencies_ = (sel_count / bootstrap_idx).astype(np.float64)
        self.mean_abs_coef_ = (sum_abs_coef / bootstrap_idx).astype(np.float32)

        # Select features
        mask = self.selection_frequencies_ >= self.threshold

        if self.max_features is not None and mask.sum() > self.max_features:
            top_idx = np.argsort(-self.selection_frequencies_, kind="mergesort")[:self.max_features]
            mask = np.zeros(p, dtype=bool)
            mask[top_idx] = True

        selected = np.where(mask)[0]
        order = np.argsort(-self.selection_frequencies_[selected], kind="mergesort")
        self.selected_features_ = selected[order]
        self.selected_feature_names_ = [feature_names[i] for i in self.selected_features_]
        self.n_features_selected_ = len(self.selected_features_)

        if self.verbose:
            logger.info(f"Selected {self.n_features_selected_} / {p} features")

        return self

    @property
    def result_view_(self):
        """Return a normalized, non-cached view of this fitted selector."""
        from sift.selection.view import as_result

        return as_result(self)

    def _select_dataframe_columns(
        self,
        X: pd.DataFrame,
        required_names: Iterable[Hashable],
        *,
        operation: str,
        selected_only: bool,
    ) -> pd.DataFrame:
        """Apply the fitted DataFrame identity contract for a public operation."""
        if getattr(self, "_fit_feature_names_generated_", False):
            raise ValueError(
                "This StabilitySelector was fitted on a positional array with "
                f"generated feature names; pass a positional ndarray to {operation}, "
                "or refit on a DataFrame to establish column names."
            )
        column_index = _feature_names_index(X.columns)
        duplicate_mask = column_index.duplicated()
        if duplicate_mask.any():
            duplicates = column_index[duplicate_mask].unique().tolist()[:5]
            raise ValueError(
                "Duplicate DataFrame column labels are not supported in "
                f"{operation}: {duplicates}. Rename columns so name-based "
                "selection is unambiguous."
            )
        names = list(required_names)
        positions = _exact_column_positions(X.columns, names)
        missing = [names[i] for i in np.flatnonzero(positions < 0)]
        if missing:
            sample = missing[:5]
            suffix = "..." if len(missing) > 5 else ""
            descriptor = "selected feature" if selected_only else "fitted feature"
            raise ValueError(
                f"X is missing {descriptor} column(s) {sample}{suffix}; "
                f"{operation} selects fitted columns by name and requires them "
                "to be present."
            )
        return X.iloc[:, positions]

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Reduce X to selected features."""
        check_is_fitted(self, ["selected_features_", "selected_feature_names_"])
        if isinstance(X, pd.DataFrame):
            selected = self._select_dataframe_columns(
                X,
                self.selected_feature_names_,
                operation="transform",
                selected_only=True,
            )
            return selected.values
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2-dimensional array-like object")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but StabilitySelector was fitted "
                f"with {self.n_features_in_} features"
            )
        return X_arr[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return names of selected columns using sklearn's transformer contract."""
        check_is_fitted(self, ["selected_features_", "selected_feature_names_", "feature_names_in_"])
        if input_features is not None:
            supplied_names = _coerce_feature_names(
                input_features, argument="input_features"
            )
            if len(supplied_names) != self.n_features_in_:
                raise ValueError(
                    "input_features must contain one name for each fitted feature"
                )
            supplied = _feature_names_index(supplied_names)
            fitted = _feature_names_index(self.feature_names_in_)
            if not supplied.equals(fitted):
                raise ValueError("input_features do not match feature_names_in_")
        return _feature_names_object_array(self.selected_feature_names_)

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **fit_params
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_info(self) -> pd.DataFrame:
        """Return feature frequencies, mean coefficient magnitude, and support."""
        check_is_fitted(
            self,
            ["feature_names_in_", "selection_frequencies_", "mean_abs_coef_"],
        )
        return pd.DataFrame({
            'feature': self.feature_names_in_,
            'frequency': self.selection_frequencies_,
            'mean_abs_coef': self.mean_abs_coef_,
            'selected': self.selection_frequencies_ >= self.threshold
        }).sort_values('frequency', ascending=False, kind="mergesort").reset_index(drop=True)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        check_is_fitted(self, ["selected_features_", "n_features_in_"])
        if indices:
            return self.selected_features_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def get_coef_stability(self) -> pd.DataFrame:
        """Return coefficient mean, std, and CV across bootstrap runs."""
        if not hasattr(self, 'coef_bootstrap_'):
            raise ValueError(
                "Coefficient matrix not available. "
                "Set store_coefs=True when creating the selector."
            )

        coef_mean = self.coef_bootstrap_.mean(axis=0)
        coef_std = self.coef_bootstrap_.std(axis=0)
        coef_cv = np.where(
            np.abs(coef_mean) > 1e-10,
            coef_std / np.abs(coef_mean),
            np.inf
        )

        return pd.DataFrame({
            'feature': self.feature_names_in_,
            'coef_mean': coef_mean,
            'coef_std': coef_std,
            'coef_cv': coef_cv,
            'frequency': self.selection_frequencies_,
            'selected': self.selection_frequencies_ >= self.threshold
        }).sort_values('frequency', ascending=False).reset_index(drop=True)

    def tune_threshold(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        thresholds: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8),
        cv: int = 3,
        scoring: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
        groups: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ) -> Tuple[float, pd.DataFrame]:
        """Choose a threshold with nested, fold-local stability selection.

        Pass the same ``sample_weight``, ``groups``, and ``time`` context used
        for :meth:`fit`. Group labels select group-disjoint outer folds; time
        values select ordered time-series folds when groups are absent. The
        context is also forwarded to each fold-local stability fit.
        """

        if not hasattr(self, 'selection_frequencies_'):
            raise ValueError("Must call fit() before tune_threshold()")
        self._validate_runtime_params()

        if not isinstance(cv, numbers.Integral) or cv <= 1:
            raise ValueError("cv must be an integer greater than 1.")
        if not isinstance(thresholds, (list, tuple, np.ndarray)) or len(thresholds) == 0:
            raise ValueError("thresholds must be a non-empty sequence of floats.")
        for thresh in thresholds:
            if not isinstance(thresh, numbers.Real) or not (0 <= float(thresh) <= 1):
                raise ValueError("thresholds must contain values in [0, 1].")

        required_context = (
            ("sample_weight", "_fit_used_sample_weight_", sample_weight),
            ("groups", "_fit_used_groups_", groups),
            ("time", "_fit_used_time_", time),
        )
        missing_context = [
            name
            for name, attr, value in required_context
            if getattr(self, attr, False) and value is None
        ]
        if missing_context:
            raise ValueError(
                "tune_threshold requires the fit-time context for: "
                + ", ".join(missing_context)
            )

        if isinstance(X, pd.DataFrame):
            X_feature_source = self._select_dataframe_columns(
                X,
                self.feature_names_in_,
                operation="tune_threshold",
                selected_only=False,
            )
            X_values = X_feature_source.to_numpy()
            # Fold-local smart sampling still needs configured group/time
            # metadata, while scoring must remain restricted to fitted
            # features. Keep those two views separate.
            X_selector_source = X if self.use_smart_sampler else X_feature_source
        else:
            X_values = np.asarray(X)
            X_selector_source = X_values
        if X_values.ndim != 2 or X_values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have {self.n_features_in_} feature columns for threshold tuning"
            )
        y_values = np.asarray(y).ravel()
        if y_values.shape[0] != X_values.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        n_rows = X_values.shape[0]
        weight_values = (
            None
            if sample_weight is None
            else ensure_weights(sample_weight, n_rows, normalize=True)
        )
        groups_values = None if groups is None else np.asarray(groups).ravel()
        time_values = None if time is None else np.asarray(time).ravel()
        for name, values in (("groups", groups_values), ("time", time_values)):
            if values is not None and values.shape[0] != n_rows:
                raise ValueError(f"{name} has {values.shape[0]} rows but X has {n_rows}")

        # TimeSeriesSplit assumes rows are already chronological. Reorder all
        # aligned inputs locally rather than requiring callers to mutate them.
        if time_values is not None and groups_values is None:
            order = np.argsort(time_values, kind="mergesort")
            X_values = X_values[order]
            X_selector_source = (
                X_selector_source.iloc[order]
                if isinstance(X_selector_source, pd.DataFrame)
                else X_selector_source[order]
            )
            y_values = y_values[order]
            time_values = time_values[order]
            if weight_values is not None:
                weight_values = weight_values[order]

        scoring = scoring or ('accuracy' if self.task == 'classification' else 'r2')
        scorer = get_scorer(scoring)
        if groups_values is not None:
            n_groups = len(np.unique(groups_values))
            if n_groups < 2:
                raise ValueError("groups must contain at least two distinct values")
            splitter = GroupKFold(n_splits=min(int(cv), n_groups))
            split_iter = splitter.split(X_values, y_values, groups_values)
        elif time_values is not None:
            if n_rows < 3:
                raise ValueError("time-aware threshold tuning requires at least 3 rows")
            splitter = TimeSeriesSplit(n_splits=min(int(cv), n_rows - 1))
            split_iter = splitter.split(X_values)
        else:
            splitter = (
                StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=self.random_state)
                if self.task == 'classification'
                else KFold(n_splits=int(cv), shuffle=True, random_state=self.random_state)
            )
            split_iter = splitter.split(
                X_values,
                y_values if self.task == 'classification' else None,
            )
        threshold_values = [float(value) for value in thresholds]
        fold_scores = {value: [] for value in threshold_values}
        fold_sizes = {value: [] for value in threshold_values}

        for train_idx, val_idx in split_iter:
            fold_selector = clone(self).set_params(
                store_coefs=False,
                verbose=False,
                callback=None,
            )
            X_selector_train = (
                X_selector_source.iloc[train_idx]
                if isinstance(X_selector_source, pd.DataFrame)
                else X_selector_source[train_idx]
            )
            fold_fit_kwargs = {}
            if weight_values is not None:
                fold_fit_kwargs["sample_weight"] = weight_values[train_idx]
            if groups_values is not None:
                fold_fit_kwargs["groups"] = groups_values[train_idx]
            if time_values is not None:
                fold_fit_kwargs["time"] = time_values[train_idx]
            if isinstance(X_selector_train, pd.DataFrame) and self.use_smart_sampler:
                fold_fit_kwargs["feature_names"] = list(self.feature_names_in_)
            fold_selector.fit(
                X_selector_train,
                y_values[train_idx],
                **fold_fit_kwargs,
            )
            frequencies = np.asarray(
                fold_selector.selection_frequencies_, dtype=np.float64
            )

            for thresh in threshold_values:
                selected = np.flatnonzero(frequencies >= thresh)
                if self.max_features is not None and selected.size > self.max_features:
                    order = np.argsort(-frequencies[selected], kind="mergesort")
                    selected = selected[order[: self.max_features]]
                fold_sizes[thresh].append(int(selected.size))
                if selected.size == 0:
                    fold_scores[thresh].append(float("nan"))
                    continue

                downstream = (
                    LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
                    if self.task == 'classification'
                    else Ridge(alpha=1.0)
                )
                model = make_pipeline(
                    SimpleImputer(strategy="mean"),
                    StandardScaler(),
                    downstream,
                )
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        downstream_fit_kwargs = {}
                        if weight_values is not None:
                            final_name = model.steps[-1][0]
                            downstream_fit_kwargs[f"{final_name}__sample_weight"] = (
                                weight_values[train_idx]
                            )
                        model.fit(
                            X_values[train_idx][:, selected],
                            y_values[train_idx],
                            **downstream_fit_kwargs,
                        )
                        score_kwargs = {}
                        if weight_values is not None:
                            score_kwargs["sample_weight"] = weight_values[val_idx]
                        score = float(
                            scorer(
                                model,
                                X_values[val_idx][:, selected],
                                y_values[val_idx],
                                **score_kwargs,
                            )
                        )
                except Exception as exc:
                    warnings.warn(
                        "Threshold tuning failed on one outer fold and recorded "
                        f"a missing score: {type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    score = float("nan")
                fold_scores[thresh].append(score)

        rows = []
        for thresh in threshold_values:
            scores = np.asarray(fold_scores[thresh], dtype=np.float64)
            finite = scores[np.isfinite(scores)]
            sizes = np.asarray(fold_sizes[thresh], dtype=np.float64)
            rows.append({
                'threshold': thresh,
                'n_features': float(np.mean(sizes)) if sizes.size else 0.0,
                'min_features': int(np.min(sizes)) if sizes.size else 0,
                'max_features': int(np.max(sizes)) if sizes.size else 0,
                'mean_score': (
                    float(np.mean(scores))
                    if scores.size and finite.size == scores.size
                    else np.nan
                ),
                'std_score': float(np.std(finite)) if finite.size else np.nan,
                'n_finite': int(finite.size),
                'n_splits': int(scores.size),
            })

        results = pd.DataFrame(rows)
        valid = results.dropna(subset=['mean_score'])
        best_thresh = threshold_values[0] if valid.empty else valid.loc[valid['mean_score'].idxmax(), 'threshold']
        if self.verbose:
            logger.info(f"Threshold tuning results (scoring={scoring}):")
            logger.info(results.to_string(index=False))
            logger.info(f"Best threshold: {best_thresh}")
        return best_thresh, results

    def set_threshold(self, threshold: float) -> 'StabilitySelector':
        """Update threshold and recompute selected features."""
        if not hasattr(self, 'selection_frequencies_'):
            raise ValueError("Must call fit() before set_threshold()")
        if not isinstance(threshold, numbers.Real) or not (0 <= float(threshold) <= 1):
            raise ValueError("threshold must be in [0, 1].")

        self.threshold = threshold
        mask = self.selection_frequencies_ >= threshold
        if self.max_features is not None and mask.sum() > self.max_features:
            top_idx = np.argsort(-self.selection_frequencies_, kind="mergesort")[:self.max_features]
            mask = np.zeros(self.n_features_in_, dtype=bool)
            mask[top_idx] = True

        selected = np.where(mask)[0]
        order = np.argsort(-self.selection_frequencies_[selected], kind="mergesort")
        self.selected_features_ = selected[order]
        self.selected_feature_names_ = [self.feature_names_in_[i] for i in self.selected_features_]
        self.n_features_selected_ = len(self.selected_features_)
        if self.verbose:
            logger.info(
                f"Updated threshold to {threshold}: "
                f"{self.n_features_selected_} features selected"
            )
        return self

    def plot_frequencies(
        self,
        top_n: int = 50,
        figsize: Optional[Tuple[float, float]] = None,
        show_coef: bool = False
    ):
        """Bar plot of selection frequencies."""
        import matplotlib.pyplot as plt

        info = self.get_feature_info().head(top_n)
        if figsize is None:
            figsize = (10, max(6, top_n * 0.25))

        fig, ax = plt.subplots(figsize=figsize)
        if show_coef:
            coef_norm = info['mean_abs_coef'] / (info['mean_abs_coef'].max() + 1e-10)
            colors = plt.cm.Blues(0.3 + 0.7 * coef_norm)
        else:
            colors = ['steelblue' if s else 'lightgray' for s in info['selected']]

        ax.barh(range(len(info)), info['frequency'], color=colors)
        ax.set_yticks(range(len(info)))
        ax.set_yticklabels(info['feature'])
        ax.axvline(self.threshold, color='red', linestyle='--', label=f'threshold={self.threshold}')
        ax.set_xlabel('Selection Frequency')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        ax.set_title(f'Stability Selection ({self.n_features_selected_} features selected)')
        plt.tight_layout()
        return fig, ax

    def plot_coef_distributions(
        self,
        features: Optional[Iterable[Hashable]] = None,
        top_n: int = 12,
    ):
        """Plot coefficient distributions across bootstrap runs."""
        if not hasattr(self, 'coef_bootstrap_'):
            raise ValueError(
                "Coefficient matrix not available. "
                "Set store_coefs=True when creating the selector."
            )
        if features is None:
            features = self.get_feature_info()['feature'].head(top_n).tolist()
        else:
            features = _coerce_feature_names(features, argument="features")
        if len(features) == 0:
            raise ValueError("features must contain at least one feature.")
        positions = _exact_column_positions(self.feature_names_in_, features)
        if np.any(positions < 0):
            missing = [features[i] for i in np.flatnonzero(positions < 0)]
            raise ValueError(
                "features must reference fitted feature names; "
                f"missing: {missing[:5]}"
            )

        import matplotlib.pyplot as plt

        n_features = len(features)
        ncols = min(4, n_features)
        nrows = int(np.ceil(n_features / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for i, (feat, idx) in enumerate(zip(features, positions)):
            ax = axes[i]
            coefs = self.coef_bootstrap_[:, idx]
            ax.hist(coefs, bins=20, edgecolor='white', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'{feat}\nfreq={self.selection_frequencies_[idx]:.2f}', fontsize=9)
            ax.set_xlabel('Coefficient')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _validate_runtime_params(self) -> None:
        """Validate estimator options before fitting or threshold tuning."""
        if self.task not in ('regression', 'classification'):
            raise ValueError(
                f"task must be 'regression' or 'classification', got '{self.task}'"
            )

        if not isinstance(self.n_bootstrap, numbers.Integral) or self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be a positive integer.")

        if not isinstance(self.sample_frac, numbers.Real) or not (0 < float(self.sample_frac) <= 1):
            raise ValueError("sample_frac must be in (0, 1].")

        if not isinstance(self.threshold, numbers.Real) or not (0 <= float(self.threshold) <= 1):
            raise ValueError("threshold must be in [0, 1].")

        if self.block_method not in ("moving", "circular", "stationary"):
            raise ValueError(
                "block_method must be one of 'moving', 'circular', or 'stationary'. "
                f"Got '{self.block_method}'."
            )

        if self.parallel_backend is not None and self.parallel_backend not in ("threads", "processes"):
            raise ValueError(
                "parallel_backend must be one of 'threads', 'processes', or None. "
                f"Got '{self.parallel_backend}'."
            )

        if self.max_features is not None:
            if not isinstance(self.max_features, numbers.Integral) or self.max_features <= 0:
                raise ValueError("max_features must be a positive integer or None.")

        if self.block_size != "auto" and (
            not isinstance(self.block_size, numbers.Integral) or self.block_size <= 0
        ):
            raise ValueError("block_size must be a positive integer or 'auto'.")

        if not isinstance(self.coef_threshold, numbers.Real) or self.coef_threshold < 0:
            raise ValueError("coef_threshold must be non-negative.")

        if self.alpha is not None and (
            not isinstance(self.alpha, numbers.Real) or self.alpha <= 0
        ):
            raise ValueError("alpha must be positive when provided.")

        if self.alpha_rule not in {"one_se", "best"}:
            raise ValueError("alpha_rule must be 'one_se' or 'best'.")

        if not isinstance(self.l1_ratio, numbers.Real) or not (0 <= float(self.l1_ratio) <= 1):
            raise ValueError("l1_ratio must be in [0, 1].")

    def _impute_with_fit_stats(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        """Mean-impute using fit-time statistics, optionally storing them."""
        X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.where(np.isfinite(X_arr), X_arr, np.nan)
        if not np.isnan(X_arr).any():
            if fit:
                self._impute_means_ = np.mean(X_arr, axis=0).astype(np.float32, copy=False)
                self._impute_means_ = np.where(np.isfinite(self._impute_means_), self._impute_means_, 0.0)
            return X_arr

        if fit:
            with np.errstate(all="ignore"):
                means = np.nanmean(X_arr, axis=0)
            self._impute_means_ = np.where(np.isfinite(means), means, 0.0).astype(np.float32, copy=False)
        elif not hasattr(self, "_impute_means_"):
            raise ValueError("Imputation statistics are unavailable. Call fit() before transforming.")

        means = np.asarray(self._impute_means_, dtype=X_arr.dtype)
        mask = ~np.isfinite(X_arr)
        if mask.any():
            X_arr = X_arr.copy()
            X_arr[mask] = np.nan
            row_idx, col_idx = np.where(mask)
            X_arr[row_idx, col_idx] = means[col_idx]
        return X_arr

    def _prep_arrays(
        self,
        X,
        y,
        sample_weight,
        feature_names
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Hashable]]:
        """Convert inputs to arrays, extract feature names."""
        exclude = set()
        if self.use_smart_sampler and self.sampler_config:
            if self.sampler_config.group_col:
                exclude.add(self.sampler_config.group_col)
            if self.sampler_config.time_col:
                exclude.add(self.sampler_config.time_col)

        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = [c for c in X.columns if c not in exclude]
            positions = _exact_column_positions(X.columns, feature_names)
            if np.any(positions < 0):
                missing = [
                    feature_names[i] for i in np.flatnonzero(positions < 0)
                ]
                raise ValueError(
                    "feature_names must reference existing DataFrame columns; "
                    f"missing: {missing[:5]}"
                )
            feature_frame = X.iloc[:, positions]
            reject_datetime_like_features(feature_frame)
            X = feature_frame.values
        else:
            reject_datetime_like_features(X)
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X must be a 2-dimensional array-like object")
            if feature_names is None:
                feature_names = [f"x{i}" for i in range(X.shape[1])]

        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names has {len(feature_names)} entries but X has "
                f"{X.shape[1]} features"
            )

        if isinstance(y, pd.Series):
            y = y.values

        y_raw = np.asarray(y)

        X = np.asarray(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, np.nan)
        if np.isnan(X).any():
            X = X.copy()

        # Handle labels properly for classification
        if self.task == 'classification':
            if pd.isna(y_raw).any():
                raise ValueError("Missing labels are not allowed for classification.")
            # Use LabelEncoder to handle string/categorical labels
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y_raw).astype(np.int32)
            self.classes_ = self._label_encoder.classes_
        else:
            if not np.isfinite(y_raw).all():
                raise ValueError("Target values must be finite for regression.")
            y = y_raw.astype(np.float64)

        sample_weight = ensure_weights(sample_weight, len(y), normalize=True).astype(np.float32)

        return X, y, sample_weight, feature_names

    def _apply_smart_sampler(
        self,
        X,
        y,
        sample_weight=None,
        feature_names=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Hashable]]:
        """Apply smart sampler to reduce data size."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("use_smart_sampler=True requires X to be a DataFrame")

        # Don't allow user sample_weight with smart sampler - they conflict
        if sample_weight is not None:
            raise ValueError(
                "Cannot use both sample_weight and use_smart_sampler=True. "
                "The smart sampler generates its own weights. Either pass sample_weight "
                "with use_smart_sampler=False, or let the smart sampler generate weights."
            )

        config = replace(self.sampler_config) if self.sampler_config is not None else SmartSamplerConfig()
        # Fix: use `is None` check to handle random_state=0
        if config.random_state is None:
            config.random_state = self.random_state if self.random_state is not None else 42
        config.verbose = self.verbose

        # For classification, disable residual-based sampling (regression on class IDs is meaningless)
        # Use geometry-only sampling (leverage + uniform floor + anchors)
        if self.task == 'classification':
            config.residual_weight_cap = 0.0

        exclude = set()
        if config.group_col:
            exclude.add(config.group_col)
        if config.time_col:
            exclude.add(config.time_col)
        if feature_names is None:
            candidate_cols = [c for c in X.columns if c not in exclude]
        else:
            # An explicit feature_names list is a feature-subset contract. Keep
            # its order, retain the sampler's group/time exclusions, and never
            # widen it back to every numeric DataFrame column.
            candidate_cols = [c for c in feature_names if c not in exclude]
        positions = _exact_column_positions(X.columns, candidate_cols)
        if np.any(positions < 0):
            missing = [candidate_cols[i] for i in np.flatnonzero(positions < 0)]
            raise ValueError(
                "feature_names for use_smart_sampler must reference existing "
                f"DataFrame columns; missing: {missing[:5]}"
            )
        candidate_frame = X.iloc[:, positions]
        reject_datetime_like_features(candidate_frame)
        if feature_names is not None:
            non_numeric = candidate_frame.select_dtypes(
                exclude=[np.number]
            ).columns.tolist()
            if non_numeric:
                raise ValueError(
                    "feature_names for use_smart_sampler must reference numeric "
                    f"columns; non-numeric: {non_numeric[:5]}"
                )
        feature_names = candidate_frame.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        dropped = [c for c in candidate_cols if c not in feature_names]
        if dropped:
            warnings.warn(
                f"Smart sampler uses numeric features only; dropping {len(dropped)} non-numeric column(s): "
                f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}",
                UserWarning,
                stacklevel=4,
            )
        if not feature_names:
            raise ValueError("No numeric feature columns available for smart sampling.")

        # Build df with encoded y BEFORE sampling
        df = X.copy()

        if isinstance(y, pd.Series):
            y_raw = y.values
        else:
            y_raw = np.asarray(y)

        if self.task == 'classification':
            if pd.isna(y_raw).any():
                raise ValueError("Missing labels are not allowed for classification.")
            # Encode labels BEFORE sampling so string labels work
            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y_raw).astype(np.int32)
            self.classes_ = self._label_encoder.classes_
            y_col = '_y_enc'
            df[y_col] = y_enc
        else:
            if not np.isfinite(y_raw).all():
                raise ValueError("Target values must be finite for regression.")
            y_col = '_y'
            df[y_col] = y_raw.astype(np.float64)

        sampled = smart_sample(
            df=df,
            feature_cols=feature_names,
            y_col=y_col,
            config=config
        )

        X_out = sampled[feature_names].values.astype(np.float32)
        weights_out = sampled['sample_weight'].values.astype(np.float32)

        if self.task == 'classification':
            y_out = sampled[y_col].values.astype(np.int32)

            # Check that all classes survived sampling
            present_classes = np.unique(y_out)
            if len(present_classes) != len(self.classes_):
                missing = set(range(len(self.classes_))) - set(present_classes)
                missing_labels = [self.classes_[i] for i in missing]
                raise ValueError(
                    f"Smart sampler dropped class(es): {missing_labels}. "
                    f"Increase sample_frac or disable use_smart_sampler for classification."
                )
        else:
            y_out = sampled[y_col].values.astype(np.float64)

        self.sampled_n_ = len(sampled)

        return X_out, y_out, weights_out, feature_names

    def _find_alpha(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ) -> float:
        """Estimate alpha via CV on subsample."""
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=min(30_000, n), replace=False)
        if time is not None:
            if groups is None:
                idx = idx[np.argsort(np.asarray(time)[idx], kind="mergesort")]
            else:
                groups_sub = np.asarray(groups)[idx]
                if len(np.unique(groups_sub)) < 2:
                    idx = idx[np.argsort(np.asarray(time)[idx], kind="mergesort")]
        cv = self._alpha_cv(idx, y, groups, time)

        if self.task == 'classification':
            model = LogisticRegression(
                penalty='l1',
                solver='saga',
                tol=1e-3,
                max_iter=2000,
                random_state=self.random_state,
                n_jobs=1,
            )
            # Preserve the sparse-selection contract of the prior
            # LogisticRegressionCV path while keeping imputation/scaling local
            # to every fold. Accuracy also selects the strongest penalty among
            # tied grid points because C is ordered from small to large.
            param_grid = {"model__C": np.logspace(-4, 4, 20)}
            scoring = "accuracy"
        else:
            y_scale = float(np.std(np.asarray(y)[idx]))
            if not np.isfinite(y_scale) or y_scale <= 0.0:
                y_scale = np.finfo(np.float64).tiny * 1e4
            alpha_grid = y_scale * np.logspace(-4, 0, 30)
            if self.l1_ratio >= 1.0:
                model = Lasso(max_iter=2000)
            else:
                model = ElasticNet(l1_ratio=self.l1_ratio, max_iter=2000)
            param_grid = {"model__alpha": alpha_grid}
            scoring = "neg_mean_squared_error"

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score=np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(
                np.asarray(X)[idx],
                np.asarray(y)[idx],
                model__sample_weight=np.asarray(sample_weight)[idx],
            )

        if self.task == 'classification':
            self._alpha_ref_weight_ = self._cv_train_weight(cv, idx, sample_weight)
            parameter = "model__C"
            best_value = float(search.best_params_[parameter])
            if self.alpha_rule == "one_se":
                best_value = self._one_se_parameter(search, parameter, prefer="smaller")
            return 1.0 / best_value
        parameter = "model__alpha"
        best_value = float(search.best_params_[parameter])
        if self.alpha_rule == "one_se":
            best_value = self._one_se_parameter(search, parameter, prefer="larger")
        return best_value

    @staticmethod
    def _one_se_parameter(search, parameter: str, *, prefer: str) -> float:
        """Return the strongest parameter within one SE of the best CV score."""
        results = search.cv_results_
        means = np.asarray(results["mean_test_score"], dtype=np.float64)
        stds = np.asarray(results["std_test_score"], dtype=np.float64)
        finite = np.isfinite(means) & np.isfinite(stds)
        if not bool(finite.any()):
            return float(search.best_params_[parameter])
        best_idx = int(np.nanargmax(np.where(finite, means, -np.inf)))
        n_splits = max(int(getattr(search, "n_splits_", 1)), 1)
        cutoff = means[best_idx] - stds[best_idx] / np.sqrt(n_splits)
        eligible = np.flatnonzero(finite & (means >= cutoff))
        values = np.asarray(
            [params[parameter] for params in results["params"]],
            dtype=np.float64,
        )
        if prefer == "larger":
            chosen = int(eligible[np.argmax(values[eligible])])
        elif prefer == "smaller":
            chosen = int(eligible[np.argmin(values[eligible])])
        else:
            raise ValueError("prefer must be 'larger' or 'smaller'")
        return float(values[chosen])

    @staticmethod
    def _cv_train_weight(cv, idx: np.ndarray, sample_weight: np.ndarray) -> float:
        """Mean total sample weight of the CV training folds used for alpha search."""
        w_idx = np.asarray(sample_weight, dtype=np.float64)[idx]
        if isinstance(cv, (int, np.integer)):
            return float(w_idx.sum()) * (int(cv) - 1) / int(cv)
        totals = [float(w_idx[np.asarray(train, dtype=np.int64)].sum()) for train, _ in cv]
        return float(np.mean(totals)) if totals else float(w_idx.sum())

    def _classification_C(self, train_weight: np.ndarray) -> float:
        """Bootstrap-fit C keeping the CV-calibrated per-sample regularization."""
        C = 1.0 / self.alpha_
        ref = getattr(self, "_alpha_ref_weight_", None)
        if ref is None or not np.isfinite(ref) or ref <= 0.0:
            return C
        total = float(np.sum(train_weight))
        if not np.isfinite(total) or total <= 0.0:
            return C
        return C * ref / total

    def _alpha_cv(self, idx: np.ndarray, y: np.ndarray, groups, time):
        if groups is not None:
            groups_sub = np.asarray(groups)[idx]
            n_groups = len(np.unique(groups_sub))
            if n_groups >= 2:
                splitter = GroupKFold(n_splits=min(3, n_groups))
                return list(splitter.split(np.zeros((len(idx), 1)), y[idx], groups_sub))
        if time is not None and len(idx) >= 4:
            return list(TimeSeriesSplit(n_splits=min(3, len(idx) - 1)).split(idx))
        return 3


def stability_select(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.6,
    n_bootstrap: int = 50,
    *,
    callback: ProgressCallback | None = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quick stability selection returning selected indices and frequencies."""
    selector = StabilitySelector(
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        callback=callback,
        **kwargs,
    )
    selector.fit(X, y)
    return selector.selected_features_, selector.selection_frequencies_


def _stability_task_features(
    task: str,
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    k: int,
    **kwargs,
) -> Union[List[Hashable], List[int]]:
    sample_weight = kwargs.pop("sample_weight", None)
    groups = kwargs.pop("groups", None)
    time = kwargs.pop("time", None)
    return_indices = kwargs.pop("return_indices", None)
    kwargs["task"] = task
    kwargs["max_features"] = k

    selector = StabilitySelector(**kwargs)
    selector.fit(X, y, sample_weight=sample_weight, groups=groups, time=time)

    if return_indices is None:
        return_indices = not isinstance(X, pd.DataFrame)
    if return_indices:
        return selector.selected_features_.tolist()
    return selector.selected_feature_names_


def stability_regression(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    k: int,
    *,
    callback: ProgressCallback | None = None,
    **kwargs,
) -> Union[List[Hashable], List[int]]:
    """Stability selection for regression.

    Returns up to ``k`` features whose selection frequency clears ``threshold``
    (``k`` caps the count via ``max_features``; it is not an exact-size
    guarantee). Never-selected features are not used to pad the result.
    """
    return _stability_task_features(
        "regression", X, y, k, callback=callback, **kwargs
    )


def stability_classif(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    k: int,
    *,
    callback: ProgressCallback | None = None,
    **kwargs,
) -> Union[List[Hashable], List[int]]:
    """Stability selection for classification.

    Returns up to ``k`` features whose selection frequency clears ``threshold``
    (``k`` caps the count via ``max_features``; it is not an exact-size
    guarantee). Never-selected features are not used to pad the result.
    """
    return _stability_task_features(
        "classification", X, y, k, callback=callback, **kwargs
    )
