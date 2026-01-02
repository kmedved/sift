import numpy as np
import pandas as pd
from typing import Iterator, Optional, List, Tuple, Union
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import (
    Lasso, LassoCV, ElasticNet, ElasticNetCV,
    LogisticRegression, LogisticRegressionCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import Parallel, delayed

from sift.sampling.smart import SmartSamplerConfig, smart_sample


# =============================================================================
# Stability Selector
# =============================================================================

def _bootstrap_indices(
    n: int,
    n_bootstrap: int,
    sample_frac: float,
    y: np.ndarray | None = None,
    task: str = "regression",
    random_state: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    subsample_size = max(2, int(n * sample_frac))
    subsample_size = min(subsample_size, n)

    is_classification = task == "classification" and y is not None
    if is_classification:
        classes = np.unique(y)
        n_classes = len(classes)
        class_indices = {c: np.where(y == c)[0] for c in classes}
        class_counts = np.array([len(class_indices[c]) for c in classes])
        if subsample_size < n_classes:
            subsample_size = n_classes

        def stratified_indices(rng_local):
            props = class_counts / class_counts.sum()
            raw = props * subsample_size
            counts = np.floor(raw).astype(int)
            counts = np.maximum(counts, 1)
            counts = np.minimum(counts, class_counts)

            total = counts.sum()
            frac = raw - np.floor(raw)

            if total < subsample_size:
                need = subsample_size - total
                room = class_counts - counts
                order = np.argsort(-frac)
                for j in order:
                    if need == 0:
                        break
                    if room[j] > 0:
                        add = min(room[j], need)
                        counts[j] += add
                        need -= add
            elif total > subsample_size:
                extra = total - subsample_size
                order = np.argsort(-counts)
                for j in order:
                    if extra == 0:
                        break
                    can_drop = counts[j] - 1
                    if can_drop > 0:
                        drop = min(can_drop, extra)
                        counts[j] -= drop
                        extra -= drop

            idx_list = [
                rng_local.choice(class_indices[c], size=counts[i], replace=False)
                for i, c in enumerate(classes)
                if counts[i] > 0
            ]
            return np.concatenate(idx_list)

    for _ in range(n_bootstrap):
        if is_classification:
            train_idx = stratified_indices(rng)
        else:
            train_idx = rng.choice(n, size=subsample_size, replace=False)

        in_bag = np.zeros(n, dtype=bool)
        in_bag[train_idx] = True
        val_idx = np.flatnonzero(~in_bag)
        yield train_idx.astype(np.int64), val_idx.astype(np.int64)


def _block_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    groups: np.ndarray,
    time: np.ndarray,
    block_size: int | str = "auto",
    block_method: str = "moving",
    y: np.ndarray | None = None,
    task: str = "regression",
    random_state: int | None = None,
    min_oob: int = 10,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Block bootstrap respecting group/time structure.

    Parameters
    ----------
    groups : array
        Group labels (e.g., player_id).
    time : array
        Time values for ordering within groups.
    block_size : int or "auto"
        "auto" uses sqrt(n_per_group).
    block_method : str
        "moving", "circular", or "stationary"
    """
    rng = np.random.default_rng(random_state)

    unique_groups = np.unique(groups)
    group_data = {}
    for g in unique_groups:
        mask = groups == g
        idx = np.where(mask)[0]
        order = np.argsort(time[idx])
        group_data[g] = idx[order]

    classes = set(np.unique(y)) if task == "classification" and y is not None else None

    valid = 0
    attempts = 0
    max_attempts = n_bootstrap * 10

    while valid < n_bootstrap and attempts < max_attempts:
        attempts += 1
        train_idx = []
        val_idx = []

        for g, sorted_idx in group_data.items():
            n_g = len(sorted_idx)
            if n_g == 0:
                continue

            bs = int(np.sqrt(n_g)) if block_size == "auto" else min(block_size, n_g)
            bs = max(1, bs)

            if block_method == "moving":
                in_bag = _moving_block_sample(sorted_idx, bs, n_g, rng)
            elif block_method == "circular":
                in_bag = _circular_block_sample(sorted_idx, bs, n_g, rng)
            elif block_method == "stationary":
                in_bag = _stationary_block_sample(sorted_idx, bs, n_g, rng)
            else:
                raise ValueError(f"Unknown block_method: {block_method}")

            in_bag_set = set(in_bag)
            oob = [i for i in sorted_idx if i not in in_bag_set]

            train_idx.extend(in_bag)
            val_idx.extend(oob)

        train_arr = np.array(train_idx, dtype=np.int64)
        val_arr = np.array(val_idx, dtype=np.int64)

        if len(val_arr) < min_oob:
            continue

        if classes is not None:
            if set(y[train_arr]) != classes or len(set(y[val_arr])) < 2:
                continue

        valid += 1
        yield train_arr, val_arr

    if valid < n_bootstrap:
        warnings.warn(f"Only generated {valid}/{n_bootstrap} valid block bootstrap splits.")


def _moving_block_sample(sorted_idx: np.ndarray, block_size: int, n: int, rng) -> list[int]:
    n_blocks = max(1, int(np.ceil(n / block_size)))
    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, max(1, n - block_size + 1))
        result.extend(sorted_idx[start:start + block_size].tolist())
    return result


def _circular_block_sample(sorted_idx: np.ndarray, block_size: int, n: int, rng) -> list[int]:
    n_blocks = max(1, int(np.ceil(n / block_size)))
    result = []
    for _ in range(n_blocks):
        start = rng.integers(0, n)
        indices = [(start + i) % n for i in range(block_size)]
        result.extend(sorted_idx[indices].tolist())
    return result


def _stationary_block_sample(sorted_idx: np.ndarray, mean_block_size: int, n: int, rng) -> list[int]:
    result = []
    p = 1.0 / max(1, mean_block_size)
    while len(result) < n:
        start = rng.integers(0, n)
        length = min(rng.geometric(p), n - start)
        result.extend(sorted_idx[start:start + length].tolist())
    return result[:n]


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

    Attributes
    ----------
    selection_frequencies_ : ndarray of shape (n_features,)
        Fraction of bootstrap runs in which each feature was selected.
    selected_features_ : ndarray
        Indices of selected features.
    selected_feature_names_ : list of str
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
        verbose: bool = True
    ):
        self.n_bootstrap = n_bootstrap
        self.sample_frac = sample_frac
        self.threshold = threshold
        self.alpha = alpha
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

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
        groups: np.ndarray | None = None,
        time: np.ndarray | None = None,
        feature_names: Optional[List[str]] = None
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
        feature_names : list of str, optional
            Feature names.

        Returns
        -------
        self
        """
        # Input validation
        if self.task not in ('regression', 'classification'):
            raise ValueError(f"task must be 'regression' or 'classification', got '{self.task}'")
        if self.use_smart_sampler and (groups is not None or time is not None):
            raise ValueError("groups/time are not supported when use_smart_sampler=True.")
        if self.use_smart_sampler:
            X, y, sample_weight, feature_names = self._apply_smart_sampler(X, y, sample_weight)
        else:
            X, y, sample_weight, feature_names = self._prep_arrays(
                X, y, sample_weight, feature_names
            )

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

        # Impute non-finite values (smart_sample may return original rows with NaNs)
        if not np.isfinite(X).all():
            from sift._impute import mean_impute

            X = mean_impute(X, copy=False)

        # Standardize
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Get alpha
        if self.alpha is None:
            self.alpha_ = self._find_alpha(X_scaled, y, sample_weight)
        else:
            self.alpha_ = self.alpha

        if self.verbose:
            task_str = 'classification' if self.task == 'classification' else 'regression'
            print(f"Stability selection ({task_str}): {self.n_bootstrap} bootstraps, "
                  f"α={self.alpha_:.4f}, threshold={self.threshold}")

        alpha = self.alpha_
        l1_ratio = self.l1_ratio
        task = self.task
        coef_threshold = self.coef_threshold

        use_block = groups is not None and time is not None
        if use_block:
            if self.verbose:
                print(f"Using block bootstrap (method={self.block_method}, size={self.block_size})")
            splits = list(_block_bootstrap_indices(
                n=n,
                n_bootstrap=self.n_bootstrap,
                groups=groups,
                time=time,
                block_size=self.block_size,
                block_method=self.block_method,
                y=y if self.task == "classification" else None,
                task=self.task,
                random_state=self.random_state,
            ))
        else:
            if self.verbose:
                print("Using i.i.d. bootstrap")
            splits = list(_bootstrap_indices(
                n=n,
                n_bootstrap=self.n_bootstrap,
                sample_frac=self.sample_frac,
                y=y if self.task == "classification" else None,
                task=self.task,
                random_state=self.random_state,
            ))

        if not splits:
            raise ValueError("No valid bootstrap splits could be generated.")

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31, size=len(splits))

        def single_run(train_idx, seed):
            idx = train_idx

            if task == 'classification':
                # C = 1/alpha for LogisticRegression
                model = LogisticRegression(
                    penalty='l1',
                    solver='saga',
                    C=1.0 / alpha,
                    max_iter=3000,
                    random_state=seed,  # Reproducibility
                    n_jobs=1  # Avoid nested parallelism
                )
            elif l1_ratio >= 1.0:
                model = Lasso(alpha=alpha, max_iter=3000)
            else:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=3000)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_scaled[idx], y[idx], sample_weight=sample_weight[idx])

            coef = model.coef_

            # Handle coefficient shapes
            if coef.ndim == 2:
                if coef.shape[0] == 1:
                    # Binary classification: preserve sign
                    coef_flat = coef[0]
                    selected = np.abs(coef_flat) > coef_threshold
                    coef_summary = coef_flat  # Signed coefficients
                else:
                    # True multiclass: aggregate across classes
                    selected = np.any(np.abs(coef) > coef_threshold, axis=0)
                    # For multiclass, take max abs (sign is ambiguous)
                    coef_summary = np.max(np.abs(coef), axis=0)
            else:
                # Regression
                selected = np.abs(coef) > coef_threshold
                coef_summary = coef.ravel()

            return selected.astype(np.int8), coef_summary.astype(np.float32)

        # Chunked execution to reduce peak memory
        # (avoids holding all n_bootstrap results in memory at once)
        n_splits = len(splits)
        chunk_size = min(20, n_splits)  # Process 20 at a time

        sel_count = np.zeros(p, dtype=np.int32)
        sum_abs_coef = np.zeros(p, dtype=np.float64)

        if self.store_coefs:
            self.coef_bootstrap_ = np.empty((n_splits, p), dtype=np.float32)

        bootstrap_idx = 0
        for chunk_start in range(0, n_splits, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_splits)
            chunk_splits = splits[chunk_start:chunk_end]
            chunk_seeds = seeds[chunk_start:chunk_end]

            chunk_results = Parallel(n_jobs=self.n_jobs, prefer=self.parallel_backend)(
                delayed(single_run)(train_idx, seed)
                for (train_idx, _), seed in zip(chunk_splits, chunk_seeds)
            )

            # Aggregate this chunk immediately, then discard
            for selected, coef_summary in chunk_results:
                sel_count += selected.astype(np.int32)
                sum_abs_coef += np.abs(coef_summary)

                if self.store_coefs:
                    self.coef_bootstrap_[bootstrap_idx] = coef_summary
                bootstrap_idx += 1

            # chunk_results goes out of scope here, memory freed

        self.selection_frequencies_ = (sel_count / n_splits).astype(np.float32)
        self.mean_abs_coef_ = (sum_abs_coef / n_splits).astype(np.float32)

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
            print(f"Selected {self.n_features_selected_} / {p} features")

        return self

    def _prep_arrays(
        self,
        X,
        y,
        sample_weight,
        feature_names
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Convert inputs to arrays, extract feature names."""
        exclude = set()
        if self.use_smart_sampler and self.sampler_config:
            if self.sampler_config.group_col:
                exclude.add(self.sampler_config.group_col)
            if self.sampler_config.time_col:
                exclude.add(self.sampler_config.time_col)

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or [c for c in X.columns if c not in exclude]
            X = X[feature_names].values
        else:
            feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

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
            y = y_raw.astype(np.float32)

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float32)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)

        return X, y, sample_weight, feature_names

    def _apply_smart_sampler(
        self,
        X,
        y,
        sample_weight=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
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

        config = self.sampler_config or SmartSamplerConfig()
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
        candidate_cols = [c for c in X.columns if c not in exclude]
        feature_names = X[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()
        dropped = [c for c in candidate_cols if c not in feature_names]
        if dropped:
            warnings.warn(
                f"Smart sampler uses numeric features only; dropping {len(dropped)} non-numeric column(s): "
                f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}"
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
            df[y_col] = y_raw.astype(np.float32)

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
            y_out = sampled[y_col].values.astype(np.float32)

        self.sampled_n_ = len(sampled)

        return X_out, y_out, weights_out, feature_names

    def _find_alpha(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray
    ) -> float:
        """Estimate alpha via CV on subsample."""
        n = X_scaled.shape[0]
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(n, size=min(30_000, n), replace=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.task == 'classification':
                # LogisticRegressionCV uses C (inverse of alpha)
                # We search over C values and convert back
                cv_model = LogisticRegressionCV(
                    penalty='l1',
                    solver='saga',
                    cv=3,
                    Cs=20,
                    max_iter=2000,
                    random_state=self.random_state,
                    n_jobs=1
                )
                cv_model.fit(X_scaled[idx], y[idx], sample_weight=sample_weight[idx])

                # C_ can be scalar or per-class array for multiclass
                C = cv_model.C_
                if np.ndim(C) > 0 and len(C) > 1:
                    C_best = float(np.mean(C))  # Average across classes
                else:
                    C_best = float(C[0]) if np.ndim(C) > 0 else float(C)
                return 1.0 / C_best
            elif self.l1_ratio >= 1.0:
                cv_model = LassoCV(cv=3, n_alphas=30, max_iter=2000)
            else:
                cv_model = ElasticNetCV(
                    l1_ratio=self.l1_ratio, cv=3, n_alphas=30, max_iter=2000
                )

            cv_model.fit(X_scaled[idx], y[idx], sample_weight=sample_weight[idx])

        return cv_model.alpha_

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Reduce X to selected features."""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_feature_names_].values
        return np.asarray(X)[:, self.selected_features_]

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **fit_params
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get DataFrame with feature selection details.

        Returns
        -------
        DataFrame with columns:
            feature: name
            frequency: selection frequency
            mean_abs_coef: mean absolute coefficient across bootstraps
            selected: whether it passed threshold
        """
        return pd.DataFrame({
            'feature': self.feature_names_in_,
            'frequency': self.selection_frequencies_,
            'mean_abs_coef': self.mean_abs_coef_,
            'selected': self.selection_frequencies_ >= self.threshold
        }).sort_values('frequency', ascending=False).reset_index(drop=True)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        if indices:
            return self.selected_features_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def get_coef_stability(self) -> pd.DataFrame:
        """
        Get coefficient stability analysis.

        Returns DataFrame with mean, std, and CV of coefficients
        across bootstrap runs for each feature.

        Note on coefficient semantics:
        - Regression: signed coefficients (can be positive or negative)
        - Binary classification: signed coefficients (positive = class 1)
        - Multiclass classification: absolute coefficients (max abs across classes)

        Requires store_coefs=True (default).
        """
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
        thresholds: List[float] = [0.4, 0.5, 0.6, 0.7, 0.8],
        cv: int = 3,
        scoring: Optional[str] = None
    ) -> Tuple[float, pd.DataFrame]:
        """
        Find optimal threshold by cross-validating downstream model performance.

        Must be called after fit(). Tests each threshold and evaluates
        ElasticNet (regression) or LogisticRegression (classification)
        performance on the selected feature subset.

        Parameters
        ----------
        X : array-like or DataFrame
            Training data (same as used in fit, or held-out).
        y : array-like
            Target values.
        thresholds : list of float
            Threshold values to test.
        cv : int
            Number of cross-validation folds.
        scoring : str, optional
            Scoring metric. Default: 'r2' for regression, 'accuracy' for classification.

        Returns
        -------
        best_threshold : float
            Threshold with highest CV score.
        results : DataFrame
            Threshold, n_features, mean_score, std_score for each threshold tested.
        """
        from sklearn.model_selection import cross_val_score

        if not hasattr(self, 'selection_frequencies_'):
            raise ValueError("Must call fit() before tune_threshold()")

        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names_in_].values
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Scale X
        X_scaled = self._scaler.transform(X)

        # Default scoring
        if scoring is None:
            scoring = 'accuracy' if self.task == 'classification' else 'r2'

        results = []
        for thresh in thresholds:
            mask = self.selection_frequencies_ >= thresh
            n_selected = mask.sum()

            if n_selected == 0:
                results.append({
                    'threshold': thresh,
                    'n_features': 0,
                    'mean_score': np.nan,
                    'std_score': np.nan
                })
                continue

            X_subset = X_scaled[:, mask]

            # Fit downstream model
            if self.task == 'classification':
                model = LogisticRegression(
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=1000
                )
            else:
                model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring)

            results.append({
                'threshold': thresh,
                'n_features': n_selected,
                'mean_score': scores.mean(),
                'std_score': scores.std()
            })

        results_df = pd.DataFrame(results)

        # Find best (highest mean score, ignoring NaN)
        valid = results_df.dropna(subset=['mean_score'])
        if len(valid) == 0:
            best_thresh = thresholds[0]
        else:
            best_idx = valid['mean_score'].idxmax()
            best_thresh = valid.loc[best_idx, 'threshold']

        if self.verbose:
            print(f"Threshold tuning results (scoring={scoring}):")
            print(results_df.to_string(index=False))
            print(f"Best threshold: {best_thresh}")

        return best_thresh, results_df

    def set_threshold(self, threshold: float) -> 'StabilitySelector':
        """
        Update threshold and recompute selected features.

        Useful after tune_threshold() to apply the optimal threshold.

        Parameters
        ----------
        threshold : float
            New threshold value.

        Returns
        -------
        self
        """
        if not hasattr(self, 'selection_frequencies_'):
            raise ValueError("Must call fit() before set_threshold()")

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
            print(f"Updated threshold to {threshold}: {self.n_features_selected_} features selected")

        return self

    def plot_frequencies(
        self,
        top_n: int = 50,
        figsize: Optional[Tuple[float, float]] = None,
        show_coef: bool = False
    ):
        """
        Bar plot of selection frequencies.

        Parameters
        ----------
        top_n : int
            Number of top features to show.
        figsize : tuple, optional
            Figure size.
        show_coef : bool
            If True, show mean coefficient as bar color intensity.
        """
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
        ax.axvline(
            self.threshold,
            color='red',
            linestyle='--',
            label=f'threshold={self.threshold}'
        )
        ax.set_xlabel('Selection Frequency')
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        ax.set_title(f'Stability Selection ({self.n_features_selected_} features selected)')
        plt.tight_layout()

        return fig, ax

    def plot_coef_distributions(self, features: Optional[List[str]] = None, top_n: int = 12):
        """
        Plot coefficient distributions across bootstrap runs.

        Parameters
        ----------
        features : list of str, optional
            Specific features to plot. If None, uses top_n by frequency.
        top_n : int
            Number of top features if features not specified.

        Requires store_coefs=True (default).
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, 'coef_bootstrap_'):
            raise ValueError(
                "Coefficient matrix not available. "
                "Set store_coefs=True when creating the selector."
            )

        if features is None:
            info = self.get_feature_info()
            features = info['feature'].head(top_n).tolist()

        n_features = len(features)
        ncols = min(4, n_features)
        nrows = int(np.ceil(n_features / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            idx = self.feature_names_in_.index(feat)
            coefs = self.coef_bootstrap_[:, idx]

            ax.hist(coefs, bins=20, edgecolor='white', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'{feat}\nfreq={self.selection_frequencies_[idx]:.2f}', fontsize=9)
            ax.set_xlabel('Coefficient')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig, axes


# =============================================================================
# Convenience Functions
# =============================================================================

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
