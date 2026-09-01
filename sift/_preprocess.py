"""Input preprocessing: validation, conversion, subsampling, encoding."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EstimatorJMI = Literal["auto", "binned", "r2", "ksg", "gaussian"]
EstimatorMRMR = Literal["classic", "gaussian"]
RelevanceMethod = Literal["f", "ks", "rf"]
CatEncoding = Literal[
    "none",
    "target_cv",
    "target",
    "loo",
    "james_stein",
    "loo_logit",
]
Formula = Literal["quotient", "difference"]
Task = Literal["regression", "classification"]

HIGHER_IS_BETTER = frozenset(
    {
        "AUC",
        "ACCURACY",
        "R2",
        "F1",
        "F1_WEIGHTED",
        "F1_MACRO",
        "ROC_AUC",
        "PRECISION",
        "RECALL",
        "NDCG",
        "MAP",
        "PRAUC",
        "MCC",
        "BALANCEDACCURACY",
    }
)
LOWER_IS_BETTER = frozenset(
    {
        "RMSE",
        "MAE",
        "LOGLOSS",
        "MSE",
        "MAPE",
        "SMAPE",
        "MULTICLASS",
        "MULTICLASSONEVSALL",
        "CROSSENTROPY",
    }
)


def infer_higher_is_better(metric: str) -> bool:
    """Infer whether higher metric values are better."""
    metric_upper = metric.upper().split(":")[0]
    if metric_upper in HIGHER_IS_BETTER:
        return True
    if metric_upper in LOWER_IS_BETTER:
        return False
    raise ValueError(
        f"Cannot infer score direction for metric {metric!r}. "
        "Pass higher_is_better=True or False explicitly."
    )


def best_score_from_dict(scores: dict, higher_is_better: bool) -> Tuple[Any, float]:
    """Return best (index, score), preferring the smallest numeric index on ties."""
    if not scores:
        return 0, float("nan")

    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        return 0, float("nan")

    def tie_key(key, position: int) -> tuple[int, float | int]:
        if isinstance(key, (int, float, np.integer, np.floating)) and not isinstance(key, bool):
            key_value = float(key)
            if np.isfinite(key_value):
                return 0, key_value
        return 1, position

    valid_items = list(valid.items())

    def ranking_key(item):
        position, (score_key, score) = item
        order_group, order_value = tie_key(score_key, position)
        if higher_is_better:
            return score, -order_group, -order_value
        return score, order_group, order_value

    if higher_is_better:
        best = max(enumerate(valid_items), key=ranking_key)[1]
    else:
        best = min(enumerate(valid_items), key=ranking_key)[1]
    return best[0], best[1]


# --- Input conversion ---


def validate_task(task: str) -> None:
    """Validate the public selector task before dispatch."""
    if task not in {"regression", "classification"}:
        raise ValueError(
            "task must be 'regression' or 'classification', "
            f"got {task!r}"
        )


def validate_classification_target(y) -> None:
    """Reject continuous or multi-output targets passed as class labels."""
    from sklearn.utils.multiclass import type_of_target

    y_raw = y.values if hasattr(y, "values") else np.asarray(y)
    if pd.isna(y_raw).any():
        raise ValueError("Missing values in y are not allowed for classification.")
    if pd.api.types.is_numeric_dtype(y_raw):
        y_num = np.asarray(y_raw, dtype=np.float64)
        if not np.isfinite(y_num).all():
            raise ValueError("Non-finite values in y are not allowed for classification.")
    target_type = type_of_target(y_raw)
    if target_type not in {"binary", "multiclass"}:
        raise ValueError(
            "Classification y must contain discrete binary or multiclass labels; "
            f"got target type {target_type!r}."
        )


def to_numpy(data, dtype=np.float32) -> np.ndarray:
    """Convert Pandas/Polars/list to numpy array."""
    if hasattr(data, "to_pandas"):
        data = data.to_pandas()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        try:
            return data.to_numpy(dtype=dtype, na_value=np.nan)
        except TypeError:
            arr = data.to_numpy()
            if arr.dtype == object:
                arr = np.where(pd.isna(arr), np.nan, arr)
            return arr.astype(dtype)
    if hasattr(data, "values"):
        return np.asarray(data.values, dtype=dtype)
    return np.asarray(data, dtype=dtype)


def extract_feature_names(X) -> Optional[List[str]]:
    """Extract column names from DataFrame, or None for ndarray."""
    if hasattr(X, "columns"):
        return list(X.columns)
    return None


def _is_temporal_feature_dtype(dtype) -> bool:
    """Recognize NumPy, pandas, and Arrow-backed datetime/duration dtypes."""
    arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    arrow_kind = (
        str(arrow_dtype).partition("[")[0]
        if arrow_dtype is not None
        else None
    )
    return (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        # pandas 2.2 does not classify Arrow duration dtypes as timedelta,
        # but they retain NumPy's temporal dtype kind.
        or getattr(dtype, "kind", None) in {"M", "m"}
        # Arrow time-of-day dtypes instead report object kind.
        or arrow_kind in {"date32", "date64", "duration", "time32", "time64", "timestamp"}
    )


def reject_datetime_like_features(X) -> None:
    """Reject NumPy, pandas, and Arrow temporal features before coercion."""
    temporal_object_types = (
        date,
        datetime,
        timedelta,
        np.datetime64,
        np.timedelta64,
        pd.Timestamp,
        pd.Timedelta,
    )

    def contains_temporal_objects(values) -> bool:
        try:
            flat = np.asarray(values, dtype=object).ravel()
        except (TypeError, ValueError):
            return False
        return any(
            value is pd.NaT or isinstance(value, temporal_object_types)
            for value in flat
        )

    if isinstance(X, pd.DataFrame):
        datetime_like = [
            name
            for i, (name, column_dtype) in enumerate(X.dtypes.items())
            if _is_temporal_feature_dtype(column_dtype)
            or (
                pd.api.types.is_object_dtype(column_dtype)
                and contains_temporal_objects(X.iloc[:, i])
            )
        ]
    else:
        try:
            array = np.asarray(X)
        except (TypeError, ValueError):
            array = None
        dtype = None if array is None else array.dtype
        datetime_like = (
            ["<array>"]
            if dtype is not None
            and (
                _is_temporal_feature_dtype(dtype)
                or (
                    pd.api.types.is_object_dtype(dtype)
                    and contains_temporal_objects(array)
                )
            )
            else []
        )
    if datetime_like:
        sample = datetime_like[:5]
        suffix = "..." if len(datetime_like) > 5 else ""
        raise ValueError(
            "Datetime or timedelta feature columns (including time-of-day values) "
            "are not supported: "
            f"{sample}{suffix}. Convert them to numeric features explicitly."
        )


# --- Validation ---


def ensure_weights(
    sample_weight: np.ndarray | None,
    n: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Validate and normalize sample weights.

    Parameters
    ----------
    sample_weight : array-like or None
        Sample weights. If None, returns uniform weights.
    n : int
        Expected number of samples.
    normalize : bool
        If True, normalize weights to mean=1.

    Returns
    -------
    w : ndarray of shape (n,)
        Validated, non-negative, finite weights.
    """
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)

    w = np.asarray(sample_weight, dtype=np.float64).ravel()

    if w.shape[0] != n:
        raise ValueError(f"sample_weight has {w.shape[0]} elements but expected {n}")
    if not np.isfinite(w).all():
        raise ValueError("sample_weight contains non-finite values")
    if np.any(w < 0):
        raise ValueError("sample_weight contains negative values")
    if not np.any(w > 0):
        raise ValueError("sample_weight must contain at least one positive value")

    if normalize:
        scale = float(np.max(w))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("sample_weight max must be finite and > 0 to normalize")
        w = w / scale
        mean = float(w.mean())
        if not np.isfinite(mean) or mean <= 0.0:
            raise ValueError("sample_weight mean must be finite and > 0 to normalize")
        w = w / mean
        # Quantize normalized weights to float32 precision, then restore their
        # mean in float64. This greatly reduces rescaling-induced ulp changes
        # that can alter tree tie-breaking without claiming exact invariance
        # for every representable input and scale.
        w = w.astype(np.float32).astype(np.float64)
        w /= float(w.mean())

    return w


def validate_k(k, *, allow_auto: bool = True) -> int | Literal["auto"]:
    """Validate a public feature-count argument."""
    if k == "auto":
        if allow_auto:
            return "auto"
        raise ValueError("k='auto' is not supported here")

    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be a positive integer")

    k_int = int(k)
    if k_int < 1:
        raise ValueError("k must be >= 1")
    return k_int


def validate_inputs(
    X, y, task: str, impute: bool = True, *, dtype=np.float64
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Validate and convert inputs.

    ``dtype`` controls the returned feature dtype. The default is float64 so
    classic filters do not lose distinctions between large-offset values (for
    example, values near ``1e8``). Callers that have a deliberate lower
    precision contract can opt into ``np.float32`` explicitly.
    """
    from sift._impute import mean_impute

    validate_task(task)

    feature_names = extract_feature_names(X)
    reject_datetime_like_features(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Either encode them first or set cat_encoding to 'target_cv', "
                "'loo', 'target', 'james_stein', or binary-only 'loo_logit'."
            )
    X_arr = to_numpy(X, dtype=np.float64)

    if impute:
        X_arr = mean_impute(X_arr, copy=True)

    X_arr = X_arr.astype(dtype, copy=False)

    if task == "classification":
        if hasattr(y, "values"):
            y_raw = y.values
        else:
            y_raw = np.asarray(y)

        validate_classification_target(y_raw)

        _, y_arr = np.unique(y_raw, return_inverse=True)
        y_arr = y_arr.astype(np.int32)
    else:
        y_arr = to_numpy(y, dtype=np.float64)
        if not np.isfinite(y_arr).all():
            raise ValueError("Non-finite values in y are not allowed for regression.")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]}")

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_arr.shape[1])]

    return X_arr, y_arr.ravel(), feature_names


def check_regression_only(task: str, estimator: str) -> None:
    """Raise if using regression-only estimator for classification."""
    regression_only = {"gaussian", "r2", "ksg"}
    if task == "classification" and estimator in regression_only:
        raise ValueError(
            f"estimator='{estimator}' is regression-only. "
            "Use estimator='binned' for classification."
        )


def resolve_jmi_estimator(estimator: str, task: str) -> str:
    """Resolve 'auto' to concrete estimator."""
    if estimator == "auto":
        return "binned" if task == "classification" else "r2"
    return estimator


# --- Subsampling ---


def subsample_xy(
    X: np.ndarray,
    y: np.ndarray,
    subsample: Optional[int],
    random_state: int,
    *,
    sample_weight: Optional[np.ndarray] = None,
    return_idx: bool = False,
) -> tuple:
    """Subsample X, y, and optionally sample_weight."""
    n = X.shape[0]
    w = ensure_weights(sample_weight, n, normalize=True)

    positive = np.flatnonzero(w > 0.0)
    if subsample is not None and positive.size > subsample:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(positive, size=subsample, replace=False)
    else:
        row_idx = positive
    X_sub, y_sub, w_sub = X[row_idx], y[row_idx], w[row_idx]

    mean = float(w_sub.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("Subsample weight mean must be finite and > 0")
    w_sub = w_sub / mean

    if return_idx:
        return X_sub, y_sub, w_sub, row_idx
    return X_sub, y_sub, w_sub


# --- Categorical encoding ---


@contextmanager
def suppress_category_encoder_pandas_warnings():
    """Hide narrow pandas 3.0 deprecation warnings emitted by category_encoders."""
    try:
        from pandas.errors import Pandas4Warning
    except (ImportError, AttributeError):
        Pandas4Warning = Warning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*future\.no_silent_downcasting.*",
            category=Pandas4Warning,
        )
        yield


class LeaveOneOutLogitEncoder:
    """Smoothed leave-one-out logit encoder for binary targets."""

    def __init__(
        self,
        cols: List[str],
        *,
        smoothing: float = 20.0,
        clip_min: float = 1e-4,
        clip_max: float = 1.0 - 1e-4,
    ):
        try:
            smoothing_float = float(smoothing)
            clip_min_float = float(clip_min)
            clip_max_float = float(clip_max)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "loo_smoothing and LOO-logit clip bounds must be finite numeric values"
            ) from exc
        if smoothing_float <= 0.0 or not np.isfinite(smoothing_float):
            raise ValueError("loo_smoothing must be positive and finite")
        if (
            not np.isfinite(clip_min_float)
            or not np.isfinite(clip_max_float)
            or not 0.0 < clip_min_float < clip_max_float < 1.0
        ):
            raise ValueError("loo_clip_min and loo_clip_max must satisfy 0 < min < max < 1")
        self.cols = list(cols)
        self.smoothing = smoothing_float
        self.clip_min = clip_min_float
        self.clip_max = clip_max_float

    @staticmethod
    def _series_with_missing_sentinel(series: pd.Series) -> pd.Series:
        sentinel = "__SIFT_MISSING_CATEGORY__"
        values = set(series.dropna().astype(object).tolist())
        while sentinel in values:
            sentinel += "_"
        return series.astype(object).where(~series.isna(), sentinel)

    @staticmethod
    def _get_column_series(X: pd.DataFrame, col: str) -> pd.Series:
        series = X.loc[:, col]
        if isinstance(series, pd.DataFrame):
            raise ValueError(
                "loo_logit encoding requires unique DataFrame column names "
                "for encoded categorical columns"
            )
        return series

    def _validate_binary_y(self, y) -> np.ndarray:
        y_arr = np.asarray(y).ravel()
        if pd.isna(y_arr).any():
            raise ValueError("loo_logit encoding requires a binary target without missing values")
        try:
            y_num = y_arr.astype(np.float64)
        except (TypeError, ValueError):
            y_num = None
        if y_num is not None:
            if not np.isfinite(y_num).all():
                raise ValueError("loo_logit encoding requires finite binary target values")
            unique_num = np.unique(y_num)
            if len(unique_num) != 2:
                raise ValueError("loo_logit encoding requires exactly two target classes")
            if set(unique_num.tolist()) == {0.0, 1.0}:
                return y_num

        unique = pd.unique(y_arr)
        if len(unique) != 2:
            raise ValueError("loo_logit encoding requires exactly two target classes")
        mapping = {unique[0]: 0.0, unique[1]: 1.0}
        return np.array([mapping[value] for value in y_arr], dtype=np.float64)

    def _logit(self, p: np.ndarray | float) -> np.ndarray | float:
        p_clipped = np.clip(p, self.clip_min, self.clip_max)
        return np.log(p_clipped / (1.0 - p_clipped))

    def _validate_sample_weight(self, sample_weight, n: int) -> np.ndarray:
        return ensure_weights(sample_weight, n, normalize=True)

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        y_arr = self._validate_binary_y(y)
        w = self._validate_sample_weight(sample_weight, len(y_arr))
        self.global_prior_ = float(np.sum(w * y_arr) / np.sum(w))
        self.global_logit_ = float(self._logit(self.global_prior_))
        self.category_maps_: dict[str, dict[object, float]] = {}

        for col in self.cols:
            series = self._series_with_missing_sentinel(self._get_column_series(X, col))
            codes, uniques = pd.factorize(series, sort=False)
            counts = np.bincount(codes, weights=w)
            sums = np.bincount(codes, weights=w * y_arr)
            p = (sums + self.smoothing * self.global_prior_) / (counts + self.smoothing)
            enc = self._logit(p)
            self.category_maps_[col] = dict(zip(uniques.tolist(), enc.tolist()))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols:
            mapping = self.category_maps_[col]
            series = self._series_with_missing_sentinel(self._get_column_series(X_out, col))
            X_out[col] = series.map(mapping).fillna(self.global_logit_).astype(float)
        return X_out

    def fit_transform(self, X: pd.DataFrame, y, sample_weight=None) -> pd.DataFrame:
        y_arr = self._validate_binary_y(y)
        w = self._validate_sample_weight(sample_weight, len(y_arr))
        self.fit(X, y_arr, sample_weight=w)
        X_out = X.copy()

        for col in self.cols:
            series = self._series_with_missing_sentinel(self._get_column_series(X_out, col))
            codes, _ = pd.factorize(series, sort=False)
            counts = np.bincount(codes, weights=w)
            sums = np.bincount(codes, weights=w * y_arr)
            p = (
                sums[codes]
                - w * y_arr
                + self.smoothing * self.global_prior_
            ) / (counts[codes] - w + self.smoothing)
            X_out[col] = self._logit(p).astype(float)
        return X_out


def target_cv_n_splits(
    y,
    *,
    target_type: Literal["auto", "continuous", "binary"] = "auto",
    cv: int = 5,
) -> int:
    """Return the usable deterministic fold count for native target encoding."""
    from sklearn.utils.multiclass import type_of_target

    if isinstance(cv, (bool, np.bool_)) or not isinstance(cv, (int, np.integer)):
        raise ValueError("target_cv cv must be an integer >= 2")
    requested = int(cv)
    if requested < 2:
        raise ValueError("target_cv cv must be an integer >= 2")
    if target_type not in {"auto", "continuous", "binary"}:
        raise ValueError("target_cv target_type must be 'auto', 'continuous', or 'binary'")

    y_arr = np.asarray(y).reshape(-1)
    if y_arr.size < 2:
        raise ValueError("target_cv requires at least two rows")
    inferred_kind = type_of_target(y_arr)
    target_kind = inferred_kind if target_type == "auto" else target_type
    if inferred_kind == "multiclass" and target_type != "continuous":
        raise ValueError(
            "cat_encoding='target_cv' does not yet support multiclass targets: "
            "sklearn expands each categorical feature to one column per class, "
            "which requires block-aware selection"
        )
    if target_kind == "binary":
        _, counts = np.unique(y_arr, return_counts=True)
        if counts.size != 2:
            raise ValueError("target_cv target_type='binary' requires exactly two classes")
        effective = min(requested, int(counts.min()))
    else:
        effective = min(requested, int(y_arr.size))
    if effective < 2:
        raise ValueError(
            "target_cv requires at least two rows per class for binary targets"
        )
    return effective


class TargetCVEncoder(TransformerMixin, BaseEstimator):
    """Cross-fitted sklearn target encoder with DataFrame-preserving output.

    The encoder intentionally preserves one output column per raw categorical
    feature.  sklearn's multiclass target encoding expands every input feature
    to one column per class, which cannot be represented by SIFT's current
    feature-selection contract until block-aware selection lands.
    """

    def __init__(
        self,
        cols: List[str],
        *,
        target_type: Literal["auto", "continuous", "binary"] = "auto",
        smooth: Literal["auto"] | float = "auto",
        cv: int = 5,
        random_state: int = 0,
        target_prior: float | None = None,
        warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    ):
        self.cols = cols
        self.target_type = target_type
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state
        self.target_prior = target_prior
        self.warmup_policy = warmup_policy

    def _validate_frame(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("target_cv encoding requires X to be a pandas DataFrame")
        if not X.columns.is_unique:
            raise ValueError("target_cv encoding requires unique DataFrame column names")
        missing = [col for col in self.cols if col not in X.columns]
        if missing:
            raise ValueError(f"target_cv columns are missing from X: {missing[:5]}")

    def _effective_cv(self, y) -> int:
        return target_cv_n_splits(y, target_type=self.target_type, cv=self.cv)

    def _requested_cv(self) -> int:
        if isinstance(self.cv, (bool, np.bool_)) or not isinstance(
            self.cv, (int, np.integer)
        ):
            raise ValueError("target_cv cv must be an integer >= 2")
        requested = int(self.cv)
        if requested < 2:
            raise ValueError("target_cv cv must be an integer >= 2")
        return requested

    def _validate_custom_options(self) -> float:
        if self.smooth == "auto":
            raise ValueError(
                "target_cv_smoothing must be an explicit non-negative float for "
                "weighted, grouped, or time-aware target_cv encoding; "
                "smooth='auto' is delegated only to sklearn's unweighted fixed-k path"
            )
        if isinstance(self.smooth, (bool, np.bool_)) or not isinstance(
            self.smooth, (int, float, np.integer, np.floating)
        ):
            raise ValueError("target_cv_smoothing must be 'auto' or a non-negative float")
        smooth = float(self.smooth)
        if not np.isfinite(smooth) or smooth < 0.0:
            raise ValueError("target_cv_smoothing must be finite and >= 0")
        if self.warmup_policy not in {"exclude", "zero_weight"}:
            raise ValueError("warmup_policy must be 'exclude' or 'zero_weight'")
        return smooth

    def _target_values(self, y) -> tuple[np.ndarray, str]:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.multiclass import type_of_target

        y_arr = np.asarray(y).reshape(-1)
        inferred = type_of_target(y_arr)
        target_kind = inferred if self.target_type == "auto" else self.target_type
        if inferred == "multiclass" and self.target_type != "continuous":
            raise ValueError(
                "cat_encoding='target_cv' does not yet support multiclass targets: "
                "sklearn expands each categorical feature to one column per class, "
                "which requires block-aware selection"
            )
        if target_kind == "binary":
            label_encoder = LabelEncoder().fit(y_arr)
            if label_encoder.classes_.size != 2:
                raise ValueError("target_cv target_type='binary' requires exactly two classes")
            self.classes_ = label_encoder.classes_
            values = label_encoder.transform(y_arr).astype(np.float64)
            if self.target_prior is not None and not 0.0 <= float(self.target_prior) <= 1.0:
                raise ValueError("target_prior must be between 0 and 1 for binary targets")
            return values, "binary"
        if target_kind != "continuous":
            raise ValueError(
                "target_cv supports only continuous regression and binary classification targets"
            )
        try:
            values = np.asarray(y_arr, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("continuous target_cv targets must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError("target_cv y contains non-finite values")
        return values, "continuous"

    @staticmethod
    def _normalized_series(series: pd.Series) -> pd.Series:
        return series.astype(object).where(~series.isna(), np.nan)

    def _fit_custom_maps(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray,
        smooth: float,
    ) -> tuple[dict[str, dict[object, float]], float]:
        active = sample_weight > 0.0
        if not bool(np.any(active)):
            raise ValueError("target_cv fit rows have zero total sample_weight")
        weights = sample_weight[active]
        y_active = y[active]
        total_weight = float(weights.sum())
        prior = float(np.dot(weights, y_active) / total_weight)
        mappings: dict[str, dict[object, float]] = {}
        for col in self.cols:
            series = self._normalized_series(X.loc[active, col])
            codes, categories = pd.factorize(
                series,
                sort=False,
                use_na_sentinel=False,
            )
            counts = np.bincount(codes, weights=weights, minlength=len(categories))
            sums = np.bincount(
                codes,
                weights=weights * y_active,
                minlength=len(categories),
            )
            encoded = (sums + smooth * prior) / (counts + smooth)
            mappings[col] = {
                category: float(value)
                for category, value in zip(categories.tolist(), encoded.tolist())
            }
        return mappings, prior

    def _apply_custom_maps(
        self,
        X: pd.DataFrame,
        mappings: dict[str, dict[object, float]],
        prior: float,
    ) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols:
            series = self._normalized_series(X_out[col])
            X_out[col] = series.map(mappings[col]).fillna(prior).astype(np.float64)
        return X_out

    def _fixed_splits(
        self,
        y: np.ndarray,
        target_kind: str,
        active: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import KFold, StratifiedKFold

        active_idx = np.flatnonzero(active)
        requested = target_cv_n_splits(
            y[active],
            target_type="binary" if target_kind == "binary" else "continuous",
            cv=self.cv,
        )
        if target_kind == "binary":
            splitter = StratifiedKFold(
                n_splits=requested,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(active_idx, y[active])
        else:
            splitter = KFold(
                n_splits=requested,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(active_idx)
        return [
            (active_idx[train_local], active_idx[valid_local])
            for train_local, valid_local in split_iter
        ]

    def _group_splits(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups,
        active: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        from sklearn.model_selection import GroupKFold

        group_arr = np.asarray(groups).reshape(-1)
        if group_arr.size != len(X):
            raise ValueError(f"groups has {group_arr.size} rows but X has {len(X)}")
        if bool(np.asarray(pd.isna(group_arr), dtype=bool).any()):
            raise ValueError("groups must not contain missing values")
        active_idx = np.flatnonzero(active)
        n_groups = int(pd.unique(group_arr[active]).size)
        n_splits = min(self._requested_cv(), n_groups)
        if n_splits < 2:
            raise ValueError(f"target_cv group folds require at least 2 groups, got {n_groups}")
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (active_idx[train_local], active_idx[valid_local])
            for train_local, valid_local in splitter.split(
                X.iloc[active_idx],
                y[active_idx],
                group_arr[active_idx],
            )
        ]

    def _time_splits(
        self,
        time,
        active: np.ndarray,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        values = np.asarray(time).reshape(-1)
        if values.size != active.size:
            raise ValueError(f"time has {values.size} rows but X has {active.size}")
        if bool(np.asarray(pd.isna(values), dtype=bool).any()):
            raise ValueError("time values must not contain missing values")
        active_idx = np.flatnonzero(active)
        active_values = values[active_idx]
        try:
            order_local = np.argsort(active_values, kind="mergesort")
            ordered = active_values[order_local]
            if ordered.dtype.kind == "O":
                for previous, current in zip(ordered[:-1], ordered[1:]):
                    if bool(current < previous):
                        raise TypeError
            elif np.asarray(ordered[1:] < ordered[:-1], dtype=bool).any():
                raise TypeError
            distinct = np.asarray(ordered[1:] != ordered[:-1], dtype=bool)
        except (TypeError, ValueError) as exc:
            raise TypeError("time values must be orderable") from exc
        boundaries = np.flatnonzero(distinct) + 1
        timestamp_groups = np.split(active_idx[order_local], boundaries)
        n_timestamps = len(timestamp_groups)
        n_splits = min(self._requested_cv(), n_timestamps)
        if n_splits < 2:
            raise ValueError(
                "target_cv time folds require at least 2 distinct timestamps"
            )
        block_groups = np.array_split(np.arange(n_timestamps), n_splits)
        blocks = [
            np.concatenate([timestamp_groups[int(i)] for i in block]).astype(
                np.int64,
                copy=False,
            )
            for block in block_groups
        ]
        warmup = blocks[0]
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        history: list[np.ndarray] = [blocks[0]]
        for block in blocks[1:]:
            splits.append((np.concatenate(history), block))
            history.append(block)
        return splits, warmup

    def _encoder_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize all pandas missing sentinels to one learned category."""
        frame = X.loc[:, self.cols].copy()
        for col in self.cols:
            series = frame[col]
            if bool(series.isna().any()):
                frame[col] = series.astype(object).where(~series.isna(), np.nan)
        return frame

    def _make_encoder(self, y):
        from sklearn.preprocessing import TargetEncoder

        self.n_splits_ = self._effective_cv(y)
        return TargetEncoder(
            target_type=self.target_type,
            smooth=self.smooth,
            cv=self.n_splits_,
            shuffle=True,
            random_state=self.random_state,
        )

    def _replace_columns(self, X: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        encoded = np.asarray(values, dtype=np.float64)
        if encoded.ndim != 2 or encoded.shape[1] != len(self.cols):
            raise ValueError(
                "cat_encoding='target_cv' must preserve one encoded column per "
                "categorical feature; multiclass expansion is not supported"
            )
        X_out = X.copy()
        for index, col in enumerate(self.cols):
            X_out[col] = encoded[:, index]
        return X_out

    def fit(self, X: pd.DataFrame, y, sample_weight=None):
        self._validate_frame(X)
        y_arr = np.asarray(y).reshape(-1)
        if len(X) != y_arr.size:
            raise ValueError(f"X has {len(X)} rows but y has {y_arr.size}")
        if self.target_prior is not None or self.warmup_policy != "zero_weight":
            raise ValueError(
                "target_prior and warmup_policy are only meaningful for time-aware "
                "target_cv fit_transform"
            )
        if sample_weight is not None:
            smooth = self._validate_custom_options()
            y_values, target_kind = self._target_values(y_arr)
            weights = ensure_weights(sample_weight, len(X), normalize=False)
            self.category_maps_, self.global_prior_ = self._fit_custom_maps(
                X,
                y_values,
                weights,
                smooth,
            )
            self.target_kind_ = target_kind
            self.fit_mode_ = "custom"
            self.n_splits_ = self._effective_cv(y_arr)
            self.encoding_cv_ = {"kind": "fixed_k", "n_splits": self.n_splits_}
            return self
        self.encoder_ = self._make_encoder(y_arr)
        self.encoder_.fit(self._encoder_input(X), y_arr)
        self.fit_mode_ = "sklearn"
        self.encoding_cv_ = {"kind": "fixed_k", "n_splits": self.n_splits_}
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y,
        sample_weight=None,
        groups=None,
        time=None,
    ) -> pd.DataFrame:
        self._validate_frame(X)
        y_arr = np.asarray(y).reshape(-1)
        if len(X) != y_arr.size:
            raise ValueError(f"X has {len(X)} rows but y has {y_arr.size}")
        if groups is not None and time is not None:
            raise ValueError(
                "cat_encoding='target_cv' does not support groups and time together"
            )
        if time is None and self.target_prior is not None:
            raise ValueError("target_prior is only meaningful for time-aware target_cv")
        if time is None and self.warmup_policy != "zero_weight":
            raise ValueError("warmup_policy is only meaningful for time-aware target_cv")
        if (
            time is not None
            and self.target_prior is not None
            and self.warmup_policy == "exclude"
        ):
            raise ValueError(
                "target_prior and warmup_policy='exclude' are mutually exclusive"
            )
        custom = sample_weight is not None or groups is not None or time is not None
        if custom:
            smooth = self._validate_custom_options()
            y_values, target_kind = self._target_values(y_arr)
            raw_weights = ensure_weights(sample_weight, len(X), normalize=False)
            active = raw_weights > 0.0
            values = np.full(
                (len(X), len(self.cols)),
                0.5 if target_kind == "binary" else 0.0,
                dtype=np.float64,
            )
            if groups is not None:
                splits = self._group_splits(X, y_values, groups, active)
                kind = "group"
                warmup = np.empty(0, dtype=np.int64)
            elif time is not None:
                splits, warmup = self._time_splits(time, active)
                kind = "time"
                if self.target_prior is not None:
                    prior = float(self.target_prior)
                    if not np.isfinite(prior):
                        raise ValueError("target_prior must be finite")
                    if target_kind == "binary" and not 0.0 <= prior <= 1.0:
                        raise ValueError(
                            "target_prior must be between 0 and 1 for binary targets"
                        )
                    values[warmup, :] = prior
            else:
                splits = self._fixed_splits(y_values, target_kind, active)
                kind = "fixed_k"
                warmup = np.empty(0, dtype=np.int64)

            for train_idx, valid_idx in splits:
                mappings, prior = self._fit_custom_maps(
                    X.iloc[train_idx],
                    y_values[train_idx],
                    raw_weights[train_idx],
                    smooth,
                )
                fold_values = self._apply_custom_maps(
                    X.iloc[valid_idx],
                    mappings,
                    prior,
                )
                values[valid_idx, :] = fold_values.loc[:, self.cols].to_numpy(
                    dtype=np.float64,
                )

            effective_weights = raw_weights.copy()
            if time is not None and self.target_prior is None:
                effective_weights[warmup] = 0.0
            if not bool(np.any(effective_weights > 0.0)):
                raise ValueError(
                    "target_cv warmup handling leaves no positive selection weight"
                )
            self.effective_sample_weight_ = (
                effective_weights
                if sample_weight is not None or warmup.size > 0
                else None
            )
            self.warmup_mask_ = np.ones(len(X), dtype=bool)
            self.warmup_mask_[warmup] = False
            self.category_maps_, self.global_prior_ = self._fit_custom_maps(
                X,
                y_values,
                raw_weights,
                smooth,
            )
            self.target_kind_ = target_kind
            self.fit_mode_ = "custom"
            self.n_splits_ = len(splits) + (1 if time is not None else 0)
            self.encoding_cv_ = {"kind": kind, "n_splits": self.n_splits_}
            return self._replace_columns(X, values)
        self.encoder_ = self._make_encoder(y_arr)
        values = self.encoder_.fit_transform(self._encoder_input(X), y_arr)
        self.fit_mode_ = "sklearn"
        self.effective_sample_weight_ = None
        self.warmup_mask_ = np.ones(len(X), dtype=bool)
        self.encoding_cv_ = {"kind": "fixed_k", "n_splits": self.n_splits_}
        return self._replace_columns(X, values)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, ["fit_mode_", "encoding_cv_"])
        self._validate_frame(X)
        if self.fit_mode_ == "custom":
            return self._apply_custom_maps(X, self.category_maps_, self.global_prior_)
        values = self.encoder_.transform(self._encoder_input(X))
        return self._replace_columns(X, values)


def encode_categoricals(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    method: CatEncoding,
    *,
    loo_smoothing: float = 20.0,
    loo_clip_min: float = 1e-4,
    loo_clip_max: float = 1.0 - 1e-4,
    sample_weight=None,
    target_type: Literal["auto", "continuous", "binary"] = "auto",
    target_cv_n_splits: int = 5,
    target_cv_smoothing: Literal["auto"] | float = "auto",
    target_prior: float | None = None,
    warmup_policy: Literal["exclude", "zero_weight"] = "zero_weight",
    groups=None,
    time=None,
) -> pd.DataFrame:
    """Apply target encoding to categorical features."""
    if method == "none":
        return X
    if method == "target_cv":
        encoder = TargetCVEncoder(
            cat_features,
            target_type=target_type,
            smooth=target_cv_smoothing,
            cv=target_cv_n_splits,
            target_prior=target_prior,
            warmup_policy=warmup_policy,
        )
        return encoder.fit_transform(
            X,
            y,
            sample_weight=sample_weight,
            groups=groups,
            time=time,
        )
    if method == "loo_logit":
        encoder = LeaveOneOutLogitEncoder(
            cat_features,
            smoothing=loo_smoothing,
            clip_min=loo_clip_min,
            clip_max=loo_clip_max,
        )
        return encoder.fit_transform(X, y, sample_weight=sample_weight)
    if sample_weight is not None:
        raise ValueError(
            "sample_weight with supervised categorical encoding is only supported "
            "for cat_encoding='loo_logit'. category_encoders-backed methods "
            "('loo', 'target', 'james_stein') do not consume sample weights."
        )
    try:
        import category_encoders as ce
    except ImportError as exc:
        raise ImportError(
            "category_encoders required for categorical encoding. "
            "Install with: pip install category_encoders"
        ) from exc

    encoders = {
        "loo": ce.LeaveOneOutEncoder,
        "target": ce.TargetEncoder,
        "james_stein": ce.JamesSteinEncoder,
    }
    encoder = encoders[method](cols=cat_features, handle_missing="return_nan")
    with suppress_category_encoder_pandas_warnings():
        return encoder.fit_transform(X, y)
