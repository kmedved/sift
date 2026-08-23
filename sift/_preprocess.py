"""Input preprocessing: validation, conversion, subsampling, encoding."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, List, Literal, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

EstimatorJMI = Literal["auto", "binned", "r2", "ksg", "gaussian"]
EstimatorMRMR = Literal["classic", "gaussian"]
RelevanceMethod = Literal["f", "ks", "rf"]
CatEncoding = Literal["none", "target", "loo", "james_stein", "loo_logit"]
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
    X, y, task: str, impute: bool = True, *, dtype=np.float32
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Validate and convert inputs.

    ``dtype`` controls the returned feature dtype. The classic filter paths
    keep ``float32``; solvers that immediately work in double precision pass
    ``np.float64`` to avoid a lossy round trip (large offsets or tiny scales
    can otherwise collapse to constants).
    """
    from sift._impute import mean_impute

    feature_names = extract_feature_names(X)
    if hasattr(X, "select_dtypes"):
        non_numeric = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if non_numeric:
            sample = non_numeric[:5]
            suffix = "..." if len(non_numeric) > 5 else ""
            raise ValueError(
                f"Non-numeric columns found: {sample}{suffix}. "
                "Either encode them first or set cat_encoding to 'loo', "
                "'target', 'james_stein', or binary-only 'loo_logit'."
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

        if pd.api.types.is_numeric_dtype(y_raw):
            try:
                y_num = np.asarray(y_raw, dtype=np.float64)
            except (TypeError, ValueError):
                y_num = None
            if y_num is not None and not np.isfinite(y_num).all():
                raise ValueError("Non-finite values in y are not allowed for classification.")

        if pd.isna(y_raw).any():
            raise ValueError("Missing values in y are not allowed for classification.")

        _, y_arr = np.unique(y_raw, return_inverse=True)
        y_arr = y_arr.astype(np.int32)
    else:
        y_arr = to_numpy(y, dtype=np.float32)
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

    if subsample is not None and n > subsample:
        rng = np.random.default_rng(random_state)
        row_idx = rng.choice(n, size=subsample, replace=False)
        X_sub, y_sub, w_sub = X[row_idx], y[row_idx], w[row_idx]
        if float(w_sub.sum()) <= 0.0:
            raise ValueError("Subsample has zero total weight; check sample_weight.")
    else:
        row_idx = np.arange(n)
        X_sub, y_sub, w_sub = X, y, w

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
) -> pd.DataFrame:
    """Apply target encoding to categorical features."""
    if method == "none":
        return X
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
