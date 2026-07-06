# SIFT API Reference

This document provides detailed API documentation for all public functions and classes in SIFT.

## Table of Contents

- [Filter-based Selection](#filter-based-selection)
  - [select_mrmr](#select_mrmr)
  - [select_jmi](#select_jmi)
  - [select_jmim](#select_jmim)
  - [select_cefsplus](#select_cefsplus)
- [Stability Selection](#stability-selection)
  - [StabilitySelector](#stabilityselector)
  - [stability_regression](#stability_regression)
  - [stability_classif](#stability_classif)
- [Boruta Selection](#boruta-selection)
  - [BorutaSelector](#borutaselector)
  - [select_boruta](#select_boruta)
  - [select_boruta_shap](#select_boruta_shap)
- [CatBoost Selection](#catboost-selection)
  - [catboost_select](#catboost_select)
  - [catboost_regression](#catboost_regression)
  - [catboost_classif](#catboost_classif)
  - [CatBoostSelectionResult](#catboostselectionresult)
- [Caching and Utilities](#caching-and-utilities)
  - [FeatureCache](#featurecache)
  - [build_cache](#build_cache)
  - [select_cached](#select_cached)
- [Automatic K Selection](#automatic-k-selection)
  - [AutoKConfig](#autokconfig)
  - [select_k_auto](#select_k_auto)
  - [select_k_elbow](#select_k_elbow)
- [Smart Sampling](#smart-sampling)
  - [SmartSamplerConfig](#smartsamplerconfig)
  - [smart_sample](#smart_sample)
  - [panel_config](#panel_config)
  - [cross_section_config](#cross_section_config)
- [Permutation Importance](#permutation-importance)
  - [permutation_importance](#permutation_importance)

---

## Filter-based Selection

### select_mrmr

```python
sift.select_mrmr(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Literal["regression", "classification"],
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
    relevance: Literal["f", "ks", "rf"] = "f",
    estimator: Literal["classic", "gaussian"] = "classic",
    formula: Literal["quotient", "difference"] = "quotient",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: Literal["none", "loo", "target", "james_stein"] = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True
) -> List[str]
```

Minimum Redundancy Maximum Relevance feature selection.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | DataFrame or ndarray | required | Feature matrix with shape (n_samples, n_features) |
| `y` | Series or ndarray | required | Target variable with shape (n_samples,) |
| `k` | int or "auto" | required | Number of features to select, or "auto" for automatic selection |
| `task` | str | required | Task type: "regression" or "classification" |
| `cache` | FeatureCache | None | Pre-computed feature cache for faster computation |
| `groups` | ndarray | None | Group labels for grouped data |
| `time` | ndarray | None | Time values for temporal ordering |
| `auto_k_config` | AutoKConfig | None | Configuration for automatic K selection |
| `sample_weight` | ndarray | None | Sample weights |
| `relevance` | str | "f" | Relevance measure: regression supports "f" (F-stat) and "rf" (Random Forest); classification supports "f", "ks" (KS-test), and "rf" |
| `estimator` | str | "classic" | Estimator type: "classic" or "gaussian" (regression only) |
| `formula` | str | "quotient" | Score formula: "quotient" (rel/red) or "difference" (rel-red) |
| `top_m` | int | None | Pre-filter to top_m features by relevance. Default: max(5*k, 250) |
| `cat_features` | list | None | List of categorical feature names |
| `cat_encoding` | str | "loo" | Categorical encoding method |
| `subsample` | int | 50_000 | Maximum samples for computation |
| `random_state` | int | 0 | Random seed for reproducibility |
| `verbose` | bool | True | Whether to print progress |

**Returns:**

`List[str]` - Selected feature names in order of selection.

**Example:**

```python
from sift import select_mrmr
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=50, n_informative=10)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(50)])

# Basic usage
selected = select_mrmr(X, y, k=10, task="regression")

# With gaussian estimator (faster for regression)
selected = select_mrmr(X, y, k=10, task="regression", estimator="gaussian")

# Automatic K selection
selected = select_mrmr(X, y, k="auto", task="regression", time=timestamps)
```

---

### select_jmi

```python
sift.select_jmi(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Literal["regression", "classification"],
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
    estimator: Literal["auto", "gaussian", "binned", "ksg", "r2"] = "auto",
    relevance: Literal["f", "ks", "rf"] = "f",
    top_m: Optional[int] = None,
    cat_features: Optional[List[str]] = None,
    cat_encoding: Literal["none", "loo", "target", "james_stein"] = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True
) -> List[str]
```

Joint Mutual Information feature selection.

JMI scores features by: `score(f) = Σ_{s ∈ S} I(f, s; y)` where I is mutual information.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | DataFrame or ndarray | required | Feature matrix |
| `y` | Series or ndarray | required | Target variable |
| `k` | int or "auto" | required | Number of features to select |
| `task` | str | required | "regression" or "classification" |
| `estimator` | str | "auto" | MI estimator: "gaussian", "binned", "ksg", "r2", or "auto" |
| `relevance` | str | "f" | Initial relevance measure: regression supports "f" and "rf"; classification supports "f", "ks", and "rf" |
| `top_m` | int | None | Pre-filter threshold |

**Returns:**

`List[str]` - Selected feature names.

**Example:**

```python
from sift import select_jmi

# Classification with KS relevance
selected = select_jmi(X, y, k=10, task="classification", relevance="ks")

# Regression with Gaussian estimator
selected = select_jmi(X, y, k=10, task="regression", estimator="gaussian")
```

---

### select_jmim

```python
sift.select_jmim(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]],
    *,
    task: Literal["regression", "classification"],
    # ... same parameters as select_jmi
) -> List[str]
```

JMI Maximization - conservative variant of JMI.

JMIM scores features by: `score(f) = min_{s ∈ S} I(f, s; y)` (uses minimum instead of sum).

**Example:**

```python
from sift import select_jmim

# More conservative selection
selected = select_jmim(X, y, k=10, task="regression")
```

---

### select_cefsplus

```python
sift.select_cefsplus(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: Union[int, Literal["auto"]] = 75,
    *,
    cache: Optional[FeatureCache] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auto_k_config: Optional[AutoKConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
    top_m: Optional[int] = None,
    corr_prune: float = 0.95,
    cat_features: Optional[List[str]] = None,
    cat_encoding: Literal["none", "loo", "target", "james_stein"] = "loo",
    subsample: Optional[int] = 50_000,
    random_state: int = 0,
    verbose: bool = True
) -> List[str]
```

CEFS+ (Conditional Entropy Feature Selection Plus) using log-determinant Gaussian MI proxy.

**Note:** Regression only.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corr_prune` | float | 0.95 | Correlation threshold for pruning highly correlated features |
| Other parameters same as `select_mrmr` |

**Returns:**

`List[str]` - Selected feature names.

**Example:**

```python
from sift import select_cefsplus

# Standard usage
selected = select_cefsplus(X, y, k=20)

# With correlation pruning
selected = select_cefsplus(X, y, k=20, corr_prune=0.9)
```

---

## Stability Selection

### StabilitySelector

```python
class sift.StabilitySelector(
    n_bootstrap: int = 50,
    sample_frac: float = 0.5,
    threshold: float = 0.6,
    alpha: Optional[float] = None,
    l1_ratio: float = 1.0,
    task: Literal["regression", "classification"] = "regression",
    max_features: Optional[int] = None,
    use_smart_sampler: bool = False,
    sampler_config: Optional[SmartSamplerConfig] = None,
    store_coefs: bool = True,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False
)
```

Sklearn-compatible stability selector using bootstrap resampling.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bootstrap` | int | 50 | Number of bootstrap iterations |
| `sample_frac` | float | 0.5 | Fraction of samples per bootstrap |
| `threshold` | float | 0.6 | Minimum selection frequency for inclusion |
| `alpha` | float | None | Regularization parameter (None = auto via CV) |
| `l1_ratio` | float | 1.0 | L1/L2 ratio: 1.0=Lasso, <1.0=ElasticNet |
| `task` | str | "regression" | Task type |
| `max_features` | int | None | Maximum features to select |
| `block_size` | int or str | "auto" | Block size for time-series/block bootstrap |
| `block_method` | str | "moving" | Block bootstrap strategy: "moving", "circular", or "stationary" |
| `use_smart_sampler` | bool | False | Enable leverage-based smart sampling |
| `sampler_config` | SmartSamplerConfig | None | Smart sampler configuration |
| `store_coefs` | bool | True | Store coefficients for analysis |
| `random_state` | int | None | Random seed |
| `n_jobs` | int | -1 | Parallel jobs for bootstrap |
| `parallel_backend` | str | "threads" | Joblib backend preference |
| `verbose` | bool | True | Print progress |

**Methods:**

```python
# Fit the selector
selector.fit(X, y, sample_weight=None, groups=None, time=None)

# Transform data to selected features
X_selected = selector.transform(X)

# Fit and transform
X_selected = selector.fit_transform(X, y)

# Get selected features
features = selector.get_support(indices=False)  # Boolean mask
feature_indices = selector.get_support(indices=True)  # Integer indices
feature_names = selector.selected_feature_names_  # Selected feature names

# Get feature information DataFrame
info = selector.get_feature_info()

# Get coefficient stability (mean, std)
coef_stability = selector.get_coef_stability()

# Tune threshold via cross-validation
best_threshold = selector.tune_threshold(X, y, thresholds=[0.4, 0.5, 0.6, 0.7], cv=3)

# Set new threshold
selector.set_threshold(0.7)

# Plotting
selector.plot_frequencies(top_n=50)
selector.plot_coef_distributions(features=None, top_n=12)
```

**Example:**

```python
from sift import StabilitySelector

# Basic usage
selector = StabilitySelector(threshold=0.6, n_bootstrap=50)
selector.fit(X, y)

# Get results
feature_indices = selector.get_support(indices=True)
feature_names = selector.selected_feature_names_
info = selector.get_feature_info()
print(info.head(10))

# With block bootstrap for time series
selector = StabilitySelector(
    threshold=0.6,
    n_bootstrap=50,
    block_method="moving",
)
selector.fit(X, y, groups=player_ids, time=timestamps)

# Tune threshold
best = selector.tune_threshold(X, y, thresholds=[0.4, 0.5, 0.6, 0.7, 0.8])
```

---

### stability_regression

```python
sift.stability_regression(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    **kwargs
) -> List[str]
```

Stability selection for regression (function API). Keyword arguments are passed to
`StabilitySelector`, including `sample_weight`, `groups`, `time`, `block_size`,
`block_method`, `use_smart_sampler`, `sampler_config`, `alpha`, `random_state`,
and `verbose`.

**Returns:**

`List[str]` - Selected feature names.

---

### stability_classif

```python
sift.stability_classif(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    **kwargs
) -> List[str]
```

Stability selection for classification (function API). Keyword arguments are passed to
`StabilitySelector`, including `sample_weight`, `groups`, `time`, `block_size`,
`block_method`, `use_smart_sampler`, `sampler_config`, `alpha`, `random_state`,
and `verbose`.

---

## Boruta Selection

### BorutaSelector

```python
class sift.BorutaSelector(
    estimator = None,
    *,
    n_estimators: Union[int, str] = "auto",
    task: Literal["regression", "classification"] = "regression",
    importance: Literal["native", "permutation", "shap"] = "native",
    max_iter: int = 50,
    alpha: float = 0.05,
    perc: int = 100,
    resolve_tentative: bool = True,
    max_features: Optional[int] = None,
    shadow_method: str = "auto",
    shadow_mode: str = "columns",
    block_size: Union[int, str] = "auto",
    cat_features: Optional[List[str]] = None,
    cat_encoding: str = "loo",
    importance_data: Literal["train", "test"] = "train",
    test_size: float = 0.3,
    shap_sample_size: Optional[int] = 2000,
    early_stop_rounds: int = 5,
    random_state: int = 0,
    verbose: bool = True
)
```

Sklearn-compatible Boruta feature selector.

**Methods:**

```python
selector.fit(X, y)
X_selected = selector.transform(X)
X_selected = selector.fit_transform(X, y)
features = selector.get_support(indices=False)
```

---

### select_boruta

```python
sift.select_boruta(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    estimator: str = "rf",
    n_estimators: Union[int, str] = "auto",
    max_iter: int = 100,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> List[str]
```

Boruta feature selection (function API).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | str | "rf" | Base estimator: "rf" (Random Forest) |
| `n_estimators` | int or "auto" | "auto" | Number of trees |
| `max_iter` | int | 100 | Maximum iterations |
| `alpha` | float | 0.05 | Significance level |
| `random_state` | int | None | Random seed |

---

### select_boruta_shap

```python
sift.select_boruta_shap(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    estimator: str = "lightgbm",
    max_iter: int = 50,
    # ... other parameters
) -> List[str]
```

SHAP-based Boruta variant using SHAP values for feature importance.

---

## CatBoost Selection

### catboost_select

```python
sift.catboost_select(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    task: Literal["regression", "classification"] = "regression",
    k: Optional[int] = None,
    min_features: int = 5,
    step_function: float = 0.67,
    feature_counts: Optional[List[int]] = None,
    selection_patience: int = 3,
    tolerance: float = 0.01,
    algorithm: Literal["shap", "permutation", "prediction", "forward", "forward_greedy"] = "shap",
    cv: Optional[Any] = None,
    n_splits: int = 3,
    test_size: float = 0.25,
    group_col: Optional[str] = None,
    sample_weight_col: Optional[str] = None,
    prefilter_k: Optional[int] = 200,
    prefilter_method: str = "catboost",
    use_stability: bool = False,
    n_bootstrap: int = 20,
    stability_threshold: float = 0.6,
    n_estimators: int = 500,
    learning_rate: Optional[float] = None,
    max_depth: Optional[int] = 6,
    eval_metric: Optional[str] = None,
    loss_function: Optional[str] = None,
    catboost_params: Optional[Dict] = None,
    steps: int = 6,
    cat_features: Optional[List[str]] = None,
    text_features: Optional[List[str]] = None,
    treat_object_as_categorical: bool = True,
    train_early_stopping_rounds: int = 20,
    gpu: bool = False,
    n_jobs: int = -1,
    higher_is_better: Optional[bool] = None,
    random_state: Optional[int] = None,
    verbose: bool = True
) -> CatBoostSelectionResult
```

Full-control CatBoost-based feature selection.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | DataFrame or ndarray | required | Feature matrix |
| `y` | Series or ndarray | required | Target variable |
| `task` | str | "regression" | Task type |
| `k` | int | None | Features to select (None = search for optimal) |
| `min_features` | int | 5 | Minimum features when searching |
| `algorithm` | str | "shap" | Selection algorithm |
| `cv` | splitter | None | Custom CV splitter (TimeSeriesSplit, GroupKFold, etc.) |
| `n_splits` | int | 3 | Number of CV splits (if cv is None) |
| `eval_metric` | str | None | Evaluation metric (auto-detected if None) |
| `loss_function` | str | None | Loss function (auto-detected if None) |
| `higher_is_better` | bool | None | Metric direction (auto-detected if None) |
| `prefilter_k` | int | 200 | Two-stage pre-filtering (inside CV) |
| `group_col` | str | None | Column name for group-aware operations |
| `sample_weight_col` | str | None | Column name containing sample weights |
| `use_stability` | bool | False | Enable stability selection |
| `n_bootstrap` | int | 20 | Bootstrap iterations for stability |
| `stability_threshold` | float | 0.6 | Selection frequency threshold |
| `catboost_params` | dict | None | Additional CatBoost parameters |
| `random_state` | int | None | Random seed |
| `verbose` | bool | True | Print progress |

**Returns:**

`CatBoostSelectionResult` - Result object with selected features and metadata.

**Example:**

```python
import sift
from sklearn.model_selection import TimeSeriesSplit

# Basic usage
result = sift.catboost_select(X, y, task="regression", k=20)

# Time series with forward selection
result = sift.catboost_select(
    X, y,
    task="regression",
    k=20,
    cv=TimeSeriesSplit(n_splits=5),
    algorithm="forward"
)

# Automatic K search with SHAP
result = sift.catboost_select(
    X, y,
    task="regression",
    k=None,  # Search
    algorithm="shap",
    prefilter_k=100
)

# Group-aware stability selection
result = sift.catboost_select(
    X, y,
    task="regression",
    k=20,
    group_col="player_id",
    use_stability=True,
    n_bootstrap=20
)
```

---

### catboost_regression

```python
sift.catboost_regression(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    algorithm: str = "forward",
    prefilter_k: Optional[int] = None,
    cv: Optional[Any] = None,
    verbose: bool = True
) -> List[str]
```

Simplified CatBoost selection for regression.

**Returns:**

`List[str]` - Selected feature names.

---

### catboost_classif

```python
sift.catboost_classif(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    algorithm: str = "forward",
    prefilter_k: Optional[int] = None,
    cv: Optional[Any] = None,
    verbose: bool = True
) -> List[str]
```

Simplified CatBoost selection for classification.

---

### CatBoostSelectionResult

```python
@dataclass
class sift.CatBoostSelectionResult:
    selected_features: List[str]
    best_k: int
    scores_by_k: Dict[int, float]
    scores_std_by_k: Dict[int, float]
    feature_importances: pd.Series
    features_by_k: Dict[int, List[str]]
    stability_scores: Optional[pd.Series]
    prefilter_features: Optional[List[str]]
    metric: str
    higher_is_better: bool
    all_scores: Optional[Dict[int, List[float]]]
```

Result container for CatBoost selection.

**Methods:**

```python
# Get score at specific K
mean, std = result.score_at_k(k=15)

# Get smallest feature set within tolerance of best
parsimonious = result.features_within_tolerance(tolerance=0.01)

# Plot scores vs K
result.plot_scores_vs_k(figsize=(10, 6))
```

---

## Caching and Utilities

### FeatureCache

```python
@dataclass
class sift.FeatureCache:
    Z: np.ndarray                      # Gaussian-transformed features
    Rxx: Optional[np.ndarray]          # Correlation matrix
    valid_cols: np.ndarray             # Valid feature indices
    row_idx: np.ndarray                # Subsampled row indices
    sample_weight: np.ndarray          # Sample weights
    feature_names: Optional[List[str]] # Feature names
```

Cache for repeated feature selection with different targets.

---

### build_cache

```python
sift.build_cache(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    sample_weight: Optional[np.ndarray] = None,
    subsample: Optional[int] = 50_000,
    random_state: int = 0
) -> FeatureCache
```

Build a feature cache for faster multi-target selection.

**Example:**

```python
from sift import build_cache, select_cached

# Build cache once
cache = build_cache(X, subsample=50000)

# Use for multiple targets
selected_y1 = select_cached(cache, y1, k=10, method="mrmr_quot")
selected_y2 = select_cached(cache, y2, k=10, method="jmi")
selected_y3 = select_cached(cache, y3, k=10, method="cefsplus")
```

---

### select_cached

```python
sift.select_cached(
    cache: FeatureCache,
    y: np.ndarray,
    k: int,
    *,
    method: Literal["cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"] = "cefsplus",
    top_m: Optional[int] = None,
    corr_prune: Union[float, str, None] = "auto",
    return_objective: bool = False,
    return_indices: bool = False
) -> Union[List[str], Tuple[List[str], np.ndarray], Tuple[List[str], List[int]], Tuple[List[str], List[int], np.ndarray]]
```

Select features from a pre-built cache.

---

## Automatic K Selection

### AutoKConfig

```python
@dataclass
class sift.AutoKConfig:
    k_method: Literal["evaluate", "elbow"] = "evaluate"
    strategy: Literal["time_holdout", "group_cv"] = "time_holdout"
    metric: Literal["rmse", "mae", "logloss", "error", "auto"] = "auto"
    max_k: int = 100
    min_k: int = 5
    val_frac: float = 0.2
    n_splits: int = 5
    random_state: int = 42
    elbow_min_rel_gain: float = 0.02
    elbow_patience: int = 3
```

Configuration for automatic K selection.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_method` | str | "evaluate" | Method: "evaluate" (CV) or "elbow" |
| `strategy` | str | "time_holdout" | Strategy for evaluate: "time_holdout" or "group_cv" |
| `metric` | str | "auto" | Evaluation metric |
| `max_k` | int | 100 | Maximum K to consider |
| `min_k` | int | 5 | Minimum K to consider |
| `val_frac` | float | 0.2 | Validation fraction for time_holdout |
| `n_splits` | int | 5 | CV splits for group_cv |
| `elbow_min_rel_gain` | float | 0.02 | Minimum relative gain for elbow |
| `elbow_patience` | int | 3 | Patience for elbow detection |

---

### select_k_auto

```python
sift.select_k_auto(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_path: List[str],
    config: AutoKConfig,
    *,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    task: Literal["regression", "classification"] = "regression"
) -> Tuple[int, List[str], pd.DataFrame]
```

Automatically select optimal K via cross-validation or holdout.

**Returns:**

- `best_k` (int): Optimal number of features
- `selected` (List[str]): Selected feature names
- `diagnostics` (pd.DataFrame): Evaluation diagnostics for each K tried

---

### select_k_elbow

```python
sift.select_k_elbow(
    objective_path: np.ndarray,
    min_k: int = 5,
    max_k: int = 100,
    min_rel_gain: float = 0.02,
    patience: int = 3
) -> Tuple[int, pd.DataFrame]
```

Select K using elbow detection on objective curve.

**Returns:**

- `best_k` (int): K at elbow point
- `diagnostics` (pd.DataFrame): Objective, gain, and relative-gain diagnostics by K

---

## Smart Sampling

### SmartSamplerConfig

```python
@dataclass
class sift.SmartSamplerConfig:
    sample_frac: float = 0.1
    group_col: Optional[str] = None
    time_col: Optional[str] = None
    min_per_group: int = 2
    pilot_sample_size: int = 50000
    leverage_batch_size: int = 200000
    svd_sample_size: Optional[int] = None
    weight_clip_quantile: float = 0.99
    residual_weight_cap: float = 0.4
    uniform_floor: float = 0.05
    anchor_fn: Optional[Callable] = None
    anchor_max_share: float = 0.4
    random_state: Optional[int] = 42
    verbose: bool = True
```

Configuration for smart sampling.

---

### smart_sample

```python
sift.smart_sample(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    config: Optional[SmartSamplerConfig] = None,
    **kwargs
) -> pd.DataFrame
```

Perform leverage-based smart sampling. `**kwargs` override fields on
`SmartSamplerConfig`, such as `group_col`, `time_col`, `sample_frac`,
`anchor_fn`, and `random_state`.

**Example:**

```python
from sift import smart_sample

sampled = smart_sample(
    df,
    feature_cols=["f1", "f2", "f3"],
    y_col="target",
    group_col="user_id",
    time_col="timestamp",
    sample_frac=0.15
)
```

---

### panel_config

```python
sift.panel_config(
    group_col: str,
    time_col: str,
    sample_frac: float = 0.15
) -> SmartSamplerConfig
```

Create configuration for temporal panel data.

---

### cross_section_config

```python
sift.cross_section_config(
    sample_frac: float = 0.2
) -> SmartSamplerConfig
```

Create configuration for cross-sectional data.

---

## Permutation Importance

### permutation_importance

```python
sift.permutation_importance(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    sample_weight: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    *,
    scoring: Union[str, Callable] = "neg_mse",
    n_repeats: int = 10,
    permute_method: Literal["auto", "global", "within_group", "block", "circular_shift"] = "auto",
    block_size: Union[int, str] = "auto",
    n_jobs: int = -1,
    parallel_backend: Literal["threads", "processes"] = "threads",
    random_state: Optional[int] = None,
) -> pd.DataFrame
```

Compute time-aware permutation importance.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | estimator | required | Fitted model with predict method |
| `X` | DataFrame or ndarray | required | Feature matrix |
| `y` | Series or ndarray | required | Target variable |
| `sample_weight` | ndarray | None | Optional sample weights |
| `groups` | ndarray | None | Group labels for group-aware permutation |
| `time` | ndarray | None | Time values for block and circular-shift permutation |
| `permute_method` | str | "auto" | Permutation strategy |
| `scoring` | str or callable | "neg_mse" | Scoring function |
| `n_repeats` | int | 10 | Number of permutation repeats |
| `block_size` | int or str | "auto" | Block size for block permutation |
| `random_state` | int | None | Random seed |
| `n_jobs` | int | -1 | Parallel jobs |
| `parallel_backend` | str | "threads" | Joblib backend preference |

**Returns:**

`pd.DataFrame` with one row per feature and columns such as `feature`,
`importance_mean`, `importance_std`, and per-repeat importance values.

**Example:**

```python
from sift import permutation_importance

# Standard permutation importance
importance = permutation_importance(model, X_test, y_test, n_repeats=10)

# Time-aware with circular shift
importance = permutation_importance(
    model, X_test, y_test,
    permute_method="circular_shift",
    groups=player_ids,
    time=timestamps,
    n_repeats=10
)
```
