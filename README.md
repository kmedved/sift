# SIFT: Feature Selection Toolbox

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SIFT** is a comprehensive feature selection library that brings together minimal-optimal and stability-focused selectors. It provides multiple state-of-the-art feature selection algorithms for both regression and classification tasks, with advanced support for time-series data, grouped/panel data, and weighted samples.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
  - [mRMR (Minimum Redundancy Maximum Relevance)](#mrmr-minimum-redundancy-maximum-relevance)
  - [JMI / JMIM (Joint Mutual Information)](#jmi--jmim-joint-mutual-information)
  - [CEFS+ (Conditional Entropy Feature Selection)](#cefs-conditional-entropy-feature-selection)
  - [Stability Selection](#stability-selection)
  - [Boruta](#boruta)
  - [CatBoost-based Selection](#catboost-based-selection)
- [Advanced Features](#advanced-features)
  - [Automatic K Selection](#automatic-k-selection)
  - [Time Series Support](#time-series-support)
  - [Grouped/Panel Data](#groupedpanel-data)
  - [Smart Sampling](#smart-sampling)
  - [Categorical Features](#categorical-features)
  - [Sample Weights](#sample-weights)
- [Algorithm Selection Guide](#algorithm-selection-guide)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multiple Algorithms**: mRMR, JMI/JMIM, CEFS+, Stability Selection, Boruta, and CatBoost-based selection
- **Both Tasks**: Full support for regression and classification
- **Time-Series Aware**: Block bootstrap, temporal holdout validation, and time-aware permutation
- **Group/Panel Data**: Group-aware bootstrap and cross-validation for hierarchical data
- **Automatic K Selection**: Built-in methods to automatically determine optimal feature count
- **Sample Weights**: Full support for weighted samples throughout
- **Categorical Features**: Multiple encoding strategies (leave-one-out, target, James-Stein)
- **Performance Optimized**: Numba JIT compilation, caching, and parallelization
- **Scikit-learn Compatible**: `StabilitySelector` and `BorutaSelector` implement sklearn's transformer interface

## Installation

SIFT is not published on PyPI. Install from source:

```bash
git clone https://github.com/kmedved/sift.git
cd sift
pip install -e .
```

### Optional Dependencies

```bash
# All optional dependencies
pip install -e ".[all]"

# Individual extras
pip install -e ".[catboost]"     # CatBoost-based selection
pip install -e ".[categorical]"  # Advanced categorical encoding (category_encoders)
pip install -e ".[test]"         # Testing dependencies (pytest)
```

## Quick Start

### Basic Feature Selection

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import select_mrmr, select_jmi, select_cefsplus

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=50, n_informative=10, noise=0.1)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(50)])

# mRMR selection
selected = select_mrmr(X, y, k=10, task="regression")
print(f"mRMR selected: {selected}")

# JMI selection
selected = select_jmi(X, y, k=10, task="regression")
print(f"JMI selected: {selected}")

# CEFS+ selection (regression only)
selected = select_cefsplus(X, y, k=10)
print(f"CEFS+ selected: {selected}")
```

### Classification Example

```python
from sklearn.datasets import make_classification
from sift import select_mrmr, select_jmi

X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=20)
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(50)])

# mRMR for classification
selected = select_mrmr(X, y, k=10, task="classification")

# JMI with different relevance measure
selected = select_jmi(X, y, k=10, task="classification", relevance="ks")
```

### Stability Selection

```python
from sift import stability_regression, StabilitySelector

# Function API
selected = stability_regression(
    X, y, k=15,
    n_bootstrap=50,
    threshold=0.6,
    random_state=42
)

# Class-based API (sklearn compatible)
selector = StabilitySelector(
    threshold=0.6,
    n_bootstrap=50,
    task="regression",
    random_state=42
)
selector.fit(X, y)
X_selected = selector.transform(X)
print(f"Selected feature names: {selector.selected_feature_names_}")
```

### CatBoost-based Selection

```python
import sift

# Simple API
selected = sift.catboost_regression(X, y, k=15, algorithm="forward")

# Full control with automatic K search
result = sift.catboost_select(
    X, y,
    task="regression",
    k=None,  # Search for optimal K
    algorithm="shap",
    prefilter_k=100,
    verbose=True
)

print(f"Best K: {result.best_k}")
print(f"Selected: {result.selected_features}")
print(f"Scores by K: {result.scores_by_k}")

# Get minimal feature set within 1% of best score
parsimonious = result.features_within_tolerance(tolerance=0.01)
```

## Algorithms

### mRMR (Minimum Redundancy Maximum Relevance)

mRMR selects features that have high relevance to the target while minimizing redundancy among selected features.

```python
from sift import select_mrmr

selected = select_mrmr(
    X, y, k=20,
    task="regression",           # or "classification"
    estimator="classic",         # "classic" or "gaussian" (regression only)
    formula="quotient",          # "quotient" (rel/red) or "difference" (rel-red)
    relevance="f",               # Regression: "f" or "rf"; classification: "f", "ks", or "rf"
    top_m=250,                   # Pre-filter to top_m by relevance
    verbose=True
)
```

**Parameters:**
- `estimator="classic"`: Uses F-statistic for relevance and Pearson correlation for redundancy
- `estimator="gaussian"`: Fast Gaussian MI proxy via correlation (regression only)
- `formula="quotient"`: score = relevance / mean(redundancy)
- `formula="difference"`: score = relevance - mean(redundancy)

### JMI / JMIM (Joint Mutual Information)

JMI considers the joint information between candidate features, selected features, and the target.

```python
from sift import select_jmi, select_jmim

# JMI: score(f) = Σ I(f, s; y) - emphasizes multivariate relevance
selected = select_jmi(
    X, y, k=20,
    task="regression",
    estimator="auto",    # "gaussian", "binned", "ksg", "r2", or "auto"
    relevance="f",
    verbose=True
)

# JMIM: score(f) = min I(f, s; y) - conservative variant
selected = select_jmim(X, y, k=20, task="regression")
```

**Estimators:**
- `"gaussian"`: Fast Gaussian copula MI proxy (regression only)
- `"binned"`: Histogram-based MI estimation
- `"ksg"`: Kraskov-Stögbauer-Grassberger nearest-neighbor estimator
- `"r2"`: Quick MI proxy via R² transformation
- `"auto"`: Automatically selects based on task

### CEFS+ (Conditional Entropy Feature Selection)

CEFS+ uses log-determinant Gaussian MI with efficient incremental updates. Regression only.

```python
from sift import select_cefsplus

selected = select_cefsplus(
    X, y, k=20,
    corr_prune=0.95,     # Prune highly correlated features
    top_m=250,           # Pre-filter threshold
    verbose=True
)
```

### Stability Selection

Stability selection identifies features that are consistently selected across bootstrap resamples.

```python
from sift import StabilitySelector, stability_regression, stability_classif

# Class-based API
selector = StabilitySelector(
    n_bootstrap=50,          # Number of bootstrap iterations
    sample_frac=0.5,         # Fraction of samples per bootstrap
    threshold=0.6,           # Minimum selection frequency
    alpha=None,              # Regularization (None = auto via CV)
    l1_ratio=1.0,            # 1.0 = Lasso, <1.0 = ElasticNet
    task="regression",       # or "classification"
    max_features=None,       # Optional cap on features
    random_state=42
)

selector.fit(X, y)

# Get selected features
feature_mask = selector.get_support(indices=False)  # Boolean mask
feature_indices = selector.get_support(indices=True)  # Integer indices
feature_names = selector.selected_feature_names_  # Selected feature names

# Feature information
info = selector.get_feature_info()  # DataFrame with frequencies, ranks

# Coefficient stability (mean, std across bootstraps)
coef_stability = selector.get_coef_stability()

# Tune threshold via cross-validation
best_threshold = selector.tune_threshold(X, y, thresholds=[0.4, 0.5, 0.6, 0.7])

# Visualization
selector.plot_frequencies(top_n=30)
selector.plot_coef_distributions(top_n=12)
```

### Boruta

Boruta is a wrapper method that creates shadow features and iteratively removes features that perform worse than random.

```python
from sift import BorutaSelector, select_boruta, select_boruta_shap

# Function API
selected = select_boruta(X, y, estimator="rf", max_iter=100, random_state=42)

# SHAP-based variant
selected = select_boruta_shap(X, y, estimator="lightgbm", max_iter=50)

# Class-based API
selector = BorutaSelector(estimator="rf", max_iter=100, random_state=42)
selector.fit(X, y)
X_selected = selector.transform(X)
```

### CatBoost-based Selection

Wrapper-based selection using CatBoost with multiple algorithms.

```python
import sift
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

# Simple APIs
selected = sift.catboost_regression(X, y, k=20, algorithm="forward")
selected = sift.catboost_classif(X, y, k=20, algorithm="shap")

# Full control API
result = sift.catboost_select(
    X, y,
    task="regression",
    k=20,                              # Number of features (None = search)
    algorithm="forward",               # Selection algorithm
    cv=TimeSeriesSplit(n_splits=5),   # Custom CV splitter
    eval_metric="RMSE",               # Evaluation metric
    prefilter_k=100,                  # Two-stage pre-filtering
    group_col="player_id",            # For group-aware bootstrap
    use_stability=True,               # Enable stability selection
    n_bootstrap=20,                   # Bootstrap iterations
    stability_threshold=0.6,          # Selection frequency threshold
    verbose=True
)
```

**Algorithms:**
| Algorithm | Speed | Description |
|-----------|-------|-------------|
| `forward` | Fast | Forward selection via iterative importance (O(K) fits) |
| `forward_greedy` | Slow | True greedy forward selection (O(K×n_features) fits) |
| `shap` | Medium | SHAP importance with RFE |
| `permutation` | Medium | Loss-function-change importance with RFE |
| `prediction` | Fast | Quick RFE via prediction changes |

## Advanced Features

### Automatic K Selection

All main selectors support `k="auto"` for automatic feature count selection.

```python
from sift import select_mrmr, AutoKConfig

# Configure auto-k behavior
config = AutoKConfig(
    k_method="evaluate",      # "evaluate" (CV) or "elbow" (diminishing returns)
    strategy="time_holdout",  # "time_holdout" or "group_cv"
    metric="rmse",           # Evaluation metric
    max_k=100,               # Maximum features to consider
    min_k=5,                 # Minimum features
    val_frac=0.2,            # Validation fraction (time_holdout)
    n_splits=5,              # CV splits (group_cv)
    elbow_min_rel_gain=0.02, # Elbow detection threshold
    elbow_patience=3         # Elbow detection patience
)

# Use with time information
selected = select_mrmr(
    X, y, k="auto",
    task="regression",
    time=timestamps,          # Required for time_holdout
    auto_k_config=config,
    verbose=True
)

# Use with group information
selected = select_mrmr(
    X, y, k="auto",
    task="regression",
    groups=group_ids,         # Required for group_cv
    auto_k_config=AutoKConfig(strategy="group_cv"),
    verbose=True
)
```

### Time Series Support

SIFT provides comprehensive time-series support across all selectors.

```python
from sklearn.model_selection import TimeSeriesSplit
import sift

# Stability selection with block bootstrap
from sift import StabilitySelector

selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    task="regression",
    block_size="auto",        # Auto-determined block size
    block_method="moving",    # "moving", "circular", or "stationary"
    random_state=42
)

# Block bootstrap respects temporal structure
selector.fit(
    X, y,
    groups=player_ids,        # Group identifier
    time=timestamps           # Temporal ordering
)

# CatBoost with time series CV
result = sift.catboost_select(
    X, y, k=20,
    cv=TimeSeriesSplit(n_splits=5),
    algorithm="forward"
)

# Time-aware permutation importance
from sift import permutation_importance

importance = permutation_importance(
    model, X, y,
    permute_method="circular_shift",  # Preserves temporal structure
    groups=group_ids,
    time=timestamps,
    n_repeats=10
)
```

**Permutation Methods:**
- `"global"`: Standard random shuffle
- `"within_group"`: Shuffle within groups
- `"block"`: Block permutation within groups
- `"circular_shift"`: Rotation within groups (preserves temporal structure)

### Grouped/Panel Data

For hierarchical data (e.g., players across seasons), SIFT supports group-aware operations.

```python
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
import sift

# Group-aware CV for CatBoost
result = sift.catboost_select(
    X, y, k=20,
    cv=GroupKFold(n_splits=5),
    group_col="player_id"           # Column name in X
)

# Group-resampled stability selection
result = sift.catboost_select(
    X, y, k=20,
    group_col="player_id",
    use_stability=True,
    n_bootstrap=20,
    stability_threshold=0.6
)
print(result.stability_scores)  # Selection frequencies

# Stability selector with group bootstrap
selector = StabilitySelector(n_bootstrap=50, threshold=0.6)
selector.fit(X, y, groups=player_ids)  # Samples entire groups
```

### Smart Sampling

For large datasets, smart sampling reduces data size while preserving information.

```python
from sift import StabilitySelector, panel_config, cross_section_config, smart_sample

# Panel data configuration
selector = StabilitySelector(
    threshold=0.6,
    use_smart_sampler=True,
    sampler_config=panel_config(
        group_col="user_id",
        time_col="timestamp",
        sample_frac=0.15
    )
)
selector.fit(df, y)

# Direct smart sampling
from sift import smart_sample

sampled_df = smart_sample(
    df,
    feature_cols=feature_names,
    y_col="target",
    group_col="user_id",
    time_col="timestamp",
    sample_frac=0.15
)

# Pre-configured samplers
from sift import panel_config, cross_section_config

# For temporal panel data
config = panel_config("user_id", "timestamp", sample_frac=0.15)

# For cross-sectional data
config = cross_section_config(sample_frac=0.2)
```

### Categorical Features

SIFT automatically detects and encodes categorical features.

```python
from sift import select_mrmr

# Automatic detection from DataFrame dtypes
selected = select_mrmr(
    X, y, k=20,
    task="regression",
    cat_encoding="loo"  # Leave-one-out encoding (default)
)

# Explicit categorical features
selected = select_mrmr(
    X, y, k=20,
    task="regression",
    cat_features=["category_col", "string_col"],
    cat_encoding="target"  # Target encoding
)
```

**Encoding Options:**
- `"none"`: No encoding (features must be numeric)
- `"loo"`: Leave-one-out encoding (default)
- `"target"`: Target encoding
- `"james_stein"`: James-Stein encoding

### Sample Weights

All selectors support sample weights for handling imbalanced data or importance weighting.

```python
from sift import select_mrmr, StabilitySelector

# Function API
selected = select_mrmr(
    X, y, k=20,
    task="regression",
    sample_weight=weights
)

# Class API
selector = StabilitySelector(threshold=0.6)
selector.fit(X, y, sample_weight=weights)
```

## Algorithm Selection Guide

| Algorithm | Speed | Best For | Task Support |
|-----------|-------|----------|--------------|
| mRMR (classic) | Fast | Quick baseline, large datasets | Both |
| mRMR (gaussian) | Very Fast | Regression with many features | Regression |
| JMI | Medium | Multivariate interactions | Both |
| JMIM | Medium | Conservative selection | Both |
| CEFS+ | Medium | Minimal-optimal subset | Regression |
| Stability | Medium | Robust selection, noisy data | Both |
| Boruta | Slow | All-relevant features | Both |
| CatBoost forward | Fast | Time series, quick exploration | Both |
| CatBoost SHAP | Slow | Interpretability, accuracy | Both |

**Recommendations:**

1. **Start with mRMR**: Fast baseline, good default choice
2. **Need robustness?** Use Stability Selection
3. **Time series data?** Use CatBoost with TimeSeriesSplit or Stability with block bootstrap
4. **Panel/grouped data?** Use group-aware bootstrap or GroupKFold
5. **Want all relevant features?** Use Boruta
6. **Need interpretability?** Use CatBoost SHAP
7. **Very large datasets?** Use smart sampling + gaussian estimators

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `select_mrmr(X, y, k, task, ...)` | Minimum Redundancy Maximum Relevance |
| `select_jmi(X, y, k, task, ...)` | Joint Mutual Information |
| `select_jmim(X, y, k, task, ...)` | JMI Maximization (conservative) |
| `select_cefsplus(X, y, k, ...)` | Conditional Entropy Feature Selection+ |
| `stability_regression(X, y, k, ...)` | Stability selection for regression |
| `stability_classif(X, y, k, ...)` | Stability selection for classification |
| `select_boruta(X, y, ...)` | Boruta feature selection |
| `select_boruta_shap(X, y, ...)` | SHAP-based Boruta |
| `catboost_regression(X, y, k, ...)` | CatBoost selection for regression |
| `catboost_classif(X, y, k, ...)` | CatBoost selection for classification |
| `catboost_select(X, y, ...)` | Full control CatBoost selection |

### Classes

| Class | Description |
|-------|-------------|
| `StabilitySelector` | Sklearn-compatible stability selector |
| `BorutaSelector` | Sklearn-compatible Boruta selector |
| `FeatureCache` | Cache for multi-target selection |
| `AutoKConfig` | Configuration for automatic K selection |
| `SmartSamplerConfig` | Configuration for smart sampling |
| `CatBoostSelectionResult` | Result container for CatBoost selection |

### Utility Functions

| Function | Description |
|----------|-------------|
| `build_cache(X, ...)` | Build feature cache for repeated selection |
| `select_cached(cache, y, k, ...)` | Select from cached features |
| `permutation_importance(model, X, y, ...)` | Time-aware permutation importance |
| `smart_sample(df, ...)` | Leverage-based smart sampling |
| `panel_config(...)` | Create panel data sampler config |
| `cross_section_config(...)` | Create cross-section sampler config |
| `select_k_auto(...)` | Automatic K selection via CV |
| `select_k_elbow(...)` | Automatic K selection via elbow method |
| `compute_objective_for_path(...)` | Compute objective along feature path |

## Project Structure

```
sift/
├── __init__.py           # Public API exports
├── api.py                # mRMR, JMI, JMIM, CEFS+ selection
├── stability.py          # Stability selection
├── boruta.py             # Boruta feature selection
├── catboost.py           # CatBoost-based selection
├── importance.py         # Permutation importance
├── _preprocess.py        # Input validation, encoding
├── _impute.py            # Missing value imputation
├── _permute.py           # Permutation strategies
├── estimators/           # MI and relevance estimators
│   ├── relevance.py      # F-stat, KS, RF relevance
│   ├── copula.py         # Gaussian copula, caching
│   └── joint_mi.py       # JMI/JMIM MI estimators
├── selection/            # Selection algorithms
│   ├── cefsplus.py       # CEFS+ implementation
│   ├── loops.py          # mRMR, JMI loops
│   ├── auto_k.py         # Automatic K selection
│   └── objective.py      # Objective computation
└── sampling/             # Smart sampling
    ├── smart.py          # Leverage-based sampling
    └── anchors.py        # Anchor strategies
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/kmedved/sift.git
cd sift
pip install -e ".[test]"

# Run tests
pytest

# Run specific test file
pytest tests/test_smoke.py -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2023 smazzanti

## Acknowledgments

SIFT implements algorithms from the feature selection literature:
- mRMR: Peng et al. (2005) "Feature selection based on mutual information"
- JMI/JMIM: Yang & Moody (1999), Bennasar et al. (2015)
- CEFS: Brown et al. (2012) "Conditional likelihood maximisation"
- Stability Selection: Meinshausen & Bühlmann (2010)
- Boruta: Kursa & Rudnicki (2010)
