# Advanced Features Guide

This guide covers advanced features in SIFT for handling complex real-world scenarios.

## Table of Contents

- [Time Series Feature Selection](#time-series-feature-selection)
- [Panel/Grouped Data](#panelgrouped-data)
- [Automatic K Selection](#automatic-k-selection)
- [Smart Sampling for Large Datasets](#smart-sampling-for-large-datasets)
- [Multi-Target Feature Selection](#multi-target-feature-selection)
- [Custom CV Strategies](#custom-cv-strategies)
- [Handling Categorical Features](#handling-categorical-features)
- [Sample Weighting](#sample-weighting)
- [Combining Multiple Methods](#combining-multiple-methods)

---

## Time Series Feature Selection

Time series data requires special handling to avoid data leakage and preserve temporal structure.

### The Problem with Standard Methods

Standard feature selection methods assume i.i.d. data:
- Bootstrap sampling breaks temporal dependencies
- Random CV splits can use future data to predict past
- Standard permutation importance destroys temporal patterns

### Solutions in SIFT

#### 1. Block Bootstrap for Stability Selection

```python
from sift import StabilitySelector

selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    task="regression",
    block_size="auto",          # sqrt(n_per_group) by default
    block_method="moving"       # "moving", "circular", or "stationary"
)

# Block bootstrap preserves temporal structure
selector.fit(
    X, y,
    groups=entity_ids,          # e.g., player_id, stock_ticker
    time=timestamps             # temporal ordering
)
```

**Block Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `moving` | Sample consecutive blocks, may repeat | Stationary series |
| `circular` | Wraps around at boundaries | Reducing edge effects |
| `stationary` | Random block lengths (geometric) | Non-stationary series |

#### 2. Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit
import sift

# Respects temporal ordering
result = sift.catboost_select(
    X, y,
    k=20,
    cv=TimeSeriesSplit(n_splits=5),
    algorithm="forward"
)
```

**Custom Blocked Time Series Split:**

```python
import numpy as np

class BlockedTimeSeriesSplit:
    """Time series split with gap between train and validation."""

    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + self.gap
            val_end = val_start + fold_size
            yield np.arange(train_end), np.arange(val_start, min(val_end, n))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Usage
cv = BlockedTimeSeriesSplit(n_splits=5, gap=10)
result = sift.catboost_select(X, y, k=20, cv=cv)
```

#### 3. Time-Aware Permutation Importance

```python
from sift import permutation_importance

# Standard permutation destroys temporal patterns
# Use circular shift instead
importance = permutation_importance(
    model, X_test, y_test,
    permute_method="circular_shift",  # Preserves temporal structure
    groups=entity_ids,
    time=timestamps,
    n_repeats=10
)
```

**Permutation Methods:**

| Method | Description | Preserves |
|--------|-------------|-----------|
| `global` | Standard shuffle | Nothing |
| `within_group` | Shuffle within groups | Group structure |
| `block` | Shuffle blocks within groups | Some temporal |
| `circular_shift` | Rotate within groups | Full temporal |

#### 4. Time-Based Holdout for Auto-K

```python
from sift import select_mrmr, AutoKConfig

config = AutoKConfig(
    k_method="evaluate",
    strategy="time_holdout",    # Use last portion as validation
    val_frac=0.2,              # Last 20% for validation
    metric="rmse"
)

selected = select_mrmr(
    X, y, k="auto",
    task="regression",
    time=timestamps,
    auto_k_config=config
)
```

---

## Panel/Grouped Data

Panel data has hierarchical structure (e.g., multiple entities over time).

### The Problem

- Same entity in train and test → data leakage
- Standard bootstrap samples rows → group leakage
- Need to respect group boundaries

### Solutions in SIFT

#### 1. Group-Aware Bootstrap

```python
from sift import StabilitySelector

selector = StabilitySelector(n_bootstrap=50, threshold=0.6)

# Bootstrap samples entire groups, not individual rows
selector.fit(X, y, groups=player_ids)
```

#### 2. GroupKFold Cross-Validation

```python
from sklearn.model_selection import GroupKFold
import sift

# Each entity appears in only one fold
result = sift.catboost_select(
    X, y, k=20,
    cv=GroupKFold(n_splits=5),
    group_col="player_id"       # Column in X
)
```

#### 3. Leave-One-Group-Out

```python
from sklearn.model_selection import LeaveOneGroupOut

# Extreme case: each entity is its own test set
cv = LeaveOneGroupOut()
result = sift.catboost_select(X, y, k=20, cv=cv, group_col="entity_id")
```

#### 4. Group-Aware Auto-K

```python
from sift import select_mrmr, AutoKConfig

config = AutoKConfig(
    k_method="evaluate",
    strategy="group_cv",       # Use GroupKFold
    n_splits=5
)

selected = select_mrmr(
    X, y, k="auto",
    task="regression",
    groups=entity_ids,
    auto_k_config=config
)
```

#### 5. Combined Time + Group

```python
# For true panel data: groups + time
selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    block_size="auto",
    block_method="moving",
)
selector.fit(
    X, y,
    groups=player_ids,
    time=game_dates,
)
```

---

## Automatic K Selection

SIFT provides two methods for automatically determining the optimal number of features.

### Method 1: Cross-Validation Evaluation

```python
from sift import select_mrmr, AutoKConfig

config = AutoKConfig(
    k_method="evaluate",       # Use CV to find best K
    strategy="time_holdout",   # or "group_cv"
    max_k=100,                 # Maximum K to try
    min_k=5,                   # Minimum K to try
    metric="rmse",             # Evaluation metric
    val_frac=0.2               # For time_holdout
)

selected = select_mrmr(X, y, k="auto", task="regression",
                       time=timestamps, auto_k_config=config)
```

### Method 2: Elbow Detection

```python
config = AutoKConfig(
    k_method="elbow",          # Find elbow in objective curve
    max_k=100,
    elbow_min_rel_gain=0.02,   # Minimum relative improvement
    elbow_patience=3           # Stop after N steps without improvement
)

selected = select_cefsplus(X, y, k="auto", auto_k_config=config)
```

### How Elbow Detection Works

1. Build feature path up to max_k
2. Compute objective (MI, score, etc.) at each K
3. Find point where marginal gain drops below threshold
4. Return features up to that point

```python
from sift import build_cache, select_k_elbow, compute_objective_for_path

# Manual elbow detection
cache = build_cache(X)
objective = compute_objective_for_path(cache, y, feature_path)
elbow_k, diagnostics = select_k_elbow(
    objective,
    min_k=5,
    max_k=100,
    min_rel_gain=0.02,
    patience=3
)
```

### Using with CatBoost

```python
import sift

# Automatic K search
result = sift.catboost_select(
    X, y,
    task="regression",
    k=None,            # Search for optimal K
    min_features=5,    # Lower bound
    # Will try k from min_features to n_features
)

print(f"Best K: {result.best_k}")
print(f"Scores by K: {result.scores_by_k}")

# Parsimonious selection
simple = result.features_within_tolerance(tolerance=0.01)
```

---

## Smart Sampling for Large Datasets

For very large datasets, smart sampling reduces computation while preserving statistical properties.

### The Problem

- 1M+ rows makes bootstrap slow
- Random sampling may miss rare patterns
- Need to preserve leverage/influence patterns

### Smart Sampling Algorithm

1. Compute leverage scores (hat matrix diagonal)
2. Sample with probability proportional to leverage
3. Weight samples by inverse probability

```python
from sift import smart_sample, panel_config

# Direct usage
sampled = smart_sample(
    df,
    feature_cols=["f1", "f2", "f3"],
    y_col="target",
    group_col="user_id",
    time_col="timestamp",
    sample_frac=0.15
)
```

### Pre-configured Samplers

```python
from sift import panel_config, cross_section_config

# For panel data
config = panel_config(
    group_col="user_id",
    time_col="timestamp",
    sample_frac=0.15
)

# For cross-sectional data
config = cross_section_config(sample_frac=0.2)
```

### Integration with Stability Selection

```python
from sift import StabilitySelector, panel_config

selector = StabilitySelector(
    threshold=0.6,
    use_smart_sampler=True,
    sampler_config=panel_config("user_id", "timestamp", sample_frac=0.15)
)

selector.fit(df, y)
```

---

## Multi-Target Feature Selection

When you have multiple targets, caching avoids redundant computation.

### Building a Cache

```python
from sift import build_cache, select_cached

# Build cache once (expensive)
cache = build_cache(X, subsample=50000, random_state=42)

# Select for multiple targets (cheap)
selected_y1 = select_cached(cache, y1, k=10, method="mrmr_quot")
selected_y2 = select_cached(cache, y2, k=10, method="jmi")
selected_y3 = select_cached(cache, y3, k=15, method="cefsplus")
```

### What Gets Cached

- Gaussian-transformed features (Z)
- Correlation matrix (Rxx)
- Valid column indices
- Subsampled row indices

### Available Methods

| Method | Description |
|--------|-------------|
| `mrmr_quot` | mRMR with quotient formula |
| `mrmr_diff` | mRMR with difference formula |
| `jmi` | Joint Mutual Information |
| `jmim` | JMI Maximization |
| `cefsplus` | CEFS+ |

---

## Custom CV Strategies

SIFT's CatBoost selector accepts any sklearn-compatible CV splitter.

### Time-Based Splits

```python
from sklearn.model_selection import TimeSeriesSplit
import sift

# Expanding window
cv = TimeSeriesSplit(n_splits=5)

# With gap
cv = TimeSeriesSplit(n_splits=5, gap=10, max_train_size=1000)

result = sift.catboost_select(X, y, k=20, cv=cv)
```

### Group-Based Splits

```python
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Each group in exactly one fold
cv = GroupKFold(n_splits=5)

# Random group splits
cv = GroupShuffleSplit(n_splits=5, test_size=0.2)

result = sift.catboost_select(X, y, k=20, cv=cv, group_col="entity_id")
```

### Stratified Splits (Classification)

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
result = sift.catboost_select(X, y, k=20, task="classification", cv=cv)
```

### Custom Splitter

```python
class PurgedGroupTimeSeriesSplit:
    """
    Time series split with purging to prevent leakage.
    Groups close in time are purged from training.
    """

    def __init__(self, n_splits=5, purge_days=30):
        self.n_splits = n_splits
        self.purge_days = purge_days

    def split(self, X, y=None, groups=None):
        # Implementation
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
```

---

## Handling Categorical Features

### Automatic Detection

```python
from sift import select_mrmr

# DataFrame with categorical columns
df = pd.DataFrame({
    "num1": np.random.randn(100),
    "num2": np.random.randn(100),
    "cat1": np.random.choice(["A", "B", "C"], 100),
    "cat2": pd.Categorical(np.random.choice(["X", "Y"], 100))
})

# Automatically detected from dtype
selected = select_mrmr(df, y, k=3, task="regression")
```

### Explicit Specification

```python
selected = select_mrmr(
    df, y, k=5,
    task="regression",
    cat_features=["cat1", "cat2"],
    cat_encoding="loo"
)
```

### Encoding Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `none` | No encoding (must be numeric) | Pre-encoded data |
| `loo` | Leave-one-out encoding | Default, low overfitting |
| `target` | Target encoding | Large categories |
| `james_stein` | James-Stein shrinkage | Small samples |

### Installation

```bash
pip install -e ".[categorical]"  # Installs category_encoders
```

---

## Sample Weighting

All selectors support sample weights.

### Use Cases

- **Importance weighting:** Recent data more important
- **Imbalanced data:** Upweight minority class
- **Survey data:** Design weights

### Filter Methods

```python
from sift import select_mrmr, select_jmi

# Create weights
weights = np.ones(len(y))
weights[-100:] = 2.0  # Recent data more important

selected = select_mrmr(X, y, k=10, task="regression", sample_weight=weights)
selected = select_jmi(X, y, k=10, task="regression", sample_weight=weights)
```

### Stability Selection

```python
from sift import StabilitySelector

selector = StabilitySelector(threshold=0.6)
selector.fit(X, y, sample_weight=weights)
```

### CatBoost

```python
import sift

# CatBoost handles weights internally
df = X.assign(weight=weights)
result = sift.catboost_select(
    df, y, k=20,
    task="regression",
    sample_weight_col="weight"
)
```

---

## Combining Multiple Methods

### Ensemble Feature Selection

```python
from sift import select_mrmr, select_jmi, select_cefsplus, stability_regression
from collections import Counter

# Run multiple methods
methods = [
    select_mrmr(X, y, k=20, task="regression", verbose=False),
    select_jmi(X, y, k=20, task="regression", verbose=False),
    select_cefsplus(X, y, k=20, verbose=False),
    stability_regression(X, y, k=20, verbose=False)
]

# Count feature occurrences
counts = Counter()
for selected in methods:
    counts.update(selected)

# Select features appearing in multiple methods
ensemble_selected = [f for f, c in counts.items() if c >= 2]
print(f"Ensemble selected {len(ensemble_selected)} features")
```

### Two-Stage Selection

```python
import sift
from sift import select_mrmr

# Stage 1: Fast pre-filter
prefiltered = select_mrmr(X, y, k=100, task="regression", estimator="gaussian")

# Stage 2: Careful selection on reduced set
X_pre = X[prefiltered]
final = sift.catboost_regression(X_pre, y, k=20, algorithm="shap")
```

### Validation-Based Selection

```python
import sift
import numpy as np

# Run selection with different random seeds
all_selections = []
for seed in range(10):
    result = sift.catboost_select(
        X, y, k=20,
        random_state=seed,
        verbose=False
    )
    all_selections.append(set(result.selected_features))

# Find consensus features
consensus = set.intersection(*all_selections)
print(f"Consensus features ({len(consensus)}): {consensus}")

# Find features selected at least 80% of time
from collections import Counter
counts = Counter()
for s in all_selections:
    counts.update(s)
stable = [f for f, c in counts.items() if c >= 8]
```

---

## Best Practices Summary

1. **Time Series Data:**
   - Use `TimeSeriesSplit` or custom temporal CV
   - Use block bootstrap with `block_method="moving"`
   - Use `circular_shift` for permutation importance

2. **Panel Data:**
   - Always provide `groups` parameter
   - Use `GroupKFold` for CV
   - Enable group-aware bootstrap

3. **Large Datasets:**
   - Use smart sampling with `use_smart_sampler=True`
   - Use gaussian estimators where possible
   - Consider two-stage selection

4. **Production Pipelines:**
   - Use stability selection or ensemble methods
   - Validate with multiple random seeds
   - Save feature importance for monitoring

5. **Feature Interactions:**
   - Use JMI or CatBoost SHAP
   - Consider forward_greedy for small K

6. **Debugging Selection:**
   - Use `verbose=True`
   - Check `feature_importances` in results
   - Compare multiple methods
