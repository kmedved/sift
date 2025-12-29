# sift: feature selection toolbox

`sift` is a feature selection toolbox that brings together minimal-optimal and stability-focused selectors.
It includes **mRMR**, **JMI/JMIM**, **CEFS+** (and related Gaussian-copula variants), and **Stability Selection**.

## Supported selectors

- **mRMR / JMI / JMIM** (classification & regression; pandas, with a polars mRMR path)
- **CEFS+** (plus `mrmr_fcd` / `mrmr_fcq` variants)
- **Stability Selection** (regression & classification)
- **CatBoost-based selection** (SHAP/permutation/forward, optional dependency)

## Installation

This project is not published on PyPI. Install it from source:

```bash
git clone https://github.com/kmedved/sift.git
cd sift
python -m pip install -e .
```

### Extras

```bash
python -m pip install -e ".[all]"
python -m pip install -e ".[polars]"
python -m pip install -e ".[categorical]"
python -m pip install -e ".[numba]"
python -m pip install -e ".[catboost]"
python -m pip install -e ".[test]"
```

## Quick examples

### mRMR / JMI / JMIM (pandas)

```python
import pandas as pd
from sklearn.datasets import make_classification
from sift import mrmr_classif

X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=10,
    n_redundant=40,
)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# mRMR (default)
selected_mrmr = mrmr_classif(X=X, y=y, K=10)

# JMI / JMIM
selected_jmi = mrmr_classif(X=X, y=y, K=10, method="jmi")
selected_jmim = mrmr_classif(X=X, y=y, K=10, method="jmim")
```

### CEFS+ (regression)

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import cefsplus_regression

X, y = make_regression(n_samples=500, n_features=30, n_informative=8, noise=0.1)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y)

selected_cefs = cefsplus_regression(X, y, K=8)
```

### Stability Selection

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import stability_regression

X, y = make_regression(n_samples=300, n_features=25, n_informative=6, noise=0.2)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

selected_stable = stability_regression(
    X,
    y,
    K=10,
    n_bootstrap=30,
    threshold=0.3,
    random_state=42,
    verbose=False,
)
```

### CatBoost feature selection (optional)

```python
import pandas as pd
from sklearn.datasets import make_regression
import sift

X, y = make_regression(n_samples=300, n_features=25, n_informative=6, noise=0.2)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
selected = sift.catboost_regression(X, y, K=10, algorithm="forward", prefilter_k=200)
```

#### Usage Examples

**Time Series Data (Your NBA Use Case)**

```python
from sklearn.model_selection import TimeSeriesSplit
import sift

# Time series split - respects temporal ordering
result = sift.catboost_select(
    X, y, K=20,
    cv=TimeSeriesSplit(n_splits=5),
    algorithm='forward',  # Fast forward selection
)
```

**Grouped Data (NBA Players Across Seasons)**

```python
from sklearn.model_selection import GroupKFold
import sift

# Group-aware split - no player appears in both train and val
result = sift.catboost_select(
    X, y, K=20,
    cv=GroupKFold(n_splits=5),
    group_col='player_id',
)
```

**Group-Aware Bootstrap Stability**

```python
import sift

# Bootstrap samples GROUPS, not rows - prevents leakage
result = sift.catboost_select(
    X, y, K=20,
    group_col='player_id',
    use_stability=True,
    n_bootstrap=20,
    stability_threshold=0.6,
)
print(result.stability_scores)  # Selection frequencies
```

**Fast Forward Selection**

```python
import sift

# O(K) model fits - much faster than RFE
selected = sift.catboost_regression(
    X, y, K=20,
    algorithm='forward',
    prefilter_k=200,
)
```

**True Greedy Forward Selection**

```python
import sift

# O(K * n_features) fits - most principled but slowest
selected = sift.catboost_regression(
    X, y, K=10,  # Use for small K
    algorithm='forward_greedy',
    prefilter_k=50,  # Prefilter first to make feasible
)
```

**Full Control with K Search**

```python
import sift

result = sift.catboost_select(
    X, y,
    task='regression',
    K=None,                    # Search for optimal K
    min_features=5,
    n_splits=5,
    eval_metric='RMSE',
    prefilter_k=200,
    verbose=True,
)

print(result.selected_features)
print(result.scores_by_k)
print(result.feature_importances.head(10))

# Get minimal feature set within 1% of best score
parsimonious = result.features_within_tolerance(tolerance=0.01)
```

**Algorithm Comparison**

| Algorithm | Speed | Accuracy | Use When | Complexity |
| --- | --- | --- | --- | --- |
| forward | ⚡⚡⚡⚡ | ★★ | Fast exploration, time series | O(K) fits |
| prediction | ⚡⚡⚡ | ★ | Quick RFE, large datasets | O(splits×steps) |
| permutation | ⚡⚡ | ★★ | Production RFE | O(splits×steps) |
| shap | ⚡ | ★★★ | Final feature set, interpretability | O(splits×steps) |
| forward_greedy | 🐢 | ★★★ | Small K, final refinement | O(K×n_features) |

**Key Features**

- Custom CV splitter: pass any sklearn splitter via `cv=` parameter.
- Forward selection: fast importance-based selection (O(K) fits).
- Group-aware bootstrap: stability selection samples groups when `group_col` provided.
- One-shot RFE: runs `select_features` once per split at `min_k`.
- Prefilter inside CV: avoids data leakage.
- Always explicit metrics: `eval_metric` and `loss_function` always set.
- Feature aggregation: combines across splits by frequency + mean-rank.
- Final model on full data: importance computed from model trained on all data.
- Multi-class SHAP: properly handles multi-dimensional SHAP output.

**Custom Splitter Examples**

```python
# Time series (no shuffle, temporal order)
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
cv = TimeSeriesSplit(n_splits=5)

# Group K-fold (each group in exactly one fold)
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)

# Leave-one-group-out
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()

# Blocked time series (custom)
class BlockedTimeSeriesSplit:
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
```

### Smart sampling (for stability selection)

Smart sampling is an optional, leverage-based subsampler that can reduce the
data size before running stability selection. It works on pandas DataFrames
and returns approximate inverse-probability weights internally, so you should
not pass `sample_weight` when `use_smart_sampler=True`.

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import StabilitySelector, panel_config

X, y = make_regression(n_samples=10000, n_features=40, n_informative=10, noise=0.3)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y
df["user_id"] = [f"user_{i % 200}" for i in range(len(df))]
df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="h")

selector = StabilitySelector(
    threshold=0.6,
    use_smart_sampler=True,
    sampler_config=panel_config("user_id", "timestamp", sample_frac=0.15),
)
selector.fit(df, y)
```

You can also call the sampler directly if you want access to the sampled
DataFrame and its generated weights:

```python
from sift import smart_sample

sampled = smart_sample(
    df,
    feature_cols=[f"f{i}" for i in range(40)],
    y_col="target",
    group_col="user_id",
    time_col="timestamp",
    sample_frac=0.15,
)
```

### mRMR with Polars

```python
import polars as pl
import sift

data = [
    (1.0, 1.0, 1.0, 7.0, 1.5, -2.3),
    (2.0, None, 2.0, 7.0, 8.5, 6.7),
    (2.0, None, 3.0, 7.0, -2.3, 4.4),
    (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
    (4.0, 5.0, 4.0, 7.0, 12.1, -5.2),
]
columns = ["target", "some_null", "feature", "constant", "other_feature", "another_feature"]

df_polars = pl.DataFrame(data=data, schema=columns)
selected = sift.polars.mrmr_regression(df=df_polars, target_column="target", K=2)
```

## Concepts and workflows

### When to use each selector

- **mRMR (and JMI/JMIM)**: Good default when you want a fast, greedy ranking of features
  based on relevance and redundancy. Use `method="jmi"` or `method="jmim"` to emphasize
  multivariate relevance over pairwise redundancy.
- **CEFS+**: Useful when you need a minimal-optimal subset and want more explicit
  balancing of relevance and redundancy for regression problems.
- **Stability Selection**: Prefer this when you want robustness across resamples,
  or you need a tunable tradeoff between sparsity and confidence in feature inclusion.

### Data expectations

- **Pandas inputs**: Most selectors accept `pandas.DataFrame` for features and
  `pandas.Series` (or array-like) for targets.
- **Polars inputs**: `sift.polars.mrmr_regression` supports `polars.DataFrame` and
  a `target_column` name.
- **Targets**: Classification targets should be discrete labels, regression targets
  should be continuous.
- **Missing values**: Prefer imputed or filtered datasets. For stability selection,
  missing values can materially affect bootstrap results.

### Output format

Most selectors return a list of feature names (or indices) in ranked order. For
stability selection, you can additionally inspect selection frequencies via the
`StabilitySelector` object when using the class-based API.

## API overview

### mRMR / JMI / JMIM

```python
from sift import mrmr_classif, mrmr_regression

# classification
selected = mrmr_classif(
    X,
    y,
    K=20,
    method="mrmr",  # "jmi" or "jmim"
    n_jobs=-1,
    verbose=False,
)

# regression
selected = mrmr_regression(X, y, K=20, n_jobs=-1, verbose=False)
```

**Key parameters**

- `K`: Number of features to select.
- `method`: `"mrmr"`, `"jmi"`, or `"jmim"` for classification.
- `n_jobs`: Parallelism for mutual information estimation.
- `verbose`: Toggle progress reporting.

### CEFS+

```python
from sift import cefsplus_regression

selected = cefsplus_regression(
    X,
    y,
    K=15,
    n_jobs=-1,
    verbose=False,
)
```

**Key parameters**

- `K`: Number of features to select.
- `n_jobs`: Parallelism for internal scoring.
- `verbose`: Toggle progress reporting.

### Stability Selection

```python
from sift import stability_classif, stability_regression

selected_cls = stability_classif(
    X,
    y,
    K=20,
    n_bootstrap=50,
    threshold=0.5,
    sample_fraction=0.75,
    random_state=0,
    verbose=False,
)

selected_reg = stability_regression(
    X,
    y,
    K=20,
    n_bootstrap=50,
    threshold=0.5,
    sample_fraction=0.75,
    random_state=0,
    verbose=False,
)
```

**Key parameters**

- `n_bootstrap`: Number of bootstrap resamples.
- `threshold`: Minimum selection frequency for inclusion.
- `sample_fraction`: Fraction of samples to draw per bootstrap.
- `random_state`: Ensures reproducibility across runs.

### Class-based stability API

```python
from sift import StabilitySelector

selector = StabilitySelector(
    threshold=0.5,
    n_bootstrap=50,
    sample_fraction=0.75,
    random_state=0,
)
selector.fit(X, y)
selected = selector.get_support()
```

Use the class-based API when you need more control (for example, toggling smart
sampling, inspecting support scores, or reusing fitted selectors).

## Practical guidance

### Reproducibility

- Set `random_state` for stability selection and any randomness inside sampling
  or resampling routines.
- Keep `n_bootstrap` fixed when comparing different runs.

### Performance tips

- Start with small `K` values and increase once you have a stable baseline.
- Use `n_jobs=-1` to parallelize mutual information estimations.
- When working with very wide datasets, consider running a coarse pre-filter
  (e.g., variance threshold) before applying mRMR or CEFS+.

### Categorical features

If you have categorical features, install the `categorical` extra and ensure
categories are encoded consistently. This helps avoid unstable mutual
information estimates due to mixed data types.

### Choosing `threshold` for stability selection

A higher `threshold` yields a smaller, more conservative feature set; a lower
threshold yields more features but with less certainty. Start around `0.5` and
tune according to your downstream model’s tolerance for false positives.

## Project layout

- `sift/`: core library code.
- `tests/`: unit tests and regression tests.
- `setup.py`: packaging metadata.

## Development

```bash
python -m pip install -e ".[test]"
pytest
```
