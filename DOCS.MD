# Sift Documentation

`sift` is a Python feature-selection toolbox for fast filter methods, stability
selection, time-aware permutation importance, Boruta/Boruta-Shap, smart sampling,
and optional CatBoost-based selection.

This document describes the current public API in this repository. The primary
imports are exposed from `sift`; advanced utilities live in submodules such as
`sift.selection`, `sift.sampling`, `sift.estimators`, and `sift._permute`.

## Installation

Install from the repository root:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[categorical]"  # category_encoders for target/LOO encoders
python -m pip install -e ".[catboost]"     # CatBoost selectors and SHAP-style backends
python -m pip install -e ".[test]"         # pytest
python -m pip install -e ".[all]"          # category_encoders + catboost
```

Core dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, and
`numba`.

## Quick Start

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import select_mrmr, select_jmi, select_jmim, select_cefsplus, select_cefsplus_binary

X_arr, y = make_regression(
    n_samples=500,
    n_features=30,
    n_informative=8,
    noise=0.2,
    random_state=0,
)
X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])

mrmr = select_mrmr(X, y, k=10, task="regression", verbose=False)
jmi = select_jmi(X, y, k=10, task="regression", verbose=False)
jmim = select_jmim(X, y, k=10, task="regression", verbose=False)
cefs = select_cefsplus(X, y, k=10, verbose=False)
binary_cefs = select_cefsplus_binary(X, (y > y.mean()).astype(int), k=10, verbose=False)
```

Most selectors return feature names for `pandas.DataFrame` inputs and synthetic
names such as `x0`, `x1`, ... for ndarray inputs. Stability wrappers can also
return integer indices.

## Feature Map

| Feature | Main entry points | Best for |
| --- | --- | --- |
| mRMR | `select_mrmr` | Fast relevance/redundancy filtering for regression or classification. |
| JMI | `select_jmi` | Joint mutual-information ranking, usually less pairwise than mRMR. |
| JMIM | `select_jmim` | Conservative JMI variant using a minimum joint score. |
| CEFS+ | `select_cefsplus` | Regression-only Gaussian-copula filter with log-det objective. |
| Binary CEFS+ | `select_cefsplus_binary` | Binary logistic-score filter using conditional Bernoulli deviance proxy. |
| Auto-k | `AutoKConfig`, `k="auto"`, `select_k_auto`, `select_k_elbow` | Choosing feature count by holdout, group CV, or objective elbow. |
| Gaussian cache | `build_cache`, `select_cached`, `FeatureCache` | Reusing transformed features across many targets or methods. |
| Stability selection | `StabilitySelector`, `stability_regression`, `stability_classif` | Robust feature sets across bootstrap resamples. |
| Smart sampling | `smart_sample`, `SmartSamplerConfig`, sampler presets | Reducing large panel/cross-section datasets before selection. |
| Boruta | `BorutaSelector`, `select_boruta` | All-relevant feature selection against shadow features. |
| Boruta-Shap | `select_boruta_shap`, `BorutaSelector(importance="shap")` | SHAP-backed Boruta using CatBoost or a compatible estimator. |
| Permutation importance | `permutation_importance` | Model-agnostic importance with grouped/time-aware permutations. |
| CatBoost selection | `catboost_select`, `catboost_regression`, `catboost_classif` | Feature selection with CatBoost RFE, SHAP, permutation, or forward search. |
| Low-level estimators | `sift.estimators.*`, `sift.selection.*` | Custom pipelines and advanced experimentation. |

Top-level exports:

```python
from sift import (
    FeatureCache,
    build_cache,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_jmi,
    select_jmim,
    select_mrmr,
    AutoKConfig,
    FeaturePathEvaluationResult,
    evaluate_feature_path,
    select_k_auto,
    select_k_elbow,
    compute_objective_for_path,
    BorutaSelector,
    BorutaResult,
    select_boruta,
    select_boruta_shap,
    MRMRSelector,
    JMISelector,
    JMIMSelector,
    CEFSPlusSelector,
    CEFSPlusBinarySelector,
    permutation_importance,
    SmartSamplerConfig,
    smart_sample,
    panel_config,
    cross_section_config,
    StabilitySelector,
    stability_regression,
    stability_classif,
    catboost_select,
    catboost_regression,
    catboost_classif,
)
```

## Data Requirements

`select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, and
`select_cefsplus_binary` accept
`pandas.DataFrame` or `numpy.ndarray` features and `pandas.Series` or array-like
targets.

Numeric requirements:

- Classic filter selectors require numeric features after optional categorical
  encoding.
- Gaussian/cache-based selectors require numeric features. Use categorical
  encoding before building a cache.
- Regression targets must be finite numeric values.
- Classification targets may be strings, categoricals, or numeric labels, but
  missing labels are not allowed.
- Feature matrices may contain non-finite values; the internal preprocessing
  mean-imputes them for most numeric selectors.

Sample weights:

- `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`,
  `select_cefsplus_binary`, `select_boruta`, `select_k_auto`,
  `StabilitySelector.fit`, `stability_regression`, `stability_classif`, and
  `permutation_importance` accept sample weights.
- `select_cefsplus_binary` also accepts `class_weight=None`, `"balanced"`, or a
  class-label dictionary. Resolved sample/class weights are used by the
  logistic score test and by `cat_encoding="loo_logit"`.
- Weights must be finite, non-negative, length-matched to `X`, and contain at
  least one positive value.
- For `k="auto"` with `k_method="evaluate"`, sample weights are used both for
  fitting the downstream prefix-scoring model and for validation metrics.
- When `StabilitySelector(use_smart_sampler=True)` is used, do not pass
  `sample_weight`; the sampler creates its own `sample_weight` column.

Categorical encoding:

- Filter selectors support `cat_features=[...]` and `cat_encoding="loo"`,
  `"target"`, `"james_stein"`, `"loo_logit"`, or `"none"`.
- `cat_encoding="loo"`, `"target"`, and `"james_stein"` require the
  `category_encoders` package. `cat_encoding="loo_logit"` is built into SIFT
  and is intended for binary targets.
- Function-style filter selectors are safe by default: supervised categorical
  encodings (`"loo"`, `"target"`, `"james_stein"`, and `"loo_logit"`) raise unless
  `allow_full_data_target_encoding=True` is passed. That flag preserves the old
  full-dataset encoding behavior and should only be used when leakage has been
  handled outside the selector.
- If `cat_features` is omitted and `X` is a DataFrame, object/category/string
  columns are auto-detected for Gaussian/JMI/CEFS+ paths in the places where the
  code supports it.
- CatBoost selectors handle categorical columns natively and treat
  object/string columns as categorical by default.

## Core Filter Selectors

Fixed `k` is an upper bound for filter selectors, not an exact-size guarantee.
`select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`,
`select_cefsplus_binary`, and `select_cached` return up to `k` features. They
can return fewer when fewer valid candidates remain after constant-feature
filtering, non-positive relevance filtering, `top_m`, explicit correlation
pruning, or other validity checks. CatBoost selectors document their own
exact-`k` behavior separately.

### `select_mrmr`

```python
from sift import select_mrmr

selected = select_mrmr(
    X,
    y,
    k=20,                    # maximum number of features to return
    task="regression",        # "regression" or "classification"
    estimator="classic",      # "classic" or "gaussian"
    relevance="f",            # regression: "f", "rf"; classification: "f", "ks", "rf"
    formula="quotient",       # "quotient" or "difference"
    top_m=None,               # default max(5*k, 250)
    sample_weight=None,
    subsample=50_000,
    random_state=0,
    n_jobs=1,
    mrmr_backend="auto",      # "auto", "serial", "blas", or "processes"
    verbose=False,
)
```

mRMR greedily balances feature relevance against redundancy with already selected
features.

Key options:

- `task`: required. Use `"regression"` for continuous `y` and
  `"classification"` for labels.
- `estimator="classic"`: uses relevance scores plus Pearson-style redundancy.
- `estimator="gaussian"`: uses rank-Gaussian transforms and Gaussian mutual
  information proxies. This path is regression-only and does not apply
  correlation pruning by default.
- `formula="quotient"`: score is relevance divided by mean redundancy.
- `formula="difference"`: score is relevance minus mean redundancy.
- `relevance="f"`: F-statistic style relevance.
- `relevance="ks"`: weighted KS relevance for classification only.
- `relevance="rf"`: random-forest relevance.
- `top_m`: prefilters candidates by relevance before greedy selection.
- `subsample`: caps rows used by the filter path. Use `None` to use all rows.
- `mrmr_backend`: controls the mRMR redundancy/cache execution path. `"serial"`
  is the original Numba path, `"blas"` uses single-process vectorized
  correlation updates for classic mRMR, and `"processes"` uses joblib process
  workers. `"auto"` resolves to `"serial"` when `n_jobs=1` and `"processes"`
  otherwise.
- `n_jobs`: worker count for `mrmr_backend="processes"`. `n_jobs=0` is invalid.
  For Gaussian mRMR, this only parallelizes rank-Gaussian cache construction
  when no prebuilt `cache` is supplied; a prebuilt cache is reused as-is.

Classification example:

```python
selected = select_mrmr(
    X,
    y_labels,
    k=15,
    task="classification",
    relevance="ks",
    verbose=False,
)
```

Gaussian mRMR example:

```python
selected = select_mrmr(
    X,
    y,
    k=25,
    task="regression",
    estimator="gaussian",
    formula="difference",
    verbose=False,
)
```

Result object example:

```python
result = select_mrmr(
    X,
    y,
    k=20,
    task="regression",
    return_result=True,
    verbose=False,
)

selected = result.selected_features
selected_indices = result.selected_indices
metadata = result.selector_metadata
ranking = result.get_feature_ranking()
```

`return_result=True` is available on `select_mrmr`, `select_jmi`,
`select_jmim`, and `select_cefsplus`. The default remains `list[str]`. The
result object currently contains selected feature names, best-effort input-column
indices, compact selector metadata, and a conservative diagnostic ranking table.
Unknown diagnostics, such as relevance for selectors that did not expose it, are
reported as `NaN` rather than guessed.

### `select_jmi`

```python
from sift import select_jmi

selected = select_jmi(
    X,
    y,
    k=20,
    task="regression",
    estimator="auto",         # "auto", "binned", "r2", "ksg", or "gaussian"
    relevance="f",
    top_m=None,
    sample_weight=None,
    verbose=False,
)
```

JMI scores candidates by joint mutual information with each already selected
feature and the target. It uses sum aggregation internally.

Estimator behavior:

- `estimator="auto"` resolves to `"r2"` for regression and `"binned"` for
  classification.
- `estimator="binned"` supports classification and regression-style continuous
  targets through quantile bins.
- `estimator="r2"`, `"ksg"`, and `"gaussian"` are regression-only.
- `estimator="ksg"` does not support `sample_weight`; weighted JMI/JMIM should
  use `"r2"` or `"binned"`.
- `estimator="gaussian"` uses the cache-compatible rank-Gaussian path. It does
  not apply correlation pruning by default.

### `select_jmim`

```python
from sift import select_jmim

selected = select_jmim(
    X,
    y,
    k=20,
    task="classification",
    estimator="binned",
    verbose=False,
)
```

JMIM is the conservative JMI variant. It uses the same options as `select_jmi`
but aggregates candidate joint information with a minimum instead of a sum.

### `select_cefsplus`

```python
from sift import select_cefsplus

selected = select_cefsplus(
    X,
    y,
    k=20,
    top_m=None,
    corr_prune=0.95,
    sample_weight=None,
    subsample=50_000,
    random_state=0,
    verbose=False,
)
```

CEFS+ is regression-only. It builds a rank-Gaussian cache, scores candidate
subsets with a log-determinant Gaussian mutual-information proxy, and optionally
prunes highly correlated candidates before selection.

Important options:

- `corr_prune`: candidate pruning threshold. A value of `0.95` drops features
  highly correlated with better-scoring candidates. CEFS+ uses this pruning by
  default. Pass `None` to disable pruning.
- `top_m`: number of relevance-prefiltered candidates to keep before CEFS+.
- `cache`: pass a `FeatureCache` to reuse the expensive transform.

### `select_cefsplus_binary`

```python
from sift import select_cefsplus_binary

selected = select_cefsplus_binary(
    X,
    y_binary,
    k=20,                 # or "auto"
    loss="logloss",
    top_m=None,
    corr_prune=0.95,
    time=None,
    groups=None,
    auto_k_config=None,
    sample_weight=None,
    class_weight=None,
    ridge=1e-4,
    refit_every=1,
    subsample=None,
    verbose=False,
)
```

Binary CEFS+ is a logistic-score analogue of CEFS+: it greedily ranks features by
conditional Bernoulli deviance reduction using a Rao/Fisher score-test update.
It is intended for binary screening, not as the same estimator as Gaussian
log-det CEFS+.

Important options:

- `loss="logloss"`: default Bernoulli/log-loss criterion. `sample_weight` and
  `class_weight` are honored directly. When `cat_encoding="loo_logit"` is used,
  the resolved binary weights also drive the leave-one-out category statistics.
- `loss="weighted_logloss"`: alias for weighted log-loss; requires explicit
  weights.
- `loss="brier"`: compatibility mode that delegates to `select_cefsplus` with
  the binary target cast to float.
- `k="auto"`: supports the same `AutoKConfig` workflow as other filter selectors.
  For `loss="logloss"`, `k_method="evaluate"` scores prefixes with a downstream
  logistic model, and `k_method="elbow"` applies elbow detection to the cumulative
  binary score-test gains. For `loss="brier"`, auto-k delegates to regular CEFS+.
- `ridge`: stabilizes the selected-feature logistic fit and candidate score-test
  information. The default score includes the same ridge term in each candidate
  denominator.
- `refit_every`: refits the selected-feature logistic null every step by default.
  The default exact path uses direct row-level score products. Larger values use
  a block-Gram accelerator: after `top_m` and correlation pruning, SIFT
  materializes the Fisher-weighted Gram and score vector over the remaining
  candidate matrix once per refit block, then scores within-block candidates
  from Schur-complement submatrices with efficient-score adjustment. Treat
  larger values as an approximate speed mode.
- `corr_prune`: absolute weighted feature-correlation pruning threshold in
  `(0, 1]`. Pass `None` to disable pruning.
- `cat_encoding="loo_logit"`: smoothed leave-one-out logit encoding for binary
  categorical features. Function-style selectors still require
  `allow_full_data_target_encoding=True` before fitting supervised encoders on
  the full matrix.
- `CEFSPlusBinarySelector(loss="brier", cat_encoding="loo_logit")` rejects
  categorical inputs because selector-class pre-encoding would otherwise use
  logit encodings while the function API maps brier compatibility mode to
  ordinary `"loo"`. Use `cat_encoding="loo"` for brier compatibility or
  `loss="logloss"` for logistic `loo_logit` encoding.
- `class_weight`: class weights are resolved on the input rows before any
  optional `subsample` draw. Returned metadata records this as
  `class_weight_scope="pre_subsample"`.
- `CEFSPlusBinarySelector` fits supervised categorical encoders against the
  validated binary `0/1` target, including for string labels. Preprocessing
  parameters such as `loss`, `cat_encoding`, `cat_features`, `class_weight`, and
  `loo_*` settings must be set on the estimator or with `set_params()` before
  `fit`; fit-time overrides for those parameters are rejected.

## Scikit-Learn Selector Classes

Sift also exposes thin transformer wrappers around the filter selectors:

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sift import MRMRSelector

pipe = make_pipeline(
    MRMRSelector(k=20, task="regression", verbose=False),
    Ridge(),
)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
```

Available classes:

- `MRMRSelector`
- `JMISelector`
- `JMIMSelector`
- `CEFSPlusSelector`
- `CEFSPlusBinarySelector`

Each class supports `fit`, `transform`, `fit_transform`, and
`get_support(indices=False)`. After fitting, the selectors expose:

- `selected_features_`
- `selected_indices_`
- `feature_names_in_`
- `n_features_in_`

For pandas inputs, `transform` validates that columns match the fitted columns
and order before selecting by stored positions.

Selector classes default to `cat_encoding="none"`. When you explicitly set
`cat_encoding` to `"target"`, `"loo"`, `"james_stein"`, or `"loo_logit"`, the
encoder is fitted only inside `fit(X_train, y_train)` and reused by
`transform(X_test)` without access to test targets. `fit_transform(X_train,
y_train)` returns the selected columns from the y-aware encoded matrix used
during selection, so sklearn pipeline training does not silently switch to
target-blind transform semantics.
These fitted encoders are not cross-fitted; use external cross-fitting or
selector-class `auto_k_mode="nested"` when you need fold-local evaluation.
Function-style selectors still reject supervised full-data categorical encoding
by default unless `allow_full_data_target_encoding=True` is passed explicitly.

## Automatic `k` Selection

The filter selectors accept `k="auto"`:

```python
from sift import AutoKConfig, select_mrmr

config = AutoKConfig(
    k_method="evaluate",      # "evaluate" or "elbow"
    auto_k_mode="prefix_only", # "prefix_only" or selector-class "nested"
    strategy="time_holdout",  # "time_holdout" or "group_cv"
    metric="rmse",            # "auto", "rmse", "mae", "logloss", "error"
    min_k=5,
    max_k=100,
    val_frac=0.2,
)

selected = select_mrmr(
    X,
    y,
    k="auto",
    task="regression",
    time=df["date"].to_numpy(),
    auto_k_config=config,
    verbose=False,
)
```

`AutoKConfig` fields:

- `k_method="evaluate"` evaluates prefixes of the selected path with a simple
  downstream model.
- `k_method="elbow"` chooses where objective gains flatten out. It requires an
  objective path and is currently supported by Gaussian/cache-backed filter
  paths such as CEFS+ and Gaussian mRMR, plus binary CEFS+ log-loss paths using
  cumulative score-test gains. Classic MRMR/JMI/JMIM auto-k paths use
  `k_method="evaluate"`; requesting elbow there raises.
- `auto_k_mode="prefix_only"` is the current public semantic. Selectors build
  one supervised feature path on the rows available to the selector, then
  evaluate prefixes of that fixed path. For `k_method="evaluate"`, the scores
  are not an unbiased estimate of refitting the full selector inside each
  split. Prefix evaluation does honor `sample_weight` when provided.
- `auto_k_mode="nested"` is implemented by the sklearn-style selector classes.
  For each split, the class fits a train-only selector path, uses the same
  `fit_transform` training representation that sklearn pipelines receive,
  transforms validation rows with train-fitted preprocessing, evaluates prefix
  lengths, and refits the final selector on all rows passed to `fit` using the
  chosen `k`. Nested mode currently supports `k_method="evaluate"` only;
  `k_method="elbow"` raises instead of being silently ignored. Function-style
  selectors still reject nested mode and keep the prefix-only contract.
- Selector classes reject `k_method="evaluate"`, `auto_k_mode="prefix_only"`,
  and explicit supervised categorical encoding together, because that would
  evaluate already target-encoded validation rows. Use nested mode or
  pre-encode/cross-fit categoricals outside the selector.
- `strategy="time_holdout"` requires `time` and validates on the latest rows.
- `strategy="group_cv"` requires `groups` and uses `GroupKFold`.
- If `auto_k_config` is omitted, selectors infer `strategy="time_holdout"` when
  `time` is supplied and `strategy="group_cv"` when only `groups` is supplied.
- `metric="auto"` resolves to RMSE for regression and log loss for
  classification.
- `min_k`, `max_k`, `val_frac`, `n_splits`, `random_state`,
  `elbow_min_rel_gain`, and `elbow_patience` control the search.

Advanced helpers:

```python
from sift import select_k_auto, select_k_elbow, compute_objective_for_path

best_k, best_features, diagnostics = select_k_auto(
    X,
    y,
    feature_path=selected_path,
    config=config,
    groups=groups,
    task="classification",
)

best_k, elbow_diagnostics = select_k_elbow(
    objective_path,
    min_k=5,
    max_k=100,
    min_rel_gain=0.02,
    patience=3,
)

objective_path = compute_objective_for_path(cache, y, selected_path)
```

### Explicit feature-path evaluation

Use `evaluate_feature_path` when you already have an ordered feature path and
want to score explicit prefixes without invoking selector `k="auto"`:

```python
from sift import evaluate_feature_path
from sklearn.linear_model import Ridge

path_result = evaluate_feature_path(
    X,
    y,
    feature_path=selected_path,
    k_grid=[5, 10, 20, 40],
    estimator=Ridge(alpha=1.0),
    scoring="rmse",           # "rmse", "mae", or callable
    splitter=(train_idx, val_idx),
    sample_weight=weights,
)

best_k = path_result.best_k
best_features = path_result.features
diagnostics = path_result.diagnostics
```

`evaluate_feature_path` supports DataFrames or arrays, feature names or integer
positions, an estimator template or estimator factory, sample weights, one
explicit train/validation split, sklearn-style splitter objects, or a random
holdout controlled by `val_frac` and `random_state`. It returns
`FeaturePathEvaluationResult` with the resolved path, tested `k` values, average
scores, best `k`, best-prefix features, and diagnostics.

## Gaussian Feature Cache

Use `build_cache` when selecting against many targets or comparing several
Gaussian/cache methods on the same `X`.

```python
from sift import build_cache, select_cached

cache = build_cache(
    X,
    sample_weight=None,
    subsample=50_000,
    random_state=0,
    compute_Rxx=True,
    n_jobs=1,
    rank_backend="serial",    # "serial" or "processes"
)

cefs = select_cached(cache, y, k=30, method="cefsplus")
jmi = select_cached(cache, y, k=30, method="jmi")
jmim = select_cached(cache, y, k=30, method="jmim")
mrmr_q = select_cached(cache, y, k=30, method="mrmr_quot")
mrmr_d = select_cached(cache, y, k=30, method="mrmr_diff")
```

Cached selectors follow the same fixed-`k` contract as the public filter
selectors: `k` is a maximum, and fewer features may be returned when fewer valid
or unpruned candidates remain.

`FeatureCache` stores:

- `Z`: rank-Gaussian transformed features for valid, non-constant columns.
- `Rxx`: optional weighted feature correlation matrix.
- `valid_cols`: original column indices kept after filtering.
- `row_idx`: rows used after optional subsampling.
- `sample_weight`: normalized weights for cached rows.
- `feature_names`: original feature names when available.

Pass `rank_backend="processes"` with `n_jobs` to parallelize the
rank-Gaussian column transforms during cache construction. Passing a prebuilt
cache into `select_mrmr(..., estimator="gaussian")` reuses that cache and does
not rebuild it; build the cache with `build_cache(..., n_jobs=..., rank_backend="processes")`
when you want parallel cache construction.

`select_cached` uses `corr_prune="auto"` by default. In that mode, CEFS+ keeps
its historical `0.95` pruning threshold, while cached Gaussian JMI, JMIM, and
mRMR do not prune correlated candidates and can return `k` features when at
least `k` candidates remain after `top_m` and validity filtering. Pass a float
such as `corr_prune=0.95` to opt into the same pruning for any cached method, or
pass `corr_prune=None` to disable pruning.

`select_cached(..., return_objective=True)` returns `(features, objective_path)`.

## Stability Selection

Stability selection repeatedly fits sparse linear models on bootstrap samples and
keeps features that appear often enough.

### Convenience wrappers

```python
from sift import stability_regression, stability_classif

selected_reg = stability_regression(
    X,
    y,
    k=20,
    threshold=0.6,
    n_bootstrap=50,
    alpha=None,          # estimated by CV when None
    l1_ratio=1.0,        # 1.0=Lasso; <1.0=ElasticNet
    sample_frac=0.5,
    random_state=0,
    verbose=False,
)

selected_cls = stability_classif(
    X,
    y_labels,
    k=20,
    threshold=0.6,
    n_bootstrap=50,
    random_state=0,
    verbose=False,
)
```

For DataFrame inputs, wrappers return feature names by default. For ndarray
inputs, they return integer indices unless `return_indices=False` is passed.

The lower-level convenience function `sift.stability.stability_select` returns
`(selected_indices, selection_frequencies)`:

```python
from sift.stability import stability_select

idx, frequencies = stability_select(
    X,
    y,
    threshold=0.6,
    n_bootstrap=50,
    task="regression",
    random_state=0,
    verbose=False,
)
```

### Class API

```python
from sift import StabilitySelector

selector = StabilitySelector(
    task="regression",
    n_bootstrap=100,
    threshold=0.65,
    max_features=25,
    store_coefs=True,
    random_state=0,
    verbose=False,
)
selector.fit(X, y)

selected_names = selector.selected_feature_names_
info = selector.get_feature_info()
coef_stability = selector.get_coef_stability()
mask = selector.get_support()
X_reduced = selector.transform(X)
```

Important options:

- `n_bootstrap`: number of resamples.
- `sample_frac`: fraction per i.i.d. bootstrap.
- `threshold`: minimum selection frequency.
- `alpha`: regularization strength. If `None`, Sift estimates it by CV.
- `l1_ratio`: regression only; use values below `1.0` for ElasticNet.
- `max_features`: hard cap after thresholding.
- `store_coefs=False`: lowers memory use but disables coefficient stability
  diagnostics and coefficient distribution plots.
- `parallel_backend="threads"`: lower memory overhead; `"processes"` isolates
  work at higher memory cost.

Diagnostics and plotting:

```python
info = selector.get_feature_info()
best_threshold, threshold_scores = selector.tune_threshold(X, y)
selector.set_threshold(best_threshold)

fig, ax = selector.plot_frequencies(top_n=30, show_coef=True)
fig, axes = selector.plot_coef_distributions(top_n=12)
```

### Block bootstrap for panels/time series

If both `groups` and `time` are passed to `fit` or the convenience wrappers,
stability selection uses a group-aware block bootstrap.

```python
selected = stability_regression(
    X,
    y,
    k=30,
    groups=df["player_id"].to_numpy(),
    time=df["game_date"].to_numpy(),
    block_size="auto",
    block_method="moving",    # "moving", "circular", or "stationary"
    verbose=False,
)
```

## Smart Sampling

Smart sampling is an informative subsampler for large datasets. It combines
leverage-score sampling, optional residual-based sampling, uniform coverage, and
anchor rows. It returns a sampled DataFrame with an approximate inverse
probability `sample_weight` column.

```python
from sift import smart_sample, panel_config

config = panel_config(
    group_col="user_id",
    time_col="timestamp",
    sample_frac=0.15,
)

sampled = smart_sample(
    df,
    feature_cols=[c for c in df.columns if c.startswith("f")],
    y_col="target",
    config=config,
)
```

`SmartSamplerConfig` fields:

- `sample_frac`: target fraction of rows to sample.
- `group_col`, `time_col`: optional grouping and ordering columns.
- `min_per_group`: minimum rows to keep per group.
- `pilot_sample_size`: rows used for residual pilot model.
- `leverage_batch_size`: memory/speed tradeoff for leverage scores.
- `svd_sample_size`: optional row subsample for approximate SVD.
- `weight_clip_quantile`: cap extreme inverse-probability weights.
- `residual_weight_cap`: maximum blend weight for residual scores.
- `uniform_floor`: minimum base probability for coverage.
- `anchor_fn`: callable that marks rows to keep preferentially.
- `anchor_max_share`: cap on anchor share within each group quota.
- `random_state`, `verbose`: reproducibility and logging.

Advanced helper:

```python
from sift.sampling.smart import leverage_scores_multi_alpha

scores = leverage_scores_multi_alpha(
    Xs,
    V,
    S,
    s2,
    leverage_batch_size=200_000,
)
```

`leverage_scores_multi_alpha` computes ridge-style leverage scores across
multiple regularization strengths. The main sampler calls it internally after a
randomized SVD.

Use smart sampling inside stability selection:

```python
from sift import StabilitySelector, panel_config

selector = StabilitySelector(
    task="regression",
    use_smart_sampler=True,
    sampler_config=panel_config("user_id", "timestamp", sample_frac=0.15),
    threshold=0.6,
    random_state=0,
    verbose=False,
)
selector.fit(df, y)
```

When `use_smart_sampler=True`, `X` must be a DataFrame. The sampler uses numeric
feature columns only and drops group/time columns from the feature matrix.

### Anchor utilities and presets

The root package exposes `panel_config` and `cross_section_config`. All sampler
presets and anchors are available from `sift.sampling`:

```python
from sift.sampling import (
    no_anchors,
    first_per_group,
    first_and_last_per_group,
    periodic_anchors,
    quantile_anchors,
    event_window_anchors,
    combine_anchors,
    panel_config,
    cross_section_config,
    financial_config,
    medical_config,
)

anchor_fn = combine_anchors(
    first_and_last_per_group,
    periodic_anchors("month"),
    quantile_anchors("usage", quantile=0.95),
    event_window_anchors("team_change", window=3),
)
```

Preset configs:

- `panel_config(group_col, time_col=None, sample_frac=0.15)`: generic grouped
  longitudinal data; anchors first and last per group.
- `cross_section_config(sample_frac=0.15)`: non-grouped cross-sectional data.
- `financial_config(ticker_col="ticker", date_col="date", sample_frac=0.15)`:
  panel data with first/last plus yearly anchors.
- `medical_config(patient_col="patient_id", time_col="visit_date",
  sample_frac=0.20)`: conservative medical longitudinal preset.

## Boruta and Boruta-Shap

Boruta compares each real feature against randomized shadow features. Sift's
implementation supports native importances, SHAP importances, sample weights,
time-aware shadow permutations, categorical encoding, and train/test importance
evaluation for held-out-compatible backends.

### Functional API

```python
from sift import select_boruta, select_boruta_shap

selected = select_boruta(
    X,
    y,
    task="regression",
    importance="native",      # "native" or "shap"
    max_iter=50,
    alpha=0.05,
    perc=100,
    max_features=30,
    shadow_method="auto",
    random_state=0,
    verbose=False,
)

selected_shap = select_boruta_shap(
    X,
    y,
    task="classification",
    max_features=25,
    random_state=0,
    verbose=False,
)
```

Set `return_result=True` to inspect the full `BorutaResult`:

```python
result = select_boruta(X, y, return_result=True, verbose=False)

accepted = result.selected_features()
ranking = result.get_feature_ranking()
accepted_mask = result.accepted_mask
tentative_mask = result.tentative_mask
rejected_mask = result.rejected_mask
```

### Class API

```python
from sift import BorutaSelector

selector = BorutaSelector(
    task="regression",
    importance="native",
    n_estimators="auto",
    max_iter=50,
    max_features=30,
    random_state=0,
    verbose=False,
)
selector.fit(X, y)

selected = selector.selected_features_
mask = selector.get_support()
X_reduced = selector.transform(X)
result = selector.result_()
```

Important options:

- `estimator`: custom estimator. If omitted, native importance uses RandomForest;
  SHAP importance uses CatBoost.
- `n_estimators="auto"`: fast bounded heuristic when Sift creates the default
  estimator.
- `importance="native"`: uses `feature_importances_` or absolute coefficients.
- `importance="shap"`: uses SHAP values. CatBoost is required unless a
  compatible estimator is supplied.
- `alpha`: accept/reject significance level.
- `perc`: shadow percentile threshold. `100` means max shadow feature.
- `resolve_tentative=True`: resolves undecided features at the end.
- `importance_data="train"` or `"test"`: compute importance on fit data or a
  held-out split. `importance_data="test"` is only supported with
  held-out-compatible backends such as `importance="shap"`; native importances
  are model attributes and raise `ValueError` in test mode.
- `shap_sample_size`: optional SHAP row subsample.
- `early_stop_rounds`: stop after repeated no-decision iterations.
- `cat_features` and `cat_encoding`: encode categorical DataFrame columns before
  fitting Boruta.

Shadow permutations:

- `shadow_method="auto"` resolves to `"circular_shift"` when `time` is present
  with or without `groups`, `"within_group"` when only `groups` are present, and
  `"global"` otherwise.
- Explicit methods are `"global"`, `"within_group"`, `"block"`, and
  `"circular_shift"`.
- For `"block"` and `"circular_shift"`, `time` is required. If `time` is
  provided without `groups`, the full dataset is treated as one ordered group.
- `shadow_mode="columns"` permutes each feature independently.
- `shadow_mode="rows"` applies the same row permutation to all columns, preserving
  cross-feature covariance.
- `group_col` and `time_col` can be column names in `X`; they are extracted and
  dropped before fitting.

## Permutation Importance

`permutation_importance` computes model-agnostic feature importance after a model
has already been fit.

For pandas inputs, Sift preserves the DataFrame path when calling
`model.predict`, so feature names and categorical/object dtypes remain available
to models that require them. NumPy inputs keep the ndarray path.

```python
from sklearn.ensemble import RandomForestRegressor
from sift import permutation_importance

model = RandomForestRegressor(random_state=0).fit(X, y)

importance = permutation_importance(
    model,
    X,
    y,
    scoring="neg_rmse",
    n_repeats=20,
    n_jobs=-1,
    parallel_backend="threads",  # "threads" or "processes"
    permute_method="auto",
    groups=df["player_id"].to_numpy(),
    time=df["game_date"].to_numpy(),
    random_state=0,
)
```

Output columns:

- `feature`
- `importance_mean`
- `importance_std`
- `baseline_score`

Scoring options:

- `"neg_mse"`
- `"neg_rmse"`
- `"neg_mae"`
- `"r2"`
- `"accuracy"`
- `"balanced_accuracy"`
- `"neg_error"`
- `"neg_logloss"`
- callable with signature `scoring(y_true, y_pred, sample_weight) -> float`

All built-in scoring strings are higher-is-better; classification error rate
and log loss are therefore exposed as `"neg_error"` and `"neg_logloss"` rather
than `"error"` or `"logloss"`.

Parallel options:

- `n_jobs`: feature chunks to evaluate in parallel.
- `parallel_backend="threads"`: lower memory and serialization overhead for
  many estimators and DataFrame-preserving prediction paths.
- `parallel_backend="processes"`: process isolation when that is preferable,
  with the usual process serialization overhead.

Permutation methods:

- `"auto"`: chooses `"circular_shift"` when `time` is present with or without
  `groups`, `"within_group"` for grouped data without `time`, and `"global"`
  otherwise.
- `"global"`: standard column shuffle.
- `"within_group"`: shuffle within each group.
- `"block"`: shuffle time-ordered blocks within each group. With `time` but no
  `groups`, the full dataset is treated as one ordered group.
- `"circular_shift"`: rotate values within each group. With `time` but no
  `groups`, the full dataset is treated as one ordered group.

## CatBoost Feature Selection

CatBoost support is optional and loaded lazily. Install it with:

```bash
python -m pip install -e ".[catboost]"
```

### Full API

```python
import sift

result = sift.catboost_select(
    X,
    y,
    k=20,                       # exact k; use None to search k
    task="regression",
    algorithm="shap",           # "shap", "permutation", "prediction", "forward", "forward_greedy"
    prefilter_k=200,
    prefilter_method="catboost",
    n_splits=3,
    n_estimators=500,
    eval_metric="RMSE",
    random_state=0,
    verbose=True,
)

selected = result.selected_features
scores = result.scores_by_k
importances = result.feature_importances
```

Algorithm choices:

| Algorithm | Behavior | Relative cost |
| --- | --- | --- |
| `"shap"` | Recursive feature elimination using SHAP importance. | Slowest, most interpretability-focused. |
| `"permutation"` | Recursive elimination using loss-function-change importance. | Medium. |
| `"prediction"` | Recursive elimination using prediction-change importance. | Faster RFE path. |
| `"forward"` | Fit once, rank by importance, evaluate prefixes. | Fast. |
| `"forward_greedy"` | True greedy forward search. | Most expensive for large feature sets. |

Prefiltering:

- `prefilter_k=None`: disable prefiltering.
- `prefilter_method="catboost"`: shallow CatBoost importance, handles
  categoricals.
- `prefilter_method="cefsplus"`: numeric Gaussian-copula prefilter. For
  classification this falls back to mRMR internally.
- `prefilter_method="mrmr"`: numeric mRMR prefilter.
- `prefilter_method="none"`: keep all features.

Cross-validation:

- Pass any sklearn-compatible splitter via `cv=`.
- If `group_col` is set, that column is extracted from `X`, dropped from the
  feature matrix, and passed as groups to supported splitters.
- `sample_weight_col` is extracted and used as CatBoost weights.
- Misspelled `group_col` or `sample_weight_col` names raise `ValueError`
  instead of silently disabling groups or weights.

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

ts_result = sift.catboost_select(
    X,
    y,
    k=20,
    cv=TimeSeriesSplit(n_splits=5),
    algorithm="forward",
)

group_result = sift.catboost_select(
    X,
    y,
    k=20,
    group_col="player_id",
    cv=GroupKFold(n_splits=5),
)
```

Optional stability stage:

```python
result = sift.catboost_select(
    X,
    y,
    k=20,
    group_col="player_id",
    use_stability=True,
    n_bootstrap=20,
    stability_threshold=0.6,
)

stability_scores = result.stability_scores
```

### Convenience wrappers

```python
selected_reg = sift.catboost_regression(
    X,
    y,
    k=20,
    algorithm="forward",
    prefilter_k=200,
    random_state=0,
)

selected_cls = sift.catboost_classif(
    X,
    y_labels,
    k=20,
    algorithm="shap",
    random_state=0,
)
```

### `CatBoostSelectionResult`

`catboost_select` returns a result object with:

- `selected_features`
- `best_k`
- `scores_by_k`
- `scores_std_by_k`
- `feature_importances`
- `features_by_k`
- `stability_scores`
- `prefilter_features`
- `metric`
- `higher_is_better`
- `all_scores`

Methods:

```python
mean, std = result.score_at_k(20)
small_set = result.features_within_tolerance(tolerance=0.01)
fig, ax = result.plot_scores_vs_k()
```

## Low-Level Selection Utilities

Advanced utilities are available from `sift.selection`.

```python
from sift.selection import mrmr_select, jmi_select

idx = mrmr_select(
    X_arr,
    relevance,
    k=20,
    formula="quotient",
    top_m=250,
    sample_weight=weights,
)

jmi_idx = jmi_select(
    X_arr,
    y_arr,
    k=20,
    relevance=relevance,
    mi_estimator="r2",
    aggregation="sum",       # "sum" for JMI, "min" for JMIM
    top_m=250,
    y_kind="continuous",
    sample_weight=weights,
)
```

These functions return integer column indices. They assume you already computed
valid relevance scores and performed input validation.

The objective helper is available when you already have a correlation matrix in
path order:

```python
from sift.selection.objective import objective_from_corr_path

objective = objective_from_corr_path(
    R_path,
    r_path,
    shrink=1e-6,
    eps=1e-12,
)
```

It computes the same cumulative Gaussian information objective used by
`compute_objective_for_path`.

## Low-Level Estimators

The `sift.estimators` package exposes three lazily loaded modules:

- `sift.estimators.relevance`
- `sift.estimators.copula`
- `sift.estimators.joint_mi`

### Relevance estimators

```python
from sift.estimators import relevance

rel_reg = relevance.f_regression(X_arr, y_arr, weights)
rel_cls = relevance.f_classif(X_arr, y_labels, weights)
rel_ks = relevance.ks_classif(X_arr, y_labels, weights)
rel_rf = relevance.rf_regression(X_arr, y_arr, weights)
rel_rf_cls = relevance.rf_classif(X_arr, y_labels, weights)
```

Available relevance methods:

- `f_regression`
- `f_classif`
- `ks_classif`
- `rf_regression`
- `rf_classif`

### Copula and Gaussian MI utilities

```python
from sift.estimators import copula

z = copula.weighted_rank_gauss_1d(x, weights)
Z = copula.weighted_rank_gauss_2d(X_arr, weights)
R = copula.weighted_correlation_matrix(Z, weights, backend="auto")
mi = copula.gaussian_mi_from_corr(r)
keep = copula.greedy_corr_prune(candidates, R, scores, threshold=0.95)
```

Important helpers:

- `FeatureCache`
- `build_cache`
- `weighted_rank_gauss_1d`
- `weighted_rank_gauss_2d`
- `weighted_correlation_matrix`
- `weighted_correlation_matrix_blas`
- `weighted_correlation_matrix_numba`
- `weighted_corr_with_vector`
- `gaussian_mi_from_corr`
- `greedy_corr_prune`

### Joint mutual-information estimators

```python
from sift.estimators import joint_mi

mi = joint_mi.r2_joint_mi(x1, x2, y, weights)
bins = joint_mi.quantile_bin_matrix(X_arr, n_bins=10)
```

Useful functions include:

- `binned_joint_mi`
- `binned_joint_mi_indexed`
- `binned_joint_mi_indexed_prebinned`
- `r2_joint_mi`
- `r2_joint_mi_indexed`
- `ksg_joint_mi`
- `quantile_bin_matrix`
- `quantile_bin_matrix_indexed`

The indexed variants are used internally to avoid repeatedly materializing
candidate arrays during greedy JMI/JMIM selection.

## Internal Preprocessing Utilities

Most users do not need these directly, but they are useful when building custom
selection flows on top of the low-level loops.

```python
from sift._impute import mean_impute
from sift._preprocess import (
    infer_higher_is_better,
    best_score_from_dict,
    to_numpy,
    extract_feature_names,
    ensure_weights,
    validate_inputs,
    check_regression_only,
    resolve_jmi_estimator,
    subsample_xy,
    encode_categoricals,
)

X_filled = mean_impute(X_arr, copy=True)
w = ensure_weights(sample_weight, n=len(y), normalize=True)
X_arr, y_arr, feature_names = validate_inputs(X, y, task="regression")
X_sub, y_sub, w_sub, row_idx = subsample_xy(
    X_arr,
    y_arr,
    subsample=50_000,
    random_state=0,
    sample_weight=w,
    return_idx=True,
)
```

Utility behavior:

- `mean_impute`: fills non-finite feature values column-wise.
- `infer_higher_is_better`: resolves metric direction for common metrics.
- `best_score_from_dict`: returns the best finite score from a dictionary.
- `to_numpy`: converts pandas, objects with `to_pandas()`, lists, and arrays to
  numpy.
- `extract_feature_names`: returns DataFrame columns when present.
- `ensure_weights`: validates and optionally mean-normalizes sample weights.
- `validate_inputs`: converts, validates, imputes, and encodes classification
  labels to integer codes.
- `check_regression_only`: blocks regression-only estimators for classification.
- `resolve_jmi_estimator`: maps `estimator="auto"` to `"r2"` or `"binned"`.
- `subsample_xy`: row-subsamples features, target, and weights.
- `encode_categoricals`: target/LOO/James-Stein encoding through
  `category_encoders`.

## Permutation Utilities

The internal permutation module is public enough for advanced use:

```python
from sift._permute import (
    resolve_permutation_method,
    build_group_info,
    permute_array,
    permute_rows,
    permute_matrix,
)

method = resolve_permutation_method("auto", groups=groups, time=time)
group_info = build_group_info(groups, time)
shadow = permute_matrix(
    X_arr,
    method=method,
    group_info=group_info,
    block_size="auto",
    seed=0,
    axis="columns",
)
```

Methods are `"auto"`, `"global"`, `"within_group"`, `"block"`, and
`"circular_shift"`. Axes are `"columns"` and `"rows"`.

## Common Workflows

### Fast numeric baseline

```python
selected = select_mrmr(
    X,
    y,
    k=50,
    task="regression",
    top_m=500,
    verbose=False,
)
```

### Regression-focused Gaussian workflow

```python
cache = build_cache(X, compute_Rxx=True, random_state=0)
cefs = select_cached(cache, y, k=50, method="cefsplus")
jmi = select_cached(cache, y, k=50, method="jmi")
```

### Grouped time-series workflow

```python
groups = df["player_id"].to_numpy()
time = df["game_date"].to_numpy()

selected = select_mrmr(
    X,
    y,
    k="auto",
    task="regression",
    groups=groups,
    time=time,
    auto_k_config=AutoKConfig(strategy="time_holdout", max_k=80),
    verbose=False,
)

importance = permutation_importance(
    fitted_model,
    X[selected],
    y,
    groups=groups,
    time=time,
    permute_method="auto",
)
```

### High-confidence all-relevant selection

```python
result = select_boruta(
    X,
    y,
    task="regression",
    max_iter=100,
    perc=100,
    return_result=True,
    random_state=0,
    verbose=False,
)

accepted = result.selected_features()
ranking = result.get_feature_ranking()
```

### Large panel data

```python
selector = StabilitySelector(
    task="regression",
    use_smart_sampler=True,
    sampler_config=panel_config("entity_id", "timestamp", sample_frac=0.10),
    threshold=0.6,
    n_bootstrap=50,
    max_features=50,
    random_state=0,
    verbose=False,
)
selector.fit(df, y)
selected = selector.selected_feature_names_
```

### Benchmark Gates

Benchmark scripts live under `benchmarks/` and are opt-in, not part of normal
pytest runs:

```bash
python benchmarks/run_benchmarks.py --quick --output benchmarks/latest.json
python benchmarks/run_benchmarks.py --full --output benchmarks/latest-full.json
```

The shared JSON schema records baseline/current wall time, optional peak memory,
semantic parity, benchmark kind, and promotion status. Promotion rows are
intended as gates; informational/smoke/parity rows are coverage and comparison
signals, not proof that an optimization was promoted. Generated benchmark JSON
files are local artifacts and are not committed by default.

## Return Types

Filter selectors:

- `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`,
  `select_cefsplus_binary`,
  `select_cached` return `list[str]` when feature names are available.
- `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, and
  `select_cefsplus_binary` return `FilterSelectionResult` when
  `return_result=True`.
- Low-level `mrmr_select` and `jmi_select` return integer indices.

Stability selection:

- `StabilitySelector.selected_feature_names_`: names.
- `StabilitySelector.selected_features_`: integer indices.
- `get_feature_info()`: DataFrame with feature, frequency, mean coefficient, and
  selected flag.
- `get_coef_stability()`: DataFrame with coefficient mean/std/CV.

Boruta:

- `select_boruta(..., return_result=False)`: list of accepted feature names.
- `select_boruta(..., return_result=True)`: `BorutaResult`.
- `BorutaResult.get_feature_ranking()`: DataFrame with feature, mean importance,
  hit count, and status.

Permutation importance:

- DataFrame sorted by `importance_mean` descending.

CatBoost:

- Convenience wrappers return `list[str]`.
- `catboost_select` returns `CatBoostSelectionResult`.

## Performance Notes

- Use `top_m` for wide datasets. Most classic/Gaussian filter selectors default
  `top_m` to `max(5*k, 250)`. `select_cefsplus_binary` leaves `top_m=None` as
  "all finite candidates"; for `k="auto"` it builds the binary path to
  `AutoKConfig.max_k`. Set `top_m` explicitly for wide binary screens.
- Use `subsample` for large row counts. Most filter selectors default to a
  `50_000` row cap; `select_cefsplus_binary` defaults to `subsample=None`.
- Build a `FeatureCache` once when running multiple Gaussian/cache selectors or
  selecting against many targets.
- Set `compute_Rxx=True` in `build_cache` when you will reuse the cache many
  times; it costs more up front but avoids repeated correlation construction.
- For classic mRMR, compare `mrmr_backend="blas"` and
  `mrmr_backend="processes"` on your data with `benchmarks/bench_mrmr.py`.
  Defaults stay serial to avoid surprise process and memory overhead. Use
  `--blas-threads 1,2,4` to sweep native BLAS thread limits; `n_jobs` controls
  only the process backend.
- Use `StabilitySelector(store_coefs=False)` for very wide datasets when
  coefficient diagnostics are not needed.
- Use CatBoost `algorithm="forward"` for quick exploration and
  `"forward_greedy"` only for smaller candidate sets.
- Use `prefilter_k` before expensive CatBoost SHAP/RFE selectors.

## Reproducibility

Set `random_state` wherever available:

- `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`,
  `select_cefsplus_binary`
- `build_cache`
- `AutoKConfig`
- `StabilitySelector` and wrappers
- `smart_sample`
- `select_boruta` / `BorutaSelector`
- `permutation_importance`
- `catboost_select` and wrappers

For exact comparisons, also keep `subsample`, `top_m`, `n_bootstrap`,
`feature_counts`, and CV splitters fixed.

## Development

Install test dependencies and run tests:

```bash
python -m pip install -e ".[test]"
pytest
```

Useful source layout:

- `sift/__init__.py`: top-level exports.
- `sift/api.py`: mRMR, JMI/JMIM, CEFS+, cache-facing public API.
- `sift/selection/`: greedy loops, CEFS+ loop, auto-k helpers, objectives.
- `sift/estimators/`: relevance, copula/Gaussian MI, joint MI estimators.
- `sift/stability.py`: stability selection and block bootstrap.
- `sift/sampling/`: smart sampler and anchor presets.
- `sift/boruta.py`: Boruta and Boruta-Shap.
- `sift/importance.py`: time-aware permutation importance.
- `sift/catboost.py`: optional CatBoost feature selection.
- `tests/`: smoke, regression, optimization, and behavior tests.
