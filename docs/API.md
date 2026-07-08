# SIFT API Reference

This page is a standalone reference for the public SIFT API in the 0.7.0
surface. For deeper examples and option notes, see the canonical manual in
[`DOCS.MD`](../DOCS.MD).

## Public Surface

| Area | Main entry points |
| --- | --- |
| Fixed-k filters | `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, `select_cefsplus_binary` |
| q-calibrated knockoffs | `select_fdr`, `KnockoffSelector`, `KnockoffSelectionResult`, `sample_knockoffs` |
| Caching | `build_cache`, `select_cached`, `FeatureCache` |
| Automatic k | `AutoKConfig`, `select_k_auto`, `select_k_elbow`, `select_k_penalized_objective` |
| Selector classes | `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector`, `KnockoffSelector` |
| Stability selection | `StabilitySelector`, `stability_regression`, `stability_classif` |
| Sampling | `smart_sample`, `SmartSamplerConfig`, `panel_config`, `cross_section_config` |
| Model importance | `permutation_importance` |
| Boruta | `BorutaSelector`, `BorutaResult`, `select_boruta`, `select_boruta_shap` |
| Optional CatBoost | `catboost_select`, `catboost_regression`, `catboost_classif` |

CatBoost entry points are lazy exports from `sift`; importing `sift` does not
require the `catboost` extra.

## Shared Selector Behavior

Function-style filter selectors accept pandas DataFrames or NumPy arrays. With
DataFrame input, selected features are returned as column labels; with ndarray
input, synthetic names such as `x0` are used.

Fixed-k filter selectors treat `k` as an upper bound. They can return fewer than
`k` features after constant-feature filtering, relevance screening, correlation
pruning, or non-positive objective checks.

Most filter selectors support:

- `sample_weight` for non-negative row weights.
- `cat_features` and `cat_encoding` for categorical preprocessing.
- `subsample` and `random_state` for Gaussian/cache-backed paths.
- `return_result=True` for a `FilterSelectionResult` where supported.

Supervised categorical encodings are conservative by default. When a function
selector would fit target encoders on the full dataset, pass
`allow_full_data_target_encoding=True` only if leakage is handled outside SIFT.

## Filter Functions

### `select_mrmr`

```python
from sift import select_mrmr

selected = select_mrmr(
    X,
    y,
    k=20,
    task="regression",        # "regression" or "classification"
    estimator="classic",      # "classic" or "gaussian"
    formula="quotient",       # "quotient" or "difference"
    relevance="f",            # regression: "f", "rf"; classification: "f", "ks", "rf"
    top_m=None,
    sample_weight=None,
    subsample=50_000,
    random_state=0,
    n_jobs=1,
    mrmr_backend="auto",
    verbose=False,
)
```

mRMR greedily balances target relevance against redundancy with already selected
features. `estimator="gaussian"` is a fast regression-only path built on the
Gaussian copula cache. `formula="quotient"` scores relevance divided by
redundancy; `formula="difference"` scores relevance minus redundancy.

### `select_jmi` and `select_jmim`

```python
from sift import select_jmi, select_jmim

jmi_features = select_jmi(
    X,
    y,
    k=20,
    task="regression",
    estimator="auto",         # "auto", "gaussian", "binned", "ksg", or "r2"
    relevance="f",
    top_m=None,
    sample_weight=None,
    verbose=False,
)

jmim_features = select_jmim(X, y, k=20, task="regression", verbose=False)
```

JMI uses joint mutual-information style scoring to prefer complementary
features. JMIM is the conservative variant that uses a minimum joint score
against already selected features.

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

CEFS+ is a regression-only Gaussian-copula filter that uses a log-determinant
conditional information objective. `corr_prune` removes highly correlated
candidates from the greedy path.

### `select_cefsplus_binary`

```python
from sift import select_cefsplus_binary

selected = select_cefsplus_binary(
    X,
    y_binary,
    k=20,
    loss="logloss",           # "logloss" or "brier"
    class_weight=None,        # None, "balanced", or a class-weight dict
    ridge=1e-4,
    refit_every=1,
    top_m=None,
    corr_prune=0.95,
    sample_weight=None,
    verbose=False,
)
```

Binary CEFS+ follows a logistic or Brier score path for Bernoulli-like targets.
It validates a binary target and honors both `sample_weight` and
`class_weight`.

## Knockoff FDR

### `select_fdr`

```python
from sift import select_fdr

result = select_fdr(
    X,
    y,
    q=0.1,
    statistic="relevance",    # "relevance" or tie-safe "cefsplus"
    n_draws=1,
    eta=0.5,
    offset=1,                 # 1 = knockoff+, 0 = modified knockoff threshold
    s_method="equi",          # "equi", "mvr", or "me"
    min_eig=1e-3,
    screen_pairs=2000,
    statistic_options=None,
    feature_groups=None,
    sample_weight=None,
    subsample=50_000,
    cache=None,
    random_state=0,
    n_jobs=1,
    verbose=False,
)

trusted = result.selected_features
ranking = result.get_feature_ranking()
metadata = result.selector_metadata
```

`select_fdr` selects a set by target FDR level `q` rather than by fixed `k`. It
builds or reuses a `FeatureCache`, samples second-order Gaussian-copula
knockoffs, computes antisymmetric `W` statistics, and applies the knockoff+
threshold.

The 0.7.0 implementation intentionally reports plug-in validity metadata:

- `fdr_control="approximate_plugin"`
- `validity_model="gaussian_copula_plugin"`
- `weighted_model` when row weights were used in cache/statistic estimation

This means exact Model-X FDR is claimed only under the fitted Gaussian-copula
feature model and valid swap-antisymmetric statistics. With estimated
correlations, shrinkage, weights, or derandomization, interpret the result as an
approximate practical knockoff filter.

Important options:

- `statistic="relevance"` is the fast default.
- `statistic="cefsplus"` enables the tie-safe greedy CEFS+ statistic. It accepts
  `statistic_options={"path_depth": int, "min_gain_ratio": float}`.
- `s_method="equi"` is fastest. `s_method="mvr"` and `"me"` use diagonal
  coordinate-descent objectives and can improve power on correlated designs.
- `n_draws > 1` redraws knockoffs and selects features with frequency at least
  `eta`; `threshold` is then `None` and `selection_frequency` is populated.
- `offset=1` is the knockoff+ threshold. `offset=0` is less conservative and is
  best read as modified-knockoff or mFDR-style control.
- `feature_groups` thresholds group-level antisymmetric statistics and expands
  selected groups back to member features. This is group discovery, not exact
  feature-level FDR inside a selected group.
- `cache` and `sample_weight` are mutually exclusive because row weights already
  live inside a prebuilt cache.

### `KnockoffSelectionResult`

```python
result.selected_features      # ordered selected feature labels
result.selected_indices       # original X column positions, when available
result.selector_metadata      # q, statistic, s_method, gamma, validity metadata
result.W                      # one row per valid feature with W diagnostics
result.threshold              # float for one draw, None for derandomized runs
result.selection_frequency    # Series for n_draws > 1, otherwise None
result.diagnostics_           # per-draw thresholds and selected sets
result.get_feature_ranking()  # sorted DataFrame
```

The ranking table includes `feature`, `W`, `rank`, `selected`,
`selection_frequency`, `selected_index`, `relevance`, and `selector`. When
`feature_groups` is used it also includes `feature_group`.

### `sample_knockoffs`

```python
from sift import build_cache, sample_knockoffs

cache = build_cache(X, compute_Rxx=True, random_state=0)
Z_tilde = sample_knockoffs(cache, s_method="equi", random_state=123)
```

`sample_knockoffs` is an advanced helper that returns one Gaussian-copula
knockoff draw in cache space. It is useful for diagnostics and custom
statistics, not for ordinary feature selection.

### `KnockoffSelector`

```python
from sift import KnockoffSelector

selector = KnockoffSelector(
    q=0.1,
    statistic="relevance",
    n_draws=1,
    offset=1,
    s_method="equi",
    feature_groups=None,
    cat_encoding="none",
    random_state=0,
    verbose=False,
)

X_selected = selector.fit_transform(X, y)
selector.result_
selector.selected_features_
selector.get_support(indices=True)
```

`KnockoffSelector` is sklearn-style, but it is q-based and does not support
`k` or `auto_k_config`. It rejects row `groups` and `time`; use
`feature_groups` for grouped feature discoveries.

## Caching

### `build_cache`

```python
from sift import build_cache

cache = build_cache(
    X,
    sample_weight=None,
    subsample=50_000,
    compute_Rxx=True,
    random_state=0,
    n_jobs=1,
)
```

`FeatureCache` stores Gaussianized features, valid columns, row indices, sample
weights, feature names, and optionally the feature-feature correlation matrix.
Use it when running many targets or cache-backed methods.

### `select_cached`

```python
from sift import select_cached

selected = select_cached(
    cache,
    y,
    k=20,
    method="cefsplus",        # "cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"
    top_m=None,
    corr_prune="auto",
    return_objective=False,
    return_indices=False,
)
```

`select_cached` reuses the cache transform and correlation work for repeated
selection against new numeric targets.

## Automatic K

```python
from sift import AutoKConfig, select_mrmr

config = AutoKConfig(
    k_method="evaluate",      # "evaluate", "elbow", or "penalized_objective"
    strategy="time_holdout",  # "time_holdout" or "group_cv"
    metric="auto",
    max_k=100,
    min_k=5,
    selection_rule="best",    # "best", "one_se", "plateau", or "tolerance"
)

selected = select_mrmr(
    X,
    y,
    k="auto",
    task="regression",
    time=timestamps,
    auto_k_config=config,
)
```

Function-style selectors use `auto_k_mode="prefix_only"`: they build one
supervised feature path and evaluate prefixes. Selector classes also implement a
nested mode for train-only fold paths where supported.

## Selector Classes

All fixed-k selector classes implement `fit`, `transform`, `fit_transform`, and
`get_support`.

```python
from sift import MRMRSelector, JMISelector, JMIMSelector
from sift import CEFSPlusSelector, CEFSPlusBinarySelector

selector = MRMRSelector(k=20, task="regression", verbose=False)
selector.fit(X, y)
X_selected = selector.transform(X)
mask = selector.get_support()
indices = selector.get_support(indices=True)
```

After fitting, selector classes expose:

- `selected_features_`
- `selected_indices_`
- `feature_names_in_`
- `n_features_in_`
- `k_` when automatic k resolved a value

`KnockoffSelector` additionally exposes `result_`.

## Stability Selection

```python
from sift import StabilitySelector, stability_regression, stability_classif

selector = StabilitySelector(
    n_bootstrap=50,
    sample_frac=0.5,
    threshold=0.6,
    alpha=None,
    l1_ratio=1.0,
    task="regression",
    max_features=None,
    block_size="auto",
    block_method="moving",
    use_smart_sampler=False,
    random_state=0,
)

selector.fit(X, y, sample_weight=None, groups=None, time=None)
info = selector.get_feature_info()
```

Convenience wrappers:

```python
selected_reg = stability_regression(X, y, k=20, random_state=0)
selected_cls = stability_classif(X, y, k=20, random_state=0)
```

Stability selection is a robust heuristic built on repeated sparse linear
models. It does not provide the same q-calibrated API as `select_fdr`.

## Smart Sampling

```python
from sift import SmartSamplerConfig, smart_sample, panel_config, cross_section_config

config = panel_config("entity_id", "date", sample_frac=0.15)
sampled = smart_sample(
    df,
    feature_cols=feature_cols,
    y_col="target",
    config=config,
)
```

Smart sampling reduces large panel or cross-section data before selection and
adds inverse-probability style sample weights for selected rows.

## Permutation Importance

```python
from sift import permutation_importance

importance = permutation_importance(
    model,
    X,
    y,
    sample_weight=None,
    groups=None,
    time=None,
    scoring="neg_mse",
    n_repeats=10,
    permute_method="auto",    # "auto", "global", "within_group", "block", "circular_shift"
    block_size="auto",
    random_state=0,
)
```

The result is a DataFrame with mean/std importance and per-repeat diagnostics.
Use grouped or time-aware permutation methods when ordinary global shuffling
would break the data-generating structure.

## Boruta

```python
from sift import BorutaSelector, select_boruta, select_boruta_shap

features = select_boruta(X, y, task="regression", random_state=0)
shap_features = select_boruta_shap(X, y, task="regression", random_state=0)

selector = BorutaSelector(
    estimator="rf",
    task="regression",
    importance="native",
    max_iter=50,
    random_state=0,
)
selector.fit(X, y)
```

Boruta is an all-relevant selector: it tries to keep every feature that beats
shadow-feature importance, not a minimal subset.

## CatBoost Selection

```python
import sift

result = sift.catboost_select(
    X,
    y,
    task="regression",
    k=20,                    # None searches over feature counts
    algorithm="forward",     # "forward", "forward_greedy", "shap", "permutation", "prediction"
    prefilter_k=200,
    cv=None,
    group_col=None,
    sample_weight_col=None,
    random_state=0,
)

features = result.selected_features
```

Convenience wrappers:

```python
reg_features = sift.catboost_regression(X, y, k=20, algorithm="forward")
cls_features = sift.catboost_classif(X, y, k=20, algorithm="forward")
```

Install with `python -m pip install -e ".[catboost]"` before using these
helpers.

## Low-Level Estimators

Advanced users can import submodules directly:

- `sift.estimators.copula` for `FeatureCache`, rank-Gaussian transforms, and
  weighted correlations.
- `sift.estimators.knockoffs` for Gaussian knockoff sampler internals:
  `fit_gaussian_knockoffs`, `sample_gaussian_knockoffs`,
  `gaussian_knockoff_mean`, and `GaussianKnockoffModel`.
- `sift.estimators.joint_mi` for mutual-information estimators.
- `sift.estimators.relevance` for relevance scorers.
- `sift.selection.path_eval` for explicit feature-path evaluation.

Low-level APIs are useful for diagnostics and extensions; the top-level
functions remain the supported user-facing surface.
