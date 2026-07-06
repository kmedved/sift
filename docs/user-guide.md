# User Guide

This guide gives the practical map for choosing a SIFT workflow. For exact
parameters and longer examples, use the full [API manual](../DOCS.MD); for
error messages and common pitfalls see [troubleshooting](troubleshooting.md).

## Choose a Selector

| Need | Start with |
| --- | --- |
| Fast fixed-k regression or classification filter | `select_mrmr` |
| Joint mutual-information ranking | `select_jmi` or `select_jmim` |
| Gaussian-copula regression filter with objective diagnostics | `select_cefsplus` |
| Binary logistic CEFS+ path | `select_cefsplus_binary` |
| Sklearn pipeline compatibility | `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector` |
| Robust selection across resamples | `StabilitySelector` |
| Large panel or cross-section subsampling | `smart_sample` |
| Model-agnostic importance after fitting a model | `permutation_importance` |
| All-relevant feature selection | `BorutaSelector` or `select_boruta` |
| CatBoost-native selection | `catboost_select`, `catboost_regression`, `catboost_classif` |

## Fixed-k Filters

```python
from sift import select_mrmr, select_jmi, select_jmim, select_cefsplus

mrmr = select_mrmr(X, y, k=25, task="regression", verbose=False)
jmi = select_jmi(X, y, k=25, task="regression", verbose=False)
jmim = select_jmim(X, y, k=25, task="regression", verbose=False)
cefs = select_cefsplus(X, y, k=25, verbose=False)
```

Fixed `k` is an upper bound. Selectors can return fewer features when constant
features, invalid scores, `top_m`, or pruning remove candidates.

## Binary CEFS+

```python
from sift import select_cefsplus_binary

selected = select_cefsplus_binary(
    X,
    y_binary,
    k=20,
    loss="logloss",
    class_weight="balanced",
    cat_encoding="loo_logit",
    verbose=False,
)
```

Use binary CEFS+ when the target is Bernoulli-like and logistic conditional
information is a better fit than a Gaussian target approximation.
`sample_weight` and `class_weight` are honored directly by `loss="logloss"`.

## Automatic Feature Count

```python
from sift import AutoKConfig, select_cefsplus

config = AutoKConfig(
    k_method="evaluate",
    strategy="time_holdout",
    min_k=5,
    max_k=80,
)

selected = select_cefsplus(X, y, k="auto", auto_k_config=config, verbose=False)
```

Function-style selectors use a prefix-only contract for auto-k: SIFT builds one
selection path, then evaluates prefixes. Sklearn-style selector classes can use
nested evaluation where supported.

Auto-k support depends on the selector route:

| Route | Supported `k_method` values |
| --- | --- |
| Classic mRMR/JMI/JMIM | `evaluate` |
| Gaussian mRMR/JMI/JMIM | `evaluate`, `elbow` |
| CEFS+ | `evaluate`, `elbow`, `penalized_objective` |
| Binary CEFS+ | `evaluate`, `elbow`, `penalized_objective` |

Unsupported modes fail before SIFT builds caches or feature paths, which keeps
configuration errors cheap to catch.

## Reuse a Gaussian Cache

```python
from sift import build_cache, select_cached

cache = build_cache(X, subsample=None, compute_Rxx=True)
mrmr = select_cached(cache, y1, k=30, method="mrmr_quot")
cefs = select_cached(cache, y2, k=30, method="cefsplus")
```

Use a cache when many selectors or targets share the same feature matrix.

## Stability Selection

```python
from sift import StabilitySelector

selector = StabilitySelector(
    task="regression",
    n_bootstrap=50,
    threshold=0.6,
    random_state=0,
    verbose=False,
)
selector.fit(X, y)
stable_features = selector.selected_feature_names_
```

Pass both `groups` and `time` to use block bootstrap for ordered panel data.

## Time-aware Permutation Importance

```python
from sift import permutation_importance

importance = permutation_importance(
    fitted_model,
    X,
    y,
    groups=group_ids,
    time=dates,
    permute_method="auto",
    scoring="neg_rmse",
    n_repeats=10,
)
```

With `time` but no `groups`, SIFT treats the dataset as one ordered group for
time-aware permutations.

## Categorical Features

Function-style selectors support `cat_features` and `cat_encoding`. Supervised
categorical encodings are guarded against full-data target leakage by default;
use them through train-only wrappers or opt in only when leakage is handled
outside SIFT. CatBoost selectors handle categorical features natively.

## Diagnostics

Many selectors can return richer metadata through `return_result=True` or
selector-specific diagnostics. The detailed behavior is documented in
[DOCS.MD](../DOCS.MD).

```python
from sift import select_cefsplus_binary

result = select_cefsplus_binary(
    X,
    y_binary,
    k="auto",
    auto_k_config=config,
    return_result=True,
    verbose=False,
)

print(result.selected_features)
print(result.selector_metadata)
```

Sklearn-style selector classes always keep their transform contract stable; pass
inspection options to the function-style selectors when you need full result
objects.
