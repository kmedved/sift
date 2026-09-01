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
| q-calibrated feature set instead of fixed k | `select_fdr` or `KnockoffSelector` |
| Sklearn pipeline compatibility | `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector`, `KnockoffSelector` |
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
For fixed-k filter calls, `groups` and `time` are rejected because they only
define auto-k evaluation splits; use `k="auto"` with a matching strategy or
omit those arguments. `KnockoffSelector` rejects row `groups` and `time` too.
Sklearn-style selector classes accept dense arrays and DataFrames; sparse
matrices are rejected during fit, transform, and inverse transform.

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

## FDR-Controlled Knockoffs

```python
from sift import select_fdr

result = select_fdr(X, y, q=0.1, n_draws=1, verbose=False)
selected = result.selected_features
ranking = result.get_feature_ranking()
```

Use knockoffs when you want a q-calibrated trusted set instead of asking for a
fixed number of features. The v1 implementation samples second-order Gaussian
knockoffs in the rank-Gaussian copula space already used by `FeatureCache`, then
applies the knockoff+ threshold to an antisymmetric feature statistic.

Read the guarantee metadata literally:

- `fdr_control="approximate_plugin"`: the default path estimates the feature
  model from data and may shrink it for numerical stability.
- `validity_model="gaussian_copula_plugin"`: exact Model-X FDR would require
  that copula model to be correct.
- `weighted_model=True`: sample weights were used as importance weights in the
  plug-in model and statistic.
- `gamma`, `lambda_min`, and `s_mean`: diagnose covariance shrinkage and
  knockoff power. Large `gamma` or tiny `s_mean` usually means highly correlated
  features; deduplicate near-copies before building the cache when power matters.

`statistic="relevance"` is the fastest compatibility default for marginal
signals. Relative power is data-dependent, and no committed quality bakeoff
establishes a universal winner. `statistic="cefsplus"` enables a tie-safe greedy
CEFS+ statistic with pair-coupled screening and objective-gain W magnitudes. It
can recover redundant signal families that a marginal statistic treats as a
single effect, but it is still slower at large `screen_pairs`/`path_depth`, so
use it as a redundancy-aware second opinion rather than a better default.
Without an explicit `path_depth`, CEFS+ starts with a q-aware bounded path and
doubles it when discoveries reach the cap. The initial and final depths are
reported in selector metadata. Set `statistic_options={"path_depth": ...}` only
when you need a hard compute cap; a saturated explicit cap emits a warning.
`statistic_options={"min_gain_ratio": 1e-4}` is an opt-in speed knob for large
CEFS+ runs.

`s_method="mvr"` and `"me"` use diagonal coordinate-descent optimizers for the
MVR and maximum-entropy knockoff objectives. They can improve power on
correlated designs where equicorrelated knockoffs are too reconstructable. Do
not judge them by `s_mean` alone: a correct MVR solution can have lower average
`s` than equicorrelated while improving the objective and selections.

Pass `feature_groups=[...]` to threshold group-level antisymmetric statistics
and expand selected groups back to active member features. This is useful for
one-hot families, lags, spline bases, or other feature families. Interpret it
as group-discovery control, not exact feature-level FDR inside each selected
group.

`n_draws > 1` redraws knockoffs and selects features whose selection frequency
is at least `eta`. This is useful for run-to-run stability, but it is also
reported as approximate. An empty result is a valid answer: no feature survived
the requested q threshold. `select_fdr` requires a finite numeric target;
continuous targets and numeric binary labels are supported. Integer-valued
multiclass targets trigger a warning because this routine treats `y` as numeric;
string/categorical labels are not accepted by `select_fdr`, so encode a
one-vs-rest target numerically for categorical multiclass tasks.

Auto-k and knockoffs answer different questions. Auto-k asks how many features
help prediction; knockoffs ask how many discoveries you can trust at a target
q. It is often useful to compare both diagnostics.

## Automatic Feature Count

```python
import numpy as np

from sift import AutoKConfig, select_cefsplus

# Zero-config CEFS+ auto-k uses the measured Auto-K v2 router.
selected = select_cefsplus(X, y, k="auto", verbose=False)

config = AutoKConfig(
    k_method="evaluate",  # or "auto", "gaussian_cv", "chi2_stop", etc.
    strategy="time_holdout",
    min_k=5,
    max_k=80,
)

timestamps = np.arange(len(X))  # replace with the real chronological key

selected = select_cefsplus(
    X,
    y,
    k="auto",
    time=timestamps,
    auto_k_config=config,
    verbose=False,
)
```

Function-style selectors use a prefix-only contract for auto-k: SIFT builds one
selection path, then evaluates prefixes. Sklearn-style selector classes can use
nested evaluation where supported.

Auto-k support depends on the selector route:

| Route | Supported `k_method` values |
| --- | --- |
| Classic mRMR/JMI/JMIM | `evaluate` |
| Gaussian mRMR/JMI/JMIM | `auto`, `evaluate`, `elbow`, `gaussian_cv`, `xfit_objective`, `stability` |
| CEFS+ | `auto`, `evaluate`, `elbow`, `penalized_objective`, `k_posterior`, `chi2_stop`, `forward_stop`, `changepoint`, `perm_gap`, `knockoff_path`, `gaussian_cv`, `xfit_objective`, `stability`, `consensus` |
| Binary CEFS+ | `auto`, `evaluate`, `elbow`, `penalized_objective`, `k_posterior`, `changepoint` |

Unsupported modes fail before SIFT builds caches or feature paths, which keeps
configuration errors cheap to catch.

For a first pass with CEFS+, use `select_cefsplus(X, y, k="auto")`; it routes
to the measured EBIC default and records `auto_routing` diagnostics in result
objects. Use `gaussian_cv` when you specifically want fold scoring, `chi2_stop`
or `forward_stop` when you need a calibrated no-signal stop, `perm_gap` when
groups/time/weights make analytic nulls suspicious, and `knockoff_path` when
you need an approximate plug-in q-calibrated returned set. `changepoint`,
`stability`, `xfit_objective`, and `knockoff_path` remain experimental or
failed-gate for automatic sizing.

## Reuse a Gaussian Cache

```python
from sift import build_cache, select_cached

cache = build_cache(X, subsample=None, compute_Rxx=True)
mrmr = select_cached(cache, y1, k=30, method="mrmr_quot")
cefs = select_cached(cache, y2, k=30, method="cefsplus")
cefs_view = select_cached(
    cache, y2, k=30, method="cefsplus", return_result=True
)
```

Use a cache when many selectors or targets share the same feature matrix. A
prebuilt cache is tied to the input row count and feature contract: named
caches require the same DataFrame column names in exact order, while positional
caches require a positional ndarray with the same row count and feature count.
Rebuild a positional cache from a DataFrame to establish named-column
alignment. Cache-backed filter-function calls reject call-time `sample_weight`
and must omit `subsample` and construction `random_state`; the cache already
fixes its sampled rows and weights. For `select_fdr`, `random_state` remains
available because it seeds a fresh knockoff draw; `sample_weight` and
`subsample` remain forbidden.
The opt-in cached `SelectionView` includes selected positions, the objective
path, relevance, and cache provenance. `return_result=True` is mutually
exclusive with the legacy `return_objective` and `return_indices` tuple flags.

## Stability Selection

```python
from sift import StabilitySelector

selector = StabilitySelector(
    task="regression",
    n_bootstrap=50,
    threshold=0.6,
    penalty=None,              # additive alias for alpha
    random_state=0,
    verbose=False,
)
selector.fit(X, y)
stable_features = selector.selected_feature_names_
X_stable = selector.transform(X)
X_restored = selector.inverse_transform(X_stable)
```

Pass both `groups` and `time` to use block bootstrap for ordered panel data.
With DataFrames, `groups="column"` and `time="column"` extract and exclude the
metadata columns; direct arrays remain positional. `penalty` is an alias for
`alpha`, and both may be supplied only when equal. Threshold tuning accepts
sklearn scorer objects as well as scorer names.
`selector.get_feature_names_out()` is the sklearn-compatible equivalent for
retrieving the selected names after fitting.
Set `output_order="legacy"` (the default) to keep descending stability-frequency
order, or `output_order="original"` to emit selected columns in fitted input
order. The same order is used by `transform`, `get_support(indices=True)`,
`get_feature_names_out`, and dense `inverse_transform`; inverse output
zero-fills unselected columns.
Block draws honor `sample_frac`; the rounded panel-wide draw budget is allocated
proportionally across groups and block windows are sampled with replacement.
Time values must be non-missing and orderable within each group.

Leaving `random_state=None` on Stability, permutation importance, or CatBoost
emits a `FutureWarning`: 0.9 remains nondeterministic, while 1.0 will default to
seed 0. Their existing `n_jobs=-1` defaults are also unchanged in 0.9.

## CatBoost Row Context

```python
from sklearn.model_selection import TimeSeriesSplit
import sift

result = sift.catboost_select(
    X,
    y,
    k=20,
    groups=group_ids,
    time=dates,
    sample_weight=weights,
    cv=TimeSeriesSplit(n_splits=5),
    random_state=0,
)
```

CatBoost accepts direct positional `groups`, `time`, and `sample_weight`
arrays. DataFrame callers may instead use `groups="group_column"` or
`time="date_column"`; `group_col` and `sample_weight_col` remain compatibility
aliases. A direct value and its alias cannot be combined. Supplied time values
must be non-missing and mutually orderable and stably order aligned rows before
the configured splitter. Use a time-aware splitter when chronological
validation is required; the default splitter remains random.

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

rich_importance = permutation_importance(
    fitted_model,
    X,
    y,
    groups=group_ids,
    time=dates,
    n_repeats=10,
    return_result=True,
)
repeat_drops = rich_importance.importances_
view = rich_importance.result_view()
```

With `time` but no `groups`, SIFT treats the dataset as one ordered group for
time-aware permutations. The historical DataFrame remains the default;
`return_result=True` adds the repeat-level matrix and a complete ranking view
without applying an arbitrary selection threshold.

## Categorical Features

Function-style selectors default to `cat_encoding="none"` and support
`cat_features` and explicit encodings. Use `cat_encoding="target_cv"` for the
built-in leakage-safe regression/binary path; it uses cross-fitted training
values and needs no optional dependency:

```python
selected = select_mrmr(
    X,
    y,
    k=10,
    task="regression",
    cat_encoding="target_cv",
    verbose=False,
)
```

Selector classes retain the full-training encoder for target-blind inference,
while `fit_transform` returns the cross-fitted training columns used for
selection. Weighted calls use SIFT's weighted m-estimate folds and require an
explicit smoothing value:

```python
selected = select_mrmr(
    X,
    y,
    k=10,
    task="regression",
    cat_encoding="target_cv",
    target_cv_smoothing=20.0,
    sample_weight=weights,
    verbose=False,
)
```

Grouped/time-aware encoding is available on auto-k evaluate routes. Set
`target_cv_n_splits` independently of the outer auto-k fold count. Group folds
exclude whole groups; time folds keep tied timestamps together and use only
strictly earlier values. Earliest time rows use an explicit target-independent
`target_prior`, or receive zero effective selection weight under
`warmup_policy="zero_weight"` (default) or `"exclude"`. Fixed-k calls continue
to reject `groups`/`time`, and multiclass target encoding remains blocked on
block-aware selection. Existing `"target"`, `"loo"`, `"james_stein"`, and
`"loo_logit"` function encodings remain guarded against full-data target
leakage; opt in only when leakage is handled outside SIFT. CatBoost selectors
handle categorical features natively.

## Diagnostics

Many selectors can return richer metadata through `return_result=True` or
selector-specific diagnostics. `sift.as_result(...)` now provides an additive
common view for `FilterSelectionResult`, `KnockoffSelectionResult`,
`BorutaResult`, `FeaturePathEvaluationResult`, and `CatBoostSelectionResult`,
plus fitted `StabilitySelector` and the opt-in `ImportanceResult` from
`permutation_importance`. Legacy result types and default returns are unchanged.

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

The common A2d access pattern is:

```python
view = sift.as_result(result, input_features=X.columns)

view.features
view.indices
view.k
view.table
view.metadata
```

These result-only views do not retain fitted preprocessing state, so transform
and proxy operations are unavailable. See [Reading Results](results.md) for the
current adapter-completeness matrix and serialization contract, and
[DOCS.MD](../DOCS.MD) for selector-specific diagnostics.

Sklearn-style selector classes always keep their transform contract stable; pass
inspection options to the function-style selectors when you need full result
objects.

A fitted stability selector supplies the same accessors and a frozen transform:

```python
from sift import StabilitySelector

selector = StabilitySelector(random_state=0, verbose=False).fit(X, y)
view = selector.result_view_
X_stable = view.transform(X)
```

Its table covers the selector's fitted candidate features; `view.indices`
keeps the existing integer positions and `view.features` supplies names.
The fitted selector itself supports dense `inverse_transform`; the frozen
`SelectionView` intentionally does not retain the fitted preprocessing state
needed for inversion.
