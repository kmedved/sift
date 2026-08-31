# SIFT API Reference

This page is a standalone reference for the public SIFT API in the 0.9
surface. For deeper examples and option notes, see the canonical manual in
[`DOCS.MD`](../DOCS.MD).

## Public Surface

| Area | Main entry points |
| --- | --- |
| Fixed-k filters | `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, `select_cefsplus_binary` |
| q-calibrated knockoffs | `select_fdr`, `KnockoffSelector`, `KnockoffSelectionResult`, `sample_knockoffs` |
| Caching | `build_cache`, `select_cached`, `FeatureCache` |
| Automatic k | `AutoKConfig`, `select_k_auto`, `select_k_elbow`, `select_k_penalized_objective`, `select_k_chi2_stop`, `select_k_perm_gap`, `select_k_gaussian_cv` |
| Selector classes | `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector`, `KnockoffSelector` |
| Stability selection | `StabilitySelector`, `stability_regression`, `stability_classif` |
| Sampling | `smart_sample`, `SmartSamplerConfig`, `panel_config`, `cross_section_config` |
| Model importance | `permutation_importance` |
| Boruta | `BorutaSelector`, `BorutaResult`, `select_boruta`, `select_boruta_shap` |
| Optional CatBoost | `catboost_select`, `catboost_regression`, `catboost_classif` |
| Logging | `set_verbosity` |
| Normalized result views | `SelectionView`, `as_result` |

CatBoost entry points are lazy exports from `sift`; importing `sift` does not
require the `catboost` extra.

Progress from calls with `verbose=True` is emitted at INFO on the `sift`
logger. It remains visible by default and propagates to application logging
handlers when configured. Use `sift.set_verbosity("debug")`,
`sift.set_verbosity("info")`, or `sift.set_verbosity(None)` to select debug,
normal progress, or silence globally; per-call `verbose=False` remains silent.

## Normalized Result Views (A2c)

`sift.as_result(result, input_features=None)` returns a `SelectionView` without
changing the legacy result object. The current A2c implementation supports
`FilterSelectionResult`, `KnockoffSelectionResult`, `BorutaResult`,
`FeaturePathEvaluationResult`, and `CatBoostSelectionResult`, plus a fitted
`StabilitySelector`; permutation importance is still pending.

```python
view = sift.as_result(result, input_features=X.columns)

view.features
view.indices
view.k
view.table
view.metadata
```

Result-only views report `metadata["input_kind"] == "unknown"`, because the
legacy objects cannot prove whether their source was named or positional.
Feature-path and CatBoost results expose normalized evaluation curves. Boruta
maps mean importance to `gain` while preserving original positional order;
CatBoost maps its retained final-model importance to `gain` and records the
source explicitly. The five result-only adapters do not transform. A fitted
stability view exposes a frozen column-subset transform while leaving inverse
transform unavailable. Proxy lookup is not available in A2c. See
[Reading Results](results.md) for the completeness matrix,
versioned JSON format, and partial-table rules.

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
- `subsample` and `random_state` for classic row sampling and uncached Gaussian
  cache construction.
- `return_result=True` for a `FilterSelectionResult` where supported.

For fixed-k calls, `groups` and `time` are rejected because they define only
auto-k evaluation splits. With a prebuilt Gaussian cache, a named cache
requires a DataFrame with the same row count and exact column order; a
positional cache requires a positional ndarray with the same row count and
feature count. Named DataFrames are rejected for positional caches because
column alignment cannot be proven. The
cache-backed filter functions reject explicit cache-construction
`subsample`/`random_state` overrides. In `select_fdr`, `random_state` instead
seeds a fresh knockoff draw and remains meaningful with a cache; `subsample`
must be omitted.

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
    corr_prune=None,
    sample_weight=None,
    subsample=50_000,
    random_state=0,
    verbose=False,
)
```

CEFS+ is a regression-only Gaussian-copula filter that uses a log-determinant
conditional information objective. `corr_prune=None` preserves possible
suppressor pairs; pass a threshold such as `0.95` to opt into duplicate-oriented
pruning.

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
    corr_prune=None,
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
    statistic="relevance",    # "relevance", "lsm", "ridge", or tie-safe "cefsplus"
    n_draws=1,
    eta=0.5,
    offset=1,                 # 1 = knockoff+, 0 = modified knockoff threshold
    s_method="equi",          # "equi", "mvr", or "me"
    min_eig=1e-3,
    screen_pairs=2000,
    statistic_options=None,
    feature_groups=None,      # labels, or "auto" for correlation-cluster representatives
    group_corr_threshold=0.7,
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
- `statistic="lsm"` is the lasso signed-max from a Gram-form LARS path on the
  analytic augmented correlation and is exactly antisymmetric
  (`statistic_options={"max_steps": int}`). Its power relative to
  `relevance` is data-dependent; no committed quality bakeoff establishes a
  universal winner.
- `statistic="ridge"` is the analytic ridge coefficient difference
  (`statistic_options={"ridge_lambda": float}`, default `0.5`).
- `statistic="cefsplus"` enables the tie-safe greedy CEFS+ statistic. It accepts
  `statistic_options={"path_depth": int, "min_gain_ratio": float}`.
- `s_method="equi"` is fastest. `s_method="mvr"` and `"me"` use diagonal
  coordinate-descent objectives and can improve power on correlated designs.
- `n_draws > 1` redraws knockoffs and selects features with frequency at least
  `eta`; `threshold` is then `None` and `selection_frequency` is populated.
  Here `q` is a per-draw threshold level, not a guarantee for the aggregated
  vote. The aggregated result reports `fdr_control="none"`,
  `q_scope="per_draw"`, and `aggregation_fdr_control="none"`.
- `offset=1` is the knockoff+ threshold. `offset=0` is less conservative and is
  best read as modified-knockoff or mFDR-style control.
- `feature_groups` thresholds a heuristic signed-maximum group aggregation and
  expands selected groups back to member features. This mode has no established
  group- or feature-level FDR control; metadata reports
  `group_fdr_control="none"`, `per_draw_fdr_control="none"`, and
  `fdr_control="none"`.
- `feature_groups="auto"` clusters features by `|corr| >= group_corr_threshold`
  and runs the filter on one representative per cluster; the result table gains
  `feature_group` and `is_representative` columns and selected clusters are
  expanded. Use it when `selector_metadata["s_median"]` is tiny (a
  `UserWarning` also flags this): near-collinear blocks leave feature-level
  knockoffs with no power. Knockoff calibration applies to the representatives;
  neither cluster- nor feature-level FDR is established after expansion. A
  single draw reports `representative_fdr_control="approximate_plugin"`, while
  `group_fdr_control`, `feature_level_fdr_control`, and `fdr_control` are
  `"none"`. Correlation clustering/linkage costs O(p^2) memory/time, so
  pre-screen very wide feature sets first.
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
It is tagged non-deterministic because changing row order can change seeded
knockoff noise assignment and selection. Zero-weight rows are removed before
RNG draws and do not consume that stream, but shuffled-row equality is not
promised.

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
Use it when running many targets or cache-backed methods. A named cache requires
the same DataFrame column names in exact order. A positional cache requires a
positional ndarray with the same row count and feature count. Rebuild rather
than passing call-time `sample_weight` or new cache-construction `subsample` or
`random_state` settings to a cache-backed filter function. `select_fdr` accepts
`random_state` with a cache because it seeds the fresh knockoff draw; its
`sample_weight` and `subsample` remain forbidden.

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
from sift import AutoKConfig, select_cefsplus, select_mrmr

# CEFS+ zero-config auto-k uses the measured Auto-K v2 router.
selected = select_cefsplus(X, y, k="auto")

config = AutoKConfig(
    k_method="evaluate",      # also "auto", "gaussian_cv", "perm_gap", etc.
    strategy="time_holdout",  # "time_holdout", "group_cv", or "kfold" for fold curves
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

`k_method="auto"` is the measured router. CEFS+ and binary CEFS+ use it when
`auto_k_config` is omitted; explicit `AutoKConfig(k_method="auto")` also works
on Gaussian mRMR/JMI/JMIM. The router records `auto_routing` in
`diagnostics_["auto_k"]`. CEFS+ currently routes to EBIC by default, uses EBIC
when `p_valid > n_eff_kish`, uses `perm_gap` for heavy weight skew, and uses
`gaussian_cv/one_se` for non-CEFS+ Gaussian selectors. No-config CEFS+ calls
with `groups` or `time` now use this router instead of the older
`evaluate/group_cv` or `evaluate/time_holdout` behavior, and no-context calls
now work. Router branches use method-specific effective floors (`0` for EBIC
and permutation-gap stops, at least `1` for Gaussian CV curves), so use an
explicit `AutoKConfig(k_method=...)` when you need a hard `min_k`. If the
selected k hits the effective maximum, the router emits a `UserWarning` and
sets `auto_routing["saturated"] = True`; treat that result as censored. When
`saturation_reason="configured_max_k"`, raise `max_k` or inspect the curve.
When `saturation_reason="candidate_path_exhausted"`, increasing `max_k` alone
cannot help; inspect valid candidates and `corr_prune`/`top_m`. When
`saturation_reason="evaluation_curve_limited"`, the candidate path still has
features but a fold/statistical limit ended the risk curve; inspect fold sample
sizes and evaluation diagnostics.

For dense weak-signal domains in Gaussian CEFS+ automatic routing, set
`auto_dense_check=True` on `AutoKConfig(k_method="auto")` to run an opt-in
`gaussian_cv` cross-check with `selection_rule="best"` after large EBIC picks;
the router warns when EBIC's detectable-feature count and the Gaussian CV
sufficiency pick differ by more than the configured ratio. Binary log-loss
CEFS+ has no dense-regime diagnostic and rejects non-default `auto_dense_*`
fields. Binary Brier selection delegates to Gaussian CEFS+ and therefore
follows the Gaussian dense-check contract.

Important `AutoKConfig` method fields:

| Field | Applies to |
| --- | --- |
| `alpha`, `m_mode`, `stop_patience` | `chi2_stop`, `forward_stop`, `perm_gap`, `changepoint` |
| `perm_B`, `perm_null`, `gap_rule` | `perm_gap` |
| `knockoff_q`, `knockoff_draws`, `knockoff_s_method`, `knockoff_return` | `knockoff_path` |
| `xfit_folds`, `xfit_mode` | `gaussian_cv`, `xfit_objective` |
| `xfit_ridge` | `gaussian_cv` |
| `ebic_gamma`, `n_eff_mode` | `penalized_objective`, `k_posterior` |
| `posterior_level`, `posterior_pick` | `k_posterior` |
| `boot_B`, `boot_mode`, `stability_rule`, `stability_pi` | `stability` |
| `floor_z`, `floor_window` | `changepoint` |
| `consensus_methods` | `consensus` |
| `auto_dense_check`, `auto_dense_min_k`, `auto_dense_min_frac`, `auto_dense_disagreement_ratio` | `auto` (Gaussian CEFS+, including the binary Brier delegate; binary log-loss rejects non-default values) |

`knockoff_path` returns an approximate Gaussian-copula plug-in selected set when
`knockoff_return="set"`. `changepoint`, `stability`, `xfit_objective`, and
`knockoff_path` remain experimental or failed-gate for automatic sizing in the
Auto-K v2 campaign; stability uses `stability_rule="max_one_se"` by default and returns
`stopped_by="stability_floor"` when chance-corrected agreement is too low.

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
- `get_feature_names_out()`

`KnockoffSelector` additionally exposes `result_`.

The Gaussian/cache-backed selector classes expose sklearn-compatible automatic
defaults: `subsample="auto"` and, except for `KnockoffSelector`,
`random_state="auto"`. Without a cache they resolve at fit time to 50,000 rows
and seed 0. With a prebuilt cache, `"auto"` means omitted, while explicit
construction overrides are rejected. `KnockoffSelector.random_state` remains
numeric because it controls each new knockoff draw even when a cache is reused.

## Stability Selection

```python
from sift import StabilitySelector, stability_regression, stability_classif

selector = StabilitySelector(
    n_bootstrap=50,
    sample_frac=0.5,
    threshold=0.6,
    alpha=None,
    alpha_rule="one_se",
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
view = selector.result_view_
```

Convenience wrappers:

```python
selected_reg = stability_regression(X, y, k=20, random_state=0)
selected_cls = stability_classif(X, y, k=20, random_state=0)
```

Stability selection is a robust heuristic built on repeated sparse linear
models. It does not provide the same q-calibrated API as `select_fdr`.
After fitting, `StabilitySelector.get_feature_names_out()` returns the selected
feature names and validates any supplied `input_features` against the fit-time
feature order.

`StabilitySelector.transform` always returns an ndarray. For a selector fitted
on a DataFrame, or on an ndarray with explicit `feature_names`, DataFrame
transforms select fitted features by name: extra and reordered columns are
accepted, while duplicate labels or missing selected columns raise.
`tune_threshold` applies the same identity checks while requiring every fitted
feature. A selector fitted on an unnamed positional ndarray cannot prove
DataFrame column identity and therefore rejects DataFrame input to either
method; keep using same-width ndarrays, provide explicit names when fitting the
ndarray, or refit on a DataFrame. Ndarray transforms are positional and must
have the fit-time feature count.

`selector.result_view_` is a dynamic, non-cached `SelectionView` over the
fitted candidate-feature namespace. It preserves selected names in
`view.features`, legacy integer positions in `view.indices`, frequencies and
mean absolute coefficients in a complete table, and exposes a frozen
`view.transform(...)`. The frozen transform retains sklearn `set_output`
configuration but not training rows or bootstrap coefficient matrices.

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

With `StabilitySelector(use_smart_sampler=True)`, `X` must be a DataFrame. An
explicit `fit(..., feature_names=[...])` sequence is an ordered feature-subset
contract: the selector does not widen it to other numeric columns. Every
surviving explicit feature must be numeric. Pass an ordered iterable such as a
list, tuple, pandas Index, or one-dimensional NumPy array; strings, bytes-like
objects, mappings, sets, scalar arrays, and matrix-like containers are rejected.
Tuple and MultiIndex column labels remain single feature names in
`get_feature_names_out`. `group_col` and `time_col` from the sampler
configuration are metadata exclusions and are removed from the feature subset
even when named explicitly. Without `feature_names`, the selector uses the
numeric DataFrame columns other than those metadata columns. Datetime and
timedelta feature columns are rejected before numeric coercion; a configured
datetime `time_col` remains valid metadata. `tune_threshold` preserves those
metadata columns for fold-local smart sampling without widening the fitted
feature subset.

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
After fitting, `BorutaSelector.get_feature_names_out()` returns the accepted
feature names and validates any supplied `input_features` against the fit-time
feature order.

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

### Progress callbacks

Long-running filter paths (including `select_cached` and the filter selector
classes), stability, Boruta, and CatBoost selectors accept
`callback(step, total, info)`. Steps are one-based and are reported after each
completed greedy-path step, bootstrap, Boruta iteration, or CatBoost split,
respectively. Each call receives a fresh snapshot dictionary; callback
exceptions propagate, and callbacks supplement rather than replace `verbose`
logging. The default `callback=None` makes no calls. Internal cross-validation
fits used by `StabilitySelector.tune_threshold()` do not re-fire the selector's
public bootstrap callback.

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
