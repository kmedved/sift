# Sift Documentation

`sift` is a Python feature-selection toolbox for fast filter methods, stability
selection, time-aware permutation importance, Boruta/Boruta-Shap, smart sampling,
and optional CatBoost-based selection.

This document describes the current public API in this repository. The primary
imports are exposed from `sift`; advanced utilities live in submodules such as
`sift.selection`, `sift.sampling`, `sift.estimators`, and `sift._permute`.

## Documentation Map

- [Standalone API reference](docs/API.md): compact function/class reference for
  the public 0.8.0 surface.
- [Algorithm guide](docs/ALGORITHMS.md): method assumptions, output contracts,
  and selection guidance.
- [Advanced workflow guide](docs/ADVANCED.md): time series, groups, smart
  sampling, caches, categorical handling, sample weights, and knockoffs.
- [User guide](docs/user-guide.md): task-oriented examples and practical
  selector usage.
- [Troubleshooting](docs/troubleshooting.md): common errors and diagnostics.
- [Release notes](docs/release-notes.md): user-facing release changes.

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
from sift import select_mrmr, select_jmi, select_jmim, select_cefsplus, select_cefsplus_binary, select_fdr

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
knockoff = select_fdr(X, y, q=0.1, verbose=False)
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
| Knockoff FDR | `select_fdr`, `KnockoffSelector`, `sample_knockoffs` | q-calibrated approximate Gaussian-copula knockoff selection and advanced sampler access. |
| Auto-k | `AutoKConfig`, `k="auto"`, `select_k_auto`, `select_k_elbow`, `select_k_penalized_objective`, `select_k_chi2_stop`, `select_k_perm_gap`, `select_k_gaussian_cv` | Choosing feature count by holdout/CV, calibrated objective stops, permutation gaps, knockoffs, stability, or information criteria. |
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
    KnockoffSelectionResult,
    sample_knockoffs,
    select_cached,
    select_cefsplus,
    select_cefsplus_binary,
    select_fdr,
    select_jmi,
    select_jmim,
    select_mrmr,
    AutoKConfig,
    FeaturePathEvaluationResult,
    FilterSelectionResult,
    evaluate_feature_path,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
    select_k_posterior,
    path_gain_pvalues,
    select_k_changepoint,
    select_k_chi2_stop,
    select_k_forward_stop,
    bootstrap_paths,
    null_objective_paths,
    select_k_perm_gap,
    select_k_stability,
    gaussian_cv_curves,
    select_k_gaussian_cv,
    select_k_xfit_objective,
    select_k_knockoff_path,
    xfit_objective_curves,
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
    KnockoffSelector,
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
    set_verbosity,
    SelectionView,
    as_result,
)
```

The 0.9 surface currently contains 58 top-level exports: the 55 legacy names,
`set_verbosity`, `SelectionView`, and `as_result`.

## Logging

Calls with `verbose=True` emit INFO records on the `sift` logger and remain
visible by default. Use `set_verbosity("debug")`, `set_verbosity("info")`, or
`set_verbosity(None)` to select debug, normal progress, or silence globally;
per-call `verbose=False` remains silent. Application logging handlers can
capture `sift` records normally.

## Data Requirements

`select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`,
`select_cefsplus_binary`, and `select_fdr` accept
`pandas.DataFrame` or `numpy.ndarray` features and `pandas.Series` or array-like
targets.

Numeric requirements:

- Classic filter selectors require numeric features after optional categorical
  encoding.
- Gaussian/cache-based selectors require numeric features. Use categorical
  encoding before building a cache.
- Regression targets must be finite numeric values.
- Classification targets must be sklearn-compatible labels: strings/categories
  or integer-valued numeric labels. Missing labels are not allowed.
- `select_fdr` treats `y` as a continuous numeric target; integer-valued labels
  with 3-20 unique values warn because categorical multiclass workflows should
  use one-vs-rest targets.
- Feature matrices may contain non-finite values; the internal preprocessing
  mean-imputes them for most numeric selectors.
- Datetime and timedelta feature columns are rejected. Convert calendar/time
  information to explicit numeric features before selection.

Sample weights:

- `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`,
  `select_cefsplus_binary`, `select_boruta`, `select_k_auto`,
  `select_k_penalized_objective`,
  `StabilitySelector.fit`, `stability_regression`, `stability_classif`, and
  `permutation_importance` accept sample weights.
- `select_fdr` accepts sample weights on the `X` path. The weights are used in
  the plug-in Gaussian-copula model and statistics; this is an
  importance-weighted approximation, not an exact weighted Model-X guarantee.
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
For fixed-k filter calls, `groups` and `time` are rejected because they only
define auto-k evaluation splits; use `k="auto"` with an appropriate strategy
or omit those arguments. `KnockoffSelector` likewise rejects row `groups` and
`time`; use `feature_groups` for grouped feature discoveries.

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
  workers. `"auto"` resolves to `"blas"` (3-10x faster than serial in the
  repository benchmark cases, not a universal guarantee, and free of process
  start-up costs); pass `"processes"` explicitly to opt into workers.
- `n_jobs`: worker count for `mrmr_backend="processes"`. `n_jobs=0` is invalid.
  For Gaussian mRMR, `n_jobs != 1` parallelizes rank-Gaussian cache
  construction with threads when no prebuilt `cache` is supplied; a prebuilt
  cache is reused as-is.
- Gaussian mRMR compares an MI relevance against an MI redundancy on the same
  scale, so once the informative features are mutually redundant a low-relevance
  low-redundancy (noise) column can outscore them under either formula. SIFT
  warns when selected features sit below the Gaussian noise floor; prefer the
  Gaussian JMI/JMIM/CEFS+ paths, which condition on the selected set, for wide
  noisy screens.

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

The additive A2d result view supports `FilterSelectionResult`,
`KnockoffSelectionResult`, `BorutaResult`, `FeaturePathEvaluationResult`, and
`CatBoostSelectionResult` without replacing any class. It also supports the
opt-in `ImportanceResult` from `permutation_importance`; a fitted
`StabilitySelector` exposes the same view through `selector.result_view_`:

```python
view = as_result(result, input_features=X.columns)

view.features
view.indices
view.k
view.table
view.metadata
```

`return_result=True` is available on `select_mrmr`, `select_jmi`,
`select_jmim`, and `select_cefsplus`. The default remains `list[str]`. The
result object currently contains selected feature names, best-effort input-column
indices, compact selector metadata, and a conservative diagnostic ranking table.
Unknown diagnostics, such as relevance for selectors that did not expose it, are
reported as `NaN` rather than guessed.

The six legacy adapters that do not wrap fitted state are result-only:
`metadata["input_kind"]` is `"unknown"`,
and transform and inverse-transform are unavailable. Feature-path evaluation
and CatBoost supply normalized score curves;
the other supported legacy results have no retained route-level curve. CatBoost
views remain partial without an explicit raw feature identity and distinguish
the target curve count from the number of stable features actually returned.
The fitted stability adapter has a complete candidate-feature table and a
frozen fitted column-subset transform; inverse transform remains unavailable.
`ImportanceResult` records named-versus-positional provenance and retains
repeat-level score drops without retaining the model or input matrix. See
[Reading Results](docs/results.md) for the current completeness
matrix and versioned JSON contract.

Cached selectors and Gaussian filter routes can opt into proxy lookup with
`return_result=True, store_proxies=True`. The view retains only a bounded
`float32` candidate-by-selected copula-correlation block (at most 64 MiB), not
`X` or the cache. Use `view.proxies(feature)` for unique labels and
`view.proxies_at(selected_index)` when labels repeat. Classic and binary
log-loss routes reject this option.

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
  targets through quantile bins. With `sample_weight`, both bin edges and
  entropy counts use the weights; zero-weight rows affect neither.
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
    corr_prune=None,
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

- `corr_prune`: optional candidate-pruning threshold. The default `None` keeps
  possible suppressor pairs. A value such as `0.95` drops features highly
  correlated with better marginally scoring candidates and can be useful when
  duplicate suppression is more important than suppressor recovery.
- `top_m`: number of relevance-prefiltered candidates to keep before CEFS+.
- `cache`: pass a `FeatureCache` to reuse the expensive transform.

### `select_fdr`

```python
from sift import select_fdr

result = select_fdr(
    X,
    y,
    q=0.1,                    # target FDR level
    statistic="relevance",    # "relevance", "lsm", "ridge", or tie-safe "cefsplus"
    n_draws=1,                # >1 enables derandomized aggregation
    eta=0.5,                  # selection-frequency cutoff for n_draws > 1
    offset=1,                 # 1 = knockoff+, 0 = modified knockoff threshold
    s_method="equi",          # "equi", "mvr", or "me"
    feature_groups=None,      # labels, or "auto" for correlation-cluster representatives
    group_corr_threshold=0.7, # |corr| cut used by feature_groups="auto"
    sample_weight=None,
    subsample=50_000,
    random_state=0,
    verbose=False,
)

selected = result.selected_features
ranking = result.get_feature_ranking()
metadata = result.selector_metadata
```

`select_fdr` is q-calibrated rather than fixed-k. It builds or reuses the
rank-Gaussian `FeatureCache`, samples second-order Gaussian knockoffs in copula
space, computes antisymmetric feature statistics, and applies the knockoff+
threshold. In v1 the public guarantee is intentionally framed as approximate
plug-in control: metadata reports `fdr_control="approximate_plugin"` and
`validity_model="gaussian_copula_plugin"`. Exact finite-sample Model-X FDR would
require the sampled Gaussian-copula model to be the true feature distribution
and the statistic to be valid under swaps.

Important options:

- `q`: target false-discovery rate, validated in `(0, 1)`.
- `statistic="relevance"`: marginal Gaussian-MI difference between each
  original feature and its knockoff. It remains the compatibility default. No
  committed quality bakeoff establishes a universal winner among statistics;
  compare `ridge` or `lsm` when conditional signal is plausible.
- `statistic="lsm"`: lasso signed-max computed from a Gram-form LARS path on
  the analytic augmented correlation and the observed original/knockoff
  target correlations. It is exactly antisymmetric under swaps and needs no
  tie handling. Its power relative to `relevance` is data-dependent; no
  committed bakeoff supports a universal advantage. It keeps the same
  approximate plug-in validity framing.
  `statistic_options=
  {"max_steps": ...}` caps the LARS steps (columns that never enter get `W=0`).
- `statistic="ridge"`: analytic ridge coefficient difference
  `|beta_j| - |beta_j~|` with `beta = (G + lambda I)^-1 [r; r~]`; cheap,
  deterministic, and antisymmetric by construction.
  `statistic_options={"ridge_lambda": 0.5}` sets the penalty.
- `statistic="cefsplus"`: tie-safe greedy CEFS+ entry-order statistic over the
  analytic augmented covariance. It uses pair-coupled screening, disables
  correlation pruning inside the augmented run, and scores entries by marginal
  objective gain. It is slower than the default statistic, so use it as a
  redundancy-aware second opinion rather than a better default.
- `statistic_options`: for `cefsplus`, `path_depth` sets an explicit hard
  greedy-path cap. Without it, SIFT starts from a q-aware bounded depth and
  doubles the depth when the discovery set reaches that cap. Metadata reports
  the initial and final depths; an explicit saturated cap emits a warning.
  `min_gain_ratio` optionally stops once the best gain is tiny relative to the
  first gain. The default leaves gain-based early stopping disabled.
- `s_method`: `"equi"` is fastest; `"mvr"` and `"me"` use diagonal
  coordinate-descent optimizers for the MVR and maximum-entropy objectives.
  They can improve power on correlated designs; `s_mean` is diagnostic only and
  is not the optimization objective.
- `n_draws`: repeat knockoff sampling and aggregate by selection frequency.
  The configured `q` applies separately to each draw. For `n_draws > 1`, the
  final frequency vote has no established FDR control; metadata therefore
  reports `q_scope="per_draw"`, `aggregation_fdr_control="none"`, and
  `fdr_control="none"` for the aggregated set.
- `eta`: selected when `selection_frequency >= eta` for `n_draws > 1`.
- `offset=1`: knockoff+ threshold. Use `offset=0` only when the modified
  knockoff/mFDR-style threshold is acceptable.
- `screen_pairs`: pair-coupled screening cap for expensive statistics.
  The default statistic does not need screening; `cefsplus` uses it to bound
  the augmented greedy problem.
- `feature_groups`: optional per-feature group labels. When supplied, the
  knockoff threshold is applied to a heuristic signed-maximum aggregation and
  selected groups are expanded back to their active member features. This mode
  has no established group- or feature-level FDR control; metadata reports
  `group_fdr_control="none"`, `per_draw_fdr_control="none"`, and
  `fdr_control="none"`.
- `feature_groups="auto"`: clusters features by absolute correlation
  (average linkage on `1 - |corr|`, cut at `1 - group_corr_threshold`), runs
  the knockoff filter on one representative (medoid) per cluster, and expands
  selected clusters to all their members. Use it when the metadata reports a
  tiny `s_median` (near-collinear blocks such as duplicates, interactions, or
  one-hot sets make individual knockoffs nearly identical to their originals
  and destroy power). The result table adds `feature_group` and
  `is_representative` columns; the discovery unit is the cluster. The expanded
  feature list has no feature-level FDR guarantee. The representative null is
  not automatically the same as the cluster null, so cluster-level FDR is also
  unclaimed: metadata reports `representative_fdr_control="approximate_plugin"`
  for a single draw, `group_fdr_control="none"`,
  `feature_level_fdr_control="none"`, and `fdr_control="none"`. Clustering and
  linkage require O(p^2) memory/time; pre-screen very wide feature sets first.
- Power diagnostics: `selector_metadata["s_median"]` and
  `["n_low_power_features"]` report the knockoff decorrelation; a
  `UserWarning` is emitted when the median `s` is below `0.05`. Also note the
  discrete nature of knockoff+: at level `q` with `offset=1` at least
  `ceil(1/q)` features must be selected before the estimated FDP can drop to
  `q`, so problems with very few true signals may legitimately return an empty
  set at small `q`.
- `cache`: pass a prebuilt `FeatureCache`; do not pass `sample_weight` with a
  cache because weights already live in the cache.
- `sample_knockoffs(cache, random_state=...)`: advanced helper that returns one
  sampled knockoff matrix in cache/copula space.

The result table contains one row per valid cache feature with `W`, selected
status, selected input-column index, relevance, and per-draw `W_draw_*`
diagnostics. When `feature_groups` is active, `diagnostics_` also includes
per-draw `group_W_draws` and `group_thresholds`. Empty selections are valid
results: they mean no feature survived the requested q threshold.

### `select_cefsplus_binary`

```python
from sift import select_cefsplus_binary

selected = select_cefsplus_binary(
    X,
    y_binary,
    k=20,                 # or "auto"
    loss="logloss",
    top_m=None,
    corr_prune=None,
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
- `loss="brier"`: Brier proxy mode that delegates to `select_cefsplus` with
  the binary target cast to float.
- `k="auto"`: supports the same `AutoKConfig` workflow as other filter selectors.
  For `loss="logloss"`, `k_method="evaluate"` scores prefixes with a downstream
  logistic model, and `k_method="elbow"` applies elbow detection to the cumulative
  binary score-test gains. `k_method="penalized_objective"` applies BIC/AIC/HQC
  style penalties to a binary log-likelihood-gain path, using refit prefixes by
  default or cumulative score-test gains when `binary_objective_mode="score_test"`.
  For `loss="brier"`, auto-k delegates to regular CEFS+.
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
- `corr_prune`: optional absolute weighted feature-correlation pruning threshold
  in `(0, 1]`. The default `None` preserves possible suppressor pairs; pass a
  threshold such as `0.95` to opt into duplicate-oriented pruning.
- `cat_encoding="loo_logit"`: smoothed leave-one-out logit encoding for binary
  categorical features. Function-style selectors still require
  `allow_full_data_target_encoding=True` before fitting supervised encoders on
  the full matrix.
- `CEFSPlusBinarySelector(loss="brier", cat_encoding="loo_logit")` rejects
  categorical inputs because selector-class pre-encoding would otherwise use
  logit encodings while the function API maps Brier proxy mode to ordinary
  `"loo"`. Use `cat_encoding="loo"` for Brier proxy mode or
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
- `KnockoffSelector`

Each listed class subclasses sklearn's `SelectorMixin` and supports `fit`, `transform`,
`fit_transform`, `get_support(indices=False)`, `get_feature_names_out()`, and
dense `inverse_transform`. After fitting, the selectors expose:

- `selected_features_`
- `selected_indices_`
- `feature_names_in_`
- `n_features_in_`
- `get_feature_names_out()` for the selected columns
- `result_` on `KnockoffSelector`

All eight public selector classes, including Boruta and Stability, accept
`output_order="legacy"` (default) to preserve each family's historical
order: filter and knockoff path order, Boruta input order, and Stability
descending selection-frequency order with stable input-position ties. Set
`output_order="original"` for ascending fitted-column order. `transform`,
`get_support(indices=True)`, and `get_feature_names_out()` follow that order;
the boolean mask remains positional. `inverse_transform` returns a dense,
full-width array with unselected columns filled with zero and is unavailable
after supervised categorical encoding. Sparse matrices are rejected during
fit, transform, and inverse transform.

Metadata routing is additive and explicit. On sklearn 1.4+, enable routing and
call `set_fit_request(...)` for metadata the configured fit path accepts before
using `cross_validate(..., params=...)`. Fixed-k filter selectors reject
`groups`/`time`; group/time requests require `k="auto"`. Knockoff exposes only
`sample_weight`. Sklearn 1.3 callers continue to pass accepted metadata
directly to `fit`.

For pandas inputs, `transform` validates that columns match the fitted columns
and order before selecting by stored positions.

For `MRMRSelector`, `JMISelector`, `JMIMSelector`, and `CEFSPlusSelector`, the
sklearn-facing `subsample="auto"` and `random_state="auto"` defaults resolve to
50,000 rows and seed 0 when building a cache from `X`. With a prebuilt cache,
`"auto"` means the construction option was omitted; explicit integer/`None`
overrides are rejected. `KnockoffSelector` uses the same `subsample="auto"`
contract, while its numeric `random_state` continues to seed each fresh
knockoff draw.

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

`KnockoffSelector` is stochastic: even with a fixed seed, changing input row
order can change the sampled rows/noise stream and therefore the result. Keep
row order fixed for exact reproducibility. Its narrow zero-weight guarantee is
that zero-weight rows are removed before seeded knockoff RNG draws, so they do
not consume that RNG stream.

## Automatic `k` Selection

The filter selectors accept `k="auto"`. For CEFS+ and binary CEFS+, omitting
`auto_k_config` uses the measured Auto-K v2 router. The current zero-config
CEFS+ route chooses EBIC, a multiplicity-aware information criterion that
cleared the synthetic benchmark campaign with D5 null calibration and D9
runtime gates. This is a behavior change from the earlier no-config CEFS+
path: calls that pass `groups` or `time` now route to EBIC instead of
`evaluate/group_cv` or `evaluate/time_holdout`, and calls without validation
context now work instead of requiring split context.

```python
from sift import select_cefsplus

selected = select_cefsplus(X, y, k="auto", verbose=False)
```

Pass `AutoKConfig` when you want a specific method or validation strategy:

```python
from sift import AutoKConfig, select_mrmr

config = AutoKConfig(
    k_method="evaluate",      # or "auto", "gaussian_cv", "perm_gap", etc.
    selection_rule="best",    # "best", "one_se", "plateau", or "tolerance"
    auto_k_mode="prefix_only", # "prefix_only" or selector-class "nested"
    strategy="time_holdout",  # "time_holdout", "group_cv", or "kfold" for fold curves
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

Four additive presets cover the common intent-first cases without changing the
flat constructor:

```python
router = AutoKConfig.default()
predictive = AutoKConfig.predictive(strategy="kfold", rule="best", n_folds=5)
discovery = AutoKConfig.discovery(alpha=0.05)
downstream = AutoKConfig.downstream("group_cv", "rmse", "best")
```

`predictive(..., n_folds=...)` writes `xfit_folds`; it does not overwrite the
distinct evaluate/nested field `n_splits`. `downstream` retains the current
evaluate contract and therefore rejects `strategy="kfold"`.

For larger configurations, module-scoped frozen option objects can be flattened
through `from_groups`:

```python
from sift.selection.auto_k_options import AutoKCVOptions, AutoKPermutationOptions

config = AutoKConfig.from_groups(
    k_method="perm_gap",
    min_k=0,
    cv=AutoKCVOptions(strategy="group_cv"),
    perm=AutoKPermutationOptions(perm_B=50, perm_null="within_group"),
)
```

The 49 original flat fields remain the only stored dataclass fields. Group
views such as `config.cv` and `config.perm` are immutable snapshots; they do
not change constructor signatures, equality, `repr`, `dataclasses.replace`, or
pickle state. `from_groups` rejects unknown fields, wrong group types, and a
field supplied both flat and through a group.

| Owner | Canonical flat fields |
| --- | --- |
| Core | `k_method`, `max_k`, `min_k`, `random_state` |
| `objective` | `objective_penalty`, `objective_penalty_weight`, `objective_n_eff`, `n_eff_mode`, `ebic_gamma`, `binary_objective_mode` |
| `test` | `alpha`, `m_mode`, `stop_patience` |
| `perm` | `perm_B`, `perm_null`, `gap_rule` |
| `knockoff` | `knockoff_q`, `knockoff_draws`, `knockoff_s_method`, `knockoff_return` |
| `cv` | `strategy`, `metric`, `val_frac`, `n_splits`, `xfit_folds`, `xfit_mode`, `xfit_ridge`, `selection_rule`, `one_se_multiplier`, `score_abs_tol`, `score_rel_tol`, `plateau_prefer`, `plateau_min_points` |
| `stability` | `boot_B`, `boot_mode`, `stability_rule`, `stability_pi` |
| `experimental` | `auto_k_mode`, `elbow_min_rel_gain`, `elbow_patience`, `posterior_level`, `posterior_pick`, `floor_z`, `floor_window`, `consensus_methods`, `auto_dense_check`, `auto_dense_min_k`, `auto_dense_min_frac`, `auto_dense_disagreement_ratio` |

The similarly named fields remain distinct: `n_splits` controls evaluate and
nested folds while `xfit_folds` controls Gaussian fold curves;
`elbow_patience` confirms an elbow while `stop_patience` controls sequential
stops; `objective_n_eff` is an explicit override while `n_eff_mode` chooses
the effective-sample-size rule. Non-default fields that the chosen method does
not consume emit a caller-facing `UserWarning`; internally routed copies do not
repeat or invent those warnings.

`AutoKConfig` fields:

- `k_method="auto"` delegates to the measured router. CEFS+ and binary CEFS+
  route no-config `k="auto"` through this path automatically. The router
  records `auto_routing` diagnostics with the chosen method, reason, and cheap
  facts such as Kish effective sample size, candidate count, weight skew, and
  the presence of groups/time. The shipped CEFS+ rule table uses EBIC as the
  measured default, switches to EBIC when `p_valid > n_eff_kish`, uses
  `perm_gap` for heavy weight skew, and sends explicit non-CEFS+ Gaussian
  auto routes to `gaussian_cv/one_se`. Router branches use method-specific
  effective floors (`0` for EBIC and permutation-gap stops, at least `1` for
  Gaussian CV curves); pass an explicit `AutoKConfig(k_method=...)` when you
  need a hard `min_k` contract. In dense weak-signal domains, set
  `auto_dense_check=True` to have large EBIC picks cross-checked against
  `gaussian_cv` with `selection_rule="best"`; the router warns when EBIC's
  detectable-feature count and the Gaussian CV sufficiency pick disagree
  sharply. The dense check is a Gaussian CEFS+ diagnostic: binary log-loss
  CEFS+ automatic routing has no dense-regime diagnostic and rejects
  non-default `auto_dense_*` settings with a `ValueError` instead of silently
  ignoring them. Binary Brier selection delegates to Gaussian CEFS+ and
  therefore follows the Gaussian dense-check contract.
- `k_method="evaluate"` evaluates prefixes of the selected path with a simple
  downstream model. `selection_rule="best"` preserves the historical argmin
  behavior. `selection_rule="plateau"` and `"tolerance"` require
  `score_abs_tol` or `score_rel_tol` and pick a practically tied prefix;
  `selection_rule="one_se"` uses split-level standard errors when at least two
  finite split scores are available.
- `k_method="elbow"` chooses where objective gains flatten out. It requires an
  objective path and is currently supported by Gaussian/cache-backed filter
  paths such as CEFS+ and Gaussian mRMR, plus binary CEFS+ log-loss paths using
  cumulative score-test gains. Once `patience` consecutive low gains confirm
  the elbow, the returned prefix ends before the first low-gain feature unless
  retaining it is required by the `min_k` floor.
  Classic MRMR/JMI/JMIM auto-k paths use
  `k_method="evaluate"`; requesting elbow there raises.
- `k_method="penalized_objective"` chooses from a CEFS+ proxy objective without
  validation context. Gaussian CEFS+ uses `n_eff * O_k - penalty * k`, where
  `O_k` is the log-det Gaussian MI proxy. Binary CEFS+ log-loss uses
  `2 * G_k - penalty * k`, where `G_k` is a prefix log-likelihood gain. This is
  proxy-objective IC selection, not downstream-performance optimization.
  `objective_penalty="ebic"` and `"ric"` add multiplicity-aware penalties for
  wide screens; EBIC/RIC use Kish effective sample size by default. EBIC is the
  measured CEFS+ default from the Auto-K v2 synthetic campaign; RIC is exposed
  for comparison and may be more conservative.
- CEFS+ also supports calibrated path methods: `k_method="chi2_stop"` tests
  each greedy gain against a max-over-candidates F null, `k_method="forward_stop"`
  applies the ordered ForwardStop accumulation rule to those p-values, and
  `k_method="changepoint"` estimates a gain-path change point. `chi2_stop` and
  `forward_stop` are the calibrated no-signal stops; `changepoint` is currently
  an experimental diagnostic and should not be used as a default. These methods
  may legitimately return `k=0` when no discovery is supported.
- `k_method="perm_gap"` reruns CEFS+ on structure-aware permuted targets and
  compares the real objective curve to the null envelope. Use `perm_null="auto"`
  by default, or set `"within_group"` / `"circular_shift"` when the null must
  preserve groups or time structure.
- `k_method="gaussian_cv"` and `k_method="xfit_objective"` build train-fold
  paths and score all prefix lengths in Gaussian-copula correlation space.
  They support `strategy="time_holdout"`, `"group_cv"`, or `"kfold"` and are
  also available on Gaussian mRMR/JMI/JMIM paths. Function-style selectors
  currently support `xfit_mode="shared_z"`; `xfit_mode="exact"` is reserved for
  fold-local cache rebuilding. `gaussian_cv` is the preferred fold-curve method;
  use `selection_rule="one_se"` for sparse support recovery and
  `selection_rule="best"` when the production question is dense-regime
  predictive sufficiency. `xfit_objective` uses a one-SE default plus a null
  guard but remains experimental pending broader gates.
- `k_method="k_posterior"` reports a pseudo-posterior over `k`, including HPD
  range, `P(k=0)`, and entropy. When `min_k > 0`, HPD is computed over
  selectable `k`, while `P(k=0)` still reports the unselectable zero-feature
  mass. It is relative evidence along one greedy path, not full Bayesian model
  averaging.
- `k_method="knockoff_path"` interleaves cache-space knockoffs with originals
  and stops from knockoff entries. The default `knockoff_return="set"` returns
  an approximate Gaussian-copula knockoff set under the plug-in feature model;
  `knockoff_return="prefix"` returns a same-size CEFS+ prefix and labels it as
  count-only rather than FDR-controlled.
- `k_method="stability"` chooses from a bootstrap stability curve and reports
  per-feature selection frequencies in the summary. It is useful as a
  reproducibility diagnostic; automatic sizing from the stability curve remains
  experimental after the Auto-K v2 benchmark gates. Bootstrap paths are
  compared only through the deepest prefix available in every run; short paths
  are not padded with fabricated feature entries.
- `k_method="consensus"` runs a configured subset of methods and returns the
  lower median `k`, with a disagreement diagnostic.
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
  `k_method="elbow"` and `k_method="penalized_objective"` raise instead of
  being silently ignored. Function-style selectors still reject nested mode and
  keep the prefix-only contract.
- Selector classes reject `k_method="evaluate"`, `auto_k_mode="prefix_only"`,
  and explicit supervised categorical encoding together, because that would
  evaluate already target-encoded validation rows. Use nested mode or
  pre-encode/cross-fit categoricals outside the selector.
- `strategy="time_holdout"` requires `time` and validates on approximately the
  requested latest-row fraction after moving the cut to a boundary between
  distinct timestamps for evaluate/fold-curve methods.
- `strategy="group_cv"` requires `groups` and uses `GroupKFold`.
- `strategy="kfold"` is available for `gaussian_cv` and `xfit_objective`.
- Wherever the route accepts row context, DataFrame callers may use
  `groups="column_name"` or `time="column_name"`. SIFT extracts that column
  positionally and removes it from the feature matrix; direct arrays remain
  positional and fixed-k filter calls still reject row context.
- If `auto_k_config` is omitted, CEFS+ and binary CEFS+ use
  `k_method="auto"`. Other selectors keep the legacy inference:
  `strategy="time_holdout"` when `time` is supplied and `strategy="group_cv"`
  when only `groups` is supplied.
- `metric="auto"` resolves to RMSE for regression and log loss for
  classification. `metric=` also accepts estimator-style sklearn scorer
  objects; their signed higher-is-better output is negated into SIFT's
  historical lower-is-better curve.
- `min_k` may be zero for discovery/error-controlled methods. Mean-one weight
  normalization makes `n_eff_mode="weight_sum"` equal to the row count even
  under skewed weights; v2 inference methods use Kish effective sample size
  when `n_eff_mode="auto"`.

Important v2 knobs:

| Field | Used by | Meaning |
| --- | --- | --- |
| `alpha` | `chi2_stop`, `forward_stop`; `perm_gap` with `gap_rule="gain_envelope"` | Path-test level. |
| `m_mode` | `chi2_stop`, `forward_stop` | Max-candidate correction width. |
| `stop_patience` | `chi2_stop`, `changepoint`; `perm_gap` with `gap_rule="gain_envelope"` | Consecutive-stop patience. |
| `perm_B`, `perm_null`, `gap_rule` | `perm_gap` | Number of null paths, structure-preserving null type, and gap decision rule. |
| `knockoff_q`, `knockoff_draws`, `knockoff_s_method`, `knockoff_return` | `knockoff_path` | Target q, derandomized draws, sampler construction, and set-vs-prefix return contract. |
| `xfit_folds`, `xfit_mode` | `gaussian_cv`, `xfit_objective` | Fold count and shared-copula mode. |
| `xfit_ridge` | `gaussian_cv` | Gaussian linear-risk ridge. |
| `ebic_gamma` | EBIC `penalized_objective`, `k_posterior` | Multiplicity penalty strength. |
| `objective_n_eff`, `n_eff_mode` | Objective/posterior and gain-test methods | Explicit effective sample size and its fallback rule. |
| `posterior_level`, `posterior_pick` | `k_posterior` | HPD mass and MAP-vs-smallest-HPD selection rule. |
| `boot_B`, `boot_mode`, `stability_rule` | `stability` | Bootstrap count/mode and sizing rule. |
| `stability_pi` | `stability` with `stability_rule="pi_threshold"` | Selection-frequency threshold. |
| `floor_z`, `floor_window` | `changepoint` | Experimental noise-floor diagnostic parameters. |
| `consensus_methods` | `consensus` | Submethods included in the lower-median consensus. |
| `auto_dense_check`, `auto_dense_min_k`, `auto_dense_min_frac`, `auto_dense_disagreement_ratio` | `auto` (Gaussian CEFS+, including the binary Brier delegate; binary log-loss rejects non-default values) | Optional dense-regime EBIC-vs-`gaussian_cv/best` warning thresholds. |

Supported function-style routes:

| Route | Supported `k_method` values |
| --- | --- |
| Classic mRMR/JMI/JMIM | `evaluate` |
| Gaussian mRMR/JMI/JMIM | `auto`, `evaluate`, `elbow`, `gaussian_cv`, `xfit_objective`, `stability` |
| CEFS+ | `auto`, `evaluate`, `elbow`, `penalized_objective`, `k_posterior`, `chi2_stop`, `forward_stop`, `changepoint`, `perm_gap`, `knockoff_path`, `gaussian_cv`, `xfit_objective`, `stability`, `consensus` |
| Binary CEFS+ | `auto`, `evaluate`, `elbow`, `penalized_objective`, `k_posterior`, `changepoint` |

Which method should I use?

| Situation | Good starting point |
| --- | --- |
| You want the measured CEFS+ default | `select_cefsplus(X, y, k="auto")` |
| You have enough rows and want predictive prefix size | `gaussian_cv` with `selection_rule="best"` and `strategy="kfold"`/`"group_cv"`, or a real holdout |
| Dense weak-signal domain where EBIC returns hundreds of detectable features | `gaussian_cv` with `selection_rule="best"` or explicit prefix-risk scoring; read EBIC as support detectability |
| You have no validation context and need an IC-style baseline | `penalized_objective` with `objective_penalty="ebic"` |
| You need a calibrated no-signal stop | `chi2_stop` or `forward_stop` with `min_k=0` |
| Weights, ranks, groups, or time structure make analytic nulls suspect | `perm_gap` with the appropriate `perm_null` |
| You need an approximate plug-in q-calibrated selected set | `knockoff_path` with `knockoff_return="set"` |
| You need to know whether the list is reproducible | `stability` diagnostics |
| Methods disagree and you want a conservative summary | `consensus` and inspect `consensus_spread` |

Recommended patterns:

```python
# Split-aware predictive selection.
AutoKConfig(
    k_method="evaluate",
    selection_rule="plateau",
    score_rel_tol=0.001,
    plateau_prefer="smallest",
    max_k=300,
)

# No validation split: CEFS+ measured default.
AutoKConfig(
    k_method="auto",
    max_k=300,
)
```

Advanced helpers:

```python
from sift import (
    AutoKConfig,
    compute_objective_for_path,
    select_k_auto,
    select_k_elbow,
    select_k_penalized_objective,
)

best_k, best_features, diagnostics = select_k_auto(
    X,
    y,
    feature_path=selected_path,
    config=config,
    groups=groups,
    task="classification",
)

objective_path = compute_objective_for_path(cache, y, selected_path)

best_k, elbow_diagnostics = select_k_elbow(
    objective_path,
    min_k=5,
    max_k=100,
    min_rel_gain=0.02,
    patience=3,
)

best_k, ic_diagnostics = select_k_penalized_objective(
    objective_path,
    config=AutoKConfig(k_method="penalized_objective"),
    objective_scale="n_eff",
    n_samples=len(y),
)
```

Research-oriented auto-k helpers are also available through an additive staged
namespace:

```python
import sift.experimental as experimental

curves = experimental.gaussian_cv_curves(...)
```

Accessing one of the 16 names through `sift.experimental` emits a
`FutureWarning` because those helpers may change before 1.0. Their existing
top-level imports remain available and warning-free throughout 0.9;
`sift.__all__` remains the same ordered 58-name surface. The option-group types
remain module-scoped and are not added to either top-level export list.

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
    scoring="rmse",           # "rmse", "mae", sklearn scorer, or callable
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
Estimator-style sklearn scorers are evaluated against the fitted estimator and
negated into the result's historical lower-is-better score convention. A
weighted evaluation requires a scorer whose score function accepts
`sample_weight`; incompatible scorers raise instead of producing an all-infinite
curve.

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
    rank_backend="serial",    # "serial", "threads", or "processes"
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
- `feature_names_are_synthetic`: whether names were generated for unnamed
  positional input instead of coming from a DataFrame.

`FeatureCache` objects are bound to the feature matrix and naming contract used
when they were built: reuse requires the same row count and feature order/names
in `X`. Cache-backed filter calls (`select_mrmr`, `select_jmi`, `select_jmim`,
and `select_cefsplus`) reject call-time `sample_weight` and explicit
`subsample` or `random_state` overrides; those values belong to cache
construction. `select_fdr` still accepts `random_state` with a cache because
there it seeds the new knockoff draw, not cache construction; its
`sample_weight` and `subsample` arguments remain forbidden.
An array-built positional cache is matched by row count and feature count and
requires a positional ndarray at selection time; named DataFrames are rejected
because the original column alignment cannot be proven. Rebuild the cache from
that DataFrame to establish its names and order.
If you persist or pickle caches across SIFT versions,
rebuild or invalidate caches created before `feature_names_are_synthetic` was
recorded; old objects cannot prove whether `x0`, `x1`, ... names were real
DataFrame columns or generated positional names.

Pass `rank_backend="threads"` (preferred; the sort and quantile transform
release the GIL, so no copies of `X` are made) or `"processes"` with `n_jobs`
to parallelize the rank-Gaussian column transforms during cache construction.
Passing a prebuilt cache into `select_mrmr(..., estimator="gaussian")` reuses
that cache and does not rebuild it; build the cache with
`build_cache(..., n_jobs=..., rank_backend="threads")` when you want parallel
cache construction.

`select_cached` uses `corr_prune="auto"` by default. It resolves to no pruning
for every method so possible suppressor pairs remain eligible and selectors can
return `k` features when enough candidates survive `top_m` and validity
filtering. Pass a float such as `corr_prune=0.95` to opt into duplicate-oriented
pruning for any cached method.

`select_cached(..., return_objective=True)` returns `(features, objective_path)`.
`select_cached(..., return_result=True)` instead returns a complete
`SelectionView` with selected positions, cache provenance, relevance, and the
objective path. It cannot be combined with `return_objective` or
`return_indices`; all four historical list/tuple shapes remain the default.

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
    alpha_rule="one_se", # selection-oriented; use "best" for predictive CV
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
`k` caps the count via `max_features`; only features whose selection frequency
clears `threshold` are returned, so wrappers can return fewer than `k`
features (including zero) instead of padding with never-selected features.

When `alpha=None`, the default `alpha_rule="one_se"` chooses the strongest
regularization within one standard error of the best CV score. This is a
selection-oriented default that reduces noise-variable frequency; use
`alpha_rule="best"` when predictive CV error is the primary objective.
For compatibility, 0.9 passes sample weights to the sparse model fits in this
alpha search but retains the historical unweighted validation score and scaler.
By contrast, `tune_threshold` forwards fit weights and scores validation rows
with the supplied weights.

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
X_restored = selector.inverse_transform(X_reduced)
# sklearn-compatible selected names:
selected_names_out = selector.get_feature_names_out()
# normalized, non-cached fitted view:
view = selector.result_view_
```

`output_order="legacy"` preserves descending selection-frequency order;
`output_order="original"` emits selected columns in fitted input order. The
same order governs transform, selected support indices, feature names, and
dense inverse transformation. The fitted selector can invert a reduced dense
matrix by zero-filling unselected columns; its frozen result view cannot.

`StabilitySelector.transform` always returns an ndarray. A selector fitted on
a DataFrame selects transformed DataFrames by fitted column name, so extra or
reordered columns are accepted; duplicate labels and missing selected columns
raise clear errors. `tune_threshold` applies the same identity checks while
requiring every fitted feature. The same named paths are available when fitting
an ndarray with explicit `feature_names`. By contrast, a selector fitted on an
unnamed positional ndarray rejects DataFrame input to either method because its
generated `x0`, `x1`, ... names cannot establish column identity. Keep using
same-width ndarrays, provide explicit names when fitting the ndarray, or refit
on a DataFrame. Ndarray transforms are positional and must retain the fit-time
feature count.

The fitted view's `features` are names while `indices` preserve the legacy
integer positions. Its complete table is over `feature_names_in_`, which is the
fitted candidate-feature namespace when an explicit subset or smart sampler is
used. `view.transform(X)` freezes the selected identity at view creation,
preserves the selector's `set_output` configuration, and does not retain
training rows or the bootstrap coefficient matrix.

Important options:

- `n_bootstrap`: number of resamples.
- `sample_frac`: fraction per bootstrap; grouped time-block fits apply it to a
  rounded panel-wide draw budget allocated across groups.
- `threshold`: minimum selection frequency.
- `alpha`: regularization strength. If `None`, Sift estimates it by CV.
- `penalty`: additive alias for `alpha`; pass one or give both the same value.
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

`tune_threshold` refits stability selection inside each outer fold. If the
original `fit` used `sample_weight`, `groups`, or `time`, pass those same arrays
to `tune_threshold`; it requires the fit-time context, uses group-disjoint or
time-ordered outer folds, and forwards the context to each fold-local selector.
Its `scoring=` argument accepts sklearn scorer objects as well as scorer names.
With DataFrames, `groups="column"` and `time="column"` are equivalent to
passing the corresponding positional arrays while excluding those columns from
the fitted feature namespace.

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
    sample_frac=0.5,
    verbose=False,
)
```

The grouped bootstrap allocates a rounded panel-wide draw budget across groups
and samples contiguous windows with replacement. `sample_frac=1.0` retains the
full-panel draw budget used by earlier releases.

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

Residual-enabled sampling keeps regression targets in float64 and estimates
row difficulty with a two-fold cross-fitted pilot, so no row receives a
residual from a model trained on that row. Non-pilot rows are assigned to one
unseen fold model so their residual scale matches the pilot rows; predictions
are not ensembled. Constant pilot targets disable the residual blend.

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

When `use_smart_sampler=True`, `X` must be a DataFrame. An explicit
`fit(..., feature_names=[...])` sequence is preserved as an ordered feature
subset rather than widened to every numeric column. Explicit feature names must
refer to numeric columns after the sampler's metadata exclusions. Pass an
ordered iterable such as a list, tuple, pandas Index, or one-dimensional NumPy
array; strings, bytes-like objects, mappings, sets, scalar arrays, and
matrix-like containers are rejected. Tuple and MultiIndex column labels remain
single feature names in `get_feature_names_out`. Configured `group_col` and
`time_col` columns remain metadata and are excluded from the feature matrix even
if they appear in `feature_names`. Without explicit names, the sampler uses
numeric columns other than those group/time columns. Datetime and timedelta
feature columns are rejected before numeric coercion; a configured datetime
`time_col` remains valid sampler metadata. Fold-local `tune_threshold` fits
retain the configured group/time metadata while scoring only fitted features.

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
selected_names_out = selector.get_feature_names_out()
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
  fitting Boruta. Supervised encodings (`"loo"`, `"target"`, `"james_stein"`,
  `"loo_logit"`) are fitted on the full dataset, which tree learners can
  invert (leave-one-out values within a category are monotone in the row's own
  target), so Boruta requires `allow_full_data_target_encoding=True` before
  applying them, exactly like the filter selectors. Prefer pre-encoding
  categoricals out-of-fold and passing `cat_encoding="none"`.
  In `importance_data="test"` mode, supervised encodings are rejected outright.
  `cat_encoding="none"` is allowed only when the categorical columns are
  already numeric/pre-encoded; raw object/category columns still fail the
  numeric-input check.

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
to models that require them. NumPy inputs keep the ndarray path. DataFrame
callers may pass `groups="column"` and `time="column"`; those metadata columns
are removed before prediction. Direct arrays remain positional.

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

rich_importance = permutation_importance(
    model,
    X,
    y,
    scoring="neg_rmse",
    n_repeats=20,
    random_state=0,
    return_result=True,
)
repeat_drops = rich_importance.importances_  # raw feature positions × repeats
view = rich_importance.result_view()
```

Output columns:

- `feature`
- `importance_mean`
- `importance_std`
- `baseline_score`

The four-column DataFrame remains the default return. With
`return_result=True`, `ImportanceResult.ranking_` is exactly that summary and
`ImportanceResult.importances_` retains each normalized score drop in original
feature-position order. `result_view()` exposes all evaluated features in
importance order with `selection_semantics="ranking_only"`; it does not invent
a cutoff or retain the fitted model or input matrix.

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

Sklearn scorer objects are accepted directly and are treated as
higher-is-better, matching sklearn's scorer convention. Legacy callbacks with
the signature `callback(y_true, y_pred, sample_weight)` remain supported and
default to higher-is-better; pass `higher_is_better=False` when the callback
returns a loss such as positive MSE. Named scorers, sklearn scorer objects, and
`ScoringSpec` use their own direction metadata and reject that override.
Custom estimator-style callbacks should be wrapped in `ScoringSpec`; bare
three-argument callables retain the legacy `(y_true, y_pred, sample_weight)`
contract because that signature is otherwise ambiguous.

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
    groups=None,
    time=None,
    sample_weight=None,
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

Choosing `k` automatically (`k=None`):

- Every feature count in the search grid (`min_features`, `step_function`, or
  `feature_counts`) is scored on the validation splits. The score-curve optimum
  is the global arg-best over those scores (exact ties prefer fewer features)
  and is reported in the verbose output as the best-scoring `k`.
- The returned `selected_features`/`result.best_k` apply a parsimony rule:
  walking down from the best-scoring count, SIFT takes the smallest count whose
  mean score stays within `tolerance` (relative to `|best_score|`, default
  `0.01`) of the best, giving up after `selection_patience` consecutive counts
  outside that band. `tolerance=0` selects the best-scoring count unless
  smaller counts tie it exactly. `result.features_within_tolerance(...)` applies
  the requested band and the fit-time `selection_patience` to the stored per-`k`
  feature lists after the fact.

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
- `groups`, `time`, and `sample_weight` accept direct positional arrays.
- DataFrame `groups="column"` and `time="column"` extract and remove those
  columns. `group_col` and `sample_weight_col` remain compatibility aliases;
  direct-plus-alias combinations raise.
- Supplied `time` is validated and stably orders every aligned row before the
  configured splitter. Use an explicitly time-aware splitter for chronological
  validation; the default remains the legacy random split.
- `sample_weight_col` is extracted and used as CatBoost weights.
- Misspelled `group_col` or `sample_weight_col` names raise `ValueError`
  instead of silently disabling groups or weights.
- `catboost_params` collisions with translated SIFT arguments emit one
  `UserWarning` while retaining the 0.9 dictionary-wins precedence.

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
- `selection_patience`

Methods:

```python
mean, std = result.score_at_k(20)
small_set = result.features_within_tolerance(tolerance=0.01)
fig, ax = result.plot_scores_vs_k()
```

### Progress callbacks

Long-running filter paths (including `select_cached` and the filter selector
classes), stability, Boruta, and CatBoost selectors accept
`callback(step, total, info)`. Steps are one-based and are reported after each
completed greedy-path step, bootstrap, Boruta iteration, or CatBoost split,
respectively. Each call receives a fresh snapshot dictionary; callback
exceptions propagate, and callbacks supplement rather than replace `verbose`
logging. The default `callback=None` makes no calls.

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

The `sift.estimators` package exposes four lazily loaded modules:

- `sift.estimators.relevance`
- `sift.estimators.copula`
- `sift.estimators.joint_mi`
- `sift.estimators.knockoffs` (`fit_gaussian_knockoffs`, `sample_gaussian_knockoffs`,
  `gaussian_knockoff_mean`, `GaussianKnockoffModel` — the advanced sampler API
  behind `select_fdr`)

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

mi = joint_mi.r2_joint_mi(selected_feature, candidate_matrix, y, weights)
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
- `infer_higher_is_better`: resolves metric direction for common metrics; unknown
  metrics require an explicit `higher_is_better` value.
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
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
```

The shared JSON schema records baseline/current wall time, optional peak memory,
semantic parity, benchmark kind, and promotion status. Promotion rows are
intended as gates; informational/smoke/parity rows are coverage and comparison
signals, not proof that an optimization was promoted. Generated benchmark JSON
files are local artifacts and are not committed by default. The knockoff
benchmark emits informational rows for the `select_fdr` timing surface.

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
- For classic mRMR, `mrmr_backend="auto"` resolves to the BLAS redundancy path;
  compare it with the explicit `serial` and `processes` options on your data
  with `benchmarks/bench_mrmr.py`. Use
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

In 0.9, omitted/`None` `random_state` values on `StabilitySelector`,
`permutation_importance`, and CatBoost selectors emit a `FutureWarning` but
remain nondeterministic. SIFT 1.0 will resolve those defaults to seed 0. The
literal-42 defaults on `AutoKConfig`, `SmartSamplerConfig`, and
`evaluate_feature_path` remain unchanged and warning-free in 0.9.

Parallel defaults also remain unchanged in 0.9: filter/cache entry points use
`n_jobs=1`, while Stability, permutation importance, and CatBoost use
`n_jobs=-1`. CatBoost GPU execution does not translate `n_jobs` into a CPU
thread count.

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
