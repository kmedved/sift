# Advanced Workflows

This guide collects SIFT patterns for real-world datasets: time ordering,
groups, large samples, repeated targets, categorical variables, sample weights,
and q-calibrated knockoffs.

## Time Series Selection

Time-ordered data needs validation and perturbation strategies that do not let
future rows influence past rows.

### Auto-k with a Time Holdout

```python
from sift import AutoKConfig, select_mrmr

config = AutoKConfig(
    k_method="evaluate",
    strategy="time_holdout",
    val_frac=0.2,
    metric="rmse",
    max_k=100,
    min_k=5,
)

selected = select_mrmr(
    X,
    y,
    k="auto",
    task="regression",
    time=timestamps,
    auto_k_config=config,
    verbose=False,
)
```

The selector builds a supervised feature path, then evaluates prefixes on the
last `val_frac` of rows after sorting by `time`.

### Stability Selection with Blocks

```python
from sift import StabilitySelector

selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    task="regression",
    block_size="auto",
    block_method="moving",    # "moving", "circular", or "stationary"
    random_state=0,
)

selector.fit(X, y, groups=entity_ids, time=timestamps)
```

Block bootstrap samples contiguous windows within groups, preserving some local
serial dependence.

### Time-Aware Permutation Importance

```python
from sift import permutation_importance

importance = permutation_importance(
    model,
    X_test,
    y_test,
    groups=entity_ids,
    time=timestamps,
    permute_method="circular_shift",
    n_repeats=10,
    random_state=0,
)
```

Use `circular_shift` or `block` when ordinary global shuffling would destroy
the structure the model relies on.

## Grouped and Panel Data

Panel data often has repeated observations per entity. Passing groups prevents
row-level resampling from mixing entity structure in places where SIFT supports
group-aware behavior.

### Group CV for Auto-k

```python
from sift import AutoKConfig, select_jmi

config = AutoKConfig(
    k_method="evaluate",
    strategy="group_cv",
    n_splits=5,
    metric="rmse",
)

selected = select_jmi(
    X,
    y,
    k="auto",
    task="regression",
    groups=entity_ids,
    auto_k_config=config,
    verbose=False,
)
```

### Group-Aware Stability

```python
selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    task="regression",
    random_state=0,
)

selector.fit(X, y, groups=entity_ids)
```

When both `groups` and `time` are supplied, stability selection uses grouped
block bootstrap.

### CatBoost with Grouped Splits

```python
from sklearn.model_selection import GroupKFold
import sift

result = sift.catboost_select(
    X,
    y,
    task="regression",
    k=20,
    cv=GroupKFold(n_splits=5),
    group_col="entity_id",
    algorithm="forward",
)
```

CatBoost helpers accept sklearn-compatible CV splitters and can read group
labels from a column in `X`.

## Automatic Feature Counts

SIFT exposes multiple auto-k modes through `AutoKConfig.k_method`:

| Method | What it does | Typical use |
| --- | --- | --- |
| `auto` | Uses the measured router and records routing diagnostics | Zero-config CEFS+ default |
| `evaluate` | Scores prefixes on a holdout or group CV | Prediction-oriented k |
| `elbow` | Stops when objective gains flatten | Fast unsupervised-ish path sizing |
| `penalized_objective` | Applies AIC/BIC/MDL/HQC/EBIC/RIC-style penalties | Parsimonious Gaussian paths; EBIC is the measured CEFS+ default |
| `chi2_stop`, `forward_stop` | Tests CEFS+ gains against a max-over-candidates null | Calibrated no-signal stops |
| `perm_gap` | Compares CEFS+ to permutation-null objective curves | Structured/weighted null calibration |
| `gaussian_cv`, `xfit_objective` | Scores train-fold paths in Gaussian-copula space | Cheap all-k CV curves; `xfit_objective` is experimental |
| `k_posterior` | Reports pseudo-posterior mass over `k` | Uncertainty diagnostics |
| `knockoff_path` | Stops from knockoff entries in a pair-aware path | Approximate plug-in q-calibrated selected sets |
| `stability` | Uses bootstrap path reproducibility | Reproducibility diagnostics; automatic sizing is experimental |
| `changepoint`, `consensus` | Change-point diagnostic and median-of-methods | Experimental diagnostic / disagreement summary |

The zero-config CEFS+ first pass is `select_cefsplus(X, y, k="auto")`, which
currently routes to EBIC based on the Auto-K v2 benchmark campaign. Prefer
`gaussian_cv` when you specifically want fold-curve scoring, `chi2_stop` or
`forward_stop` for calibrated no-signal stops, and `perm_gap` when
groups/time/weights make analytic nulls suspicious. Inspect `changepoint`,
`stability`, `xfit_objective`, and `knockoff_path` diagnostics before trusting
their selected `k`; they remain experimental or failed-gate for automatic
sizing. This router replaces the older no-config CEFS+ split-routing behavior:
passing `groups` or `time` no longer implies `evaluate/group_cv` or
`evaluate/time_holdout`. Router branches also use method-specific effective
floors, so set an explicit method when a hard `min_k` is part of the contract.
In dense weak-signal domains, EBIC can be best read as a count of detectable
conditional signal; use `gaussian_cv` or an explicit prefix-risk curve when the
production question is predictive sufficiency.

Selection rules for `evaluate` include:

- `best`: choose the best validation score.
- `one_se`: choose a simpler prefix within one standard error.
- `plateau`: choose a point on a score plateau.
- `tolerance`: choose the smallest prefix within an absolute or relative
  tolerance of the best score.

Function selectors use prefix-only mode. Selector classes can use nested mode
where implemented, which fits a train-only path inside each validation fold.

```python
from sift import MRMRSelector, AutoKConfig

selector = MRMRSelector(
    k="auto",
    task="regression",
    auto_k_config=AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        auto_k_mode="nested",
        selection_rule="one_se",
    ),
    verbose=False,
)

selector.fit(X, y, groups=entity_ids)
```

Use nested mode when the validation estimate matters more than runtime.

## Smart Sampling

Smart sampling reduces large data before selection while keeping influential
rows and preserving group/time anchors.

```python
from sift import panel_config, smart_sample

config = panel_config(
    group_col="entity_id",
    time_col="timestamp",
    sample_frac=0.15,
)

sampled = smart_sample(
    df,
    feature_cols=feature_cols,
    y_col="target",
    config=config,
)
```

For stability selection:

```python
from sift import StabilitySelector, panel_config

selector = StabilitySelector(
    threshold=0.6,
    use_smart_sampler=True,
    sampler_config=panel_config("entity_id", "timestamp", sample_frac=0.15),
)

selector.fit(df, y)
```

Do not pass external `sample_weight` with `use_smart_sampler=True`; the sampler
creates weights for the retained rows.

## Repeated Targets and Caches

Use `build_cache` when the same `X` feeds many target vectors.

```python
from sift import build_cache, select_cached

cache = build_cache(
    X,
    sample_weight=weights,
    subsample=50_000,
    compute_Rxx=True,
    random_state=0,
)

first = select_cached(cache, y1, k=20, method="cefsplus")
second = select_cached(cache, y2, k=20, method="jmi")
third = select_cached(cache, y3, k=20, method="mrmr_quot")
```

A cache stores row subsampling, weights, feature names, valid columns, and the
rank-Gaussian representation. Do not pass new `sample_weight` values to a
selector when using a prebuilt cache; rebuild the cache with the desired
weights.

`select_fdr` also accepts a cache:

```python
from sift import select_fdr

result = select_fdr(cache=cache, y=y, q=0.1, random_state=0, verbose=False)
```

With a cache, `subsample` must be omitted and sample weights must already be in
the cache.

## Knockoff Workflows

Use knockoffs when a q-calibrated discovery set is more useful than a fixed
feature count.

```python
from sift import select_fdr

result = select_fdr(
    X,
    y,
    q=0.1,
    statistic="relevance",
    s_method="mvr",
    random_state=0,
    verbose=False,
)

ranking = result.get_feature_ranking()
```

Review these metadata fields:

| Field | Meaning |
| --- | --- |
| `fdr_control` | `"approximate_plugin"` in the default 0.7.0 path |
| `validity_model` | `"gaussian_copula_plugin"` |
| `weighted_model` | Whether non-uniform cache weights were used |
| `gamma` | Covariance shrinkage applied before sampling |
| `lambda_min` | Minimum eigenvalue after shrinkage checks |
| `s_mean` | Average knockoff separation diagnostic |
| `n_zero_weight_variance_features` | Inactive features under positive-weight support |

### Derandomized Knockoffs

```python
result = select_fdr(
    X,
    y,
    q=0.1,
    n_draws=11,
    eta=0.6,
    random_state=0,
    verbose=False,
)
```

For `n_draws > 1`, SIFT samples multiple knockoff draws and selects features
whose selection frequency is at least `eta`. This improves run-to-run stability
but remains part of the approximate plug-in contract.

### CEFS+ Knockoff Statistic

```python
result = select_fdr(
    X,
    y,
    q=0.1,
    statistic="cefsplus",
    statistic_options={"path_depth": 25, "min_gain_ratio": 1e-4},
    screen_pairs=1000,
    random_state=0,
    verbose=False,
)
```

The CEFS+ statistic is tie-safe and pair-coupled. It is useful as a
redundancy-aware second opinion, but it is slower than the default relevance
statistic. If selection count equals `selector_metadata["path_depth"]`, the cap
may be binding.

### Feature Groups

```python
groups = ["base_a", "base_a", "base_b", "base_b", "standalone"]

result = select_fdr(
    X,
    y,
    q=0.1,
    feature_groups=groups,
    random_state=0,
    verbose=False,
)
```

Feature groups threshold group-level antisymmetric statistics and then expand
selected groups back to member features. Use this for known one-hot families,
lag packs, spline bases, or other feature families. Interpret the result as
group discovery, not exact feature-level FDR within each selected group.

### KnockoffSelector

```python
from sift import KnockoffSelector

selector = KnockoffSelector(q=0.1, random_state=0, verbose=False)
selector.fit(X, y)

selector.selected_features_
selector.result_.selector_metadata
```

`KnockoffSelector` is q-based. It does not accept `k`, row `groups`, `time`, or
`auto_k_config`.

## Categorical Features

Function selectors support explicit categorical configuration:

```python
selected = select_mrmr(
    X,
    y,
    k=20,
    task="regression",
    cat_features=["league", "position"],
    cat_encoding="loo",
    allow_full_data_target_encoding=True,
    verbose=False,
)
```

Encoding options:

| Encoding | Notes |
| --- | --- |
| `none` | Input must already be numeric |
| `loo` | Leave-one-out encoding via `category_encoders` |
| `target` | Target encoding via `category_encoders` |
| `james_stein` | Shrinkage encoding via `category_encoders` |
| `loo_logit` | Built-in binary-target leave-one-out logit encoding |

For selector classes, use `cat_encoding` on the estimator constructor. If a
class was fitted with supervised categorical encoding on a DataFrame,
`transform` also requires a DataFrame so columns can be validated and encoded.

## Sample Weights

Sample weights are accepted by the main function selectors, stability
selection, permutation importance, Boruta paths, and cache construction.

```python
weights = np.ones(len(y))
weights[-100:] = 2.0

selected = select_cefsplus(
    X,
    y,
    k=20,
    sample_weight=weights,
    verbose=False,
)
```

Rules of thumb:

- Weights must be finite, non-negative, and include at least one positive row.
- Weighted caches should be rebuilt when weights change.
- Weighted `select_fdr` runs are approximate importance-weighted plug-in
  knockoff filters; do not read them as exact weighted Model-X guarantees.
- For binary CEFS+, combine `sample_weight` and `class_weight` only when the
  resulting weighting matches the estimand you want.

## Combining Methods

Different selectors answer different questions. It is often useful to compare
several diagnostics before settling on a production feature set.

```python
from collections import Counter
from sift import select_mrmr, select_jmi, select_cefsplus, select_fdr

paths = [
    select_mrmr(X, y, k=30, task="regression", verbose=False),
    select_jmi(X, y, k=30, task="regression", verbose=False),
    select_cefsplus(X, y, k=30, verbose=False),
]

knockoff = select_fdr(X, y, q=0.1, random_state=0, verbose=False)

counts = Counter()
for path in paths:
    counts.update(path)

consensus = [name for name, count in counts.items() if count >= 2]
trusted = list(knockoff.selected_features)
```

A practical workflow:

1. Use mRMR or JMI for a fast path.
2. Use auto-k or downstream CV to choose a predictive prefix.
3. Use `select_fdr` to identify q-calibrated trusted discoveries.
4. Use stability selection or Boruta when robustness or all-relevant behavior
   matters.
5. Use CatBoost selection when the final model class is tree-based and
   nonlinear interactions are central.

## Troubleshooting Cues

- Empty `select_fdr` result: valid outcome; inspect `W`, raise `q`, consider
  `offset=0`, or use derandomized draws.
- Large knockoff `gamma`: near-duplicate or ill-conditioned features may be
  reducing power.
- Auto-k chooses too many features: try `selection_rule="one_se"` or a
  tolerance rule.
- Stability selection is unstable: increase `n_bootstrap`, tune `threshold`, or
  inspect coefficient distributions.
- CatBoost selection is slow: lower `prefilter_k`, use `algorithm="forward"`,
  or provide a smaller candidate panel from a filter selector.
