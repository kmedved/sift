# Advanced Workflows

This guide collects SIFT patterns for real-world datasets: time ordering,
groups, large samples, repeated targets, categorical variables, sample weights,
and q-calibrated knockoffs. See the [glossary](glossary.md) for
[row metadata](glossary.md#row-metadata), [`target_cv`](glossary.md#target_cv),
and [approximate plugin](glossary.md#approximate-plugin) knockoff language.

## Time Series Selection

Time-ordered data needs validation and perturbation strategies that do not let
future rows influence past rows.

### Auto-k with a Time Holdout

```python
import numpy as np
import pandas as pd

from sift import AutoKConfig, select_mrmr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))
timestamps = np.arange(len(X))  # replace with the real chronological key

config = AutoKConfig(
    k_method="evaluate",
    strategy="time_holdout",
    val_frac=0.2,
    metric="rmse",
    max_k=10,
    min_k=2,
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

The selector builds a supervised feature path, then evaluates prefixes on
approximately the requested `val_frac` of latest rows after sorting by `time`.
The cut moves to the nearest boundary between distinct timestamps, so the exact
row fraction can differ. Supply `time=timestamps` (or the equivalent array)
whenever using `strategy="time_holdout"`; equal timestamps stay in the same
split. Timestamps must be non-missing and mutually orderable.

### Stability Selection with Blocks

```python
import numpy as np
import pandas as pd

from sift import StabilitySelector

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))
entity_ids = np.repeat(np.arange(20), 10)
timestamps = np.tile(np.arange(10), 20)

selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    task="regression",
    block_size="auto",
    block_method="moving",    # "moving", "circular", or "stationary"
    random_state=0,
    verbose=False,
)

selector.fit(X, y, groups=entity_ids, time=timestamps)
```

Block bootstrap samples contiguous windows within groups, preserving some local
serial dependence. `sample_frac` controls the rounded total draw budget (with
replacement); at `sample_frac=1.0` the historical full-panel budget is used.
Time values must be non-missing and orderable within each group.

### Time-Aware Permutation Importance

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from sift import permutation_importance

rng = np.random.default_rng(0)
X_test = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
y_test = 2.0 * X_test["x0"] - 1.5 * X_test["x1"] + rng.normal(scale=0.3, size=len(X_test))
entity_ids = np.repeat(np.arange(20), 10)
timestamps = np.tile(np.arange(10), 20)
model = Ridge().fit(X_test, y_test)

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
import numpy as np
import pandas as pd

from sift import AutoKConfig, select_jmi

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))
entity_ids = np.repeat(np.arange(20), 10)

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
import numpy as np
import pandas as pd

from sift import StabilitySelector

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))
entity_ids = np.repeat(np.arange(20), 10)

selector = StabilitySelector(
    n_bootstrap=50,
    threshold=0.6,
    task="regression",
    random_state=0,
    verbose=False,
)

selector.fit(X, y, groups=entity_ids)
```

### Within and between panel transforms

Regression filters accept `within="groups"` or `within="two_way"` so ranks see
within-entity (and optionally within-time) variation. Weighted entity means
are subtracted from `X` and `y` before ranks; `two_way` alternates entity and
time demeaning for a fixed five iterations. Sklearn `transform` still returns
the selected raw columns. Auto-k evaluate, Gaussian CV, and xfit-objective
fit those means on training rows only; an entity unseen in training falls
back to the training grand mean.

```python
import numpy as np
import pandas as pd

from sift import select_mrmr

rng = np.random.default_rng(0)
n_groups, n_time = 20, 10
groups = np.repeat(np.arange(n_groups), n_time)
n = n_groups * n_time
X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"x{i}" for i in range(6)])
entity = 1.5 * rng.normal(size=n_groups)[groups]
X["x0"] = X["x0"] + entity
y = entity + 0.8 * X["x1"] + rng.normal(scale=0.2, size=n)
select_mrmr(
    X,
    y,
    k=2,
    task="regression",
    groups=groups,
    within="groups",
    verbose=False,
)
```

A `return_result=True` ranking then includes `within_relevance` and
`between_relevance`. `between_relevance` summarizes weighted entity means:
it has only entity-level support, is degenerate with two or fewer
positive-mass entities, and is not on the same scale as `within_relevance`.
Demeaning can remove all variation, including singleton-only groups; the
result is then an empty selection or a no-within-signal error. Prebuilt
caches, classification, datetime/timedelta columns, and non-fold auto-k
methods are rejected rather than silently ignored.

When both `groups` and `time` are supplied, stability selection uses grouped
block bootstrap.

### CatBoost with Grouped Splits

<!-- sift-doc: requires=catboost -->

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sift

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + rng.normal(scale=0.3, size=len(X))
X["entity_id"] = np.repeat(np.arange(20), 10)

result = sift.catboost_select(
    X,
    y,
    task="regression",
    k=5,
    min_features=2,
    n_estimators=50,
    cv=GroupKFold(n_splits=5),
    group_col="entity_id",
    algorithm="forward",
    random_state=0,
    verbose=False,
)
```

CatBoost helpers accept sklearn-compatible CV splitters and can read group
labels from a column in `X`. `group_col` and `sample_weight_col` are permanent
aliases for the trailing `groups`/`sample_weight` arrays; neither spelling is
deprecated and neither warns.

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
conditional signal; use `gaussian_cv` with `selection_rule="best"` or an
explicit prefix-risk curve when the production question is predictive
sufficiency. The one-SE rule remains useful for sparse support recovery, but it
can cut too far past a shallow dense-risk knee.

Selection rules for `evaluate` include:

- `best`: choose the best validation score.
- `one_se`: choose a simpler prefix within one standard error.
- `plateau`: choose a point on a score plateau.
- `tolerance`: choose the smallest prefix within an absolute or relative
  tolerance of the best score.

Function selectors use prefix-only mode. Selector classes can use nested mode
where implemented, which fits a train-only path inside each validation fold.

```python
import numpy as np
import pandas as pd

from sift import AutoKConfig, MRMRSelector

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))
entity_ids = np.repeat(np.arange(20), 10)

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
import numpy as np
import pandas as pd

from sift import panel_config, smart_sample

rng = np.random.default_rng(0)
feature_cols = [f"x{i}" for i in range(10)]
df = pd.DataFrame(rng.normal(size=(200, 10)), columns=feature_cols)
df["entity_id"] = np.repeat(np.arange(20), 10)
df["timestamp"] = np.tile(np.arange(10), 20)
df["target"] = 2.0 * df["x0"] - 1.5 * df["x1"] + rng.normal(scale=0.3, size=len(df))

config = panel_config(
    group_col="entity_id",
    time_col="timestamp",
    sample_frac=0.3,
)

sampled = smart_sample(
    df,
    feature_cols=feature_cols,
    y_col="target",
    config=config,
    verbose=False,
)
```

For stability selection, name the feature subset explicitly: the sampler needs
the group and time columns to stay in the frame, and `feature_names` is what
keeps them (and the target) out of the candidate set.

```python
import numpy as np
import pandas as pd

from sift import StabilitySelector, panel_config

rng = np.random.default_rng(0)
feature_cols = [f"x{i}" for i in range(10)]
df = pd.DataFrame(rng.normal(size=(200, 10)), columns=feature_cols)
df["entity_id"] = np.repeat(np.arange(20), 10)
df["timestamp"] = np.tile(np.arange(10), 20)
y = 2.0 * df["x0"] - 1.5 * df["x1"] + rng.normal(scale=0.3, size=len(df))

selector = StabilitySelector(
    threshold=0.6,
    n_bootstrap=20,
    use_smart_sampler=True,
    sampler_config=panel_config("entity_id", "timestamp", sample_frac=0.3),
    random_state=0,
    verbose=False,
)

selector.fit(df, y, feature_names=feature_cols)
```

Do not pass external `sample_weight` with `use_smart_sampler=True`; the sampler
creates weights for the retained rows.

## Repeated Targets and Caches

Use `build_cache` when the same `X` feeds many target vectors.

```python
import numpy as np
import pandas as pd

from sift import build_cache, select_cached

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
weights = np.ones(len(X))
y1 = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))
y2 = X.iloc[:, 8:].sum(axis=1) + rng.normal(size=len(X))
y3 = X.iloc[:, 4:16].sum(axis=1) + rng.normal(size=len(X))

cache = build_cache(
    X,
    sample_weight=weights,
    subsample=50_000,
    compute_Rxx=True,
    random_state=0,
)

first = select_cached(cache, y1, k=8, method="cefsplus")
second = select_cached(cache, y2, k=8, method="jmi")
third = select_cached(cache, y3, k=8, method="mrmr_quot")
```

A cache stores row subsampling, weights, feature names, valid columns, and the
rank-Gaussian representation. A named cache requires a DataFrame with the same
row count and exact column order. A positional cache requires a positional
ndarray with the same row count and feature count; rebuild it from a DataFrame
to establish named-column alignment. Cache-backed filter functions reject new
`sample_weight` values and explicit `subsample`/`random_state` overrides;
rebuild the cache with the desired rows, weights, or construction seed.
`select_fdr` is different only for `random_state`: with a cache it seeds the
new knockoff draw, as in the example below. Its `subsample` argument must be
omitted and its sample weights must already be stored in the cache.

`select_fdr` also accepts the cache built just above:

<!-- sift-doc: continues -->

```python
from sift import select_fdr

result = select_fdr(cache=cache, y=y1, q=0.1, random_state=0, verbose=False)
```

With a cache, `subsample` must be omitted and sample weights must already be in
the cache. `random_state` controls this draw; it does not rebuild the cache.

## Knockoff Workflows

Use knockoffs when a q-calibrated discovery set is more useful than a fixed
feature count.

```python
import numpy as np
import pandas as pd

from sift import select_fdr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

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
| `fdr_control` | `"approximate_plugin"` in the default 0.9 path |
| `validity_model` | `"gaussian_copula_plugin"` |
| `weighted_model` | Whether non-uniform cache weights were used |
| `gamma` | Covariance shrinkage applied before sampling |
| `lambda_min` | Minimum eigenvalue after shrinkage checks |
| `s_mean` | Average knockoff separation diagnostic |
| `n_zero_weight_variance_features` | Inactive features under positive-weight support |

### Derandomized Knockoffs

```python
import numpy as np
import pandas as pd

from sift import select_fdr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

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
import numpy as np
import pandas as pd

from sift import select_fdr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

result = select_fdr(
    X,
    y,
    q=0.1,
    statistic="cefsplus",
    statistic_options={"path_depth": 16, "min_gain_ratio": 1e-4},
    screen_pairs=200,
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
import numpy as np
import pandas as pd

from sift import select_fdr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

# One label per column of X, in column order.
groups = ["base_a", "base_a", "base_b", "base_b"] + [f"standalone_{i}" for i in range(16)]

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
import numpy as np
import pandas as pd

from sift import KnockoffSelector

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

selector = KnockoffSelector(q=0.1, random_state=0, verbose=False)
selector.fit(X, y)

selector.selected_features_
selector.result_.selector_metadata
```

`KnockoffSelector` is q-based. It does not accept `k`, row `groups`, `time`, or
`auto_k_config`.
Because knockoff rows and noise are sampled stochastically, seeded results can
still change when input row order changes; preserve row order for exact
reproducibility. The narrow zero-weight guarantee is that zero-weight rows are
removed before seeded knockoff RNG draws, so they do not consume that RNG
stream.

## Categorical Features

`cat_encoding="target_cv"` is the recommended path. It is SIFT's own
cross-fitted encoder, it needs no optional dependency, and it is leakage-safe by
construction, so it takes no opt-in flag:

```python
import numpy as np
import pandas as pd

from sift import select_mrmr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"x{i}" for i in range(8)])
X["league"] = rng.choice(["nba", "wnba", "gleague"], size=len(X))
X["position"] = rng.choice(["guard", "wing", "big"], size=len(X))
league_effect = X["league"].map({"nba": 2.0, "wnba": 0.0, "gleague": -2.0})
y = league_effect + 1.5 * X["x0"] + rng.normal(scale=0.3, size=len(X))

selected = select_mrmr(
    X,
    y,
    k=4,
    task="regression",
    cat_features=["league", "position"],
    cat_encoding="target_cv",
    verbose=False,
)
```

`target_cv` emits **centered category effects**, not raw category means: out-of-fold
training rows emit `fold_encoding - fold_training_prior` and inference rows emit
`full_fit_encoding - full_training_prior`. An unknown or unseen category maps to
a zero centered effect. That is what closes the fold-marker leak: a unique ID, a
group proxy, or a timestamp proxy never appears in its own fold's training rows,
so it emits a constant zero, carries zero relevance, and cannot be selected
ahead of a real feature.

**Know the boundary of that guarantee.** Centering neutralizes only
*unseen-in-fold* emissions; it is not a defence against high cardinality as
such. A level that appears two or more times in a fold's training rows still
transmits those sibling rows' targets — ordinary target-encoding behavior — so a
*near*-unique identifier stays selectable when its rows share a latent target.
On a 300-identifier fixture with two rows each, `corr(enc(id), y)` is about 0.88
and `select_mrmr(k=2)` picks `id` first. That is genuine cross-row information
rather than leakage, so the numerics are deliberately unchanged. If it must not
reach selection, drop ID-like columns, or pass `groups=` so all of an
identifier's rows land in the same fold — under `groups=` the same column
encodes to exactly zero.

Encoding options:

| Encoding | Notes |
| --- | --- |
| `target_cv` | Built-in cross-fitted, fold-centered target encoding; no extra dependency, no opt-in flag |
| `none` | Input must already be numeric |
| `loo` | Leave-one-out encoding via `category_encoders` |
| `target` | Target encoding via `category_encoders` |
| `james_stein` | Shrinkage encoding via `category_encoders` |
| `loo_logit` | Built-in binary-target leave-one-out logit encoding |

The four supervised encoders in that table — `loo`, `target`, `james_stein`,
and `loo_logit` — fit on the full dataset, so a function-style selector refuses
them until you pass `allow_full_data_target_encoding=True`. That flag is in turn
rejected alongside `target_cv`, which is cross-fitted by construction.
`KnockoffSelector` rejects `target_cv` outright, because target-derived
preprocessing breaks the Model-X FDR claim.

Selector classes default to `cat_encoding="none"`. For selector classes, use
`cat_encoding` on the estimator constructor. If a
class was fitted with supervised categorical encoding on a DataFrame,
`transform` also requires a DataFrame so columns can be validated and encoded.

## Sample Weights

Sample weights are accepted by the main function selectors, stability
selection, permutation importance, Boruta paths, and cache construction.

```python
import numpy as np
import pandas as pd

from sift import select_cefsplus

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))

weights = np.ones(len(y))
weights[-100:] = 2.0

selected = select_cefsplus(
    X,
    y,
    k=6,
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

import numpy as np
import pandas as pd

from sift import select_cefsplus, select_fdr, select_jmi, select_mrmr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

paths = [
    select_mrmr(X, y, k=10, task="regression", verbose=False),
    select_jmi(X, y, k=10, task="regression", verbose=False),
    select_cefsplus(X, y, k=10, verbose=False),
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
3. Use `select_fdr` for a q-calibrated set under its documented plug-in
   Gaussian-copula assumptions.
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
