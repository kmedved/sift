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
time demeaning until convergence (at most 200 passes). Sklearn `transform` still returns
the selected raw columns. Auto-k evaluate, Gaussian CV, and xfit-objective
fit those means on training rows only. Unseen entity levels use the training
grand mean for the entity effect; unseen time levels add no time effect. A
warning reports affected rows. Routes where none can be seen are
rejected up front: `within="two_way"` validates only under
`k_method="gaussian_cv"` or `"xfit_objective"` with `strategy="kfold"`, which
run on the Gaussian path (`select_cefsplus`, or `estimator="gaussian"` for
mRMR, JMI and JMIM). `within="groups"` also validates those two methods with
`strategy="time_holdout"`, and `evaluate` with `time_holdout`, when entities
persist across the holdout boundary. The `kfold` route still warns when every
row of an entity or period lands in one validation fold; drop or pool such
thin levels, or raise `AutoKConfig.xfit_folds`. When thin levels dominate, a
whole validation fold can hold no seen level, which raises; more folds make
that likelier, so drop or pool those levels or lower `xfit_folds`.

`include`, `exclude` and `candidates` need an auto-k method that truncates the
conditioned path, while `gaussian_cv` and `xfit_objective` rebuild an
unconditioned one. Combined with `within`, `k="auto"` therefore works only for
`within="groups"` with `evaluate` and `time_holdout`. Otherwise pass a fixed
`k` (a fixed-`k` `within="groups"` call takes no `time`), or drop the
conditioning keywords and use one of the Gaussian-path methods above.

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
methods are rejected rather than silently ignored. Two-way result metadata
reports the path-building fit as `within_two_way_iterations`,
`within_two_way_converged`, and `within_two_way_max_residual`; a fit that
stops at the 200-pass cap also warns.

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

`sample_rows` keeps a fraction of a large table so an expensive model trains
faster. Its default `method="uniform"` is a simple random sample.
`method="chronological"` draws with equal probability, spread over each
entity's timeline with entity and season quotas. The opt-in
`method="residual"` draws rows in proportion to how wrong a cheap pilot model
is on them, half evenly, along a random order, and gives each kept row a
mean-one fitting weight.

Typical workflow on a large panel: select features on all eligible rows when
that is affordable, then train the final model on a 20% chronological sample
of all eligible rows, using the selected features. On the panel below the
residual draw tied chronological sampling and took longer, so it is opt-in,
for expensive, weight-aware fits. The evidence for selecting on a sample is
thinner. If selection is too slow, start with a 20% chronological sample with
season strata, as the example does.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sift import panel_config, sample_rows, select_mrmr, smart_sample

rng = np.random.default_rng(0)
feature_cols = [f"x{i}" for i in range(10)]
df = pd.DataFrame(rng.normal(size=(20_000, 10)), columns=feature_cols)
df["entity_id"] = np.repeat(np.arange(200), 100)
df["timestamp"] = np.tile(np.arange(100), 200)
df["target"] = 2.0 * df["x0"] - 1.5 * df["x1"] + rng.standard_t(3, size=len(df))

# 1. If selection on every row is too slow, select on a chronological sample.
screen = smart_sample(df, feature_cols, "target",
                      panel_config("entity_id", "timestamp", size=0.2))
chosen = select_mrmr(screen[feature_cols], screen["target"], k=3, task="regression",
                     sample_weight=screen["sample_weight"].to_numpy())

# 2. Train on a 20% chronological sample of all rows, with the chosen features.
s = sample_rows(size=0.2, method="chronological", groups=df["entity_id"],
                time=df["timestamp"], random_state=1)
model = HistGradientBoostingRegressor().fit(
    s.take(df[chosen]), s.take(df["target"]), sample_weight=s.weights
)

# Opt-in alternative for an expensive, weight-aware learner:
r = sample_rows(df[chosen], df["target"], size=0.2, method="residual", random_state=1)
```

### How much to sample

These numbers come from a benchmark on a ~770k-row player-season panel: 5
targets, scored in 12 target/window combinations, refitting a gradient-boosted
model on 50 features chosen beforehand, with the learning rate pinned. Each
figure is the mean RMSE gap to a full-data fit on the 9 stable combinations,
over 20 fresh seeds.

The residual column is an earlier draw that ran systematically along each
entity's timeline. That draw can line up with residuals that repeat every
`1 / fraction` rows, so the shipped residual method draws along a random order
instead. A targeted check at 20% of rows (20 seeds, the same pilot) found that
the random order loses most of the edge:

- Every draw had exactly the requested number of rows.
- On the 9 stable combinations, the random-order draw was 0.39% [0.23, 0.54]
  worse than the time-ordered draw, and 0.18% [0.02, 0.34] worse than
  chronological sampling at 20% (0.19% [0.02, 0.38] worse than chronological
  at 25%, which takes about the same time).
- Across all 12 combinations it tied chronological sampling at 20%: +0.03%
  [−0.41, +0.52].
- Sample plus fit took a median 23.1 s, against 19.3 s for chronological
  sampling at 20%.
- The time-ordered draw also spread its rows along each timeline. Whether that
  explains its edge was not tested.

| Rows kept | Residual, time-ordered draw | Chronological (equal-probability) | Uniform random (12-seed screen) |
|---|---|---|---|
| 10% | +0.58% | +0.99% | about +1.7% |
| 15% | +0.30% | +0.61% | about +1.2% |
| 20% | +0.17% | +0.39% | about +0.8% |
| 25% | +0.19% | +0.37% | about +0.7% |

- **The time-ordered draw beat chronological sampling on the stable
  combinations:** by 0.40%, 0.30%, 0.21% and 0.17% at 10%, 15%, 20% and 25% of
  rows. The 25% interval just reaches 0.
- **On the 3 noise-dominated combinations (all forward windows), the gap to
  full data means little.** Chronological samples and time-ordered residual
  samples both beat the full-data fit there, by 2-5%. Against chronological
  sampling the time-ordered draw's test RMSE was +1.2% at 10% (interval −0.4%
  to +2.7%), then −1.5%, −2.1% and −1.1% at 15%, 20% and 25%. So at 10% it
  only ties chronological sampling across all 12 combinations (−0.02%); at 20%
  it is 0.68% better.
- **Every draw had exactly the requested number of rows.**
- **Count the pilot's time.** Drawing the sample, pilot included, takes about
  2.7 s on 770k rows with one thread; a full-data fit there takes about 47 s.
  Matched for total time instead of rows, the edge over chronological sampling
  on the stable combinations was 0.1-0.25%.

Outside that panel, `benchmarks/bench_row_sampling.py` compared a residual
sample with a uniform sample of the same size over 10 seeds. These runs also
used the earlier draw, systematic along the input order (by group and time on
the synthetic panel), and have not been rerun with the random order. Three
datasets were used: a 400k-row synthetic panel, the CDC diabetes survey (250k
rows, binary classification) and California housing (20k rows).

| Fit on the sample | Smart vs uniform, 10% of rows | Smart vs uniform, 20% of rows |
|---|---|---|
| Gradient boosting, test loss | −0.7% to −2.3% | −0.6% to −2.0% |
| Linear models, test loss | about −0.05%; California −0.6% | about −0.04%; California +1.2% |
| `select_mrmr`, downstream score | −0.8% (synthetic) to +0.25% (CDC) | about 0 |

The pilot pays for itself only when the downstream fit is expensive. On all
three datasets, the pilot plus a 20% linear fit took longer than a linear fit
on every row (CDC: 0.29 s against 0.07 s), and on California boosting saved no
time either.

None of the selection differences was significant, but the selection
evidence is thin: weighted `select_mrmr` on the synthetic panel and CDC only,
with the downstream model trained on all rows. At 10% on CDC the interval
reaches +0.7%, so a small loss is not ruled out, and the panel benchmark above
fixed its features before sampling. The quality of
`StabilitySelector(use_smart_sampler=True)` has not been benchmarked. The one
cell where the residual sample did worse is a linear model on California
housing at 20%: 1.2% worse than a uniform sample, though not significantly
over 10 seeds. The residual sample landed at the full-data fit, while the
uniform samples beat the full-data fit by 1.2%. Dropping the 56 training rows
with an extreme feature value (|z| > 8) improves the full-data linear fit by
2.5%. Weighting aims the residual sample at the full-data fit, so it can
inherit that fit's weaknesses, but these runs did not establish why uniform
subsamples do better.

Filter selectors cap their input at `subsample=50_000` rows by default. They
reach the cap by thinning uniformly and keeping the weights, which leaves a
residual sample's weighting intact. Draw about 50k rows for selection, or pass
`subsample=None` to use the whole sample.

### What did not work

None of the following beat the pilot residual in that benchmark:

- smoothing the residual score (isotonic or gradient-boosted variance models);
- better or ensembled pilots, including the full-data model;
- leverage scores, covariate-shift weights or change-point scores;
- two-round resampling;
- keeping the hardest rows outright instead of sampling them.

Optimal-subsampling theory for observed labels scales a row's probability with
its residual, usually times a term that depends on its features and the model;
the residual method keeps only the residual part, an empirical choice. The MVS
score `sqrt(g**2 + mean(g**2))` (Ibragimov & Gusev, 2019), used instead of the
even share, was about 0.1% better on the stable combinations but 0.9% (12-seed
screen) to 2.2% (20-seed run) worse on the noise-dominated ones, so the
residual method does not use it.

### Cautions

- **Pass the weights to every downstream fit.** Unweighted, a model fitted on a
  residual sample over-represents the rows the pilot found hard.
- **Pin automatic hyperparameters.** If a learner derives its learning rate or
  leaf regularisation from the weights' effective sample size, set them to what
  an unweighted fit of the same size would use.
- **Judge noisy targets by test error.** On targets that are mostly noise, a
  sample can beat the full-data fit, so distance to the full fit is the wrong
  yardstick. The even share is what keeps the draw from chasing noise; 0.5 is
  the only benchmarked `uniform_mix`.
- **The returned weights are for fitting.** They are `1 / inclusion_prob`
  rescaled to mean one. What a weighted fit gains or loses against a
  full-data fit depends on the learner. To estimate a fixed full-data total,
  weight by the raw `1 / inclusion_prob`, which is unbiased because every row
  can be drawn.
- **External weights.** Do not pass external `sample_weight` with
  `StabilitySelector(use_smart_sampler=True)`; the sampler creates the weights
  for the rows it keeps.

### Chronological sampling at a fixed row budget

Use `sample_rows(method="chronological")`, or `chronological_sample` on a
DataFrame, when nearby observations for the same entity are often similar,
or to keep every season and entity represented. It allocates a proportional
share within each season and entity, then spreads the draws over the ordered
observations. It is the method to start with on panels, and the baseline a
residual draw has to beat.

```python
import numpy as np
import pandas as pd

from sift import sample_rows
from sift.sampling import chronological_sample

train_df = pd.DataFrame({
    "entity_id": np.repeat(np.arange(50), 40),
    "timestamp": np.tile(np.arange(40), 50),
})
train_df["season"] = train_df["timestamp"] // 10

positions = chronological_sample(
    train_df, len(train_df) // 10, group_col="entity_id",
    time_col="timestamp", stratum_col="season", random_state=42,
)
sampled = train_df.iloc[positions]

# The same draw through sample_rows; its weights are all 1.
s = sample_rows(size=0.1, method="chronological", groups=train_df["entity_id"],
                time=train_df["timestamp"], strata=train_df["season"],
                random_state=42)
assert (s.indices == positions).all() and (s.weights == 1).all()
```

Split off validation first. Omit `stratum_col` to allocate across whole
entity histories. Groups with fewer rows than the sampling interval
may receive zero slots. Every row has the same marginal inclusion
probability, so an equal-row training objective uses equal weights.

For stability selection on a chronological sample, name the feature subset
explicitly. The sampler needs the group and time columns to stay in the frame,
and `feature_names` keeps them (and the target) out of the candidate set. The
sampler draws one 30% chronological sample, and the bootstraps resample it.

```python
import numpy as np
import pandas as pd

from sift import StabilitySelector, panel_config

rng = np.random.default_rng(0)
feature_cols = [f"x{i}" for i in range(10)]
df = pd.DataFrame(rng.normal(size=(2_000, 10)), columns=feature_cols)
df["entity_id"] = np.repeat(np.arange(200), 10)
df["timestamp"] = np.tile(np.arange(10), 200)
y = 2.0 * df["x0"] - 1.5 * df["x1"] + rng.normal(scale=0.3, size=len(df))

selector = StabilitySelector(
    threshold=0.6,
    n_bootstrap=20,
    use_smart_sampler=True,
    sampler_config=panel_config("entity_id", "timestamp", size=0.3),
    random_state=0,
    verbose=False,
)

selector.fit(df, y, feature_names=feature_cols)
```

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

Classic caches (`ClassicFeatureCache`, built by `build_classic_cache`) follow
the same source-contract rule. A cache is not a resampling recipe: changing
rows, weights, or construction options requires rebuilding it.

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
| `min_feasible_q` | Necessary count bound `1/min(m)` over completed draws, not a sufficient discovery condition |
| `n_tested` | Minimum completed-draw tested count; `0` when `tested_state="not_run"` |
| `n_tested_per_draw` | Actual post-screening `m` per completed draw |
| `n_eligible` | Pre-screen discovery-unit count (not a completed tested count) |
| `tested_state` | `"post_screening"` after completed draws; `"not_run"` when no draw or pair-screen ran |
| `n_infeasible_draws` | Completed `offset=1` draws with `m·q < 1`; not a claim that the aggregate is empty |
| `n_discoveries_offset_0` | Reported discovery **features** the same `W` would select at `offset=0` (expanded members; not tested-group count) |

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

The default statistic is `relevance`, and 1.0 kept it (settled 2026-09-21)
after the seeded public-API comparison in the
[knockoff statistic bakeoff](knockoff-statistic-bakeoff.md), where `ridge`
reduced realized FDP but lost power on correlated designs. That study is not
a validity upgrade.

For `n_draws > 1`, SIFT samples multiple knockoff draws and selects features
whose selection frequency is at least `eta`. This improves run-to-run stability
but remains part of the approximate plug-in contract and reports
`fdr_control="none"` for the voted set.

Opt-in `aggregation="evalues"` (Ren and Barber 2022) instead forms per-draw
knockoff e-values `e_j = m · 1{W_j ≥ T_q} / (1 + #{W ≤ −T_q})` on the common
tested universe `T` with `m = |T|`, averages them, and applies e-BH at `q`.
A coordinate screened out of one draw stays in `T` with `e_j = 0` for that
draw and does not shrink that draw's denominator. Validated ungrouped mode
is `relevance` (`W = g(r)-g(r̃)`) and `ridge` (swap-invariant Gram solve);
it keeps `fdr_control="approximate_plugin"` and does not upgrade
Gaussian-copula plug-in exchangeability. `lsm`, `cefsplus`, grouped runs, and
a screening union that was not fixed before the statistics stay exploratory.
The recorded bound is the aggregate null expectation, not `E[e_j] ≤ 1` for
each null. `n_draws == 1` and `offset=0` are rejected with this aggregation so
legacy calls stay unchanged.

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

The CEFS+ statistic is tie-safe and pair-coupled. It is an exploratory
alternate statistic, slower than the default relevance statistic, and its
adaptive path is not covered by the approximate-plugin sign-flip claim. If selection count equals `selector_metadata["path_depth"]`, the cap
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
