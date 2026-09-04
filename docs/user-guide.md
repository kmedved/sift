# Tutorial

This page walks one selection job from a first pass to a decision. The outcome
is a compact, inspectable feature set for a small regression panel: recover the
planted signals, then decide whether to change `k`, add leakage-safe encoding,
switch validation, or stop. It is not a catalog of every SIFT entry point.

Exact parameters live in the [generated API reference](reference/index.md).
Error messages live in [troubleshooting](troubleshooting.md). The
[glossary](glossary.md) defines [fixed-k](glossary.md#fixed-k),
[auto-k](glossary.md#auto-k), and the other contracts used below. The full
[API manual](https://github.com/kmedved/sift/blob/main/DOCS.MD) remains the
complete configuration source.

## 1. State the outcome and create the example

You will start with 200 rows and 12 numeric columns whose target depends on
`x0`, `x1`, and `x2`. Keep this fixture small and seeded so later comparisons
are about the selector contract, not a new dataset.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))
```

Later sections recreate a panel of this size so you can copy one step. The
planted signals stay `x0`, `x1`, and `x2` unless a branch says otherwise.

## 2. Choose the selection contract

Do not pick a method by familiarity. Follow the canonical
[selector decision tree](choosing-a-selector.md) first. It chooses by output
contract, then treats sklearn wrappers, automatic feature counts, caches, and
row sampling as workflow modifiers.

This walkthrough takes the fast ordered filter path because the job is a
compact ranking for a continuous target. An all-relevant Boruta need leaves
this path at that tree. Stability frequencies, q-calibrated knockoffs, and
CatBoost ranking appear later as explicit branches; do not convert those
examples into a second choice table.

## 3. Run a simple fixed-k first pass

```python
import numpy as np
import pandas as pd

from sift import select_cefsplus, select_jmi, select_jmim, select_mrmr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))

mrmr = select_mrmr(X, y, k=6, task="regression", verbose=False)
jmi = select_jmi(X, y, k=6, task="regression", verbose=False)
jmim = select_jmim(X, y, k=6, task="regression", verbose=False)
cefs = select_cefsplus(X, y, k=6, verbose=False)
```

You should see the three planted signals among the six names. If they are
missing, inspect the data contract before changing `k` or the selector.

Fixed `k` is an upper bound. Selectors can return fewer features when constant
features, invalid scores, `top_m`, or pruning remove candidates.
For fixed-k filter calls, `groups` and `time` are rejected because they only
define auto-k evaluation splits; use `k="auto"` with a matching strategy or
omit those arguments. `KnockoffSelector` rejects row `groups` and `time` too.
Sklearn-style selector classes accept dense arrays and DataFrames; sparse
matrices are rejected during fit, transform, and inverse transform. The
observed public behavior for those input kinds, categoricals, datetime
columns, weights, and group/time metadata is the
[data-type support matrix](data-type-support.md).

## 4. Add categoricals and sample weights

Function-style selectors default to `cat_encoding="none"` and support
`cat_features` and explicit encodings. Use `cat_encoding="target_cv"` for the
built-in leakage-safe regression/binary path; it uses cross-fitted training
values and needs no optional dependency.

`target_cv` emits **centered category effects**, not raw category means: each
value is the category estimate minus the training prior that produced it. An
unknown or unseen category therefore maps to a zero centered effect (the
global-mean estimate before centering). That is what makes the path safe for
high-cardinality columns: a unique ID, a group proxy, or a timestamp proxy is
never present in its own fold's training rows, so it emits a constant zero and
carries no relevance instead of encoding a fold-identifying prior.

**Know the boundary of that guarantee.** Centering neutralizes only
*unseen-in-fold* emissions. It removes the fold marker; it is not a defence
against high cardinality as such. A level that appears two or more times in a
fold's training rows still transmits those sibling rows' targets — ordinary
target-encoding behavior — so a *near*-unique identifier stays selectable when
its rows share a latent target. On a 300-identifier fixture with two rows each,
`corr(enc(id), y)` is about 0.88 and `select_mrmr(k=2)` picks `id` first. That
is genuine cross-row information rather than leakage, so SIFT does not remove
it. If it must not reach selection, drop ID-like columns, or pass `groups=` so
all of an identifier's rows land in the same fold — under `groups=` the same
column encodes to exactly zero.

Selector classes retain the full-training encoder for target-blind inference,
while `fit_transform` returns the cross-fitted training columns used for
selection. Weighted calls use SIFT's weighted m-estimate folds. The default
`target_cv_smoothing="auto"` uses the unweighted empirical-Bayes formula with
every count replaced by weighted row mass, so weight `m` and `m` duplicated
rows encode identically. An explicit float remains available when you want to
fix the shrinkage:

```python
import numpy as np
import pandas as pd

from sift import select_mrmr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"x{i}" for i in range(8)])
X["league"] = rng.choice(["nba", "wnba", "gleague"], size=len(X))
league_effect = X["league"].map({"nba": 2.0, "wnba": 0.0, "gleague": -2.0})
y = league_effect + 1.5 * X["x0"] + rng.normal(scale=0.3, size=len(X))
weights = np.ones(len(X))
weights[-50:] = 2.0  # emphasize the most recent rows

selected = select_mrmr(
    X,
    y,
    k=4,
    task="regression",
    cat_features=["league"],
    cat_encoding="target_cv",
    target_cv_smoothing=20.0,  # or leave it at the default "auto"
    sample_weight=weights,
    verbose=False,
)
```

`league` should appear beside `x0`. If a high-cardinality identifier outranks
both, drop it or pass `groups=` rather than turning off centering.

Existing `"target"`, `"loo"`, `"james_stein"`, and `"loo_logit"` function
encodings remain guarded against full-data target leakage; opt in only when
leakage is handled outside SIFT. `allow_full_data_target_encoding=True` is
rejected with `target_cv`, which is cross-fitted by construction, and
`KnockoffSelector` rejects `target_cv` entirely because target-derived
preprocessing breaks the Model-X FDR claim. CatBoost selectors handle
categorical features natively.

Grouped and time-aware encoding is available on auto-k evaluate routes, not on
this fixed-k call. Set `target_cv_n_splits` independently of the outer auto-k
fold count. Group folds exclude whole groups; time folds keep tied timestamps
together and use only strictly earlier values. Earliest time rows emit a
centered neutral effect (zero) when you supply an explicit target-independent
`target_prior`, or receive zero effective selection weight under
`warmup_policy="zero_weight"` (default) or `"exclude"`. Fixed-k calls continue
to reject `groups`/`time`, and multiclass target encoding remains blocked on
block-aware selection.

### When the target is binary

Use binary CEFS+ when the target is Bernoulli-like and logistic conditional
information is a better fit than a Gaussian target approximation.
`sample_weight` and `class_weight` are honored directly by `loss="logloss"`.
`cat_encoding="target_cv"` is the leakage-safe default choice here; the
binary-only `cat_encoding="loo_logit"` is a full-data supervised encoder and so
needs `allow_full_data_target_encoding=True` before a function-style selector
will run it.

```python
import numpy as np
import pandas as pd

from sift import select_cefsplus_binary

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
X["position"] = rng.choice(["guard", "wing", "big"], size=len(X))
score = 2.0 * X["x0"] - 1.5 * X["x1"] + rng.normal(scale=0.3, size=len(X))
y_binary = (score > score.median()).astype(int)

selected = select_cefsplus_binary(
    X,
    y_binary,
    k=6,
    loss="logloss",
    class_weight="balanced",
    cat_features=["position"],
    cat_encoding="target_cv",
    verbose=False,
)
```

Expect `x0` and `x1` in the selected set. Treat `position` as a real candidate
only when it carries target information beyond the numeric signals.

## 5. Let SIFT choose k, then change validation

Function-style selectors use a prefix-only contract for auto-k: SIFT builds one
selection path, then evaluates prefixes. Sklearn-style selector classes can use
nested evaluation where supported.

For a first pass with CEFS+, use `select_cefsplus(X, y, k="auto")`. Zero-config
CEFS+ auto-k uses the measured Auto-K v2 router; it routes to the measured EBIC
default and records the route under
`diagnostics_["auto_k"]["auto_routing"]`. Use `gaussian_cv` when you
specifically want fold scoring, `chi2_stop` or `forward_stop` when you need a
calibrated no-signal stop, and `perm_gap` when groups/time/weights make
analytic nulls suspicious. `knockoff_path` can be chosen for an approximate
plug-in q-calibrated returned set if you accept that it did not pass the
automatic-sizing gate. `changepoint`, `stability`, and `xfit_objective` remain
experimental or failed-gate for automatic sizing.

```python
import numpy as np
import pandas as pd

from sift import select_cefsplus

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))

result = select_cefsplus(X, y, k="auto", return_result=True, verbose=False)
selected = result.selected_features
```

`selected` is a prefix of the CEFS+ path, not a new objective. If it is empty
or far longer than the three planted signals, read `result.selector_metadata`
before forcing `min_k`/`max_k`.

Auto-k support depends on the selector route:

| Route | Supported `k_method` values |
| --- | --- |
| Classic mRMR/JMI/JMIM | `evaluate` |
| Gaussian mRMR/JMI/JMIM | `auto`, `evaluate`, `elbow`, `gaussian_cv`, `xfit_objective`, `stability` |
| CEFS+ | `auto`, `evaluate`, `elbow`, `penalized_objective`, `k_posterior`, `chi2_stop`, `forward_stop`, `changepoint`, `perm_gap`, `knockoff_path`, `gaussian_cv`, `xfit_objective`, `stability`, `consensus` |
| Binary CEFS+ | `auto`, `evaluate`, `elbow`, `penalized_objective`, `k_posterior`, `changepoint` |

Unsupported modes fail before SIFT builds caches or feature paths, which keeps
configuration errors cheap to catch.

Passing `groups` or `time` does not change what the filter path optimizes. On
`k="auto"` they define evaluation splits: group folds hold out whole groups,
and time splits score prefixes on later rows. That is validation, not a
different selector. The next example keeps the same CEFS+ path and only changes
how prefixes are scored:

```python
import numpy as np
import pandas as pd

from sift import AutoKConfig, select_cefsplus

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))

config = AutoKConfig(
    k_method="evaluate",  # or "auto", "gaussian_cv", "chi2_stop", etc.
    strategy="time_holdout",
    min_k=2,
    max_k=10,
)

timestamps = np.arange(len(X))  # replace with the real chronological key

result = select_cefsplus(
    X,
    y,
    k="auto",
    time=timestamps,
    auto_k_config=config,
    return_result=True,
    verbose=False,
)
selected = result.selected_features
```

If the time-aware prefix disagrees with the iid auto-k set, trust the split
that matches how you will score the downstream model. Do not treat the
disagreement as a reason to switch from CEFS+ to mRMR. The next section
inspects this `result`.

## 6. Inspect the result before changing course

Many selectors can return richer metadata through `return_result=True` or
selector-specific diagnostics. `sift.as_result(...)` provides an additive
common view for `FilterSelectionResult`, `KnockoffSelectionResult`,
`BorutaResult`, `FeaturePathEvaluationResult`, and `CatBoostSelectionResult`,
plus fitted `StabilitySelector` and the opt-in `ImportanceResult` from
`permutation_importance`. Legacy result types and default returns are unchanged.

The common access pattern is:

<!-- sift-doc: continues -->

```python
import sift

view = sift.as_result(result, input_features=X.columns)

view.features
view.indices
view.k
view.table
view.metadata
view.diagnostics
```

The minimum diagnostics that should change the next decision are:

- `view.k` and `view.features`: is the returned set empty, saturated at
  `max_k`, or missing the signals you already know are real?
- `view.table`: ranks, scores, or reasons for the candidate columns, not just
  the selected names.
- `view.diagnostics["auto_k"]["auto_routing"]` (equivalently the legacy result
  diagnostics) for the auto-k route. It is not stored on `view.metadata`.
- `view.metadata` for encoding choices and completeness flags such as
  `table_complete` and `transform_available`.

These result-only views do not retain fitted preprocessing state, so transform
and proxy operations are unavailable. See [Reading Results](results.md) for the
current adapter-completeness matrix and serialization contract, and
[DOCS.MD](https://github.com/kmedved/sift/blob/main/DOCS.MD) for
selector-specific diagnostics.

Sklearn-style selector classes always keep their transform contract stable; pass
inspection options to the function-style selectors when you need full result
objects.

The `continues` block above inspects the time-aware auto-k `result` from the
previous example. That is the only reason it shares a namespace; copy both
blocks together.

## 7. Add a robustness or trust branch

Stability selection and knockoffs answer different questions. Stability asks
how often a feature is selected under resampling. Knockoffs ask how many
discoveries you can trust at a target q. Neither replaces the other's
contract, and neither is the next default step after a clean first pass.

### When you need resampling frequencies

```python
import numpy as np
import pandas as pd

from sift import StabilitySelector

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))

selector = StabilitySelector(
    task="regression",
    n_bootstrap=50,
    threshold=0.6,
    random_state=0,
    verbose=False,
)
selector.fit(X, y)
stable_features = selector.selected_feature_names_
X_stable = selector.transform(X)
X_restored = selector.inverse_transform(X_stable)
```

Features below `threshold` stay out even when a single full-data fit would keep
them. Compare `stable_features` with the fixed-k names; disagreement is the
diagnostic.

Pass both `groups` and `time` to use block bootstrap for ordered panel data.
With DataFrames, `groups="column"` and `time="column"` extract and exclude the
metadata columns; direct arrays remain positional. With automatic alpha, groups
alone can choose GroupKFold and time alone TimeSeriesSplit; with fixed alpha
they are validated but do not change iid bootstrap.
`penalty` is an alias for `alpha`, and both may be supplied only when equal.
Both spellings are permanent: neither is deprecated and neither warns.
Threshold tuning accepts sklearn scorer objects as well as scorer names.
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

A fitted stability selector supplies the same accessors and a frozen transform:

<!-- sift-doc: continues -->

```python
view = selector.result_view_
X_stable = view.transform(X)
```

Its table covers the selector's fitted candidate features; `view.indices`
keeps the existing integer positions and `view.features` supplies names.
The fitted selector itself supports dense `inverse_transform`; the frozen
`SelectionView` intentionally does not retain the fitted preprocessing state
needed for inversion.

### When you need a q-calibrated trusted set

```python
import numpy as np
import pandas as pd

from sift import select_fdr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(300, 20)), columns=[f"x{i}" for i in range(20)])
y = X.iloc[:, :12].sum(axis=1) + rng.normal(size=len(X))

result = select_fdr(X, y, q=0.1, n_draws=1, random_state=0, verbose=False)
selected = result.selected_features
ranking = result.get_feature_ranking()
```

An empty `selected` list is a valid q-threshold answer, not a failed run. Auto-k
asks how many features help prediction; knockoffs ask how many discoveries you
can trust at a target q. Compare both diagnostics rather than converting one
into the other.

The v1 implementation samples second-order Gaussian knockoffs in the
rank-Gaussian copula space already used by `FeatureCache`, then applies the
knockoff+ threshold to an antisymmetric feature statistic.

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
is at least `eta`. This is useful for run-to-run stability, but the aggregated
vote is not an FDR guarantee: q remains per-draw, and the reported
`fdr_control` becomes `"none"`. An empty result is a valid answer: no feature
survived the requested q threshold. `select_fdr` requires a finite numeric
target; continuous targets and numeric binary labels are supported.
Integer-valued multiclass targets trigger a warning because this routine treats
`y` as numeric; string/categorical labels are not accepted by `select_fdr`, so
encode a one-vs-rest target numerically for categorical multiclass tasks.

## 8. Reuse work, then consider specialized row context

These steps are optional. Skip them unless the job actually has a shared
numeric matrix, a CatBoost final model, or an already-fitted estimator.

### When many selectors share one numeric matrix

```python
import numpy as np
import pandas as pd

from sift import build_cache, select_cached

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y1 = X.iloc[:, :6].sum(axis=1) + rng.normal(scale=0.3, size=len(X))
y2 = X.iloc[:, 6:].sum(axis=1) + rng.normal(scale=0.3, size=len(X))

cache = build_cache(X, subsample=None, compute_Rxx=True)
mrmr = select_cached(cache, y1, k=6, method="mrmr_quot")
cefs = select_cached(cache, y2, k=6, method="cefsplus")
cefs_view = select_cached(
    cache, y2, k=6, method="cefsplus", return_result=True
)
```

The two cached selections should differ because `y1` and `y2` use disjoint
signal blocks. If they do not, the cache is not the matrix you think it is.

A prebuilt cache is tied to the input row count and feature contract: named
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

Measured cost context for large `n`/`p`, including when a cache pays off, is
the [runtime and scaling guide](runtime-scaling.md).

Leaving `random_state=None` on StabilitySelector, permutation importance, or
CatBoost emits a `FutureWarning`: 0.9 remains nondeterministic, while 1.0 will
default to seed 0. Their existing `n_jobs=-1` defaults are also unchanged in
0.9.

### When selection should follow a CatBoost model

This is not a required next step after CEFS+. Use it when the final model is
CatBoost and the optional extra is acceptable.

<!-- sift-doc: requires=catboost -->

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import sift

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + rng.normal(scale=0.3, size=len(X))
group_ids = np.repeat(np.arange(20), 10)
dates = np.tile(np.arange(10), 20)
weights = np.ones(len(X))

result = sift.catboost_select(
    X,
    y,
    k=5,
    min_features=2,
    n_estimators=50,
    groups=group_ids,
    time=dates,
    sample_weight=weights,
    cv=GroupKFold(n_splits=3),
    random_state=0,
    verbose=False,
)
```

CatBoost accepts direct positional `groups`, `time`, and `sample_weight`
arrays. DataFrame callers may instead use `groups="group_column"` or
`time="date_column"`; `group_col` and `sample_weight_col` are permanent
aliases, not deprecated spellings, and neither warns.
A direct value and its alias cannot be combined. Supplied time values
must be non-missing and mutually orderable and stably order aligned rows before
the configured splitter. The example pairs `groups` with `GroupKFold`; a
time-only splitter such as `TimeSeriesSplit` ignores `groups` (scikit-learn
warns), so pass `time=` alone when chronological validation is required. The
default splitter remains random.

### When you already have a fitted estimator

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from sift import permutation_importance

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 10)), columns=[f"x{i}" for i in range(10)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + rng.normal(scale=0.3, size=len(X))
group_ids = np.repeat(np.arange(20), 10)
dates = np.tile(np.arange(10), 20)
fitted_model = Ridge().fit(X, y)

importance = permutation_importance(
    fitted_model,
    X,
    y,
    groups=group_ids,
    time=dates,
    permute_method="auto",
    scoring="neg_rmse",
    n_repeats=10,
    random_state=0,
)

rich_importance = permutation_importance(
    fitted_model,
    X,
    y,
    groups=group_ids,
    time=dates,
    n_repeats=10,
    return_result=True,
    random_state=0,
)
repeat_drops = rich_importance.importances_
view = rich_importance.result_view()
```

With `time` but no `groups`, SIFT treats the dataset as one ordered group for
time-aware permutations. The historical DataFrame remains the default;
`return_result=True` adds the repeat-level matrix and a complete ranking view
without applying an arbitrary selection threshold. Rank on held-out rows when
the question is out-of-sample behavior of that fitted model, not a new SIFT
filter path.

## 9. Checklist and next steps

Before leaving this page:

1. You have a seeded panel and a first-pass selected set.
2. The method matches the [selector decision tree](choosing-a-selector.md)
   contract you actually need.
3. Categorical columns use `target_cv` unless leakage is handled outside SIFT.
4. `k="auto"` was an explicit choice, and `groups`/`time` were used only where
   they change splits or resampling rather than a fixed-k filter.
5. You inspected `view.features`, `view.k`, `view.table`, `view.metadata`,
   and `view.diagnostics` before changing the selector.
6. Stability frequencies and knockoff q were added only when those are the
   questions, not as a default second opinion.

Then read the canonical pages instead of enlarging this walkthrough:

- [Generated API reference](reference/index.md) for signatures and defaults.
- [Reading Results](results.md) for adapter coverage and serialization.
- [Data-type support matrix](data-type-support.md) for ndarray, DataFrame,
  categoricals, sparse input, datetime columns, weights, groups, and time.
- [Runtime and scaling](runtime-scaling.md) for measured cost and when a cache
  is worth building.
- [Troubleshooting](troubleshooting.md) for validation errors.
- [Glossary](glossary.md) for k, q, FDR, `target_cv`, and related terms.
- [Full API manual](https://github.com/kmedved/sift/blob/main/DOCS.MD) for
  complete configuration, longer examples, and selector-specific diagnostics.
- [Advanced workflows](ADVANCED.md) for sampling, panels, and other modifiers
  that are not part of this first path.
