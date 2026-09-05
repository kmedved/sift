# Glossary

SIFT-specific terms for the current public APIs. These are product meanings,
not generic Python vocabulary. Formal error-control claims and heuristic
methods are distinguished below. For signatures see the
[generated API reference](reference/index.md); for method math see the
[algorithm guide](ALGORITHMS.md).

## All-relevant

A selection contract that tries to keep every feature that carries signal,
including redundant members of the same family. [Boruta](#boruta) is
all-relevant. Contrast [fixed-k](#fixed-k) filters, which budget a path of
size `k`.

## Approximate plugin

The honesty label on SIFT knockoff FDR metadata. `select_fdr` and
`KnockoffSelector` report `fdr_control="approximate_plugin"` when the
Gaussian-copula knockoff model is a fitted plug-in, not a proven true Model-X
distribution. An empty selected set is a valid q-calibrated outcome. This is
not an exact finite-sample FDR theorem. For `n_draws > 1`, [`q`](#q) is a
per-draw target; the aggregated vote has no FDR guarantee and reports
`fdr_control="none"` (with analogous aggregation metadata).

## Auto-k

Asking SIFT to choose how many features to keep from an already-built
[feature path](#feature-path), via `k="auto"` and `AutoKConfig`. Auto-k
changes the stopping rule, not the underlying selector's objective. Fixed-k
function filters reject unused [row metadata](#row-metadata) `groups` and
`time` unless [`within`](#within) is set; on auto-k those arguments define
split construction, and with `within` they also fit fold-local demeaning.

## Between relevance

Entity-level association between a feature and the target after collapsing
rows to weighted group means. Exposed as `between_relevance` on filter
ranking tables when [`within`](#within) is set. It has only the observed
entity-level support, not independent row-level evidence; with two or fewer
positive-mass entities the summary is degenerate. Its magnitude need not be
comparable to `within_relevance`. Contrast [within](#within).

## Boruta

An all-relevant wrapper: compare each original feature with [shadow
features](#shadow-feature) under a fitted model, then confirm, reject, or
leave tentative. It is a heuristic, not an FDR procedure. The only importance
backends are `native` and `shap`. Permutation options control how shadows
are built; `sift.permutation_importance` is a separate post-fit ranking API.
`select_boruta` / `BorutaSelector` support native or SHAP
(`importance="native"` or `"shap"`). `select_boruta_shap` is the
function-style convenience for the same SHAP backend as
`BorutaSelector(importance="shap")` (CatBoost by default, or an explicit
compatible estimator plus the `shap` extra); it is not the same return or
API object as the fitted selector class.

## Conditional gain

The extra target information a candidate adds given features already on the
path. Gaussian CEFS+ uses a log-determinant / conditional-MI proxy; binary
CEFS+ uses a logistic score-test increment. Contrast pairwise
[redundancy](#redundancy) in classic mRMR. See the
[algorithm guide](ALGORITHMS.md).

## False discovery proportion

The realized fraction of selected features that are null on one run. FDR
procedures target the *expected* FDP. SIFT does not report a live FDP; `q`
is the target level, not a measured error rate on your sample.

## False discovery rate

The expected [FDP](#false-discovery-proportion). In SIFT, only the knockoff
filter (`select_fdr` / `KnockoffSelector`) is framed as q-calibrated
discovery, and only as an [approximate plugin](#approximate-plugin) under
the fitted knockoff model. Multi-draw aggregation (`n_draws > 1`) drops that
claim; see [approximate plugin](#approximate-plugin). Stability frequencies
and Boruta hits are not FDR control.

## Feature cache

`FeatureCache` from `build_cache`: a rank-Gaussian transform of a numeric
matrix, optional copula correlations, retained rows/weights, and name
provenance. `select_cached` and Gaussian routes reuse it. Weights belong on
`build_cache`; call-time `sample_weight` on `select_cached` is rejected.
Categorical and datetime feature columns are not cached; encode or convert
them first.

## Feature path

The ordered sequence of features a greedy selector adds, one at a time.
Fixed-k returns a prefix of length at most `k`. Auto-k walks the same path
and applies a [stopping rule](#stopping-rule). Path order is the default
filter [result view](#result-view) order unless `output_order="original"`.

## Fixed-k

Requesting at most `k` features. `k` is an upper bound: constants, `top_m`,
pruning, or non-finite scores can return fewer. Fixed-k filter functions and
classes reject `groups`/`time`. Knockoffs are sized by `q`, not `k`.

## Gaussian copula

The working model that treats rank-transformed features as jointly Gaussian.
SIFT's [rank-Gaussian transform](#rank-gaussian-transform) plus Gaussian MI
or knockoff sampling is a copula plug-in, not a claim that the raw data are
Gaussian.

## Groups

Row labels for entities, folds, or clusters. They are
[row metadata](#row-metadata), not candidate features. DataFrame sugar
`groups="column"` drops that column from X. On auto-k they define split
construction (`group_cv` and related strategies). On `StabilitySelector`,
`groups` and `time` together activate the block bootstrap; when automatic
alpha selection is used, groups alone can choose GroupKFold, while with a
fixed alpha they are validated but do not change the iid bootstrap. On
permutation they restrict shuffles; on `smart_sample` they are `group_col`.

## Inclusion weights

Approximate inverse-probability weights written by `smart_sample` into a
`sample_weight` column (mean-normalized). They reduce sampling bias; they
are not exact Horvitz–Thompson weights. Downstream selectors consume them as
ordinary [sample weights](#sample-weight).

## Joint mutual information

JMI scores a candidate by relevance plus complementary information with the
selected set; JMIM replaces the sum with a minimum and is more conservative.
These are filter-path heuristics, not FDR procedures.

## k

The requested path length or cap. See [fixed-k](#fixed-k) and
[auto-k](#auto-k). Stability wrappers use `k` as `max_features`. CatBoost
`k` is a requested count on a wrapper curve, with a separate parsimony rule
when `k` is omitted.

## Knockoff plus

The knockoff+ threshold (`offset=1`, the SIFT default): a feature enters only
if its [W](#w-statistic) clears a cutoff that yields either nothing or at
least about `1/q` discoveries. `offset=0` is knockoff (not knockoff+). Empty
sets are valid. Metadata reports `min_feasible_q = 1/min(m)` over completed
draws as a necessary count bound, not a sufficient discovery condition.
`n_tested` is that minimum post-screening count; `n_tested_per_draw` is
per-draw truth. `n_eligible` is the pre-screen discovery-unit count;
`tested_state="not_run"` means no draw or pair-screen ran. `m` is
post-screening and post-conditioning (group-level when grouped), not raw
`p`. `n_discoveries_offset_0` counts reported discovery **features** from
the same `W` at `offset=0` (expanded group/cluster members), not tested
groups. Included conditioning features are not discoveries. A knockoff+
warning is per infeasible draw (`m·q < 1`); it does not mean the aggregated
selection is empty. Existing `approximate_plugin` / heuristic FDR labels
are unchanged.

## Knockoffs

Synthetic covariates exchangeable with originals under a feature model.
SIFT samples second-order Gaussian-copula knockoffs and thresholds
antisymmetric [W](#w-statistic) statistics. See [Model-X](#model-x) and
[approximate plugin](#approximate-plugin). `sample_knockoffs` is a generator,
not a selector.

## Leakage

Using held-out, future, or target-derived information that the selection
contract forbids. Centered [`target_cv`](#target_cv) blocks unseen-in-fold
fold markers. Legacy supervised encoders (`loo`, `target`, `james_stein`,
`loo_logit`) are not cross-fitted. Function-style filters and all Boruta
entry points (including `BorutaSelector`) require
`allow_full_data_target_encoding=True` to run those modes; sklearn filter
selector classes and `KnockoffSelector` do not. `loo_logit` is
in-library; `loo` / `target` / `james_stein` additionally need
`category_encoders`. `KnockoffSelector` rejects `target_cv`; its
`loo_logit` path warns and sets `fdr_control="none"`.

## Model-X

Candès–Fan–Janson–Lv knockoffs: valid if knockoffs are generated from the
covariate distribution, without modeling Y given X. SIFT implements a
Gaussian-copula plug-in of that idea. The guarantee holds only to the extent
the fitted copula matches the true X law; metadata says
[approximate plugin](#approximate-plugin).

## Out-of-fold

A value computed on rows that were not used to fit the object that produced
it. [`target_cv`](#target_cv) emits OOF encodings during selection
(`fold_encoding - fold_training_prior`) and a target-blind map at inference.

## Permutation importance

Score drop when a column is shuffled, averaged over repeats, for an already
fitted model. SIFT adds group/time-aware shuffle schemes. It ranks; it does
not select a cutoff. Support for string or datetime columns is
model-dependent.

## q

The target FDR level for knockoff selection (for example `q=0.1`). It is not
a feature count and not a measured [FDP](#false-discovery-proportion). With
`n_draws > 1` it is a per-draw target; the aggregated set is not
q-calibrated. See [approximate plugin](#approximate-plugin).

## Rank-Gaussian transform

Map each numeric column to weighted mid-ranks, then to standard normal
quantiles, then weighted-standardize. Gaussian correlations of the result
are copula (rank) correlations of the raw columns. This is the cache
transform, not a Gaussianity claim on X.

## Redundancy

How much a candidate duplicates features already selected. Classic mRMR uses
mean pairwise redundancy; Gaussian paths and CEFS+ use conditional remaining
variance / [conditional gain](#conditional-gain).

## Relevance

Univariate association with the target (F, KS, RF, or Gaussian MI, depending
on the estimator). High relevance does not imply low [redundancy](#redundancy).

## Result view

`SelectionView` from `sift.as_result(...)`. Legacy result objects expose
`.result_view()`; a fitted `StabilitySelector` exposes `.result_view_`. One
accessor surface (`features`, `indices`, `support_`, `table`, `curve`,
`metadata`) over the legacy result types. It does not replace those types.
Incomplete adapters set `table_complete=False` rather than inventing rows.

## Row metadata

Per-row context that is not a candidate feature: [groups](#groups),
[time](#time), and [sample weights](#sample-weight). Acceptance differs by
entry point; see the [data-type support matrix](data-type-support.md).

## Sample weight

Non-negative finite row weights, normalized internally where the API
documents it. Zero-weight rows contribute no fitting or statistic mass where
weights are supported. Cache and knockoff preprocessing remove them before
retained-row and RNG work; shape-preserving encoders or transforms may still
emit an output row for every input row. On caches, pass weights to
`build_cache`. `smart_sample` *writes* a weight column; it does not take one
as input.

## Selection curve

The normalized k-indexed diagnostic `k`, `criterion`, `criterion_se`,
`selected` exposed by auto-k and feature-path [result views](#result-view)
and by `CatBoostSelectionResult` adapters. Plain fixed-k filter, knockoff,
Boruta, stability, and permutation results do not invent a curve. Auto-k
`knockoff_path` and `consensus` explicitly report why a curve is unavailable.

## Selection rule

How auto-k turns a score curve into a k (`best`, `one_se`, `plateau`,
`tolerance`, and method-specific stops). Distinct from the filter objective
that built the path. See [stopping rule](#stopping-rule).

## Shadow feature

A permuted copy of a real column, used by [Boruta](#boruta) as a noise
baseline. Beating shadows is an all-relevant heuristic, not FDR control.

## Smart sampling

`smart_sample`: row reduction by leverage and residual hard cases, with
optional group/time columns. It does not select features. See
[inclusion weights](#inclusion-weights).

## Stability frequency

The fraction of resamples in which a feature was selected. Thresholding
frequency is a robustness diagnostic, not Meinshausen–Bühlmann error control
and not knockoff FDR.

## Stability selection

Repeated sparse linear fits on resamples; keep features above a frequency
threshold. SIFT's implementation is a practical heuristic. It is not
`select_fdr`.

## Stopping rule

The rule that truncates a [feature path](#feature-path): a fixed `k`, an
auto-k [selection rule](#selection-rule), a knockoff `q` threshold, a
Boruta confirmation test, or a stability frequency cutoff. These are not
interchangeable contracts.

## Target encoding

Replacing a categorical column with a number derived from the target.
SIFT's leakage-aware option is [`target_cv`](#target_cv). Legacy supervised
modes (`loo`, `target`, `james_stein`, `loo_logit`) are not cross-fitted.
Function-style filters and all Boruta entry points (including
`BorutaSelector`) require `allow_full_data_target_encoding=True` to run
them; sklearn filter selector classes and `KnockoffSelector` do not.
`loo_logit` is in-library; `loo` / `target` / `james_stein` need
`category_encoders`. `KnockoffSelector` rejects `target_cv` and, on
`loo_logit`, warns and sets `fdr_control="none"`.

## target_cv

SIFT's cross-fitted, fold-centered target encoder. Training emits
`fold_encoding - fold_training_prior`; inference emits
`full_fit_encoding - full_training_prior`. Unseen categories map to a zero
centered effect. Centering removes unseen-in-fold fold markers; a level
seen twice in-fold can still carry sibling-row target information.

## Time

A chronological key for holdouts, time-aware folds, block shuffles, or
`smart_sample`'s `time_col`. It is [row metadata](#row-metadata). On auto-k,
`time` defines split construction (for example a time holdout), not merely
split size. On `StabilitySelector` it activates the block bootstrap only
together with `groups`; with automatic alpha it can choose TimeSeriesSplit,
and with a fixed alpha it is validated without changing the iid bootstrap.
Datetime *feature* columns are rejected by selectors; convert them to
numeric features if they belong in X.

## W statistic

The antisymmetric knockoff score `W_j = f(Z_j, Z_j')` used to threshold
discoveries. Large positive W favors the original over its knockoff. SIFT's
default statistic is `relevance`; that default is not an exact-FDR upgrade.

## Within

Optional regression-filter panel transform (`within="groups"` or
`"two_way"`). Weighted entity means, and for two-way a fixed five-iteration
alternation with time means, are subtracted from `X` and `y` before ranks.
Validation folds fit those means on training rows only; unseen entities fall
back to the training grand mean. Demeaning can remove all variation,
including singleton-only groups, and then the selection is empty or the
call raises that no within-entity signal remains. Ranking tables then
include `within_relevance` (the selector relevance on the demeaned data)
and [`between_relevance`](#between-relevance). Sklearn `transform` still
returns selected raw columns.
