# Reading Results

SIFT 0.9 introduces an additive `SelectionView` without replacing any legacy
result type. Existing functions still return the same lists, tuples, DataFrames,
or result classes by default. The [glossary](glossary.md) defines
[result view](glossary.md#result-view) and
[selection curve](glossary.md#selection-curve). The completed core adapter slice supports
`FilterSelectionResult`, `KnockoffSelectionResult`, `BorutaResult`,
`FeaturePathEvaluationResult`, `CatBoostSelectionResult`, and the opt-in
`ImportanceResult`. It also adapts a fitted `StabilitySelector` through its
dynamic `result_view_` property.

Cache-backed filters have their own additive entry point:
`select_cached(..., return_result=True)` returns a complete `SelectionView`
with selected positions, relevance, objective-path diagnostics, and cache
provenance. The four historical list/tuple return shapes remain unchanged when
the flag is omitted or false.

## The common five accessors

Request the existing rich result first, then adapt it. Supplying
`input_features` gives the adapter the exact ordered raw-column identity when
the legacy result cannot reconstruct it completely.

```python
import numpy as np
import pandas as pd

import sift

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"x{i}" for i in range(12)])
y = 2.0 * X["x0"] - 1.5 * X["x1"] + X["x2"] + rng.normal(scale=0.3, size=len(X))

result = sift.select_mrmr(
    X,
    y,
    k=6,
    task="regression",
    return_result=True,
    verbose=False,
)
view = sift.as_result(result, input_features=X.columns)
```

The same five accessor lines work for all seven entry-point families:

<!-- sift-doc: continues -->

```python
view.features
view.indices
view.k
view.table
view.metadata
```

`features` preserves the selector's legacy selected-feature order. `indices`
contains raw input positions when they are known, and `k` is the number of
selected features. Filter `feature_blocks` keeps that raw-width convention:
`view.k` and `metadata["n_columns_selected"]` are expanded column counts,
while selector metadata `k` / `n_blocks_selected` count additional blocks.
`n_blocks_selected_total` counts include blocks plus those additional blocks.
Every supplied `feature_blocks` map uses those block metadata units, including
identity (all-singleton) maps. Identity parity is the selected features,
ranking scores, and `refit_every` cadence, not no-block legacy
`metadata["k"]`, which still counts include columns in the total selected
width. `view.k` stays the raw expanded count in both cases.
With `cat_encoding="onehot"`, `features` / `indices` / `view.k` stay in the
raw category namespace while `metadata["n_encoded_columns_selected"]` and
sklearn `transform` / `get_feature_names_out` report dummy width.
`table` is a defensive-copy alias for `raw_table`.
`metadata` is also copied and includes `schema_version`, `table_complete`,
`transform_available`, and column-identity hashes.

## Current adapter coverage

| Input to `sift.as_result` | Status | Raw table completeness | Curve | Transform |
| --- | --- | --- | --- | --- |
| `FilterSelectionResult` | supported | complete when the ranking covers every raw position; otherwise available rows only | normalized auto-k route curve; empty standardized table for fixed-k | unavailable |
| `KnockoffSelectionResult` | supported | complete from `n_features_input` plus explicit `reason_dropped` rows; otherwise available rows only | empty standardized table | unavailable |
| `BorutaResult` | supported | complete from the result's full positional feature arrays | empty standardized table | unavailable |
| `CatBoostSelectionResult` | supported | complete with an explicit identity that uniquely resolves every known feature; otherwise known feature rows only | normalized direction-aware score curve | unavailable |
| `FeaturePathEvaluationResult` | supported | complete when `input_features` uniquely resolves the path; otherwise path rows only | normalized evaluation curve | unavailable |
| fitted `StabilitySelector` | supported | complete over the fitted candidate-feature namespace | empty standardized table | frozen fitted column subset; inverse unavailable |
| `ImportanceResult` from `permutation_importance(..., return_result=True)` | supported | complete in original feature-position order | empty standardized table | unavailable |
| `SelectionView` from `select_cached(..., return_result=True)` | supported directly | complete from the named cache contract | empty standardized table | unavailable |

The seven-family core adapter acceptance criterion is complete. The historical
permutation-importance DataFrame remains the default return and is deliberately
not guessed by `as_result`, because it lacks repeat-level positional
provenance; request `return_result=True` to obtain `ImportanceResult`. Passing
an existing `SelectionView` is an identity operation. Bare legacy list or tuple
returns are also rejected with guidance to request the corresponding result.

## Tables and partial identity

Filter tables expose the available subset of `feature`, `selected_index`,
`path_rank`, `selected`, and `relevance`. Automatic-k filter routes retain the
same complete ranking their fixed-k twins already produced, so an auto-k view
has one row per raw column rather than one row per selected feature.

Knockoff tables additionally map the knockoff statistic `W` to `gain` and retain
available relevance, selection frequency, and feature-group columns.
`select_fdr` records the raw input width as `n_features_input`, which is
distinct from `n_features` (the post-screening column count the filter actually
ran on). The view derives `support_` and the raw table width from
`n_features_input`, so both are available without passing `input_features`, and
every column the filter could not use gets an explicit `reason_dropped` row:
`"constant"` for a column removed while building the copula cache (it has no
`W` row at all) and `"zero_weight_variance"` for a cached column carrying no
weighted variance. Without an explicit identity those rows carry the raw
position and a null `feature`; passing `input_features` names them. Legacy
knockoff results that predate `n_features_input` keep the previous partial
behavior. Tables are ordered by raw position when
the positions form a complete identity; `path_rank` preserves selection order.
Boruta tables retain original input order and map mean importance to `gain`,
with `hits` and `boruta_status` as diagnostics. Feature-path tables add
`feature_path_rank`; without explicit input identity they deliberately leave
raw positions unknown. CatBoost tables map the retained final-model importance
to `gain`, identify its source in metadata, and retain target-k stability and
first-split prefilter diagnostics without treating either as a raw column list.
Importance tables retain raw position order, map the per-repeat population mean
to `gain`, and use importance order as `path_rank`. Every evaluated feature is
present with `selected=True`; metadata labels this `selection_semantics` as
`ranking_only`, so it is not mistaken for a thresholded subset. Unavailable
metric columns are omitted rather than synthesized.

A fitted stability view follows the selector's own `output_order`. `features`,
`indices`, the table's `path_rank`, and the frozen `transform` all use the order
that `get_feature_names_out()`, `get_support(indices=True)`, and `transform`
already use, and `metadata["output_order"]` records which one applied. With the
default `output_order="legacy"` that is descending selection frequency; with
`output_order="original"` it is ascending fitted position.

Stability tables retain fitted candidate order, bootstrap selection frequency,
mean absolute coefficient, and that same coefficient magnitude as `gain`.
Actual `selected_features_` indices are the membership authority, so a
`max_features` cap is represented correctly even when additional rows exceed
the frequency threshold. An explicit DataFrame feature subset or smart-sampler
metadata exclusion means the fitted candidate namespace can be narrower than
the original DataFrame; the view records this as
`raw_namespace="fitted_candidate_features"` rather than claiming positions for
columns the selector did not fit.

Legacy selection-result objects do not reliably record whether their original
matrix was a DataFrame or a positional ndarray. Those result-only views therefore report
`metadata["input_kind"] == "unknown"`. Passing `input_features` establishes an
ordered raw identity and `raw_columns_hash`, but it does not rewrite that
historical provenance as known. `metadata["table_complete"]` says whether every
raw input position is represented by row-level information.

New `StabilitySelector` fits record whether the fit input was a DataFrame or a
positional ndarray. The fitted view therefore reports `input_kind="dataframe"`
or `"positional"`; older pickles that predate that private provenance marker
fall back to `"unknown"` because names such as `x0` cannot distinguish a real
DataFrame label from an older generated positional label.

`selected_index` is the positional authority when labels repeat. The raw table
retains positions instead of collapsing duplicate labels. None of the six
result-only adapters enable a name-based `transform`, so callers should use
`indices` and `support_` for positional work. Proxy lookup is the one
selection-time capability that does survive adaptation: a Gaussian filter
result selected with `store_proxies=True` carries its correlation block through
`sift.as_result`, and `proxies_at` is the unambiguous accessor there when raw
labels repeat.

## Curves, serialization, and plotting

Automatic-k filter routes publish a normalized curve with exactly the columns
`k`, `criterion`, `criterion_se`, and `selected`. The producer builds it from
the route's own diagnostics and stores it in
`diagnostics_["auto_k_curve"]`, so adapters never guess which method-specific
diagnostic column is the criterion. `metadata["criterion"]` names the source
diagnostic column and `metadata["criterion_direction"]` is `higher_is_better`
or `lower_is_better`; `metadata["curve_route"]` records the route that ran,
which is the routed method rather than `"auto"` when the router chose for you.
`selected` marks the k the route actually returned, so a stop rule floored by
`min_k` stays truthful, and a route that returned zero features marks no row.

| Route | `criterion` | `criterion_se` | Direction |
| --- | --- | --- | --- |
| `evaluate`, `gaussian_cv`, `xfit_objective` | `score` | `score_se` | higher is better |
| `penalized_objective` (and `auto` when it routes there) | `penalized_score` | — | higher is better |
| `elbow` | `objective` | — | higher is better |
| `k_posterior` | `post` | — | higher is better |
| `perm_gap` | `gap` | `gap_se` | higher is better |
| `stability` | `phi` | `phi_se` | higher is better |
| `changepoint` | `log_scaled_gain` | — | higher is better |
| `chi2_stop` | `p_max` | — | lower is better |
| `forward_stop` | `Y_running_mean` | — | lower is better |
| `knockoff_path`, `consensus` | curve unavailable | — | — |

`knockoff_path` and `consensus` deliberately have no curve: their diagnostics
carry one row per candidate feature and knockoff draw, and one row per member
method's k vote, so neither is a k-indexed criterion path. Those views report
`curve_available=False` with an explicit
`metadata["curve_unavailable_reason"]` rather than a fabricated curve.

Fixed-k filter, knockoff, and Boruta result views expose an empty DataFrame with
the stable columns `k`, `criterion`, `criterion_se`, and `selected` because
those legacy objects retain no route-level curve. Feature-path views normalize their
tested grid and lower-is-better score into those columns. Their producer stores
population fold SD, so `criterion_se` is `std / sqrt(n_finite - 1)` when every
fold is finite and at least two folds exist; otherwise it is missing. CatBoost
views preserve the raw metric and publish `criterion_direction` as `minimize` or
`maximize`. Their `selected` curve row is the returned target/parsimony count,
which can exceed the number of features that pass stability thresholding;
standard error is derived only when retained per-split scores establish the
finite-split count. Note that the CatBoost and feature-path adapters predate
the auto-k curve and still spell `criterion_direction` as `minimize`/`maximize`;
auto-k filter routes use `lower_is_better`/`higher_is_better`. Read the value,
not one fixed pair of words, until the vocabularies are unified.

`ImportanceResult.importances_` is a defensive-copy matrix with raw feature
positions on rows and permutation repeats on columns. Its mean and population
standard deviation (`ddof=0`) reproduce the legacy summary exactly. The repeat
matrix is diagnostic data, not a feature-count curve.

`view.to_dict()` returns a JSON-safe payload. Both the top-level payload and its
metadata carry `schema_version="1"`; tables use pandas `orient="split"` form.
Consumers should ignore unknown keys so later schema additions remain
compatible.

Conversion is lossless rather than best-effort. `pd.NA`, `pd.NaT`, and
non-finite floats become `null`; dates, times, datetimes, and timedeltas become
ISO 8601 strings; dataclasses become their `dataclasses.asdict` form. An object
with no defined JSON representation raises `TypeError` instead of emitting a
`repr()` that would leak a memory address and could not be read back.

A mapping whose keys are all strings — including the payload root and normal
metadata — stays an ordinary JSON object. Only a mapping containing a non-string
key uses a tagged, order-preserving envelope, because plain string coercion
would silently merge keys such as `1` and `"1"`:

```json
{
  "__sift_mapping__": "typed_key_entries",
  "entries": [
    {"key": {"type": "builtins.int", "value": 1}, "value": "int"},
    {"key": {"type": "builtins.str", "value": "1"}, "value": "str"}
  ]
}
```

Each key token carries the key's concrete type plus its JSON-safe value, so
distinct keys stay distinct through `json.dumps`/`json.loads`. Both forms are
part of schema version `"1"`: a consumer that meets a mapping carrying the
`__sift_mapping__` tag must read `entries` instead of the object's own keys.

`plot()` uses a stored curve when available and otherwise plots `gain` or
`relevance`; it raises `NotImplementedError` when a partial table has no plotted
metric. Matplotlib is imported only when `plot()` is called.

## Transform and proxy storage

The six result-only adapters wrap result objects, not fitted selector state.
They do not retain encoders or the source matrix, so `transform()` and
`inverse_transform()` raise `NotImplementedError` and
`metadata["transform_available"]` is false.

A fitted stability view is the exception: `transform()` uses a frozen copy of
only the fitted feature identity and selected positions. It preserves the
selector's ndarray/DataFrame validation and sklearn `set_output` configuration,
does not retain training rows or coefficient matrices, and does not change if
the original selector is refit or its threshold is changed later. Stability has
no inverse transform, so `inverse_transform()` remains unavailable.

Proxy lookup is an explicit selection-time opt-in on cached selectors and
Gaussian filter routes. Pass `return_result=True, store_proxies=True` to
`select_cached`, to `select_cefsplus`, or to Gaussian `select_mrmr`,
`select_jmi`, and `select_jmim` calls. Binary CEFS+ supports the same option
only in Brier mode, which delegates to Gaussian CEFS+. Classic and binary
log-loss routes reject the option instead of silently ignoring it.

The view stores only the post-screening candidate-by-selected copula
correlation block as `float32`; it never retains `X` or a cache. Storage is
capped at 64 MiB and an oversized request fails rather than truncating the
block. `view.proxies(name, r_min=0.8)` returns unselected candidates above the
absolute-correlation threshold. When raw labels repeat, use
`view.proxies_at(selected_index, r_min=0.8)` for unambiguous positional access.
The proxy block is deliberately omitted from `to_dict()`; its presence, byte
count, and candidate count are recorded in metadata. Without the explicit
option, both proxy accessors raise with guidance to rerun selection.

`view.redundancy_report(r_min=0.8)` lists every qualifying edge across the
selected set, with raw positions next to labels so duplicate names stay
identifiable. `view.proxy_clusters(r_min=0.8)` groups selected features with
their qualifying stand-ins by connected components on that same block: a
candidate correlated with two selected anchors joins those anchors, and a
selected feature with no stand-ins is a singleton cluster. Signed
correlations are kept in the edge report; clustering uses absolute
correlation. Strictly monotone transforms of columns keep rank-Gaussian
clusters; a strictly decreasing transform of one column flips that edge's
signed correlation. Fitted `StabilitySelector(..., store_proxies=True)` views also
fill `cluster_frequency` as the fraction of completed resamples in which any
cluster member was selected. The `cluster_frequency` column is always
present; without a resample payload its values are nullable rather than
invented numbers. Resample indicators are omitted from `to_dict()`.
