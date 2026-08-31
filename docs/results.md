# Reading Results

SIFT 0.9 introduces an additive `SelectionView` without replacing any legacy
result type. Existing functions still return the same lists, tuples, DataFrames,
or result classes by default. The current A2c slice supports
`FilterSelectionResult`, `KnockoffSelectionResult`, `BorutaResult`,
`FeaturePathEvaluationResult`, and `CatBoostSelectionResult`. It also adapts a
fitted `StabilitySelector` through its dynamic `result_view_` property.

## The common five accessors

Request the existing rich result first, then adapt it. Supplying
`input_features` gives the adapter the exact ordered raw-column identity when
the legacy result cannot reconstruct it completely.

```python
import sift

result = sift.select_mrmr(
    X,
    y,
    k=20,
    task="regression",
    return_result=True,
    verbose=False,
)
view = sift.as_result(result, input_features=X.columns)
```

The same five accessor lines work for all six adapters shipped through A2c:

```python
view.features
view.indices
view.k
view.table
view.metadata
```

`features` preserves the selector's legacy selected-feature order. `indices`
contains raw input positions when they are known, and `k` is the number of
selected features. `table` is a defensive-copy alias for `raw_table`.
`metadata` is also copied and includes `schema_version`, `table_complete`,
`transform_available`, and column-identity hashes.

## Current adapter coverage

| Input to `sift.as_result` | Status | Raw table completeness | Curve | Transform |
| --- | --- | --- | --- | --- |
| `FilterSelectionResult` | supported | complete when the ranking covers every raw position; otherwise available rows only | empty standardized table | unavailable |
| `KnockoffSelectionResult` | supported | contains valid knockoff features; complete only when those positions cover the supplied raw identity | empty standardized table | unavailable |
| `BorutaResult` | supported | complete from the result's full positional feature arrays | empty standardized table | unavailable |
| `CatBoostSelectionResult` | supported | complete with an explicit identity that uniquely resolves every known feature; otherwise known feature rows only | normalized direction-aware score curve | unavailable |
| `FeaturePathEvaluationResult` | supported | complete when `input_features` uniquely resolves the path; otherwise path rows only | normalized evaluation curve | unavailable |
| fitted `StabilitySelector` | supported | complete over the fitted candidate-feature namespace | empty standardized table | frozen fitted column subset; inverse unavailable |
| permutation-importance DataFrame | planned | not implemented | not implemented | not implemented |

Workstream A is therefore still in progress. Passing the remaining planned
permutation-importance family currently raises `TypeError`; it is not silently
interpreted as another result family. Passing an existing `SelectionView` is an identity operation.
Bare legacy list or tuple returns are also rejected with guidance to rerun the
selector with `return_result=True`.

## Tables and partial identity

Filter tables expose the available subset of `feature`, `selected_index`,
`path_rank`, `selected`, and `relevance`. Knockoff tables additionally map the
knockoff statistic `W` to `gain` and retain available relevance, selection
frequency, and feature-group columns. Tables are ordered by raw position when
the positions form a complete identity; `path_rank` preserves selection order.
Boruta tables retain original input order and map mean importance to `gain`,
with `hits` and `boruta_status` as diagnostics. Feature-path tables add
`feature_path_rank`; without explicit input identity they deliberately leave
raw positions unknown. CatBoost tables map the retained final-model importance
to `gain`, identify its source in metadata, and retain target-k stability and
first-split prefilter diagnostics without treating either as a raw column list.
Unavailable metric columns are omitted rather than synthesized.

Stability tables retain fitted candidate order, bootstrap selection frequency,
mean absolute coefficient, and that same coefficient magnitude as `gain`.
Actual `selected_features_` indices are the membership authority, so a
`max_features` cap is represented correctly even when additional rows exceed
the frequency threshold. An explicit DataFrame feature subset or smart-sampler
metadata exclusion means the fitted candidate namespace can be narrower than
the original DataFrame; the view records this as
`raw_namespace="fitted_candidate_features"` rather than claiming positions for
columns the selector did not fit.

Legacy result objects do not reliably record whether their original matrix was
a DataFrame or a positional ndarray. A result-only A2c view therefore reports
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
retains positions instead of collapsing duplicate labels. The five result-only
adapters do not enable name-based transforms or proxy lookup, so callers should
use `indices` and `support_` for positional work.

## Curves, serialization, and plotting

Filter, knockoff, and Boruta result views expose an empty DataFrame with the
stable columns `k`, `criterion`, `criterion_se`, and `selected` because those
legacy objects retain no route-level curve. Feature-path views normalize their
tested grid and lower-is-better score into those columns. Their producer stores
population fold SD, so `criterion_se` is `std / sqrt(n_finite - 1)` when every
fold is finite and at least two folds exist; otherwise it is missing. CatBoost
views preserve the raw metric and publish `criterion_direction` as `minimize` or
`maximize`. Their `selected` curve row is the returned target/parsimony count,
which can exceed the number of features that pass stability thresholding;
standard error is derived only when retained per-split scores establish the
finite-split count.

`view.to_dict()` returns a JSON-safe payload. Both the top-level payload and its
metadata carry `schema_version="1"`; tables use pandas `orient="split"` form.
Consumers should ignore unknown keys so later schema additions remain
compatible.

`plot()` uses a stored curve when available and otherwise plots `gain` or
`relevance`; it raises `NotImplementedError` when a partial table has no plotted
metric. Matplotlib is imported only when `plot()` is called.

## Transform and proxy limitations

The five result-only adapters wrap result objects, not fitted selector state.
They do not retain encoders or the source matrix, so `transform()` and
`inverse_transform()` raise `NotImplementedError` and
`metadata["transform_available"]` is false.

A fitted stability view is the exception: `transform()` uses a frozen copy of
only the fitted feature identity and selected positions. It preserves the
selector's ndarray/DataFrame validation and sklearn `set_output` configuration,
does not retain training rows or coefficient matrices, and does not change if
the original selector is refit or its threshold is changed later. Stability has
no inverse transform, so `inverse_transform()` remains unavailable.

Likewise, A2c does not yet add selection-time `store_proxies` plumbing.
`proxies()` raises `NotImplementedError`; it never captures `X` implicitly.
Proxy-correlation storage, its explicit option, and its memory cap remain part
of the unfinished Workstream A scope.
