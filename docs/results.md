# Reading Results

SIFT 0.9 introduces an additive `SelectionView` without replacing any legacy
result type. Existing functions still return the same lists, tuples, DataFrames,
or result classes by default. In the current A1 slice, normalized views support
`FilterSelectionResult` and `KnockoffSelectionResult` only.

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

The same five accessor lines work for the two adapters shipped in A1:

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

| Input to `sift.as_result` | A1 status | Raw table completeness | Curve | Transform |
| --- | --- | --- | --- | --- |
| `FilterSelectionResult` | supported | complete when the ranking covers every raw position; otherwise available rows only | empty standardized table | unavailable |
| `KnockoffSelectionResult` | supported | contains valid knockoff features; complete only when those positions cover the supplied raw identity | empty standardized table | unavailable |
| `BorutaResult` | planned | not implemented | not implemented | not implemented |
| `CatBoostSelectionResult` | planned | not implemented | not implemented | not implemented |
| `FeaturePathEvaluationResult` | planned | not implemented | not implemented | not implemented |
| fitted `StabilitySelector` | planned | not implemented | not implemented | not implemented |
| permutation-importance DataFrame | planned | not implemented | not implemented | not implemented |

Workstream A is therefore still in progress. Passing one of the five planned
types currently raises `TypeError`; it is not silently interpreted as another
result family. Passing an existing `SelectionView` is an identity operation.
Bare legacy list or tuple returns are also rejected with guidance to rerun the
selector with `return_result=True`.

## Tables and partial identity

Filter tables expose the available subset of `feature`, `selected_index`,
`path_rank`, `selected`, and `relevance`. Knockoff tables additionally map the
knockoff statistic `W` to `gain` and retain available relevance, selection
frequency, and feature-group columns. Tables are ordered by raw position when
the positions form a complete identity; `path_rank` preserves selection order.
Unavailable metric columns are omitted rather than synthesized.

Legacy result objects do not reliably record whether their original matrix was
a DataFrame or a positional ndarray. A result-only A1 view therefore reports
`metadata["input_kind"] == "unknown"`. Passing `input_features` establishes an
ordered raw identity and `raw_columns_hash`, but it does not rewrite that
historical provenance as known. `metadata["table_complete"]` says whether every
raw input position is represented by row-level information.

`selected_index` is the positional authority when labels repeat. The raw table
retains positions instead of collapsing duplicate labels. The A1 result-only
adapters do not enable name-based transforms or proxy lookup, so callers should
use `indices` and `support_` for positional work.

## Curves, serialization, and plotting

In A1, `curve` is an empty DataFrame with the stable columns `k`, `criterion`,
`criterion_se`, and `selected`. Normalization of the selector-specific Auto-K
curves belongs to the remaining Workstream A adapters.

`view.to_dict()` returns a JSON-safe payload. Both the top-level payload and its
metadata carry `schema_version="1"`; tables use pandas `orient="split"` form.
Consumers should ignore unknown keys so later schema additions remain
compatible.

`plot()` uses a stored curve when available and otherwise plots `gain` or
`relevance`; it raises `NotImplementedError` when a partial table has no plotted
metric. Matplotlib is imported only when `plot()` is called.

## Transform and proxy limitations

The two A1 adapters wrap result objects, not fitted selector state. They do not
retain encoders or the source matrix, so `transform()` and `inverse_transform()`
raise `NotImplementedError` and `metadata["transform_available"]` is false.

Likewise, A1 does not yet add selection-time `store_proxies` plumbing.
`proxies()` raises `NotImplementedError`; it never captures `X` implicitly.
Proxy-correlation storage, its explicit option, and its memory cap remain part
of the unfinished Workstream A scope.
