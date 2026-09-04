# V2 — extract the permutation-importance result adapter

## Equivalence contract

- **Inputs covered:** `ImportanceResult.result_view()` and
  `sift.as_result(ImportanceResult, input_features=...)`, for DataFrame and
  positional identity, malformed ranking/snapshot fields, and repeated
  importance arrays.
- **Ordering preserved:** copy `_as_importance_result` verbatim; preserve
  `ranking_indices`, raw position reconstruction, table row order, and path
  ranks.
- **Tie-breaking:** unchanged; the adapter consumes the producer's ranking.
- **Error semantics:** preserve every exception type/message/cause and add no
  forwarding wrapper frame.
- **Laziness:** retain the existing unconditional exact `ImportanceResult`
  import/check and dispatch order in `as_result`. The core module
  `sift.importance` is imported for every `as_result` call; do not claim that
  unrelated dispatch avoids it. Lazily import `_as_importance_result` only
  inside `if type(obj) is ImportanceResult`. The sibling must not import
  `sift.importance`. Unrelated dispatch must not import
  `sift.selection.view_importance` (guard that name only; do not guard
  `sift.importance`).
- **Private source methods:** keep the duck-typed calls
  `result._adapter_snapshot()` and `result._matches_original_features(input_features)`
  exactly. The latter is identity-sensitive (`id()` based); do not rewrite it
  or its control flow.
- **Short-circuit evaluation:** unchanged by verbatim extraction.
- **Floating-point:** preserve float64 conversion, `mean`, population `std`,
  exact `allclose` settings, NaN behavior, and array copy points.
- **RNG/hash order:** no RNG or hashing in this adapter.
- **Observable side effects:** preserve deep-copy isolation of metadata,
  diagnostics, and repeat-level arrays.
- **Public API:** `SelectionView`, `as_result`, `ImportanceResult`, and the
  58-name `sift.__all__` surface remain in their historical modules.
- **Internal imports:** moved `_as_importance_result` lives only in
  `sift.selection.view_importance` and reports that `__module__`; it is not
  re-exported from `sift.selection.view`. There are no external callers or
  patchers of this private name.
- **Dependencies:** the sibling may import only `SelectionView`,
  `_coerce_feature_names`, `_coerce_indices`, `_labels_equal`,
  `_numeric_vector`, and `_strict_integer` from `view.py`, plus `copy`,
  `Mapping`, `Real`, `Any`, NumPy, and pandas. Keep
  `from __future__ import annotations`.
- **Serialization:** the `ImportanceResult` and `SelectionView` classes do not
  move; result snapshots and legacy pickle identity are unchanged.
- **Circular imports:** `view.py` does not bottom-import or eagerly bind
  `view_importance`. Fresh-interpreter sibling import must succeed without a
  cycle and without loading `sift.catboost_common`.
- **Runtime provenance:** do not rerun the 18-cell benchmark for this
  extraction. The sidecar hashes every `sift/**/*.py`; refresh it once after
  the full three-module split is committed.

## Planned edit

- Add `sift/selection/view_importance.py` with the verbatim adapter.
- Remove that function from `view.py`.
- Lazy-import the moved adapter inside the exact ImportanceResult branch.

## Verification

- [x] AST/body equality against the extracted `_as_importance_result` body.
- [x] Importance adapter and mutation-isolation tests pass under
  warnings-as-errors.
- [x] Fresh-process sibling import does not create a cycle, does not import
  `sift.catboost_common`, and reports
  `__module__ == "sift.selection.view_importance"`.
- [x] Public-spine checks stay current (no duplicated V1 public-module tests).
- [ ] Complete suite excluding only the known runtime-provenance binding
  remains green; provenance is refreshed once after the full split.
- [x] Ruff including F401 and `git diff --check` are recorded with this slice.
