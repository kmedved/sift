# V3 — extract the Boruta result adapter

## Equivalence contract

- **Inputs covered:** `BorutaResult.result_view()` and
  `sift.as_result(BorutaResult, input_features=...)`, including accepted,
  tentative, and rejected statuses; hit/iteration bounds; explicit raw
  identity; malformed arrays; and duplicate labels.
- **Ordering preserved:** copy `_as_boruta_result` verbatim; raw table stays in
  feature order, selected features/indices stay in `flatnonzero` order, and
  path ranks retain accepted-feature order.
- **Tie-breaking:** none in the adapter.
- **Error semantics:** preserve exception types/messages/causes and add no
  forwarding wrapper frame.
- **Laziness:** retain the existing unconditional exact `BorutaResult`
  import/check in `as_result`. The core module `sift.boruta` is already loaded
  by `import sift` / package initialization; do **not** claim that
  `sift.boruta` stays out of `sys.modules`. Lazily import only
  `sift.selection.view_boruta` inside `if type(obj) is BorutaResult`. The
  sibling itself must not import `sift.boruta`. Unrelated dispatch and bare
  `import sift` / `import sift.selection.view` must leave `view_boruta`
  unloaded.
- **Short-circuit evaluation:** unchanged by verbatim extraction.
- **Floating-point:** preserve `_numeric_vector` validation and copies of mean
  importance and shadow thresholds exactly.
- **RNG/hash order:** no RNG or hashing in the adapter.
- **Observable side effects:** none; preserve array/diagnostic copy isolation.
- **Public API:** `BorutaResult`, `SelectionView`, `as_result`, and the 58-name
  public spine remain in their historical modules.
- **Internal imports:** `_as_boruta_result` moves only to
  `sift.selection.view_boruta` and reports that `__module__`; it is not
  re-exported from `sift.selection.view`. No repository caller imports or
  patches it from `view.py`.
- **Dependencies:** the sibling may import only `SelectionView`,
  `_coerce_feature_names`, `_labels_equal`, `_numeric_vector`,
  `_strict_integer`, and `_strict_integer_vector` from `view.py`, plus `Any`,
  NumPy, and pandas, with future annotations.
- **Serialization:** `BorutaResult` and `SelectionView` classes do not move;
  result dataclass fields and pickle identity remain unchanged.
- **Circular imports:** `view.py` does not bottom-import or eagerly bind
  `view_boruta`. Fresh-interpreter sibling/function import must succeed without
  a cycle.
- **Runtime provenance:** defer the single benchmark refresh until the full
  three-module split is committed.

## Planned edit

- Add `sift/selection/view_boruta.py` with the verbatim adapter.
- Remove that function from `view.py`.
- Lazy-import the moved adapter inside the exact BorutaResult branch.

## Verification

- [x] AST/body equality against the extracted `_as_boruta_result` body.
- [x] Boruta adapter/error/pickle tests pass under warnings-as-errors.
- [x] Bare `import sift` and unrelated dispatch leave the sibling unloaded;
  direct sibling import succeeds without a cycle and reports
  `__module__ == "sift.selection.view_boruta"`. The sibling itself has no
  `sift.boruta` import edge, although package initialization already loads the
  core `sift.boruta` module.
- [x] Public-spine checks remain green.
- [ ] Complete suite has only the known runtime-provenance failure.
- [x] Ruff including F401 and `git diff --check` are recorded with this slice.
