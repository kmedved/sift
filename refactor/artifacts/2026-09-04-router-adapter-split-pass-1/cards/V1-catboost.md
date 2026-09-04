# V1 — extract the CatBoost result adapter

## Equivalence contract

- **Inputs covered:** `CatBoostSelectionResult.result_view()` and
  `sift.as_result(CatBoostSelectionResult, input_features=...)`, including
  absent/complete raw identity, duplicate unobserved raw labels, malformed
  legacy fields, failed split scores, both metric directions, and target-k
  saturation.
- **Ordering preserved:** copy the existing adapter bodies without rewriting
  loops, sorting, table construction, or selected-feature iteration.
- **Tie-breaking:** preserve `min(..., key=(-score, k))` or `(score, k)` exactly.
- **Error semantics:** same exception types, messages, and causal chains; no
  forwarding wrapper frames.
- **Laziness:** `as_result` retains the exact CatBoost type guard
  (`__module__ == "sift.catboost_common"` and exact class identity) and only
  then lazily imports `_as_catboost_result` from `sift.selection.view_catboost`.
  Unrelated dispatch must not import `sift.catboost_common`,
  `sift.selection.view_catboost`, or the optional `catboost` package.
- **Short-circuit evaluation:** unchanged by verbatim extraction.
- **Floating-point:** preserve all float64 conversion, `isclose`, `mean`,
  population-standard-deviation, and standard-error operations in the same
  order.
- **RNG/hash order:** no RNG; label-token JSON ordering and feature-key
  insertion order remain unchanged.
- **Observable side effects:** none; warning/log behavior remains absent.
- **Public API:** `SelectionView`, `as_result`, the 58-name `sift.__all__`, and
  their `__module__` values stay `sift.selection.view`.
- **Internal imports:** moved `_catboost_*` helpers and `_as_catboost_result`
  live in `sift.selection.view_catboost` and report that `__module__`. They are
  **not** re-exported from `sift.selection.view`. `view_catboost.py` imports
  only `SelectionView`, `_label_token`, `_coerce_feature_names`, and
  `_strict_integer` from `view.py`, plus `json`, `math`, `Iterable`, `Mapping`,
  `Set`, `Real`, `Any`, NumPy, and pandas. It never imports
  `sift.catboost_common` or `catboost`.
- **Monkeypatch behavior:** no repository test patches a CatBoost adapter
  helper. Existing patches on other `view`, `filter_auto_k`, and `auto_k`
  names are untouched.
- **Serialization:** `CatBoostSelectionResult` and `SelectionView` classes do
  not move; legacy dataclass fields and pickle identity remain unchanged.
- **Circular imports:** `view.py` does not bottom-import or eagerly bind
  `view_catboost`. The CatBoost branch of `as_result` is the only import site.

## Planned edit

- Add one cohesive sibling, `sift/selection/view_catboost.py`, containing the
  CatBoost-only helpers and `_as_catboost_result`.
- Keep `SelectionView`, codec helpers, `as_result`, and all non-CatBoost
  adapters in `view.py`.
- In the CatBoost branch of `as_result`, after the existing lazy exact-type
  import/check, lazily import `_as_catboost_result` from
  `sift.selection.view_catboost` and call it.

## Verification

- [x] `tests/test_selection_view.py` CatBoost and optional-import cases pass
  under warnings-as-errors.
- [x] `tests/contracts/test_public_spine.py` keeps the exact 58-name surface.
- [x] A boundary regression pins four *separate* fresh interpreters:
  `import sift`; `import sift.selection.view`;
  `import sift.selection.view_catboost`; and
  `from sift.selection.view_catboost import _as_catboost_result`.
  `import sift` (and `import sift.selection.view`) must not load
  `sift.catboost_common` or `sift.selection.view_catboost`. Direct sibling
  imports must not load `sift.catboost_common` and must report adapter
  `__module__ == "sift.selection.view_catboost"`. Unrelated `as_result`
  dispatch does not import `sift.catboost_common` or
  `sift.selection.view_catboost`; public `__module__` values remain
  `sift.selection.view`.
- [ ] Do **not** rerun the 18-cell runtime benchmark after each small
  extraction. `tests/test_runtime_scaling_benchmark.py::test_committed_runtime_evidence_and_documented_table_are_bound`
  is expected to fail until the three-module split is complete: `view.py`
  changed and `view_catboost.py` is a new hashed source. Policy: finish and
  review the full split, commit the implementation, then rerun the runtime
  benchmark **once** from that clean commit and commit the refreshed
  CSV/sidecar/docs evidence. Until then, every full-suite check isolates
  exactly that one provenance failure; all other tests must pass. Do not
  weaken or skip the provenance test in code.
- [x] Ruff and `git diff --check` are recorded with the implementation slice.
- [x] `view.py` falls by about 460 lines; total source LOC may rise slightly
  for the sibling's module/import boilerplate.
