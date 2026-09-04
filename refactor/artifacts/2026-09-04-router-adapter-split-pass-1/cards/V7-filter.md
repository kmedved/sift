# V7 — extract the FilterSelectionResult adapter cluster

## Equivalence contract

- **Cluster moved:** `_normalize_filter_table`, `_CRITERION_DIRECTIONS`,
  `_normalize_auto_k_curve`, and `_as_filter_result`. The three functions have
  no external callers beyond their cluster and the exact `as_result` branch;
  the constant is used only by the curve normalizer.
- **Shared facade layer retained:** do not delete any `view.py` top-level
  definition. After V7, six helpers have no in-module caller but remain
  required by siblings: `_coerce_feature_names`, `_strict_integer_vector`,
  `_numeric_vector`, `_validate_selected_identity`, `_selection_path_ranks`,
  and `_append_rows_like`; the last is also imported at module scope by
  `tests/test_selection_view.py`.
- **Inputs covered:** full/partial filter rankings, explicit/implicit raw
  identities and widths, selected-position consistency, nullable ranking
  positions, classic/Gaussian/binary filters, fixed-k and every normalized
  auto-k curve availability mode, malformed producer payloads, diagnostics,
  and proxy-correlation storage.
- **Ordering/dtypes:** preserve stable selected-index sorting only for complete
  tables, path ranks, feature identity/token semantics, nullable positions,
  relevance handling, curve row order/dtypes, selection flags, and copies.
- **Errors:** preserve messages, causes, and all frame chains; add no forwarding
  wrapper. Pin the direct depth-3 and nested depth-5 paths plus both distinct
  depth-4 paths: an error from a retained helper must end in `view.py`, while
  an error from a moved normalization helper must end in `view_filter.py`.
- **Laziness/cycles:** retain the unconditional local import and exact type
  check for `FilterSelectionResult` in `as_result`; lazy-import the sibling
  only inside that branch. Keep the two imports inside `_as_filter_result`
  function-local. The sibling must not import `sift.selection.result` or
  `filter_auto_k` at module import time.
- **Dependencies:** from `view.py`, import `CURVE_COLUMNS`, `SelectionView`,
  `_coerce_boolean_series`, `_coerce_feature_names`, `_coerce_indices`,
  `_coerce_position_series`, `_labels_equal`, `_selection_path_ranks`, and
  `_validate_selected_identity`; plus `copy`, `Mapping`, `Any`, NumPy, pandas.
- **Private/public compatibility:** keep every retained helper at its historic
  module path. Preserve `FilterSelectionResult`, constructors, pickle,
  result-view method, `SelectionView`, `as_result`, public modules, and spine.
  Dataclass equality holds when `ranking_`/`diagnostics_` are `None`, but raises
  `ValueError` on populated DataFrame fields; preserve both behaviors and do
  not assert plain equality over a real producer result.
- **Runtime provenance:** defer the one source-hash refresh until the complete
  split of all three originally oversized modules.

## Planned edit

- Add `sift/selection/view_filter.py` with the constant and verbatim bodies of
  the three functions.
- Remove only those cluster definitions from `view.py`; expected import
  deletion is none.
- Lazy-import `_as_filter_result` inside the existing exact branch.
- Add only `view_filter` to the sibling-laziness guard and minimal direct
  fresh-interpreter probes.

## Verification

- [x] Three function args/bodies and `_CRITERION_DIRECTIONS` AST match base
  `2370631`. `_as_filter_result` still starts with the two function-local
  imports of `AUTO_K_CURVE_KEY` and `_PROXY_CORRELATIONS_ATTR`.
- [x] `tests/test_selection_view.py`, `tests/test_filter_results.py`,
  `tests/test_selection_view_proxies.py`, and
  `tests/contracts/test_public_spine.py` passed under warnings-as-errors
  (**303 passed**).
- [x] Bare `import sift` / `import sift.selection.view` leave
  `sift.selection.view_filter` unloaded. Direct sibling/function import
  succeeds without a cycle and reports
  `__module__ == "sift.selection.view_filter"`. The sibling has no
  module-level `filter_auto_k` or `result` import.
- [ ] Accepted test fix pending final Codex/Opus verification:
  `test_filter_result_dispatch_loads_only_its_adapter_sibling` dispatches
  `_full_filter_result(["a"], selected_indices=[0])` under a monkeypatched
  `builtins.__import__`. It forbids `sift.catboost_common` and the six
  unrelated siblings, allows `view_filter`, and asserts `features == ["a"]`
  plus `"sift.selection.view_filter" in sys.modules`.
- [x] `_coerce_feature_names`, `_strict_integer_vector`, `_numeric_vector`,
  `_validate_selected_identity`, `_selection_path_ranks`, and
  `_append_rows_like` remain in `view.py`. F401 reported no facade import
  deletion.
- [x] Full suite: **1 failed, 1,998 passed, 40 skipped**. The only failure is
  `tests/test_runtime_scaling_benchmark.py::test_committed_runtime_evidence_and_documented_table_are_bound`.
  Provenance test was not edited, weakened, or skipped.
- [x] Ruff (default select and `--select F401,F841` on all view modules and
  the touched test) and `git diff --check` are clean.
