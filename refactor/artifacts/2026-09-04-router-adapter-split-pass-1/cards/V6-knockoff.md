# V6 — extract the KnockoffSelectionResult adapter cluster

## Equivalence contract

- **Functions moved together:** `_knockoff_dropped_inputs` and
  `_as_knockoff_result`. Their only external cluster call is the exact
  `as_result` branch; no production or test code imports either name.
- **Facade helpers retained:** `_append_rows_like` remains defined in
  `view.py`, preserving its existing private import path and direct tests.
  `_selection_path_ranks`, `_coerce_position_series`,
  `_coerce_boolean_series`, and all shared identity/numeric helpers also stay.
  Do not remove any other function definition from `view.py`: after V6,
  `_append_rows_like`, `_strict_integer_vector`, and `_numeric_vector` have no
  in-module caller but remain required by siblings, and `_append_rows_like` is
  also imported at module scope by `tests/test_selection_view.py`.
- **Inputs covered:** complete/partial legacy results, explicit and absent raw
  identities, pre-screened/dropped columns, raw versus post-screening widths,
  optional W columns, malformed metadata/W frames, duplicate/out-of-range
  positions, empty selections, dtype widening/no-widening, and real
  `select_fdr` output.
- **Ordering/dtypes:** preserve stable raw-position sort, selected path ranks,
  optional-column order, nullable `Int64`/boolean widening only when rows are
  appended, unchanged NumPy dtypes when none are appended, and copy isolation.
- **Errors:** preserve all messages, causes, and frame depths; add no forwarding
  wrapper. Pin all three current strata from `sift.as_result`: 3 frames for an
  adapter-body error, 4 for a retained helper called directly by the adapter,
  and 5 for a retained helper reached through `_knockoff_dropped_inputs`.
- **Laziness:** retain the unconditional local import and exact type check for
  `KnockoffSelectionResult` in `as_result`; lazy-import the sibling only inside
  that exact branch. The sibling must not import `knockoff_filter` or the result
  class. Add only `view_knockoff` to the sibling-laziness guard.
- **Dependencies:** import from `view.py`: `SelectionView`,
  `_append_rows_like`, `_coerce_boolean_series`, `_coerce_feature_names`,
  `_coerce_indices`, `_coerce_position_series`, `_labels_equal`,
  `_selection_path_ranks`, `_strict_integer`, and
  `_validate_selected_identity`; plus `copy`, `Mapping`, `Any`, NumPy, pandas.
- **Compatibility:** preserve `_append_rows_like` in `view.py`; keep
  `KnockoffSelectionResult`, `SelectionView`, `as_result`, constructors,
  result-view method, pickle, public module identities, and 58-name spine.
- **Runtime provenance:** defer the one source-hash refresh until the complete
  split of all three originally oversized modules.

## Planned edit

- Add `sift/selection/view_knockoff.py` with verbatim args/bodies for the two
  cluster functions.
- Remove only those two definitions from `view.py`; retain every shared helper.
  Expected import deletion: none.
- Lazy-import `_as_knockoff_result` inside the existing exact branch.
- Extend sibling lazy-import and fresh-interpreter tests minimally.

## Verification

- [x] Both moved args/bodies AST-identical to base `2370631` (`ast.dump` of
  `_knockoff_dropped_inputs` and `_as_knockoff_result`).
- [x] `tests/test_selection_view.py`, `tests/test_knockoff_filter.py`,
  `tests/contracts/test_select_fdr_compat.py`, and
  `tests/contracts/test_public_spine.py` passed under warnings-as-errors
  (**353 passed**). Includes dropped/no-drop dtype contracts, real
  `select_fdr` views, and `_append_rows_like` tests.
- [x] Bare `import sift` / `import sift.selection.view` and unrelated dispatch
  leave `sift.selection.view_knockoff` unloaded; direct sibling/function
  import succeeds without a cycle and reports
  `__module__ == "sift.selection.view_knockoff"`. The sibling has no
  `knockoff_filter` / `KnockoffSelectionResult` import edge.
- [x] `_append_rows_like` remains defined in `view.py` and its private import
  tests pass. `_strict_integer_vector` and `_numeric_vector` remain. F401
  reported no facade import deletion.
- [x] Full suite: **1 failed, 1,996 passed, 40 skipped**. The only failure is
  `tests/test_runtime_scaling_benchmark.py::test_committed_runtime_evidence_and_documented_table_are_bound`.
  Provenance test was not edited, weakened, or skipped.
- [x] Ruff (default select and `--select F401,F841` on all view modules and
  the touched test) and `git diff --check` are clean.
