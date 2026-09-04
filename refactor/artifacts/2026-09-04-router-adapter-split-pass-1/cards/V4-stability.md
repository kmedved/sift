# V4 — extract the fitted StabilitySelector adapter

## Equivalence contract

- **Inputs covered:** `StabilitySelector.result_view_` for named/positional
  fits, explicit identities, fitted/failed-refit state, both output orders,
  threshold/max-feature selection, legacy provenance, coefficient diagnostics,
  tuple/same-repr labels, set-output transforms, clone, and pickle.
- **Ordering preserved:** copy `_as_stability_selector` verbatim; preserve
  mergesort ranking, threshold membership, selected index/name validation,
  table order, `output_order`, and frozen-transform order.
- **Tie-breaking:** stable `mergesort` and positional tie order remain exact.
- **Error semantics:** preserve `check_is_fitted`, all validation messages and
  causes, and add no forwarding wrapper frame.
- **Laziness:** preserve the module/qualname guard and exact
  `StabilitySelector` identity check in `as_result`; lazy-import the adapter
  only after that check. The sibling must not import `sift.stability`.
- **Floating-point:** preserve frequency/coef validation, threshold comparison,
  stable sorting, and copies without reordering operations.
- **RNG:** no RNG is consumed by adaptation.
- **Observable side effects:** preserve construction of the frozen selector,
  copied sklearn output config, bound transform method, and the absence of live
  training/coefficient state.
- **Public API:** `StabilitySelector`, `SelectionView`, `as_result`, and the
  58-name public spine remain in their historical modules.
- **Private state:** preserve reads of all fitted and provenance attributes,
  including optional `coef_bootstrap_`, `_fit_feature_names_generated_`,
  `_fit_input_kind_`, `_sklearn_output_config`, and `_fit_used_*` flags.
- **Dependencies:** the sibling imports `SelectionView`,
  `_coerce_feature_names`, `_coerce_indices`, `_labels_equal`,
  `_numeric_vector`, `_strict_integer`, and `_validate_selected_identity` from
  `view.py`; `ordered_indices`/`validate_output_order` directly from
  `_selector_compat`; plus `copy`, `math`, `Real`, `Any`, NumPy, and pandas.
  Keep `check_is_fitted` as the function's first local import, matching the
  source body exactly; do not hoist it to module scope. The sibling does not
  import `sift.stability`.
- **Serialization:** the selector/view classes do not move; frozen view
  transforms and existing selector/view pickle behavior remain unchanged.
- **Runtime provenance:** refresh once after the full three-module split.

## Planned edit

- Add `sift/selection/view_stability.py` with the verbatim adapter.
- Remove the function from `view.py`; remove the now-dead
  `from sift._selector_compat import ordered_indices, validate_output_order`
  line.
- Lazy-import the moved adapter inside the exact StabilitySelector branch.

## Verification

- [x] AST/body equality against base `2370631`: `ast.dump` of
  `_as_stability_selector` args+body from
  `237063195707c3a45571b8c59824b29f52665f61:sift/selection/view.py` matches
  `sift/selection/view_stability.py`. First body statement remains
  `from sklearn.utils.validation import check_is_fitted`.
- [x] `tests/test_stability_result_view.py`, `tests/test_selection_view.py`,
  and `tests/contracts/test_public_spine.py` passed under warnings-as-errors
  (**303 passed**).
- [x] Bare `import sift` / `import sift.selection.view` and unrelated dispatch
  leave `sift.selection.view_stability` unloaded; direct sibling/function
  import succeeds without a cycle and reports
  `__module__ == "sift.selection.view_stability"`. The sibling has no
  `sift.stability` import edge.
- [x] Public spine remains green.
- [x] Full suite: **1 failed, 1,992 passed, 40 skipped**. The only failure is
  `tests/test_runtime_scaling_benchmark.py::test_committed_runtime_evidence_and_documented_table_are_bound`
  (source-hash binding). Provenance test was not edited, weakened, or skipped.
- [x] Ruff (default select and `--select F401,F841`) and `git diff --check`
  are clean.
