# V5 — extract the FeaturePathEvaluationResult adapter cluster

## Equivalence contract

- **Inputs covered:** complete and partial `FeaturePathEvaluationResult` views,
  real `evaluate_feature_path` results, explicit/raw feature identities,
  duplicate and ambiguous labels, empty/all-failed paths, malformed scores and
  diagnostics, floating-point edge cases, copy isolation, and legacy
  constructor/pickle behavior.
- **Functions moved together:** `_path_result_scores`,
  `_numeric_values_equal`, `_validate_path_diagnostics`,
  `_resolve_path_positions`, and `_as_feature_path_result`. These form one
  private adapter cluster and have no call sites outside the cluster and the
  exact `as_result` branch.
- **Ordering and ties:** preserve tested-k order, lower-score then lower-k
  best selection, feature-path order, stable positional identity resolution,
  table order, and nullable integer dtypes.
- **Floating point:** preserve exact `np.isclose(..., rtol=0, atol=0,
  equal_nan=True)`, finite/positive-infinity rules, float64 conversion guards,
  standard-error formula, NaN placement, and copies without changing operation
  order.
- **Errors:** preserve every validation message, explicit cause, and frame
  depth; add no forwarding wrapper. Pin the three current origin depths: 3
  frames for adapter-body errors, 4 for cluster-helper errors, and 6 for
  errors originating in retained `view.py` helpers.
- **Laziness:** retain the existing local import and exact identity check for
  `FeaturePathEvaluationResult` in `as_result`; lazy-import the sibling only
  inside that exact branch. The sibling must not import
  `sift.selection.path_eval`.
- **Dependencies:** the sibling imports `SelectionView`, `_coerce_feature_names`,
  `_coerce_indices`, `_label_token`, `_labels_equal`, `_numeric_vector`,
  `_strict_integer`, and `_strict_integer_vector` from `view.py`; plus `math`,
  `Mapping`, `Real`, `Any`, NumPy, and pandas. It imports no result class.
- **Retained facade helpers:** do not remove `_strict_integer_vector` or
  `_numeric_vector` from `view.py`. This move leaves them without an in-module
  caller, but accepted sibling adapters import them. No module-level import in
  `view.py` is expected to become dead in this slice.
- **Public API:** `FeaturePathEvaluationResult`, `evaluate_feature_path`,
  `SelectionView`, `as_result`, and all public symbol modules remain unchanged.
- **Serialization:** the result/view classes do not move; the legacy
  constructor and pickle contracts stay exact. The dataclass's current plain
  `__eq__` raises `ValueError` because its DataFrame field has ambiguous truth;
  preserve that behavior and compare pickle round-trips field-by-field.
- **Runtime provenance:** defer the single source-hash refresh until all three
  oversized source modules have been split.

## Planned edit

- Add `sift/selection/view_path.py` containing the five-function cluster with
  verbatim bodies.
- Remove those five definitions from `view.py`. Retain all other facade
  definitions, including `_strict_integer_vector` and `_numeric_vector`.
  Expected import deletion: none; stop and re-check if F401 suggests one.
- Lazy-import `_as_feature_path_result` inside the existing exact
  `FeaturePathEvaluationResult` branch.
- Extend generic lazy-adapter and fresh-interpreter import tests minimally.
  Add only `sift.selection.view_path` to the lazy guard; core
  `sift.selection.path_eval` remains an unconditional local dispatch import.

## Verification

- [x] AST args+body equality for all five moved functions against base
  `2370631` (`ast.dump` of `_path_result_scores`, `_numeric_values_equal`,
  `_validate_path_diagnostics`, `_resolve_path_positions`,
  `_as_feature_path_result`).
- [x] `tests/test_selection_view.py`, `tests/test_feature_path_evaluation.py`,
  and `tests/contracts/test_public_spine.py` passed under warnings-as-errors
  (**291 passed**).
- [x] Bare `import sift` / `import sift.selection.view` and unrelated dispatch
  leave `sift.selection.view_path` unloaded; direct sibling/function import
  succeeds without a cycle and reports
  `__module__ == "sift.selection.view_path"`. The sibling has no
  `sift.selection.path_eval` import edge. Core `path_eval` remains an
  unconditional local dispatch import in `as_result`.
- [x] Public spine remains green. `_strict_integer_vector` and
  `_numeric_vector` remain in `view.py`; F401 reported no dead facade imports.
- [x] Full suite: **1 failed, 1,994 passed, 40 skipped**. The only failure is
  `tests/test_runtime_scaling_benchmark.py::test_committed_runtime_evidence_and_documented_table_are_bound`.
  Provenance test was not edited, weakened, or skipped.
- [x] Ruff (default select and `--select F401,F841` on all view modules and
  the touched test) and `git diff --check` are clean.
