# R2 — extract shared auto-k count and guard primitives

This is a cohesion/file-size split, not import-time decoupling. Route,
router, and consensus callers stay in `sift.selection.filter_auto_k`. The
facade re-exports the moved names as identity aliases so existing imports
and module-level monkeypatch seams remain effective.

## Equivalence contract

- **Cluster moved:** `auto_k_summary`, `_zero_capable_effective_min_k`,
  `_effective_max_k`, `_require_eval_split_context`, `_print_selected_k`,
  `_select_elbow_count`, `_select_penalized_count`, `_select_posterior_count`,
  `_objective_n_eff`, and `_gain_test_candidate_inputs`.
- **Callers retained:** do not move Gaussian/classic/binary/consensus route
  bodies. `_consensus_method_k` still looks up `_gain_test_candidate_inputs`
  on the facade, so
  `tests/test_auto_k_v2.py::test_consensus_gain_tests_preserve_panel_semantics`
  keeps proving that seam.
- **Facade:** `sift.selection.filter_auto_k` re-exports all ten names
  (`X as X` identity). Unused `logger` and `build_candidate_panel` imports
  leave the facade because their only in-module users moved.
- **`auto_k_module`:** the sibling binds
  `from sift.selection import auto_k as auto_k_module` (the module object,
  never selector functions by value). The facade keeps the same binding for
  remaining callers. `filter_auto_k.auto_k_module is
  filter_auto_k_common.auto_k_module`. Attribute patches through
  `filter_auto_k.auto_k_module` remain effective.
- **Laziness/cycles:** the sibling is a leaf. It does not import
  `filter_auto_k`. Source imports are typing, numpy, pandas, logger,
  `auto_k` as `auto_k_module`, `AutoKConfig`, and `build_candidate_panel`.
- **Public API:** no `sift.__all__` change.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/filter_auto_k_common.py` with verbatim bodies.
- Import the ten names into `filter_auto_k.py` as the stable facade.
- Add a small identity / module-object / leaf-import contract test.

## Verification

- [x] AST of all ten definitions matches base `2370631`. Remaining facade
  function bodies are unchanged. Whole-module FunctionDef accounting:
  base 45 = facade 33 + common 10 + curve 2.
- [x] Focused tests under warnings-as-errors:
  `tests/test_auto_k_v2.py`, `tests/test_cefsplus_binary.py`,
  `tests/test_filter_results.py`, `tests/contracts/test_autok_ergonomics.py`,
  `tests/test_release_readiness.py`, `tests/test_filter_auto_k_curve_facade.py`,
  `tests/test_filter_auto_k_common_facade.py`
  (**280 passed / 3 skipped**).
- [x] Ruff (default select and `--select F401,F841`) on changed Python files
  and `git diff --check` are clean.
- [x] Accepted by Codex and Opus. Codex reproduced the **280 passed / 3
  skipped** warnings-as-errors gate and exact ten-function accounting; Opus
  independently verified leaf imports, identity aliases, shared module-object
  patches, logger identity, and the existing consensus monkeypatch seam.
