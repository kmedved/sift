# R3 — extract binary auto-k route functions

This is a cohesion/file-size split, not import-time decoupling. Downstream
imports in `filter_payloads.py` stay on the `sift.selection.filter_auto_k`
facade. Shared count/guard primitives are imported from
`filter_auto_k_common.py`, not back through the facade.

## Equivalence contract

- **Cluster moved:** `select_binary_elbow`, `select_binary_penalized`,
  `select_binary_posterior`, `select_binary_changepoint`, and
  `select_binary_evaluate`.
- **Facade:** `sift.selection.filter_auto_k` re-exports all five names
  (`X as X` identity). After the move, unused `cefsplus_binary_common`
  imports leave the facade; `select_k_changepoint` stays because Gaussian
  and consensus callers still use it.
- **Shared helpers:** the sibling imports
  `_select_elbow_count`, `_select_penalized_count`, `_select_posterior_count`,
  `_print_selected_k`, `_effective_max_k`, `_zero_capable_effective_min_k`,
  `_objective_n_eff`, `_require_eval_split_context`, and `auto_k_summary`
  from `filter_auto_k_common`.
- **`auto_k_module`:** the sibling binds
  `from sift.selection import auto_k as auto_k_module`. Attribute patches
  through `filter_auto_k.auto_k_module.select_k_auto` still reach
  `select_binary_evaluate`. Existing proof:
  `tests/test_cefsplus_binary.py` near line 963.
- **Laziness/cycles:** the sibling is a leaf. It does not import
  `filter_auto_k`. Binary types/utilities come from
  `cefsplus_binary_common`; `select_k_changepoint` comes from
  `auto_k_stop`.
- **Warnings:** these five functions emit no warnings; no wrappers or extra
  stack frames.
- **Public API:** no `sift.__all__` change. `filter_payloads.py` imports
  remain on the facade.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/filter_auto_k_binary.py` with verbatim bodies.
- Import the five names into `filter_auto_k.py` as the stable facade.
- Add a small identity / module-object / leaf-import contract test.

## Verification

- [x] AST of all five definitions matches base `2370631`. Remaining facade
  function bodies are unchanged. Whole-module FunctionDef accounting:
  base 45 = facade 28 + common 10 + curve 2 + binary 5.
- [x] Focused tests under warnings-as-errors:
  `tests/test_cefsplus_binary.py`, `tests/test_auto_k_v2.py`,
  `tests/test_filter_results.py`, `tests/contracts/test_auto_k_context_compat.py`,
  `tests/contracts/test_autok_ergonomics.py`,
  `tests/test_select_k_auto_no_target_leak.py`,
  `tests/test_filter_auto_k_curve_facade.py`,
  `tests/test_filter_auto_k_common_facade.py`,
  `tests/test_filter_auto_k_binary_facade.py`
  (**360 passed / 4 skipped**).
- [x] Ruff (default select and `--select F401,F841`) on changed Python files
  and `git diff --check` are clean.
- [x] Accepted by Codex and Opus. Codex reproduced a **354 passed / 3
  skipped** binary/seam gate and exact five-function accounting; Opus
  independently verified runtime signatures, import orders, shared module
  patches, dead-import removal, and the existing evaluator seam test.
