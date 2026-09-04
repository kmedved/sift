# R4 — extract cache, evaluation, and classic auto-k helpers

This is a cohesion/file-size split, not import-time decoupling. Gaussian and
consensus callers of `_cached_filter_path` stay in `filter_auto_k.py`, so
`monkeypatch.setattr(filter_auto_k, "_cached_filter_path", fake)` still
intercepts their bare-name lookups. `filter_payloads.py` keeps importing
`prepare_filter_eval_data` and `select_filter_classic_auto_k` from the facade.

## Equivalence contract

- **Cluster moved:** `prepare_filter_eval_data`, `_cached_filter_path`,
  `_cache_uses_synthetic_feature_names`,
  `_require_positional_cache_dataframe_alignment`, and
  `select_filter_classic_auto_k`.
- **Facade:** `sift.selection.filter_auto_k` re-exports all five names
  (`X as X` identity). `ensure_weights` leaves the facade because its only
  in-module user moved. `EvalData` stays on the facade; the sibling repeats
  the same alias so the moved return annotation remains resolvable.
- **Shared helpers:** the sibling imports `_require_eval_split_context`,
  `_print_selected_k`, `_effective_max_k`, and `auto_k_summary` from
  `filter_auto_k_common`, and `ensure_weights` from `sift._preprocess`.
- **`auto_k_module`:** the sibling binds
  `from sift.selection import auto_k as auto_k_module`. Attribute patches
  through `filter_auto_k.auto_k_module` still reach
  `select_filter_classic_auto_k`.
- **Monkeypatch:** `_cached_filter_path` remains a facade global. Existing
  tests in `tests/test_auto_k_v2.py` remain the seam proof.
- **Laziness/cycles:** the sibling is a leaf. It does not import
  `filter_auto_k`. The historical function-local `select_cached` import
  inside `_cached_filter_path` is unchanged.
- **Public API:** no `sift.__all__` change. `filter_payloads.py` imports
  remain on the facade.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/filter_auto_k_cache.py` with verbatim bodies.
- Import the five names into `filter_auto_k.py` as the stable facade.
- Add a small identity / module-object / leaf-import contract test.

## Verification

- [x] AST of all five definitions matches base `2370631`. Remaining facade
  function bodies are unchanged. Whole-module FunctionDef accounting:
  base 45 = facade 23 + common 10 + curve 2 + binary 5 + cache 5.
- [x] Focused tests under warnings-as-errors:
  `tests/contracts/test_cached_shapes.py`, `tests/test_auto_k_v2.py`,
  `tests/test_filter_results.py`, `tests/test_select_k_auto_no_target_leak.py`,
  `tests/contracts/test_auto_k_context_compat.py`,
  `tests/test_filter_auto_k_curve_facade.py`,
  `tests/test_filter_auto_k_common_facade.py`,
  `tests/test_filter_auto_k_binary_facade.py`,
  `tests/test_filter_auto_k_cache_facade.py`
  (**257 passed / 1 skipped**).
- [x] Ruff (default select and `--select F401,F841`) on changed Python files
  and `git diff --check` are clean.
- [x] Accepted by Codex and Opus. Codex verified exact five-function
  accounting and **243 passed** across cache/weight/context contracts; Opus
  independently verified runtime signatures, import orders, `EvalData`
  equivalence, and all three live `_cached_filter_path` patch seams.
