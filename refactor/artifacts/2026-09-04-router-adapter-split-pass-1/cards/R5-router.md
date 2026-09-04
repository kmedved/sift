# R5 — extract auto-k router configuration helpers

This is a cohesion/file-size split, not import-time decoupling. Downstream
imports in `filter_payloads.py` stay on the `sift.selection.filter_auto_k`
facade, including `_AUTOK_FIELD_DEFAULTS` as the same object.

## Equivalence contract

- **Cluster moved:** `auto_k_mode_label`, `_auto_route_facts`,
  `_AUTOK_FIELD_DEFAULTS`, `_strip_router_only_fields`, and
  `_auto_route_config`.
- **Gaussian routes retained:** `select_gaussian_auto_path` and every
  Gaussian route body stay in the facade. A 12-route move would break live
  facade monkeypatch seams for `_cached_filter_path` (and related helpers)
  or abandon verbatim/leaf extraction. The facade remains the cohesive
  Gaussian/auto/consensus orchestration owner.
- **Facade:** `sift.selection.filter_auto_k` re-exports all five names
  (`X as X` identity). `_AUTOK_FIELD_DEFAULTS` is the same object the
  sibling created. `dataclasses.replace` stays on the facade because
  Gaussian/consensus callers still use it.
- **Laziness/cycles:** the sibling is a true leaf. It imports only
  `dataclasses.replace`, numpy, and `AutoKConfig`. It does not import
  `filter_auto_k`.
- **Public API:** no `sift.__all__` change. `filter_payloads.py` is
  unchanged.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/filter_auto_k_router.py` with verbatim bodies.
- Import the five names into `filter_auto_k.py` as the stable facade.
- Add a small identity / leaf-import contract test.

## Verification

- [x] AST of the four functions and `_AUTOK_FIELD_DEFAULTS` matches base
  `2370631`. Remaining facade function bodies are unchanged.
  Whole-module FunctionDef accounting:
  base 45 = facade 19 + common 10 + curve 2 + binary 5 + cache 5 + router 4.
- [x] Focused tests under warnings-as-errors:
  `tests/test_auto_k_v2.py`, `tests/contracts/test_autok_ergonomics.py`,
  `tests/test_filter_results.py`, `tests/test_release_readiness.py`,
  `tests/test_select_k_auto_no_target_leak.py`,
  `tests/test_filter_auto_k_curve_facade.py`,
  `tests/test_filter_auto_k_common_facade.py`,
  `tests/test_filter_auto_k_binary_facade.py`,
  `tests/test_filter_auto_k_cache_facade.py`,
  `tests/test_filter_auto_k_router_facade.py`
  (**270 passed / 1 skipped**).
- [x] Ruff (default select, `F401,F841`, and `F821`) on changed Python files
  and `git diff --check` are clean. `filter_payloads.py` has no diff.
- [x] Accepted by Codex and Opus. Codex reproduced the **270 passed / 1
  skipped** gate and exact cluster accounting; Opus independently verified
  constant identity through `filter_payloads`, import orders, every router
  branch/reason, labels, and the deliberate Gaussian-route retention.
