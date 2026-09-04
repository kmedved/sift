# A1 — extract auto-k penalty/objective helpers

This is a cohesion/file-size split, not import-time decoupling.
`select_k_penalized_objective` and `select_k_posterior` stay in
`sift.selection.auto_k` and look up the helpers by bare name. Downstream
`filter_auto_k_common` still reads `auto_k_module._objective_weight_diagnostics`.

## Equivalence contract

- **Cluster moved:** `_resolve_n_eff_mode`, `_penalty_weight`, `_log_comb`,
  `_resolve_ebic_gamma`, `_penalty_array`, and
  `_objective_weight_diagnostics`.
- **Retained on the facade:** `AutoKConfig`, `validate_auto_k_config`,
  `select_k_auto`, `select_k_penalized_objective`, `select_k_posterior`,
  and public exports/docs. `AutoKConfig.__module__` remains
  `sift.selection.auto_k`.
- **Facade:** `sift.selection.auto_k` re-exports all six names (`X as X`
  identity). `gammaln` leaves the facade because its only in-module user
  moved; `ensure_weights` stays because evaluate-path callers still use it.
- **Laziness/cycles:** the sibling is a true leaf at runtime. It imports
  numpy, `scipy.special.gammaln`, and `sift._preprocess.ensure_weights`.
  `AutoKConfig` is imported only under `TYPE_CHECKING`. It does not
  runtime-import `sift.selection.auto_k`.
- **Consumers:** `auto_k_module._objective_weight_diagnostics` remains the
  same object for `filter_auto_k_common`. Tests that call
  `auto_k_module._log_comb` and `auto_k_module._resolve_ebic_gamma` keep
  working through the facade aliases.
- **Public API:** no `sift.__all__` change.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/auto_k_objective.py` with verbatim bodies.
- Import the six names into `auto_k.py` as the stable facade.
- Add a small identity / leaf-import / AutoKConfig module / consumer
  identity contract test.

## Verification

- [x] AST of all six definitions matches base `2370631`. Remaining facade
  function bodies are unchanged. Whole-module FunctionDef accounting:
  base 28 = facade 22 + objective 6.
- [x] Focused tests under warnings-as-errors:
  `tests/test_auto_k_v2.py`, `tests/contracts/test_autok_ergonomics.py`,
  `tests/test_select_k_auto_no_target_leak.py`,
  `tests/test_auto_k_objective_facade.py`
  (**191 passed / 1 skipped**).
- [x] `scripts/generate_api_reference.py --check` (58 exports plus index).
  Ruff (default select, `F401,F841`, and `F821`) and `git diff --check`
  are clean. A1 did not edit the filter_auto_k family, `filter_payloads.py`,
  or docs.
- [x] Accepted by Codex and Opus. Codex reproduced **199 passed / 1
  skipped**, exact AST accounting, generated-reference freshness, and the
  dataclass/pickle contract. Opus independently verified import orders,
  module-attribute consumers, objective numerics/errors, and the unchanged
  58-name public-spine fingerprint.
