# A2 — extract private score-curve selection-rule helpers

This is a cohesion/file-size split, not import-time decoupling. Public
`choose_k_from_score_curve` stays natively defined on `sift.selection.auto_k`
because it calls `validate_auto_k_config` and is imported by
`auto_k_xfit` / `auto_k_nested`. It resolves `_RULE_SELECTORS` on the facade.

## Equivalence contract

- **Cluster moved:** `_score_curve_tolerance`, `_choose_best_rule`,
  `_choose_one_se_rule`, `_mark_tolerance`, `_choose_tolerance_rule`,
  `_selected_plateau_ks`, `_choose_plateau_rule`, and `_RULE_SELECTORS`.
- **Retained on the facade:** `choose_k_from_score_curve`, `AutoKConfig`,
  `validate_auto_k_config`, `select_k_auto`, and public exports/docs.
  `AutoKConfig.__module__` remains `sift.selection.auto_k`.
- **Facade:** `sift.selection.auto_k` re-exports all eight names (`X as X`
  identity). `_RULE_SELECTORS` is the same dict object; its values are the
  moved helper functions.
- **Warnings:** `_choose_one_se_rule` keeps `stacklevel=3`. With no wrapper
  and `choose_k_from_score_curve` still the immediate caller, the one-SE
  fallback warning remains caller-facing.
- **Laziness/cycles:** the sibling is a true leaf at runtime. It imports
  numpy and `warnings`. pandas and `AutoKConfig` are `TYPE_CHECKING`-only.
  It does not runtime-import `sift.selection.auto_k`.
- **Public API:** no `sift.__all__` change.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/auto_k_score.py` with verbatim bodies.
- Import the eight names into `auto_k.py` as the stable facade.
- Add a small identity / leaf-import / selector-map contract test.

## Verification

- [x] AST of the seven functions and `_RULE_SELECTORS` matches base
  `2370631`. `choose_k_from_score_curve` remains native and body-identical.
  Whole-module FunctionDef accounting:
  base 28 = facade 15 + objective 6 + score 7.
- [x] Focused tests under warnings-as-errors:
  `tests/test_select_k_auto_no_target_leak.py`, `tests/test_auto_k_v2.py`,
  `tests/contracts/test_autok_ergonomics.py`,
  `tests/test_auto_k_score_facade.py`,
  `tests/test_auto_k_objective_facade.py`
  (**193 passed / 1 skipped**).
- [x] `scripts/generate_api_reference.py --check` (58 exports plus index).
  Ruff (default select, `F401,F841`, and `F821`) and `git diff --check`
  are clean. A2 did not edit docs or the filter_auto_k family.
- [x] Accepted by Codex and Opus. Codex independently reproduced the AST
  accounting, focused warnings-as-errors gate, public-reference freshness,
  lint, and whitespace checks. Opus additionally verified import-order safety,
  facade and downstream import identities, dataclass/public-spine contracts,
  and the caller-facing one-SE warning location; it reported no actionable
  defect.
