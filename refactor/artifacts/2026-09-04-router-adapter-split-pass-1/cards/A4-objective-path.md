# A4 — extract public `compute_objective_for_path`

This is a cohesion/file-size split, not import-time decoupling. The public
import path remains `sift.selection.auto_k.compute_objective_for_path`. After
the verbatim definition, the sibling assigns
`compute_objective_for_path.__module__ = "sift.selection.auto_k"` so pickle
and generated-reference lookup stay on the historical facade path.

## Equivalence contract

- **Cluster moved:** `compute_objective_for_path` (full docstring, local
  imports, and body).
- **Facade:** `from sift.selection.auto_k_path import
  compute_objective_for_path as compute_objective_for_path`. The object
  exposed by the sibling, facade, `sift.selection`, and top-level `sift` is
  identical. `sift.api` never exported this name and is unchanged.
- **Laziness/cycles:** module-scope runtime imports are `typing` and numpy.
  `FeatureCache` is `TYPE_CHECKING`-only. Function-local imports of
  `sift.estimators.copula`, `sift.selection.objective`, and
  `sift.selection.knockoff_filter` stay inside the function. The sibling
  does not runtime-import `sift.selection.auto_k`.
- **Facade cleanup:** the unused `TYPE_CHECKING` `FeatureCache` import/block
  is removed from `auto_k.py`.
- **Public API:** no `sift.__all__` change. Generated-reference source path
  remains `sift.selection.auto_k.compute_objective_for_path`.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/auto_k_path.py` with the verbatim function plus the
  historical `__module__` assignment.
- Remove the original definition from `auto_k.py`.
- Add a small identity / pickle / signature / leaf / cached-uncached
  contract test.

## Verification

- [x] AST of `compute_objective_for_path` matches base `2370631` (function
  definition only, before the `__module__` assignment). Remaining facade
  function bodies are unchanged. Whole-module FunctionDef accounting:
  base 28 = facade 13 + objective 6 + score 7 + elbow 1 + path 1.
- [x] Focused tests under warnings-as-errors:
  `tests/test_auto_k_path_facade.py`, `tests/test_cefsplus.py`,
  `tests/test_optimizations.py::TestObjectivePathHelper`,
  `tests/test_auto_k_v2.py`, `tests/contracts/test_public_spine.py`,
  `tests/contracts/test_autok_ergonomics.py`,
  `tests/test_docstring_coverage.py`, `tests/test_docstring_examples.py`,
  `tests/test_generated_api_reference.py`,
  `tests/test_auto_k_objective_facade.py`,
  `tests/test_auto_k_score_facade.py`,
  `tests/test_auto_k_elbow_facade.py`
  (**363 passed / 4 skipped**).
- [x] `scripts/generate_api_reference.py --check` (58 exports plus index).
  Ruff (default select and explicit `F401,F841,F821`) is clean on the A4
  files; `git diff --check` is clean. A4 did not edit docs or the
  filter_auto_k family.
- [x] Accepted by Codex and Opus. Codex reproduced exact AST/function
  accounting, identity/pickle behavior, cache variants, and a 427-passed
  focused gate. Opus additionally ran a 700-case base-vs-current differential
  sweep with zero mismatches and verified validation order, lazy import
  timing, the experimental route, and generated-reference resolution.
