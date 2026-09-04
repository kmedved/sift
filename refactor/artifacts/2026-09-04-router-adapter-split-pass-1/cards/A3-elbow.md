# A3 — extract public `select_k_elbow`

This is a cohesion/file-size split, not import-time decoupling. The public
import path remains `sift.selection.auto_k.select_k_elbow`. After the
verbatim definition, the sibling assigns
`select_k_elbow.__module__ = "sift.selection.auto_k"` so pickle and
introspection keep the historical facade path.

## Equivalence contract

- **Cluster moved:** `select_k_elbow` (full docstring and body).
- **Facade:** `from sift.selection.auto_k_elbow import select_k_elbow as
  select_k_elbow`. The object exposed by the sibling, facade,
  `sift.selection`, `sift.api`, and top-level `sift` is identical.
- **Laziness/cycles:** the sibling is a true leaf. Runtime imports are
  `typing.Tuple`, numpy, and pandas. It does not import
  `sift.selection.auto_k` or any other SIFT module.
- **Warnings/errors:** no wrappers and no extra stack frames. Validation
  messages, empty-path `(0, empty DataFrame)`, mergesort-free elbow scan,
  and diagnostic dtypes/columns stay unchanged.
- **Public API:** no `sift.__all__` change. Generated-reference source path
  remains `sift.selection.auto_k.select_k_elbow`.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/auto_k_elbow.py` with the verbatim function plus the
  historical `__module__` assignment.
- Remove the original definition from `auto_k.py`.
- Add a small identity / pickle / signature / leaf / output contract test.

## Verification

- [x] AST of `select_k_elbow` matches base `2370631` (function definition
  only, before the `__module__` assignment). Remaining facade function
  bodies are unchanged. Whole-module FunctionDef accounting:
  base 28 = facade 14 + objective 6 + score 7 + elbow 1.
- [x] Focused tests under warnings-as-errors:
  `tests/test_auto_k_elbow_facade.py`,
  `tests/test_public_api_untested_exports.py`,
  `tests/contracts/test_public_spine.py`,
  `tests/test_docstring_coverage.py`, `tests/test_docstring_examples.py`,
  `tests/test_generated_api_reference.py`,
  `tests/test_auto_k_objective_facade.py`,
  `tests/test_auto_k_score_facade.py`
  (**254 passed / 4 skipped**).
- [x] `scripts/generate_api_reference.py --check` (58 exports plus index).
  Ruff (default select and explicit `F401,F841,F821`) is clean on the A3
  files; `git diff --check` is clean. A3 did not edit docs or the
  filter_auto_k family.
- [x] Accepted by Codex and Opus. Codex reproduced exact AST/function
  accounting, public identities, pickle resolution, the generated-reference
  check, and a 445-passed focused gate. Opus additionally ran a 1,620-case
  base-vs-current differential sweep with zero mismatches and verified the
  alias loader path. Local `mkdocs build --strict` remains an integration-gate
  check because the current environment lacks mkdocstrings/griffe/pymdownx;
  generated-reference freshness itself is green.
