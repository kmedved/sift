# A5 — extract AutoKConfig and validation/config helpers

This is the final original three-module split slice. It is a
cohesion/file-size split, not import-time decoupling. Public import paths
remain `sift.selection.auto_k.AutoKConfig` (and the three other pinned
callables). After the verbatim definitions, the sibling assigns
`__module__ = "sift.selection.auto_k"` on `AutoKConfig`,
`validate_auto_k_config`, `resolve_auto_k_config`, and
`with_effective_k_bounds`.

## Equivalence contract

- **Cluster moved:** 31 top-level nodes from `@dataclass AutoKConfig`
  through `resolve_auto_k_config`, including all validation constants/state
  and the eight functions/context manager.
- **Facade re-exports:** 30 of 31 names as `X as X`. `_DEFAULT_AUTOK_CONFIG`
  is not re-exported: it is lazily rebound in the leaf, has zero consumers,
  and a by-value alias would be silently stale.
- **Incidental facade attributes removed:** 15 unique zero-consumer names
  that were previously importable from `sift.selection.auto_k` only because
  the facade imported them for the moved cluster: `AutoKCVOptions`,
  `AutoKExperimentalOptions`, `AutoKKnockoffOptions`,
  `AutoKObjectiveOptions`, `AutoKPermutationOptions`,
  `AutoKStabilityOptions`, `AutoKTestOptions`, `AUTO_K_OPTION_GROUP_TYPES`,
  `ContextVar`, `Iterator`, `contextmanager`, `dataclass`,
  `dataclass_fields`, `replace`, and `warn_external`. Plus the separately
  documented `_DEFAULT_AUTOK_CONFIG`. This is accepted surface narrowing of
  non-exported incidental imports, not a public API change.
- **Typing globals:** the facade retains `from typing import Any as Any,
  List, Literal, Optional, Tuple` so `typing.get_type_hints(AutoKConfig)`
  still sees `Any` and `Literal` through the historical module.
- **Known introspection loss (accepted):** `inspect.getsource(AutoKConfig)`
  raises `OSError` and `inspect.getsourcefile` points at the facade, because
  the class is physically in the sibling while `__module__` is historical.
  Method source still works. No machinery is added to mask this.
- **Seams:** `select_k_auto` stays native. Facade-resident lookups bind to
  the extracted validation functions.
  `resolve_auto_k_config.__globals__ is vars(auto_k_config)`.
- **Public API:** no `sift.__all__` change. Generated-reference paths stay
  on `sift.selection.auto_k`.
- **Runtime provenance:** still deferred until the implementation commit
  after this split is accepted.

## Planned edit

- Add `sift/selection/auto_k_config.py` with the verbatim 31-node block
  plus four historical `__module__` assignments.
- Re-export 30 names from `auto_k.py`; drop dead imports.
- Add a facade/config contract test covering identity, pickle, hints,
  ContextVar suppression, lazy default, seams, and the accepted
  getsource loss.

## Verification

- [x] AST of all 31 moved nodes matches base `2370631`. No lost/new/duplicate
  function/class bodies across facade and siblings. Whole-module
  FunctionDef+ClassDef accounting: base 29 = facade 5 + config 9 +
  objective 6 + score 7 + elbow 1 + path 1.
- [x] Focused tests under warnings-as-errors (Opus set plus A1–A4 facade
  tests): **882 passed / 15 skipped**.
- [x] `scripts/generate_api_reference.py --check` (58 exports plus index).
  Ruff (default select and explicit `F401,F841,F821`) is clean on the A5
  files; `git diff --check` is clean. A5 did not edit docs or the
  filter_auto_k family.
- [x] Strict docs: `uv run --isolated --extra docs mkdocs build --strict
  --site-dir /private/tmp/sift-mkdocs-a5` succeeded using the declared docs
  extra. MkDocs 1.6.1; documentation built cleanly.
- [x] Accepted by Codex and Opus. Codex focused gate **882 passed / 15
  skipped**. Opus ran 288 `validate_auto_k_config` and 32
  `resolve_auto_k_config` differentials with zero mismatches. No defects.
