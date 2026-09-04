# R1 — extract the filter auto-k curve payload cluster

This is a cohesion/file-size split, not import-time decoupling. Downstream
consumers (`filter_payloads.py`, `view_filter.py`) intentionally remain on
the stable `sift.selection.filter_auto_k` facade.

## Equivalence contract

- **Cluster moved:** `AUTO_K_CURVE_KEY`, `AUTO_K_CURVE_COLUMNS`,
  `_AUTO_K_CURVE_CRITERIA`, `_AUTO_K_CURVE_UNAVAILABLE`,
  `_auto_k_curve_unavailable`, and `build_auto_k_curve_payload`.
- **Retained on the facade:** `auto_k_summary` stays implemented in
  `filter_auto_k.py` (19 internal callers, no external consumers). Moving it
  would create an artificial sibling dependency and an asymmetric patch point.
- **Facade:** `sift.selection.filter_auto_k` re-exports the moved objects
  (`is` identity). Unused names use the explicit `X as X` form so Ruff F401
  does not drop them. Downstream imports stay on `filter_auto_k`.
- **Ordering/dtypes:** preserve mergesort-by-`k` curve construction, exact
  column order `k`, `criterion`, `criterion_se`, `selected`, and int64/float64
  /bool dtypes.
- **Errors/unavailable routes:** preserve unavailable-reason strings and
  knockoff_path/consensus maps verbatim.
- **Laziness/cycles:** the sibling imports numpy/pandas plus `typing.Optional`.
  It does not import `filter_auto_k` or `AutoKConfig`.
- **Monkeypatch:** no test patches the moved names. Internal
  `auto_k_summary(...)` lookups remain native facade functions.
- **Public API:** no `sift.__all__` change. Curve payload key remains
  `"auto_k_curve"`.
- **Runtime provenance:** still deferred until the full three-module split.

## Planned edit

- Add `sift/selection/filter_auto_k_curve.py` with verbatim curve-payload
  bodies.
- Import those names into `filter_auto_k.py` as the stable facade.
- Keep `auto_k_summary` in the facade with its exact base-commit body.
- Add a small facade/payload contract test, including one unavailable-route
  branch.

## Verification

- [x] AST args/bodies and constant values of the six moved names match base
  `2370631`. `auto_k_summary` AST matches the base facade body. Remaining
  facade function bodies are unchanged.
- [x] Focused filter/result/curve/import tests under warnings-as-errors:
  `tests/test_filter_auto_k_curve_facade.py`, `tests/test_filter_results.py`,
  `tests/test_selection_view.py`, `tests/contracts/test_public_spine.py`
  (**291 passed**).
- [x] Ruff (default select and `--select F401,F841` on the three R1 Python
  files) and `git diff --check` are clean.
- [x] Accepted by Codex and Opus. Codex reproduced the focused **291 passed**
  gate and AST equivalence; Opus independently verified the import-order
  matrix, facade identity, payload edge branches, and full suite (**2,001
  passed / 40 skipped / 1 expected provenance failure**).
