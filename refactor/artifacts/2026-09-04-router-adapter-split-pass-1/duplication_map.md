# Boundary and coupling map — 2026-09-04-router-adapter-split-pass-1

This roadmap item is an explicitly requested architectural split, not a
duplicate-code collapse. The simplification skill's positive-LOC-savings gate
therefore does not apply; its isomorphism, callsite, and verification gates do.

## V1 — `selection/view.py`

- Core schema/class: serialization and validation helpers plus
  `SelectionView` (lines 34–1182).
- Filter/knockoff/Boruta/path adapters: lines 1185–2021.
- CatBoost adapter: lines 2024–2486.
- Stability adapter: lines 2489–2766.
- Permutation-importance adapter: lines 2769–2971.
- Dispatcher: `as_result` at lines 2974–3110.
- Compatibility imports: top-level `sift.SelectionView`/`sift.as_result`, local
  lazy imports across result classes, and tests importing private
  `_append_rows_like` and `_json_safe`.

## R1 — `selection/filter_auto_k.py`

- Normalized curve payload and summary: lines 87–234.
- automatic router/dense diagnostic: lines 237–525.
- Gaussian per-rule dispatch: lines 528–1463.
- consensus route: lines 1466–1749.
- classic-cache route: lines 1752–1845.
- binary routes: lines 1848–2139.
- Compatibility imports: `filter_payloads.py` imports the broad private/public
  surface. Tests monkeypatch `_cached_filter_path`, `_consensus_method_k`,
  `_run_gaussian_routed_path`, `_run_auto_dense_check`, `select_k_chi2_stop`,
  `bootstrap_paths`, `gaussian_cv_curves`, `select_k_gaussian_cv`, and
  `auto_k_module.select_k_auto` on this original module.

## A1 — `selection/auto_k.py`

- `AutoKConfig` and config validation/resolution: lines 50–1008.
- score-curve rules: lines 1011–1183.
- evaluate-prefix selector: lines 1186–1607.
- elbow selector: lines 1610–1784.
- objective/posterior selectors: lines 1787–2416.
- objective helper: lines 2419–2587.
- Compatibility imports: 30 repository files; public top-level exports include
  `AutoKConfig`, `select_k_auto`, `select_k_elbow`,
  `select_k_penalized_objective`, `select_k_posterior`, and
  `compute_objective_for_path`. Other Auto-K modules import validation helpers.
  One no-target-leak test monkeypatches `auto_k_module.select_k_auto`; moved
  functions must not bypass that seam.

## Non-negotiable observables

- No change to public or current internal import paths and signatures.
- Preserve `AutoKConfig.__module__`, dataclass equality/repr/replace/pickle, and
  SelectionView JSON/schema behavior.
- Preserve exception types/messages, warning categories/counts/stack levels,
  logging, output ordering, tie breaking, floating-point/RNG operation order,
  and lazy imports.
- Original-module monkeypatches must still intercept the call sites they pin.
