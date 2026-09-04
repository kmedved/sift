# Ledger — 2026-09-04-router-adapter-split-pass-1

| order | ID | commit | files | facade lines before/after | total LOC delta | tests | lints |
| ---: | --- | --- | --- | --- | ---: | --- | --- |
| 1 | V1 | committed fa74d63 | `view.py`, `view_catboost.py`, boundary tests | 3,113 → 2,650 | +21 | 1,986 / 40, plus one provenance deselection | clean |
| 2 | V2 | committed fa74d63 | `view.py`, `view_importance.py`, boundary tests | 2,650 → 2,447 | +22 | 1,988 / 40, plus one known provenance failure | clean |
| 3 | V3 | committed fa74d63 | `view.py`, `view_boruta.py`, boundary tests | 2,447 → 2,355 | +19 | 1,990 / 40, plus one known provenance failure | clean |
| 4 | V4 | committed fa74d63 | `view.py`, `view_stability.py`, boundary tests | 2,355 → 2,076 | +23 | 1,992 passed / 40 skipped / 1 known provenance failure | clean |
| 5 | V5 | committed fa74d63 | `view.py`, `view_path.py`, boundary tests | 2,076 → 1,761 | +24 | 1,994 passed / 40 skipped / 1 known provenance failure | clean |
| 6 | V6 | committed fa74d63 | `view.py`, `view_knockoff.py`, boundary tests | 1,761 → 1,610 | +26 | 1,996 passed / 40 skipped / 1 known provenance failure | clean |
| 7 | V7 | committed fa74d63 | `view.py`, `view_filter.py`, boundary tests | 1,610 → 1,409 | +23 | 1,998 passed / 40 skipped / 1 known provenance failure | clean |
| 8 | R1 | committed fa74d63 | `filter_auto_k.py`, `filter_auto_k_curve.py`, boundary test | 2,139 → 2,005 | +18 | 2,001 passed / 40 skipped / 1 expected provenance failure | clean |
| 9 | R2 | committed fa74d63 | `filter_auto_k.py`, `filter_auto_k_common.py`, boundary test | 2,005 → 1,843 | +22 | focused 280 passed / 3 skipped | clean |
| 10 | R3 | committed fa74d63 | `filter_auto_k.py`, `filter_auto_k_binary.py`, boundary test | 1,843 → 1,548 | +30 | focused 360 passed / 4 skipped | clean |
| 11 | R4 | committed fa74d63 | `filter_auto_k.py`, `filter_auto_k_cache.py`, boundary test | 1,548 → 1,421 | +26 | focused 257 passed / 1 skipped | clean |
| 12 | R5 | committed fa74d63 | `filter_auto_k.py`, `filter_auto_k_router.py`, boundary test | 1,421 → 1,324 | +16 | focused 270 passed / 1 skipped | clean |
| 13 | A1 | committed fa74d63 | `auto_k.py`, `auto_k_objective.py`, boundary test | 2,587 → 2,479 | +21 | focused 199 passed / 1 skipped | clean |
| 14 | A2 | committed fa74d63 | `auto_k.py`, `auto_k_score.py`, boundary test | 2,479 → 2,388 | +22 | Codex focused 199 passed / 1 skipped; Opus broader 305 passed / 1 skipped; full suite not rerun | clean |
| 15 | A3 | committed fa74d63 | `auto_k.py`, `auto_k_elbow.py`, boundary test | 2,388 → 2,212 | +12 | Codex focused 445 passed / 4 skipped; Opus 1,620-case differential sweep, 0 mismatches | clean |
| 16 | A4 | committed fa74d63 | `auto_k.py`, `auto_k_path.py`, boundary test | 2,212 → 2,039 | +11 | Codex focused 427 passed / 4 skipped; Opus 700-case differential sweep, 0 mismatches | clean |
| 17 | A5 | committed fa74d63 | `auto_k.py`, `auto_k_config.py`, boundary test | 2,039 → 1,097 | +48 | Codex focused 882 passed / 15 skipped; Opus 288 validate + 32 resolve differentials, 0 mismatches; strict MkDocs green | clean |
