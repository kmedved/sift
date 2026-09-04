# Baseline — 2026-09-04-router-adapter-split-pass-1

- Captured: 2026-09-04
- Branch: `codex/0.9x-router-splits`
- Base commit: `237063195707c3a45571b8c59824b29f52665f61`
- Scope: behavior-preserving extraction from `selection/view.py`,
  `selection/filter_auto_k.py`, and `selection/auto_k.py`

## Test suite

- Command: `LOKY_MAX_CPU_COUNT=8 /opt/anaconda3/bin/python -W error -m pytest -q`
- Result: **1,982 passed / 0 failed / 40 skipped** in 118.10 seconds
- Output summary: `tests_before.txt`

## Source size and complexity

Complexity is a local AST branch-count proxy (`1 +` branches, Boolean operands,
comprehensions, and match/try/with constructs), used only for before/after
comparison in this pass.

| module | lines | functions | mean proxy CC | max proxy CC |
| --- | ---: | ---: | ---: | ---: |
| `sift/selection/view.py` | 3,113 | 68 | 9.37 | 52 (`SelectionView.__init__`) |
| `sift/selection/filter_auto_k.py` | 2,139 | 45 | 5.27 | 14 (`select_gaussian_auto_path`) |
| `sift/selection/auto_k.py` | 2,587 | 42 | 7.60 | 103 (`validate_auto_k_config`) |
| **total** | **7,839** | **155** | — | — |

Direct repository files importing or referring to the modules: `view.py` 11,
`filter_auto_k.py` 7, and `auto_k.py` 30. The detailed public/internal import
and monkeypatch seams are in `duplication_map.md`.

## Lint and warnings

- Command: `/opt/anaconda3/bin/python -m ruff check sift tests scripts benchmarks`
- Result: **all checks passed**
- The complete suite was run under `-W error`; warning ceiling is zero.

## Behavioral goldens

The repository's existing contract tests are the golden oracle for this
library-only refactor. In particular, `tests/test_selection_view.py`,
`tests/test_auto_k_v2.py`, `tests/contracts/test_autok_ergonomics.py`, and
`tests/test_select_k_auto_no_target_leak.py` pin result schemas, route choices,
warnings, errors, selection order, and the original-module monkeypatch seams.
Each extraction must run its focused slice plus the complete suite.
