# SIFT v1.0.2 release — in progress

Objective: publish GitHub release v1.0.2 from `main` as requested by the owner on 2026-10-26, with the same gates as v1.0.1. PyPI publication is excluded.

Scope: 1.0.2 ships the opt-in `build_cache(column_batch_size=...)` and allocation trims (2026-10-15) and the three edge-case fixes merged in PR #107 (`6d8acc7`). The release-preparation branch sets version 1.0.2, dates the notes, re-measures the runtime evidence from its clean release-source commit, and runs the release checklist in `docs/development.md` before merging with a merge commit and publishing.

Previous release: v1.0.1 (2026-10-09) merged as `b3fe977` via PR #106; tag `v1.0.1` points there. Published artifact SHA-256: wheel `a2c96d772df8577f7ef4cba4965153a260008fa822171f0d4d11f2b00d12b8fb`; source archive `72739959d58f58fe97fdb49cbc04608a18f5f99cf12bae72d198f78959c3e917`.
