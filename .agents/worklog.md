# SIFT v1.0.1 — complete

Objective: merge PR #106 with its ancestry intact, publish GitHub release v1.0.1, and verify the actual uploaded artifacts. The user explicitly approved merge and publication on October 9, 2026. PyPI publication is excluded. Work used GPT-6.1 Sol xhigh without other workers.

Completed: PR #106 merged at `b3fe977861f5cd42e572d9a9a8ebd09cf8493f05`; `main`, `origin/main`, and tag `v1.0.1` point to that commit. Its tree matches tested PR head `ba2c8f4f8a0a546e7625514e54c61e8dbd3cda38`. Original evidence ancestors `055d9b5` and `5ffc877` and clean release-source `b615108` remain in its history. All 18 data and selection fingerprints are unchanged. Release: https://github.com/kmedved/sift/releases/tag/v1.0.1 . Published notes match the prepared dated 1.0.1 section.

Decisive verification: all six active PR jobs passed in `37996645127`; all eight manual release-gate jobs passed in `37996646928`. Release workflow `37998951299` passed, including tag/version matching, builds, Twine, installed-wheel smoke, and uploads. Automatic merged-main CI `37998936781` passed all six active jobs; the two schedule/manual-only jobs were correctly skipped. Both actual uploaded archives were downloaded; their digests match GitHub metadata, and all 85 tracked package source/typing files in each archive match the tagged tree. Both pass Twine. The published wheel clean-installed outside the checkout and passed `scripts/verify_wheel_install.py`; package and distribution metadata both report 1.0.1.

Published artifact SHA-256: wheel `a2c96d772df8577f7ef4cba4965153a260008fa822171f0d4d11f2b00d12b8fb`; source archive `72739959d58f58fe97fdb49cbc04608a18f5f99cf12bae72d198f78959c3e917`. The release assets on GitHub are the record; the local `dist/1.0.1-*` copies were removed in the 2026-10-26 repository cleanup.

Since then: `main` is `1.0.2.dev0` (opt-in `build_cache(column_batch_size=...)` and allocation trims, 2026-10-15; unreleased). The 2026-10-26 cleanup removed the merged release branches and local build leftovers and fixed three remaining edge cases; no tag, release or PyPI publication.
