# SIFT 0.9 roadmap worklog

## Objective and constraints

- Complete the full owner-approved 0.9.1 and ordered 0.9.x roadmap in TODO.MD and docs/specs/0.9-product-layer.md; the goal is active.
- Grok 4.6 is primary coder; Codex independently verifies; Claude Opus xhigh reviews concurrently with Codex between stages. Return accepted defects to Grok. The owner approved private SIFT source/diff/test transmission to xAI and Anthropic; the approval blocker is resolved.
- Resume the exact provider sessions in this Codex task. No PyPI publication. Preserve unrelated edits. Codex owns commits, pushes and merges.
- Grok session: 6942c6e1-14da-42e0-8ef2-27e77a0ab942, workspace / medium, last completed stage 52.
- Opus session: 72bc11aa-3fc5-4cae-8b5a-9197b89f270c, read-only / xhigh, last completed stage 43.
- Use the installed grok_run.py and claude_cli_run.py launchers with the same cwd and envelope; do not start fresh.

## Completed milestones

- 0.9.1 generated reference, selector decision tree, data-type matrix, glossary, tutorial, clean runtime evidence and integration are complete.
- Router/adapter split merged PR #78 at 3b9ac0a.
- F1 conditioning merged PR #79 at 14bfb5c109b75f9e79ceed85d02562a8520d3af3. Implementation b2a11bd, runtime artifact 682af22. All six required GitHub CI jobs passed; verified live.
- Current development version remains 0.9.1.dev0; v0.9.0 is immutable at 94bae05.

## Current stage: F2 integration

- Branch codex/0.9x-f2-proxy-reports, base 14bfb5c. F2 implementation is committed at 081ca04552c0d662691a968b64455bea141d2ba8 and accepted by independent Codex plus Opus review; CI/merge pending.
- Additions: SelectionView.redundancy_report / proxy_clusters; opt-in StabilitySelector(store_proxies=True), actual 16 MiB bool indicator cap, 64 MiB float32 candidate-by-selected proxy cap checked before rank/correlation work, direct p-by-k Gram, no training X retained.
- Accepted fixes verified: selected-selected cluster edges; truthful memory accounting; stale/absent proxy availability; exact variation and selected constants; positive-weight filtering before proxy imputation; no extra default-path diagnostic matrix.
- Both original and missing-value zero-weight probes pass. Graph oracle: 0 mismatches / 200. Direct weighted block matches the full oracle exactly after float32 storage.
- Opus stage 42 found no additional correctness defects; stage 43 accepted the final imputation/memory correction. Optional stale-block wording was not changed: a direct probe confirms refitting at the updated threshold with store_proxies=True restores availability.
- Verification: final full integration run 2085 passed / 40 skipped under -W error, including runtime source/artifact/table freshness. F2 has 31 tests; 113 affected tests also passed. Opus final affected slice 160 passed / 1 skipped, docs examples 172 passed / 12 skipped. Repo Ruff, both generators and strict MkDocs passed.
- Runtime evidence refreshed from clean 081ca04552c0d662691a968b64455bea141d2ba8: dirty=false, empty status, 73 source hashes, 18 rows with 7 samples each, native threads all 1, all selection/data fingerprints unchanged. CSV SHA-256 21728e50c161260abc4e17cf82204d390e9363774f225539e505182e55c602d1.
- Next action: commit the refreshed artifact/docs, push PR, run required GitHub CI and merge. Then F7.

## Remaining ordered roadmap

F7 within/between transforms; F8a knockoff UX, F8b validated e-values, F8c statistic bakeoff; F3 block selection; E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector and purged splits; classic-selector caches; unsupervised ordinal/frequency fallbacks. Do not mark the overall goal complete at F2.

## Runtime and tooling

- Test/benchmark Python: /opt/anaconda3/bin/python (3.12.7; NumPy 1.26.4, pandas 2.2.2, sklearn 1.5.1). Set LOKY_MAX_CPU_COUNT=8 and native thread limits to 1 for tests.
- Docs Python: /private/tmp/sift-docs-venv/bin/python; strict build output /private/tmp/sift-f2-docs.YX4vkk. Base test Python lacks pymdownx.
- Runtime runner: benchmarks/bench_runtime_scaling.py --full --warmup-runs 1 --timing-repeats 7 --output benchmarks/results/runtime_scaling_2026-09-03.csv. Stable basename; source commit and dirty=false must be captured before measurement/artifact writes.
- Specify --repo kmedved/sift on gh commands because an upstream remote also exists.
