# SIFT 0.9 roadmap worklog

## Objective and constraints

- Complete the full owner-approved 0.9.1 and ordered 0.9.x roadmap in TODO.MD and docs/specs/0.9-product-layer.md; the goal is active.
- Grok 4.6 is primary coder; Codex independently verifies; Claude Opus xhigh reviews concurrently with Codex between stages. Return accepted defects to Grok. The owner approved private SIFT source/diff/test transmission to xAI and Anthropic; the approval blocker is resolved.
- Resume the exact provider sessions in this Codex task. No PyPI publication. Preserve unrelated edits. Codex owns commits, pushes and merges.
- Grok session: 6942c6e1-14da-42e0-8ef2-27e77a0ab942, workspace / medium, last completed stage 55.
- Opus session: 72bc11aa-3fc5-4cae-8b5a-9197b89f270c, read-only / xhigh, last completed stage 45.
- Use the installed grok_run.py and claude_cli_run.py launchers with the same cwd and envelope; do not start fresh.

## Completed milestones

- 0.9.1 generated reference, selector decision tree, data-type matrix, glossary, tutorial, clean runtime evidence and integration are complete.
- Router/adapter split merged PR #78 at 3b9ac0a.
- F1 conditioning merged PR #79 at 14bfb5c109b75f9e79ceed85d02562a8520d3af3. Implementation b2a11bd, runtime artifact 682af22. All six required GitHub CI jobs passed; verified live.
- F2 proxy/redundancy and cluster reports merged PR #80 at 17fe3bfd541202db4d87727fdeed789fb976caa6. Implementation 081ca04, runtime artifact a495f33. All six required CI jobs passed; main fast-forwarded locally.
- Current development version remains 0.9.1.dev0; v0.9.0 is immutable at 94bae05.

## Current stage: F7 integration

- Branch codex/0.9x-f7-panel-transforms, base 17fe3bf. Grok's F7 implementation and corrections are accepted by Codex and resumed Opus stage 45; no code defect remains from those reviews. Not yet committed or merged.
- Public regression filters/wrappers support weighted within='groups' and five-iteration within='two_way', before ranks. Evaluate, gaussian_cv and xfit_objective use train-only means; unseen groups use the training grand mean. Views expose within/between relevance; transform output and omitted-option behavior are unchanged. Classification, prebuilt cache, nested and non-fold auto-k combinations explicitly reject within.
- Four verified corrections: stable anchored group means (no false large-offset/group-constant signal); datetime/timedelta rejection before conversion; proxy screening uses demeaned y; direct select_k_auto validates the new option/task/context. No absolute constant threshold added. Small signals at 1e-13 still select correctly.
- Codex: 338 affected passed / 2 skipped; independent weighted-selection/zero-row/raw-transform probe and six group/two-way CV arithmetic folds passed. Opus: 452 passed / 11 skipped and all four targeted repros fixed. Source hashes unchanged during review.
- Full pre-integration run: 2105 passed / 40 skipped, one glossary-order failure, one artifact-freshness check deferred. Grok stage 55 fixed only the glossary ordering; its seven tests pass. Ruff, generators and strict MkDocs passed. Final full run follows artifact refresh.
- Review synthesis/protocol: /private/tmp/sift_f7_review_findings.md and /private/tmp/sift_f7_review_protocol.md. Codex probe scripts: /private/tmp/sift_f7_codex_probe.py and /private/tmp/sift_f7_cv_probe.py.
- Next: commit accepted implementation, refresh runtime from that clean commit, update artifact table/source references, full tests/CI, then merge. F8 code has not started; /private/tmp/sift_f8_reference_notes.md records a primary-source lookup for the next stage.

## Remaining ordered roadmap

After F7 integration: F8a knockoff UX, F8b validated e-values, F8c statistic bakeoff; F3 block selection; E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector and purged splits; classic-selector caches; unsupervised ordinal/frequency fallbacks. Do not mark the overall goal complete at F7.

## Runtime and tooling

- Test/benchmark Python: /opt/anaconda3/bin/python (3.12.7; NumPy 1.26.4, pandas 2.2.2, sklearn 1.5.1). Set LOKY_MAX_CPU_COUNT=8 and native thread limits to 1 for tests.
- Docs Python: /private/tmp/sift-docs-venv/bin/python; strict build output /private/tmp/sift-f7-review-site. Base test Python lacks pymdownx.
- Runtime runner: benchmarks/bench_runtime_scaling.py --full --warmup-runs 1 --timing-repeats 7 --output benchmarks/results/runtime_scaling_2026-09-03.csv. Stable basename; source commit and dirty=false must be captured before measurement/artifact writes.
- Specify --repo kmedved/sift on gh commands because an upstream remote also exists.
