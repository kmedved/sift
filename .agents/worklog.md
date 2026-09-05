# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: complete the owner-approved 0.9.1 closeout and ordered 0.9.x roadmap in TODO.MD and docs/specs/0.9-product-layer.md.
- Grok4.6 is primary coder; Codex independently verifies; Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic is approved.
- Resume exact sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium, completed stage65; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c read-only/xhigh, completed stage55. No provider active; caller prompts removed. Installed launchers, same cwd/envelope, no fresh session/turn cap. Redact incidental sc_token output.
- Codex owns commits/push/merge. Preserve unrelated changes. No PyPI publication; GitHub Releases/tags remain separate owner actions. Development0.9.1.dev0; v0.9.0 immutable94bae05.

## Completed milestones

- 0.9.1 docs/matrix/glossary/tutorial/integration and router split: PR78 merged3b9ac0a.
- F1 conditioning: PR79 merged14bfb5c (implementationb2a11bd, artifact682af22).
- F2 proxy/cluster reports: PR80 merged17fe3bf (implementation081ca04, artifacta495f33).
- F7 panel transforms: PR81 merged76e4d51 (implementation2d04754, artifactd3938e4).
- F8a knockoff UX: PR82 merged40f8af77cccebf5b1842e932cda0a9464dae38fa (implementation308dfe1, artifact0c62a44, test-only CI correction449f063).
- F8b e-value aggregation: PR83 merged f70a3da66fd6d2d8dbefefd0f17c045bd260826f (implementation dde1f50, artifact/status 2a05c66). All six required jobs passed at exact head2a05c66 in run33948407577. Local main fast-forwarded; merged tree exactly matches reviewed/tested head.
- All six required GitHub CI jobs passed for each milestone. F8a full local gate2118 passed/40 skipped under pytest7 and pytest9. Pytest9 exposed unmatched warning cases fixed in449f063; preserve strict warning categories/messages.

## Latest completed stage: F8b

- Reviewed implementation dde1f50b2f9aa68539d5bd7c25003cc9cf8e1027 and refreshed evidence2a05c66b73b95e2550c72c83b123967835145ca6 merged as PR83. No remaining integration blocker.
- Opt-in aggregation="evalues" requires n_draws>1/offset1; common tested universe m, zero padding, averaging, e-BH, wrapper/results/rankings/views and provenance implemented. Omitted-option selections/draws/metadata remain unchanged.
- Validated configurations: ungrouped relevance/ridge, fixed-before-statistics universe, no inherited downgrade; retain approximate_plugin, aggregate-null expectation bound only. CEFS+/LSM, grouped/representative expansion, varying screening unions and supervised encodings are exploratory as applicable. Per-draw validity distinguishes invalid statistics from aggregate-only screening downgrades.
- Codex/Opus final review accepted. Six primary corrections, two metadata propagation fixes and the legacy recursion guard are closed. Saturating CEFS+ and tied/truncated LSM violate sign-flip; legacy statistics were not rewritten, but new e-value validation excludes them. Encoded nested representative results are downgraded only for opted-in e-value runs. Excluded group members remain unselected with zero evidence.
- Review artifacts: /private/tmp/sift_f8b_review_findings.md and /private/tmp/sift_f8b_review_protocol.md; initial packet /private/tmp/sift-f8b-review/. Probes: /private/tmp/sift_f8b_codex_probe.py, /private/tmp/sift_f8b_adaptive_data_probe.py, /private/tmp/sift_f8b_parity_oracle.py, /private/tmp/sift_f8b_legacy_wrapper_probe.py.
- Evidence: 20 focused F8b tests pass; 12 baseline function cases + actual baseline wrapper parity and 4 public literal arithmetic cases pass. Opus53 final targeted78 passed/6 skipped. Earlier full pre-artifact run2135 passed/40 skipped/1 runtime-binding deselection (before last two tests). All reviewer checkout hash manifests unchanged. Ruff/generators/strict docs pass. No remaining review defect.
- Final local integration gate: 2138 passed / 40 skipped under pytest9/-W error, no deselections, in65s. Runtime binding, both generators, strict MkDocs, Ruff and whitespace checks pass. This supersedes the pre-artifact run. All18 data/selection fingerprints match the preceding artifact.
- Completed commit/push, exact-head CI verification and merge. No further F8b review or runtime refresh needed without new evidence.

## Current stage: F8c accepted implementation, retained run next

- Branch `codex/0.9x-f8c-statistic-bakeoff`, base `f70a3da`. Grok64/65 wrote the public-API bakeoff and fixed all5 review findings. Codex and resumed Opus54/55 independently accept; checkout hashes unchanged through both reviews. Protocol/findings: /private/tmp/sift_f8c_review_protocol.md and /private/tmp/sift_f8c_codex_findings.md. No production statistic/default change.
- Fixed study: independent/AR1/block/dense-weak × relevance/lsm/ridge/cefsplus, n800/p40, seeds0–29, q0.1/offset1/equi/one draw, unchanged statistic options; one warmup + one timed call. Paired data and actual knockoff draws verified. Descriptive FDP/power/cost and SEs, not a validity proof; retain F8b LSM/CEFS+ caveats.
- Evidence fixes: immutable pre-run source/environment, end-source/commit guard,75 source hashes, persisted selected indices/timing samples/effective pools, phase-aware warnings including failures, strict finite JSON. Historical/current LSM wording corrected. No retained full artifact yet.
- Verification: full pytest9/-W error2148 passed/40 skipped in65s; Ruff, both generators, strict MkDocs, existing runtime binding and whitespace pass. Optional redundant guard tests deferred; independent live probes pass.
- Next: commit implementation; run `benchmarks/bench_knockoff_statistic_bakeoff.py --full --output benchmarks/results/knockoff_statistic_bakeoff.csv` from clean commit without concurrent providers/tests. Then retained artifact/report binding and scoped1.0 recommendation, full CI and merge. Existing runtime_scaling evidence is unchanged and valid.

## Runtime evidence

- Current artifact binds clean F8b dde1f50b2f9aa68539d5bd7c25003cc9cf8e1027, captured2026-09-05T05:49:34.274465+00:00; dirty=false, status[], 74 source hashes, 18 cells/7 samples/all pools1. CSV SHA25662a00e7abc2d6b53ace375158250d24d858a69e4d1d8e8d415a268e93daa84ee.
- Stable basename benchmarks/results/runtime_scaling_2026-09-03.csv and .provenance.json. Runtime binding guard is live and passes. No concurrent tests/providers during benchmark. All18 data/selection fingerprints unchanged; wide/baseline FDR ratio8.0x is descriptive, not a quality or asymptotic claim.

## Remaining ordered roadmap

F8c statistic bakeoff; F3 blocks; E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks. Keep overall goal active at intermediate milestones.

## Tooling

- Tests: /private/tmp/sift-pytest9.5nI4QH/bin/python (pytest9.1.1, system scientific deps). Base /opt/anaconda3/bin/python: Python3.12.7, NumPy1.26.4/pandas2.2.2/sklearn1.5.1, pytest7.4.4. Keep base environment unchanged; benchmark/ruff/generators use base Python. No local CatBoost/category_encoders; CI covers optional deps.
- Native thread pools1; LOKY_MAX_CPU_COUNT8. PYTHONPATH="$PWD" for /private/tmp probes.
- Docs: /private/tmp/sift-docs-venv/bin/python; strict MkDocs output /private/tmp/sift-f8b-review-site.
- Always gh --repo kmedved/sift and explicit PR number. Merge with --merge after exact-head CI; retain implementation ancestry for runtime provenance.
