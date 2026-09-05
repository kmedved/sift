# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: complete the owner-approved 0.9.1 closeout and ordered 0.9.x roadmap in TODO.MD and docs/specs/0.9-product-layer.md.
- Grok4.6 is primary coder; Codex independently verifies; Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic is approved.
- Resume exact sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium, completed stage63; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c read-only/xhigh, completed stage53. No provider active; caller prompts removed. Installed launchers, same cwd/envelope, no fresh session/turn cap. Redact incidental sc_token output.
- Codex owns commits/push/merge. Preserve unrelated changes. No PyPI publication; GitHub Releases/tags remain separate owner actions. Development0.9.1.dev0; v0.9.0 immutable94bae05.

## Completed milestones

- 0.9.1 docs/matrix/glossary/tutorial/integration and router split: PR78 merged3b9ac0a.
- F1 conditioning: PR79 merged14bfb5c (implementationb2a11bd, artifact682af22).
- F2 proxy/cluster reports: PR80 merged17fe3bf (implementation081ca04, artifacta495f33).
- F7 panel transforms: PR81 merged76e4d51 (implementation2d04754, artifactd3938e4).
- F8a knockoff UX: PR82 merged40f8af77cccebf5b1842e932cda0a9464dae38fa (implementation308dfe1, artifact0c62a44, test-only CI correction449f063).
- All six required GitHub CI jobs passed for each milestone. F8a full local gate2118 passed/40 skipped under pytest7 and pytest9. Pytest9 exposed unmatched warning cases fixed in449f063; preserve strict warning categories/messages.

## Current stage: F8b integration

- Branch codex/0.9x-f8b-evalues, base40f8af7; reviewed implementation committed at dde1f50b2f9aa68539d5bd7c25003cc9cf8e1027. Refreshed runtime evidence/docs are ready to commit. Remote main verified still40f8af7. No unrelated changes.
- Opt-in aggregation="evalues" requires n_draws>1/offset1; common tested universe m, zero padding, averaging, e-BH, wrapper/results/rankings/views and provenance implemented. Omitted-option selections/draws/metadata remain unchanged.
- Validated configurations: ungrouped relevance/ridge, fixed-before-statistics universe, no inherited downgrade; retain approximate_plugin, aggregate-null expectation bound only. CEFS+/LSM, grouped/representative expansion, varying screening unions and supervised encodings are exploratory as applicable. Per-draw validity distinguishes invalid statistics from aggregate-only screening downgrades.
- Codex/Opus final review accepted. Six primary corrections, two metadata propagation fixes and the legacy recursion guard are closed. Saturating CEFS+ and tied/truncated LSM violate sign-flip; legacy statistics were not rewritten, but new e-value validation excludes them. Encoded nested representative results are downgraded only for opted-in e-value runs. Excluded group members remain unselected with zero evidence.
- Review artifacts: /private/tmp/sift_f8b_review_findings.md and /private/tmp/sift_f8b_review_protocol.md; initial packet /private/tmp/sift-f8b-review/. Probes: /private/tmp/sift_f8b_codex_probe.py, /private/tmp/sift_f8b_adaptive_data_probe.py, /private/tmp/sift_f8b_parity_oracle.py, /private/tmp/sift_f8b_legacy_wrapper_probe.py.
- Evidence: 20 focused F8b tests pass; 12 baseline function cases + actual baseline wrapper parity and 4 public literal arithmetic cases pass. Opus53 final targeted78 passed/6 skipped. Earlier full pre-artifact run2135 passed/40 skipped/1 runtime-binding deselection (before last two tests). All reviewer checkout hash manifests unchanged. Ruff/generators/strict docs pass. No remaining review defect.
- Final local integration gate: 2138 passed / 40 skipped under pytest9/-W error, no deselections, in65s. Runtime binding, both generators, strict MkDocs, Ruff and whitespace checks pass. This supersedes the pre-artifact run. All18 data/selection fingerprints match the preceding artifact.
- Next: commit refreshed evidence/status, push PR, verify all six required exact-head CI jobs, merge with merge commit. No F8c implementation yet.

## Runtime evidence

- Current artifact binds clean F8b dde1f50b2f9aa68539d5bd7c25003cc9cf8e1027, captured2026-09-05T05:49:34.274465+00:00; dirty=false, status[], 74 source hashes, 18 cells/7 samples/all pools1. CSV SHA25662a00e7abc2d6b53ace375158250d24d858a69e4d1d8e8d415a268e93daa84ee.
- Stable basename benchmarks/results/runtime_scaling_2026-09-03.csv and .provenance.json. Runtime binding guard is live and passes. No concurrent tests/providers during benchmark. All18 data/selection fingerprints unchanged; wide/baseline FDR ratio8.0x is descriptive, not a quality or asymptotic claim.

## Remaining ordered roadmap

F8b integration; F8c statistic bakeoff; F3 blocks; E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks. Keep overall goal active at intermediate milestones.

## Tooling

- Tests: /private/tmp/sift-pytest9.5nI4QH/bin/python (pytest9.1.1, system scientific deps). Base /opt/anaconda3/bin/python: Python3.12.7, NumPy1.26.4/pandas2.2.2/sklearn1.5.1, pytest7.4.4. Keep base environment unchanged; benchmark/ruff/generators use base Python. No local CatBoost/category_encoders; CI covers optional deps.
- Native thread pools1; LOKY_MAX_CPU_COUNT8. PYTHONPATH="$PWD" for /private/tmp probes.
- Docs: /private/tmp/sift-docs-venv/bin/python; strict MkDocs output /private/tmp/sift-f8b-review-site.
- Always gh --repo kmedved/sift and explicit PR number. Merge with --merge after exact-head CI; retain implementation ancestry for runtime provenance.
