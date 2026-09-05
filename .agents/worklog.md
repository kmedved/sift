# SIFT 0.9 roadmap worklog

## Objective and constraints

- Complete the full owner-approved 0.9.1 and ordered 0.9.x roadmap in TODO.MD and docs/specs/0.9-product-layer.md; the goal is active.
- Grok 4.6 is primary coder; Codex independently verifies; Claude Opus xhigh reviews concurrently with Codex between stages. Return accepted defects to Grok. The owner approved private SIFT source/diff/test transmission to xAI and Anthropic.
- Resume exact provider sessions in this task. No PyPI publication. Preserve unrelated edits. Codex owns commits, pushes and merges. GitHub Releases/tags remain separate owner actions.
- Grok session 6942c6e1-14da-42e0-8ef2-27e77a0ab942, workspace / medium, last completed stage 59.
- Opus session 72bc11aa-3fc5-4cae-8b5a-9197b89f270c, read-only / xhigh, last completed stage 49.
- Use installed grok_run.py / claude_cli_run.py launchers with the same cwd/envelope, no fresh session or turn cap. No provider run is active; caller prompts are removed after completion.

## Completed milestones

- 0.9.1 generated reference, decision tree, data-type matrix, glossary, tutorial and integration complete. Router/adapter split PR #78 merged at 3b9ac0a.
- F1 conditioning PR #79 merged at 14bfb5c (implementation b2a11bd, artifact 682af22).
- F2 proxy/cluster reports PR #80 merged at 17fe3bf (implementation 081ca04, artifact a495f33).
- F7 panel within/between PR #81 merged at 76e4d5164e857d74f09d77f629f1bfc77d1a42a4 (implementation 2d04754, artifact d3938e4).
- All six required GitHub CI jobs passed for each merged capability. F7 final full local gate: 2107 passed / 40 skipped under -W error, no deselections. Development remains 0.9.1.dev0; v0.9.0 is immutable at 94bae05.

## Current stage: F8a CI warning-filter correction (uncommitted)

- PR #82 head `0c62a44` failed five test jobs in run 33944942510 on two tests that used `pytest.warns(UserWarning, match="no FDR claim applies")`. Pytest 9.1.1 re-emits the additional intentional F8a knockoff+ feasibility warning (`offset=1`, `m=3`, `q=0.2`, `m*q<1`) under warnings-as-errors; local pytest 7.4.4 did not. Production behavior is correct and unchanged.
- Test-only fix: `tests/contracts/test_target_cv_encoding.py::test_knockoff_legacy_supervised_encoding_warns_and_drops_the_fdr_claim` and `tests/test_release_readiness.py::test_knockoff_fdr_claims_stay_honest_under_categorical_encoding` now require both the legacy FDR-downgrade warning and the feasibility warning. Every captured warning must be a `UserWarning` before message checks; unrelated categories are no longer filtered away. Metadata / no-FDR assertions are unchanged. No production, docs, or default changes.
- Grok58/59 CI corrections accepted by Codex and Opus48/49. Full pytest9 run: 2118 passed / 40 skipped, no deselections; after final category-assert tightening, both affected modules plus runtime binding: 146 passed. Direct injection rejects unexpected RuntimeWarning, DeprecationWarning and UserWarning in both tests. Grok verified decisive tests under pytest7 and pytest9. Source hashes and artifact binding unchanged. No remaining CI-correction blocker.
- Runtime artifact from clean `308dfe1` remains valid: only tests and this worklog change. F8b/F8c not started.
- New diagnostics distinguish actual post-screening tested units from pre-screen eligibility, grouped/representative units from reported feature counts, and not-run early returns. Offset-0 counterfactual reuses the same W without extra draws. Selections, defaults and existing FDR labels remain unchanged.
- Four accepted corrections: per-draw warnings no longer claim aggregate impossibility when group counts vary; constant-target returns no longer invent completed screened counts; encoding C4 expects only the new feasibility warning; warn_external attributes warnings to the caller.
- Codex: 120 affected tests passed under -W error; direct zero-target and cluster counterfactual probes passed. Ruff, both generators and strict MkDocs passed. Opus47: 431 passed / 10 skipped and 12 HEAD/current A/B configurations with unchanged selections/shared W/existing metadata. Review hashes unchanged. No remaining review defect.
- Review synthesis/protocol: /private/tmp/sift_f8a_review_findings.md and /private/tmp/sift_f8a_review_protocol.md. Probe: /private/tmp/sift_f8a_codex_probe.py. Frozen initial packet: /private/tmp/sift-f8a-review/.
- Final full integration run after artifact refresh: 2118 passed / 40 skipped under -W error in 65s, no deselections. This supersedes the earlier pre-artifact run's fixed C4 warning failure and deferred runtime-binding test.
- Next: commit/push this test-only CI correction on PR #82, verify all six required CI jobs at the new exact head, then merge. No further review round is needed absent a new concrete defect.

## Runtime evidence

- Current artifact binds clean F8a implementation 308dfe1ad30c052c3bcf09567dbe65322934b973, captured 2026-09-05T04:30:27.939633+00:00, dirty=false and empty status, 74 source hashes, 18 cells / 7 samples / all native pools=1. All data/selection fingerprints match the prior evidence. CSV SHA256 e3b4f82b70d0ac7e65ee6ab586aa59dcad7681913029650e1cde655da30399ac.
- Stable basename benchmarks/results/runtime_scaling_2026-09-03.csv and .provenance.json; bound displayed table/commit/hash references updated. Measured wide/baseline FDR ratio 8.1x is descriptive, not a power or asymptotic complexity claim.

## Remaining ordered roadmap

F8a integration; F8b validated e-values; F8c statistic bakeoff; F3 block selection; E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector and purged splits; classic caches; unsupervised ordinal/frequency fallbacks. Do not mark the overall goal complete at F8a.

## Tooling

- Test/benchmark Python /opt/anaconda3/bin/python (3.12.7, NumPy 1.26.4, pandas 2.2.2, sklearn 1.5.1); LOKY_MAX_CPU_COUNT=8, all native thread limits=1. Local environment lacks CatBoost/category_encoders; CI covers them.
- Prefer /private/tmp/sift-pytest9.5nI4QH/bin/python for test runs (pytest9.1.1 with the same base scientific dependencies); base/docs Python has pytest7.4.4 and can hide unmatched pytest.warns warnings. Base environment was not modified. Keep benchmark Python unchanged for provenance continuity.
- Docs Python /private/tmp/sift-docs-venv/bin/python; strict build output /private/tmp/sift-f8a-review-site. Use PYTHONPATH="$PWD" for /private/tmp probe scripts.
- Specify --repo kmedved/sift for gh commands (there is an upstream remote). Merge with --merge to retain implementation ancestry for source provenance, after exact-head CI passes.
