# SIFT 0.9 roadmap worklog

## Objective and constraints

- Complete the full owner-approved 0.9.1 and ordered 0.9.x roadmap in TODO.MD and docs/specs/0.9-product-layer.md; the goal is active.
- Grok 4.6 is primary coder; Codex independently verifies; Claude Opus xhigh reviews concurrently with Codex between stages. Return accepted defects to Grok. The owner approved private SIFT source/diff/test transmission to xAI and Anthropic.
- Resume exact provider sessions in this task. No PyPI publication. Preserve unrelated edits. Codex owns commits, pushes and merges. GitHub Releases/tags remain separate owner actions.
- Grok session 6942c6e1-14da-42e0-8ef2-27e77a0ab942, workspace / medium, last completed stage 57.
- Opus session 72bc11aa-3fc5-4cae-8b5a-9197b89f270c, read-only / xhigh, last completed stage 47.
- Use installed grok_run.py / claude_cli_run.py launchers with the same cwd/envelope, no fresh session or turn cap. No provider run is active; caller prompts are removed after completion.

## Completed milestones

- 0.9.1 generated reference, decision tree, data-type matrix, glossary, tutorial and integration complete. Router/adapter split PR #78 merged at 3b9ac0a.
- F1 conditioning PR #79 merged at 14bfb5c (implementation b2a11bd, artifact 682af22).
- F2 proxy/cluster reports PR #80 merged at 17fe3bf (implementation 081ca04, artifact a495f33).
- F7 panel within/between PR #81 merged at 76e4d5164e857d74f09d77f629f1bfc77d1a42a4 (implementation 2d04754, artifact d3938e4).
- All six required GitHub CI jobs passed for each merged capability. F7 final full local gate: 2107 passed / 40 skipped under -W error, no deselections. Development remains 0.9.1.dev0; v0.9.0 is immutable at 94bae05.

## Current stage: F8a accepted, integration pending

- Branch codex/0.9x-f8a-knockoff-ux, base 76e4d51. Grok56 implementation and Grok57 corrections are accepted by Codex and Opus47; uncommitted. F8b/F8c not started.
- New diagnostics distinguish actual post-screening tested units from pre-screen eligibility, grouped/representative units from reported feature counts, and not-run early returns. Offset-0 counterfactual reuses the same W without extra draws. Selections, defaults and existing FDR labels remain unchanged.
- Four accepted corrections: per-draw warnings no longer claim aggregate impossibility when group counts vary; constant-target returns no longer invent completed screened counts; encoding C4 expects only the new feasibility warning; warn_external attributes warnings to the caller.
- Codex: 120 affected tests passed under -W error; direct zero-target and cluster counterfactual probes passed. Ruff, both generators and strict MkDocs passed. Opus47: 431 passed / 10 skipped and 12 HEAD/current A/B configurations with unchanged selections/shared W/existing metadata. Review hashes unchanged. No remaining review defect.
- Review synthesis/protocol: /private/tmp/sift_f8a_review_findings.md and /private/tmp/sift_f8a_review_protocol.md. Probe: /private/tmp/sift_f8a_codex_probe.py. Frozen initial packet: /private/tmp/sift-f8a-review/.
- Earlier pre-artifact full run: 2113 passed / 40 skipped / 1 failed (encoding C4 warning expectation, now fixed) / 1 deselected (runtime binding deferred). Do not report that as a completed full gate.
- Next: commit reviewed implementation and status docs, run runtime benchmark from that clean commit without concurrent test/reviewer workloads, update bound artifact/table/provenance references, run final full suite without deselections, commit artifact, push PR, verify six required CI jobs, merge.

## Runtime evidence

- Current artifact still binds F7 clean implementation 2d047549ec7bdd1fb9f4fdd303b07251b84b5eb6, dirty=false, 74 source hashes, 18 cells / 7 samples / native pools=1; CSV SHA256 e3bd31662ed54efec6e91540477b41e4b8eead5294cbfd60f0d25e0d6a1be545. Must refresh for F8a before final full gate.
- Stable artifact basename benchmarks/results/runtime_scaling_2026-09-03.csv and .provenance.json. Run --full --warmup-runs 1 --timing-repeats 7 from clean implementation commit; update docs/runtime-scaling.md and commit/hash references. Compare all 18 data/selection fingerprints against the previous artifact.

## Remaining ordered roadmap

F8a integration; F8b validated e-values; F8c statistic bakeoff; F3 block selection; E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector and purged splits; classic caches; unsupervised ordinal/frequency fallbacks. Do not mark the overall goal complete at F8a.

## Tooling

- Test/benchmark Python /opt/anaconda3/bin/python (3.12.7, NumPy 1.26.4, pandas 2.2.2, sklearn 1.5.1); LOKY_MAX_CPU_COUNT=8, all native thread limits=1. Local environment lacks CatBoost/category_encoders; CI covers them.
- Docs Python /private/tmp/sift-docs-venv/bin/python; strict build output /private/tmp/sift-f8a-review-site. Use PYTHONPATH="$PWD" for /private/tmp probe scripts.
- Specify --repo kmedved/sift for gh commands (there is an upstream remote). Merge with --merge to retain implementation ancestry for source provenance, after exact-head CI passes.
