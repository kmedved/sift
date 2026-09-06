# SIFT 0.9 roadmap worklog

## Objective and constraints

Complete the approved 0.9.1 closeout and ordered 0.9.x roadmap. Every implementation is now accepted; the final ordinal/frequency stage still needs integration and merge before the active unbudgeted goal is complete.

- Grok4.6 primary coder, native6942c6e1-14da-42e0-8ef2-27e77a0ab942, workspace/medium. Codex independently verifies and owns Git/CI. Opus native72bc11aa-3fc5-4cae-8b5a-9197b89f270c, read-only/xhigh, reviews concurrently with Codex between stages. Resume exact sessions, no caps/fresh sessions. If Opus quota fails, Codex works directly; no quota failure through97.
- Source transmission approved for this task; exclude secrets/unrelated repos. Use exact python3 ~/.codex/skills/grok/scripts/grok_run.py and claude-cli/scripts/claude_cli_run.py launchers. Never sandbox_permissions. Prompts in /private/tmp mode600; delete after terminal. Redact sc_token. Poll exact handles; observation timeout is not termination. No concurrent checkout writes during provider ownership; freeze review source.
- Preserve unrelated edits. No PyPI, new tag/release, old-PR or PID cleanup. Keep0.9.1.dev0 and immutablev0.9.0. Merge commits, not squash, preserve clean benchmark-source ancestry.

## Completed milestones and goal coverage

All15 scoped PR78-92 are merged; all merge commits were verified ancestors of current base47977b3. Actual committed source/tests inspected against the original goal, not only trackers:

| Requirement | Merged evidence | Decisive coverage |
| --- | --- | --- |
| 0.9.1 matrix, glossary, tutorial; router/adapter splits | PR78 / 3b9ac0a | executable26x8 matrix, glossary/links/tutorial tests; actual split-module delegation |
| F1 conditioning | PR79 / 14bfb5c | independent conditional Schur-gain oracle |
| F2 proxy/redundancy clusters | PR80 / 17fe3bf | edge values, bridging clusters, positional identity, storage cap |
| F7 within/between | PR81 / 76e4d51 | weighted group-mean and within-signal tests |
| F8a knockoff UX | PR82 / 40f8af7 | effective-group minimum-q and qualified FDR tests |
| F8b e-values | PR83 / f70a3da | literal e-value/eBH arithmetic, common-m zero padding, symmetry guards |
| F8c statistic bakeoff | PR84 / cfd2f64 | frozen480-record paired quality artifact; defaults unchanged |
| F3 blocks | PR85 / e9a4464 | joint-block gain oracle, atomic support, block-count/width/df auto-k tests |
| E4 one-hot blocks | PR86 / c5e1d51 | train-only weighted vocabulary, unknown/missing, raw/encoded output widths |
| F9 compare | PR87 / 7314355 | spies prove selector and model refit only on outer training rows |
| Manifests | PR88 / c29e99e | versioned JSON, typed identities, optional data hashing, no retainedX |
| F4 Stabilized | PR89 / 18cdf40 | frequencies match manual resampling oracle |
| F5 multi-target | PR90 / 0ee513b | joint logdet/weighted oracle, df=q*k, multi-output evaluation |
| F6 ModelSelector/purged splits | PR91 / 323da27 | shared native/generic backend, nested fit-row spies, exact purged/tied-time folds |
| Classic caches | PR92 / 47977b3 | exact result/curve parity, no target/relevance cache, duplicate-name walls |

Every listed stage passed its full local gate and all6exact-head required CI jobs before merge. F6 local2372/40, CI34035101543 on e9cbf1d; classic local2386/40, CI34038231034 on fbd0ea0. Merged-tree equality and clean-source ancestry verified. No need repeat earlier audits. No PyPI upload exists in the asset-only GitHub-release workflow.

## Final encoding implementation accepted

Branch codex/0.9x-unsupervised-encoding from47977b36b378e4f43f656e192b9aa7b9bb9efb29. Grok112 implemented,113-115 corrected reproduced defects. Grok115 terminal0 unified85832 (2026-09-06T15:19:29UTC); Opus97 terminal0 unified48919. All providers stopped, caller prompts deleted, review97 hashes unchanged before Codex tracker edits. No known correctness blocker remains.

- Additive dependency-free ordinal/frequency on existing cat_encoding APIs;66exports/defaults unchanged. Positive training-mass vocabulary, ordinal0..C-1/unknown-1, frequency proportions/unknown0; missing observed-only; y/class_weight excluded from encoding. Fixed inference maps and numeric training output.
- Actual evaluate/group/time/nested and GaussianCV/xfit fold encoders train locally; evaluate/time path maps train-only. In-sample EBIC remains in-sample; no full-path holdout-blind claim for prefix-only evaluation.
- Within and valid-column/block composition verified; Brier/logloss encoding weights remain separate from scoring/class weights; finite large weights normalize safely. Private Brier weight plumbing; public advanced override applies only to unsupervised modes.
- Explicit scoped limits remain: resampled stability/knockoff_path/consensus, prebuilt caches without encoding provenance, Boruta test-importance. No added APIs on ModelSelector/nativeCatBoost/select_fdr; no FDR upgrade.
- Codex94 found5bugs despite Opus94 no findings. Codex95 found double encoding/docstring gaps; Codex96 found override applicability. All reproduced and returned to same Grok, now closed. Reports /private/tmp/sift_unsup_codex94_findings.md through _codex96_findings.md and _opus94_report.md through _opus97_report.md.
- Final Codex97:190focused tests-Werror,4TargetCV default/None/array override cases with exact full-curve equality,13deferred/binary fold-map/weight checks. Codex96:271focused tests10skip,8fixed/nondeferred training-value checks; original within/block repros repaired. Opus97:204tests plus explicit frequency-map inheritance probe. No extra review needed absent new evidence.
- Ruff/API66/matrix/diff checks pass; strict MkDocs passed in1.47s. Full suite deliberately waits for clean-source runtime refresh, not a waived binding test.

## Integration next

1. Finish strict docs, commit accepted implementation and tracker updates.
2. Refresh runtime from that clean commit with no heavy concurrent work. Verify dirty=false/status[],18cases x7samples, all pools1, current source hashes, exact CSV binding and18unchanged data/selection fingerprints.
3. Update runtime page/table/SHA and benchmarks README; run full pytest-Werror/no exclusions, Ruff/API66/matrix/strict docs. Commit evidence.
4. Push/PR; all6required CI on exact head; merge --merge --match-head-commit; mainff, merged-tree equality and clean-source ancestry. Final original-goal audit then update_goal complete. No PyPI.

## Retained evidence and commands

Runtime (about to be refreshed): clean source8f4499985ccdc56fb205d11cc5d8a17c041b32f2, capture2026-09-06T14:01:21.110628UTC/generated14:01:50.027707UTC, dirtyfalse/status[],85hashes,18cases/7samples/pools1. benchmarks/results/runtime_scaling_2026-09-03.csv/.provenance.json, CSVshae2b2e139b913b44cfd89df458be767ddb127e9381ebea004018844d2619b604a, docsratio8.0. Previous18fingerprints unchanged.

Frozen F8c quality MUST NOT refresh: clean sourceae904b8af02037eb66cd649384c4665dba17049d, capture2026-09-05T06:32:30.226365UTC, dirtyfalse/status[],75hashes/480records/pools1. knockoff_statistic_bakeoff.csv/.provenance.json, CSVsha40d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. Preserve CRLF/historical binding and existing FDR/default caveats. Hash/source ancestry reverified.

- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python (pytest9.1.1); base /opt/anaconda3/bin/python (3.12.7, numpy1.26.4,pandas2.2.2,sklearn1.5.1,scipy1.13.1,numba0.60); docs /private/tmp/sift-docs-venv/bin/python. No optional dependency installs; native CI required.
- Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=8; PYTHONPATHcwd for scratch.
- Full tests: testpython -m pytest -q -W error, NO exclusions. Ruff testpython -m ruff check sift tests scripts. Basepython scripts/generate_api_reference.py --check and scripts/generate_data_type_matrix.py --check. Strict MkDocs into mktemp /private/tmp dir. git diff --check.
- Runtime: basepython benchmarks/bench_runtime_scaling.py --full --warmup-runs 1 --timing-repeats 7 --output benchmarks/results/runtime_scaling_2026-09-03.csv; VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 too. Audit processes before/during/after; never kill unrelated work. Ordinary desktop activity disclosed.
- CI6: test(3.10),test(3.11),test(3.12),test-catboost,min-pins,wheel-smoke. gh --repo kmedved/sift. No squash. Test/docs-only fixes need decisive checks plus CI, not another production review/runtime.
