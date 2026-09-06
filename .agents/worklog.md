# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: finish approved0.9.1 closeout and ordered0.9.x roadmap in TODO.MD/docs/specs/0.9-product-layer.md. Keep goal active at intermediate milestones.
- Grok4.6 primary coder, Codex independently verifies, Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic approved.
- Resume exact native sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium, completed75; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c read-only/xhigh, completed65. Next coding stage76 for E4 after F3 integration; next reviewer66. No fresh session/turn cap. Grok compacted during67/70/74. Launchers: python3 ~/.codex/skills/grok/scripts/grok_run.py and ~/.codex/skills/claude-cli/scripts/claude_cli_run.py. Redact incidental sc_token output.
- No checkout review/edits during writer; freeze during review. Persist future hash baselines outside checkout, not only transient tool store. Codex owns Git/CI. Preserve unrelated changes. No PyPI; tags/GitHub Releases separate owner actions. Development0.9.1.dev0, v0.9.0 immutable94bae05.

## Completed milestones

- 0.9.1 docs/matrix/glossary/tutorial/integration and router split: PR78 merged3b9ac0a (implementationfa74d63).
- F1 conditioning: PR79 merged14bfb5c (implementationb2a11bd, artifact682af22).
- F2 proxy/cluster reports: PR80 merged17fe3bf (implementation081ca04, artifacta495f33).
- F7 panel transforms: PR81 merged76e4d51 (implementation2d04754, artifactd3938e4).
- F8a knockoff UX: PR82 merged40f8af7 (implementation308dfe1, artifact0c62a44, test449f063).
- F8b e-values: PR83 mergedf70a3da (implementationdde1f50, artifact2a05c66). Validated opt-in ungrouped relevance/ridge fixed universe; CEFS+/LSM/grouped/varying-screen/supervised-encoding modes explicitly exploratory. No default or FDR validity upgrades.
- F8c bakeoff: PR84 mergedcfd2f645ee5404fb3cb85be0a69cde3e7e699a26 (implementationae904b8, artifact/report0b74737, CRLF40bec7d, ULP-test20d07d1). All6 required jobs passed at exact20d07d18f0ae782db0df8ebc33c1bed4fffe5363, run33951312155; merged tree matched tested head. Keep relevance default on the committed4-DGP grid; not mixed-sign/suppressor/p>>n evidence.

## F3: implementation and independent reviews accepted; integration pending

- Branch codex/0.9x-f3-feature-blocks, basecfd2f645ee5404fb3cb85be0a69cde3e7e699a26. Reviewed implementation committed2586a2cf0c8fdc4ee57d8d55e2a773d5ba7463bd (one trailing test-file blank line removed before final clean commit). Runtime refreshed from that clean commit; evidence/status commit next. No known in-scope correctness blocker. No provider run active for this task.
- Fixed-k: joint Gaussian residual log-det; classic mRMR/JMI/JMIM max-over-member configured estimators; atomic screening/pruning/F1/proxy panels; knockoff alias. Filter auto means {block}__{level}; knockoff auto retains correlation clustering. Constant members expand honestly; store_proxies raises if selected correlations are unavailable.
- Auto-k: auto/evaluate/elbow/penalized_objective/gaussian_cv/xfit_objective use additional-block prefixes. Model rank separate from eligible-block multiplicity: EBIC d log n+2gamma log C(B,k); RIC block adaptation2d log B. Copula rank excludes constants/duplicates and conditions on includes. Nested honors effective fit overrides including None, real labels and usable fold prefixes; calibrated multimember routes explicitly unsupported. Scalar/identity numerical paths preserved.
- Binary log-loss: actual joint conditional logistic score with cross-member Fisher/nuisance penalty-gradient adjustment; atomic blocks and fixed/auto/evaluate/elbow/penalized prefixes. Weighted logistic-design rank, not copula transforms. Refit cadence in additional-block steps. Gaussian CV/xfit/calibrated binary routes unsupported; brier delegates Gaussian. No new calibration/FDR guarantee.
- Metadata: every supplied block map including identity uses additional-discovery k/n_blocks_selected/selected_blocks; n_blocks_selected_total includes forced blocks; n_columns_selected and view.k raw width. No-block legacy metadata k unchanged. Identity parity means selections/scores/cadence; explicit documented metadata difference.
- Grok74 fixed4 Codex findings: unsupported Sphinx role; rounded weighted constants falsely valid in genuine blocks; expected constant padding falsely counted as refit failure; include blocks inflate discovery metadata. Grok75 moved invalid-include validation before all-invalid early return and documented/pinned identity metadata semantics. Exact positive-weight constancy scoped to genuine blocks; scalar/identity legacy weighted constancy unchanged intentionally.
- Codex evidence:80 independent Gaussian determinant comparisons max8.88e-16;13 fixed-k and10 auto-k acceptance probes pass;72 binary row/Gram vs full-quadratic comparisons max3.91e-14; weighted scaling/zero rows/rank/refit/prefix/labels checks pass. Original four correction acceptance plus all-invalid include/empty-path repro pass.
- Final local gate after75 and clean runtime refresh:2200 passed/40 skipped, no deselections,67.72s, pytest9/-W error. Ruff, both generators, strict docs, whitespace, current runtime binding and historical F8c quality evidence all pass. Earlier57 affected/doc-reference tests plus exact public repro also pass.
- Opus65 targeted acceptance no findings:232 passed/3 skipped, independent validation-order/partial-validity/weighted/scalar/identity metadata comparisons. Opus63/64 accepted core math and corrections. Reviews63/64 tracked+untracked hashes independently unchanged; review65 reports no writes, but transient hash baseline was lost during host/tool-state pause, so no independent hash-equality claim for65. Final source behavior/code checked; do not replay completed review for bookkeeping.
- Evidence outside checkout: /private/tmp/sift_f3_binary_codex_findings73.md, ..._followon74.md, ..._codex_accept74.py, ..._codex_math73.py; /private/tmp/sift_f3_codex_accept.py (total-count assertion updated for corrected contract), ..._auto_codex_accept71.py. Native Opus final65 is in its session JSONL. Optional repeated docstring clause observation rejected as unnecessary; result documentation already explicit.
- Next: commit refreshed evidence/status, push/open integration PR, wait for all6 required jobs at exact head, merge preserving clean-source ancestry. Then E4. Local final gate complete; do not repeat it without a code/contract change.

## Retained evidence

- Frozen F8c quality: sourceae904b8af02037eb66cd649384c4665dba17049d, captured2026-09-05T06:32:30.226365+00:00, dirty=false/status[],75 hashes;480 records, pools1, no failures/warnings. benchmarks/results/knockoff_statistic_bakeoff.csv/.provenance.json; SHA25640d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. Preserve CRLF .gitattributes and historical-source binding. Summary-only floats allow rel1e-14/abs1e-15 for proved Python ULP differences. Do not refresh F8c for F3.
- Runtime refreshed: source2586a2cf0c8fdc4ee57d8d55e2a773d5ba7463bd, captured2026-09-06T01:53:11.416666+00:00, dirty=false/status[],75 hashes,18 cells/7 samples/pools1. Stable runtime_scaling_2026-09-03.csv/.provenance.json; SHA2564a56af6b5cd1fef80366703e4529e93e16a0acd7bf1dd52ff6dd23efc850765b. docs/runtime-scaling.md table+commit+checksum and benchmarks/README.md source updated/bound.18/18 data/selection fingerprints match retained baseline. Waited for another local SIFT test run and transient Git job to finish; ordinary desktop background activity remains, so timings are descriptive, not an A/B speed claim. F8c artifacts untouched.

## Remaining order and tooling

- E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks.
- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python pytest9.1.1. Base /opt/anaconda3/bin/python Python3.12.7/scientific deps for benchmark/Ruff/generators (base pytest7.4.4 not used). Docs /private/tmp/sift-docs-venv/bin/python. No localCatBoost/category_encoders; CI covers; no installs.
- Pools1, LOKY_MAX_CPU_COUNT8, PYTHONPATH="$PWD" for scratch probes. Strict docs output /private/tmp/sift-f3-review-site.
- Always gh --repo kmedved/sift and explicit PR number. Required jobs3.10/3.11/3.12/test-catboost/min-pins/wheel-smoke. Merge --merge with exact-head success, never squash (retain clean-source ancestry). CI fetch-depth0 supports historical artifact verification.
