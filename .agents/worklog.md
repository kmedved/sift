# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: finish approved0.9.1 closeout and ordered0.9.x roadmap in TODO.MD/docs/specs/0.9-product-layer.md. Keep goal active at intermediate milestones.
- Grok4.6 primary coder, Codex independently verifies, Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic approved.
- Resume exact native sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium, completed78 exit0; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c read-only/xhigh, completed68 exit0. Next coding79/review69 on the same goal after E4 integration. No fresh session/turn cap. Grok compacted during67/70/74/77. Launchers: python3 ~/.codex/skills/grok/scripts/grok_run.py and ~/.codex/skills/claude-cli/scripts/claude_cli_run.py. Redact incidental sc_token output.
- No checkout review/edits during writer; freeze during review. Persist future hash baselines outside checkout, not only transient tool store. Codex owns Git/CI. Preserve unrelated changes. No PyPI; tags/GitHub Releases separate owner actions. Development0.9.1.dev0, v0.9.0 immutable94bae05.

## Completed milestones

- 0.9.1 docs/matrix/glossary/tutorial/integration and router split: PR78 merged3b9ac0a (implementationfa74d63).
- F1 conditioning: PR79 merged14bfb5c (implementationb2a11bd, artifact682af22).
- F2 proxy/cluster reports: PR80 merged17fe3bf (implementation081ca04, artifacta495f33).
- F7 panel transforms: PR81 merged76e4d51 (implementation2d04754, artifactd3938e4).
- F8a knockoff UX: PR82 merged40f8af7 (implementation308dfe1, artifact0c62a44, test449f063).
- F8b e-values: PR83 mergedf70a3da (implementationdde1f50, artifact2a05c66). Validated opt-in ungrouped relevance/ridge fixed universe; CEFS+/LSM/grouped/varying-screen/supervised-encoding modes explicitly exploratory. No default or FDR validity upgrades.
- F8c bakeoff: PR84 mergedcfd2f645ee5404fb3cb85be0a69cde3e7e699a26 (implementationae904b8, artifact/report0b74737, CRLF40bec7d, ULP-test20d07d1). All6 required jobs passed at exact20d07d18f0ae782db0df8ebc33c1bed4fffe5363, run33951312155; merged tree matched tested head. Keep relevance default on the committed4-DGP grid; not mixed-sign/suppressor/p>>n evidence.
- F3 atomic blocks: PR85 mergede9a4464deb3745ec1bf129ce556ca1740208ab42 (implementation2586a2c, artifact/status9f91781). All6 required jobs passed on exact9f917817083b5b48eb4625bcd0a5bae6113b4564, run34005199518. Merged tree exactly equals tested head. Local main fast-forwarded. No PyPI.

## Current stage: E4 accepted; integration closeout

- Branch codex/0.9x-e4-onehot-blocks, basee9a4464deb3745ec1bf129ce556ca1740208ab42. Grok78 and resumed Opus68 completed exit0. Codex and Opus accept the bounded E4 implementation; no verified correctness blocker remains. Tracked+untracked hashes unchanged through Opus68 against /private/tmp/sift_e4_review68_hashes.txt. Provider prompts removed; no active provider.
- E4 contracts: fitted capped one-hot encoding, positive-weight vocab/pooling, stable typed categories/raw labels, atomic raw/encoded blocks and conditioning, training-only encoding on supported validation routes, raw results plus complete encoded identity/raw_feature table, fixed wrapper inference schema. Existing defaults/math retained; caches/within/knockoffs/Boruta reject. No new FDR claim.
- Codex independent acceptance /private/tmp/sift_e4_codex_accept78.py: whole-block public results/scored widths across Gaussian/classic/binary; explicit-dummy score oracle including forced bases and k=0; typed relevance/raw-parent mapping; exact legacy positional weighted-call parity; nested compound paths all PASS. Focused E4/F3 tests58 passed; Ruff/API58/matrix/strictMkDocs/diffcheck passed. Earlier category/fold-local/proxy acceptance and manual Gaussian/binary oracle pass.
- Opus68 report /private/tmp/sift_e4_opus68_report.md: no remaining findings; independently verified scored matrices, multi-categorical/multi-block grids, typed mapping/weights, and nonvacuous fold guard.21+165 tests passed; no writes. Optional nested-test suggestion already covered by the new compound-block test; no extra test/review round needed. Synthesis /private/tmp/sift_e4_review67_synthesis.md records resolved findings.
- Next: commit accepted implementation, refresh current runtime evidence alone from that clean commit, update bound docs/artifact and commit, run full final gates with no exclusions, push/PR, verify exact-head6 CI jobs, merge --merge. Do not rerun frozen F8c quality. No PyPI; no tags/releases.
- Previous turn produced decisive acceptance evidence plus verified reviewer wait. This turn observed clean Opus completion and began integration; no blocker. F3 integrated PR85, no redo. OldPR43/51 remain open/outofscope; PID35265 no longer exists, no process killed.
- Full ordered goal remains active; next feature after E4 merge is F9 leakage-safe compare, then manifests/F4/F5/F6/classic caches/ordinal-frequency.

## Retained evidence

- Frozen F8c quality: sourceae904b8af02037eb66cd649384c4665dba17049d, captured2026-09-05T06:32:30.226365+00:00, dirty=false/status[],75 hashes;480 records, pools1, no failures/warnings. benchmarks/results/knockoff_statistic_bakeoff.csv/.provenance.json; SHA25640d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. Preserve CRLF .gitattributes and historical-source binding. Summary-only floats allow rel1e-14/abs1e-15 for proved Python ULP differences. Do not refresh F8c for F3.
- Runtime refreshed: source2586a2cf0c8fdc4ee57d8d55e2a773d5ba7463bd, captured2026-09-06T01:53:11.416666+00:00, dirty=false/status[],75 hashes,18 cells/7 samples/pools1. Stable runtime_scaling_2026-09-03.csv/.provenance.json; SHA2564a56af6b5cd1fef80366703e4529e93e16a0acd7bf1dd52ff6dd23efc850765b. docs/runtime-scaling.md table+commit+checksum and benchmarks/README.md source updated/bound.18/18 data/selection fingerprints match retained baseline. Waited for another local SIFT test run and transient Git job to finish; ordinary desktop background activity remains, so timings are descriptive, not an A/B speed claim. F8c artifacts untouched.

## Remaining order and tooling

- E4 one-hot blocks; F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks.
- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python pytest9.1.1. Base /opt/anaconda3/bin/python Python3.12.7/scientific deps for benchmark/Ruff/generators (base pytest7.4.4 not used). Docs /private/tmp/sift-docs-venv/bin/python. No localCatBoost/category_encoders; CI covers; no installs.
- Pools1, LOKY_MAX_CPU_COUNT8, PYTHONPATH="$PWD" for scratch probes. Strict docs output /private/tmp/sift-f3-review-site.
- Always gh --repo kmedved/sift and explicit PR number. Required jobs3.10/3.11/3.12/test-catboost/min-pins/wheel-smoke. Merge --merge with exact-head success, never squash (retain clean-source ancestry). CI fetch-depth0 supports historical artifact verification.
