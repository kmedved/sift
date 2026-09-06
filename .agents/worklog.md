# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: finish approved0.9.1 closeout and ordered0.9.x roadmap in TODO.MD/docs/specs/0.9-product-layer.md. Keep goal active at intermediate milestones.
- Grok4.6 primary coder, Codex independently verifies, Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic approved.
- Resume exact native sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium completed83 exit0; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c readonly/xhigh completed72 exit0. No fresh session/turn cap. Grok compacted during67/70/74/77/81. Launchers python3 ~/.codex/skills/grok/scripts/grok_run.py and ~/.codex/skills/claude-cli/scripts/claude_cli_run.py; redact incidental sc_token output.
- Latest owner steering: if Opus usage runs out, Codex should do the work directly and continue without waiting for reset. No quota failure in completed72.
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
- E4 one-hot blocks: PR86 mergedc5e1d51b6938c9499239bb319753fd708b736014 (implementation826e10b, artifact/status1a99aad). All6 required jobs passed on exact1a99aadc699cfb8f74439960fa476a108dbbf307, run34010206243. Merged tree exactly equals tested head, clean source826e10b remains an ancestor, local main fast-forwarded. No PyPI.

## Current stage: F9 accepted, CI and merge next

- Branch codex/0.9x-f9-compare, basec5e1d51b6938c9499239bb319753fd708b736014. Implementation6ecbe637606ebd0246df8bc2808c43ee8a877424 accepted and committed after Grok82/Opus72; evidence/status commit and CI/merge next. Additive compare/CompareResult exports bring surface to60. Preserves the initial E4 tracker-closeout edits.
- Default CV uses fresh fold-local selector fit_transform, fresh downstream fit, held-out-only scoring, exact row/target/weight/context slicing. Reports actual raw identity and additional-block units; accepts standard sklearn get_support and Stability contracts. True empty sets use weighted Dummy priors/means. in_sample_path retains fitted encodings, discovery order independently of output_order, whole blocks and protected include at k0. Typed raw labels are not positional indices. Every table/bookkeeping has protocol labels; to_dict is JSON-safe, not a manifest.
- Decisive Codex verification: typed_path/prefix/fold/accept80 public oracles all PASS;14 F9 tests pass under -W error. Includes exact target_cv Pipeline and weighted empty-prior oracles; failed-refit metadata cleanup passed Gaussian/binary/Knockoff. Ruff/API60/data-type matrix/strictdocs/diff clean. Opus72 independently35 tests plus typed-label/Stability/generic path oracles, no findings within focused scope; prior review71 accepted atomic/output-order/include-only corrections and review70 accepted eight original groups. Frozen tracked+untracked hashes unchanged through72; all provider prompts removed. No further review without new evidence.
- Clean implementation6ecbe637606ebd0246df8bc2808c43ee8a877424 committed. Runtime refresh completed from clean6ecbe63: captured2026-09-06T05:09:16.328696+00:00 through05:09:46.153309+00:00, dirtyfalse/status[],76sourcehashes,18cells,7samples,pools1; CSVsha d3639b42baf2b80a0930e0444f8947cf08df08ecdbf24d9ac94eb5bd88261c1e.18/18data/selection fingerprints match prior. Bound docs table/ratio8.2/source/checksum+README updated. No overlapping SIFT workers; unrelated low-CPU provider sessions only, descriptive not A/B speed evidence. Full local suite2243passed40skipped/noexclusions in79.04s; Ruff/API60/matrix/strictdocs/diff pass.
- Before CI Codex verified test-only newer-sklearn incompatibility: three F9 synthetic selector fixtures call BaseEstimator._validate_data, absent in upstream sklearn1.7.2 (supported dependency band >=1.3,<2). Grok83 replaced those private calls with public n_features_in_/feature_names_in_ on the fixtures; assertions and generic selector contracts unchanged. Codex independently confirmed the three public-fixture edits and14 F9 tests passed -W error; Ruff/diff clean. Production code and bound runtime artifacts untouched; prior full2243/40 plus exact-head CI are the integration evidence. Checkout released for Codex closeout (focused test verification/evidence-status commit/exacthead6CI/merge--merge). No new whole-code review for test-only compatibility. No PyPI/tags/releases; F8c quality frozen.
- Next implementation after F9 integration: reproducibility manifests, then F4/F5/F6/classic caches/categorical fallbacks. OldPR43/51 outofscope; PID35265 no longer exists.

## Retained evidence

- Frozen F8c quality: sourceae904b8af02037eb66cd649384c4665dba17049d, captured2026-09-05T06:32:30.226365+00:00, dirty=false/status[],75 hashes;480 records, pools1, no failures/warnings. benchmarks/results/knockoff_statistic_bakeoff.csv/.provenance.json; SHA25640d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. Preserve CRLF .gitattributes and historical-source binding. Summary-only floats allow rel1e-14/abs1e-15 for proved Python ULP differences. Do not refresh F8c for later roadmap stages.
- Runtime refreshed from6ecbe637606ebd0246df8bc2808c43ee8a877424, captured2026-09-06T05:09:16.328696+00:00, generated05:09:46.153309+00:00, dirty=false/status[],76hashes,18cells/7samples/pools1. Stable runtime_scaling_2026-09-03.csv/.provenance.json, CSVsha d3639b42baf2b80a0930e0444f8947cf08df08ecdbf24d9ac94eb5bd88261c1e. Docs table/ratio8.2/commit/checksum+README bound;18/18data/selection fingerprints match prior826e10b run. No overlapping SIFT workers; unrelated low-CPU providers/ordinarydesktop activity, descriptive not A/B speed evidence. F8c artifacts untouched.

## Remaining order and tooling

- F9 leakage-safe compare; manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks.
- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python pytest9.1.1. Base /opt/anaconda3/bin/python Python3.12.7/scientific deps for benchmark/Ruff/generators (base pytest7.4.4 not used). Docs /private/tmp/sift-docs-venv/bin/python. No localCatBoost/category_encoders; CI covers; no installs.
- Pools1, LOKY_MAX_CPU_COUNT8, PYTHONPATH="$PWD" for scratch probes. Strict docs output /private/tmp/sift-f9-final-site.
- Always gh --repo kmedved/sift and explicit PR number. Required jobs3.10/3.11/3.12/test-catboost/min-pins/wheel-smoke. Merge --merge with exact-head success, never squash (retain clean-source ancestry). CI fetch-depth0 supports historical artifact verification.
