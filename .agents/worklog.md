# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: finish approved0.9.1 closeout and ordered0.9.x roadmap in TODO.MD/docs/specs/0.9-product-layer.md. Keep goal active at intermediate milestones.
- Grok4.6 primary coder, Codex independently verifies, Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic approved.
- Resume exact native sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium completed91 exit0; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c readonly/xhigh completed79 exit0. Next Grok92 for a verified integration defect or F5, then Opus80 if available. No fresh session/turn cap. Grok compacted during67/70/74/77/81/89. Launchers python3 ~/.codex/skills/grok/scripts/grok_run.py and ~/.codex/skills/claude-cli/scripts/claude_cli_run.py; redact incidental sc_token output.
- Latest owner steering: if Opus usage runs out, Codex should do the work directly and continue without waiting for reset. No quota failure through completed79.
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

- F9 leakage-safe compare: PR87 merged7314355a7002766a0a5265953d24bdc11746664b (implementation6ecbe63, artifact/status/testcompat69410cb). All6 required CI jobs passed at exact69410cb5bbc5b50751cfa89a92227eb66fbe9a67, run34013589434; merged tree exactly matches tested head, clean-source ancestry preserved. Full local2243/40/noexclusions plus focused14 after test-only compatibility fix; no PyPI.

- Reproducibility manifests: PR88 merged c29e99ecd005fe4cb4a3af7fa26946d3e23ee014 at2026-09-06T06:53:30Z (clean implc322671, artifact/status/glossarymove bb651a2). All6CI jobs passed at exactbb651a27f7394bab0a99ad73efe6678be382f517, run34017406858; mergedtree==testedhead, clean-sourceancestor, localmainffverifiedclean. AcceptedGrok87/Opus76; Codex39focused/allpublicrepros. Full local2261pass40skip/oneglossaryfailure70.63s; Grok88pureheadingmove7tests, Codex10glossary/runtimebindingtests; CIfullall6greenafterfix. StaticRuff/API60/matrix/strictdocs/diff passed. Acceptance/report /private/tmp/sift_manifests_accept76.md,/private/tmp/sift_manifests_opus76_report.md. NoPyPI.

## Current stage: F4 Stabilized (accepted, integration pending)

- Branch `codex/0.9x-stabilized`, clean implementation committed as `7cc4f6617678f0370f044c6c176703ffae9c92c2`. Grok90's eight original groups remain closed. Grok91 fixed the three remaining review78 defects. Surface 61. No FDR/math/default changes. F5/F6 still open.
- (1) Named ndarray stays an ndarray for generic bases (mixed hashable SelectKBest); SIFT wrappers still get a named frame or `feature_names`. (2) Manifest `configured_options` is a sanitized fit-time snapshot; mutating the shared base or wrapper params without refit does not rewrite it; output_order stays a live presentation fact. (3) Frequency `n_rows_used` comes from fitted `selector_metadata_` when known (CEFS+ subsample 40 on a 100-row draw); otherwise unknown. Draw/unique sizes stay in diagnostics only.
- Accepted after Codex151 focused tests and three independent public repros; Opus79 independently passed704/7 and confirmed all original eight plus final three defect groups closed. Grok F4 tests23/affected552/6. Ruff/API61/matrix/strict docs/diff checks pass. All344 frozen79 hashes verified unchanged after reviewer exit; providers released, prompts deleted.
- Opus's optional per-draw row-count list is not a correctness blocker: scalar actual rows are explicitly unknown when counts vary, draw/unique diagnostics retained. Do not add another correction/review round solely for this enhancement. Acceptance/report: /private/tmp/sift_stabilized_accept79.md and /private/tmp/sift_stabilized_opus79_report.md.
- Clean-source runtime refreshed and bound docs updated. Full local -W error/no-exclusions suite passed2290/40 in70.03s; strict docs/diff pass after bound-doc refresh. Next evidence/status commit; all6CI exact head; merge --merge. No F5 before F4 integration closes.
## Retained evidence

- Frozen F8c quality: sourceae904b8af02037eb66cd649384c4665dba17049d, captured2026-09-05T06:32:30.226365+00:00, dirty=false/status[],75 hashes;480 records, pools1, no failures/warnings. benchmarks/results/knockoff_statistic_bakeoff.csv/.provenance.json; SHA25640d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. Preserve CRLF .gitattributes and historical-source binding. Summary-only floats allow rel1e-14/abs1e-15 for proved Python ULP differences. Do not refresh F8c for later roadmap stages.
- Runtime refreshed from7cc4f6617678f0370f044c6c176703ffae9c92c2, captured2026-09-06T08:20:39.489935+00:00/generated08:21:08.747732+00:00, dirty=false/status[],79sourcehashes,18cells/7samples/pools1. Stable runtime_scaling_2026-09-03.csv/.provenance.json; CSVsha9c3b567557295f6fd6b761b37197208ef101e382df799c904773ace1ddcf7888. Bounddocs table/ratio8.0/source/checksum+README updated;18/18data/selection fingerprints match previousc322671 artifact. No concurrent SIFT jobs/providers; idle retiredDarkofit children and unrelated lowCPUClaude only. Descriptive desktop timing, not A/Bspeed evidence. F8c untouched.

## Remaining order and tooling

- F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks.
- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python pytest9.1.1. Base /opt/anaconda3/bin/python Python3.12.7/scientific deps for benchmark/Ruff/generators (base pytest7.4.4 not used). Docs /private/tmp/sift-docs-venv/bin/python. No localCatBoost/category_encoders; CI covers; no installs.
- Pools1, LOKY_MAX_CPU_COUNT8, PYTHONPATH="$PWD" for scratch probes. Strict docs output /private/tmp/sift-stabilized-final-site.
- Always gh --repo kmedved/sift and explicit PR number. Required jobs3.10/3.11/3.12/test-catboost/min-pins/wheel-smoke. Merge --merge with exact-head success, never squash (retain clean-source ancestry). CI fetch-depth0 supports historical artifact verification.
