# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active goal: finish approved0.9.1 closeout and ordered0.9.x roadmap in TODO.MD/docs/specs/0.9-product-layer.md. Keep goal active at intermediate milestones.
- Grok4.6 primary coder, Codex independently verifies, Claude Opus xhigh reviews concurrently between stages. Return verified defects to Grok. Relevant private source transmission to xAI/Anthropic approved.
- Resume exact native sessions: Grok6942c6e1-14da-42e0-8ef2-27e77a0ab942 workspace/medium completed87 exit0; Opus72bc11aa-3fc5-4cae-8b5a-9197b89f270c readonly/xhigh completed76 exit0. No fresh session/turn cap. Grok compacted during67/70/74/77/81. Launchers python3 ~/.codex/skills/grok/scripts/grok_run.py and ~/.codex/skills/claude-cli/scripts/claude_cli_run.py; redact incidental sc_token output.
- Latest owner steering: if Opus usage runs out, Codex should do the work directly and continue without waiting for reset. No quota failure through completed76.
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

## Current stage: reproducibility manifests

- Branch codex/0.9x-reproducibility-manifests, base7314355a7002766a0a5265953d24bdc11746664b. F9 integrated; task-owned tracker closeout edits preserved. Manifests accepted after Grok87 / Opus76; implementation commit and clean-source runtime refresh next. If Opus quota fails, Codex continues directly per owner.
- Implemented `reproducibility_()` on SelectionView, FilterSelectionResult, KnockoffSelectionResult, and CompareResult. Schema `"1"` JSON via existing `_json_safe`/`_columns_hash`: export-time sift/numpy/pandas/sklearn/scipy/numba/BLAS/git; result-time shape and typed column hash; opt-in data hash; cache `n_rows_original`/`feature_names_are_synthetic` when retained; configured vs effective metadata; seeds only if stored; compare fold_bookkeeping reused. No new `__all__` name (surface stays 60). No raw X retention; hash_data default False. Legacy unknown provenance stays unknown; export environment is labelled `captured_at=export`. Cache-backed extra now also stores the synthetic-names flag. Compare diagnostics add n_rows/n_features/raw_columns_hash/input_kind/random_state. Constructors/pickles/defaults unchanged. Public tests in tests/test_reproducibility_manifest.py.
- Existing entry points: sift/selection/view.py (SelectionView slots/provenance/schema1/to_dict/_json_safe/_columns_hash), view_* adapters, result.py, compare.py (CompareResult,fold_bookkeeping). Normative docs/specs/0.9-product-layer.md F9 and view-provenance subsection; TODO.MD. Manifest API must clearly distinguish captured run context from export-time/caller-supplied facts; unknown provenance stays unknown.
- Last decisive accepted F9 evidence: independent typed_path/prefix/fold/accept80 oracles; full local2243passed40skip/noexclusions79.04s; subsequent test-only sklearn fixture compatibility14passed. Ruff/API60/matrix/strictdocs/diff pass. Grok83 and Opus72 exit0; no quota hit. All6 required CI jobs pass at exact69410cb5bbc5b50751cfa89a92227eb66fbe9a67 (run34013589434). PR87 merge7314355 exactly equals tested tree, clean source6ecbe63 ancestor, localmain ffverified. No repeat F9 reviews/gates/benchmarks absent changed dependency.
- Accepted Grok87/Opus76 (exit0/noquota/no findings). Codex39manifest/F9/filter tests2.37s; all review74 and75 public repros pass, including actual rows, typedkeys, AutoKpenalty/RNG, Dummy/prefixmodelcontexts, GroupKFoldflag, complete non-data selector options, and real in_sample_path cache summary/noasdictcopy/retention. Opus504focused/7skip;19manifesttests. Ruff(allsource/tests/scripts/benchmarks), API60, matrixcheck, strictdocs, diff pass; frozen76hashes unchanged. Evidence /private/tmp/sift_manifests_accept76.md and /private/tmp/sift_manifests_opus76_report.md. No further review absent new evidence. Nonblocking inherited partial-cache-fact limit for cache-only FDR: original/used rows retained, dedicated synthetic-name/cacheavailability history absent; exporter reports unknown rather than inventing. Next implementationcommit, standalonecleanruntime18cells7samples, bounddocs/fullgate, evidencecommit/exact6CI/merge--merge. No PyPI; F8c untouched.
- Remaining after manifests: F4 Stabilized, F5 multi-target CEFS+, F6 ModelSelector/purged splits, classic caches, unsupervised ordinal/frequency fallbacks. No PyPI/tags/releases. OldPR43/51 outside scope; PID35265 gone.

## Retained evidence

- Frozen F8c quality: sourceae904b8af02037eb66cd649384c4665dba17049d, captured2026-09-05T06:32:30.226365+00:00, dirty=false/status[],75 hashes;480 records, pools1, no failures/warnings. benchmarks/results/knockoff_statistic_bakeoff.csv/.provenance.json; SHA25640d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. Preserve CRLF .gitattributes and historical-source binding. Summary-only floats allow rel1e-14/abs1e-15 for proved Python ULP differences. Do not refresh F8c for later roadmap stages.
- Runtime refreshed from6ecbe637606ebd0246df8bc2808c43ee8a877424, captured2026-09-06T05:09:16.328696+00:00, generated05:09:46.153309+00:00, dirty=false/status[],76hashes,18cells/7samples/pools1. Stable runtime_scaling_2026-09-03.csv/.provenance.json, CSVsha d3639b42baf2b80a0930e0444f8947cf08df08ecdbf24d9ac94eb5bd88261c1e. Docs table/ratio8.2/commit/checksum+README bound;18/18data/selection fingerprints match prior826e10b run. No overlapping SIFT workers; unrelated low-CPU providers/ordinarydesktop activity, descriptive not A/B speed evidence. F8c artifacts untouched.

## Remaining order and tooling

- Manifests; F4 Stabilized; F5 multi-target CEFS+; F6 ModelSelector/purged splits; classic caches; unsupervised ordinal/frequency fallbacks.
- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python pytest9.1.1. Base /opt/anaconda3/bin/python Python3.12.7/scientific deps for benchmark/Ruff/generators (base pytest7.4.4 not used). Docs /private/tmp/sift-docs-venv/bin/python. No localCatBoost/category_encoders; CI covers; no installs.
- Pools1, LOKY_MAX_CPU_COUNT8, PYTHONPATH="$PWD" for scratch probes. Strict docs output /private/tmp/sift-f9-final-site.
- Always gh --repo kmedved/sift and explicit PR number. Required jobs3.10/3.11/3.12/test-catboost/min-pins/wheel-smoke. Merge --merge with exact-head success, never squash (retain clean-source ancestry). CI fetch-depth0 supports historical artifact verification.
