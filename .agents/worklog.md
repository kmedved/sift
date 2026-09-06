# SIFT 0.9 roadmap worklog

## Objective and constraints

- Active unbudgeted goal: approved 0.9.1 closeout and ordered 0.9.x roadmap. All through F5 merged; F6 implementation accepted, integration pending; classic caches and unsupervised ordinal/frequency categorical fallbacks follow. Do not complete the goal at an intermediate milestone.
- Grok4.6 primary coder in same native session `6942c6e1-14da-42e0-8ef2-27e77a0ab942` workspace/medium; Codex verifies and owns Git/CI; Opus same native `72bc11aa-3fc5-4cae-8b5a-9197b89f270c` readonly/xhigh reviews between stages. Relevant source transmission approved. Latest owner: if Opus quota fails, Codex continues directly, no reset wait. No quota failure through91.
- Grok106 terminal0; Opus91 terminal0; all prompts deleted after completion and review hashes unchanged. Next Grok107 is classic caches after F6 merge. No fresh sessions/caps. Observation timeouts are not termination; poll exact live handles. Launchers `python3 ~/.codex/skills/grok/scripts/grok_run.py` and `python3 ~/.codex/skills/claude-cli/scripts/claude_cli_run.py`, `--cwd "$PWD" --task-file /private/tmp/... --mode workspace|read-only --effort medium|xhigh`. Redact incidental sc_token; do not pass sandbox_permissions.
- No checkout review/edits during writer ownership; freeze source during review. Handles/hash baselines outside checkout. Preserve unrelated changes. No PyPI/tag/release or old-PR/PID cleanup. Keep0.9.1.dev0 and immutablev0.9.0.

## Completed milestones

- Through F8c: PR78docs/router3b9ac0a,79F114bfb5c,80F217fe3bf,81F776e4d51,82F8a40f8af7,83F8bf70a3da,84F8ccfd2f64. Keep relevance default and approximate-plugin/exploratory FDR caveats.
- PR85F3e9a4464 (2586a2c/9f91781),86E4c5e1d51 (826e10b/1a99aad),87F97314355 (6ecbe63/69410cb),88manifestsc29e99e (c322671/bb651a2),89F418cdf40 (7cc4f66/b54a138/bfe5765). Exact-head six required jobs and source ancestry accepted.
- F5 PR90 merged0ee513b9a6ba9ec78397d482f93587ec1fee5f3c at2026-09-06T09:49:04Z; clean implementationc7ea9daaf48a3de0cee6c5e2a2c59efcc7b90f70; evidence/testedheadfe14516d09b8ffda9f9c8244924a5bc8519c4a53. All6CI(run34025486997), mergedtree==testedhead and clean-source ancestry verified. Local2303passed40skipped-Werror/noexclusions68.96s, Ruff/API61/matrix/strictdocs. Factorreuse13vs1266 exact selection/objective2.22e-16, descriptive microtiming only. F5 acceptance /private/tmp/sift_multitarget_accept81.md.

## Current stage: F6 accepted, integration pending

- Branch `codex/0.9x-model-selector` from0ee513b. Required-estimator ModelSelector(RFE/forward/stability, group/timeCV, nested scoring, additiveview), PurgedTimeSeriesSplit/GroupPurgedTimeSeriesSplit.64exports. Generic and native CatBoost share internal prepare/evaluate/choose/finalize orchestration; retain distinct numerical backends and native SHAP/Pool/voting/padding/defaults.
- Primitives97/83 accepted:20tests,630pairwiseoraclefolds,12largeoffsetinteger and8reversemixeddomain probes; allocation112788543→1603983bytes exact5foldindices, descriptiveallocationonly. /private/tmp/sift_purged_accept97.md.
- Production102/88 accepted: Codex532passed5skipped-Werror24.33s,41exact committed-native-orchestration comparisons and warningcaller, configuredCV/precomputed/weightedemptybaseline/snapshotoracles. Opus88 210/2 plus shared4phase/all4warning probes. /private/tmp/sift_modelselector_accept88.md. NativeCatBoost absent locally; requiredCI must execute native contracts.
- Matrix103/104/89 accepted: ModelSelector row yes/yes/cond/no/no/yes/cond/cond; disclosed Ridgebaseline, raw-permutation categoricalpipeline, searched-count group/timecontext. All169oldtable/note rows unchanged. Corrected width-preserving, not order-preserving, encoder wording.
- New concrete pipeline alignment defect from89 observation independently reproduced: reverse-transform coefficient auto selectednoise; permutation selectedsignal.105 fixed flat but missed nested-final;106 validates every preprocessing segment along same unwrap chain, including one-step outer, carrying intermediate names. Safe nested scaling retained; declared-name semantic trust documented. No broad re-audit.
- Final acceptance91: Codex267passed3skipped-Werror23.55s (model/purged/matrix/views/manifests),8auto/coefrejections+4permutationrecoveries+3safeautorecoveries. FullrepoRuff/API64/diff clean. Opus91 33tests plus deeper/wrapped/name-carrying probes; no actionable findings. All review91 hashes unchanged. /private/tmp/sift_modelselector_accept91.md, _codex91_findings.md, _opus91_report.md. No further review round without new evidence.
- Next: clean implementation commit; noheavyoverlap/processaudit; isolated full18cell runtime(1warmup/7repeats), bind docs/check18fingerprints; full-Werror/noexclusions/static/docs; evidencecommit; pushPR/all6exactheadCI/mergecommit. No F6commit/runtime/PR yet.

## Retained evidence

- Runtime sourcec7ea9daaf48a3de0cee6c5e2a2c59efcc7b90f70; captured2026-09-06T09:39:13.512006+00:00/generated09:39:42.748079+00:00, dirtyfalse/status[],80sourcehashes,18cases/7samples/pools1. Stable basename benchmarks/results/runtime_scaling_2026-09-03.csv/.provenance.json; CSVsha29b728a983fc122565ed16d5add1cb2795d777d6955030e5564420941382943e; bounddocsratio8.1/table/source/sha+benchmarksREADME. All18fingerprints match prior7cc4f66. Descriptive timings, not A/B evidence. Refresh only after clean F6implementationcommit.
- Frozen F8cquality sourceae904b8af02037eb66cd649384c4665dba17049d; capture2026-09-05T06:32:30.226365+00:00, dirtyfalse/status[],75hashes/480records/pools1. knockoff_statistic_bakeoff.csv/.provenance.json CSVsha40d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf. PreserveCRLF/historicalbinding and summaryfloatrel1e-14/abs1e-15. No refresh/defaultflip/strongerFDRclaims.

## Tooling and integration gates

- Tests /private/tmp/sift-pytest9.5nI4QH/bin/python (pytest9.1.1); base /opt/anaconda3/bin/python (3.12.7,NumPy1.26.4,pandas2.2.2,sklearn1.5.1,SciPy1.13.1,Numba0.60); docs /private/tmp/sift-docs-venv/bin/python. No CatBoost/category_encoders local installs;CIcovers.
- Env LOKY_MAX_CPU_COUNT8/OPENBLAS_NUM_THREADS1/OMP_NUM_THREADS1/MKL_NUM_THREADS1; PYTHONPATH="$PWD" scratch. Full suite -Werror noexclusions after runtimebindingrefresh. Strictdocs output /private/tmp. Ruff via testpython -m ruff.
- Runtime command basepython benchmarks/bench_runtime_scaling.py --full --warmup-runs1 --timing-repeats7 --output benchmarks/results/runtime_scaling_2026-09-03.csv (actual flags require spaces). Singlethread VECLIB/NUMEXPR too; no overlapping heavywork. Update generatedtable/docsSHA/CSVsha/FDRwide-baselineratio and benchmarksREADME viaapply_patch.
- All6 requiredCI: test3.10/3.11/3.12,test-catboost,min-pins,wheel-smoke. Explicit gh --repo kmedved/sift; merge with --merge --match-head-commit, never squash; fetch/mainff and verify mergedtree==testedhead/clean-sourceancestor. Test/docs-only compatibility fixes need focused proof+CI, not new productionreview/runtime.
- Remaining order afterF6merge: standardizedclassic caches → leakage-free unsupervisedordinal/frequency categorical fallbacks. Goal remains active.
