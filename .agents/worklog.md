# SIFT audit corrections — implementation complete

## Objective and authority

Address all agreed findings in Fable's post-roadmap audit of d038698. The user authorized implementation, local commits and the existing runtime refresh, then approved pushing this branch, opening a PR, running required CI on the exact PR head, and merging without squashing once green. Branch: codex/0.9.1-audit-corrections; base: main. Version remains 0.9.1.dev0. No tag, release, PyPI publication, unrelated cleanup, or other external action is authorized.

## Completed work

- Wrong-result/routing fixes: compare prefix identity and purged time forwarding; conditioning iterables; dropped-constant block units; nonfinite callable importance; duplicate categorical labels; rolling training caps after purging; supported weighted multi-target routes and early dense-check errors.
- Composition/validity fixes: frozen-cache resampling rejection; multiplicity-preserving deduplication for supported inner-CV bases or explicit restrictions; unseen-level within-validation guard; 1-D transform copy and two-way approximation disclosure; constant-selected proxy restrictions, typed empty reports, float32 boundary policy and refit guidance; nested knockoff validity, feasibility rounding, e-BH input validation and representative-evidence labeling.
- Manifest/API fixes: deterministic supported identities and streamed object hashes; long-configuration digests; opt-in caller context hashes with honest completeness and selection-time limits; cache privacy; effective multi-target metadata; legacy delegates; replayable unseeded StabilitySelector roots and controlled declared Stabilized base seeds; numeric onehot no-op; conditioned-auto preflight; compare val_frac/empty-design/unsupported-label behavior; supported duck-array row counting.
- Documentation: complete feature/export/provenance entries, base-dependent support matrix, future 1.0 owner decisions, raw-unit scoring and typed ordinal limits, within/proxy behavior, and bakeoff no-discovery/floor interpretation. Unsupported CEFS+ recommendations removed.
- Qualified claims remain qualified: raw-unit multi-target scoring, nominal repr-ordered typed encoding, empty-selection FDP=0, and the selected-selected report/cluster distinction were not silently redefined.

Grok implemented the initial corrections, then exhausted its balance; Codex completed the unfinished work. Resumed Opus read-only/xhigh reviews plus independent Codex probes closed all accepted findings. No additional speculative review or optimization round is pending.

## Decisive verification

- Accepted source full warnings-as-errors run: 2480 passed, 40 skipped, with only the then-stale runtime source binding failing. No production source changed afterward.
- After the authorized refresh: existing runtime/provenance/table and release-ledger tests, 5 passed. Strict MkDocs rebuilt successfully. Ruff, API generator (66 exports), support-matrix check, and diff checks passed.
- Runtime: unchanged 18 method/workload cases, seed 20260903, one warm-up and seven timed calls, single-threaded native pools. All 18 settings, data hashes, and selection hashes match the previous reference; all 86 source hashes and the rendered table bind to the refreshed artifact.
- Clean measured source: 06c569ecc5d6b8e7b489c34f530fb20306175737, dirty=false, captured 2026-09-08T00:11:23.971160+00:00. Runtime CSV SHA256 c37f97b0b9ffbb5126df22f375c3fb7cdee31cf73c9b6e4f338843307b2a9cdc. Source and refreshed evidence are retained in separate local commits.
- Historical quality evidence is unchanged: CSV SHA256 40d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf; JSON SHA256 bd84c19ca731cfb92e553c308c74969f547484355448b676fbad89a0d0606cd9.
- Hash-memory correction preserved the exact 2000x10 string-fixture digest while median tracemalloc peak fell from 5,970,868B to 164,267B. This is traced allocation, not process RSS or a runtime speedup. Runtime refresh occurred without launched SIFT tests/reviews; ordinary desktop background activity remained.

## Integration

All agreed corrections and the local evidence gate are complete. The authorized integration requires passing CI on the exact PR head and a merge commit preserving measured source 06c569e. The PR checks and Git ancestry are authoritative for integration status; there is no further local implementation or review round pending. Historical runtime evidence remains in Git history. Tagging, releases and publication remain outside the authorized task.
