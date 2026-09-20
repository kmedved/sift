# SIFT 0.9.1 closure — in progress

Objective: complete Fable's paused closure fixes on a local integration branch, verify them, refresh clean-source runtime evidence, and hand Astra a reviewable result. No push, PR, main merge, tag, release or version bump; the owner version-name decision remains open.

State: main clean at c72900c. Integration branch audit/0.9.1-closure-fixes contains Fable's CI provenance fix and seven merge commits preserving all closure branches. Known compare and manifest failures were repaired; one-hot conditioning reports raw names for function and wrapper routes; remaining code/docstring gaps and Markdown pass are integrated. Frozen knockoff bakeoff artifacts are unchanged.

Decisive checks: integrated suite 2760 passed, 41 skipped, 1 runtime-binding test deselected pending evidence refresh; 272 focused docs tests passed with 15 skips; strict MkDocs, Ruff, API generator and data-type generator passed. The 12-case compare and 4-case path-evaluation A/B matched c72900c except intended classification stratification. Local CatBoost environment: 75 marked tests and 41 adapter/view tests passed. CI provenance fix was validated in a fresh clone by Fable; local workflow-dispatch confirmation requires later PR/merge.

Next: commit the integrated code/docs to make the source clean, run the full runtime-scaling benchmark with frozen configuration, verify 18 data/selection fingerprints against existing sidecar, update CSV/sidecar/documented table and checksum, run the previously deselected binding test and final checks, then send review callback to Astra. Report any deferred test-quality-only items honestly.

Blockers: none. The version name remains an owner decision.
