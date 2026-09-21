# SIFT 0.10.1 release and 1.0 development transition — in progress

Objective: publish GitHub-only v0.10.1 with Python 3.13 support and the advance 1.0 migration notice while retaining 0.10 defaults, then update reviewed PR #101 to 1.0.0.dev0, merge it with exact-head CI, and leave final 1.0.0 as a separate decision. PyPI is excluded.

State: Phase A is complete. PR #102 passed all six active jobs at `7cd4d70` and merged as `96a38f1`; its tree matches the tested head. Merged-main run `35654976557` passed. GitHub Release v0.10.1 and tag point to `96a38f1`; release workflow `35655498007` passed and attached a metadata-checked, clean-installed wheel and sdist. Phase B merge `0cc4397` joins released main to PR #101 while preserving the reviewed implementation and clean-source evidence history. The development version is now 1.0.0.dev0, with a distinct unreleased 1.0 section above the published 0.10.1 announcement.

Next: commit the 1.0.0.dev0 source state, refresh required clean-source runtime binding with unchanged data/selection fingerprints, and update PR #101. Require all six active checks on its exact ready head, merge with a merge commit, and verify merged-main CI and clean synchronization. Do not tag or publish final 1.0.0 or PyPI.
