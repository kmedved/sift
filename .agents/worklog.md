# SIFT 1.0.1 release preparation — in progress

Objective: merge approved PR #105 with its history intact and prepare exact 1.0.1 notes and distributions. Publishing a tag or GitHub Release is not authorized; PyPI is excluded. Use GPT-6.1 Sol xhigh without other workers.

State: PR #105 merged as `5489807` after CI `37995689754` passed all six active jobs at head `484b0d7`. The merge tree exactly matches that head, and evidence ancestors `055d9b5` and `5ffc877` are retained. Branch `codex/1.0.1-release` prepares version 1.0.1 and dated notes without new API or default changes. Version and release-note checks passed 118 tests; the focused release/docs/benchmark/evidence checks passed 77. Ruff, generated API and diff checks passed. Runtime evidence was rerun from clean release source `b615108`: only the version file changed among bound sources, and all 18 data and selection fingerprints are unchanged.

Next: build and clean-install the distributions, open the release-preparation PR, and complete the existing CI and benchmark gates at its final head. Stop before merging this preparation PR or publishing any tag or release.
