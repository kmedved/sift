# SIFT 1.0.1 release preparation — in progress

Objective: merge approved PR #105 with its history intact and prepare exact 1.0.1 notes and distributions. Publishing a tag or GitHub Release is not authorized; PyPI is excluded. Use GPT-6.1 Sol xhigh without other workers.

State: PR #105 merged as `5489807` after CI `37995689754` passed all six active jobs at head `484b0d7`. The merge tree exactly matches that head, and evidence ancestors `055d9b5` and `5ffc877` are retained. Local main was fast-forwarded cleanly. Branch `codex/1.0.1-release` prepares version 1.0.1 and dated notes without new API or default changes.

Next: commit the release version, refresh source-bound runtime evidence, build and validate wheel/sdist, complete the documented release gates, and open a release-preparation PR. Stop with concrete publishing readiness.
