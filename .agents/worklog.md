# Python 3.13 CI support — in progress

Objective: add Python 3.13 to the existing SIFT CI, fix only demonstrated compatibility failures, and update current support documentation. Deliver a scoped PR with all required checks green; Astra reviews before merge. No 1.0 API/default changes, version bump, tag, release, or PyPI action.

State: topic branch `codex/python313-ci` starts from clean main `3617060092512cd917e984f2cd7745da3846637d` (version 0.10.1.dev0). Existing test matrix now includes 3.13 and preinstalls `numba>=0.61` only there, matching the repository's documented wheel constraint. The new job inherits the matrix's full-suite and provenance steps. Historical deferral comments were removed. No source or runtime artifact changed.

Next: commit and push this CI-only first head, read the actual Python 3.13 and existing PR checks, repair only demonstrated failures, then update current support docs/metadata from the observed result and rerun exact-head CI. Callback to Astra with the green PR for review before merge.
