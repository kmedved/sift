# SIFT 1.0.0 GitHub release — in progress

Objective: publish final GitHub-only v1.0.0 from the approved implementation, attach verified wheel and source distributions, then advance development to 1.0.1.dev0 through a separate PR. PyPI is excluded.

State: branch `codex/1.0.0-release` starts from clean main `a68695a`, whose six-job merged-main run `35656950667` passed. The release source now sets version 1.0.0, dates the finalized 1.0 notes, and preserves the approved defaults and compatibility boundaries. Runtime evidence must be rebound after the version-source change before release delivery.

Next: commit the release source state, refresh clean-source runtime evidence with unchanged data/selection fingerprints, run the repository release checks, and deliver through a merge-commit PR with exact-head CI. Publish and verify v1.0.0, then advance to 1.0.1.dev0 through a separate merge-commit PR. Do not publish to PyPI.
