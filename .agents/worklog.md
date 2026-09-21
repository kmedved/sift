# SIFT 1.0.0 GitHub release — in progress

Objective: publish final GitHub-only v1.0.0 from the approved implementation, attach verified wheel and source distributions, then advance development to 1.0.1.dev0 through a separate PR. PyPI is excluded.

State: branch `codex/1.0.0-release` starts from clean main `a68695a`, whose six-job merged-main run `35656950667` passed. Commit `bb86a65` sets version 1.0.0, dates the finalized 1.0 notes, and preserves the approved defaults and compatibility boundaries. Runtime evidence was rerun from clean `bb86a65`: all 18 data and selection fingerprints are unchanged and only `sift/__init__.py` changed among bound sources. Focused release/default/runtime checks pass 127 with 1 dependency skip; Ruff, generated API, and diff checks pass.

Next: commit the refreshed evidence, build and clean-install the wheel/sdist, run the existing dispatch release gates, and deliver through a merge-commit PR with exact-head CI. Publish and verify v1.0.0, then advance to 1.0.1.dev0 through a separate merge-commit PR. Do not publish to PyPI.
