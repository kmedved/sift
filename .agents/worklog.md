# SIFT 0.10.1 release and 1.0 development transition — in progress

Objective: publish GitHub-only v0.10.1 with Python 3.13 support and the advance 1.0 migration notice while retaining 0.10 defaults, then update reviewed PR #101 to 1.0.0.dev0, merge it with exact-head CI, and leave final 1.0.0 as a separate decision. PyPI is excluded.

State: release branch `codex/0.10.1-release` starts from clean main `612368e`. It contains only the independently separable announcement/policy commits from PR #101 plus their CatBoost notice correction. Commit `aef2e8b` sets the v0.10.1 version/date while excluding the approved 1.0 implementation. Runtime evidence was rerun from clean `aef2e8b`: all 18 data and selection fingerprints are unchanged; the source hashes now bind the already-landed Python 3.13 classifier and the release version. Representative old-default contracts passed 241 with 3 dependency skips. Runtime/release/public-spine checks pass 119/119; Ruff, generated API, and diff checks pass.

Next: commit the refreshed evidence, build and clean-install the local wheel/sdist, open and merge the release PR after exact-head CI, then publish and verify v0.10.1. Afterward integrate released main into PR #101, set 1.0.0.dev0, refresh any required source binding, pass exact-head CI, and merge with a merge commit. Do not publish final 1.0.0 or PyPI.
