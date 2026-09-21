# SIFT 0.10.1 release and 1.0 development transition — in progress

Objective: publish GitHub-only v0.10.1 with Python 3.13 support and the advance 1.0 migration notice while retaining 0.10 defaults, then update reviewed PR #101 to 1.0.0.dev0, merge it with exact-head CI, and leave final 1.0.0 as a separate decision. PyPI is excluded.

State: release branch `codex/0.10.1-release` starts from clean main `612368e`. It contains only the independently separable announcement/policy commits from PR #101 plus their CatBoost notice correction. The release version/date and release documentation now target v0.10.1; the approved 1.0 implementation remains excluded. Runtime evidence must be rebound after the version source change before release checks and delivery.

Next: commit the release source state, refresh clean-source runtime evidence with unchanged data/selection fingerprints, run the smallest release checks including representative old-default contracts and distribution smoke, open and merge the release PR after exact-head CI, then publish and verify v0.10.1. Afterward integrate released main into PR #101, set 1.0.0.dev0, refresh any required source binding, pass exact-head CI, and merge with a merge commit. Do not publish final 1.0.0 or PyPI.
