# SIFT 0.10.0 GitHub release — in progress

Objective: release 0.10.0 from the merged closure work, attach source and wheel distributions to a GitHub Release, then advance main to 0.10.1.dev0. Owner authorized the necessary PRs, merge commits, tag, and GitHub Release; PyPI publication is excluded.

Current state: primary main and origin/main are clean at bd13aeda23ddd0059f64a14b216cc5e1b9a9046f. PR #96 and its exact-head CI passed; merged-main manual workflow 35517554434 passed all six active jobs, including benchmark-smoke. No scheduled latest-dependency job has run on this commit. The existing `test-latest-deps` job only admits `schedule`, so this branch adds `workflow_dispatch` eligibility without changing its steps or dependency set.

Next: validate and merge this narrow CI change through a PR; dispatch the existing workflow on merged main and require the actual latest-dependency job result. Resolve only demonstrated release blockers, then prepare 0.10.0 version and dated notes, build/check the distributions, pass exact-head release-preparation CI, merge, tag and publish the GitHub-only release, verify assets, and bump the development version through a separate PR. Preserve the runtime evidence and frozen bakeoff artifacts.
