# SIFT 0.10.0 GitHub release — in progress

Objective: release 0.10.0 from the reviewed closure, attach source and wheel distributions to a GitHub Release, then advance main to 0.10.1.dev0. Owner authorized the release and its necessary merge-commit PRs; PyPI publication is excluded.

Current state: PR #97 merged as f155725, allowing `test-latest-deps` on workflow_dispatch without changing its steps. Manual run 35546467408 targets that merge commit and is pending behind the automatic main run. The release-preparation branch changes the package version to 0.10.0 and dates the release notes 2026-09-20. The runtime-scaling test binds `sift/__init__.py`, so a clean-source evidence refresh is required after this version change.

Next: require the actual latest-dependency result; resolve only demonstrated release blockers. Then commit and refresh clean-source runtime evidence, build/check the release distributions, pass exact-head PR CI, and merge. Tag the tested release commit and publish the GitHub-only release; verify assets and linkage. Finally, bump to 0.10.1.dev0 through a separate PR and refresh the runtime binding required by that version change. Keep primary main clean and preserve frozen bakeoff artifacts.
