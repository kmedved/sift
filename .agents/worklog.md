# SIFT 0.10.0 GitHub release — in progress

Objective: release 0.10.0 from the reviewed closure, attach source and wheel distributions to a GitHub Release, then advance main to 0.10.1.dev0. Owner authorized the release and its necessary merge-commit PRs; PyPI publication is excluded.

Current state: PR #97 merged as f155725, allowing `test-latest-deps` on workflow_dispatch without changing its steps. Manual run 35546467408 passed all seven jobs, including latest dependencies and benchmark-smoke. This branch sets version 0.10.0 and dates the release notes 2026-09-20. Runtime evidence was refreshed from clean source commit dfb26b5; all 18 data and selection fingerprints match the previous artifact, and its binding tests pass. The wheel and sdist build, pass Twine metadata checks, and the wheel installs and verifies in a fresh venv. Local suite: 2,785 passed, 41 skipped with `LOKY_MAX_CPU_COUNT=8` for this machine's loky core-count probe; both documented quick benchmark commands also passed.

Next: push this release-preparation branch, await exact-head PR CI, and merge. Tag the tested release commit and publish the GitHub-only release; verify assets and linkage. Finally, bump to 0.10.1.dev0 through a separate PR and refresh the runtime binding required by that version change. Keep primary main clean and preserve frozen bakeoff artifacts.
