# SIFT 0.10.0 release and development bump — in progress

Objective: finish the approved GitHub-only 0.10.0 release with source and wheel assets, then set main to 0.10.1.dev0. PyPI publication is excluded.

Current state: PR #97 enabled manual latest-dependency CI; run 35546467408 passed all seven jobs. PR #98 prepared 0.10.0 and all five exact-head PR gates passed; it merged as b97e94d with an identical tree. Tag v0.10.0 points to b97e94d; the GitHub Release is published with wheel and sdist assets. Release packaging run 35547707092 passed every step. This branch sets the development version to 0.10.1.dev0 and opens its unreleased notes section. The runtime-scaling test binds `sift/__init__.py`, requiring a clean-source evidence refresh after this version change.

Next: commit the dev-version source change, refresh runtime evidence from that clean commit, pass focused local checks, push a separate dev-bump PR, await exact-head CI, merge with a merge commit, and leave primary main clean and synchronized. Preserve frozen bakeoff artifacts.
