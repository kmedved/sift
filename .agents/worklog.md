# SIFT 1.0.0 release and 1.0.1 development bump — in progress

Objective: publish final GitHub-only v1.0.0 from the approved implementation, attach verified wheel and source distributions, then advance development to 1.0.1.dev0 through a separate PR. PyPI is excluded.

State: v1.0.0 is published at merge `42c1e6d` through PR #103. Exact-head CI `35658690511`, the eight-job manual dispatch `35658721847`, merged-main CI `35659208975`, and release workflow `35659829103` passed. The published wheel (`8505f605...b262`) and sdist (`692a8c65...ad23`) match GitHub's SHA-256 digests, pass Twine, and the wheel clean-installs as 1.0.0. No PyPI publication occurred. Branch `codex/1.0.1-dev` now advances source to `1.0.1.dev0` without product changes.

Next: commit the development version state, refresh the source-bound runtime evidence, run focused checks, and deliver the bump through a separate merge-commit PR with exact-head and merged-main CI.
