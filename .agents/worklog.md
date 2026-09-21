# SIFT 1.0.0 release and 1.0.1 development bump — in progress

Objective: publish final GitHub-only v1.0.0 from the approved implementation, attach verified wheel and source distributions, then advance development to 1.0.1.dev0 through a separate PR. PyPI is excluded.

State: v1.0.0 is published at merge `42c1e6d` through PR #103. Exact-head CI `35658690511`, the eight-job manual dispatch `35658721847`, merged-main CI `35659208975`, and release workflow `35659829103` passed. The published wheel (`8505f605...b262`) and sdist (`692a8c65...ad23`) match GitHub's SHA-256 digests, pass Twine, and the wheel clean-installs as 1.0.0. No PyPI publication occurred. Branch `codex/1.0.1-dev` advances source to `1.0.1.dev0` without product changes. Runtime evidence was rerun from clean `edfa6b2`: all 18 data and selection fingerprints are unchanged and only `sift/__init__.py` changed among bound sources. Public-spine, runtime-evidence, docs-smoke, and docs-example contracts pass; Ruff, generated API, and diff checks pass. The first docs-example run hit the host's known physical-core probe warning under warnings-as-errors; all 108 examples passed with 8 dependency skips after setting `LOKY_MAX_CPU_COUNT=4`.

Next: push the development branch, deliver it through a separate merge-commit PR with exact-head CI, verify the merge tree, then require merged-main CI before closeout.
