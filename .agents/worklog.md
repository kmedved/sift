# SIFT Python 3.11 floor and audited branch cleanup

## Objective and authority

Raise the minimum supported Python to 3.11, add pyarrow to the Python 3.12 CI test job, land the changes, close superseded PRs #43 and #51, delete the exact audited allowlist of 27 local and 85 origin feature branches, prune four stale worktree registrations, and finish clean and synchronized on main. The user explicitly authorized commits, push, PR, merge, and these audited deletions. No tags, releases, PyPI publication, unrelated cleanup, stashes, archive branches, or backup worktrees.

## Current state

- Delivery branch `codex/python311-ci-cleanup` starts from fetched origin/main at e533d42ef7a2b59f7f601cc83797239358605124.
- Audit allowlist and expected tip SHAs are in /private/tmp/sift-branch-audit-20260915/cleanup_candidates.json.
- All 27 local and 85 fetched origin feature refs still match their audited tip SHAs; the four audited worktrees remain the only dry-run prune targets.
- Metadata, CI, contribution guide, development guide, and unreleased release notes now state the Python 3.11 floor; the Python 3.12 test job installs PyArrow.

## Decisive verification

- Focused docs and Arrow contract slice: 8 passed on Python 3.11.16 with PyArrow installed.
- Ruff, generated API reference check, and `git diff --check` passed.
- Wheel built successfully and declares `Requires-Python: >=3.11`.
- Local strict MkDocs was unavailable in the designated environment; the Python 3.12 PR job installs docs dependencies and runs it.
- Initial PR CI exposed one shared failure: historical runtime evidence bound `pyproject.toml` despite recording the actual measured package versions. The test now excludes packaging metadata from runtime-source invalidation; no benchmark or frozen artifact was rewritten.

## Next action

Verify the focused runtime-evidence contract, update PR #95, then land through green CI and perform the approved close/delete/prune cleanup.

## Blockers

None.
