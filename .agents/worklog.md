# SIFT Python 3.11 floor and audited branch cleanup — complete

## Objective and authority

Raise the minimum supported Python to 3.11, add PyArrow to the Python 3.12 CI test job, land the changes, close superseded PRs #43 and #51, delete the exact audited allowlist of 27 local and 85 origin feature branches, prune four stale worktree registrations, and finish clean and synchronized on main. No tags, releases, PyPI publication, unrelated cleanup, stashes, archive branches, or backup worktrees.

## Completed work

- PR #95 merged with merge commit 5a9b4f0ad5634d3df8cc5ee93c5bc3c622056de8. Package metadata now requires Python 3.11; CI tests Python 3.11 and 3.12, runs minimum pins on 3.11, and installs PyArrow in the 3.12 job.
- Historical runtime evidence remains unchanged. Its binding test now excludes packaging metadata because the provenance records the actual measured package versions.
- Superseded PRs #43 and #51 are closed.
- All 27 audited local feature branches and 85 audited origin feature branches were deleted against their expected tip SHAs. The temporary delivery branch was also deleted locally and remotely.
- The four audited nonexistent scratch worktree registrations were pruned. Only the primary main worktree remains.

## Decisive verification

- Required PR checks passed on exact head 5dbd0961fb31afeb2c4ea2b63270d5e0adb2a449: Python 3.11, Python 3.12 with PyArrow and strict docs build, minimum pins, CatBoost, and clean-wheel smoke.
- Local focused docs, Arrow, and runtime-evidence slice: 9 passed. Ruff, generated API reference, wheel metadata (`Requires-Python: >=3.11`), and diff checks passed.
- Post-cleanup live verification found all 27 audited local and 85 audited origin refs absent, both delivery refs absent, PRs #43/#51 closed, PR #95 merged, one local branch (`main`), one origin head (`main`), no stale worktree registrations, and no stashes.
- Main is clean and synchronized with origin after this closeout commit is pushed.

## Blockers

None.
