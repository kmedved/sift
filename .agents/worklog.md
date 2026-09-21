# Python 3.13 CI support — in progress

Objective: add Python 3.13 to SIFT CI, fix demonstrated compatibility failures, and update current support docs/metadata. Deliver a scoped green PR for Astra review before merge. No 1.0 API/default changes, version bump, tag, release, or PyPI action.

State: PR #100 (https://github.com/kmedved/sift/pull/100) first head 3e61851 added 3.13 to the existing matrix with `numba>=0.61` only there. First run 35548902794 resolved Python 3.13.15 with NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1 and Numba 0.67.0. Its sole failure was the existing shared-DataFrame equality assertion for `KnockoffSelectionResult` (2,791 passed, 34 skipped); all five prior gates passed. Commit 63ad499 restores pre-3.13 tuple comparison. Runtime evidence was rerun from that clean source commit: all 18 data and selection fingerprints match, only `sift/selection/knockoff_filter.py` changed among bound sources. Focused selection-view and runtime-binding checks pass 179/179; Ruff and diff check pass.

Next: commit the refreshed evidence and docs, push a new PR head, and read exact-head CI. If green, update current support metadata and docs from observed success, rerun exact-head CI, then callback to Astra for review. Preserve frozen bakeoff artifacts and do not merge.
