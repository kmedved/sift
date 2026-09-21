# Python 3.13 CI support — in progress

Objective: add Python 3.13 to SIFT CI, fix demonstrated compatibility failures, and update current support docs/metadata. Deliver a scoped green PR for Astra review before merge. No 1.0 API/default changes, version bump, tag, release, or PyPI action.

State: PR #100 (https://github.com/kmedved/sift/pull/100) first head 3e61851 added 3.13 to the existing matrix with an interpreter-specific `numba>=0.61` preinstall. First run 35548902794 resolved Python 3.13.15 with NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, Numba 0.67.0 and failed one of 2,792 active tests: the existing shared-DataFrame equality assertion for `KnockoffSelectionResult`. Python 3.13's generated dataclass equality evaluates the DataFrame boolean; an explicit tuple comparison restores the earlier identity short-circuit. The focused test and Ruff pass locally. Source changed in `sift/selection/knockoff_filter.py` and runtime evidence must be refreshed from a clean commit before pushing.

Next: commit the narrow source fix, rerun the 18-case runtime artifact from clean source and verify unchanged data/selection fingerprints, update its docs and binding, then push. Read exact-head CI; if green, update current support metadata and docs from observed evidence, rerun CI and callback to Astra for review. Preserve frozen bakeoff artifacts and do not merge yet.
