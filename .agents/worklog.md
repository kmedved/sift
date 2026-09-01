# SIFT 0.9 worklog

- Objective: Complete the 0.9.0b1 advanced `SelectionView` operations without moving the later F2 cluster/report program into this milestone.
- Latest steering: Proceed after Workstream C commit `8a17aab`.
- Constraints: Preserve legacy result classes, constructors, default returns, pickles, and the ordered 58-name top-level surface; never retain source `X`; proxy storage is explicit, float32, position-safe with duplicate labels, and capped at 64 MiB.
- Decisions: Limit initial proxy producers to `select_cached` and Gaussian filter routes, including the binary Brier delegate. Defer non-Gaussian producers and cluster-frequency aggregation to F2. Reject partial-table plots rather than presenting incomplete data as complete.
- Completed: Three independent read-only audits confirmed the narrow gaps. Implemented bounded selection-time proxy storage, duplicate-safe name/position lookup, unsupported-route validation, defensive proxy normalization, pickle preservation for opt-in filter results, partial-plot degradation, docs, release notes, and installed-wheel coverage. Workstreams A, B2, and C are complete.
- Current state: Changes are verified and ready to commit on `main`, which was 12 commits ahead of `origin/main` before this commit.
- Next action: Commit the completed 0.9.0b1 stage; Workstream D sklearn integration is next.
- Blockers: None.
- Decisive verification: Primary affected matrix `436 passed, 9 skipped`; base-environment compatibility slice `240 passed`; full Ruff and compile checks clean; clean-built wheel installed with current dependencies and passed `scripts/verify_wheel_install.py` from outside the source tree.
