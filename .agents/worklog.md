# SIFT 0.9 worklog

- Objective: Complete 0.9 Workstream E leakage-safe categorical encoding while preserving every 0.8 categorical/default contract; E1 and E2 are complete.
- Latest steering: Proceed through Workstream E2 and land the verified stage.
- Constraints: Keep sklearn `>=1.3,<2`, existing unsafe encoders behind `allow_full_data_target_encoding=True`, fixed-k rejection of `groups`/`time`, the 58-name top-level surface, and current default `cat_encoding="none"`. One-hot remains blocked on F3 block-aware selection.
- Decisions: E1 uses sklearn `TargetEncoder.fit_transform` for unweighted regression/binary training and target-blind `transform` for inference. E2 uses explicit finite nonnegative smoothing for weighted/group/time modes, fold-local weighted maps, whole-group holdouts, strict-history tied-timestamp folds, and effective zero weights for temporal warmup when no explicit prior is supplied. It preserves fixed-k `groups`/`time` rejection and one raw output column per feature, so multiclass expansion and one-hot remain blocked on F3.
- Completed: Workstream D is committed at `8263e00`. E1 is committed at `63e32c3`. E2 adds weighted, grouped, and temporal `target_cv` encoding across function filters, nested auto-k evaluation, selector wrappers, binary CEFS+, Boruta, and knockoffs, with truthful metadata and target-blind inference.
- Current state: E2 implementation, docs, and regression tests are committed on `main`. The worktree is clean; no push was requested.
- Next action: Proceed to the remaining Workstream E categorical fallbacks; one-hot waits for F3 block-aware selection.
- Blockers: None.
- Decisive verification: E2's focused contract passes 29 tests on sklearn 1.7.1. The affected route matrix passes 508 with 17 skipped; docs/release checks pass 65. The base full suite passes 1,573 with 23 skipped and the primary sklearn 1.7 suite passes 1,666 with 16 skipped. Ruff, compileall, and `git diff --check` are clean.
