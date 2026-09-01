# SIFT 0.9 worklog

- Objective: Complete 0.9 Workstream E leakage-safe categorical encoding while preserving every 0.8 categorical/default contract; E1 is complete and E2 contextual folds remain.
- Latest steering: Proceed after Workstream D commit `8263e00`.
- Constraints: Keep sklearn `>=1.3,<2`, existing unsafe encoders behind `allow_full_data_target_encoding=True`, fixed-k rejection of `groups`/`time`, the 58-name top-level surface, and current default `cat_encoding="none"`. One-hot remains blocked on F3 block-aware selection.
- Decisions: Add `cat_encoding="target_cv"` using sklearn `TargetEncoder.fit_transform` for unweighted regression/binary training and target-blind `transform` for inference. Preserve one raw column per selected feature, so multiclass expansion rejects until block-aware selection. Reject weights/groups/time until E2 assigns the already-specified prior, warmup, smoothing, and fold options to public signatures; never fall back to full-data encoding.
- Completed: Workstream D is committed at `8263e00`. E1 now covers function filters, sklearn wrappers, binary CEFS+, Boruta, split-local auto-k evaluation, inference unknown/missing rules, conditional result/fitted metadata, and the no-extra high-cardinality leakage regression.
- Current state: E1 code, tests, documentation, and the rev-13 campaign status are fully verified and ready to commit.
- Next action: Commit E1, then settle and implement E2 weighted/group/time fold policy on the next explicit proceed.
- Blockers: None.
- Decisive verification: The 12-test E1 contract passes on sklearn 1.3.2, 1.5.1, and 1.7.1. Existing affected categorical/selector suites pass 231 with 14 skipped; docs/public-contract checks pass 214; the full primary suite passes 1,649 with 16 skipped. Ruff, compileall, and `git diff --check` are clean.
