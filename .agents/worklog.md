# SIFT 0.9 worklog

- Objective: Complete 0.9 Workstream D sklearn integration without changing any 0.8 selection, return, default, or legacy-order contract.
- Latest steering: Proceed after 0.9.0b1 commit `662b352`.
- Constraints: Keep sklearn `>=1.3,<2`, the 49-field `AutoKConfig`, the ordered 58-name top-level surface, dense-only selector inputs, and every selector family's legacy transform order.
- Decisions: Add `output_order="original"` as an opt-in while defaulting to legacy order. Make row-metadata requests explicit and reject configured requests the selected fit path cannot consume. Scope private sklearn CV/pipeline calls out of an outer routing context. Document, rather than silently change, contextual inner Ridge CV and fully weighted Stability alpha scoring.
- Completed: All eight public selectors subclass `SelectorMixin`; support masks, ordered indices/names/transforms, dense inverse transforms, sparse rejection, tags, and metadata-request guards are implemented. Group metadata routes through Pipeline/cross_validate on sklearn 1.5.1 and 1.7.1. Six common estimator checks are pinned for every class. Public docs, release notes, and the rev-12 campaign status describe the shipped and compatibility-gated behavior.
- Current state: Workstream D is complete and its code, contracts, documentation, and packaged wheel are verified.
- Next action: Begin Workstream E categoricals on the next explicit proceed.
- Blockers: None.
- Decisive verification: The expanded D contract and related focused tests pass 55 on sklearn 1.5.1; the original 37-test D matrix passes on sklearn 1.7.1; affected selector suites pass 247 with 7 skipped; the full primary suite passes 1,637 with 16 skipped. Ruff, compileall, and `git diff --check` are clean. The isolated wheel build installs from `dist` and passes a 58-export MRMR fit/transform/inverse smoke test.
