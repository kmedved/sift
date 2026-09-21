# SIFT 1.0 default contract — implementation in progress

Objective: implement the owner-approved `docs/specs/1.0-compatibility-proposal.md` on `codex/1.0-default-contract`, deliver a tested review PR, and keep version/release/merge actions separate. No 1.0 release has shipped.

Decisions: only the Stability/permutation/CatBoost `random_state=None` defaults become 0; public `verbose=True` becomes false; their `n_jobs=-1` defaults become 1; ten selector transformer defaults become `output_order="original"`. Explicit None, explicit legacy order, parallel options, nested learned-prefix order, seed-42 defaults, filter/cache seed semantics, all 66 exports, returns, aliases, sklearn 1.3 floor, and algorithm/count defaults remain.

State: policy/announcement/proposal commit `aebab8d` is independent. Implementation and contract edits are locally complete. Focused contracts passed 327/327 (2 skips), all contracts passed 493/493 (7 skips), and the broader affected slice passed 530/530 (16 skips) after updating quiet-default logging expectations. A first full local suite reached 2,784 passed / 41 skipped with only one stale transform-order assertion plus the expected runtime source-binding failure; the order assertion is fixed and its focused test passes. Ruff and diff check pass. No runtime artifact has been refreshed yet.

Next: commit implementation source/tests/docs, rerun the 18-case benchmark from that clean source commit, verify fingerprints and update binding docs, then run decisive checks, push a draft 1.0 PR, and read exact-head CI. Preserve frozen bakeoff evidence.
