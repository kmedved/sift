# SIFT 1.0 default contract — implementation review

Objective: implement the owner-approved `docs/specs/1.0-compatibility-proposal.md` on `codex/1.0-default-contract` and deliver a tested draft PR. Version, release, merge and publication actions remain separate; 1.0 has not shipped.

Decisions implemented: only the Stability/permutation/CatBoost `random_state=None` defaults become 0; public `verbose=True` becomes false; their `n_jobs=-1` defaults become 1; ten selector transformer defaults become `output_order="original"`. Explicit None, explicit legacy order, parallel options, nested learned-prefix order, seed-42 defaults, filter/cache seed semantics, all 66 exports, returns, aliases, sklearn 1.3 floor, and algorithm/count defaults remain. CatBoost dictionaries still win collisions and warn accurately. Experimental and ledger messaging matches retained imports and policy.

State: `aebab8d` is the independent advance announcement/policy commit; `e6021bd` is implementation and contracts; `7943521` binds runtime evidence to the clean implementation commit. All 18 runtime data and selection fingerprints are unchanged; frozen bakeoff evidence is untouched. Local full suite: 2,786 passed, 41 skipped. Contracts: 493 passed, 7 skipped. Ruff and diff check pass; generated API reference is current. Strict MkDocs and optional CatBoost coverage will run in CI with their installed extras.

Next: push the branch, open a draft 1.0 implementation PR, await all exact-head CI jobs, repair only demonstrated failures, then callback to Astra for review. Do not merge or release.
