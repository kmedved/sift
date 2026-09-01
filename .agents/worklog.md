# SIFT 0.9 worklog

- Objective: Complete Workstream C from `docs/specs/0.9-product-layer.md`: additive `AutoKConfig` presets, immutable option-group views/builders, field-semantics documentation and unused-field validation, plus `sift.experimental`.
- Latest steering: Proceed after B2 commit `910d475`.
- Constraints: Preserve all 49 flat `AutoKConfig` dataclass fields and their defaults/signature/equality/repr/replace/pickle behavior; preserve the exact 58-name `sift.__all__`; no algorithmic/default-path changes; minimum sufficient implementation and verification.
- Decisions: Treat option groups as synthesized immutable views and builder inputs only; keep overlapping fields distinct; experimental access warns without removing current top-level exports.
- Completed: B2 committed; repository was clean at start of C. Three independent read-only C audits resolved preset mappings, option ownership, namespace exports, and acceptance checks. C code, focused contract tests, release notes, API/manual documentation, campaign status, and installed-wheel verification are complete.
- Current state: Workstream C is complete. The flat dataclass remains 49 fields and the ordered top-level surface remains 58 names; no algorithmic defaults or selections changed.
- Next action: None for C; commit the completed workstream.
- Blockers: None.
- Decisive verification: rich affected suites 346 passed / 6 skipped; final rich contracts 97 passed; base contracts 203 passed; docs smoke 4 passed; clean Python 3.12 wheel install verified; full Ruff, compileall, and diff checks clean.
