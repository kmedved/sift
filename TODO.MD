# TODO

Current state and open work. Delivered work lives in `docs/release-notes.md`.

## Version and release state

- `v0.9.0` is published at `94bae05` with the verified wheel and source
  distribution attached. Development has moved to `0.9.1.dev0`.
- Full suite here (Python 3.12, numpy 1.26 / pandas 2.2 / scikit-learn 1.5.1, no
  `catboost` or `category_encoders`): the F8b integration run gives **2,138 passed,
  40 skipped** under warnings-as-errors. All public docstring
  examples and the six remaining manual sources still execute independently.
- Releases are GitHub Releases with the built sdist and wheel attached. **No
  PyPI publication, ever.** The workflow builds, metadata-checks, clean-installs
  and attaches the artifacts; tagging `v` + `sift.__version__` and creating the
  Release stay explicit owner actions. Pre-release checks: `docs/development.md`.
- [x] A numpydoc docstring for every `sift.__all__` export, pinned by two tests:
      `tests/test_docstring_coverage.py` for structure (summary line, 8 non-empty
      lines, a `name : type` entry for every signature parameter including
      `*args`/`**kwargs`, `Returns`/`Yields` for functions, `Examples` for every
      export) and `tests/test_docstring_examples.py`, which executes each
      docstring's `>>>` statements under warnings-as-errors.
- [x] CI execution of every fenced `python` block in the manual set:
      `tests/test_docs_examples.py` for the six `docs/` manuals and `DOCS.MD`,
      `tests/test_docs_smoke.py` for the README.
- [x] 0.9.0 release preparation: bump `sift.__version__` to `0.9.0`, finalize
      the release-notes section, and run the pre-release smoke from
      `docs/development.md` on the release PR.
- [x] Owner release action: `v0.9.0` and its GitHub Release were published at
      `94bae05`; the asset-only workflow attached both verified distributions
      and did not publish to PyPI.

## Target encoding status

- `cat_encoding="target_cv"` is leakage-safe by fold centering: training rows
  emit `fold_encoding - fold_training_prior` and inference rows
  `full_fit_encoding - full_training_prior`, so an unseen category maps to zero
  rather than to a prior that identifies its own fold. Unique-ID, group-proxy
  and timestamp-proxy columns now encode to constant zero.
- The residual boundary is documented, not removed, and deliberately so: a level
  seen twice or more inside a fold still transmits its sibling rows' targets, so
  a *near*-unique identifier whose rows share a latent target stays selectable
  (measured `corr(enc, y) ~ 0.88` on 300 IDs x 2 rows). Pinned by the three
  `test_near_unique_ids_*` / `test_grouping_an_identifiers_rows_*` cases in
  `tests/contracts/test_target_cv_encoding.py`.
- The full-data supervised encoders (`loo`, `target`, `james_stein`,
  `loo_logit`) still need `allow_full_data_target_encoding=True` and are not
  cross-fitted; `KnockoffSelector` rejects `target_cv` outright.

## Re-scoped to 0.9.1 (owner decision, 2026-09-02)

0.9.0 kept two G-docs items — a numpydoc docstring for every `sift.__all__`
export, and CI execution of every fenced `python` block in the manual set — and
both are done. The rest of §G-docs in `docs/specs/0.9-product-layer.md` moves to
0.9.1:

- [x] Generated API reference from those docstrings, which retires `docs/API.md`.
- [x] One selector decision tree replacing the five "which selector" tables.
- [x] A measured runtime/scaling table from a committed script with provenance.
- [x] A data-type support matrix.
- [x] A glossary.
- [x] Convert `docs/user-guide.md` into the promised tutorial.
- [x] Refresh `runtime_scaling_2026-09-03` from clean implementation commit
      `dde1f50b2f9aa68539d5bd7c25003cc9cf8e1027` with `dirty=false`.
- [x] 0.9.1 integration closeout: pushed, required GitHub CI jobs passed, and
      merged as PR #78 at `3b9ac0a`. PyPI publication did not follow.

Do not treat the tutorial page or the clean runtime-scaling refresh as open
documentation work. The 0.9.1 integration closeout is merged; PyPI remains
out of scope.

The `group_col`/`sample_weight_col` and `StabilitySelector` `alpha`/`penalty`
aliases are permanent as of the same decision: no deprecation, no 1.0 removal.

## 0.9.x capability backlog

Authoritative list and design detail: `docs/specs/0.9-product-layer.md` §F, in
the accepted order from `docs/specs/0.9-next-steps.md` Stage 4.

- [x] Router/adapter split of `selection/view.py`, `selection/filter_auto_k.py`,
      and `selection/auto_k.py`. Merged as PR #78 at `3b9ac0a` (implementation
      commit `fa74d63`).
- [x] F1 conditioning sets (`include=`, `exclude=`, `candidates=`).
      Merged as PR #79 at `14bfb5c` (implementation `b2a11bd`, artifact
      `682af22`; required GitHub CI green).
- [x] F2 proxy/redundancy and cluster reports.
      Merged as PR #80 at `17fe3bf` (implementation `081ca04`, artifact
      `a495f33`; all six required GitHub CI jobs green).
- [x] F7 panel within/between transforms.
      Merged as PR #81 at `76e4d51` (implementation `2d04754`, artifact
      `d3938e4`; Codex/Opus review accepted, full local suite and all six
      required GitHub CI jobs green).
- [x] F8a knockoff feasibility and UX metadata, implementation `308dfe1`.
      Merged as PR #82 at `40f8af7` (artifact `0c62a44`, CI test correction
      `449f063`). Codex/Opus review accepted; clean runtime evidence, full
      local gate and all six required GitHub CI jobs passed.
- [x] F8b opt-in e-value aggregation, implementation `dde1f50`: Codex/Opus
      review accepted; validated relevance/ridge and explicit exploratory modes.
      Merged as PR #83 at `f70a3da` (artifact `2a05c66`); clean runtime
      evidence, full local gate and all six required GitHub CI jobs passed.
- [x] F8c statistic bakeoff, implementation `ae904b8`, retained CSV SHA256
      `40d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf`.
      Codex/Opus accepted; merged as PR #84 at `cfd2f64` after all six required
      CI jobs passed on `20d07d1`. No 0.9 default change.
- [x] F3 block/group-aware selection (`k` counts blocks). Fixed-k Gaussian/
      classic, block-aware auto-k, and binary log-loss joint blocks
      (`evaluate`/`elbow`/`penalized_objective`/`auto`) are implemented.
      Codex/Opus review accepted; clean implementation `2586a2c` and refreshed
      runtime evidence `9f91781`. PR #85 merged at `e9a4464` after all six
      required CI jobs passed on the exact head; local gate 2200 passed / 40 skipped.
- [x] E4 one-hot encoding on atomic blocks. Codex/Opus accepted; clean
      implementation `826e10b` and bound runtime refresh `1a99aad`.
      PR #86 merged at `c5e1d51` after all six CI jobs passed on the exact
      head. Local gate 2221 passed / 40 skipped, no exclusions.
- [x] F9 leakage-safe `sift.compare`, with selector fitting inside every fold.
      Codex/Opus accepted, clean implementation `6ecbe63`; bound runtime
      refresh `69410cb`. PR #87 merged at `7314355` after all six required
      CI jobs passed on the exact head. Full local gate 2243 passed / 40 skipped,
      no exclusions; focused fixture-compatibility tests also pass.
- [x] Reproducibility manifests (`result.reproducibility_`).
      Codex/Opus review accepted; clean implementation `c322671` and bound
      runtime refresh complete. Full local run: 2261 passed / 40 skipped,
      with its sole glossary-order failure now fixed and targeted checks green.
      All six CI jobs passed on `bb651a2`; PR #88 merged at `c29e99e`.
- [x] F4 `Stabilized(selector)` meta-selector, after F8b.
      Codex/Opus accepted the implementation on `codex/0.9x-stabilized`.
      Clean implementation `7cc4f66` has refreshed `dirty=false` runtime evidence.
      Full local gate: 2290 passed / 40 skipped, no exclusions; 26 focused
      checks pass after a test-only sklearn 1.3 compatibility correction.
      All six CI jobs passed on `bfe5765`; PR #89 merged at `18cdf40`.
      F6 and later remain open.
- [x] F5 multi-target CEFS+. Implementation on
      `codex/0.9x-multi-target-cefsplus` (joint 2-D `y`, multi-output
      evaluate, `df=q·k` IC after derivation). Codex/Opus accepted after
      verified corrections; 375 focused checks passed / 2 skipped plus
      independent conditional/weighted public probes. Clean implementation
      `c7ea9da` has bound `dirty=false` runtime evidence (18/18 fingerprints
      unchanged). Full local gate: 2303 passed / 40 skipped, no exclusions;
      strict docs and static checks pass. All six CI jobs passed on `fe14516`;
      PR #90 merged at `0ee513b`, with clean-source ancestry preserved.
- [ ] F6 generic `ModelSelector` with purged/embargoed time-series splits.
      Split primitives accepted. Generic `ModelSelector.fit` and the CatBoost
      preset `catboost_select` share the internal `run_selection` backend
      contract (`_GenericModelBackend` / `_CatBoostNativePreset`) on
      `codex/0.9x-model-selector`. Codex/Opus accepted the implementation,
      including the executable support-matrix row and raw-column pipeline
      importance guard. Latest focused gate: 267 passed / 3 skipped.
      Clean-commit runtime refresh, full local gates and exact-head CI/merge
      remain pending.
- [ ] Standardized classic-selector caches, then leakage-free unsupervised
      `ordinal`/`frequency` categorical fallbacks.

## Closed history

The Oracle/source review, the follow-up LLM review, the 2026-08 deep dive, the
0.7.0 knockoff bundle, the 0.8.0 performance and correctness bundle, and the 0.9
product-layer campaign (sklearn contracts, the centered `target_cv` correction,
`SelectionView`/`as_result`, Auto-K presets and option groups,
`sift.experimental`, CI hardening) are closed; per-item detail and the
deprecation ledger are in `docs/release-notes.md`. 0.9.0 supersedes the 0.8.0
work, which was never tagged or released.
