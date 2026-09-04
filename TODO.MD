# TODO

Current state and open work. Delivered work lives in `docs/release-notes.md`.

## Version and release state

- `v0.9.0` is published at `94bae05` with the verified wheel and source
  distribution attached. Development has moved to `0.9.1.dev0`.
- Full suite here (Python 3.12, numpy 1.26 / pandas 2.2 / scikit-learn 1.5.1, no
  `catboost` or `category_encoders`): `python -m pytest -q` gives **1,982 passed,
  40 skipped**. The matrix-stage count was 1,973 / 40. All public docstring
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
      `5f3f1078cac76477562249d456fca7c01c6edcd4` with `dirty=false`.
- [ ] 0.9.1 integration closeout: commit this refreshed artifact, push, run
      full GitHub CI (including strict MkDocs and optional-dependency jobs),
      and merge. PyPI publication does not.

Do not treat the tutorial page or the clean runtime-scaling refresh as open
documentation work. GitHub CI has not been claimed for this closeout yet.

The `group_col`/`sample_weight_col` and `StabilitySelector` `alpha`/`penalty`
aliases are permanent as of the same decision: no deprecation, no 1.0 removal.

## 0.9.x capability backlog

Authoritative list and design detail: `docs/specs/0.9-product-layer.md` §F, in
the accepted order from `docs/specs/0.9-next-steps.md` Stage 4.

- [ ] F1 conditioning sets (`include=`, `exclude=`, `candidates=`).
- [ ] F2 proxy/redundancy and cluster reports.
- [ ] F7 panel within/between transforms.
- [ ] F8a knockoff feasibility and UX metadata; F8b/F8c validated e-value
      aggregation and the statistic bakeoff.
- [ ] F3 block/group-aware selection (`k` counts blocks).
- [ ] F4 `Stabilized(selector)` meta-selector, after F8b.
- [ ] F5 multi-target CEFS+; F6 generic `ModelSelector` with purged/embargoed
      time-series splits.
- [ ] Outside §F: split `selection/view.py`, `selection/filter_auto_k.py` and
      `selection/auto_k.py` along their adapter/router boundaries before any of
      the above; then the reusable standardized/cache path for classic
      selectors, leakage-free unsupervised `ordinal`/`frequency` fallbacks, and
      reproducibility manifests (`result.reproducibility_`).

## Closed history

The Oracle/source review, the follow-up LLM review, the 2026-08 deep dive, the
0.7.0 knockoff bundle, the 0.8.0 performance and correctness bundle, and the 0.9
product-layer campaign (sklearn contracts, the centered `target_cv` correction,
`SelectionView`/`as_result`, Auto-K presets and option groups,
`sift.experimental`, CI hardening) are closed; per-item detail and the
deprecation ledger are in `docs/release-notes.md`. 0.9.0 supersedes the 0.8.0
work, which was never tagged or released.
