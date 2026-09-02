# TODO

Current state and open work. Delivered work lives in `docs/release-notes.md`.

## Version and release state

- `sift.__version__` is `0.9.0a1`. A small release PR bumps it to `0.9.0` and
  retitles the release-notes section; a later commit moves it to `0.9.1.dev0`.
- Full suite here (Python 3.12, numpy 1.26 / pandas 2.2 / scikit-learn 1.5.1, no
  `catboost` or `category_encoders`): `python -m pytest -q` gives **1,965 passed,
  39 skipped** in ~110 s. 37 skips are optional-dependency gates (`catboost`,
  `category_encoders`, the four literal-block docstring examples) and 2 are the
  sklearn name-based transform skips on `BorutaSelector`/`StabilitySelector`.
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
- [ ] Remaining 0.9.0 gate: the release PR. Bump `sift.__version__` from
      `0.9.0a1` to `0.9.0`, retitle the release-notes section, run the
      pre-release smoke (`docs/development.md`), and merge. The owner then cuts
      the `v0.9.0` tag and GitHub Release; a separate commit afterwards moves
      `__version__` to `0.9.1.dev0`.

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

- [ ] Generated API reference from those docstrings, which retires `docs/API.md`.
- [ ] One selector decision tree replacing the five "which selector" tables.
- [ ] A measured runtime/scaling table from a committed script with provenance.
- [ ] A data-type support matrix.
- [ ] A glossary.

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
