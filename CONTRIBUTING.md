# Contributing to SIFT

Thanks for helping improve SIFT. This repository moves quickly, so the best
contributions are small, tested, and explicit about which public API surface
they change.

## Development Setup

SIFT requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Install optional extras only when the change needs them:

```bash
python -m pip install -e ".[categorical]"
python -m pip install -e ".[catboost]"
python -m pip install -e ".[docs]"
python -m pip install -e ".[all]"
```

Core dependencies are declared in `pyproject.toml`. Optional extras are kept
small so importing `sift` does not require CatBoost or categorical encoders.

## Working Rules

- Keep public behavior backwards compatible unless the change is explicitly a
  release-breaking cleanup.
- Update tests and docs in the same patch when a public function, selector
  class, result object, warning, or metadata key changes.
- Prefer the shared validation and preprocessing helpers over local one-off
  parsing.
- Keep selector behavior deterministic when `random_state` is fixed.
- Do not broaden optional dependencies into required dependencies.

## Documentation Expectations

Public API changes should update the relevant docs:

- `DOCS.MD` is the canonical full manual and is checked by docs smoke tests.
- `README.md` should remain a fast orientation page.
- `docs/reference/` is generated from `sift.__all__`; run
  `python scripts/generate_api_reference.py` after changing the public surface.
- `docs/data-type-support.md` is generated from live public-entry-point
  probes; rerun `python scripts/generate_data_type_matrix.py` after changing
  accepted input kinds or row-metadata contracts.
- `docs/ALGORITHMS.md` explains method behavior and assumptions.
- `docs/ADVANCED.md` covers workflow patterns such as time splits, caches,
  sample weights, smart sampling, and knockoffs.
- `docs/release-notes.md` should receive a user-facing note for release-surface
  changes.

For 0.7.0-era knockoff work, be precise about the guarantee language:
`select_fdr` reports approximate plug-in Gaussian-copula validity metadata
unless the fitted feature model is the true Model-X distribution.

## Running Tests

Run the focused tests that match your change, then run the docs smoke test if
you touched public docs or exports.

```bash
python -m pytest -q tests/test_smoke.py
python -m pytest -q tests/test_docs_smoke.py
python scripts/generate_api_reference.py --check
mkdocs build --strict
```

Useful focused slices:

```bash
python -m pytest -q tests/test_selector_classes.py
python -m pytest -q tests/test_knockoff_filter.py
python -m pytest -q tests/test_knockoff_sampler.py
python -m pytest -q tests/test_knockoff_fdr_control.py
python -m pytest -q tests/test_boruta.py
python -m pytest -q tests/test_catboost_selection.py
python -m pytest -q tests/test_smart_sample_basic.py
```

Before a release-oriented patch, run:

```bash
python -m pytest -q
```

## Code Style

Use the style already present in the touched module. In general:

- Prefer type hints on public functions and dataclasses.
- Keep comments focused on non-obvious invariants or algorithmic choices.
- Use NumPy/pandas/scikit-learn APIs directly instead of ad hoc parsing when
  possible.
- Keep result tables stable: column names and ordering are part of the practical
  API for downstream users.
- Raise `ValueError` with actionable messages for invalid user input.

## Adding a Selector or Public Option

A selector or option is not merge-ready until it has:

- Input validation tests.
- At least one deterministic smoke test on pandas input.
- ndarray behavior checked when ndarray input is supported.
- Sample-weight tests if the option claims weight support.
- Documentation in `DOCS.MD` and the relevant focused docs.
- Release notes when it changes the shipped surface.

For sklearn-style selector classes, also check:

- `fit`, `transform`, `fit_transform`, and `get_support`.
- pandas column-order validation.
- failed-refit state cleanup.
- `sklearn.base.clone` compatibility when constructor defaults use sentinels or
  mutable-looking values.

## Pull Request Checklist

- [ ] Tests pass locally for the affected area.
- [ ] `tests/test_docs_smoke.py` passes when exports or docs changed.
- [ ] New public APIs are documented.
- [ ] Release notes mention user-facing behavior changes.
- [ ] Optional dependencies remain optional.
- [ ] Randomized behavior is reproducible under `random_state`.

## Issue Reports

For bugs, include:

- SIFT version or commit.
- Python version and operating system.
- Minimal reproducible example.
- Input shape and dtypes.
- Expected result and actual result.
- Full traceback for exceptions.

For feature requests, include the workflow you are trying to support and the
shape of the API you would like to use.
