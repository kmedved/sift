# Development

This guide covers local setup, validation, and release-oriented checks for SIFT.

## Setup

Use Python 3.10, 3.11, or 3.12.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Optional dependencies:

```bash
python -m pip install -e ".[categorical]"
python -m pip install -e ".[catboost]"
python -m pip install -e ".[all]"
```

## Validation

Run the full test suite:

```bash
python -m pytest -q
```

Run focused slices while working on specific areas:

```bash
python -m pytest tests/test_docs_smoke.py -q
python -m pytest tests/test_cefsplus.py tests/test_cefsplus_binary.py -q
python -m pytest tests/test_knockoff_sampler.py tests/test_knockoff_filter.py tests/test_knockoff_fdr_control.py -q
python -m pytest tests/test_select_k_auto_no_target_leak.py tests/test_jmi_weights.py -q
python -m pytest tests/test_mrmr_parallel.py tests/test_filter_results.py -q
python -m pytest tests/test_stability_selection.py tests/test_block_bootstrap.py -q
python -m pytest tests/test_selector_classes.py tests/test_boruta_groups_without_time_split.py -q
python -m pytest tests/test_importance.py tests/test_permute.py -q
```

Check formatting-sensitive diffs:

```bash
python -m ruff check sift tests
git diff --check
```

For selector-layer refactors, keep the dispatcher contract honest with:

```bash
rg "direct_result|select_binary_api|build_binary_result|spec\\.name|match spec|is_binary|if ctx\\.estimator|Optional\\[Callable\\]" sift tests
awk 'length($0)>120 {print FILENAME ":" FNR ":" length($0)}' \
  sift/selection/filter_api.py \
  sift/selection/filter_payloads.py \
  sift/selection/filter_auto_k.py
```

The first command should only return unrelated hits; the second should print
nothing.

## Documentation Checks

`DOCS.MD` is the detailed API manual and is used as the package long
description. The docs smoke tests verify that documented top-level exports and
install extras match the package.

```bash
python -m pytest tests/test_docs_smoke.py -q
```

When adding or renaming public exports, update both `sift/__init__.py` and
`DOCS.MD`. If the export is a first-screen workflow such as `select_fdr`, also
update `README.md`, [docs/user-guide.md](user-guide.md), and the release notes.

When moving private selector internals, also check docs and tests for stale
module names:

```bash
rg "mrmr_api|jmi_api|cefsplus_api|cefsplus_binary_api|filter_api_common|api_helpers|stability_api|catboost_api" \
  README.md DOCS.MD docs tests sift
```

## Benchmarks

Benchmark scripts live under `benchmarks/` and emit promotion-oriented JSON.

```bash
python benchmarks/run_benchmarks.py --quick --output /tmp/sift-benchmarks.json
```

Use the individual benchmark scripts when working on a hot path:

```bash
python benchmarks/bench_mrmr.py --quick --output /tmp/bench-mrmr.json
python benchmarks/bench_jmi.py --quick --output /tmp/bench-jmi.json
python benchmarks/bench_permutation.py --quick --output /tmp/bench-permutation.json
python benchmarks/bench_cefsplus.py --quick --output /tmp/bench-cefsplus.json
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
python benchmarks/bench_stability.py --quick --output /tmp/bench-stability.json
```

## CI and Releases

GitHub Actions run tests on Python 3.10, 3.11, and 3.12, plus optional CatBoost
coverage. Publishing is triggered from GitHub releases through the package
upload workflow.

Before release-oriented promotion, run:

```bash
python -m pytest -q
python -m pytest tests/test_docs_smoke.py -q
python -m pytest tests/test_benchmarks.py -q
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
python benchmarks/run_benchmarks.py --quick --output /tmp/sift-benchmarks.json
git diff --check
```

For the 0.7.0 knockoffs bundle, the focused smoke is:

```bash
python -m pytest tests/test_docs_smoke.py tests/test_benchmarks.py -q
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
```

## Generated Files

Tests and Numba compilation may create `__pycache__`, `.nbc`, `.nbi`, and
`catboost_info` artifacts. Keep generated artifacts out of documentation or
source commits unless the project intentionally tracks them.
