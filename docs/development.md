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
python -m pytest tests/test_mrmr_parallel.py tests/test_jmi_hot_loop.py -q
python -m pytest tests/test_stability_selection.py tests/test_block_bootstrap.py -q
python -m pytest tests/test_importance.py tests/test_permute.py -q
```

Check formatting-sensitive diffs:

```bash
git diff --check
```

## Documentation Checks

`DOCS.MD` is the detailed API manual and is used as the package long
description. The docs smoke tests verify that documented top-level exports and
install extras match the package.

```bash
python -m pytest tests/test_docs_smoke.py -q
```

When adding or renaming public exports, update both `sift/__init__.py` and
`DOCS.MD`.

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
python benchmarks/run_benchmarks.py --quick --output /tmp/sift-benchmarks.json
git diff --check
```

## Generated Files

Tests and Numba compilation may create `__pycache__`, `.nbc`, `.nbi`, and
`catboost_info` artifacts. Keep generated artifacts out of documentation or
source commits unless the project intentionally tracks them.
