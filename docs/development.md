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

### Test markers

`pyproject.toml` sets `testpaths = ["tests"]`, so a bare `python -m pytest`
collects the suite from anywhere in the repository. Three markers are registered:

| marker | meaning |
| --- | --- |
| `slow` | Monte-Carlo or large-fixture tests: the `test_knockoff_fdr_control.py` seed loops, the Auto-K null-calibration simulation, the 12k-row D10 design, and the 25k-row knockoff sampler draw. |
| `catboost` | Needs the optional `catboost` dependency. |
| `categorical` | Needs the optional `category_encoders` dependency. |

The markers sit beside the existing `pytest.importorskip` gates rather than
replacing them, so the suite still skips cleanly without the optional packages;
the markers exist so a run can *select* those tests. Useful selections:

```bash
python -m pytest -m "not slow" -q          # ~11 tests fewer, ~60s faster
python -m pytest -m slow -q                # the Monte-Carlo tests only
python -m pytest -m "catboost or categorical" -q
```

### Warning policy

`filterwarnings` starts from `error`: any warning that escapes a test fails the
run. The allowlist is deliberately tiny and every entry names an exact message
prefix *and* a category — never a bare category. Adding an entry requires a
comment saying why the project cannot fix the warning at its source.

The current allowlist has one entry: loky's `DeprecationWarning` about calling
`fork()` from a multi-threaded process. It is emitted by
`joblib/externals/loky/backend/fork_exec.py` whenever the process backend starts
workers after a thread pool exists, it depends on how many threads happen to be
running at fork time (so it appears and disappears between runs on the same
machine), and nothing in this project can prevent it.

Everything else is handled at the site instead of suite-wide:

- A warning a test *intends* to trigger is asserted with `pytest.warns`. Where a
  call emits several legitimate warnings, record them all and assert the one you
  mean — `with pytest.warns(UserWarning) as record:` followed by
  `assert any("..." in str(w.message) for w in record)`. pytest 7 silently
  discarded non-matching warnings inside `pytest.warns`; pytest 8+ re-emits them,
  where `error` turns them into failures, so the plain `match=` form is fragile
  for calls with more than one advisory.
- A warning a single test incidentally triggers gets a local
  `@pytest.mark.filterwarnings("ignore:...")` with a comment, so the exemption
  stays visible next to the test that needs it.
- A warning that says an option is inert (`AutoKConfig.<field> is set but
  k_method=... does not use it`) means the fixture is wrong: drop the inert
  field. `sift/selection/auto_k.py::_warn_unused_method_fields` is the authority
  on which field each method consumes.

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

`README.md` is the package long description; its links must therefore be
absolute and render correctly outside a repository checkout. `DOCS.MD` is the
detailed API manual. The docs smoke tests verify the README examples,
documented top-level exports, and install extras.

```bash
python -m pytest tests/test_docs_smoke.py -q
```

When adding or renaming public exports, update both `sift/__init__.py` and
`DOCS.MD`. If the export is a first-screen workflow such as `select_fdr`, also
update `README.md`, [docs/user-guide.md](user-guide.md),
[docs/results.md](results.md), and the release notes.

### Docstring standard

Every name in `sift.__all__` carries a substantive
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstring.
This is a 0.9.0 release gate, not a style preference: the docstrings are the
source the generated API reference will be built from in 0.9.1, so an export
without one silently removes a page from that reference.

Two tests enforce the standard. `__version__` is exempt from both; every other
name in `sift.__all__` is in scope.

`tests/test_docstring_coverage.py` checks structure:

| requirement | applies to |
| --- | --- |
| a non-empty summary line | every export |
| at least 8 non-empty docstring lines | every export |
| every signature parameter named as a `name : type` entry | functions, under `Parameters`; classes, under `Parameters` or `Attributes`, read off `__init__` |
| a `Returns` or `Yields` section | functions |
| an `Examples` section with at least one runnable `>>>` statement | every export, except the four optional-dependency exports pinned by name in `LITERAL_EXAMPLE_EXPORTS` (`select_boruta_shap`, `catboost_select`, `catboost_regression`, `catboost_classif`), whose examples are literal blocks that must name the dependency; the set is asserted exactly, so it cannot grow silently |

The parameter check includes `*args` and `**kwargs`: a variadic in the signature
has to appear as its own entry, spelled with the stars. For a class the two
sections are pooled, so a constructor argument documented under `Attributes`
rather than `Parameters` still satisfies the check.

`tests/test_docstring_examples.py` checks that the examples run. It parses each
docstring with `doctest` and executes the `>>>` statements under
warnings-as-errors, so an example that emits an unasserted `UserWarning` or
`FutureWarning` fails the suite. It deliberately does **not** compare printed
output: NumPy 2 scalar reprs differ across the CI matrix, so an expected-output
line would pin the docstring to one interpreter. It also runs the `Examples`
sections of the public methods and properties that exported classes define
inside `sift` (ids like `SelectionView.proxies`), executes a statement whose
documented output is a traceback inside `pytest.raises` with the exception
type checked, and runs each case in a fresh namespace with a temporary
working directory. Only three things leave an example unrun — a leading
`# doctest: +SKIP` (a later one skips just that statement), a missing
CatBoost extra when the section uses it, and the pinned literal blocks above.

**What neither test checks: defaults and accepted values.** Nothing compares the
`default=...` text or the listed choices in a description against the actual
signature, so a docstring can name a stale default and stay green. That stays a
review responsibility.

Beyond the mechanical floors, a good docstring also has a `Returns` section
naming the concrete type rather than a bare `object`, and `Raises` for the
validation errors the entry point owns. A class documents its constructor
parameters and its fitted attributes (the trailing-underscore ones) rather than
repeating the function docstring it wraps. Keep parameter descriptions honest
about defaults that are scheduled to change in 1.0 — the deprecation ledger in
[docs/release-notes.md](release-notes.md) is the list.

### Executable documentation blocks

Every fenced `python` block in the manual set executes in CI. The covered files
are:

- `README.md`
- `DOCS.MD`
- `docs/API.md`
- `docs/user-guide.md`
- `docs/ADVANCED.md`
- `docs/troubleshooting.md`
- `docs/results.md`

**Blocks are standalone by default.** Each block runs in a fresh namespace, so
it must build its own `X`, `y`, and imports. Prefer tiny synthetic data
(a few hundred rows and a handful of columns) over `make_regression` at scale:
the runner allows roughly 20 s per block, and the suite runs under
warnings-as-errors, so a block that emits an unasserted `UserWarning` or
`FutureWarning` fails the run exactly as a test would.

Three HTML comments change how a block is handled. Put the marker immediately
before the fence; one blank line between marker and fence is allowed.

| marker | effect |
| --- | --- |
| `<!-- sift-doc: skip reason="..." -->` | do not execute; the reason is required |
| `<!-- sift-doc: requires=catboost -->` | execute only when the named module imports; any module name works (`matplotlib`, `category_encoders`), and several `requires=` tokens may be combined in one marker |
| `<!-- sift-doc: continues -->` | execute in the *previous* block's namespace |

Use `continues` only where the narrative genuinely builds across two blocks —
fitting in one and inspecting the fitted object in the next, for example. A
chain of `continues` blocks fails as a unit and is much harder to debug than
one self-contained block, and a reader who lands on the page from a search
result cannot copy it.

`bash` blocks are not executed; neither are blocks in the specs, the release
notes, `docs/architecture.md`, `docs/ALGORITHMS.md`, or `CONTRIBUTING.md`.

All three gates — docstring coverage, docstring examples, and manual block
execution — run as part of the ordinary `python -m pytest -q`, so a docs-only
change still needs a suite run before it ships.

### Stale references

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

GitHub Actions run tests on Python 3.10, 3.11, and 3.12, plus a `min-pins` job on
the declared dependency floors, optional CatBoost coverage, a clean-wheel
installation smoke test, a scheduled latest-dependency canary, and a scheduled
quick benchmark gate. Every job sets `timeout-minutes` and uses `cache: pip`, and
the workflow declares a `concurrency` group that cancels superseded pull-request
runs while letting branch and scheduled runs finish so their artifacts stay
complete. Published GitHub releases build and metadata-check source and wheel
distributions, clean-install the exact wheel, and attach those distributions to
the GitHub Release. This project does not publish those artifacts to PyPI.

### The `min-pins` job

`min-pins` installs every direct runtime requirement at its advertised floor and
then installs the package with `--no-deps` so pip cannot quietly upgrade them:

```bash
python -m venv /tmp/sift-min-pins
/tmp/sift-min-pins/bin/python -m pip install \
  "numpy==1.24.*" "pandas==2.0.*" "scikit-learn==1.3.*" "scipy==1.10.*" \
  "numba==0.59.*" "joblib==1.3.*" "threadpoolctl==3.1.*" pytest
/tmp/sift-min-pins/bin/python -m pip install -e . --no-deps
/tmp/sift-min-pins/bin/python -m pytest -q
```

The floors are mutually consistent and resolve to numpy 1.24.4, pandas 2.0.3,
scikit-learn 1.3.2, scipy 1.10.1, numba 0.59.1, joblib 1.3.2 and
threadpoolctl 3.1.0. numba 0.59 constrains numpy to `<1.27`, which is what makes
1.24 the binding floor rather than an arbitrary one. CI runs this on Python 3.10;
3.11 is a valid local stand-in, since every one of these pins ships wheels for
both.

Two behaviours differ at the floor and are worth knowing before you debug a
failure there: older LAPACK returns a slightly smaller minimum eigenvalue, so the
Gaussian knockoff shrinkage advisory fires where it does not on newer NumPy/SciPy;
and `matplotlib` is not a declared dependency, so the one plotting test skips.

### Python 3.13

Still not enabled, but the reason it was deferred is gone. numba ships cp313
wheels from 0.61.0 (llvmlite 0.44.0), above the 0.59 floor, and a local Python
3.13.15 run with numba 0.67 reached 1,565 passed / 3 failed. Those three were
dependency versions rather than the interpreter — they broke the 3.11/3.12
matrix identically, because `scikit-learn>=1.3,<2` and `numpy>=1.24,<3` resolve
straight to them — and they are the numpy 2.5 failures now fixed. A 3.13 run has
not been repeated since, so enable the job below and read its first result
rather than assuming it is green.

### Verified dependency band

**The whole declared band is supported — there is no ceiling below what
`pyproject.toml` allows.** Each row is a full-suite run under the
warnings-as-errors policy:

| dependency set | result |
| --- | --- |
| floors (numpy 1.24.4 / pandas 2.0.3 / sklearn 1.3.2 / scipy 1.10.1 / numba 0.59.1), Python 3.11 | green — 1,566 passed / 30 skipped |
| **base** (numpy 1.26.4 / pandas 2.2.2 / sklearn 1.5.1), Python 3.12 | green — 1,967 passed / 39 skipped (10 doc-block and 4 docstring-example skips are optional-dependency gates) |
| numpy 2.4.6 / pandas 2.3.3 / sklearn 1.7.2 | green |
| numpy 2.5.2 / pandas 2.3.3 / sklearn 1.7.2 / scipy 1.18.1 / numba 0.67.0, Python 3.12 | green — 1,680 passed / 30 skipped |
| **latest** — numpy 2.5.2 / pandas 3.0.5 / sklearn 1.9.0 / scipy 1.18.1 / numba 0.67.0, Python 3.12 | green — 1,680 passed / 30 skipped |

Only the base row is re-measured every time this page is touched; it is the
current count on this tree. The other rows were measured at earlier commits in
the 0.9 campaign and are not re-run per commit, so their absolute counts trail
the suite. Read them as green/not-green, and re-measure a row before quoting its
number.

The previously recorded ceiling (scikit-learn `<1.8`, numpy `<2.5`) is gone. Its
13 failures are closed and described in `docs/release-notes.md` under
*Stage 2 — latest-dependency compatibility*; the largest group, nine
`target_cv` failures on scikit-learn 1.9, were a `FutureWarning` from
`TargetEncoder(shuffle=..., random_state=...)` that the Stage 1 encoder rewrite
had already eliminated by dropping that backend entirely.

Reproduce the latest row with:

```bash
python3.12 -m venv /tmp/sift-latest
/tmp/sift-latest/bin/pip install "numpy>=2.5" "scikit-learn>=1.9" "pandas>=3" \
  scipy numba joblib threadpoolctl pytest
/tmp/sift-latest/bin/pip install -e . --no-deps
/tmp/sift-latest/bin/python -m pytest -q
```

numpy 2.5 ships cp312 wheels and above, so this row needs Python 3.12+; on 3.11
pip resolves numpy 2.4.6 instead.

Two failure modes are worth knowing before pinning anything new. scikit-learn
1.9 routes dataframe validation through narwhals, which rejects duplicate column
labels in both `fit` and `predict` — that is an estimator-side limit, not a SIFT
one, and SIFT still keeps duplicate labels distinct by position. And pinned
float32 goldens derived from LAPACK plus float32 BLAS are only reproducible to
about one ulp across NumPy/SciPy builds, so compare them with a tolerance;
same-seed determinism inside one interpreter is exact and should stay pinned
exactly.

### Enabling a Python 3.13 job

Now that the newer set is supported, this job can be added to
`.github/workflows/test.yml`:

```yaml
  test-python313:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v6
        with:
          python-version: '3.13'
          cache: pip
      - run: |
          python -m pip install --upgrade pip
          pip install "numba>=0.61"
          pip install -e ".[test]"
      - run: pytest
```

`numba>=0.61` is the only 3.13-specific pin required. This is a support job on
the supported dependency set, not a 3.13/min-pins combination.

### Auto-K gate summarizer

The scheduled `benchmark-smoke` job regenerates the Auto-K G1-G6 gate table from
the committed raw campaign CSVs and uploads it as the `sift-auto-k-gate-table`
artifact. The summarizer is deterministic — `tests/test_auto_k_gate_summary.py`
pins the exact output bytes of a synthetic fixture — so the job also verifies the
regenerated table against the committed one.

That verification is **numeric, not `cmp`**. Gate floats are rendered with 12
significant digits and compared with `rtol=1e-9`; every other cell must match
exactly. A raw `repr` differed in the 17th digit between macOS/arm64 and Linux
CI, so a byte-for-byte comparison would fail on platform-dependent last-ulp
summation rather than on a real change to the inputs or the aggregation. Use
`--verify-against` rather than `cmp`:

```bash
python benchmarks/summarize_auto_k_gates.py \
  --main benchmarks/results/auto_k_v2_main.csv \
  --null benchmarks/results/auto_k_v2_null.csv \
  --timing benchmarks/results/auto_k_v2_d9.csv \
  --fixed-k-path-timing benchmarks/results/auto_k_v2_d9_fixed_k_path_2026-08-31.csv \
  --oracle-aggregation mean \
  --output /tmp/auto_k_v2_gates_mean_oracle.csv \
  --verify-against benchmarks/results/auto_k_v2_gates_mean_oracle_2026-08-31.csv
```

Every argument is required, including `--oracle-aggregation`, so the denominator
convention is always recorded rather than inferred. The summarizer verifies the
path-timing provenance sidecar by hashing its recorded source files *at the commit
the sidecar names*, which is why the job checks out with `fetch-depth: 0`; a
shallow clone cannot resolve that commit and the summarizer fails closed.

Before release-oriented promotion, run:

```bash
python -m pytest -q
python -m pytest tests/test_docs_smoke.py -q
python -m pytest tests/test_benchmarks.py -q
python -m build --wheel
python -m twine check dist/*
python -m venv /tmp/sift-wheel-smoke
/tmp/sift-wheel-smoke/bin/python -m pip install --force-reinstall dist/*.whl
(cd /tmp && /tmp/sift-wheel-smoke/bin/python "$OLDPWD/scripts/verify_wheel_install.py")
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
python benchmarks/run_benchmarks.py --quick --output /tmp/sift-benchmarks.json
git diff --check
```

Release tags must match `v` plus `sift.__version__` — `v0.9.0` for this release.
The release workflow verifies the exact wheel it attaches to the existing GitHub
Release. There is no publication step and no package index: releasing SIFT means
creating a GitHub Release with the checked source and wheel distributions
attached, and nothing else.

For the 0.9.0 release bundle, the focused smoke is:

```bash
python -m pytest tests/test_docs_smoke.py tests/test_benchmarks.py -q
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
```

## Generated Files

Tests and Numba compilation may create `__pycache__`, `.nbc`, `.nbi`, and
`catboost_info` artifacts. Keep generated artifacts out of documentation or
source commits unless the project intentionally tracks them.
