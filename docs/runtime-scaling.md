# Runtime and Scaling

These measurements compare six dependency-free, end-to-end selector paths on
one machine. They are orientation data, not latency guarantees and not evidence
that one selector has better statistical quality than another. Method names
follow the [glossary](glossary.md) ([fixed-k](glossary.md#fixed-k) filters
versus q-calibrated [knockoffs](glossary.md#knockoffs)).

## Workloads and method settings

All three workloads use the same seeded independent-normal design with eight
signal columns. The regression target is a noisy linear combination of those
columns; the binary target is a balanced thresholded version of the same latent
signal.

| workload | rows (`n`) | features (`p`) | purpose |
| --- | ---: | ---: | --- |
| baseline | 2,000 | 100 | reference shape |
| tall | 20,000 | 100 | isolate a 10x row increase |
| wide | 2,000 | 500 | isolate a 5x feature increase |

The five fixed-size filters use `k=10`, `subsample=None`, `random_state=0`,
and their normal candidate-screen defaults. The named variants are classic
BLAS mRMR, R2 JMI/JMIM, Gaussian CEFS+, and log-loss binary CEFS+.
`fdr_relevance` is `select_fdr(q=0.1, statistic="relevance", n_draws=1)` and
returns a q-calibrated set rather than ten features.

Each method/workload combination runs in a fresh process with one warm-up and
seven timed calls. BLAS, OpenMP, NumExpr, and Numba are limited to one thread.
The table reports linearly interpolated p50 and p95 wall time. The artifact also
retains p99 and every raw sample. “M cells/s” is `n * p / p50`; it is an input
work-rate aid, not an algorithm-independent throughput score. Peak RSS is the
maximum resident size of the whole worker after imports, data generation,
warm-up, and timing—not incremental selector allocation.

## Recorded run

<!-- runtime-scaling-table:start -->
| workload | n | p | method | p50 s | p95 s | peak RSS MB | M cells/s | selected |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 2,000 | 100 | `mrmr_classic` | 0.0011 | 0.0017 | 259.4 | 186.01 | 10 |
| baseline | 2,000 | 100 | `jmi_r2` | 0.0035 | 0.0043 | 272.5 | 56.70 | 10 |
| baseline | 2,000 | 100 | `jmim_r2` | 0.0039 | 0.0045 | 277.5 | 50.73 | 10 |
| baseline | 2,000 | 100 | `cefsplus` | 0.0113 | 0.0115 | 274.0 | 17.73 | 10 |
| baseline | 2,000 | 100 | `cefsplus_binary` | 0.0172 | 0.0179 | 295.6 | 11.61 | 10 |
| baseline | 2,000 | 100 | `fdr_relevance` | 0.0144 | 0.0148 | 290.6 | 13.86 | 0 |
| tall | 20,000 | 100 | `mrmr_classic` | 0.0073 | 0.0075 | 306.3 | 273.42 | 10 |
| tall | 20,000 | 100 | `jmi_r2` | 0.0164 | 0.0170 | 335.1 | 121.60 | 10 |
| tall | 20,000 | 100 | `jmim_r2` | 0.0169 | 0.0175 | 336.5 | 118.01 | 10 |
| tall | 20,000 | 100 | `cefsplus` | 0.1259 | 0.1272 | 355.5 | 15.89 | 10 |
| tall | 20,000 | 100 | `cefsplus_binary` | 0.1400 | 0.1409 | 393.3 | 14.29 | 10 |
| tall | 20,000 | 100 | `fdr_relevance` | 0.1442 | 0.1458 | 455.6 | 13.87 | 0 |
| wide | 2,000 | 500 | `mrmr_classic` | 0.0047 | 0.0049 | 374.3 | 211.33 | 10 |
| wide | 2,000 | 500 | `jmi_r2` | 0.0079 | 0.0083 | 407.0 | 127.32 | 10 |
| wide | 2,000 | 500 | `jmim_r2` | 0.0081 | 0.0087 | 405.4 | 123.85 | 10 |
| wide | 2,000 | 500 | `cefsplus` | 0.0535 | 0.0542 | 419.6 | 18.68 | 10 |
| wide | 2,000 | 500 | `cefsplus_binary` | 0.0585 | 0.0590 | 554.1 | 17.09 | 10 |
| wide | 2,000 | 500 | `fdr_relevance` | 0.1125 | 0.1251 | 484.5 | 8.89 | 0 |
<!-- runtime-scaling-table:end -->

On this design, classic mRMR is the fastest path. R2 JMI and JMIM cluster
together. CEFS+ and binary CEFS+ pay more for conditional path updates. The
wide knockoff run grows more sharply than its row-matched baseline: five times
as many columns took about 7.8 times as long, consistent with the covariance
work being width-sensitive. Three shapes are not enough to estimate a formal
complexity exponent, so this page does not claim one.

The knockoff rows selected zero features. That is a valid result at `q=0.1` and
does not invalidate their timings; it also means the `selected` column must not
be read as a power comparison. Use a quality benchmark, not this runtime table,
to compare statistical recovery.

## Scope limits

The table deliberately excludes routes whose budgets are controlled by a
different primary knob:

- stability selection scales with `n_bootstrap` and any alpha-tuning CV;
- Boruta scales with `max_iter`, tree count, and the importance backend;
- permutation importance scales with feature count times `n_repeats` and the
  fitted model's prediction cost;
- smart sampling scales with its SVD and optional residual-pilot work;
- CatBoost and Boruta-SHAP require optional dependencies and model budgets.

Their focused harnesses remain under `benchmarks/`. Mixing reduced iteration
counts into the table above would look comparable while measuring different
contracts.

## Provenance and reproduction

The recorded run used CPython 3.12.7 on macOS arm64, NumPy 1.26.4, pandas
2.2.2, scikit-learn 1.5.1, SciPy 1.13.1, Numba 0.60.0, and one OpenBLAS
0.3.23.dev thread. It ran from clean implementation commit
`826e10b78de692f93eee70552aba3a70e7232370`; `dirty=false`, captured before
measurement and artifact creation. That is release-grade evidence for 0.9.1.

The [CSV artifact](https://github.com/kmedved/sift/blob/main/benchmarks/results/runtime_scaling_2026-09-03.csv) is
SHA-256 `27ffe9c72b7510062d926d7f8cb7aae3a3436480b677b692669c6b8858ffe25c`.
Its [provenance sidecar](https://github.com/kmedved/sift/blob/main/benchmarks/results/runtime_scaling_2026-09-03.provenance.json)
binds that checksum, the command, environment, raw samples, effective options,
data and selection fingerprints, thread-pool state, Git status, and SHA-256 for
the runner plus every executed `sift/*.py` source file.

Run a comparison artifact without overwriting the recorded evidence:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python benchmarks/bench_runtime_scaling.py \
  --full --warmup-runs 1 --timing-repeats 7 \
  --output /tmp/sift-runtime-scaling.csv
```

Compare ratios and raw distributions on your own deployment hardware. Do not
compare these absolute times directly with a run that changes thread limits,
warm-up policy, dependency versions, or selector options.
