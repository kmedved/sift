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
| baseline | 2,000 | 100 | `mrmr_classic` | 0.0010 | 0.0015 | 251.3 | 198.96 | 10 |
| baseline | 2,000 | 100 | `jmi_r2` | 0.0035 | 0.0040 | 272.8 | 57.57 | 10 |
| baseline | 2,000 | 100 | `jmim_r2` | 0.0035 | 0.0039 | 274.7 | 57.80 | 10 |
| baseline | 2,000 | 100 | `cefsplus` | 0.0104 | 0.0112 | 290.3 | 19.19 | 10 |
| baseline | 2,000 | 100 | `cefsplus_binary` | 0.0160 | 0.0164 | 263.7 | 12.48 | 10 |
| baseline | 2,000 | 100 | `fdr_relevance` | 0.0130 | 0.0137 | 282.6 | 15.36 | 0 |
| tall | 20,000 | 100 | `mrmr_classic` | 0.0071 | 0.0073 | 301.9 | 280.53 | 10 |
| tall | 20,000 | 100 | `jmi_r2` | 0.0166 | 0.0170 | 331.0 | 120.80 | 10 |
| tall | 20,000 | 100 | `jmim_r2` | 0.0163 | 0.0170 | 334.0 | 122.70 | 10 |
| tall | 20,000 | 100 | `cefsplus` | 0.1168 | 0.1199 | 352.2 | 17.12 | 10 |
| tall | 20,000 | 100 | `cefsplus_binary` | 0.1322 | 0.1352 | 370.1 | 15.13 | 10 |
| tall | 20,000 | 100 | `fdr_relevance` | 0.1338 | 0.1377 | 409.5 | 14.95 | 0 |
| wide | 2,000 | 500 | `mrmr_classic` | 0.0049 | 0.0052 | 348.8 | 205.94 | 10 |
| wide | 2,000 | 500 | `jmi_r2` | 0.0089 | 0.0094 | 375.6 | 112.70 | 10 |
| wide | 2,000 | 500 | `jmim_r2` | 0.0080 | 0.0084 | 377.0 | 125.58 | 10 |
| wide | 2,000 | 500 | `cefsplus` | 0.0487 | 0.0502 | 360.8 | 20.51 | 10 |
| wide | 2,000 | 500 | `cefsplus_binary` | 0.0564 | 0.0596 | 590.4 | 17.73 | 10 |
| wide | 2,000 | 500 | `fdr_relevance` | 0.1037 | 0.1054 | 403.3 | 9.64 | 0 |
<!-- runtime-scaling-table:end -->

On this design, classic mRMR is the fastest path. R2 JMI and JMIM cluster
together. CEFS+ and binary CEFS+ pay more for conditional path updates. The
wide knockoff run grows more sharply than its row-matched baseline: five times
as many columns took about 8.0 times as long, consistent with the covariance
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
`63ad4999c580a220cb4632b935584a5c90128042`; `dirty=false`, captured at
`2026-09-21T00:55:26.661331+00:00` before measurement and artifact creation.
This is a clean-source local runtime reference for post-release development,
not a release or CI approval. All 18 data and selection fingerprints match
the previous reference. No SIFT tests or review runs were launched during
timing; normal desktop background activity remained. This refresh is not a
controlled before/after speed comparison.

The [CSV artifact](https://github.com/kmedved/sift/blob/main/benchmarks/results/runtime_scaling_2026-09-03.csv) is
SHA-256 `1777a7557f8ca5252237cbfac9a3f00be99cb3c76a61f6c7a820821e1060df01`.
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
