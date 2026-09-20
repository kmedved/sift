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
| baseline | 2,000 | 100 | `mrmr_classic` | 0.0010 | 0.0014 | 258.8 | 201.80 | 10 |
| baseline | 2,000 | 100 | `jmi_r2` | 0.0030 | 0.0033 | 269.6 | 67.37 | 10 |
| baseline | 2,000 | 100 | `jmim_r2` | 0.0035 | 0.0043 | 285.7 | 57.30 | 10 |
| baseline | 2,000 | 100 | `cefsplus` | 0.0104 | 0.0109 | 274.8 | 19.31 | 10 |
| baseline | 2,000 | 100 | `cefsplus_binary` | 0.0161 | 0.0169 | 274.7 | 12.44 | 10 |
| baseline | 2,000 | 100 | `fdr_relevance` | 0.0129 | 0.0134 | 275.8 | 15.48 | 0 |
| tall | 20,000 | 100 | `mrmr_classic` | 0.0070 | 0.0074 | 304.2 | 284.91 | 10 |
| tall | 20,000 | 100 | `jmi_r2` | 0.0164 | 0.0174 | 335.5 | 121.71 | 10 |
| tall | 20,000 | 100 | `jmim_r2` | 0.0160 | 0.0170 | 333.3 | 125.32 | 10 |
| tall | 20,000 | 100 | `cefsplus` | 0.1155 | 0.1163 | 341.6 | 17.31 | 10 |
| tall | 20,000 | 100 | `cefsplus_binary` | 0.1334 | 0.1359 | 390.1 | 14.99 | 10 |
| tall | 20,000 | 100 | `fdr_relevance` | 0.1349 | 0.1364 | 443.8 | 14.82 | 0 |
| wide | 2,000 | 500 | `mrmr_classic` | 0.0046 | 0.0048 | 359.6 | 216.35 | 10 |
| wide | 2,000 | 500 | `jmi_r2` | 0.0077 | 0.0083 | 372.0 | 129.59 | 10 |
| wide | 2,000 | 500 | `jmim_r2` | 0.0081 | 0.0086 | 352.2 | 123.27 | 10 |
| wide | 2,000 | 500 | `cefsplus` | 0.0492 | 0.0494 | 397.6 | 20.33 | 10 |
| wide | 2,000 | 500 | `cefsplus_binary` | 0.0582 | 0.0589 | 489.2 | 17.19 | 10 |
| wide | 2,000 | 500 | `fdr_relevance` | 0.1057 | 0.1064 | 480.0 | 9.46 | 0 |
<!-- runtime-scaling-table:end -->

On this design, classic mRMR is the fastest path. R2 JMI and JMIM cluster
together. CEFS+ and binary CEFS+ pay more for conditional path updates. The
wide knockoff run grows more sharply than its row-matched baseline: five times
as many columns took about 8.2 times as long, consistent with the covariance
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
`a325e31d4e3d646cea3f230a2fe9d0e63deca4cb`; `dirty=false`, captured at
`2026-09-20T14:39:10.503490+00:00` before measurement and artifact creation.
This is a clean-source local runtime reference for the closure fixes,
not a release or CI approval. All 18 data and selection fingerprints match
the previous reference. No SIFT tests or review runs were launched during
timing; normal desktop background activity remained. This refresh is not a
controlled before/after speed comparison.

The [CSV artifact](https://github.com/kmedved/sift/blob/main/benchmarks/results/runtime_scaling_2026-09-03.csv) is
SHA-256 `1ecf6c2fc52fb5b6f7cc7fa15003d690001b84915a4f236cef8af7e09651db4e`.
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
