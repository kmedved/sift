# Knockoff statistic bakeoff

This page is the F8c evidence record for the 1.0 default-statistic decision:
keep `statistic="relevance"` or flip the default to `ridge`. It is **not** a
change to the 0.9 default, and it is **not** a Model-X proof.

Measured false-discovery proportions on these Gaussian designs are empirical
calibration checks. They do not validate knockoff exchangeability and they do
not upgrade `fdr_control="approximate_plugin"`. Adaptive CEFS+ and
tied/truncated LSM **do not have a general sign-flip proof**; their bakeoff
rows are quality and runtime measurements only. See [knockoffs](glossary.md#knockoffs),
[approximate plugin](glossary.md#approximate-plugin), and the
[algorithm guide](ALGORITHMS.md).

## Protocol

The runner is `benchmarks/bench_knockoff_statistic_bakeoff.py`. Every cell
calls public `sift.select_fdr` with the documented 0.9 defaults:

| option | value | notes |
| --- | --- | --- |
| `q` | `0.1` | public default |
| `offset` | `1` | knockoff+ |
| `s_method` | `"equi"` | public default |
| `n_draws` | `1` | public default |
| `statistic_options` | omitted | no tuning |
| `aggregation` | omitted | not an F8b e-value study |

Designs and knockoff draws share seeds, so relevance/lsm/ridge/cefsplus see
the same `X`, `y`, and `random_state` on each replicate.

| design | covariance | signals | amplitude |
| --- | --- | --- | --- |
| `independent` | identity | first 12 columns | 1.8 → 1.1 |
| `ar1` | AR(1) ρ=0.5 | first 12 columns | 1.8 → 1.1 |
| `block` | 5×5 blocks, ρ=0.7 | first 12 columns | 1.8 → 1.1 |
| `dense_weak` | identity | first 20 columns | 0.70 → 0.45 |

Noise is i.i.d. Gaussian with SD 1. True coefficients are the first
`n_signal` entries, all positive and contiguous. That same-sign contiguous
support is part of this study, not a claim about mixed-sign or suppressor
designs. `dense_weak` uses weaker amplitudes than the other three; on this
grid it is **not** an empirically hard detection problem (relevance power
about 0.99).

Full retained study: `n=800`, `p=40`, 30 data seeds `{0,…,29}`, one warm-up
call and one timed call per cell, native thread pools limited to 1. Smoke:
`n=160`, `p=16`, 2 seeds, no warm-up — functional only, not the 1.0 record.

Failed calls are stored as `status=failed` with the exception text. They are
not rewritten as empty selections. Warnings are stored on the row.

## Commands

Smoke (implementation check only):

```bash
LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/anaconda3/bin/python benchmarks/bench_knockoff_statistic_bakeoff.py \
  --smoke --output /tmp/knockoff-statistic-bakeoff-smoke.csv
```

Retained full study, from a clean implementation commit, with no concurrent
tests or providers:

```bash
LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /opt/anaconda3/bin/python benchmarks/bench_knockoff_statistic_bakeoff.py \
  --full --output benchmarks/results/knockoff_statistic_bakeoff.csv
```

The retained artifacts are
[`benchmarks/results/knockoff_statistic_bakeoff.csv`](https://github.com/kmedved/sift/blob/main/benchmarks/results/knockoff_statistic_bakeoff.csv)
and
[`benchmarks/results/knockoff_statistic_bakeoff.provenance.json`](https://github.com/kmedved/sift/blob/main/benchmarks/results/knockoff_statistic_bakeoff.provenance.json).
Provenance stores the SHA-256 of the **whole CSV**, plus per-seed selected
indices, finite timing samples, structured warnings, and effective thread
pools. It does not invent extra per-row file digests. Source hashes are
captured before the study runs; git dirty reflects that start snapshot, not
the later presence of the output files.

## Retained results

Retained full run from clean implementation commit
`ae904b8af02037eb66cd649384c4665dba17049d`, captured
`2026-09-05T06:32:30.226365+00:00`, `dirty=false` and empty status, 75 source
hashes. CSV SHA256
`40d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf`.
Environment: `/opt/anaconda3/bin/python` 3.12.7, NumPy 1.26.4, pandas 2.2.2,
scikit-learn 1.5.1, SciPy 1.13.1, numba 0.60.0, threadpoolctl 3.5.0, macOS
arm64. Native pools recorded as 1. Zero failed cells and zero warnings.
Codex ran this command with no concurrent tests or reviewers; ordinary
macOS/remote-desktop background processes were present. Millisecond-scale
desktop timings are descriptive and are not a performance guarantee.

<!-- knockoff-statistic-bakeoff-table:start -->
Study `full`. Cells are mean ± SE over completed seeds; failed seeds are counted, not converted to empty selections.

| design | statistic | n ok | n failed | n warned | FDP | power | discoveries | runtime s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| independent | `relevance` | 30 | 0 | 0 | 0.007 ± 0.005 | 0.989 ± 0.005 | 11.967 ± 0.102 | 0.002 ± 0.000 |
| independent | `lsm` | 30 | 0 | 0 | 0.000 ± 0.000 | 1.000 ± 0.000 | 12.000 ± 0.000 | 0.005 ± 0.000 |
| independent | `ridge` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.997 ± 0.003 | 11.967 ± 0.033 | 0.004 ± 0.000 |
| independent | `cefsplus` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.244 ± 0.075 | 2.933 ± 0.906 | 0.003 ± 0.000 |
| ar1 | `relevance` | 30 | 0 | 0 | 0.010 ± 0.005 | 0.997 ± 0.003 | 12.100 ± 0.074 | 0.002 ± 0.000 |
| ar1 | `lsm` | 30 | 0 | 0 | 0.000 ± 0.000 | 1.000 ± 0.000 | 12.000 ± 0.000 | 0.005 ± 0.000 |
| ar1 | `ridge` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.881 ± 0.045 | 10.567 ± 0.540 | 0.004 ± 0.000 |
| ar1 | `cefsplus` | 30 | 0 | 0 | 0.003 ± 0.003 | 0.025 ± 0.025 | 0.333 ± 0.333 | 0.004 ± 0.000 |
| block | `relevance` | 30 | 0 | 0 | 0.061 ± 0.012 | 0.967 ± 0.011 | 12.433 ± 0.257 | 0.002 ± 0.000 |
| block | `lsm` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.908 ± 0.046 | 10.900 ± 0.556 | 0.005 ± 0.000 |
| block | `ridge` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.422 ± 0.084 | 5.067 ± 1.010 | 0.004 ± 0.000 |
| block | `cefsplus` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.000 |
| dense_weak | `relevance` | 30 | 0 | 0 | 0.022 ± 0.007 | 0.987 ± 0.006 | 20.200 ± 0.206 | 0.002 ± 0.000 |
| dense_weak | `lsm` | 30 | 0 | 0 | 0.000 ± 0.000 | 1.000 ± 0.000 | 20.000 ± 0.000 | 0.005 ± 0.000 |
| dense_weak | `ridge` | 30 | 0 | 0 | 0.002 ± 0.002 | 0.997 ± 0.002 | 19.967 ± 0.058 | 0.004 ± 0.000 |
| dense_weak | `cefsplus` | 30 | 0 | 0 | 0.000 ± 0.000 | 0.900 ± 0.037 | 18.000 ± 0.733 | 0.005 ± 0.000 |

Paired `ridge - relevance` on shared seeds:

| design | n paired | power diff | FDP diff | runtime s diff |
| --- | ---: | ---: | ---: | ---: |
| independent | 30 | 0.008 ± 0.005 | -0.007 ± 0.005 | 0.002 ± 0.000 |
| ar1 | 30 | -0.117 ± 0.045 | -0.010 ± 0.005 | 0.002 ± 0.000 |
| block | 30 | -0.544 ± 0.082 | -0.061 ± 0.012 | 0.002 ± 0.000 |
| dense_weak | 30 | 0.010 ± 0.007 | -0.020 ± 0.006 | 0.001 ± 0.000 |
<!-- knockoff-statistic-bakeoff-table:end -->

**Recommendation for the 1.0 owner decision, on this evidence:** retain
`statistic="relevance"`. Paired ridge-minus-relevance power is +0.0083
(independent), −0.1167 (AR(1)), −0.5444 (block), +0.010 (dense-weak). Ridge
reduces realized FDP but materially loses power on the correlated designs and
costs more here. Relevance FDP means are about 0.0073 / 0.0103 / 0.0606 /
0.0215, all sampled below `q=0.1`. That is an empirical calibration check on
these Gaussian draws, not a formal FDR certificate and not an upgrade of
`approximate_plugin`.

This does **not** claim universal dominance, suppressor or mixed-sign
support, or `p ≫ n` coverage. Standard errors describe Monte Carlo sampling
variability of this fixed study; they are not a significance test. Adaptive
CEFS+ and tied/truncated LSM still have no general sign-flip proof; their
rows are quality/runtime measurements only. The 0.9 default remains
`relevance`; any 1.0 change stays an owner decision.

LSM has strong measured quality here, but its missing general sign-flip
guarantee prevents recommending it as a validity-preserving default replacement.

The earlier uncommitted AR(1) contrast (power 0.39 vs 0.90) used a different
unretained design. This run does not reproduce or generalize that claim, and
it is not a universal disproof of ridge.

## What this study does not do

- It does not change `select_fdr` defaults in 0.9.
- It does not retune `ridge_lambda`, LSM `max_steps`, or CEFS+ `path_depth`.
- It does not study F8b e-value aggregation or grouped knockoffs.
- It does not treat realized FDP as a sign-flip proof for LSM or CEFS+.
