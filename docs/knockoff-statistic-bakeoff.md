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
| `independent` | identity | 12 on the full grid (`p=40`) | 1.8 → 1.1 |
| `ar1` | AR(1) ρ=0.5 | 12 | 1.8 → 1.1 |
| `block` | 5×5 blocks, ρ=0.7 | 12 | 1.8 → 1.1 |
| `dense_weak` | identity | 20 | 0.70 → 0.45 |

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

The CSV is accompanied by `knockoff_statistic_bakeoff.provenance.json`.
Provenance stores the SHA-256 of the **whole CSV**, plus per-seed selected
indices, finite timing samples, structured warnings, and effective thread
pools. It does not invent extra per-row file digests. Source hashes are
captured before the study runs; git dirty reflects that start snapshot, not
the later presence of the output files.

## Retained results

The committed full-run numbers are filled after Codex runs the command above
from a clean source commit. Until then this table is a skeleton, not a 1.0
recommendation.

<!-- knockoff-statistic-bakeoff-table:start -->
Full retained study not yet written. Run the `--full` command from a clean
implementation commit, then replace this block with the runner summary.
<!-- knockoff-statistic-bakeoff-table:end -->

A 1.0 recommendation, when issued, will be scoped to these four Gaussian
designs, `q=0.1`, `offset=1`, `s_method="equi"`, `n_draws=1`, and will not
claim universal dominance. The uncommitted review-time AR(1) contrast (power
0.39 vs 0.90) remains a hypothesis until the retained run.

## What this study does not do

- It does not change `select_fdr` defaults in 0.9.
- It does not retune `ridge_lambda`, LSM `max_steps`, or CEFS+ `path_depth`.
- It does not study F8b e-value aggregation or grouped knockoffs.
- It does not treat realized FDP as a sign-flip proof for LSM or CEFS+.
