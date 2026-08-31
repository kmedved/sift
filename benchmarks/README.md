# Benchmarks

Promotion-oriented benchmark scripts for SIFT. Each script measures one hot path
and emits JSON records that the aggregator consumes for release gating.
Knockoff rows are informational in 0.7.0: they verify the new `select_fdr`
timing surface and catch smoke regressions without acting as promotion gates.

## Quick Start

Run the full suite in quick mode (fast, smaller problem sizes) and write a
single aggregated JSON file:

```bash
python benchmarks/run_benchmarks.py --quick --output /tmp/sift-benchmarks.json
```

Use `--full` for the larger problem sizes used in release decisions. Add
`--continue-on-fail` to keep going past a failing script:

```bash
python benchmarks/run_benchmarks.py --full --continue-on-fail \
  --output /tmp/sift-benchmarks-full.json
```

`run_benchmarks.py` invokes each script in `SCRIPTS` as a subprocess, collects
its `--output` JSON, and reports any rows whose `promotion_status` starts with
`blocked`.

## Individual Scripts

| Script | What it measures |
| --- | --- |
| `bench_mrmr.py` | Classic mRMR backends (`serial`, `blas`, `processes`) plus selected-feature parity. |
| `bench_jmi.py` | Classic JMI hot loops: binned prebin reuse and R2 weighted-correlation paths, with parity checks. |
| `bench_permutation.py` | Permutation importance on DataFrame and ndarray inputs, grouped/time-aware permutations. |
| `bench_filters.py` | End-to-end function-style filter selectors with promotion JSON. |
| `bench_cefsplus.py` | CEFS+ wall time and allocation-sensitive options. |
| `bench_knockoffs.py` | Gaussian-copula knockoff cache/model/mean/sample/stat/threshold timing, including derandomized draws and a CEFS+ smoke case. |
| `bench_stability.py` | Stability-selection split streaming and fit memory. |
| `bench_catboost.py` | CatBoost split helpers and optional tiny selector smoke cases (skipped without CatBoost). |
| `auto_k_designs.py` | Shared synthetic DGP registry and support-scoring helpers for auto-k gates. |
| `bench_auto_k.py` | Auto-k synthetic DGP harness with oracle-risk rows and support-recovery metrics. |
| `bench_auto_k_path_timing.py` | Focused D9 fixed-k `select_cached` path timing with a checksum-bound provenance sidecar. |
| `bench_utils.py` | Shared helpers and `SCHEMA_VERSION` for the emitted JSON. |

Each script accepts:

```
--quick / --full     Problem-size preset.
--output PATH        Path to write the benchmark JSON.
```

Run one at a time when working on a single hot path:

```bash
python benchmarks/bench_mrmr.py --quick --output /tmp/bench-mrmr.json
python benchmarks/bench_jmi.py --quick --output /tmp/bench-jmi.json
python benchmarks/bench_permutation.py --quick --output /tmp/bench-permutation.json
python benchmarks/bench_stability.py --quick --output /tmp/bench-stability.json
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
python benchmarks/bench_auto_k.py --designs D1,D2,D5 --seeds 3 \
  --output /tmp/bench-auto-k.csv
```

`bench_auto_k.py --methods ...` accepts comma-separated methods including
`penalized/ebic`, `penalized/ric`, `k_posterior`, `chi2_stop`,
`forward_stop`, `changepoint`, `perm_gap`, `xfit_objective`, `gaussian_cv`,
`stability`, and `consensus`, plus the historical baselines.

## Recorded Auto-K v2 Campaign

`python benchmarks/bench_auto_k.py` was run on 2026-07-08 for the Auto-K v2
promotion campaign. The full result files are committed under
`benchmarks/results/`:

- `auto_k_v2_main.csv`: D1-D8, 30 seeds, all v2 methods plus baselines
  (4,560 rows after the appended `gaussian_cv/best` sweep).
- `auto_k_v2_null.csv`: D5 null-calibration deep run, 50 seeds.
- `auto_k_v2_d9.csv`: D9 full-size timing run, 2 seeds, with `--full`.
- `auto_k_v2_catboost.csv`: guarded CatBoost transfer run, D1-D3, 10 seeds.
- `auto_k_v2_summary.csv`, `auto_k_v2_gates.csv`, and
  `auto_k_v2_catboost_summary.csv`: derived aggregate tables.

`summarize_auto_k_gates.py` provides the schema-validated, deterministic G1-G6
aggregation path. The original 2026-07-08 run did not commit its fixed-k path
timing input, so that value must not be reconstructed from the post-path
`fixed_k=50` dispatch rows. Record new evidence from a clean commit instead:

```bash
python benchmarks/bench_auto_k_path_timing.py \
  --full --seeds 2 --seed-start 0 \
  --timing-repeats 5 --warmup-runs 1 --thread-limit 1 \
  --output benchmarks/results/auto_k_v2_d9_fixed_k_path_YYYY-MM-DD.csv
```

The runner uses the same D9 path construction as `bench_auto_k.py`: on the full
design it resolves `k=100` and `top_m=500`, builds the cache first, then times
the complete `select_cached(..., return_indices=True, return_objective=True)`
call. Cache construction is deliberately outside the denominator. One warm-up
is discarded and the CSV records the median of five calls for each seed. Its
strict schema remains `design,seed,benchmark,runtime_s`, with `design=D9` and
`benchmark=fixed_k_select_cached`, so it can be consumed directly by the
summarizer. The automatically written `.provenance.json` sidecar records every
raw timing sample, the effective config and shapes, the command, commit and
dirty state, package/BLAS environment, source hashes, and the CSV checksum.
Use `--quick --seeds 1` only for smoke testing, and never feed a quick artifact
to a release gate.

Regenerate a dated table with one declared oracle convention as follows:

```bash
python benchmarks/summarize_auto_k_gates.py \
  --main benchmarks/results/auto_k_v2_main.csv \
  --null benchmarks/results/auto_k_v2_null.csv \
  --timing benchmarks/results/auto_k_v2_d9.csv \
  --fixed-k-path-timing benchmarks/results/auto_k_v2_d9_fixed_k_path_YYYY-MM-DD.csv \
  --oracle-aggregation mean \
  --output benchmarks/results/auto_k_v2_gates_mean_oracle_YYYY-MM-DD.csv
```

The path-timing seeds must exactly match the D9 method-timing CSV. `mean` is the
declared convention for the dated canonical recomputation. The resulting table
is reproducible from its named inputs, but its G5 path-only ratios combine the
legacy July method timings with a newly measured denominator; it is not a
retroactive measurement of the missing July denominator or a same-run hardware
comparison. The mixed-convention `auto_k_v2_gates.csv` therefore remains an
unchanged legacy artifact. Do not replace it with the dated table unless a
complete contemporaneous campaign justifies that migration.

The 2026 campaign recorded `penalized/ebic` as the measured default for CEFS+
`k="auto"`. It passes the program-level bar, is calibrated on D5, is
recorded as effectively free relative to the CEFS+ path build on D9, and works
for binary CEFS+ without a fold-scoring bridge. The exact G5 path ratio is the
non-reproducible field described above. `gaussian_cv` stays a useful power-user
predictive curve but missed the D9 runtime target and D3 accuracy gate in this
campaign. The additional `gaussian_cv/best` row is a dense-regime sizing
variant: it improves D4 dense-weak behavior and D3 relative to
`gaussian_cv/one_se`, but it still does not replace EBIC as the zero-config
default. `changepoint`, `stability`, `xfit_objective`, and `knockoff_path`
remain experimental or failed-gate for automatic sizing.

Legacy published program-level gate summary (retained unchanged pending the
missing provenance):

| method | mean regret D1-D3+D7 | std(k)/oracle | D5 P(k>3) | D5 max k | D9 runtime ratio | program |
| --- | --- | --- | --- | --- | --- | --- |
| penalized/ebic | 0.001708 | 0.08048 | 0 | 1 | 0.004453 | PASS |
| chi2_stop | 0.001607 | 0.03988 | 0 | 1 | 0.004442 | PASS |
| forward_stop | 0.001293 | 0.07097 | 0 | 1 |  | PASS |
| perm_gap | 0.001561 | 0.05270 | 0 | 1 |  | PASS |
| gaussian_cv | 0.003651 | 0.03739 | 0 | 3 | 9.374 | PASS |
| gaussian_cv/best | 0.001022 | 0.07229 | 0 | 3 | 11.238 | PASS |
| k_posterior | 0.001708 | 0.08048 | 0 | 1 |  | PASS |
| consensus | 0.001575 | 0.04100 | 0 | 1 |  | PASS |
| knockoff_path | 0.2483 | 0.3419 | 0.06667 | 12 |  | FAIL |
| xfit_objective | 0.03778 | 1.543 | 0.1333 | 69 | 9.559 | FAIL |
| stability | 0.1969 | 0.1051 | 0 | 1 |  | FAIL |
| changepoint | 0.03010 | 1.029 | 1.000 | 80 |  | FAIL |

D5 deep null calibration:

| method | P(k>3) | max k | mean k |
| --- | --- | --- | --- |
| chi2_stop | 0 | 1 | 0.04 |
| forward_stop | 0 | 1 | 0.04 |
| knockoff_path | 0.12 | 12 | 0.90 |
| penalized/ebic | 0 | 1 | 0.16 |
| perm_gap | 0 | 2 | 0.16 |

D8-vs-D2 structure-honesty deltas, measured as mean regret on D8 minus D2:

| method | D2 | D8 | D8-D2 |
| --- | --- | --- | --- |
| chi2_stop | 0.00007443 | 0.00007431 | -0.00000012 |
| consensus | 0.00007443 | 0.00007431 | -0.00000012 |
| gaussian_cv | 0 | 0 | 0 |
| stability | 0 | 0 | 0 |
| k_posterior | 0.0003003 | 0.0004390 | 0.0001387 |
| penalized/ebic | 0.0003003 | 0.0004390 | 0.0001387 |
| forward_stop | 0.0003612 | 0.0005133 | 0.0001521 |
| perm_gap | 0.00004094 | 0.0004241 | 0.0003832 |
| knockoff_path | 0.001629 | 0.002844 | 0.001215 |
| xfit_objective | 0.0005186 | 0.002000 | 0.001482 |
| changepoint | 0.01591 | 0.02674 | 0.01082 |

D9 full-size timing:

| method | mean k | mean regret | selection runtime s |
| --- | --- | --- | --- |
| fixed_k=50 | 50.0 | 0.0005060 | 0.000003 |
| elbow | 16.0 | 0.00002757 | 0.000130 |
| chi2_stop | 15.0 | 0.00001267 | 0.000554 |
| penalized/ebic | 15.0 | 0.00001267 | 0.000556 |
| penalized/bic | 16.0 | 0.00007072 | 0.000557 |
| evaluate/time_holdout/best | 100.0 | 0.0009807 | 0.2243 |
| perm_gap | 15.0 | 0.00001267 | 0.9351 |
| evaluate/one_se | 16.0 | 0.00002757 | 1.043 |
| gaussian_cv | 15.0 | 0.00001267 | 2.103 |
| gaussian_cv/best | 15.0 | 0.00001267 | 2.521 |
| xfit_objective | 15.0 | 0.00001267 | 2.144 |

D10 full production-scale dense design (`n=90k`, `p=700`, grouped dense weak
signal; 2 seeds):

| method | mean k | mean regret | selection runtime s |
| --- | --- | --- | --- |
| gaussian_cv/best/group_cv | 213.0 | 0.005380 | 6.652 |
| gaussian_cv/best | 210.5 | 0.007658 | 6.782 |
| penalized/ebic | 199.5 | 0.01948 | 0.001961 |
| gaussian_cv | 193.5 | 0.02377 | 6.657 |

CatBoost transfer, D1-D3:

| design | method | mean regret | median abs k error | runtime s |
| --- | --- | --- | --- | --- |
| D1 | chi2_stop | 0 | 0 | 0.000613 |
| D1 | consensus | 0 | 0 | 0.05039 |
| D1 | evaluate/one_se | 0 | 1.0 | 0.07136 |
| D1 | gaussian_cv | 0 | 0 | 0.04845 |
| D1 | penalized/ebic | 0 | 0.5 | 0.000359 |
| D2 | chi2_stop | 0.0000079 | 0 | 0.000560 |
| D2 | consensus | 0.0000079 | 0 | 0.05196 |
| D2 | evaluate/one_se | 0.0000079 | 1.0 | 0.07450 |
| D2 | gaussian_cv | 0.0000079 | 0 | 0.04930 |
| D2 | penalized/ebic | 0.0000079 | 0 | 0.000349 |
| D3 | chi2_stop | 0.006729 | 42.5 | 0.000579 |
| D3 | consensus | 0.006143 | 42.5 | 0.04769 |
| D3 | evaluate/one_se | 0.003735 | 36.5 | 0.07114 |
| D3 | gaussian_cv | 0.01273 | 47.0 | 0.04625 |
| D3 | penalized/ebic | 0.004189 | 41.0 | 0.000327 |

The complete per-design x method aggregate is in
`benchmarks/results/auto_k_v2_summary.csv`.

The focused 0.7.0 knockoffs smoke is:

```bash
python benchmarks/bench_knockoffs.py --quick --output /tmp/bench-knockoffs.json
```

## Recorded Knockoff Full Run

`python benchmarks/bench_knockoffs.py --full --output /tmp/bench-knockoffs-full.json`
on 2026-07-07. These rows are informational sanity data for the
Gaussian-copula knockoff cache/model/mean/sample/stat/threshold surface, not
hard release gates:

| stat | n | p | draws | cache s | fit s | mean s | sample s | stat s | total s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relevance | 2,000 | 100 | 1 | 0.020 | 0.001 | 0.000 | 0.001 | 0.002 | 0.024 |
| relevance | 2,000 | 100 | 5 | 0.020 | 0.001 | 0.000 | 0.003 | 0.001 | 0.026 |
| cefsplus | 2,000 | 100 | 1 | 0.020 | 0.001 | 0.000 | 0.001 | 0.001 | 0.022 |
| relevance | 50,000 | 500 | 1 | 3.398 | 0.009 | 0.000 | 0.092 | 0.019 | 3.520 |
| relevance | 50,000 | 500 | 11 | 3.446 | 0.009 | 0.022 | 0.908 | 0.232 | 4.633 |
| cefsplus | 50,000 | 500 | 1 | 3.467 | 0.009 | 0.000 | 0.096 | 0.023 | 3.596 |
| relevance | 50,000 | 2,000 | 1 | 14.980 | 0.294 | 0.000 | 0.804 | 0.067 | 16.153 |
| relevance | 50,000 | 2,000 | 11 | 14.748 | 0.302 | 0.279 | 5.915 | 0.912 | 22.243 |
| cefsplus | 50,000 | 2,000 | 1 | 14.752 | 0.295 | 0.000 | 0.783 | 0.136 | 15.965 |

## Output Schema

Aggregated output is a JSON object:

```json
{
  "schema": "<SCHEMA_VERSION>",
  "records": [ { "...": "..." } ],
  "failures": [ ["script", "reason"] ]
}
```

Individual scripts write the shared object form; the aggregator also accepts a
bare records list for legacy or ad hoc scripts. Each record carries a
`benchmark_kind` (default `promotion`) and a `promotion_status` field; rows
whose status starts with `blocked` fail the promotion gate. `bench_knockoffs.py`
emits schema-wrapped records with `benchmark_kind="informational"` and
`promotion_status="informational"`.

## Adding a New Benchmark

1. Create `bench_<area>.py` under `benchmarks/` that exposes `--quick`,
   `--full`, and `--output`.
2. Emit records using the helpers in `bench_utils.py` so the schema stays
   consistent.
3. Append the script name to the `SCRIPTS` list in `run_benchmarks.py`.
4. Document it in this README.

See [docs/development.md](../docs/development.md) for the release-time
benchmark checklist.
