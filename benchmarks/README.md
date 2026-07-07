# Benchmarks

Promotion-oriented benchmark scripts for SIFT. Each script measures one hot path
and emits JSON records that the aggregator consumes for release gating.

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
```

## Recorded Knockoff Full Run

`python benchmarks/bench_knockoffs.py --full --output /tmp/bench-knockoffs-full.json`
on 2026-07-07:

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

Individual scripts may emit either the full object form or a bare records list
(both are accepted by the aggregator). Each record carries a
`benchmark_kind` (default `promotion`) and a `promotion_status` field; rows
whose status starts with `blocked` fail the promotion gate.

## Adding a New Benchmark

1. Create `bench_<area>.py` under `benchmarks/` that exposes `--quick`,
   `--full`, and `--output`.
2. Emit records using the helpers in `bench_utils.py` so the schema stays
   consistent.
3. Append the script name to the `SCRIPTS` list in `run_benchmarks.py`.
4. Document it in this README.

See [docs/development.md](../docs/development.md) for the release-time
benchmark checklist.
