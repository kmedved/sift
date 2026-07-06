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
```

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
