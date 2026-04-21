# TODO

This file tracks the public-release work from the Oracle/source review plus the follow-up LLM review. Top-level bullets are the actionable checklist. Notes under each item preserve the implementation and test context without turning every subtask into its own checkbox.

## Release Gate

- [ ] Clear the public-release block.
  Notes: release-blocking correctness items from both reviews are fixed or explicitly documented. Current local state after the scorer/grid/warning fixes: base `python -m pytest -ra` passed with 274 passed, 10 skipped, and no warnings; darko `/Users/kmedved/miniconda3/envs/darko/bin/python -m pytest -ra` passed with 336 passed, 2 skipped, and no warnings. The remaining skips are optional-dependency/version-conditional coverage for `catboost` and `category_encoders`; remaining open work is performance/feature follow-up unless a release policy requires benchmarks before publishing.
  Latest integration note: the five-agent follow-up added public modules under
  `sift/selectors.py`, `sift/scoring.py`, `sift/selection/result.py`, and
  `sift/selection/path_eval.py`. `/Users/kmedved/miniconda3/envs/darko/bin/python -m pytest -ra`
  passed with 336 passed and 2 skipped. Keep this gate open until all new
  imported modules/tests/benchmarks are included in the release commit.

## Top Priorities

- [x] Fix target-encoding leakage in filter selectors.
- [x] Make `BorutaSelector(importance_data="test")` honest for native importances.
- [x] Fix Gaussian rank transform tie handling.
- [x] Add hard validation for crashy or misleading edge cases.
  Notes: covered for invalid `k`, weighted KSG JMI, zero-weight cache subsamples, cached target length, CatBoost metadata column typos, stability sample weights/options, Boruta options, permutation methods/axes, CatBoost public options, smart sampler config, and smart sampler config mutation.
- [x] Fix Boruta-Shap multiclass default estimator.
- [x] Decide and document time-only permutation semantics.
- [x] Benchmark and optimize classic mRMR/JMI hot loops.
  Notes: classic mRMR now has opt-in `serial`, `blas`, and `processes`
  backends plus `benchmarks/bench_mrmr.py`. JMI now reuses prebinned selected
  columns for binned JMI and uses precomputed/vectorized weighted correlations
  for R2 JMI, with `benchmarks/bench_jmi.py`.

## Correctness Findings

### 1. Full-data target encoding leaks target information

- [x] Make supervised categorical encoders leakage-safe or explicitly unsafe in function-style selectors.
  Scope: `sift/api.py::_prepare_xy_classic`, `_mrmr_gaussian`, `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, and `sift/_preprocess.py::encode_categoricals`.
  Notes: function-style selectors now reject `cat_encoding in {"target", "loo", "james_stein"}` by default when categorical columns would be encoded on the full dataset. `allow_full_data_target_encoding=True` explicitly opts into the old unsafe behavior. Prebuilt caches do not require the flag because no full-data encoder is fitted.
  Tests: high-cardinality categorical guard tests cover default rejection, explicit opt-in, `k="auto"`, `cat_encoding="none"`, no-categorical numeric inputs, and prebuilt-cache behavior.

### 2. Boruta `importance_data="test"` does not use test data for native importances

- [x] Fix or reject `importance_data="test"` with `importance="native"`.
  Scope: `sift/boruta.py::BorutaSelector._compute_importance` and `_get_native_importance`.
  Notes: native test-data mode now raises a clear `ValueError`; use SHAP or another held-out-compatible backend when available. `DOCS.MD` matches this behavior.
  Tests: native test-data mode raises, native train-data mode still runs, and SHAP test-data mode runs when optional dependencies are available.

### 3. Gaussian copula rank transform mishandles ties

- [x] Assign the same weighted midrank to equal feature values.
  Scope: `sift/estimators/copula.py::weighted_rank_gauss_1d`.
  Notes: group sorted equal values into tie blocks and map the same weighted midpoint rank to every row in a block. Confirm downstream callers still work: `build_cache`, Gaussian mRMR/JMI/JMIM, CEFS+, and `compute_objective_for_path`.
  Tests: equal values produce equal transformed values; weighted ties use weighted midranks; binary features produce two unique transformed values.

### 4. Auto-k feature paths use validation targets

- [x] Clarify or fix auto-k path-selection semantics.
  Scope: `sift/api.py::_auto_k_gaussian`, `_auto_k_classic`, and `sift/selection/auto_k.py::select_k_auto`.
  Notes: documented current behavior as `auto_k_mode="prefix_only"` and reserved `auto_k_mode="nested"` for a future train-split refit implementation. Nested mode now raises instead of silently using prefix-only behavior. Prefix evaluation now honors `sample_weight` in downstream model fitting and validation metrics.
  Tests: default and explicit prefix-only match; nested mode raises in `select_k_auto`, `select_mrmr`, `select_jmi`, and `select_cefsplus`; weighted prefix scoring can change the selected `k`, and public selectors pass aligned weights through.

### 5. Invalid `k` can crash inside selection loops

- [x] Add central `k` validation across public selectors.
  Scope: `sift/api.py`, `sift/selection/loops.py`, and `sift/selection/cefsplus.py`.
  Notes: fixed `k` should be integer `>= 1`; `k == "auto"` should be allowed only where supported; decide whether `k > n_features` caps with warning or raises. Classic and Gaussian paths should match.
  Tests: `k=0` and `k=-1` raise `ValueError` across mRMR/JMI/JMIM/CEFS+ variants.

### 6. KSG JMI silently ignores `sample_weight`

- [x] Reject weighted KSG JMI.
  Scope: `sift/selection/loops.py::jmi_select` and `sift/estimators/joint_mi.py::ksg_joint_mi`.
  Notes: `ksg_joint_mi` does not support weights, so `select_jmi(..., estimator="ksg", sample_weight=w)` should raise.
  Tests: weighted KSG raises; unweighted KSG still works.

### 7. `build_cache` can subsample only zero-weight rows

- [x] Make cache subsampling safe with zero weights.
  Scope: `sift/estimators/copula.py::build_cache`.
  Notes: after subsampling, require positive total sampled weight, or sample only positive-weight rows. Consider weight-proportional subsampling when weights are provided.
  Tests: one positive-weight row plus many zero-weight rows gives deterministic safe behavior; cache weights remain finite.

### 8. Cached selectors cannot validate target length against the original matrix

- [x] Add original-row-count metadata to `FeatureCache`.
  Scope: `sift/estimators/copula.py::FeatureCache`, `sift/selection/cefsplus.py::select_cached`, and `sift/selection/auto_k.py::compute_objective_for_path`.
  Notes: store `n_rows_original` in `build_cache` and require `len(y) == cache.n_rows_original` before indexing with `cache.row_idx`.
  Tests: short and long target arrays both raise `ValueError`.

### 9. `corr_prune` applies to Gaussian mRMR/JMI/JMIM, not only CEFS+

- [x] Decide and enforce the intended `corr_prune` contract.
  Scope: `sift/selection/cefsplus.py::select_cached`.
  Notes: `corr_prune="auto"` now means CEFS+ uses its historical `0.95` pruning threshold, while cached Gaussian mRMR/JMI/JMIM do not prune by default. Passing a float to `select_cached` explicitly opts any cached method into pruning; `None` disables pruning.
  Tests: cached Gaussian mRMR/JMI/JMIM can return `k` with duplicated correlated features by default/`None`; CEFS+ default pruning remains intentional.

### 10. `permutation_importance` drops DataFrame metadata and categorical dtypes

- [x] Preserve DataFrame inputs through permutation importance.
  Scope: `sift/importance.py::permutation_importance`.
  Notes: keep a DataFrame path that preserves names and categorical/object dtypes, passes DataFrames to `model.predict`, and keeps an optimized ndarray path for ndarray-only models.
  Tests: dummy DataFrame-requiring model works; CatBoost categorical workflow works or raises a clear unsupported-mode error.

### 11. CatBoost `group_col` and `sample_weight_col` are silently ignored if misspelled

- [x] Raise on missing CatBoost metadata columns.
  Scope: `sift/catboost.py::catboost_select`.
  Notes: missing `group_col` and `sample_weight_col` should raise immediately so group-aware splitting cannot silently fall back because of a typo.
  Tests: missing `group_col` and missing `sample_weight_col` both raise.

### 12. Stability selection accepts invalid sample weights

- [x] Reuse central sample-weight validation in stability selection.
  Scope: `sift/stability.py::StabilitySelector._prep_arrays`.
  Notes: validate length, finiteness, non-negativity, and positive total weight; normalize consistently, likely via `sift._preprocess.ensure_weights`.
  Tests: mismatched, negative, `NaN`, and all-zero weights raise; scaling weights by a constant is stable.

### 13. `StabilitySelector._apply_smart_sampler` mutates caller config

- [x] Copy smart sampler config before mutation.
  Scope: `sift/stability.py::StabilitySelector._apply_smart_sampler`.
  Notes: use something like `dataclasses.replace` so caller-provided `verbose`, `random_state`, and `residual_weight_cap` are not changed.
  Tests: original config remains unchanged after fit, including classification-specific `residual_weight_cap`.

### 14. `StabilitySelector.tune_threshold` skips fit-time imputation

- [x] Reuse fit-time imputation statistics in `tune_threshold`.
  Scope: `sift/stability.py::StabilitySelector.tune_threshold`.
  Notes: `fit` now stores train-time imputation means and `tune_threshold` reuses those means before applying the fit scaler.
  Tests: fitting with NaNs then tuning threshold does not fail, and a regression test verifies imputation uses train means.

### 15. Runtime option validation is inconsistent

- [x] Add explicit runtime validation for enum-like options.
  Scope: `sift/boruta.py`, `sift/catboost.py`, `sift/_permute.py`, and `sift/sampling/smart.py`.
  Notes: Boruta validates `task`, `importance`, `importance_data`, `shadow_method`, `shadow_mode`, and `block_size`; permutation validates methods/axes and group/time lengths; stability validates task, block, parallel, and numeric options; CatBoost validates task, algorithm, prefilter, step/stability options, and GroupKFold group requirements; smart sampler validates config fields and rejects unknown overrides.
  Tests: invalid choices raise clear `ValueError`/`TypeError` messages with valid choices or field names.

### 16. Stale README examples referenced APIs that do not exist

- [x] Promote `DOCS.MD` and remove stale `README.md`.
  Scope: `README.md`, `DOCS.MD`, and `setup.py`.
  Notes: `README.md` was removed; `setup.py` now uses `DOCS.MD` for the package long description; `DOCS.MD` is the source of truth unless the API changes.

- [x] Finish docs/API alignment against `DOCS.MD`.
  Scope: `DOCS.MD`, `setup.py`, `sift/__init__.py`, `sift/api.py`, and `sift/catboost.py`.
  Notes: `DOCS.MD` top-level exports now have smoke coverage against `sift.__all__`, and documented install extras are checked against `setup.py`.
  Tests: `tests/test_docs_smoke.py` verifies exports and extras.

### 17. Boruta-Shap default multiclass classifier uses binary CatBoost loss

- [x] Choose `MultiClass` for default SHAP classification estimator when `y` has 3+ classes.
  Scope: `sift/boruta.py::BorutaSelector._get_default_estimator` and `BorutaSelector.fit`.
  Notes: default estimator construction now sees `y`; SHAP classification uses `Logloss` for binary targets and `MultiClass` for 3+ classes. User-supplied estimators are still preserved.
  Tests: binary and multiclass default loss behavior is covered when `catboost` is installed.

### 18. Time-only permutation currently falls back to global shuffle

- [x] Decide whether `time` without `groups` should use an implicit single group.
  Scope: `sift/_permute.py::resolve_permutation_method`, `build_group_info`, `sift/importance.py::permutation_importance`, and Boruta shadow permutations.
  Notes: `resolve_permutation_method("auto", groups=None, time=...)` now resolves to `"circular_shift"`. Time-only `"block"` and `"circular_shift"` treat the full dataset as one ordered group; docs describe the same behavior for permutation importance and Boruta shadows.
  Tests: low-level permutation tests cover time-only auto/block/circular-shift behavior, and permutation importance has a time-only smoke test.

## Efficiency And Speedups

### A. Classic mRMR redundancy updates

- [x] Benchmark and consider BLAS for classic mRMR redundancy updates.
  Scope: `sift/selection/loops.py::_weighted_corr_with_last` and `mrmr_loop_incremental`.
  Notes: `select_mrmr` now supports `n_jobs` plus `mrmr_backend="auto"`,
  `"serial"`, `"blas"`, and `"processes"`. The BLAS path prestandardizes once
  and updates redundancy with `Z.T @ (w * Z[:, last]) / w.sum()`. The process
  path chunks candidate columns with joblib workers. Gaussian mRMR can parallelize
  rank-Gaussian cache construction via `build_cache(..., n_jobs=..., rank_backend="processes")`.
  Benchmark: `benchmarks/bench_mrmr.py --quick --n-jobs 2` reported parity for
  all rows; on `n=5_000, p=200, k=20`, classic BLAS was 3.32x serial, classic
  processes was slower at 0.02x serial due process overhead, and Gaussian
  processes was 1.92x serial. A follow-up `--blas-threads 1,2,4` sweep showed
  classic BLAS was effectively flat at about 3.3x serial on the quick case, so
  native BLAS threading did not add meaningful speed there.

### B. Binned JMI selected-column reuse

- [x] Reuse pre-binned selected columns in JMI.
  Scope: `sift/selection/loops.py::jmi_select`.
  Notes: binned JMI now uses `X_binned[:, last]` instead of re-binning selected
  columns with `_quantile_bin`; targeted spy tests cover the call count.
  Benchmark: `benchmarks/bench_jmi.py --quick` showed parity and a small binned
  speedup because entropy/counting work dominates after the rebinning removal.

### C. Smart sampling residual pilot

- [x] Skip unnecessary residual-pilot preparation in smart sampling.
  Scope: `sift/sampling/smart.py::smart_sample`.
  Notes: `y` is converted to float only when residual weighting is enabled; `residual_weight_cap == 0` now supports classification-like non-float target columns.
  Benchmark: `benchmarks/bench_stability.py` includes a residual-disabled smart-sampler row with non-float labels.

### D. Permutation importance parallel backend

- [x] Add and benchmark thread/process backend selection for permutation importance.
  Scope: `sift/importance.py::permutation_importance`.
  Notes: added `parallel_backend="threads" | "processes"` and pass it through
  to `Parallel(..., prefer=...)`. DataFrame and ndarray tests cover thread/process
  parity and invalid backend validation.
  Benchmark: `benchmarks/bench_permutation.py` covers DataFrame and ndarray
  inputs across thread/process backends.

### E. Stability split materialization

- [x] Stream stability splits instead of materializing all at once.
  Scope: `sift/stability.py::StabilitySelector.fit`.
  Notes: split iterators are consumed in chunks and coefficient storage is sliced to the number of valid generated splits.
  Benchmark: `benchmarks/bench_stability.py` covers split generation and fit memory modes.

### F. R2 JMI vectorized correlations

- [x] Precompute standardized correlations for R2 JMI.
  Scope: `sift/selection/loops.py::jmi_select` and `sift/estimators/joint_mi.py::r2_joint_mi_indexed`.
  Notes: `jmi_select` now standardizes the post-`top_m` candidate matrix once,
  precomputes feature-target correlations, and computes selected-feature
  correlations with BLAS/vectorized scoring. The existing indexed helper remains
  as the low-allocation compatibility path.
  Benchmark: quick `benchmarks/bench_jmi.py` rows showed about 3.7x unweighted
  and 4.3x weighted R2 speedups with selected-feature parity.

### G. Reusable standardized feature cache

- [ ] Design a reusable standardized/cache path for classic selectors.
  Scope: `sift/api.py`, `sift/selection/loops.py`, and `sift/estimators/copula.py`.
  Notes: cache mean/std, standardized `Z`, optional `Z.T @ W @ Z`, `r_y`, and valid columns for reuse across classic mRMR and R2-JMI.

### H. CEFS+ loop allocation

- [ ] Reduce CEFS+ loop allocation pressure.
  Scope: `sift/selection/cefsplus.py::cefsplus_loop`.
  Notes: benchmark current allocations, reuse scratch arrays where possible, and consider BLAS/Cholesky-based redesigns.
  Benchmark: `top_m=[250, 1_000, 5_000]`, `k=[50, 200]`, `compute_Rxx=True/False`.

### I. Filter result objects

- [x] Add `FilterSelectionResult` behind `return_result=True`.
  Scope: `sift/api.py`, `sift/selection/cefsplus.py`, and `sift/selection/loops.py`.
  Notes: default return type remains a list. `return_result=True` returns a
  compact `FilterSelectionResult` with selected names, best-effort input-column
  indices, selector metadata, and an on-demand conservative ranking table.
  Richer diagnostics such as redundancy, objective path, and candidate features
  remain future polish.

### J. Smart sampler probability accumulation

- [x] Replace dictionary accumulation in `smart_sample` with dense/vectorized probabilities.
  Scope: `sift/sampling/smart.py::smart_sample`.
  Notes: inclusion probabilities now accumulate into a dense `pi` array with indexed updates. Output rows are returned in ascending original row order.
  Benchmark: deterministic tests cover row/weight stability; grouped panel timing is covered by the benchmark runner.

### K. Bootstrap OOB construction

- [x] Reduce Python set/list OOB work in bootstrap generators.
  Scope: `sift/stability.py::_block_bootstrap_indices` and `sift/catboost.py` bootstrap helpers.
  Notes: stability block bootstrap now uses vectorized set differences for OOB rows. CatBoost split helpers are covered in benchmark smoke rows and remain unchanged where already vectorized.
  Benchmark: `benchmarks/bench_stability.py` and `benchmarks/bench_catboost.py`.

### L. Permutation importance copy pressure

- [x] Reuse worker-local scratch copies in permutation importance.
  Scope: `sift/importance.py::_permutation_importance_dataframe` and `_permutation_importance_array`.
  Notes: feature chunks now use one worker-local scratch DataFrame/array and
  restore the permuted column in-place after each score, preserving DataFrame
  metadata and avoiding a full copy per feature task.
  Benchmark: `benchmarks/bench_permutation.py` includes DataFrame and ndarray
  smoke/full modes.

## Feature Additions

- [x] Add scikit-learn transformer classes for filter selectors.
  Notes: `MRMRSelector`, `JMISelector`, `JMIMSelector`, and `CEFSPlusSelector`
  now provide `fit`, `transform`, `fit_transform`, `get_support`,
  `selected_features_`, `selected_indices_`, and `feature_names_in_`.
  DataFrame transforms validate fitted column order, and sklearn
  `set_params(...)` flows through to the next fit.

- [x] Add explicit fitted categorical encoders inside selector classes.
  Notes: selector classes now default to `cat_encoding="none"`. When supervised encoders are explicitly enabled, `fit_transform` returns the selected columns from the y-aware encoded training matrix used for selection, `transform` applies the fitted encoder to future rows without target access, and prefix-only/evaluate auto-k rejects the unsafe supervised-encoder combination. Failed refits clear prior fitted state. These encoders are fitted, not cross-fitted.

- [ ] Add cross-fitted categorical encoding modes for selection workflows.
  Notes: consider `cat_encoding="target_cv"` / `"loo_cv"` for DataFrame inputs, using fold-local encoders during path construction and evaluation. This is distinct from the current explicit unsafe full-data opt-in.

- [x] Implement `auto_k_mode="nested"` for selector classes.
  Notes: selector classes refit train-only selector paths inside time-holdout or group-CV splits, evaluate prefixes with the same `fit_transform` training representation used by sklearn pipelines plus fold-local validation preprocessing/weights, reject prebuilt caches, reject `k_method="elbow"` until a nested-elbow contract exists, and refit final features on all rows passed to `fit`. Function-style selectors keep the prefix-only contract and still reject nested mode. Class and function selectors now share auto-k strategy inference, so groups-only calls infer group CV without an explicit config.

- [x] Add a first-class feature-path evaluation helper.
  Notes: `sift.selection.evaluate_feature_path` and the top-level
  `sift.evaluate_feature_path` now support explicit `k_grid` prefix validation,
  estimator templates or factories, custom splitter objects or one explicit
  train/validation split, sample weights, and regression scoring via `rmse`,
  `mae`, or a callable. It returns `FeaturePathEvaluationResult` with best `k`,
  best-prefix features, scores, and a diagnostics DataFrame. Multi-metric
  selection rules, per-split cached paths, and nested selector refits remain
  future extensions.

- [x] Add a shared scoring registry.
  Notes: unify metric names and higher-is-better/lower-is-better semantics
  across permutation importance, auto-k, future feature-path evaluation, and
  CatBoost diagnostics. Preserve `permutation_importance`'s explicit
  higher-is-better string contract: use `neg_error` and `neg_logloss`, not plain
  `error` or `logloss`. `permutation_importance` now uses `sift.scoring` for
  built-in scorer lookup while preserving callable scoring.

- [ ] Add grouped feature selection.
  Notes: support feature families such as one-hot sets, lags, splines, embeddings, and interactions. Start with post-processing where selecting one feature includes its group; decide whether `k` counts groups or raw features.

- [ ] Add purged / embargoed time-series split utilities.
  Notes: add `PurgedTimeSeriesSplit` and `GroupPurgedTimeSeriesSplit`, make them scikit-learn-compatible, and document use with `catboost_select`, `select_k_auto`, and permutation importance.

- [x] Add time-only permutation support or explicit documentation.
  Notes: single time series are treated as one implicit ordered group for auto/block/circular-shift permutation; Boruta and permutation importance use the same shared behavior.

- [x] Add selected-feature result summary.
  Notes: `FilterSelectionResult.get_feature_ranking()` now returns a conservative selected-feature table with `feature`, `rank`, `selected`, `selected_index`, `relevance`, and selector metadata. Metrics not exposed by the selector are left as `NaN`.

- [ ] Add richer unified selector diagnostics.
  Notes: include unselected candidates where available, non-NaN selector-specific relevance, redundancy at selection, objective gains, candidate-pool size, filtered/pruned features, and objective/path diagnostics.

- [ ] Add reproducibility manifests.
  Notes: expose `result.reproducibility_` with package version, selector name, seeds, selected features, column list or feature hash, key parameters, and dependency versions where available.

## Refactor Plan

- [x] Step 1: Add central validation.
  Notes: validation is now applied across API selectors, Boruta, CatBoost, stability, permutation, smart sampler, weights, and cache paths. Further cleanup can still centralize duplicated helper shapes, but the release-blocking validation gaps are covered.

- [x] Step 2: Fix leakage semantics.
  Notes: prioritize target encoding, auto-k path construction, CatBoost missing `group_col`, Boruta `importance_data="test"`, and safe/default-or-explicitly-unsafe behavior.

- [x] Step 3: Fix Gaussian cache correctness.
  Notes: add `FeatureCache.n_rows_original`, fix tie handling, validate zero-weight subsamples, validate cached target length, and clarify `corr_prune`.

- [x] Step 4: Introduce filter result objects.
  Notes: current public return behavior is preserved; `return_result=True`
  exposes compact result objects for diagnostics/tests.

- [x] Step 5: Add class-based selectors.
  Notes: sklearn-style wrappers, explicit fitted categorical encoders with
  guardrails, and selector-class nested auto-k are in place.

- [x] Step 6a: Add benchmark foundation and promote completed hot-loop optimizations.
  Notes: mRMR BLAS/process backends, vectorized R2 JMI, binned JMI reuse, stability split streaming, smart sampler dense probability accumulation, bootstrap OOB construction, and permutation importance backend/DataFrame work now have tests and benchmark coverage. CEFS+ allocation pressure remains benchmarked but not promoted as a public optimization yet.

- [ ] Step 6b: Finish remaining cache/CEFS+ allocation work.
  Notes: reusable standardized/cache design and CEFS+ allocation-pressure reduction remain open optimization work.

## Test Plan

- [x] Add filter selector validation tests.
  Notes: `k <= 0`, weighted KSG JMI, cached target length, Gaussian tie behavior, and explicit `corr_prune` return behavior.

- [x] Add categorical leakage tests.
  Notes: top-level selectors with `k="auto"` and supervised encoders; high-cardinality ID tests for time holdout and group CV.

- [x] Add auto-k semantic tests.
  Notes: prefix-only/default equivalence, nested-mode rejection, fold-local target encoding in `select_k_auto`, and sample-weighted prefix scoring.

- [x] Add Boruta tests.
  Notes: native `importance_data="test"`, native train mode, SHAP train/test smoke coverage, invalid option choices, shadow row mode, and default SHAP binary/multiclass classification are covered.

- [x] Add stability selection tests.
  Notes: invalid sample weights, runtime option validation, smart sampler config immutability, and `tune_threshold` fit-time imputation reuse are covered.

- [x] Add CatBoost tests.
  Notes: missing metadata columns, invalid public options, invalid `step_function`, custom `GroupKFold` without groups, and documented `k > n_features` cap-with-warning behavior are covered when `catboost` is installed.

- [x] Add permutation importance tests.
  Notes: DataFrame-preserving prediction, ndarray prediction, classification scoring, `balanced_accuracy`, binary and multiclass `neg_logloss`, invalid plain `error`/`logloss` scoring, missing `predict_proba` for `neg_logloss`, `n_repeats <= 0`, X/y length mismatch, grouped time auto behavior, and time-only permutation smoke coverage are present. Remaining optional coverage: CatBoost categorical workflow.

- [x] Add Gaussian rank-Gauss row-order invariance test.
  Notes: tie-heavy weighted row-permutation coverage maps transformed values
  back to original row ids and asserts equality.

- [x] Add documentation smoke tests.
  Notes: verify documented extras exist in `setup.py` and compare `DOCS.MD` top-level exports with `sift.__all__`.

## Benchmark Suite

- [x] Add a benchmark suite under `benchmarks/`.
  Notes: `bench_mrmr.py`, `bench_jmi.py`, `bench_permutation.py`,
  `bench_filters.py`, `bench_cefsplus.py`, `bench_stability.py`,
  `bench_catboost.py`, and `run_benchmarks.py` now share a promotion JSON schema.
  Track wall time, optional peak memory, selected feature agreement, and
  benchmark kind/promotion status where applicable. Informational/smoke/parity
  rows are not treated as promoted optimizations. Benchmark scripts write row
  metadata and return success unless they crash; `run_benchmarks.py` owns the
  aggregate promotion-gate decision.

Benchmark matrix:

| Benchmark | Variables | Metrics |
| --- | --- | --- |
| Classic mRMR | `n`, `p`, `k`, `top_m` | time, memory, selected overlap |
| JMI R2 | `n`, `p`, `k`, `top_m` | time, selected overlap |
| JMI binned | `n`, `top_m`, bins | time, memory |
| CEFS+ | `top_m`, `k`, `compute_Rxx` | time, memory, objective |
| Stability | `n_bootstrap`, `p`, `store_coefs` | time, memory |
| Smart sampler | `n`, group cardinality, anchors, residual weighting | time, memory, inclusion probabilities |
| Permutation importance | `p`, `n_repeats`, backend, DataFrame vs ndarray | time, memory |
| CatBoost forward/RFE | `k`, `prefilter_k`, splits | fits, time, score |

## Open Questions

- [x] Is `DOCS.MD` intended to describe the current API while README may be older?
  Decision: remove `README.md` and promote `DOCS.MD`.

- [x] Should target encoders inside function-style selectors be allowed at all?
  Decision: allowed only with explicit `allow_full_data_target_encoding=True`; default behavior rejects supervised full-data encoding.
- [x] Should `k="auto"` mean nested selector refitting for this release?
  Decision: function-style selectors remain `auto_k_mode="prefix_only"` by default and still reject nested mode. Selector classes now support opt-in `auto_k_mode="nested"` for train-split refits.
- [x] Should filter functions return exactly `k` features or up to `k` features?
  Decision: fixed `k` is an upper bound for filter selectors. They return up to
  `k`, because constant-feature filtering, non-positive relevance filtering,
  `top_m`, pruning, and validity checks can leave fewer honest candidates.
- [x] Should `time` without `groups` imply a single time-series group for permutation?
  Decision: yes. Time-only auto/block/circular-shift permutation treats the full dataset as one ordered group.
- [x] Is `corr_prune` intended for all Gaussian cached methods or only CEFS+?
  Decision: CEFS+-only by default; direct `select_cached(..., corr_prune=<float>)` is the explicit opt-in for cached Gaussian mRMR/JMI/JMIM.
- [x] Should `importance_data="test"` for Boruta native importances become permutation-style importance?
  Decision: no for now; native/test raises. A future held-out permutation backend should be explicitly named rather than overloading native importances.
