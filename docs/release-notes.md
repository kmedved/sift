# Release Notes

## Unreleased

### Performance

- Greedy correlation pruning and knockoff thresholding now use vectorized and
  sorted-search implementations. JMI updates candidate scores in bulk,
  bootstrap stability reuses its indicator/intersection state across path
  depths, and multi-draw knockoff selectors reuse fitted Gaussian models and
  draw-invariant augmented correlations.
- Fold/bootstrap-local Gaussian panels now screen with bounded column blocks
  before materializing float64 candidates, preserving stable two-pass moments
  without full-panel centered copies. Binary CEFS+ correlation pruning also
  uses bounded blocks instead of a dense all-candidate correlation matrix.
- The auto-k benchmark harness discards a warm-up run and reports the median
  of three timed runs by default (`--timing-repeats` controls the count).
- Gaussian cache construction (`build_cache`, all Gaussian/cache selectors,
  `select_fdr`): the weighted rank-Gaussian transform now scatters instead of
  re-sorting, uses a shared template for equal-weight untied columns, and
  gained a `rank_backend="threads"` option (used automatically when
  `n_jobs != 1`). Serial builds are about 1.8x faster and threaded builds up
  to ~10x faster on 50k x 2000 inputs; outputs are bitwise identical for
  float64 weights. Low-precision (float32) weight arrays are now accumulated in
  float64, which fixes a systematic tie/tail error in the weighted target
  transform used by cache selectors.
- The CEFS+ greedy loop is now a BLAS-free partial-Cholesky recursion:
  O(k^2 m) instead of O(k^3 m), and it no longer calls tiny BLAS products from
  Numba. On machines whose NumPy and SciPy ship different OpenBLAS builds this
  removes a thread-pool thrash that made `perm_gap`, `stability`, and
  `gaussian_cv` auto-k paths 4-7x slower than necessary. Paths are identical
  except in near-degenerate collinear panels, where the new recursion is the
  numerically correct greedy choice. `objective_from_corr_path` uses the same
  recursion.
- Binary CEFS+ refits warm-start from the previous prefix and stop on relative
  objective convergence; long paths and `k="auto"` (EBIC refit) are 5-8x faster
  with identical selections. The binary path now stays in float64 end to end
  and drops constants by standard deviation (`> 1e-12`) like the Gaussian
  cache, so large-offset or tiny-scale informative columns are no longer lost.
- `select_mrmr(mrmr_backend="auto")` now resolves to the BLAS redundancy path
  for every `n_jobs` (3-10x faster than the serial Numba loop; the process
  backend remains an explicit opt-in). The `f_regression`, `f_classif`, and
  standardization kernels sweep rows instead of columns (about 10x faster).
  The row-order-preserving traversal itself is bitwise equivalent; separately,
  regression relevance and JMI/mRMR standardization use a `1e-24` variance
  floor so genuinely varying tiny-scale features remain scale invariant instead
  of being treated as constants.
- `smart_sample` clips only touched inclusion probabilities per group and
  `quantile_anchors` uses a vectorized group quantile (about 2x faster on large
  grouped panels, identical output).
- KSG joint-MI neighbor counting is vectorized (about 3x faster).

### Knockoffs

- Added `statistic="lsm"` (lasso signed-max from a Gram-form LARS path on the
  analytic augmented correlation) and `statistic="ridge"` (analytic ridge
  coefficient difference). Both are exactly antisymmetric under original/
  knockoff swaps; `lsm` is markedly more powerful than the marginal
  `relevance` default on correlated designs while keeping the same approximate
  plug-in validity framing. Options: `statistic_options={"max_steps": ...}`
  for `lsm`, `{"ridge_lambda": ...}` for `ridge`.
- Added `feature_groups="auto"` with `group_corr_threshold`: features are
  clustered by absolute correlation, knockoffs run on one representative per
  cluster, and selected clusters are expanded. This restores power for
  near-collinear blocks. Representative-level plug-in calibration does not
  establish cluster- or feature-level FDR after expansion, and metadata says so
  explicitly. Correlation clustering/linkage has O(p^2) scaling.
- `select_fdr` now warns when the knockoff decorrelation is too small to have
  power (median `s < 0.05`) and reports `s_median` and
  `n_low_power_features` in the metadata.

### Correctness and validation

- Low-level classic selectors reject non-positive `top_m`; cached Gaussian
  selectors reject non-finite targets and correlation-pruning thresholds above
  one. Binary target metadata preserves raw labels and numeric ordering.
- Block permutations always move at least one block. Degenerate auto-k folds
  fall back with an explicit warning, non-finite score curves use the method
  floor, and chi-square floor clamps report `stopped_by="floored"`.
- CatBoost stability and CV modes are mutually exclusive; custom splitters are
  signature-checked, user overfitting-detector parameters are preserved, and
  stability output is capped at the selected k. Stability-selection alpha and
  threshold tuning now fit preprocessing inside their validation folds.
  Threshold tuning aligns DataFrame columns to fit order and preserves supplied
  sample weights, groups, and time through group-disjoint/time-ordered folds.
- Multi-draw knockoff metadata and docs explicitly distinguish per-draw q
  calibration from frequency aggregation, which has no aggregate FDR
  guarantee. Consensus gain tests preserve `m_mode`/panel eigenvalue semantics
  and each consensus submethod receives a distinct deterministic RNG stream.
- Boruta now requires `allow_full_data_target_encoding=True` before fitting a
  supervised categorical encoder (`loo`, `target`, `james_stein`,
  `loo_logit`) on the full dataset, matching the filter selectors. Tree
  learners can invert leave-one-out encodings and accept pure-noise
  high-cardinality categoricals otherwise.
- `ensure_weights` quantizes normalized weights to float32 precision and then
  restores their mean in float64. This greatly reduces rescaling-induced ulp
  changes that can alter tree tie-breaking; it is not an exact invariance claim
  for every representable input and scaling constant.
- Stability selection classification alpha search preserves accuracy-scored
  sparse-model selection while fitting imputation and scaling inside each CV
  fold. The chosen `C` is rescaled by total training weight for each bootstrap
  so the per-sample regularization matches the CV calibration.
- Gaussian mRMR/JMI/JMIM reject non-finite regression targets instead of
  silently treating them as neutral ranks; Gaussian mRMR warns when it selects
  features whose relevance is at the noise floor.
- `evaluate_feature_path` treats integer path entries as positions even under
  duplicate column labels and passes the real target to stratified splitters.
- Selector classes route CEFS+ / binary CEFS+ `k="auto"` without a config
  through the Auto-K router like the function API (including `loss="brier"`).
- `smart_sample` raises if the input already contains a `sample_weight`
  column instead of overwriting it.

- Expanded experimental auto-k selection with EBIC/RIC penalties,
  pseudo-posterior `k` diagnostics, calibrated CEFS+ gain stops,
  permutation-gap null envelopes, closed-form Gaussian CV curves,
  knockoff-path stopping, bootstrap stability, changepoint diagnostics,
  consensus diagnostics, and a synthetic auto-k harness.
- Changed no-config CEFS+ and binary CEFS+ `k="auto"` to use the measured
  Auto-K v2 router. Calls with `groups` or `time` now route to EBIC by default
  instead of the older `evaluate/group_cv` or `evaluate/time_holdout` path, and
  calls without validation context now work. The router uses method-specific
  effective floors, so pass an explicit `AutoKConfig` when you need a hard
  `min_k` or the legacy evaluate behavior.
- Auto-K router diagnostics now flag saturated/censored results in
  `auto_routing["saturated"]` and emit a `UserWarning` when the selected k hits
  the effective maximum.
- Added an opt-in dense-regime Auto-K diagnostic:
  `AutoKConfig(k_method="auto", auto_dense_check=True)` cross-checks large
  EBIC picks against `gaussian_cv` with `selection_rule="best"` and warns when
  detectable-feature count and downstream-size proxy disagree sharply.

### CI

- The CatBoost dependency job now runs the full test suite with all optional
  dependencies installed. The redundant Python 3.11 Numba job was removed;
  Numba is a required dependency and remains covered by every base matrix job.

## 0.7.0

- Added q-calibrated Gaussian-copula knockoff selection: `select_fdr`,
  `KnockoffSelector`, `sample_knockoffs`, feature-group thresholding, and
  approximate plug-in validity metadata.
- Added `benchmarks/bench_knockoffs.py` for the new knockoff timing surface.
  Its quick/full records are informational smoke data rather than promotion
  gates.
- Accelerated Gaussian cache construction by vectorizing the weighted
  rank-Gaussian transform; this benefits all Gaussian/cache selectors, not only
  knockoffs.
- Added tie-safe `statistic="cefsplus"` for knockoffs, with objective-gain
  scoring and optional `min_gain_ratio` early stopping for large runs.
- Implemented diagonal coordinate-descent `s_method="mvr"` and `"me"` optimizers
  for the MVR and maximum-entropy knockoff objectives.
- Weighted `build_cache(..., subsample=...)` now samples from positive-weight
  rows. Seeded weighted caches can choose different rows than pre-release builds;
  unweighted seeded caches preserve the old row choices.
- Knockoff noise now uses NumPy float32 standard-normal draws. Seeded knockoff
  samples can differ from pre-release builds for the same `random_state`.
- Documentation and metadata consistently frame knockoff FDR control as an
  approximate plug-in Gaussian-copula result unless the fitted feature model is
  the true Model-X distribution.
- Added standalone API, algorithm, advanced-workflow, and contributing docs
  aligned with the 0.7.0 public surface.
- Release documentation now links the `select_fdr` workflow, `KnockoffSelector`,
  `sample_knockoffs`, and the focused docs/benchmark smoke checks from the
  README, development guide, benchmark guide, and release tracker.
