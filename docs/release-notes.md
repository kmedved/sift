# Release Notes

## Unreleased

### Breaking changes and migration

- `stability_regression(..., k=…)` and `stability_classif(..., k=…)` no longer
  pad short selections with never-selected features: `k` caps the count and
  `threshold` gates membership, so wrappers can now return fewer than `k`
  features (including zero). Rank by `selection_frequencies_` yourself if you
  need a fixed-size list.
- `k='auto'` (router) calls now emit a `UserWarning` when they select zero
  features, and `select_cefsplus` warns when `y` contains only 3-20 distinct
  integer-valued levels (labels-shaped targets). Selector classes reject 1-D
  `X` with a `ValueError` instead of an `IndexError`. The no-config router
  routes time-context non-CEFS+ Gaussian selectors to
  `gaussian_cv/time_holdout` with `selection_rule="best"` (previously the
  `one_se` request fell back to `best` with a warning), no longer re-warns
  about `auto_dense_*` fields it already consumed, and
  `StabilitySelector.selection_frequencies_` is now float64.

- Prebuilt Gaussian caches now enforce their full source contract. Named caches
  require the same row count and exact DataFrame names/order; positional caches
  require a positional ndarray with the same row and feature counts. Reordered,
  renamed, or duplicate columns raise for named caches, and a DataFrame cannot
  consume a positional cache. Positional caches cannot detect reordered ndarray
  columns, so callers must preserve their original positions. Cached filter
  calls reject call-time `sample_weight`, `subsample`, and construction
  `random_state`; `select_fdr` still accepts `random_state` because it seeds a
  fresh knockoff draw, while its `subsample` remains forbidden. Rebuild
  persisted caches that predate `feature_names_are_synthetic`.
- Fixed-k filter calls now reject `groups` and `time`; remove those arguments or
  use `k="auto"` with the matching evaluation strategy. `KnockoffSelector`
  rejects row `groups`/`time` in every mode; its `feature_groups` option groups
  features, not observations.
- Datetime and timedelta feature columns, including NumPy datetime/timedelta
  arrays, now raise before numeric coercion in classic, cache, and Boruta paths.
  Derive explicit numeric calendar or elapsed-time features before selection.
- Function-style filters using `task="classification"` follow sklearn's
  discrete-target contract. String, categorical, integer, and integer-valued
  floating labels remain valid; non-integral numeric class codes such as
  `0.5`/`1.5` are classified as continuous and rejected. Re-encode those values
  as categories or integer IDs. `select_fdr` is separate: it requires a finite
  numeric target and does not use the classification-task contract.
- Time holdout moves the requested cut to the nearest boundary between distinct
  timestamps, preferring the smaller boundary on an exact tie. `val_frac` is
  therefore approximate and row counts can change. Fewer than two rows,
  all-tied, missing, or mutually unorderable timestamps now raise.
- Classic numeric filter feature matrices now stay in float64, preventing
  large-offset signals from collapsing. Their core feature-array footprint is
  therefore roughly twice the former float32 path, and peak memory can be
  higher because of copies or solver workspaces. BLAS runtime and native-thread
  contention remain workload-dependent; benchmark representative data when
  choosing the mRMR backend.

### Correctness, API, and documentation

- Smart-sampler regression targets now remain float64, are robustly centered
  on the pilot median, and use two-fold cross-fitted predictions for every row.
  This prevents large-offset target collapse and in-sample residual optimism;
  non-pilot rows use one unseen fold model to preserve a comparable residual
  scale, and constant pilot targets now disable the residual blend.
- Stability selectors now reject one-dimensional inputs with a clear
  `ValueError`. Elbow selection accepts integer objective paths, validates its
  direct arguments, and stops before the first feature in a patience-confirmed
  flat-gain run rather than retaining that zero-gain feature, subject to the
  configured `min_k` floor.
- `permutation_importance` accepts sklearn scorer objects and `ScoringSpec`, and
  exposes `higher_is_better` only for legacy loss callbacks, avoiding a second
  direction flip for signed scorers. CatBoost result objects
  persist `selection_patience`, so `features_within_tolerance()` uses the same
  consecutive-miss rule as fit-time selection. Auto-k saturation warnings now
  distinguish a configured `max_k` cap, exhaustion of the candidate path, and
  a fold/statistical limit that ends an evaluation curve before the path.
- Gaussian/cache-backed sklearn selector constructors use
  `subsample="auto"`, resolving to 50,000 rows only at fit time. MRMR, JMI,
  JMIM, and CEFS+ wrappers also use `random_state="auto"`, resolving to seed 0;
  `KnockoffSelector.random_state` remains numeric because it seeds each fresh
  draw. These literals preserve explicit cache-override rejection while
  satisfying sklearn's default-constructible estimator parameter contract.
- Weighted binned JMI/JMIM now use weighted quantile edges as well as weighted
  entropy counts. Zero-weight rows do not affect binning, and multiplying all
  weights by a positive constant does not change the estimand. Integer
  frequency ratios are reduced by any common global factor rather than
  treating that factor as extra replicated sample size.
- Grouped time-block stability bootstrap now honors `sample_frac` with a rounded
  panel-wide draw budget allocated across unequal groups. Moving, circular, and
  stationary windows draw with replacement and preserve the full-panel budget
  at `sample_frac=1.0`.
- Documentation now records cache/X compatibility and rejected cache overrides,
  fixed-k group/time rejection, the `cat_encoding="none"` default, and the
  stochastic row-order sensitivity of `KnockoffSelector`. Knockoff statistic
  power comparisons are intentionally left data-dependent pending a committed
  quality bakeoff.
- All sklearn-style selector wrappers now expose `get_feature_names_out()`.
  `KnockoffSelector` is tagged and documented as row-order-sensitive despite a
  fixed seed; zero-weight rows are still removed before knockoff RNG draws.

## 0.8.0

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
  and drops only exactly constant columns like the Gaussian cache, so
  large-offset or tiny-scale informative columns are no longer lost.
- Binary CEFS+, R² JMI, ridge knockoffs, and stability bootstrap fits now apply
  narrow one-thread native-pool scopes around repeated matrix operations.
  This prevents multiple OpenBLAS/OpenMP runtimes from oversubscribing one
  another; `threadpoolctl` is now a direct dependency.
- `select_mrmr(mrmr_backend="auto")` now resolves to the BLAS redundancy path
  for every `n_jobs` (3-10x faster than the serial Numba loop in the repository
  benchmark cases, not a universal guarantee; the process backend remains an
  explicit opt-in). The `f_regression`, `f_classif`, and
  standardization kernels sweep rows instead of columns (about 10x faster).
  The row-order-preserving traversal itself is bitwise equivalent; separately,
  regression relevance and JMI/mRMR standardization use exact-constancy checks
  so genuinely varying tiny-scale features remain scale invariant instead of
  being treated as constants.
- `smart_sample` clips only touched inclusion probabilities per group and
  `quantile_anchors` uses a vectorized group quantile (about 2x faster on large
  grouped panels, identical output).
- KSG joint-MI neighbor counting is vectorized (about 3x faster).

### Knockoffs

- Added `statistic="lsm"` (lasso signed-max from a Gram-form LARS path on the
  analytic augmented correlation) and `statistic="ridge"` (analytic ridge
  coefficient difference). Both are exactly antisymmetric under original/
  knockoff swaps and keep the same approximate plug-in validity framing. Power
  relative to the marginal `relevance` default is data-dependent; no universal
  advantage is claimed without a committed quality bakeoff. Options:
  `statistic_options={"max_steps": ...}` for `lsm`,
  `{"ridge_lambda": ...}` for `ridge`.
- Added `feature_groups="auto"` with `group_corr_threshold`: features are
  clustered by absolute correlation, knockoffs run on one representative per
  cluster, and selected clusters are expanded. This restores power for
  near-collinear blocks. Representative-level plug-in calibration does not
  establish cluster- or feature-level FDR after expansion, and metadata says so
  explicitly. Correlation clustering/linkage has O(p^2) scaling.
- `select_fdr` now warns when the knockoff decorrelation is too small to have
  power (median `s < 0.05`) and reports `s_median` and
  `n_low_power_features` in the metadata.
- CEFS+ knockoff paths no longer have a silent ten-discovery default cap. The
  implicit path depth starts from a q-aware bound and expands when discoveries
  saturate it; explicit saturated caps warn and depth metadata records the
  initial and final values.

### Correctness and validation

- Function selectors now reject unknown `task` values and continuous
  classification targets. Regression targets remain float64 across classic,
  Gaussian, stability, and smart-sampling paths, preventing large-offset target
  collapse.
- Prebuilt Gaussian caches reject call-time `sample_weight` instead of silently
  ignoring it. Seeded knockoff caches discard zero-weight rows before sampling,
  so irrelevant rows do not consume RNG draws.
- Stability auto-k excludes the tautological `k=p` agreement endpoint.
  Threshold tuning uses scale-equivariant ridge scoring, and automatic
  stability regularization defaults to a one-standard-error rule; users can
  request prediction-optimal CV with `alpha_rule="best"`.
- CEFS+ correlation pruning is now opt-in. The unpruned default preserves
  suppressor pairs, while `corr_prune=0.95` remains available for
  duplicate-oriented diversity.
- Function-style categorical encoding defaults to `"none"`, matching the
  full-data leakage guard instead of selecting a supervised encoder that is
  rejected by default.
- Fixed-k filter results now retain complete rankings, relevance, path scores,
  selected indices, and method diagnostics. Routed auto-k metadata omits
  strategy and selection-rule fields for methods that did not use them.

- Fixed `catboost_select(k=None)` reporting the wrong `best_k`: the old scan
  walked down from the largest count and stopped after `selection_patience`
  non-improvements even though every count had already been scored, so a
  better small prefix could be missed (a curve with its optimum at the
  smallest count returned the largest count). The score-curve optimum is now
  the global arg-best; `tolerance` and `selection_patience` implement a separate
  parsimony rule for the returned feature count (smallest count within the
  tolerance band of the best, giving up after `selection_patience` consecutive
  misses). Both parameters are now validated. Automatic selections can
  therefore move to smaller counts than before.
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
- The distribution is now published as `sift-feature-selection` (while the
  import remains `sift`) to avoid the occupied `Sift` PyPI project. Wheels
  exclude benchmark packages, PyPI renders the concise README with absolute
  links, and the release workflow uses a separate verified build plus OIDC
  Trusted Publishing. Critical Ruff checks and a scheduled quick benchmark
  promotion gate now run in CI.

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
