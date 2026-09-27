# Release Notes

## 1.0.1 (unreleased)

A patch release. The 1.0 defaults, all 66 exports and every public signature
are unchanged, and calls that already worked select the same features, except
for the input-dependent changes listed under Compatibility. Each item below
compares with 1.0.0.

### Compatibility

- A prebuilt `FeatureCache` or `ClassicFeatureCache` meets `cat_encoding`
  with one rule on `select_mrmr`, `select_jmi`, `select_jmim`,
  `select_cefsplus`, their selector classes and `KnockoffSelector`: any
  encoding other than `"none"` raises one message, naming the encoding and the
  columns, once there is a column to encode (a `cat_features` column of a
  DataFrame, else its object/category/string columns; for an ndarray, an
  in-range `cat_features` position or its `x{i}` name). With nothing to
  encode, the encoding is inert and the call returns the `cat_encoding="none"`
  selection. Consequences:
  - Calls that silently ignored a supervised encoding (`"target"`, `"loo"`,
    `"james_stein"`, `"loo_logit"`) beside a cache and a column to encode now
    raise: the Gaussian function APIs on a DataFrame or on an ndarray with
    positional `cat_features`, and the selector classes and `KnockoffSelector`
    on an ndarray with positional `cat_features`. The ignored encoding also
    skipped the `allow_full_data_target_encoding` opt-in. Pass
    `cat_encoding="none"` with a cache built from encoded data.
  - Calls with nothing to encode that used to raise the cache error now run:
    one-hot, ordinal and frequency on the filter selector classes, ordinal and
    frequency on `KnockoffSelector`, one-hot on the function APIs, any encoding
    with a classic cache, and any call whose `cat_features` names no column (a
    misspelt name, an out-of-range position). For one-hot, a call of the last
    kind instead reaches one-hot's own error: "onehot cat_features are not
    columns of X" on a DataFrame, and a `TypeError` ("requires a pandas
    DataFrame") instead of the cache `ValueError` on an ndarray.
- `cat_encoding="target_cv"` encodes a column with a single positive-weight
  level in its fitting rows (for example a constant or all-missing
  categorical) as exactly zero, in every fold and at inference. It used to
  become rounding noise (about 1e-17) that every route could select: the
  Gaussian routes kept it and the rank transform inflated it, and the classic
  MI and binary CEFS+ estimators ranked the noise as informative. It is now
  dropped as constant on every route, which can change other picks and auto-k
  choices, and naming it in `include` raises the "dropped as constant" error.
- `AutoKConfig.random_state` is validated where a rule reads it: `perm_gap`,
  `knockoff_path`, `stability`, `gaussian_cv` / `xfit_objective` with
  `strategy="kfold"`, a `consensus` with any of those members (its
  `gaussian_cv` / `xfit_objective` members read the seed whatever the
  strategy), and the rule `k_method="auto"` routes to. A seed counts as the
  integer it equals, so Python and NumPy integers and booleans, integral
  floats, `Decimal` / `Fraction` values, 0-d arrays and `IntEnum` members work
  exactly as before. `None`, a string and a non-integral number now raise a
  `ValueError` naming the rule; they used to raise a bare or wrapped
  `TypeError` (`None`) or be truncated by `int()` (`1.5`, `"7"`). A 1-d array,
  which worked with NumPy's deprecation warning, raises too. The
  `auto_dense_check=True` cross-check skips itself with its warning for such a
  seed, naming the reason; 1.0.0 skipped it for `None` but ran it on the
  truncated seed for `1.5` or `"7"`. The dense check is diagnostic, so the
  selection is the same either way. Out-of-range integers that NumPy
  rejected get a message naming the range. Rules that never read the seed
  accept any value, and a `consensus` with a `gaussian_cv` / `xfit_objective`
  member no longer warns that `random_state` is unused.
- The ordinal, frequency and one-hot encoders identify datetime and duration
  levels exactly:
  - `cat_encoding="ordinal"` orders datetimes by instant and durations by
    length at any magnitude. Datetimes outside pandas' nanosecond range
    (before 1677 or after 2262, aware values included), `np.datetime64` /
    `np.timedelta64` values beyond about ±292,000 years, large
    non-nanosecond durations and count-multiplied units such as `m8[3D]` used
    to rank after strings, wrap or misorder, as did `np.datetime64` values
    with a count multiplier or in `ps` / `fs` / `as`. `Decimal` and `Fraction`
    levels used to sort by their text and now sort numerically with ints and
    floats. Other datetime and duration values, and every other level type,
    keep their codes.
  - An `np.timedelta64` in an object column is identified by its exact
    length. Most units (`W`, `D`, `h`, `m`, `s`, `ms`, `us`) used to crash in
    `int()`; nanosecond, generic, year, month and sub-nanosecond values used to
    merge with the integer of the same count (`np.timedelta64(5, "ns")` and `5`
    were one level). Equal lengths are one level in any unit (`5000 ps` and
    `5 ns`), years and months count in months (one year is `...__12 months`),
    and a count multiplier counts (one tick of `m8[2Y]` is 24 months, no longer
    one year; one tick of `m8[3D]` is three days).
  - An object column mixing `np.datetime64` units such as years and
    attoseconds no longer raises `OverflowError`; each unit stays its own
    level, and equal instants in two units, which 1.0.0 merged only when
    NumPy's hash happened to collide (the epoch, raw counts -1 and -2), are now
    always two levels. An aware non-nanosecond `pd.Timestamp` outside years
    1-9999 no longer crashes, and the two readings of a repeated wall-clock
    hour (a DST fold) are two levels instead of one whose code depended on row
    order.
- `evaluate_feature_path` imputes `+inf` / `-inf` like missing values and
  computes the fill without overflow, so its scores change for a training
  column that contains an infinity or whose finite values sum past the float
  range (both used to be filled with `0.0`).
- Several error and warning messages are reworded (listed under Fixes), and
  some calls that fail for more than one reason report a different reason
  first: `CEFSPlusBinarySelector` with a non-binary target reports the target
  error before a category_encoders `ImportError` or a weight error, and an
  invalid `within` value is reported before nested auto-k errors. View metadata gains `proxy_correlations_stale`
  (`StabilitySelector`, `Stabilized`) and `proxy_input_numeric` (`Stabilized`
  over non-numeric input), two-way results gain `within_two_way_converged` /
  `within_two_way_max_residual`, and CatBoost manifests gain their resolved
  `higher_is_better`. Tests that match old text or exact key sets need
  updating.

### Fixes

- `reproducibility_()` manifests report sift's own git state. When sift is
  installed into a virtualenv inside a different checkout (for example a
  git-ignored `.venv`), `environment.git_commit` and `git_dirty` are now
  `None`; they used to show that project's HEAD, and `git_dirty=False` even
  after the installed sift was edited. An exported `GIT_DIR`,
  `GIT_WORK_TREE`, `GIT_COMMON_DIR` or other repository-selecting git variable
  no longer redirects them, and a `status.showUntrackedFiles=no` setting no
  longer hides a new untracked module from `git_dirty`. Checkouts, editable
  installs and git worktrees of sift still report their commit and dirty
  state.
- `StabilitySelector.fit`, `stability_regression`, `stability_classif` and
  `stability_select` accept a single-column target (an `(n, 1)` array or a
  one-column DataFrame) and select exactly what the 1-D target selects; they
  used to crash with a NumPy `TypeError` (regression) or emit scikit-learn's
  `DataConversionWarning` (classification). A wider, 3-D, zero-width or
  scalar target, or one whose length differs from `X`, raises a `ValueError`
  naming the shape or both row counts, and `tune_threshold` rejects a wide
  target too.
- `select_fdr`, `KnockoffSelector` and `Stabilized(aggregation="evalues")`
  raise a `ValueError` naming `random_state` for a `None` or non-integer
  knockoff seed, which used to fail with a bare `TypeError` from inside the
  filter; negative seeds, which already raised NumPy's `ValueError`, get the
  same message. `sample_knockoffs` names `random_state` for a float, string or
  negative seed and lists the forms it accepts. On all four, a NumPy boolean
  now seeds as the integer it equals, as a Python `bool` does; it used to fail
  inside `SeedSequence`. Integer seeds give identical draws.
- `evaluate_feature_path` results record the estimator, the splitter (or the
  default holdout with its `val_frac`), digests of `time` / `event_end`, and
  `random_state` when it seeded the holdout in their reproducibility manifest
  (`seeds.available` was `false`). `catboost_select` results record their
  options and seed. A hand-assembled `FeaturePathEvaluationResult` or
  `CatBoostSelectionResult` reports `captured_at="unknown"` instead of
  claiming selection-time capture, and a CatBoost result whose column labels
  have no deterministic token (a pandas `Interval`, say) records
  `columns_hash: null` instead of raising.
- `compare` accepts a pandas `Period` or a one-column `time`; hashing it for
  the split record used to raise. Period values have a deterministic manifest
  token (frequency and ordinal), and row metadata with no deterministic token
  is recorded as `{"status": "opaque", "reason": "no_deterministic_token"}`.
  `PurgedTimeSeriesSplit` and `GroupPurgedTimeSeriesSplit` accept a `Period`
  time axis of one frequency and split on its ordinals, so `compare` and
  `evaluate_feature_path` can purge on it. The embargo is an integer count of
  ordinal steps of the frequency's base unit, not of periods: months for
  `'M'` and for `'2M'` (so `embargo=2` covers one `'2M'` period), minutes for
  `'15min'`. A Categorical of Periods is rejected like every categorical, with
  a remedy.
- `evaluate_feature_path` fills `+inf` / `-inf` cells, like missing ones, with
  the mean of the column's finite training values; a column that contained an
  infinity used to have all its non-finite cells filled with `0.0`, and some
  columns leaked NumPy `RuntimeWarning`s despite the documented silent
  imputation.
- Panel `within=` auto-k errors and warnings name routes that work and never
  recommend the route that produced them:
  - `AutoKConfig(k_method="evaluate", strategy="kfold")` on the filter
    functions and selector classes no longer answers "use time_holdout or
    group_cv" (dead ends under `within`); the zero-config CEFS+ router,
    non-fold methods such as `elbow`, and nested auto-k no longer suggest
    `evaluate` for `within="two_way"`; `select_k_auto` no longer suggests
    `select_k_elbow`, which takes no `within`.
  - The guidance says that `gaussian_cv` / `xfit_objective` need the Gaussian
    path (`estimator="gaussian"` for mRMR, JMI and JMIM), that they also
    validate `within="groups"` under `time_holdout`, and that `select_k_auto`
    cannot validate `within="two_way"`.
  - Under `strategy="kfold"`, the partially-unseen-levels warning advises
    pooling thin entities or periods or raising `AutoKConfig.xfit_folds`, and
    the no-seen-level error explains that more folds make such a fold more
    likely; under `time_holdout` both point to the Gaussian `kfold` route.
  - Where the within and conditioning rejections used to point at each other
    (`within` with `include` / `exclude` / `candidates` and `k="auto"`), one
    message now says no auto-k method serves both under `"two_way"` (use a
    fixed `k`, or drop the keywords on the Gaussian path) and names the one
    route that does under `"groups"` (`k_method="evaluate"` with
    `strategy="time_holdout"`, or a fixed `k` without `time`). Structural,
    configuration, cache and conditioning errors still come first; errors
    found while preparing the data (y length, non-finite y, negative weights,
    non-numeric X) now follow this message.
- `SelectionView.table` keeps `within_relevance` and `between_relevance`
  whenever the filter ranking has them, so `view.table["between_relevance"]`
  no longer raises `KeyError` when the score is NaN (two or fewer entities).
  `within="two_way"` results also report `within_two_way_converged` and
  `within_two_way_max_residual` next to `within_two_way_iterations`.
- `include` errors name exactly the raw columns that cannot be conditioned on,
  on every route, all of them, in the caller's `include` order: the one-hot
  selector classes no longer blame every `include` column;
  `select_cefsplus_binary` and `CEFSPlusBinarySelector` (including
  `loss="brier"`) no longer quote a one-hot dummy such as `city__NY`; the
  classic mRMR/JMI/JMIM routes name the column instead of an encoded
  position; `select_fdr` / `KnockoffSelector` name the column and mention a
  cache only when one was passed; and the low-level
  `sift.selection.mrmr_select` / `jmi_select` say "include positions".
- Cache errors come in the order a caller can act on:
  `CEFSPlusBinarySelector` and nested auto-k reject a prebuilt cache before
  the encoding rule advises re-encoding, a `cat_features` NumPy array no
  longer fails with NumPy's truth-value error, and a bare `str`
  `cat_features` gets a message explaining that each character was read as a
  column name.
- The filter selector classes with a 2-D `y` and a supervised `cat_encoding`
  raise the function API's message instead of a shape error such as "X has
  240 rows but y has 480".
- A `Decimal("sNaN")` category value is missing, like `Decimal("NaN")`, for
  the ordinal, frequency, one-hot, `target_cv` and `loo_logit` encoders
  instead of crashing them.
- `Stabilized(store_proxies=True)` suggests editing a block only when the
  selected constant column is a member of a multi-member `feature_blocks`
  entry of its base; otherwise it names the column.
- `SelectionView.proxies`, `proxies_at`, `redundancy_report` and
  `proxy_clusters` no longer blame a threshold change when proxies were never
  stored. A source that cannot store proxies (classic or log-loss filter
  routes, one-hot encoding, `ModelSelector`, knockoffs) is pointed only at the
  `store_proxies=True` routes that take its kind of target, and a `Stabilized`
  fit on non-numeric input is told to encode those columns first. A view whose
  stored block was invalidated by `StabilitySelector.set_threshold` still
  explains that the lower threshold needs a refit;
  `metadata["proxy_correlations_stale"]` says which case applies.
- The runtime-scaling benchmark records the version of the SIFT source it
  measured rather than whatever distribution metadata is installed.

### Documentation

- Corrected the README's output-order example, which on 1.0 printed input
  order for the selector it labelled as legacy; it now passes
  `output_order="legacy"`, and a test binds its printed orders to its
  comments. The README gains an "Upgrading to 1.0" section with the explicit
  settings that restore 0.10 behavior.
- Completed the 1.0.0 notes with a "Breaking changes and migration"
  subsection: which transform methods and result views follow input order,
  the removed `random_state=None` `FutureWarning`, the CatBoost collision
  notice, and a per-entry-point table of restore settings.
- Corrected the 0.10.0 note on one-hot encoding (prefix-only auto-k fits the
  path-building encoder on every row; only held-out scoring refits per fold),
  added the 0.10.0 changes to 0.9.0 APIs that the notes had omitted
  (`evaluate_feature_path` `time=` / `event_end=` and generator splitters,
  `proxies_at(include_selected=)`, the full encoding list in the non-numeric
  error), and restored the deprecation ledger's 0.9 export count to the 58
  names v0.9.0 shipped.
- Documented the prebuilt-cache `cat_encoding` rule in one place (the cache
  section of the manual) and fixed the `CEFSPlusSelector` Raises entry that
  described it backwards; `compare` documents when its stratified default
  passes through scikit-learn's "least populated class" warning and how an
  explicit `cv=` avoids it.
- Corrected docstrings: `include` / `exclude` / `candidates` references in
  `select_fdr` and `select_cached`, the fold-local encoding in
  `xfit_objective_curves` / `gaussian_cv_curves`, what `compare`'s
  `random_state` seeds, `KnockoffSelector.eta` (it rescores only
  `n_discoveries_offset_0`), the `purged_kfold` cap's distance rule and
  tie-break, `AutoKConfig.random_state`, and the two-way demeaning
  convergence criterion in every `within` docstring (largest removed level
  mean scaled by the column's weighted SD, below `1e-10`, not a relative
  change).
- Updated pages and trackers that still described 0.9-era plans as current.

## 1.0.0 (2026-09-21)

### Breaking changes and migration

This subsection was added after the release to spell out the upgrade; the
behavior it describes is what 1.0.0 shipped. The README has a short
[Upgrading to 1.0](https://github.com/kmedved/sift/blob/main/README.md#upgrading-to-10)
guide.

- Selector transforms now follow fitted input order. On the ten selector
  classes, `transform`, `fit_transform`, `get_feature_names_out()`,
  `get_support(indices=True)` and `inverse_transform` emit the selected
  columns in ascending input position instead of selection order, and the
  `sift.as_result` views of a fitted `StabilitySelector`, `ModelSelector` or
  `Stabilized` (`features`, `indices`) follow the same order. Which features
  are selected, `selected_features_`, result tables and the lists returned by
  the function API are unchanged. Refit a downstream model that was trained on
  0.10 transform output, or keep `output_order="legacy"`.
- Omitted `random_state` on `StabilitySelector`, `permutation_importance` and
  `catboost_select` (and the `stability_regression`, `stability_classif`,
  `catboost_regression` and `catboost_classif` wrappers) is now seed `0`, so
  those calls are reproducible, and the `FutureWarning` that 0.9 and 0.10
  emitted for `random_state=None` is removed. An explicit `random_state=None`
  still draws fresh entropy, now without a warning. The same entry points now
  default to `n_jobs=1`. Because seed `0` and one worker are now translated
  CatBoost settings, a `catboost_params` `random_seed` or `thread_count`
  override now emits the existing collision `UserWarning`; the dictionary
  value still wins.
- Public `verbose` defaults are now `False`.

To keep 0.10 behavior, pass the old values explicitly. Not every entry point
has all four settings:

| entry point | explicit 0.10 settings |
| --- | --- |
| `StabilitySelector` | `output_order="legacy", verbose=True, n_jobs=-1, random_state=None` |
| `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector`, `KnockoffSelector`, `Stabilized` | `output_order="legacy", verbose=True` |
| `BorutaSelector` | `verbose=True` (its legacy order already was input order) |
| `ModelSelector` | `output_order="legacy"` (its `verbose` default was already `False`) |
| `permutation_importance` | `n_jobs=-1, random_state=None` |
| `catboost_select` | `n_jobs=-1, random_state=None, verbose=True` |
| `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, `select_cefsplus_binary`, `select_fdr`, `select_boruta`, `select_boruta_shap`, `SmartSamplerConfig` | `verbose=True` |

### Compatibility

- Implemented the defaults announced in 0.10.1 for 1.0 development. Omitted
  `random_state=None` defaults in stability, permutation importance and
  CatBoost are now `0`; public progress defaults are quiet; existing public
  `n_jobs=-1` defaults are now `1`; and the ten selector transformers default
  to original input order. Explicit `None`, `verbose=True`, parallel worker
  settings and `output_order="legacy"` remain supported.
- Existing seed-42 and cache-aware defaults, all 66 exports, result forms,
  permanent aliases, selector mathematics and the scikit-learn 1.3 floor are
  unchanged. CatBoost dictionaries retain override precedence and their
  corrected collision warning.

## 0.10.1 (2026-09-21)

### Compatibility

- Added Python 3.13 to the full-suite CI matrix. `KnockoffSelectionResult`
  equality again handles a shared pandas payload without evaluating the
  DataFrame as a boolean, preserving its behavior from earlier Python versions.
- Announced the approved 1.0 compatibility defaults in advance. Omitted
  `random_state=None` defaults in stability, permutation importance and
  CatBoost become `0`; public progress defaults become quiet; existing
  `n_jobs=-1` defaults become `1`; and selector transforms default to original
  input order. Seed-42 and cache-aware defaults, all 66 exports, existing
  result forms, permanent aliases, selector mathematics and the scikit-learn
  1.3 floor remain unchanged. The flips are reserved for 1.0, with no new
  per-call deprecation warnings merely for omitted options. Explicit legacy
  settings remain available. CatBoost dictionaries still override translated
  SIFT options; because seed `0` and worker count `1` are now translated
  defaults, an existing `catboost_params` `random_seed` or `thread_count`
  override starts emitting the existing collision `UserWarning` while its
  dictionary value continues to win. Warnings-as-errors callers using those
  overrides may need to handle that notice.

## 0.10.0 (2026-09-20)

### Compatibility

- Raised the minimum supported Python version from 3.10 to 3.11. CI now tests
  Python 3.11 and 3.12, and the Python 3.12 job installs PyArrow so Arrow dtype
  contracts run instead of being skipped.

### Features

- Added additive `Stabilized(selector)` to threshold selection frequency of any
  cloneable sklearn-style selector or SIFT wrapper (`n_resamples=30`,
  `resample="half"|"bootstrap"|"blocks"`, `threshold=0.6`, `random_state=0`).
  Half draws without replacement; bootstrap draws with replacement; blocks use
  the existing group/time block-bootstrap helper and require both `groups` and
  `time` at fit time. Row metadata is never a constructor array. Frequencies
  are float64 over the raw fitted width. `aggregation="evalues"` reuses
  native full-data `KnockoffSelector` e-value derandomization and rejects
  resample/threshold overrides; it does not average e-values across bootstrap
  datasets or claim FDR for ordinary frequency voting. Opt-in F2 proxy storage
  and an additive `SelectionView` / reproducibility manifest are included.
  `StabilitySelector` and KnockoffSelector defaults are unchanged. The public
  surface grows to 66 names in this 0.10.0 release. Not a 1.0
  preset rewrite.
- Added additive `sift.compare` for leakage-safe selector comparison. Factories
  refit inside each training fold; a fresh downstream estimator is scored on
  the untouched fold. Results include score distributions, explicit-unit mean
  `k`, and raw-feature selection-frequency overlap. `mode="in_sample_path"`
  is a labelled in-sample prefix diagnostic. Empty selected sets score an
  intercept-only predictor. `groups`/`time` are forwarded only to selectors
  that accept them. Reproducibility manifests are implemented as
  `reproducibility_()` on results, `SelectionView`, and `CompareResult`
  (schema `"1"` JSON: export-time environment/BLAS/package-bound git, original
  vs used rows, typed column hash, opt-in caller data hash, retained cache
  provenance, known new-run configuration/seeds, compare selector/estimator/
  splitter snapshots and fold fingerprints). The comparison API is additive;
  existing selector defaults and return types are unchanged.
- Added additive `cat_encoding="onehot"` on filter function APIs and sklearn
  filter wrappers. Each raw categorical becomes `{column}__{level}` dummies
  selected as one F3 block. Default `onehot_max_levels=32` (positive-weight
  mass, then label); remainder levels share `other`; unknown values join
  `other` when pooling created that remainder, otherwise they are all-zero;
  missing is its own level. Selected names/indices/support stay raw; encoded
  transform names match dummy width. Nested `evaluate` learns the vocabulary
  inside training folds. Prefix-only evaluate, Gaussian CV, xfit and auto
  routing refit the encoder on each training fold for held-out scoring, but
  the path-building encoder that fixes the dummy columns is fit on every row
  (on the training partition under `k_method="evaluate"` with
  `strategy="time_holdout"`); the map is target-blind, so this is not target
  leakage. (Corrected after release: this note first said the prefix-only
  routes learn the vocabulary on training folds.)
  Caches, `within`, knockoffs, and Boruta raise. `target_cv` and no-encoding
  paths are unchanged.
- Added additive `cat_encoding="ordinal"` and `"frequency"` on existing
  `cat_encoding` surfaces (filter functions/wrappers, binary CEFS+, Boruta
  `importance_data='train'`, `KnockoffSelector`, multi-target CEFS+).
  Target-blind numeric maps; unknown `-1` / `0`; explicit `sample_weight` only
  (not class-weight). Held-out scoring fits maps on training-fold raw `X`;
  evaluate/time-holdout path maps use the train partition. Prefix-only ranking may
  still use a full-data path. Prebuilt caches, resampled auto-k, and Boruta
  `importance_data='test'` raise. No new public exports. No FDR upgrade.
- Added joint multi-target Gaussian CEFS+ for numeric 2-D targets (`n×q`,
  `q≥2`). It uses target covariance and a `q·k` information-criterion
  dimension; one-column targets retain the 1-D path. Other selectors reject
  multi-target input with a clear error.
- Added generic `ModelSelector` with RFE/forward/stability strategies and
  additive `PurgedTimeSeriesSplit` / `GroupPurgedTimeSeriesSplit` splitters.
  Purging and embargo are applied at validation boundaries; group-aware
  splitting keeps entity identities disjoint between training and validation.
- Added `ClassicFeatureCache` and `build_classic_cache` for reusable numeric
  classic-selector statistics. Cache reuse requires the original source
  contract; row resampling and call-time weights/subsampling are not silently
  re-applied to a prebuilt cache.
- Added additive `feature_blocks=` on the fixed-k filter selectors,
  `select_cached`, the sklearn filter wrappers, and as an alias of knockoff
  `feature_groups`. Explicit dicts map block labels to members; unlisted
  columns stay singletons. Filter `"auto"` groups `{block}__{level}` one-hot
  prefixes only (ordinary underscores are not split). Knockoff `"auto"` remains
  correlation clustering. `k` counts additional blocks;
  `n_columns_selected` reports raw width; `SelectionView.k` stays
  `len(features)`. Gaussian CEFS+ uses joint residual log-det gain; classic
  mRMR/JMI/JMIM aggregate the configured column estimators. Filter `k="auto"`
  scores complete block prefixes for `evaluate`, `elbow`,
  `penalized_objective`, `gaussian_cv`, `xfit_objective`, and auto routing
  (grids stay in additional-block units; selected prefixes expand to raw
  columns). Column-step calibrated rules raise. Binary log-loss CEFS+ uses a
  joint logistic block score on `evaluate`/`elbow`/`penalized_objective`/
  `auto`; Gaussian CV/xfit and calibrated binary stops raise.
  `loss="brier"` delegates to Gaussian blocks. No-block calls and grouped FDR
  validity are unchanged.
- Added additive `include=`, `exclude=`, and `candidates=` keywords on the
  public filter selectors, `select_cached`, the sklearn filter wrappers, and
  `select_fdr`. `include` initializes the actual greedy/conditional state
  before step 1; `exclude` and `candidates` constrain the discovery pool.
  `k` counts additional discoveries. Conditioned features are prepended to
  the output in caller order and are not treated as discoveries. Omitted
  keywords leave existing calls unchanged. Auto-k methods that rebuild an
  unconditioned path (`stability`, `knockoff_path`, `consensus`,
  `gaussian_cv`, `perm_gap`, `xfit_objective`) reject conditioning instead
  of approximating. `k_method="auto"` is checked after routing, and
  `auto_dense_check` is rejected when conditioning is active.
  Knockoff `include`/`exclude`/`candidates` require `include_provenance`;
  only `prespecified` and `sample_split` keep FDR-compatible metadata, while
  `data_derived` is labeled exploratory with `fdr_control="none"`. FDR
  applies to discoveries only; the include set is residualized out of the
  Gaussian-copula knockoff model.
- Added `SelectionView.redundancy_report` and `SelectionView.proxy_clusters`
  on stored proxy correlations: `redundancy_report` lists selected-to-
  unselected-candidate edges, while `proxy_clusters` includes selected-
  selected edges in its selected-anchored connected components, with exact
  per-cluster selection frequencies on `StabilitySelector(store_proxies=True)`
  resamples.
  `store_proxies` remains opt-in and default-false; omitted calls are
  unchanged. Cluster frequency is nullable when no resample payload exists.
- Added regression-filter `within="groups"` and `within="two_way"` panel
  transforms, with weighted entity means and alternating entity/time
  demeaning until convergence before ranks. Supported auto-k scoring fits means
  on training folds only. Unseen entity levels use the training grand mean for
  that effect; unseen time levels add no time effect. A warning reports affected
  rows; routes where none can be seen are rejected.
  Result views expose within/between relevance; raw transform output and
  calls omitting `within` are unchanged. Prebuilt caches, classification,
  nested mode and non-fold auto-k methods reject the new option explicitly.
  Stable anchored means preserve group-constant large-offset data and genuine
  small signals; removing all feature variation raises a clear error.
- Added opt-in knockoff `aggregation="evalues"` for `n_draws > 1` and
  `offset=1`. Draws keep the same knockoff samples as the omitted frequency
  vote; e-values use the common tested universe, zero-pad screened-out
  coordinates, average, and apply e-BH. Validated ungrouped FDR remains
  `approximate_plugin` only for `relevance` and `ridge`, whose sign-flip
  identities are algebraic. `lsm` and `cefsplus` e-value runs are exploratory
  because truncated LARS and adaptive CEFS+ paths are not swap-equivariant.
  Grouped/representative mode, varying pair-screens, and supervised encodings
  also report `fdr_control="none"` with matching aggregation validity fields.
  Rankings and normalized views carry the `evalue` column. Omitted aggregation
  is unchanged.
- Added `benchmarks/bench_knockoff_statistic_bakeoff.py`, a seeded public
  `select_fdr` comparison of `relevance`, `lsm`, `ridge`, and `cefsplus` on
  independent, AR(1), block-correlated, and dense-weak designs. The retained
  full run (commit `ae904b8`, CSV SHA256 `40d4e7944b81b012996f9c9f08327b1c7f2be33a4eee766f9af7a0a482c88acf`)
  recommends keeping `relevance` for the 1.0 owner decision on that grid:
  ridge reduced realized FDP but lost power on correlated designs. The 0.9
  default remains `relevance`. Realized FDP does not upgrade
  `approximate_plugin` and does not prove sign-flip for LSM or CEFS+.

### Fixes and behavior changes

- Fixed purged time splits to apply `max_train_size` after purge, embargo and
  group exclusion; mixed numeric timeline/end dtypes now use the timeline's
  embargo rule, and unsupported time dtypes fail clearly.
- `compare` now accepts purged splitters and `event_end`, keeps each candidate
  on paired folds, validates holdout settings, handles generated split pairs,
  and stratifies its default classification folds. `in_sample_path` refuses
  ambiguous column identity; empty selections and empty result tables have
  consistent contracts.
- `Stabilized` now prevents duplicate source rows from leaking across an
  inner-CV base's folds under replacement resampling, preserves multiplicity
  with weights where supported, seeds each resample independently, and rejects
  stale prebuilt caches. Unsupported block resampling reports the compatible
  alternatives.
- Panel `within` scoring now warns on partially unseen validation levels and
  rejects split strategies where no levels can be seen. Two-way demeaning
  iterates to convergence on unbalanced panels; `between_relevance` is not
  reported as a finite score when too few entities exist.
- Ordinal categories now follow natural numeric order or declared ordered
  categorical order. Integral float and integer scoring values match fitted
  categories across ordinal, frequency and one-hot transforms without
  changing the fitted vocabulary.
- Conditioning now rejects integer positions on named DataFrames with
  actionable guidance and reports raw categorical names when an included
  one-hot level is unusable. Sklearn wrappers reject one-shot conditioning
  iterators at fit time instead of silently changing on refit.
- Grouped knockoff e-values now distinguish representatives from cluster
  members; members have `NaN` e-values and a separate representative value.
  e-BH helpers reject invalid values, and nested validity labels reflect the
  actual selection route.
- Auto-k fixes include block-unit elbow curves after dead columns are
  removed, conditioned-route preflight, valid weighted multi-target routing
  to EBIC, and the `min_feasible_q` boundary. `ModelSelector` handles missing
  callable importance and records its effective splitter.
- Encoders reject duplicate column labels; one-hot with no categorical
  ndarray columns is a no-op. Unsupported multi-target inputs now raise a
  direct error in Boruta, comparison, stability and permutation importance.
- Reproducibility manifests now distinguish input targets, weights, groups,
  time, long option lists, seeds and cache provenance; prebuilt caches no
  longer serialize raw training arrays. Deterministic `Decimal`, UUID,
  complex and fraction labels can be hashed again. Export metadata records
  Python, platform and repository dirtiness without absolute BLAS paths.
  Manifest schema `1` is documented before its first release.
- `evaluate_feature_path` (a 0.9.0 API) gained keyword-only `time=` and
  `event_end=`, forwarded to splitters that declare them, so the purged
  splitters apply their label-horizon purge from there. It also accepts a
  generator of `(train_idx, val_idx)` pairs as `splitter`, which 0.9.0
  rejected with a `TypeError`.
- `SelectionView.proxies_at` (a 0.9.0 API) accepts `include_selected=True`,
  which also lists selected-to-selected edges (the edges `proxy_clusters`
  merges on), as the new `redundancy_report` does. The default output is
  unchanged.
- The "Non-numeric columns found" error lists every accepted `cat_encoding`,
  including `ordinal` and `frequency`.

### Documentation

- Replaced the handwritten `docs/API.md` page with a generated reference page
  for every `sift.__all__` export. MkDocs and mkdocstrings now render the
  numpydoc source of truth, while CI checks the committed page inventory and
  runs `mkdocs build --strict`. Legacy Sphinx-role markers in docstrings were
  normalized to inline code so they render cleanly in the generated pages.
- Replaced five overlapping selector-choice tables with one canonical decision
  tree. The README, manual, tutorial, and algorithm guide now link to that
  page instead of maintaining separate recommendations.
- Added a runtime/scaling guide backed by a reproducible six-method benchmark.
  Its committed CSV and provenance sidecar retain all raw timing samples,
  environment and thread-pool state, effective options, data and selection
  fingerprints, clean-commit Git state (`dirty=false`), and hashes for the
  runner and executed SIFT sources. The [runtime guide](runtime-scaling.md)
  records the current bound commit and CSV checksum.
- Added an executable data-type support matrix over the public selector entry
  points. Cells are live probes of numeric ndarray/DataFrame input, categoricals,
  sparse matrices, datetime/timedelta feature columns, sample weights, groups,
  and time, classified as supported, rejected, conditional, or dependency-gated.
  The probes do not change selector mathematics or public defaults.
- Added a canonical glossary of SIFT-specific terms, linked from MkDocs
  navigation, the root documentation maps, the selector/user/algorithm/advanced
  guides, and the generated API and data-type pages.
- Converted `docs/user-guide.md` from a topic catalog into a task-oriented
  tutorial. The page now walks one selection job from a first pass through
  diagnostics, with stability, knockoff FDR, cache reuse, CatBoost, and
  permutation importance as explicit branches rather than a second choice
  table. MkDocs navigation labels the page as Tutorial.

### Compatibility

- Stability classification now constructs sparse logistic models without the
  deprecated explicit `penalty=` argument on scikit-learn 1.8 and newer,
  while preserving the `penalty="l1"`/`"l2"` path on the supported 1.3-1.7
  releases. This removes a scikit-learn 1.9 `FutureWarning` that failed
  warnings-as-errors runs in `stability_classif`, automatic alpha selection,
  and classification threshold tuning.

## 0.9.0

0.9.0 supersedes the never-published 0.8.0 work — the last published release is v0.7.0, so
everything recorded under `0.8.0` below reaches users for the first time here. 0.9.0 is
strictly additive apart from the breaking changes enumerated in the next section: every other
0.8-style call keeps its selections, return types, output ordering, and defaults.

The work below was staged in the working tree as `0.9.0a1` and `0.9.0b1`; neither was ever
published, and both are folded into this single section.

### Breaking changes and migration

- `stability_regression(..., k=…)` and `stability_classif(..., k=…)` no longer
  pad short selections with never-selected features: `k` caps the count and
  `threshold` gates membership, so wrappers can now return fewer than `k`
  features (including zero). Rank by `selection_frequencies_` yourself if you
  need a fixed-size list.
- `k='auto'` (router) calls now emit a `UserWarning` when they select zero
  features, and `select_cefsplus` warns when `y` contains only 3-20 distinct
  integer-valued levels (labels-shaped targets). Selector classes reject 1-D
  `X` with a `ValueError` instead of an `IndexError`. Binary log-loss CEFS+
  automatic routing rejects non-default `auto_dense_*` options with a
  `ValueError` (there is no log-loss dense-regime diagnostic; the fields were
  previously warned about, and stripping that warning would have made them
  silently ignored). Binary Brier selection delegates to Gaussian CEFS+ and
  retains its dense-check behavior. `StabilitySelector` rejects duplicate
  DataFrame column labels and duplicate, empty, missing, scalar, or unordered
  explicit `feature_names` at fit; pass an ordered iterable such as a list,
  tuple, pandas Index, or one-dimensional NumPy array. Transform validates
  duplicate labels and missing selected DataFrame columns; `tune_threshold`
  uses the same identity helper but requires every fitted feature column. A
  failed fit or refit now leaves the selector unfitted. Column identity is
  exact for tuple/MultiIndex and missing-value labels, and unhashable labels
  are rejected clearly. A
  selector fitted on an unnamed positional ndarray rejects DataFrame input to
  `transform` or `tune_threshold`, because its generated names cannot establish
  column identity. Continue passing positional ndarrays, provide explicit
  `feature_names` when fitting the ndarray, or refit on a DataFrame before
  passing DataFrames to those methods.
  The no-config router
  routes time-context non-CEFS+ Gaussian selectors to
  `gaussian_cv/time_holdout` with `selection_rule="best"` (previously the
  `one_se` request fell back to `best` with a warning), no longer re-warns
  about `auto_dense_*` fields it already consumed, and
  `StabilitySelector.selection_frequencies_` is now float64.
- `StabilitySelector(use_smart_sampler=True)` now honors an explicit
  `feature_names` sequence as an ordered feature subset instead of widening it
  to every numeric DataFrame column, so existing calls can produce different
  selections and output widths. Omit `feature_names` to retain the former
  all-numeric behavior. Configured group/time columns remain sampler metadata
  and are excluded from an explicit subset. Datetime and timedelta feature
  columns are rejected before numeric coercion, while a configured datetime
  `time_col` remains valid metadata. Fold-local `tune_threshold` fits retain
  required sampler metadata while scoring only the fitted feature subset.

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
- Datetime and timedelta feature columns, including native and object-typed
  NumPy arrays and Arrow date, duration, timestamp, and time-of-day dtypes,
  now raise before numeric coercion in classic, cache, Boruta, and
  stability-selection paths. Derive explicit numeric calendar or elapsed-time
  features before selection.
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
- `feature_names_in_` is now sklearn's required one-dimensional NumPy object
  array rather than a list, on every selector class that exposes it. Because it
  is an ndarray, `selector.feature_names_in_ == ["a", "b"]` is an element-wise
  comparison, and asserting on it raises `ValueError: The truth value of an
  array with more than one element is ambiguous`. Compare with
  `list(selector.feature_names_in_) == ["a", "b"]` or
  `np.array_equal(selector.feature_names_in_, ["a", "b"])` instead.

### sklearn contracts

- All eight public selector classes now subclass `SelectorMixin` and expose
  support masks, ordered support indices, selected feature names, and dense
  `inverse_transform`. Sparse matrices are rejected consistently during fit,
  transform, and inverse transform.
- Added `output_order="legacy"|"original"`. The default preserves filter and
  knockoff selection order, Boruta input order, and Stability descending
  selection-frequency order; `"original"` is the additive input-order option.
- Added explicit, version-gated sklearn metadata routing. The dependency floor
  remains `scikit-learn>=1.3,<2`: 1.3 callers pass fit metadata directly, while
  sklearn 1.4+ can route requested metadata through Pipeline and
  `cross_validate(params=...)`. Fixed-k filters reject group/time requests,
  Knockoff exposes only weights, and smart-sampler conflicts fail before fit.
- Scoped private `RidgeCV`, `GridSearchCV`, and threshold-tuning pipelines out
  of an outer estimator's routing context, preserving their historical fit
  semantics on sklearn 1.5 and 1.7. The compatibility audit documents, without
  silently changing selections, that inner auto-k Ridge alpha CV does not yet
  consume group/time context and Stability automatic-alpha validation scoring
  remains unweighted in 0.9.
- Pinned a common green sklearn estimator-check list and selector tags across
  all classes. All audited selectors handle non-finite feature values and
  require `y`; only Knockoff is tagged non-deterministic.
- Gaussian/cache-backed sklearn selector constructors use
  `subsample="auto"`, resolving to 50,000 rows only at fit time. MRMR, JMI,
  JMIM, and CEFS+ wrappers also use `random_state="auto"`, resolving to seed 0;
  `KnockoffSelector.random_state` remains numeric because it seeds each fresh
  draw. These literals preserve explicit cache-override rejection while
  satisfying sklearn's default-constructible estimator parameter contract.
- All sklearn-style selector wrappers now expose `get_feature_names_out()`.
  `KnockoffSelector` is tagged and documented as row-order-sensitive despite a
  fixed seed; zero-weight rows are still removed before knockoff RNG draws.
- Fixed a regression that made every non-Knockoff selector crash on duck-typed
  array input. Dense fit validation no longer calls `np.iscomplexobj` on the
  raw object, so array-likes that refuse `__array_function__` dispatch (such as
  the wrapper behind sklearn's `check_sample_weights_not_an_array` and
  `check_transformer_data_not_an_array`) are materialized through `__array__`
  and validated normally. DataFrame and sparse handling is unchanged, and wide
  frames are still checked column-wise without a second materialization.
- The complex-input error now reads `Complex data not supported by SIFT
  selectors`, matching the wording sklearn's `check_complex_data` requires.
- `feature_names_in_` is now sklearn's required one-dimensional NumPy object
  array on all eight selector classes. Positional fits keep their generated
  `x0...` names in that public attribute, which is unchanged 0.8 behavior; a
  private `_fit_feature_names_generated_` marker (already present on
  `StabilitySelector`) now records named-versus-positional provenance on the
  filter selectors, `KnockoffSelector`, and `BorutaSelector` as well.
  The ndarray-comparison migration note is in *Breaking changes and migration*
  above.
- Filter and Knockoff `transform` now raise sklearn's standard feature-name
  mismatch message for all-string DataFrame columns, naming the unexpected and
  missing labels instead of only reporting that columns differ. Non-string
  column labels keep SIFT's existing strict order/identity message.
- `CEFSPlusBinarySelector` declares the legacy `binary_only=True` tag, so
  sklearn's common checks coerce `y` to two classes rather than tripping the
  selector's own validation. sklearn 1.6 replaced that flat tag with
  `Tags.classifier_tags.multi_class`, which only exists for estimators typed as
  classifiers; the selector remains a transformer and leaves `classifier_tags`
  unset rather than misdeclaring its estimator type to obtain a tag.
- Pinned `check_complex_data`, `check_sample_weights_not_an_array`,
  `check_transformer_data_not_an_array`, and (for the order-strict transforms)
  `check_dataframe_column_names_consistency` alongside the existing green-check
  list, plus a duck-array fit regression test for every selector class.

### Leakage-safe target encoding

- Added `cat_encoding="target_cv"` for regression and binary DataFrame inputs.
  One SIFT encoder serves every fold kind; it requires no `category_encoders`
  extra, preserves one output column per raw feature, normalizes missing values
  to one learned category, and maps unseen inference categories to a zero
  centered effect (the global-mean estimate before centering). The centering
  correction and its metadata repairs are in the bullets below.
- Function filter results conditionally record the fixed-fold encoding kind and
  effective split count. Selector classes and Boruta retain the full-training
  encoder for target-blind `transform`, expose the same information through
  `categorical_encoding_metadata_`, and return the cross-fitted selected
  training columns from `fit_transform` where applicable.
- Added `target_cv_n_splits`, `target_cv_smoothing`, `target_prior`, and
  `warmup_policy` to filter and Boruta entry points. Weighted, grouped, and
  time-aware folds accept `target_cv_smoothing="auto"` alongside an explicit
  numeric value (see the `"auto"` bullet below; they briefly required the
  explicit value). Group folds exclude held-out
  groups; time folds keep ties together, use strictly earlier history, and
  remove earliest no-history rows from selection unless a target-independent
  prior is supplied. Contextual filter calls remain limited to auto-k evaluate
  routes, while fixed-k `groups`/`time` rejection is unchanged.
- Existing defaults and unsafe expert encoders are unchanged. Multiclass is
  still rejected until block-aware expansion exists.
- **`cat_encoding="target_cv"` now emits centered category effects.**
  Out-of-fold training rows emit `fold_encoding - fold_training_prior` and
  inference rows emit `full_fit_encoding - full_training_prior`. An unknown or
  unseen category maps to a zero centered effect, i.e. the global-mean estimate
  before centering, rather than to a prior that identifies its own fold. This is
  a behavior change: encoded values are now effects around zero, not raw
  category means.
- This closes a real leak. A unique-ID column, a group proxy under
  `groups`/GroupKFold, and a timestamp proxy under `time` were each encoded with
  their complement folds' prior, which is anti-correlated with the row's own
  fold. On a 600-row, 8-seed regression fixture the ID entered `select_mrmr`'s
  top three in 8/8 seeds with `corr(enc(id), y) ~ -0.09`; group proxies reached
  `|corr| 0.38` and timestamp proxies `0.97`. After centering all three columns
  are constant zero, carry zero relevance, and are selected in 0/8 seeds.
- **Scope of that guarantee: centering neutralizes only unseen-in-fold
  emissions.** It removes the fold marker; it is not a defence against high
  cardinality as such. A level that appears two or more times in a fold's
  training rows still transmits those sibling rows' targets — ordinary
  target-encoding behavior — so a *near*-unique identifier can still be
  selected. Measured on a 300-identifier / 2-rows-each fixture whose rows share
  a latent target, `corr(enc(id), y) ~ 0.88` and `select_mrmr(k=2)` picks `id`.
  That is genuine cross-row information, not leakage, so the numerics are
  deliberately unchanged. If it must not reach selection, drop ID-like columns
  or pass `groups=` so all of an identifier's rows land in one fold — with
  `groups=` the same column encodes to exactly zero. The boundary is pinned by
  `test_near_unique_ids_with_a_shared_target_stay_selectable_by_design`,
  `test_near_unique_ids_without_a_shared_target_are_not_selected`, and
  `test_grouping_an_identifiers_rows_into_one_fold_removes_the_residual`.
- All `target_cv` routing now goes through SIFT's own encoder so one engine
  carries the guarantee; sklearn's `TargetEncoder` does not expose the per-fold
  priors the contract needs. Unweighted fixed-k folds keep the previous split
  construction (`KFold`/`StratifiedKFold(shuffle=True, random_state=...)`) and
  reproduce sklearn's `smooth="auto"` empirical-Bayes shrinkage exactly, now
  generalized to weighted rows. Smoothing options, group exclusion, strict-history
  time folds, tied timestamps, effective weights, the
  one-raw-column/one-encoded-column contract, and missing-as-its-own-category are
  unchanged.
- **`target_cv_smoothing="auto"` now works on the weighted, grouped, and
  time-aware paths too**, which is what the weighted generalization above always
  described. Those calls previously raised `ValueError: target_cv_smoothing must
  be an explicit non-negative float ...`, so
  `select_mrmr(..., cat_encoding="target_cv", sample_weight=...)` and
  `select_cefsplus_binary(..., cat_encoding="target_cv",
  class_weight="balanced")` failed on the default smoothing. The weighted prior
  is the integer formula with every count replaced by weighted row mass
  (`prior = sum(w*y)/sum(w)`, `s2y = sum(w*(y-prior)^2)/sum(w)`, `w_i` the
  category's weighted mass, `ssd_i` its weighted sum of squared deviations), so
  weight `m` and `m` duplicated rows give identical encodings — verified exactly
  (0.0 max difference), and the full-fit map still matches sklearn's `"auto"` to
  2.8e-17. No case was found in which `"auto"` is undefined but an explicit
  float is not: `ensure_weights` already rejects negative, non-finite, and
  all-zero weights; target-CV also rejects individually finite frequency
  weights whose aggregate mass overflows float64. A fitting slice with no
  positive weight mass — the one genuinely undefined case, where neither the
  weighted prior nor the weighted target variance exists — still raises for
  both.
- **`target_cv_smoothing="auto"` is now invariant to an additive shift of the
  target.** The empirical-Bayes shrinkage used to build its per-category and
  global variances from raw weighted moments, reconstructing each
  within-category scatter as `sum(w*y^2) - w_i*mean_i^2`. On an offset target
  the two terms agree to about sixteen digits while their difference is the
  small quantity being sought, so `lambda_i` — and therefore every emitted
  effect — was dominated by rounding error. Measured on a 300-row, 6-level
  regression fixture, `fit_transform(X, y)` and `fit_transform(X, y + 1e8)`
  differed by up to 0.19 in the centered out-of-fold encoding (0.31 under
  time-aware folds) and by 0.05 in the target-blind `transform`; on a near-tie
  design `select_mrmr(k=1, cat_encoding="target_cv")` flipped from `["cat"]` to
  `["numeric"]`, and `select_cefsplus(k=1)` flipped the other way on a sibling
  design. Every moment is now accumulated on `y - prior` with a two-pass
  weighted sum of squared deviations, and the shrinkage is applied in centered
  space, so the out-of-fold, full-fit, weighted, grouped, and time-aware paths
  all encode `y + c` exactly as they encode `y` — agreement is ~4e-16 against
  the target the shifted run actually sees, and ~2e-9 against raw `y`, limited
  beyond that only by float64's own ~1.5e-8 resolution near 1e8 — the offset
  target cannot represent `y` any more exactly than that. The shrinkage remains
  scale *equivariant*: scaling `y` by `s` leaves `lambda_i` unchanged and scales
  the effects by `s`. Ordinary-scale encodings are unchanged to the last ulp,
  sklearn `smooth="auto"` parity is preserved, and binary `target_cv` — a 0/1
  target has no offset to cancel — is pinned unchanged. An explicit
  `target_cv_smoothing` float was never badly affected (4.4e-8 at the same
  offset, i.e. float noise), which is what localized the defect to the
  empirical-Bayes path; it is centered too and is now exact as well.
- Target-CV now fails clearly when finite individual `sample_weight` values
  sum to a non-finite frequency mass. Previously, values such as `1e308` could
  overflow only when aggregated, yield a NaN target prior, and silently turn
  every encoded category effect into zero. The check occurs inside the shared
  map fitter, so direct, out-of-fold, grouped, temporal, and public selector
  routes share the same contract.
- Earliest temporal rows with an explicit target-independent `target_prior` now
  emit a centered neutral effect (zero) instead of the raw prior value; without
  one they still retain zero effective selection weight.
- Encoding metadata is producer-owned. Results carry only the nested
  `encoding_cv={"kind": ..., "n_splits": ...}` shape read from the fitted
  encoder; the stray top-level `kind`/`n_splits` keys that classic and Gaussian
  function results emitted are gone. `BinaryPathRun` now carries the encoder's
  actual metadata, so the binary time route reports the four active folds
  instead of reconstructing five from zero-weight rows.
- Encoding metadata is attached only when encoding actually ran. A requested but
  absent `cat_features` column is ignored silently, matching the legacy
  `loo`/`loo_logit` convention, instead of raising `KeyError: 'encoding_cv'`
  from `select_cefsplus_binary(..., return_result=True)`.
- `allow_full_data_target_encoding=True` combined with
  `cat_encoding="target_cv"` now raises a clear `ValueError` at the function,
  selector-class, binary, and Boruta entry points instead of being silently
  ignored.
- `KnockoffSelector` rejects `cat_encoding="target_cv"`: target-derived
  preprocessing undermines Model-X exchangeability. The 0.8 supervised encodings
  (`"loo"`, `"target"`, `"james_stein"`, `"loo_logit"`) remain available there
  for compatibility, but now emit a `UserWarning` and report
  `fdr_control="none"` with a `validity_note` in the result metadata. Function
  parity is deliberately not the fix: `select_fdr` gains no `cat_encoding`
  parameter.

### Result views

- Added `SelectionView` and `sift.as_result(...)` without changing legacy return
  types, constructors, or defaults. The first A1 slice adapts
  `FilterSelectionResult` and `KnockoffSelectionResult`, and adds matching
  `.result_view()` methods. The same five accessors expose selected names,
  positions, count, the available raw table, and copied metadata.
- The A2a slice adds non-replacing adapters and `.result_view()` methods for
  `BorutaResult` and `FeaturePathEvaluationResult`. Boruta retains a complete
  positional table and maps mean importance to `gain`; feature-path views leave
  discarded raw positions unknown unless explicit input names resolve uniquely,
  and expose the tested lower-is-better score as a normalized curve.
- The A2b slice adds the same non-replacing adapter and `.result_view()` method
  for `CatBoostSelectionResult`. It preserves the target-k versus returned-count
  distinction, normalizes the direction-aware score curve, derives standard
  errors only when raw split scores provide a denominator, and keeps raw
  identity partial unless the caller supplies it explicitly.
- The A2c slice adds a dynamic `StabilitySelector.result_view_` and
  `sift.as_result(fitted_selector)` adapter. Its complete table uses fitted
  candidate order, maps mean absolute coefficient to `gain`, preserves capped
  selection membership from the legacy integer indices, and exposes a frozen
  column-subset transform without retaining training rows or bootstrap
  coefficient matrices. New fits record DataFrame-versus-positional provenance;
  no constructor parameter, legacy fitted attribute, return type, or default
  changes.
- The A2d slice completes the seven-family core adapter coverage. The default
  `permutation_importance` return remains its exact four-column DataFrame;
  `return_result=True` opts into module-scoped `ImportanceResult`, whose
  defensive-copy repeat matrix is aligned to original feature positions.
  Its complete view preserves duplicate labels positionally and marks every
  evaluated feature as a `ranking_only` report rather than inventing a subset
  threshold.
- Views serialize to JSON-safe schema version `"1"`, preserve positional
  identity in `selected_index`, report incomplete tables explicitly, and use
  `input_kind="unknown"` when a legacy result cannot prove whether its source
  was named or positional. Partial views now reject table-only plots instead of
  presenting incomplete data as complete.
- Added explicit bounded proxy storage to `select_cached` and Gaussian filter
  result paths. `return_result=True, store_proxies=True` retains only the
  post-screening candidate-by-selected copula correlations as `float32`, with
  a 64 MiB cap and no retained `X` or cache. Name lookup rejects ambiguous
  duplicate labels and `proxies_at(...)` provides positional access. Existing
  calls, return types, and serialized default results are unchanged.
- A fitted `StabilitySelector` view now applies the selector's own
  `output_order`. `view.features`, `view.indices`, the raw table's `path_rank`,
  and the frozen `view.transform` follow the same order as
  `get_feature_names_out()`, `get_support(indices=True)`, and `transform`; the
  frozen transformer copies `output_order` instead of silently reverting to the
  `"legacy"` default, and `metadata["output_order"]` records which order applied.
- Automatic-k filter producers keep the complete feature ranking they already
  computed. `select_mrmr`/`select_jmi`/`select_jmim` with `k="auto"`, every
  Gaussian auto-k route, and binary CEFS+ auto-k now populate `ranking_`, so an
  auto-k `SelectionView` has one row per raw column and
  `metadata["table_complete"] is True` instead of only the selected rows.
- Automatic-k routes publish a normalized curve with exactly the columns `k`,
  `criterion`, `criterion_se`, and `selected`, built producer-side from each
  route's diagnostics and stored in `diagnostics_["auto_k_curve"]`.
  `metadata["criterion"]` names the source diagnostic column,
  `metadata["criterion_direction"]` is `"higher_is_better"` or
  `"lower_is_better"`, and `metadata["curve_route"]` records the routed method.
  `knockoff_path` and `consensus` report `curve_available=False` with an
  explicit `metadata["curve_unavailable_reason"]`, because their diagnostics are
  per-feature draws and per-method votes rather than a k-indexed criterion path.
  Adapters consume only the normalized payload; `view.py` no longer guesses
  method-specific diagnostic columns.
- `select_fdr` metadata gains `n_features_input` (the raw input width) plus
  `dropped_feature_positions`/`dropped_feature_reasons`, all distinct from the
  existing post-screening `n_features`. Knockoff views therefore build
  `support_` and a complete raw table without requiring `input_features`, and
  every dropped column gets an explicit `reason_dropped` row (`"constant"` or
  `"zero_weight_variance"`). Legacy results without the new keys keep the
  previous partial behavior.
- `view.to_dict()` no longer merges mapping keys or emits `repr()` fallbacks.
  Ordinary string-key mappings — including the payload root and metadata — stay
  plain JSON objects; only a mapping containing a non-string key uses a tagged,
  ordered `{"__sift_mapping__": "typed_key_entries", "entries": [...]}` envelope
  with typed key tokens, so `1` and `"1"` both survive a JSON round trip.
  `pd.NA`/`pd.NaT` become `null`, datetimes become ISO strings, dataclasses use
  `dataclasses.asdict`, and unsupported objects raise a clear `TypeError`.
  `schema_version` stays `"1"`; mixed-key envelopes are part of schema 1.
- Legacy `FilterSelectionResult` and `KnockoffSelectionResult` fields, defaults,
  and pickle formats are unchanged, and fixed-k `ranking_` semantics are
  unchanged.

### Conventions

- DataFrame callers may use `groups="column"` and `time="column"` wherever
  those row arrays are accepted. SIFT extracts the metadata positionally and
  removes it from the feature namespace; direct arrays remain positional, and
  fixed-k filters continue to reject row context.
- CatBoost selection adds trailing `groups`, `time`, and `sample_weight`
  arrays while retaining `group_col` and `sample_weight_col` aliases. Alias
  conflicts raise, supplied time values are validated and stably order aligned
  rows before the configured splitter, and translated-parameter collisions
  emit a `UserWarning` while preserving the 0.9 `catboost_params`-wins rule.
- `StabilitySelector(penalty=...)` is an additive alias for `alpha`; unequal
  simultaneous values raise. Threshold tuning, explicit feature-path
  evaluation, and auto-k evaluation accept estimator-style sklearn scorer
  objects. Path and auto-k routes negate signed scorer output into their
  historical lower-is-better curves.
- `select_cached(..., return_result=True)` returns a complete `SelectionView`
  with cache provenance, selected positions, relevance, and objective-path
  diagnostics. Its four legacy list/tuple forms and default remain unchanged.
- The existing `None` defaults on Stability, permutation importance, and
  CatBoost now emit a caller-facing `FutureWarning` when used; they remain
  nondeterministic in 0.9 and will resolve to seed 0 in 1.0. Literal-42
  defaults and all existing `n_jobs` defaults remain unchanged in 0.9.
- `permutation_importance` accepts sklearn scorer objects and `ScoringSpec`, and
  exposes `higher_is_better` only for legacy loss callbacks, avoiding a second
  direction flip for signed scorers.
- Long-running fixed and Auto-K filter paths, `select_cached`, filter selector
  classes, stability bootstraps, Boruta iterations, and CatBoost splits now
  accept an additive `callback(step, total, info)` hook. Calls are one-based,
  happen after completed units, receive fresh metadata dictionaries, and
  propagate callback exceptions. `callback=None` retains the original kernels,
  selections, return types, defaults, and logging behavior. Fold-local fits
  inside `StabilitySelector.tune_threshold()` remain silent instead of
  restarting the public bootstrap callback sequence for every fold.
- Progress output now uses the `sift` package logger at INFO instead of direct
  `print` calls. Existing `verbose` defaults and silence behavior are unchanged;
  `sift.set_verbosity("debug"|"info"|None)` is an additive global control. An
  application handler whose level rejects INFO no longer suppresses the
  default fallback progress stream.
  Every package warning now declares its category and a caller-facing stack
  level without changing warning counts or categories, and CEFS+ path-depth
  saturation reports the effective depth actually used.

### Auto-k ergonomics

- Added `AutoKConfig.default()`, `.predictive(...)`, `.discovery(...)`, and
  `.downstream(...)` presets. Predictive fold counts map to `xfit_folds`, not
  the distinct evaluate/nested `n_splits` field.
- Added `AutoKConfig.from_groups(...)` and seven immutable module-scoped option
  group types. They flatten immediately into the unchanged 49 fields; direct
  flat construction, defaults, equality, representation, replacement, and
  pickle contracts are unchanged. Unknown, wrong-type, and conflicting group
  inputs fail before construction.
- Completed method-level unused-field warnings, including conditional EBIC,
  permutation-envelope, plateau, and stability-threshold options. Warnings
  point to the caller and are suppressed for internal router/consensus copies.
- Added the 16-name `sift.experimental` namespace. Access through it emits a
  `FutureWarning`; all 58 ordered top-level exports remain available and
  warning-free throughout 0.9.
- Auto-k saturation warnings now distinguish a configured `max_k` cap,
  exhaustion of the candidate path, and a fold/statistical limit that ends an
  evaluation curve before the path.

### Correctness

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
- CatBoost result objects persist `selection_patience`, so
  `features_within_tolerance()` uses the same consecutive-miss rule as fit-time
  selection.
- Weighted binned JMI/JMIM now use weighted quantile edges as well as weighted
  entropy counts. Zero-weight rows do not affect binning, and multiplying all
  weights by a positive constant does not change the estimand. Integer
  frequency ratios are reduced by any common global factor rather than
  treating that factor as extra replicated sample size.
- Grouped time-block stability bootstrap now honors `sample_frac` with a rounded
  panel-wide draw budget allocated across unequal groups. Moving, circular, and
  stationary windows draw with replacement and preserve the full-panel budget
  at `sample_frac=1.0`.
- Finite weighted knockoff-variance reductions use `np.dot` instead of NumPy's
  matmul ufunc path, avoiding false divide/overflow warnings observed with
  NumPy 2.2 while preserving selections and statistics.
- `StabilitySelector.get_coef_stability` no longer emits NumPy's 0/0
  `RuntimeWarning` for a feature no bootstrap ever selected. The coefficient of
  variation `coef_std / |coef_mean|` was computed inside `np.where`, which
  evaluates both branches, so a feature with a zero mean and a zero standard
  deviation divided 0 by 0 before the guard could pick the `inf` fallback. The
  ratio is now computed under `np.errstate(divide="ignore", invalid="ignore")`
  and the `np.where` fallback still yields `inf`. This surfaced as a `DOCS.MD`
  example failure on the Python 3.11 CI job (NumPy 2.4.6), where the lasso path
  left a feature entirely unselected.
- `sift.as_result` on a `KnockoffSelectionResult` with dropped inputs no longer
  triggers pandas' all-NA concat `FutureWarning`. The dropped-column rows were
  concatenated as an object-typed all-NA frame onto the typed table, which
  pandas 2.x flags and warnings-as-errors turns into a failure. The new
  `_append_rows_like` helper in `sift/selection/view.py` builds each appended row
  column by column in the table's own dtype. One visible consequence: when
  dropped rows are appended, a NumPy `int64` or `bool` column that the new rows
  do not carry becomes the nullable `Int64` or `boolean` dtype, because those
  NumPy dtypes have no missing value. Float columns keep their dtype and take
  `NaN`; object and extension-dtype columns are unchanged.
- Removed an unused `gaussian_mi_from_corr` import from
  `sift/selection/auto_k_xfit.py`.

### CI and packaging

- Added `[tool.pytest.ini_options]`: `testpaths = ["tests"]` and the registered
  markers `slow`, `catboost`, and `categorical`. The optional-dependency markers
  sit beside the existing `pytest.importorskip` gates rather than replacing them,
  so the suite still skips cleanly when `catboost` or `category_encoders` is
  absent; `slow` covers the `test_knockoff_fdr_control.py` seed loops, the Auto-K
  null-calibration simulation, the 12k-row D10 design, and the 25k-row knockoff
  sampler draw, and `-m "not slow"` removes about 60 seconds from a local run.
- Warnings are now errors. The audited allowlist holds exactly one entry: loky's
  `DeprecationWarning` about `fork()` in a multi-threaded process, which joblib
  emits from `loky/backend/fork_exec.py` and which this project cannot address.
  It is genuinely intermittent — it depends on how many threads exist at fork
  time, and it appeared in one full run and not the next on the same machine.
  Every other warning is handled where it occurs: warnings a test intends are
  asserted with `pytest.warns`, warnings a single test incidentally triggers get
  a local `@pytest.mark.filterwarnings`, and fixtures that set an `AutoKConfig`
  field the chosen `k_method` does not consume simply stopped setting it. No
  category is blanket-ignored.
- Fixed three `pytest.warns` assertions in `tests/test_knockoff_filter.py` that
  only ever passed by accident. `select_fdr` emits two legitimate advisories on
  those near-collinear designs; pytest 7.4.4 silently discarded the one that did
  not match, while pytest 8+ re-emits it. The tests now record all `UserWarning`s
  and assert the intended message, which is stable across pytest versions and
  across the supported NumPy/SciPy range.
- `tests/test_stability_selection.py` no longer hard-imports `matplotlib`, which
  is not a declared runtime or test dependency; the plotting test now skips.
  It would have failed the standard CI job as written.
- `.github/workflows/test.yml`: `cache: pip` on every `setup-python` step,
  `timeout-minutes` on every job, and a top-level `concurrency` group that
  cancels superseded pull-request runs while letting branch and scheduled runs
  finish. The scheduled `benchmark-smoke` job now also regenerates the Auto-K
  G1-G6 gate table from the committed raw CSVs, verifies it against the
  committed artifact with `summarize_auto_k_gates.py --verify-against`, and
  uploads it as `sift-auto-k-gate-table`; it checks out with `fetch-depth: 0`
  because the summarizer verifies its provenance sidecar by hashing recorded
  sources at the commit the sidecar names, which a shallow clone cannot
  resolve. The comparison is **not** a byte-for-byte `cmp`: gate floats are
  rendered with 12 significant digits and compared with `rtol=1e-9`, which
  absorbs last-ulp summation differences between BLAS builds and operating
  systems (a raw `repr` differed in the 17th digit between macOS/arm64 and
  Linux CI). Every non-float cell must still match exactly, and the summarizer's
  own fixture test still pins exact output bytes.
- Added a `min-pins` job that installs every direct runtime floor exactly
  (numpy 1.24, pandas 2.0, scikit-learn 1.3, scipy 1.10, numba 0.59, joblib 1.3,
  threadpoolctl 3.1) and then `pip install -e . --no-deps`. **These floors had
  never been executed anywhere.** They were pre-validated locally on Python 3.11
  and are green: 1,566 passed, 30 skipped, under the new warning policy. The
  floors are mutually consistent and resolve to numpy 1.24.4 / pandas 2.0.3 /
  scikit-learn 1.3.2 / scipy 1.10.1 / numba 0.59.1 / joblib 1.3.2 /
  threadpoolctl 3.1.0, so **no floor in `pyproject.toml` needs to be raised**.
- A Python 3.13 job is deferred rather than added, because a job that cannot pass
  is worse than none. The interpreter is not the blocker: numba ships cp313
  wheels from 0.61.0 and a local 3.13.15 run with numba 0.67 reached 1,565 passed
  / 3 failed. The blockers are dependency versions the library does not yet
  support, and **they are not specific to 3.13 — they break the existing 3.11 and
  3.12 matrix jobs identically**, because `scikit-learn>=1.3,<2` and
  `numpy>=1.24,<3` resolve straight to them. When this was written the band
  stopped at scikit-learn `<1.8` and numpy `<2.5`, with 13 open failures.
  Those are now closed — see *Latest-dependency compatibility* below,
  which records the current verified band. `docs/development.md` carries the
  band and the ready-to-enable job definition.
- The 0.9 compatibility matrix now covers every public export behaviorally and
  expands the high-risk cross-products across fixed/Auto-K filters, cache tuple
  shapes and defaults, group/time contexts, categoricals, smart sampling,
  `select_fdr`, CatBoost, sklearn-style wrappers, stability, Boruta, knockoffs,
  and permutation importance. Internal deprecation helpers have exact
  warn-and-forward tests. The deterministic Auto-K gate summarizer now has a
  dedicated D9 fixed-path timing runner with checksum-bound environment/source
  provenance. The summarizer now requires the sidecar and verifies its full-run
  mode, clean state, artifact checksum, seed set, and source hashes against the
  recorded Git commit rather than the later working tree. The
  clean `88a8705` run is committed as
  `auto_k_v2_d9_fixed_k_path_2026-08-31.csv` with
  `auto_k_v2_d9_fixed_k_path_2026-08-31.provenance.json`, and the explicit
  mean-oracle recomputation is
  `auto_k_v2_gates_mean_oracle_2026-08-31.csv`. The mixed-convention legacy gate
  CSV remains intentionally unchanged; the dated G5 ratio is labeled cross-run
  evidence rather than a reconstruction of the missing July denominator.
- Distribution metadata now uses the SPDX `MIT` license expression and ships
  the `py.typed` marker declared by PEP 561. Release CI builds and metadata-checks
  source and wheel distributions, clean-installs the exact wheel, verifies its
  license and typed-package metadata, and rejects leaked repository-only packages.
  The exact distributions are attached to the GitHub Release but are not published
  to PyPI.

### Latest-dependency compatibility

The whole declared band is now exercised, not just its floors. The newest
resolution `pyproject.toml` allows — numpy 2.5.2, scikit-learn 1.9.0,
pandas 3.0.5, scipy 1.18.1, numba 0.67.0 on Python 3.12 — is green at
1,680 passed / 30 skipped under the warnings-as-errors policy. No default,
selection behavior, return type, or public API changed, and no version ceiling
was added to `pyproject.toml`.

- **The nine scikit-learn 1.9 `target_cv` failures needed no new fix.** Their
  cause was not numeric drift and not a renamed API: 1.9 deprecates
  `TargetEncoder(shuffle=..., random_state=...)` in favour of passing a CV
  generator as `cv`, and the resulting `FutureWarning` became an error under the
  new policy. The Stage 1 target-encoding rewrite had already removed that call
  site — every fold kind now runs through SIFT's own engine, which constructs
  `KFold`/`StratifiedKFold(shuffle=True, random_state=...)` itself — so nothing
  in `sift/` still constructs a scikit-learn `TargetEncoder`. Verified directly:
  sklearn 1.5.1 and 1.9.0 produce bit-identical `TargetEncoder` output on the
  same fixture, and all `tests/contracts/test_target_cv_encoding.py` cases pass
  on 1.9.0 unchanged. The rewrite is also forward-compatible with 1.11, where
  those two parameters are removed outright.
- **Duplicate DataFrame column labels are now a scikit-learn limitation, not a
  SIFT one.** From 1.9 its dataframe validation runs through narwhals, which
  raises `DuplicateError` for repeated column names in `fit` *and* `predict`, so
  no estimator can be handed such a frame. SIFT still passes `X` through to
  `model.predict` untouched and still keeps duplicate labels distinct by
  position; the regression test now proves that with a positional stub predictor
  instead of a `LinearRegression`, which is the only part of it that scikit-learn
  no longer permits.
- **The pinned knockoff draw is compared to float32 tolerance.**
  `mean_op`/`noise_chol` come out of LAPACK (`eigh`, `cho_factor`) and are then
  applied as float32 BLAS GEMMs, neither of which is bit-stable across
  NumPy/SciPy builds; numpy 2.5.2 + scipy 1.18.1 reproduces the pinned block to a
  max relative deviation of 6.4e-8, under one float32 ulp. This is not library
  nondeterminism: the same-seed, same-interpreter assertion in that test is
  still an exact `assert_array_equal` and still passes. Only the cross-version
  golden moved to `assert_allclose(rtol=1e-6)`, about 8 ulp.
- **The temporal-label hash test constructs NaT with an explicit unit.**
  NumPy 2.5 deprecates the generic (bare) `timedelta64` unit, which the policy
  turns into an error. The library never constructs one — it only reads
  `.dtype`/`str()` off labels a caller supplies — so the change is confined to
  the test fixture, and it still distinguishes `None`, datetime64 NaT, and
  timedelta64 NaT.
- **Fixed a merge-latent failure that was not dependency-related at all.**
  `test_routes_without_a_k_curve_say_why[consensus]` fails identically on numpy
  1.26/sklearn 1.5: the warnings-as-errors policy and the result-view test
  arrived on separate branches, and the 12-feature fixture makes the four
  consensus submethods disagree by 3x, tripping auto-k's ill-determined-k
  advisory. The advisory is correct behavior and is asserted directly in
  `tests/test_auto_k_v2.py`, so the test carries a local
  `@pytest.mark.filterwarnings` naming that exact message, per the audited
  policy.

### Documentation

- Documentation now records cache/X compatibility and rejected cache overrides,
  fixed-k group/time rejection, the `cat_encoding="none"` default, and the
  stochastic row-order sensitivity of `KnockoffSelector`. Knockoff statistic
  power comparisons are intentionally left data-dependent pending a committed
  quality bakeoff.
- Every public export now carries a substantive numpydoc docstring, replacing
  the one-line stubs that made `help(select_mrmr)` useless. Two tests pin the
  surface for every name in `sift.__all__` except `__version__`, so a new export
  cannot ship undocumented. `tests/test_docstring_coverage.py` requires a
  non-empty summary line, at least 8 non-empty docstring lines, every signature
  parameter — `*args` and `**kwargs` included — named as a `name : type` entry
  under `Parameters` for functions or under `Parameters`/`Attributes` for classes
  (read off `__init__`), a `Returns` or `Yields` section for functions, and an
  `Examples` section with at least one runnable `>>>` statement for every
  export — except the four optional-dependency exports pinned by name
  (`select_boruta_shap`, `catboost_select`, `catboost_regression`,
  `catboost_classif`), whose examples are literal blocks that must name the
  dependency. `tests/test_docstring_examples.py` parses each docstring (and the
  `Examples` sections of exported classes' public methods) with `doctest` and
  executes the `>>>` statements under warnings-as-errors, running documented
  tracebacks inside `pytest.raises`; it does not compare printed output,
  because NumPy 2 scalar reprs differ across the CI matrix, and it leaves unrun
  only examples marked `# doctest: +SKIP`, those that need CatBoost, and the
  pinned literal blocks. Neither test compares
  documented defaults or accepted values against the signature — that stays a
  review responsibility.
- Every fenced `python` code block in the manual now executes in CI. The README
  blocks already ran; `DOCS.MD`, `docs/API.md`, `docs/user-guide.md`,
  `docs/ADVANCED.md`, `docs/troubleshooting.md`, and `docs/results.md` join them,
  for 118 blocks in total: 108 execute in the base environment and 10 are gated
  on an optional dependency by a `requires=` marker (`catboost`, `matplotlib`).
  Blocks are standalone by default — a fresh namespace per block, building its
  own imports and data. Nine blocks carry a `continues` marker and inherit the
  previous block's namespace; every one of them is an honest inspect-after-fit
  step, where the narrative fits a selector in one block and reads the fitted
  object in the next (three in `DOCS.MD`, three in `docs/API.md`, one each in
  `docs/user-guide.md`, `docs/ADVANCED.md`, and `docs/results.md`). No block is
  skipped for a reason the runner cannot see: a bare `skip` directive must state
  a reason or the suite fails, and the manual set currently uses none.
- The CatBoost row-context example in `docs/user-guide.md` now pairs `groups`
  with `GroupKFold`. It previously passed `groups` to a `TimeSeriesSplit`, which
  ignores them and makes scikit-learn warn — a warnings-as-errors failure on the
  CatBoost CI job. The surrounding prose now says which splitter goes with which
  row context: pass `time=` alone when chronological validation is what you
  want, and the default splitter stays random.
- README and `TODO.MD` were refreshed for the 0.9 surface: `SelectionView` and
  `sift.as_result`, `cat_encoding="target_cv"`, `output_order`, sklearn metadata
  routing, the Auto-K presets and option groups, and `sift.experimental`. The
  stale version, test-count, target-encoding, and release text is gone.
- **The deprecation ledger's two alias removals are struck.** The CatBoost
  `group_col`/`sample_weight_col` aliases and stability's `alpha` are permanent:
  no `FutureWarning` is wired for them and none is planned, so nothing in the
  ledger claims a deprecation that the code does not emit. Any future removal
  needs a full warning cycle first. The two warnings SIFT does emit under this
  policy are the `random_state=None` default-use `FutureWarning`
  (`StabilitySelector.fit`, `permutation_importance`, `catboost_select`, and
  their wrappers) and the `sift.experimental` namespace `FutureWarning`; both
  have production tests.
- The generated API reference (mkdocs/mkdocstrings), the retirement of
  `docs/API.md`, the single "which selector" decision tree, the measured
  runtime/scaling table with its committed script, the data-type support matrix,
  and the glossary are re-scoped to **0.9.1**. They are documentation debt, not
  release blockers.

### Deprecation ledger (flips in 1.0)

Reproduced verbatim from §4 of `docs/specs/0.9-product-layer.md`. The table
started as a 0.9 proposal; its 1.0 column now records the narrower contract
approved on 2026-09-21. The changes are announced in 0.10.x and take effect only
in 1.0. The 0.9 column was edited on 2026-09-21 as well (reworded seed and
return rows, a new seed-42 row); its `sift.__all__` cell, which then read
"66 exports", was corrected back to the 58 that v0.9.0 shipped.

| item | 0.9 state | 1.0 state |
| --- | --- | --- |
| `random_state=None` defaults | Stability, permutation importance and CatBoost families warn on omitted/default `None` use | default to `0`; retain explicit `None` where supported and remove the obsolete future-default warning |
| existing seed-42 and cache-aware defaults | `AutoKConfig`, path evaluation and smart sampling use `42`; filter functions use omission sentinels and wrappers use `"auto"` | unchanged |
| `verbose` default | `True`, logging-backed; logging formatting/routing may differ, but selection/returns/default progress behavior does not | `False` |
| `n_jobs=-1` defaults (stability, permutation, CatBoost) | unchanged, documented | `1` |
| `transform` output order | `"legacy"` default: filter path/selection order, Boruta original order, Stability descending selection frequency with stable original-index ties; `"original"` opt-in | `"original"` default |
| `sift.__all__` | 58 exports (0.10.0 added 8, for 66) | retain all 66 exports and the existing `sift.experimental` access path |
| list/tuple/result returns | `select_cached` keeps list/tuple forms and offers `return_result=True`; `as_result` is additive | unchanged; result objects remain opt-in |
| CatBoost `group_col`/`sample_weight_col` | aliases beside arrays | unchanged — permanent alias (removal struck 2026-09-02; any future removal needs a full warning cycle first) |
| stability `alpha` | joined by `penalty=` alias | unchanged — permanent alias (removal struck 2026-09-02; any future removal needs a full warning cycle first) |
| overlapping `AutoKConfig` fields | audited + documented in 0.9; no deprecation | unchanged |
| `AutoKConfig` option groups | flat dataclass fields are canonical; group objects are builders/read-only views and are not simultaneous constructor fields | unchanged |
| `StabilitySelector.selected_features_` index/name asymmetry | unchanged, documented; `view.features` is the unified accessor | unchanged unless separately decided — **not** auto-flipped |
| sklearn floor | 1.3 | retain 1.3 |

## 0.8.0 (never published; folded into 0.9.0)

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
  coefficient difference). Ridge is antisymmetric under original/knockoff
  swaps. Correction from the 0.9 review: truncated/tied LSM is not generally
  swap-antisymmetric; its empirical quality is not a sign-flip proof. The
  legacy calls keep the approximate plug-in validity label. Power
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
- The built distribution is named `sift-feature-selection` while the import
  remains `sift`, avoiding a distribution-name clash with the occupied `Sift`
  project. Wheels exclude benchmark packages, and release automation verifies
  the exact wheel before attaching the source and wheel distributions to the
  GitHub Release without publishing them to PyPI. Critical Ruff checks and a
  scheduled quick benchmark promotion gate run in CI.

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
