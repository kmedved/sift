# Auto-K v2 — Smart automatic `k` selection for CEFS+

Status: proposed (idea slate for implementation)
Companions: [fdr-knockoffs.md](fdr-knockoffs.md), [knockoffs-followups.md](knockoffs-followups.md)

This document specifies a slate of new automatic-`k` methods for the CEFS+
(and general Gaussian/cache-backed) selection path, plus the evaluation
harness that decides which of them survive. Each idea section is
implementation-ready: the math, the exact integration points, config surface,
tests, and numeric acceptance gates. Build **Phase 0 (the harness) first** —
every idea is judged by the same yardstick, and half the value of this effort
is finally having that yardstick.

Primary use case: `select_cefsplus(X, y, k="auto")` on tabular regression
data, n from ~1k to ~100k, p from ~50 to ~5000, often with `time` and/or
`groups` structure and sometimes `sample_weight`. The selected features
usually feed a GBM downstream.

---

## A. Why the current auto-k methods underperform (read before implementing)

### A.1 Inventory of what exists

| Method | Mechanism | Files |
|---|---|---|
| `k_method="evaluate"` | Build one supervised path, score prefixes with Ridge/LogisticRegression on a time holdout or GroupKFold; pick via `best`/`one_se`/`plateau`/`tolerance` | [auto_k.py](../../sift/selection/auto_k.py) `select_k_auto`, [auto_k_core.py](../../sift/selection/auto_k_core.py) `evaluate_numeric_prefixes` |
| `auto_k_mode="nested"` | Same, but a train-only selector path per split (selector classes only, `evaluate` only) | [auto_k_nested.py](../../sift/selection/auto_k_nested.py) |
| `k_method="elbow"` | Relative-gain threshold with patience on the in-sample objective path | `select_k_elbow` in [auto_k.py](../../sift/selection/auto_k.py) |
| `k_method="penalized_objective"` | maximize `n_eff·obj(k) − penalty·k`, penalty ∈ {BIC, AIC, HQC, MDL, custom} | `select_k_penalized_objective` |
| `select_fdr` | Knockoff W-statistic thresholding at FDR level q (different contract: returns a set, not a path prefix) | [knockoff_filter.py](../../sift/selection/knockoff_filter.py) |

Key shared machinery (reuse, do not reinvent):

- The CEFS+ objective path. `cefsplus_loop_with_objective` returns
  `obj[t] = log|Σ_S| − log|Σ_{y,S}| = −log(1 − R²_t) = 2·Î(y; S_t)`, where
  `R²_t = r_S'R_S⁻¹r_S` is the squared multiple correlation of (copula-space)
  y on the first t selected features. The per-step gain is
  `g_t = obj(t) − obj(t−1) = −log(1 − ρ̂²_t)` with `ρ̂_t` the **sample partial
  correlation** of the entering feature with y given the previous t−1.
- `compute_objective_for_path(cache, y, feature_path)` evaluates that same
  objective for an arbitrary ordered path against one full cache. Reuse it for
  full-cache path checks, but **not** as the cross-fit primitive: fold and
  bootstrap methods must build fold/replicate-local `R_path, r_path` and call
  `objective_from_corr_path` directly after local centering/scaling (§C.1).
- `sample_knockoffs(cache, ...)` draws Gaussian-copula knockoffs in cache
  space (IDEA-4).
- `choose_k_from_score_curve(diagnostics, config, lower_is_better)` applies
  the `best`/`one_se`/`plateau`/`tolerance` rules to any score curve — new
  curve-producing methods should route through it rather than duplicating
  rule logic.

### A.2 Failure mode 1: BIC-family penalties are structurally too weak here

`penalized_objective` treats `n·g_t` as a χ²(1) log-likelihood-ratio gain and
charges `log n` (BIC) per feature. But the greedy step does not test *one*
feature — it takes the **maximum** gain over `m_t ≈ p − t + 1` candidates.
Under the null (no remaining signal), the expected maximum of m independent
χ²(1) draws is ≈ `2·log m − log log m + O(1)` (Gumbel domain). So the
selection keeps accepting null features whenever

```
2·log p  >  log n      ⇔      n  <  p²
```

which is almost always true in this library's regime (n=2000, p=200 →
2·log p ≈ 10.6 vs log n ≈ 7.6: BIC accepts nulls indefinitely). AIC (2) and
HQC (2 log log n) are hopeless. This is not a bug in the implementation — it
is the classic post-selection multiplicity gap, and it has a classic fix
(EBIC/RIC, IDEA-7) plus proper sequential tests (IDEA-1/2/3).

A second, smaller defect: `_objective_weight_diagnostics` defaults
`n_eff = weight_sum`, but `ensure_weights(normalize=True)` normalizes to
**mean 1**, so `weight_sum ≡ n` regardless of weight skew. The Kish effective
size `(Σw)²/Σw²` is already computed but only reported as a diagnostic. Every
new inference-flavored method below must use Kish by default (§C.4).

### A.3 Failure mode 2: the elbow is uncalibrated

`select_k_elbow` stops when `Δobj / max(|obj|, 1) < min_rel_gain`. Both the
numerator scale (gains shrink like 1/n_eff on nulls) and the denominator
(total accumulated MI, which depends on signal strength) are data-dependent,
so a fixed `min_rel_gain=0.02` means completely different things at n=1k vs
n=50k, or R²=0.05 vs R²=0.9. There is a *known* null scale for gains —
`max-of-m χ²(1)/n_eff` — and the elbow ignores it. IDEA-10 replaces this with
a noise-floor-calibrated changepoint; IDEA-1 replaces it with an exact test.

### A.4 Failure mode 3: `evaluate` fights a flat curve with a noisy ruler

Three compounding issues:

1. **Flat objective near the optimum.** With a regularized downstream model,
   adding 20 borderline features often changes holdout RMSE by <0.1%. The
   argmin of a flat curve estimated from one time-holdout split is close to a
   random draw from a wide plateau — hence the large run-to-run variance in
   `best_k`, and why `one_se`/`plateau`/`tolerance` were added as band-aids.
2. **Prefix-only bias.** The path is built on all rows (including validation
   rows), so prefix scores are mildly optimistic in a k-dependent way. The
   honest `nested` mode exists but only for selector classes and only for
   `evaluate`, and it's the most expensive option in the library.
3. **Proxy mismatch and hyperparameter noise.** RidgeCV-per-path alpha,
   logistic C=1.0 — the proxy model's inductive bias differs from the GBM the
   features actually feed, and alpha selection adds its own variance.

The deep lesson: "argmin of a predictive curve" is the wrong target when the
curve is flat. The defensible targets are either (a) *support recovery* —
stop when the evidence that remaining features carry conditional signal is
exhausted (IDEAs 1, 2, 3, 4, 7, 8, 10), or (b) *predictive sufficiency* —
smallest k statistically indistinguishable from the best k (IDEAs 5, 6, 9,
plus existing `one_se`/`plateau` rules on much cheaper, denser curves). The
harness scores both targets explicitly.

---

## B. Phase 0 — the evaluation harness (build this first)

Nothing in §D gets merged as a default until it clears gates measured by this
harness. All ideas are cheap to rank once this exists.

### B.1 Files

- `benchmarks/auto_k_designs.py` — ground-truth data generators (importable
  by tests as well; keep dependency-light: numpy + the library itself).
- `benchmarks/bench_auto_k.py` — CLI runner:
  `python benchmarks/bench_auto_k.py --designs D1,D2,D5 --methods bic,ebic,chi2_stop --seeds 30 --output out.csv [--model ridge|catboost] [--full]`.
- `tests/test_auto_k_v2.py` — fast deterministic subset (small n/p, ≤3 seeds
  per assertion, total runtime target < 60s).
- Recorded results table appended to `benchmarks/README.md` (house
  convention).

### B.2 Designs

Each design is a function `make(seed) -> (X_df, y, meta)` plus
`sample_test(seed, n_test) -> (X_test_df, y_test)` drawing **fresh** data from
the same DGP (we own the DGP; fresh-draw risk is the cleanest ground truth).
`meta` carries: `true_support` (list of ints or list of blocks), `k_star`
(effective support size; `None` where undefined), optional `groups`, `time`.
Default train sizes below; `--full` doubles n and p.

| ID | Design | Purpose |
|---|---|---|
| D1 | iid N(0,1) X, p=200, n=2000; 10 signals, `β = linspace(1.5, 0.5, 10)`, σ=1 | clean sparse baseline |
| D2 | AR(1) ρ=0.6, p=200, n=2000; 10 signals at spaced indices {0,20,40,…} plus one adjacent pair {120,121} | correlated design; adjacency stress |
| D3 | 8 latent signals, each expanded into a block of 5 noisy copies (within-block corr ≈0.95) + 160 pure-noise features; `k_star = 8` **blocks** | redundancy: interacts with `corr_prune`; support scored block-wise |
| D4 | 40 weak signals, equal β with each contributing ~0.25% of Var(y), p=400, n=5000 | dense-weak: no sharp k*; predictive regret only |
| D5 | global null: β=0, p=200, n=2000 | calibration: error-controlled methods must not fire |
| D6 | heavy-tail/monotone: X ~ lognormal and t(3) columns, y = Σ β_j f_j(x_j) + ε with monotone f_j (exp/log/cube) on the D1 signal pattern | copula-robustness of the null math |
| D7 | n=300, p=2000, 8 signals, iid X | n≪p: screening + multiplicity stress; EBIC territory |
| D8 | D2 signal pattern + group random intercepts (200 groups of 10 rows, group effect σ_g=1) and a slow time drift added to y; `groups`, `time` in meta | structure honesty; naive-permutation negative control |
| D9 | n=50_000, p=2000, 15 signals (timing only, 2 seeds) | runtime budget check |

Implementation notes:

- D3's block-wise scoring: a *true positive* is a block with ≥1 selected
  member; a *false positive* is any selected pure-noise feature; precision
  uses selected-feature counts with within-block extras counted as neither TP
  nor FP (they're redundant-but-not-wrong). Encode this in one scoring helper
  with unit tests.
- D8's group effect makes rows within a group positively correlated, which is
  exactly what breaks iid permutation nulls and iid CV — the point of the
  design.
- Follow the style of `_reference_design` in
  [tests/test_knockoff_fdr_control.py](../../tests/test_knockoff_fdr_control.py);
  factor generators so both tests and benchmarks import them.

### B.3 Protocol (per design × seed × method)

1. Draw train data; build one cache (`compute_Rxx=True` when p ≤ 4000,
   default subsample); build the **full-data CEFS+ path** to
   `max_k = min(p, max(4·k_star_guess, 100))` with
   `return_objective=True`. This path is shared by all path-based methods so
   they differ only in the stopping decision.
2. Run the auto-k method → `k̂` (fold-based methods build their own internal
   fold paths but still *return* a k̂ applied to the shared full-data path —
   this mirrors the production `k="auto"` contract of returning
   `path[:k̂]`).
3. Risk curve, computed **once per design × seed** and shared across methods:
   for k on a dense grid (every k ≤ 30, every 5 to 100, every 25 beyond),
   fit `RidgeCV(alphas=np.logspace(-3, 3, 10))` on train `X[path[:k]]`
   (raw feature space, standardized), evaluate RMSE on a fresh
   `n_test=20_000` draw. `--model catboost` swaps in a small fixed-config
   CatBoost (guard the import) for a GBM-transfer check on D1–D3 only.
4. Oracle: `k_oracle = argmin_k rmse(k)` on the dense grid;
   `rmse_null` = RMSE of predicting the train mean.
5. Emit one CSV row:
   `design, seed, method, k_hat, k_oracle, k_star, rmse_hat, rmse_oracle, regret_frac, support_precision, support_recall, support_f1, k_dispersion_group, saturated_min, saturated_max, runtime_s, notes`
   where the headline metric is

   ```
   regret_frac = (rmse(k̂) − rmse(k_oracle)) / (rmse_null − rmse(k_oracle))
   ```

   i.e. the fraction of the achievable signal thrown away by the k choice
   (0 = oracle, 1 = as bad as no features). Robust to the design's absolute
   noise scale, comparable across designs.

### B.4 Baselines (run in every benchmark invocation)

`elbow` (defaults), `penalized/bic`, `evaluate/time_holdout/best`,
`evaluate/one_se`, plus two humility rows: `fixed_k=50` and `oracle`. Every
idea's report is a delta against these.

### B.5 Acceptance gates

An idea graduates from "experimental" to "documented candidate default" only
if it passes its gates. Record all results in `benchmarks/README.md` either
way — negative results are results.

- **G1 (accuracy):** on D1, D2, D3, D7 — median `|k̂ − k_oracle|` ≤ the best
  baseline's median, AND mean `regret_frac` ≤ best baseline's mean + 0.01.
- **G2 (calibration; only methods with an α or q):** on D5 with 50 seeds and
  the method's floor disabled (`min_k=0` for empty-capable methods),
  P(k̂ > 3) ≤ level + 2·SE where
  SE = √(level·(1−level)/50).
- **G3 (dense-weak):** on D4, mean `regret_frac` ≤ `evaluate/one_se` + 0.02
  (must not blow up; winning not required).
- **G4 (structure honesty):** on D8 with `groups`/`time` supplied, mean
  `regret_frac` within 0.02 of the same method's D2 result. (Also run
  `perm_gap` with `null="permute"` on D8 as a *documented negative control* —
  it is expected to fail; that demonstrates why structure-aware nulls exist.)
- **G5 (runtime):** path-only methods (1, 2, 7, 8, 10) ≤ 1.5× the fixed-k
  `select_cached` wall time on D9; resampling methods (3, 4, 9) within their
  documented B-multiple; fold methods (5, 6) ≤ 0.5× current
  `evaluate/time_holdout` wall time on D9.
- **G6 (decision stability):** std(k̂) across seeds ≤ 1.5× the best
  baseline's std on D1–D3.

Program-level success (what "auto-k finally works" means): at least one
method (or the consensus combiner) achieves **mean regret_frac ≤ 0.02 on
D1–D3+D7, std(k̂)/k_oracle ≤ 0.35, and G2-calibrated null behavior**. That
method becomes the recommended `AutoKConfig` default in DOCS.MD.

---

## C. Shared plumbing (one refactor PR before the ideas)

### C.1 `CandidatePanel` — factor the candidate funnel out of `select_cached`

Every idea below needs the same objects `select_cached` builds internally:
screened candidate indices, their correlation matrix, target correlations,
relevances. Extract (no behavior change; `select_cached` becomes a thin
wrapper):

```python
# sift/selection/panel.py
@dataclass(frozen=True)
class CandidatePanel:
    cand: np.ndarray          # candidate positions into cache valid-space, after screen+prune
    original: np.ndarray      # original input-column positions for cand
    R: np.ndarray             # (m, m) float64 candidate correlation matrix
    r: np.ndarray             # (m,) float64 target correlations
    rel: np.ndarray           # (m,) gaussian_mi_from_corr(r)
    p_valid: int              # number of valid features BEFORE screening (multiplicity base)
    n_eff_kish: float         # (Σw)²/Σw² over cache.sample_weight
    n_eff_sum: float          # Σw (≡ n_rows_cache for mean-1 weights)
    names: list[str] | None

def build_candidate_panel(cache, y, k, *, top_m=None, corr_prune="auto",
                          method="cefsplus", zy=None) -> CandidatePanel: ...
```

`zy` lets callers inject a pre-transformed target of length
`cache.Z.shape[0]` (permutation nulls pass a permuted-and-re-ranked target
without touching the rest); when `zy` is passed, `y` is used only for shape
validation or may be `None`. Keep the existing screening rule
(`top_m = max(5k, 250)` by `|r|`) and pruning (`greedy_corr_prune` at 0.95
for CEFS+) byte-identical; add a regression test that `select_cached` output
is unchanged on a seeded case.

Add a second, lower-level primitive for fold and bootstrap methods:

```python
def local_standardize(Z, w, *, columns=None) -> np.ndarray: ...
def local_corr_panel(Z, zy, w, *, top_m, corr_prune, method,
                     Rxx=None, local_standardize=True) -> CandidatePanel: ...
def score_path_from_corr(R_path, r_path, *, shrink=1e-6, eps=1e-12) -> np.ndarray: ...
```

This is load-bearing. The current weighted correlation helpers compute
uncentered weighted moments and are valid only when `Z` and `zy` are already
standardized under the same weights. Row-slicing the full-cache `Z`, changing
fold weights, or Bayesian-bootstrap reweighting breaks that invariant unless
the selected rows/weights are centered and scaled again before forming `R`
and `r`. Use the same mean/variance convention as the existing
`weighted_standardize` helpers (zero-variance columns are excluded or
neutralized before correlation). Setting the diagonal to 1 on uncentered,
non-unit-variance row slices can make the matrix indefinite and corrupt the
Schur complements. `xfit_mode="shared_z"` therefore means "reuse full-data
marginal ranks, then fold-local center/scale before correlations", not "take
raw row slices of `Z` and call `weighted_correlation_matrix`".

### C.2 Config and dispatch

- Extend `AutoKConfig.k_method` literals as each idea lands:
  `"chi2_stop"`, `"forward_stop"`, `"perm_gap"`, `"knockoff_path"`,
  `"xfit_objective"`, `"gaussian_cv"`, `"k_posterior"`, `"stability"`,
  `"changepoint"`, `"consensus"` — plus new `objective_penalty` values
  `"ebic"`, `"ric"`.
- New fields are flat on `AutoKConfig` (house style), prefixed per method
  (full table in §C.5). `validate_auto_k_config` must validate every new
  field's range **and warn (`UserWarning`) when a method-specific field is
  set to a non-default value while `k_method` doesn't use it** — the
  knockoffs FIX-6 typo-blindness lesson, adapted to a flat dataclass.
- Dispatch: register handlers in `_gaussian_spec` in
  [filter_api.py](../../sift/selection/filter_api.py) via
  `auto_k_handlers["<k_method>"] = make_auto_gaussian(method_func, GAUSSIAN_<NAME>)`,
  with the new `select_gaussian_*_path` orchestrators in
  [filter_auto_k.py](../../sift/selection/filter_auto_k.py) following the
  existing `select_gaussian_elbow_path` template (build path via
  `_cached_filter_path`, compute k, emit `auto_k_summary`). Update
  `auto_k_mode_label` in the same PR as the first new method; the current
  label map raises before dispatch under the default `verbose=True`.
- Method eligibility is explicit, not inferred from "has an objective path":
  CEFS+ gets all methods; Gaussian mRMR/JMI/JMIM get only predictive
  fold-scoring methods (`xfit_objective`, `gaussian_cv`, and possibly
  `stability` as a path-reproducibility diagnostic) where the path is merely
  being evaluated. `chi2_stop`, `forward_stop`, EBIC/RIC, posterior, and
  changepoint are CEFS+-only because their null calibration assumes the
  greedy step maximizes the same conditional objective gain being tested.
- `resolve_auto_k_config` (no-config inference) is untouched; new methods
  are opt-in via explicit `AutoKConfig`. `_require_evaluate_context` already
  no-ops for non-evaluate methods; add per-method context requirements in the
  new orchestrators (e.g. `gaussian_cv` with `strategy="kfold"` needs
  nothing; `perm_gap` with `null="within_group"` needs `groups`).
- Exact fold modes need raw feature data, not just a prebuilt cache. Extend
  the Gaussian auto-k runner contract to pass `source_X`/encoded X plus
  `cache_was_prebuilt`, or create a specialized runner for fold methods.
  `xfit_mode="exact"` must reject prebuilt-cache calls unless the caller also
  supplies fold-rebuildable X; cache-only calls expose only `"shared_z"` and
  label it approximate. If X needs supervised categorical encoding, exact
  mode must use fold-local encoding like selector-class nested mode or reject
  the configuration; full-data target encoding is not exact.
- Empty-selection semantics are method-specific. Error-controlled methods
  (`chi2_stop`, `forward_stop`, `knockoff_path`, and posterior `P(k=0)`/MAP
  paths) may legitimately return `k=0`; their "no discovery" fallback is 0,
  not `min_k`. Relax `AutoKConfig.min_k` validation to allow 0, update
  `auto_k_summary` for `effective_min_k=0`, and keep predictive methods on
  positive effective floors unless the method explicitly opts into empty
  selection.
- Every method returns `(best_k, diagnostics_df)` and flows through
  `auto_k_summary` with `extra` fields named in its section. Respect
  method-specific `min_k`/`max_k` clamping semantics, the
  `_print_selected_k` verbose convention, and the warn-and-fallback pattern
  (degenerate inputs → method floor or `max_k` with a `UserWarning`, never a
  crash).
- Exports: public helpers (`select_k_chi2_stop`, etc.) from
  `sift/selection/__init__.py` and `sift/__init__.py` alongside
  `select_k_elbow`. DOCS.MD "Automatic k Selection" section, user-guide, and
  troubleshooting ("which auto-k method should I use" decision table) updated
  in the same PR as each method.

### C.3 Numerics conventions

- All p-value math in float64 with log-space forms:
  `scipy.stats.f.logsf` for tail probabilities; Šidák via
  `p = -expm1(m * log1p(-p1))` and the small-p branch `p ≈ m·p1` when
  `p1 < 1e-12`. For ForwardStop, compute
  `Y_t = -log1p(-clip(p_t, 0, 1-eps))`; when possible, use log-survival
  quantities directly to avoid rounding a highly non-significant max-p to
  exactly 1. scipy ≥1.10 is already a hard dependency.
- Gains can be ≈ −1e-16 from the eps floors in the log-det recursions
  (knockoffs FIX-1 lesson): clamp `g_t = max(g_t, 0.0)` before any transform.
- Determinism: use `np.random.SeedSequence(config.random_state).spawn(B)` and
  `np.random.default_rng(child)` for replicate streams. Do not use
  `Generator.spawn`; the project supports NumPy 1.24.
- Statistical methods must clamp their effective path length before computing
  tails or debias terms. For any formula using
  `ν_t = n_eff − t − 1`, require `ν_t > 0`; in practice set
  `stat_max_k = min(requested_max_k, floor(n_eff) - 2)` (and fold-local
  analogues for validation curves), record the clamp in diagnostics, and warn
  or return the method floor if the statistical max is below the floor.

### C.4 Effective sample size policy

New field `n_eff_mode: Literal["auto", "kish", "weight_sum"] | float =
"auto"` consumed by every v2 method and accepted by
`select_k_penalized_objective`. `"auto"` resolves by method: Kish for
EBIC/RIC/posterior and all new inference-flavored methods; the historical
`weight_sum` behavior for legacy BIC/AIC/HQC unless the caller explicitly
requests Kish. Kish is `n_eff = (Σw)² / Σw²` over the cache's (subsampled)
weights. Docs must say that mean-1 normalization makes `weight_sum ≡ n`
regardless of weight skew.
Where a formula needs "n at step t", use `ν_t = n_eff − t − 1` and the
statistical max-k clamp in §C.3.

### C.5 New `AutoKConfig` fields (complete table)

| Field | Default | Used by | Validation |
|---|---|---|---|
| `n_eff_mode` | `"auto"` | all v2 + penalized objective | `"auto"`, `"kish"`, `"weight_sum"`, or float > 1 |
| `min_k` | existing | all | relax to int ≥ 0; predictive methods coerce effective floor to ≥1 unless explicitly empty-capable |
| `alpha` | `0.05` | chi2_stop, forward_stop, perm_gap | 0 < α < 1 |
| `m_mode` | `"all"` | chi2_stop, forward_stop | `"all"`, `"panel"`, `"li_ji"` |
| `stop_patience` | `2` | chi2_stop, changepoint | int ≥ 1 |
| `perm_B` | `20` | perm_gap | int ≥ 10 |
| `perm_null` | `"auto"` | perm_gap | `"auto"`, `"permute"`, `"circular_shift"`, `"within_group"` |
| `gap_rule` | `"tibshirani"` | perm_gap | `"tibshirani"`, `"argmax"`, `"gain_envelope"` |
| `knockoff_q` | `0.2` | knockoff_path | 0 < q < 1 |
| `knockoff_draws` | `1` | knockoff_path | int ≥ 1 |
| `knockoff_s_method` | `"equi"` | knockoff_path | `"equi"`, `"mvr"`, `"me"` |
| `knockoff_return` | `"set"` | knockoff_path | `"set"`, `"prefix"` |
| `xfit_folds` | `5` | xfit_objective, gaussian_cv | int ≥ 2 |
| `xfit_mode` | `"shared_z"` | xfit_objective, gaussian_cv | `"shared_z"`, `"exact"` |
| `xfit_ridge` | `1e-3` | gaussian_cv | ≥ 0 |
| `strategy` | (existing) | + `"kfold"` for fold methods | extend literal set |
| `ebic_gamma` | `"auto"` | penalized ebic, k_posterior | `"auto"` or 0 ≤ γ ≤ 1 |
| `posterior_level` | `0.9` | k_posterior | 0 < level < 1 |
| `posterior_pick` | `"map"` | k_posterior | `"map"`, `"smallest_in_hpd"` |
| `boot_B` | `30` | stability | int ≥ 10 |
| `boot_mode` | `"bayes"` | stability | `"bayes"`, `"half"` |
| `stability_rule` | `"max_one_se"` | stability | `"max_one_se"`, `"pi_threshold"` |
| `stability_pi` | `0.6` | stability (pi_threshold) | 0.5 < π ≤ 1 |
| `floor_z` | `2.5` | changepoint | > 0 |
| `floor_window` | `0.2` | changepoint | 0 < w ≤ 0.5 (fraction) or int ≥ 5 |
| `consensus_methods` | `("ebic", "chi2_stop", "perm_gap", "gaussian_cv")` | consensus | non-empty tuple of implemented methods |

---

## D. The ideas

Shared notation: `obj(k)` the CEFS+ objective path, `g_t ≥ 0` the per-step
gain, `n_eff` per §C.4, `p` = `CandidatePanel.p_valid` (candidates before
screening), `m_t = p − t + 1` remaining-candidate count at step t,
`ν_t = n_eff − t − 1` the residual degrees of freedom (t−1 conditioning
features + intercept-equivalent + the tested feature). Any method using
`ν_t` must clamp `max_k` so `ν_t > 0` for every evaluated step (§C.3).

### D.0 The one identity everything below uses

Adding feature j at step t with observed gain g is equivalent (in copula
space, under Gaussianity) to a partial-F test of "j ⊥ y | S_{t−1}":

```
ρ̂²_t = 1 − e^{−g_t}                         (sample partial correlation²)
F_t   = ν_t · (e^{g_t} − 1)  ~  F(1, ν_t)    under H0, exactly (Gaussian iid)
n_eff · g_t                  →  χ²(1)         asymptotically
```

The rank-Gauss transform, the weights, and the greedy's data-chosen
conditioning set S_{t−1} all make "exactly" into "approximately"; that
approximation is precisely what the D5/D6/D8 designs test empirically.

---

### IDEA-1 — `chi2_stop`: sequential max-gain testing (the calibrated elbow)

**What.** Stop the path at the first (patience-smoothed) step whose gain is
not significant *as a maximum over the remaining candidates*. Pure math on
the existing objective path; zero extra selector cost.

**Math.** Single-candidate p-value at step t:
`p⁽¹⁾_t = SF_{F(1, ν_t)}(ν_t·(e^{g_t} − 1))`. The greedy took a max over m_t
candidates, so correct: Šidák `p_t = 1 − (1 − p⁽¹⁾_t)^{m_eff_t}` (log-space
per §C.3). `m_eff_t` by `m_mode`:

- `"all"` (default, conservative): `m_eff_t = p − t + 1`. Rationale: the
  top-m screening picked the largest |r| out of all p, so the step-1 max over
  the panel *is* the max over all p; later steps approximately so.
- `"panel"`: `len(panel.cand) − t + 1` (anti-conservative; diagnostic only).
- `"li_ji"`: effective number of tests from the panel eigenvalues
  (Li & Ji 2005): `m_eff = Σ_i [1(λ_i ≥ 1) + (λ_i − ⌊λ_i⌋)]` for eigenvalues
  λ_i of `panel.R`, scaled by `(p − t + 1)/len(cand)` to extrapolate off-panel
  candidates. One `eigvalsh` on an m×m matrix (m ≤ ~500: milliseconds).

Note `corr_prune` removes near-duplicates *before* the max; their test
statistics are ≈ perfectly correlated with retained features, so the
effective count is essentially unchanged — one docstring sentence, no
adjustment.

**Stopping rule.** `B_t = 1{p_t > alpha}`. k̂ = `t* − 1` where t* is the
start of the first run of `stop_patience` consecutive B_t = 1 with
`t* − 1 ≥ effective_min_k`; if the first run starts at t=1 and the method
allows empty selections, k̂ = 0. If no such run, k̂ = effective max (set
`selected_at_effective_max_k`). Patience exists because greedy ordering is
not monotone in signal (a masked signal can enter after a null feature);
patience=2 recovers those at bounded FDR cost. For α-semantics, a no-signal
fallback must be 0, not an arbitrary positive `min_k`.

**Implementation.** New module `sift/selection/auto_k_stop.py`:

```python
def path_gain_pvalues(objective_path, *, n_eff, p_candidates, m_mode="all",
                      panel_eigs=None) -> np.ndarray: ...
def select_k_chi2_stop(objective_path, config, *, n_eff, p_candidates,
                       panel_eigs=None) -> tuple[int, pd.DataFrame]: ...
```

Orchestrator `select_gaussian_chi2_path` in filter_auto_k.py (template:
`select_gaussian_penalized_path`). Diagnostics columns: `k, objective,
gain, F_stat, nu, m_eff, p_single, p_max, significant, selected` + summary
extras `{"alpha": ..., "m_mode": ..., "n_eff": ..., "stopped_by": "test"|"max_k"}`.

**Cost.** O(K) on top of the path. Nothing else.

**Tests.**
- Exact-null unit test: for iid Gaussian data with zero signal, generated
  directly in copula space (skip rank transform), `p⁽¹⁾_1` is U(0,1) across
  200 seeds at n=500, p=1 candidate (KS test p > 0.01) — validates the F
  identity against the actual `cefsplus_loop_with_objective` output.
- Deterministic toy: hand-computed 3-feature example where feature gains and
  p-values are verified against `scipy.stats.f.sf` by direct regression
  residual computation.
- D5-style null (small: n=500, p=50, 5 seeds): k̂ ≤ 2 in all seeds at α=0.05,
  `min_k=0`.
- D1-style signal (n=1000, p=50, 8 strong signals, 3 seeds): 6 ≤ k̂ ≤ 12.

**Gates.** G1, G2 (level = alpha), G5, G6. Expected weakness: D4 (dense weak
— sequential testing under-selects when every effect is tiny); G3 is the
watch-item, not a kill criterion.

---

### IDEA-2 — `forward_stop`: FDR-controlled stopping via accumulation tests

**What.** Same per-step p-values as IDEA-1, but instead of "stop at first
failure", choose k with a guaranteed false-discovery-rate interpretation
using the ForwardStop rule for ordered hypotheses (G'Sell et al. 2016).

**Math.** With p-values `p_1, …, p_L` along the path (Šidák-corrected as in
IDEA-1) define `Y_t = −log1p(−clip(p_t, 0, 1−eps))` (≈0 for strong steps,
Exp(1) for null steps). ForwardStop:

```
k̂ = max{ k ∈ [max(1, min_k), max_k] : (1/k) · Σ_{t≤k} Y_t ≤ alpha }
```

(fallback 0 if the set is empty; record `stopped_by`). Under
independent p's with U(0,1) nulls this controls FDR of the selected prefix at
α. Our p's are sequentially dependent, so the guarantee is approximate —
which is exactly what harness gate G2 measures. Robustness bonus over
IDEA-1: one interior large p (a masked-then-revealed signal) doesn't
terminate the path; the *average* has to degrade.

**Implementation.** Same module as IDEA-1; `select_k_forward_stop(...)`
consuming `path_gain_pvalues`. Diagnostics: `k, p_max, Y, Y_running_mean,
eligible, selected`. ~50 lines beyond IDEA-1.

**Tests.** Unit: hand-built p-value sequences (e.g. p = [1e-8]×5 then
U(0,1)-ish values) recover the expected k̂ for α ∈ {0.1, 0.2}; monotonicity
in α (larger α → k̂ non-decreasing). Sims: same as IDEA-1 with level = α
interpreted as FDR over selected prefix: on D1-style sims report realized
prefix FDP = 0 when k̂=0 else (#selected nulls)/k̂, mean ≤ α + 0.05.

**Gates.** G1, G2 (as FDP of prefix), G5, G6. Kill criterion: if realized
FDP on D2 (correlated) exceeds 2α systematically, the independence
approximation is too hot — demote to diagnostic-only alongside IDEA-1.

---

### IDEA-3 — `perm_gap`: permutation-null objective envelope (gap statistic)

**What.** Empirically calibrate the whole objective curve instead of trusting
the F/χ² approximations: rerun the greedy on B permuted targets, compare the
real curve to the null envelope, stop where they merge. This is Tibshirani's
gap statistic transplanted from clustering to a selection path, with
structure-aware nulls. It is the method that survives when the copula/weights
break IDEA-1's analytic null — and the pair (IDEA-1, IDEA-3) cross-validate
each other.

**Math.** For b = 1..B draw a null target `y⁽ᵇ⁾` (see null modes below),
build its panel and greedy objective path `obj⁽ᵇ⁾(k)` for k ≤ max_k.

```
Gap(k) = obj(k) − mean_b obj⁽ᵇ⁾(k)
s_k    = sd_b(obj⁽ᵇ⁾(k)) · sqrt(1 + 1/B)
```

Under exhausted signal, real gains and null gains are draws from the same
max-over-remaining-candidates distribution, so E[Gap] rises while signal
remains and then goes *flat* (not down). Rules:

- `"tibshirani"` (default): k̂ = smallest k ∈ [min_k, max_k−1] with
  `Gap(k) ≥ Gap(k+1) − s_{k+1}`; fallback argmax.
- `"argmax"`: argmax_k Gap(k), smallest tie.
- `"gain_envelope"`: stop at first run (patience `stop_patience`) of
  `g_t ≤ mean_b g⁽ᵇ⁾_t + z_α · sd_b(g⁽ᵇ⁾_t)` — a per-step permutation test;
  reuse the IDEA-1 run logic.

**Null modes** (`perm_null`): `"permute"` iid permutation of y;
`"circular_shift"` (requires `time`): sort rows by time and mirror the
existing [\_permute.py](../../sift/_permute.py) circular-shift convention
(`rng.integers(1, n)` per group). If we want to exclude tiny shifts, add a
separate `perm_min_shift_frac` option instead of silently changing existing
null semantics. `"within_group"` (requires `groups`): permute y within
groups — preserves group effects; `"auto"`: time → shift, else groups →
within_group, else permute.

**Critical honesty details.**

1. Permute **raw y first, then slice, then re-transform**. Concretely:
   `y_arr = to_numpy(y, dtype=np.float32).ravel()`; convert `groups` and
   `time` positionally with `np.asarray(...).reshape(-1)`, never pandas label
   indexing. For structured nulls, draw the permutation/shift on the full
   `y_arr` using the full `time`/`groups` arrays so autocorrelation and group
   structure are preserved; only then take
   `y_b_cache = y_b_full[cache.row_idx]` and compute
   `zy⁽ᵇ⁾ = weighted_rank_gauss_1d(y_b_cache, cache.sample_weight)`. Applying
   a circular shift after cache subsampling destroys the time-series structure.
   With weights, permuting the already-transformed `zy` is *not* the same
   thing (ranks are weight-dependent); always permute the raw target against
   fixed (X, w) rows.
2. **Re-screen per permutation.** The top_m screen and `corr_prune` are
   y-dependent; the null run must repeat them with `|r⁽ᵇ⁾|`
   (`build_candidate_panel(..., zy=zy_b)`). Reusing the real-y panel would
   understate the null max (screening already threw away the features a null
   run would have found), inflating Gap and overselecting. With `cache.Rxx`
   present, per-permutation panels are cheap slices; without it, each
   permutation pays one `top_m × top_m` weighted correlation — document, and
   recommend `compute_Rxx=True` for p ≤ ~4000.

**Implementation.** `sift/selection/auto_k_resample.py`:

```python
def null_objective_paths(cache, y, *, B, max_k, null, time=None, groups=None,
                         top_m, corr_prune, random_state) -> np.ndarray  # (B, max_k)
def select_k_perm_gap(objective_path, null_paths, config) -> tuple[int, pd.DataFrame]
```

plus orchestrator. Null paths that exhaust early (prune) are extended flat
(no further gains). Diagnostics: `k, objective, null_mean, null_sd, gap,
gap_se, selected` + extras `{"perm_B", "perm_null", "gap_rule"}`.

**Cost.** ≈ B × (one rank-1d transform + one corr-with-vector + panel slice +
greedy). With Rxx and p=2000, top_m=500, B=20: ~20× the greedy loop, well
under the current `evaluate` cost. D9 budget: ≤ 60s.

**Tests.** Unit: with B fixed permutation seeds, diagnostics reproducible;
flat-extension logic; tibshirani rule on a synthetic Gap curve with known
knee. Sims: D5-style null → k̂ ≤ min_k+1 in ≥ 90% of seeds; D1-style → k̂
within [k*−2, k*+4]; **negative control**: D8-style grouped data with
`null="permute"` overselects (assert k̂ > k*+10 in most seeds) while
`null="within_group"` does not — this test documents *why* the option
exists.

**Gates.** G1, G2 (level ≈ implied by rule; report empirically), G4
(load-bearing for this idea), G5 (B-multiple). This is the expected
workhorse for weighted/copula-weird data.

---

### IDEA-4 — `knockoff_path`: knockoff-interleaved path stopping

**What.** Augment the candidate panel with Gaussian-copula knockoffs (sampler
already shipped), run **one pair-aware** CEFS+ greedy over originals ∪ knockoffs, and
stop the path by counting knockoff entries: knockoffs entering the path are
direct eyewitnesses of "this deep in the path, entries are noise." Distinct
from `select_fdr(statistic="cefsplus")`: no W-thresholding machinery, one
draw, and the output is a stop position on the path (the CEFS+ semantics the
user actually consumes), at ~2–4× the cost of a plain run.

**Math.** For each screened pair j let `e_j` = the step at which the *first*
member of the pair (original or knockoff) enters the joint path, and
`L_j = +1` if the original entered first, `−1` otherwise. Order pairs by
`e_j`. By pairwise exchangeability of nulls, the labels of null pairs are
iid fair coins regardless of entry order, so Selective SeqStep+
(Barber–Candès 2015, Thm 3) applies to the running estimate

```
FDP̂(i) = (1 + #{L_j = −1 : rank ≤ i}) / max(1, #{L_j = +1 : rank ≤ i})
î      = max{ i : FDP̂(i) ≤ knockoff_q }        (0 if none)
```

Selected set = originals with L=+1 among the first î pairs; `k̂ = |selected|`.
FDR ≤ q applies to that **set**, up to the second-order/copula approximation
of the knockoff construction (same caveat as `select_fdr`; measured, not
assumed). A CEFS+ prefix of length `k̂` is a different object and does not
inherit this q interpretation.

**Implementation.** In `auto_k_stop.py` (or a small `auto_k_knockoff.py`):

1. `Zt = sample_knockoffs(cache, s_method=config.knockoff_s_method, random_state=...)`.
2. Screening must be **pair-symmetric** — reuse the screening convention from
   [knockoff_filter.py](../../sift/selection/knockoff_filter.py) (its
   antisymmetry tests prove it pair-safe). Do NOT reuse the plain top-|r|
   original-only screen.
3. `corr_prune` must be **disabled** for the joint run (pruning could delete
   one member of a pair, destroying exchangeability) — hard-code, document.
4. Joint panel: stack screened original and knockoff columns (n × 2m) and
   compute the empirical weighted correlation matrix and target correlations.
   Do **not** call the ordinary `cefsplus_loop`: its positional tie-breaking
   on a stacked array favors originals over knockoffs and breaks
   swap-exchangeability. Reuse `_cefsplus_incremental_scores` from
   `knockoff_filter.py`, which already has pair-aware tie neutralization, and
   extend it (or factor a sibling helper) to return the internal selected
   entry sequence and gains, not just the final antisymmetric scores.
5. Compute the pair entry table from that returned entry sequence, then FDP̂
   path, î, k̂. Do not reconstruct order by sorting nonzero gains; gains need
   not be monotone.
6. Return per `knockoff_return`: `"set"` (default) → the selected originals
   themselves, with q/FDR metadata; `"prefix"` → full-data original CEFS+
   prefix of length k̂ for callers that require a prefix, explicitly labeled
   `fdr_controlled=False` / `count_only=True` because q applies only to the
   set. Diagnostics carry both objects plus the pair table
   (`feature, entry_step, label, entry_gain`) and the FDP̂ path.
7. `knockoff_draws > 1`: repeat with `SeedSequence.spawn` children, aggregate
   sets by selection frequency or k̂ by median depending on `knockoff_return`,
   and report per-draw k̂ spread in diagnostics.

**Cost.** One knockoff GEMM sample + one (2m)² correlation + one 2m-wide
greedy ≈ 2–4× fixed-k selection. No refits, no folds.

**Tests.** Unit: pair-table construction on a crafted path (hand-set entry
order); exact-tie antisymmetry where a 0-gain original/knockoff pair is
neutralized rather than original-first; SeqStep+ arithmetic against
hand-computed FDP̂; prune-disabled assertion; determinism per seed. Sims:
D5-null → k̂ = 0 in ≥ 80% of seeds at q=0.2; D1-style → recall ≥ 0.8 of true
support in selected set, realized FDP ≤ 0.3 mean over 15 seeds (mirror the
bars in `test_cefsplus_reference_calibration_and_power`).

**Gates.** G1, G2 (level q), G5, G6. Known risk: at small n the knockoff
doubling costs power (signals can lose the race to their own knockoffs);
expect it to shine at n ≥ ~5k and to be dominated by IDEA-3 below that —
the harness will say.

---

### IDEA-5 — `xfit_objective`: cross-fitted, debiased objective curve

**What.** The honest version of the objective path, at correlation-math cost:
select on fold-train rows, evaluate the *objective* (not a ridge model) on
fold-validation rows, debias the known null drift, pick the argmax. No
downstream model, no hyperparameters, no metric choice.

**Math.** For fold f with train rows T_f and validation rows V_f: build the
train panel from fold-train locally standardized correlations, run the greedy
on train correlations → ordered path `π_f = (j_1, j_2, …)`. Evaluate on
validation by locally standardizing the validation rows/weights, extracting
`R_Vf[π_f], r_Vf[π_f]`, and calling `objective_from_corr_path` directly.
Do **not** call `compute_objective_for_path` unless the cache object is a
true fold-local validation cache with `row_idx=np.arange(n_fold)`, fold-local
weights, and a fold-local target transform.

For a null feature at step t, the validation partial correlation is centered
at 0 but has positive log-gain drift. Use the exact Gaussian null expectation
when possible:

```
ν_{f,t}      = n_eff,Vf − t − 1
drift_{f,t} = digamma((ν_{f,t} + 1) / 2) − digamma(ν_{f,t} / 2)
```

This is `≈ 1/ν_{f,t}` for large ν; the older `1/(n_eff,Vf − t)` formula is
only a first-order shortcut.

```
D_f(k) = obj_f(k) − Σ_{t=1}^k drift_{f,t}
D(k)   = mean_f D_f(k),   se(k) = sd_f(D_f(k))/√F
```

Signal steps gain their true conditional MI (O(1) vs the O(1/n) drift);
null steps gain ≈ 0 after debias. k̂ via
`choose_k_from_score_curve(lower_is_better=False)` with the full existing
rule set (`best` default; `one_se` supported since fold-level curves give a
real SE).

**Folds.** `strategy="time_holdout"` → single early/late split (val_frac);
`"group_cv"` → GroupKFold(n_splits); new `"kfold"` → shuffled KFold(n_splits,
random_state). `xfit_folds` aliases n_splits for the fold methods.

**Modes.** `xfit_mode="shared_z"` (default): keep the full-cache marginal
ranks, but center/scale `Z` and `zy` under each fold's rows and weights before
forming correlations. Leakage is limited to the marginal rank transform having
seen all rows — second-order; document. `"exact"`: rebuild train and
validation caches per fold (honest to the letter; used by the harness once to
measure the shared_z gap; if negligible, shared_z stays default). Exact mode
requires raw/encoded X; reject prebuilt-cache-only calls.

**Implementation.** `sift/selection/auto_k_xfit.py`:

```python
def xfit_objective_curves(cache, y, *, config, groups=None, time=None,
                          top_m, corr_prune, method, source_X=None,
                          cache_was_prebuilt=False) -> pd.DataFrame
    # per-fold: local-standardized train panel → greedy path
    #           → local-standardized val objective → debiased D_f(k)
def select_k_xfit_objective(curves, config) -> tuple[int, pd.DataFrame]
```

Path per fold uses train-row weighted correlations after fold-local
centering/scaling; validation correlations are restricted to the fold path's
features (k × k — small) and are also validation-local centered/scaled.
Diagnostics: `k, score_mean(=D), score_se, per-fold columns via split_scores
tuple` — reuse `build_score_curve_diagnostics` so the plateau/one_se plumbing
just works. Extras: `{"xfit_mode", "xfit_folds", "debias": True}`.

**Cost.** F × (fold panel + greedy + O(K²) objective). No sklearn, no
encoders. Expect ≥ 10× faster than current `evaluate`, evaluated at *every*
k (no coarse grid) — better plateau resolution for free.

**Tests.** Unit: local centering/scaling under fold weights changes `R`/`r`
relative to raw row-sliced `Z`, and xfit uses the local values; debias term
formula against simulated null partial correlations (mean of
`−log(1−ρ̂²)` over 2000 reps close to the digamma expression); fold plumbing
determinism; shared_z vs exact equivalence on unweighted data within
tolerance. Sims: D1: argmax of D within [k*−2, k*+3] in ≥ 80% of seeds;
D5-null: D(k) has no significant positive slope (fraction of seeds with
argmax > 5 bounded).

**Gates.** G1, G3 (this one *should* do well on dense-weak), G4, G5 (must
beat evaluate on runtime decisively), G6.

---

### IDEA-6 — `gaussian_cv`: closed-form CV risk curve (no models, every k)

**What.** The `evaluate` semantics — out-of-sample predictive risk vs k —
without fitting a single sklearn model: in copula space the "model" for a
prefix is linear with coefficients available in closed form from the train
correlation matrix, and its validation risk is a quadratic form in the
validation correlations. This is `evaluate` with the ridge replaced by exact
linear algebra: ~100× cheaper, dense in k, no alpha noise, honest
(fold-train paths), and it plugs straight into the existing
`one_se`/`plateau`/`tolerance` rules.

**Math.** Copula space with fold-local centering/scaling. Fold f: train
correlations (R_f, r_f) over the fold path prefix; validation correlations
(R̃_f, r̃_f), each computed from rows/weights standardized inside that fold
split.

```
β_k    = (R_f[:k,:k] + λI)⁻¹ r_f[:k]          λ = xfit_ridge (default 1e-3)
risk_f(k) = 1 − 2·β_k'r̃_f[:k] + β_k'R̃_f[:k,:k]β_k
```

`risk_f(k)` is the exact out-of-sample normalized MSE of the Gaussian-model
predictor — the population quantity RidgeCV+RMSE estimates noisily. Compute
for **all** k ≤ max_k via incremental Cholesky of R_f (rank-1 border
updates, the codebase's signature move): total O(K³/3) flops — milliseconds
at K=300. Aggregate across folds → `build_score_curve_diagnostics` →
`choose_k_from_score_curve(lower_is_better=True)`; default
`selection_rule="one_se"` (the SE is finally trustworthy: F fold curves, no
model-fitting noise inside them).

**Relationship to IDEA-5.** Same folds, panels, and paths — implement both
in `auto_k_xfit.py` sharing all plumbing; they differ only in the score
computed from (train, val) correlations. IDEA-5 measures *information*
(tuning-free, debiased argmax); IDEA-6 measures *transferable prediction*
(matches user intent, supports parsimony rules). Ship both; the harness
picks the default.

**Implementation.** `select_k_gaussian_cv(...)` in `auto_k_xfit.py`; share
the same fold builder and local-standardized correlation code as
`xfit_objective`; jitter retry on Cholesky failure (add 10λ, warn);
diagnostics identical in shape to the evaluate curve (drop-in for downstream
tooling) plus `{"proxy": "gaussian_linear_copula", "xfit_ridge": λ}`.

**Tests.** Unit: risk formula equals brute-force
`1 − 2βᵀr̃ + βᵀR̃β` computed by explicit lstsq on small locally standardized
matrices; local fold standardization changes the weighted curve on a
skew-weight split; equals (up to transform) actual holdout MSE of a linear
model fit on rank-Gauss-transformed features (n=2000 sim, correlation > 0.99
between curves); incremental-Cholesky path equals per-k direct solves. Sims:
D1/D2: one_se pick within [k*−2, k*+4]; curve
monotone-decreasing-then-flat shape sanity.

**Gates.** G1, G3, G4, G5 (target: ≤ 0.05× current evaluate runtime), G6.
Expected to be the strongest all-rounder and the leading candidate for the
new recommended default. Watch-item: copula-linear proxy vs GBM transfer on
D6 (the `--model catboost` harness check exists for exactly this).

---

### IDEA-7 — `ebic` / `ric`: multiplicity-corrected information criteria

**What.** Fix §A.2 inside the existing `penalized_objective` machinery: add
penalties that charge for the max-over-p selection, not just for the
parameter. One day of work; the highest value-per-line in this document.

**Math.** Choose k maximizing (Chen & Chen 2008; Foster & George 1994):

```
EBIC_γ(k) = n_eff·obj(k) − k·log(n_eff) − 2γ·log C(p, k)
RIC(k)    = n_eff·obj(k) − 2k·log(p)
```

with `log C(p,k) = gammaln(p+1) − gammaln(k+1) − gammaln(p−k+1)` (exact —
also correct once k is not ≪ p, where the k·log p simplification breaks).
`ebic_gamma="auto"` → `γ = 0.0 if p <= 1 else min(1, max(0,
1 − log(n_eff)/(2·log(p))))` — the Chen–Chen consistency threshold; degrades
gracefully to plain BIC when n ≥ p² or when there is only one valid
candidate. Note EBIC with γ=1 is exactly the Scott–Berger
multiplicity-corrected uniform model-size prior, so no separate `size_prior`
option is needed.

The null-acceptance condition from §A.2 becomes: accept a null max-gain
(~2 log p) only if `2·log p > log n + 2γ·log p`, i.e. never, for γ ≥ ½
(large-p limit) — the criterion is *calibrated to the greedy max* rather
than accidentally fighting it.

**Implementation.** In [auto_k.py](../../sift/selection/auto_k.py):

- Generalize `select_k_penalized_objective` to accept a per-k penalty
  *array*: internally build `penalty[k]` (BIC/AIC/HQC stay
  `weight·k`; EBIC/RIC use the forms above). Keep the public diagnostics
  columns, adding `penalty_kind, ebic_gamma, n_candidates`.
- New required argument for the new kinds: `n_candidates` (= panel
  `p_valid`, the count before screening/pruning), plumbed from the
  orchestrator. Raise a clear error if missing. Never use the post-screened
  panel width `len(panel.cand)` for EBIC/RIC multiplicity; that would
  under-penalize after aggressive screening.
- `n_eff_mode` honored (§C.4). EBIC/RIC/posterior default to Kish. Legacy
  BIC/AIC/HQC may keep their old `weight_sum` behavior only when explicitly
  requested through `n_eff_mode="weight_sum"` or a documented legacy branch.
- Binary CEFS+: EBIC ports verbatim to the refit log-likelihood gains path
  (`select_binary_penalized`, objective_scale=2.0):
  `−2·loglik + k log n_eff + 2γ log C(p,k)`; wire `n_candidates`,
  `n_eff_mode`, and `ebic_gamma` through the binary orchestrator too.

**Tests.** Unit: exact `log C(p,k)` vs `math.comb` for small p; γ="auto"
formula; per-k penalty array matches scalar path for BIC (regression). Sims:
D7-style (n=300, p=2000, 8 signals, 5 seeds): BIC k̂ > 30 (documents the
failure), EBIC-auto k̂ ∈ [5, 14]; D1-style: EBIC within [k*−2, k*+2] in ≥
80% of seeds; monotonicity: k̂(γ=0) ≥ k̂(γ=0.5) ≥ k̂(γ=1).

**Gates.** G1, G5, G6. G2 does not apply (no α), but D5 null: k̂ ≤ 2 in ≥
90% of seeds. Expected weakness: D4 (EBIC is a support-recovery criterion;
it will under-select dense weak signals — documented, not fixed).

---

### IDEA-8 — `k_posterior`: pseudo-posterior over k (uncertainty, not just a point)

**What.** The same quantities as IDEA-7, exponentiated and normalized into a
distribution over k — because half the practical pain of auto-k is not
knowing whether the choice was sharp or arbitrary. Output: MAP k̂, an HPD
credible set, `P(k=0)`, and entropy. When the credible set is wide the user
(or the consensus combiner, IDEA-11) knows the data don't pin k down and
parsimony rules should win.

**Math.** Grid k ∈ {0} ∪ [min_k grid … max_k] (k=0 has obj=0 — a calibrated
"no signal" mass):

```
log π̃(k) = ½·[ n_eff·obj(k) − k·log(n_eff) ] − γ·log C(p,k)
π(k)     = softmax over the grid (logsumexp)
```

(unit-information Gaussian prior on coefficients ⇒ the ½·BIC Laplace core;
γ-weighted binomial size prior ⇒ multiplicity correction). MAP ≡ EBIC argmax
by construction. HPD set: sort k by π descending, accumulate to
`posterior_level`, report the k-range envelope. `posterior_pick="map"` or
`"smallest_in_hpd"` (parsimony semantics).

**Honesty note (put verbatim in the docstring):** this is a *pseudo*-
posterior computed along one greedy path — it does not integrate over model
space and inherits the greedy's path-dependence. Its value is calibrated-ish
relative weighting, validated empirically: the harness records
**coverage** = fraction of runs with k_oracle ∈ HPD(0.9). Adopt the
diagnostic if coverage ≥ 0.7 on D1–D3; otherwise ship it clearly labeled
"relative evidence, not coverage".

**Implementation.** ~80 lines in auto_k.py next to
`select_k_penalized_objective` (shares the penalty builder from IDEA-7).
Diagnostics: `k, objective, log_post, post, in_hpd, selected` + extras
`{"posterior_level", "hpd_lo", "hpd_hi", "p_zero", "entropy"}`.

**Tests.** Unit: normalization; MAP = EBIC argmax; strong-signal toy → HPD
width ≤ 3; null toy → `P(k=0) > 0.5`. Sims: coverage measurement wired into
the harness output (`notes` column).

**Gates.** Piggybacks IDEA-7's gates; additionally the coverage bar above
decides how it's documented, not whether it merges.

---

### IDEA-9 — `stability`: bootstrap stability curve for k

**What.** Choose k where the selected *set* stops being reproducible under
data perturbation. Signals are stable across bootstrap reweightings; the
noise tail reshuffles every draw. This is the only idea whose target is
*reliability of the feature list itself* — often what practitioners actually
want from k — and it doubles as per-feature confidence output
(`π_j` = selection frequency), a UX win independent of k.

**Math.** B replicates. `boot_mode="bayes"`: replicate weights
`w⁽ᵇ⁾ = w ∘ E⁽ᵇ⁾`, `E⁽ᵇ⁾_i ~ Exp(1)` iid (Bayesian bootstrap — keeps every
row, but still requires replicate-local centering/scaling before correlation
formation);
`"half"`: uniform half-sampling without replacement (sets complementary-half
weights to 0). For each replicate: recompute `r⁽ᵇ⁾` and the panel
correlation matrix under `w⁽ᵇ⁾` by restandardizing the fixed marginal-rank
`Z` under replicate weights (Z held fixed — the marginal transform is
frozen; document as an approximation, same class as xfit shared_z),
re-screen, re-prune, rerun the greedy → path⁽ᵇ⁾. Selection frequencies
`π_j(k) = (1/B)·Σ_b 1{j ∈ prefix_k(path⁽ᵇ⁾)}`. Chance-corrected stability
(Nogueira, Sechidis & Brown 2018):

```
Φ(k) = 1 − mean_j [ B/(B−1) · π_j(k)·(1−π_j(k)) ]  /  [ (k/p)·(1 − k/p) ]
```

(j ranges over all p_valid features; k̄ = k exactly since every replicate
selects k features). At the boundaries k=0 or k=p the denominator is zero;
define `Φ(k)=1.0` when all replicate prefixes are identically empty/full, and
otherwise skip the boundary from curve-based maximization. Rules:
`"max_one_se"` (default): k̂ = largest k with
`Φ(k) ≥ Φ(k_max*) − se(Φ(k_max*))`, se via jackknife over replicates; if
`max_k Φ(k) < 0.5`, return the zero-capable floor with
`stopped_by="stability_floor"` because chance-corrected agreement below 0.5
is mostly chance-level;
`"pi_threshold"`: k̂ = #{j : π_j(max_k) ≥ stability_pi} (Meinshausen–
Bühlmann flavor; different semantics and can overselect under same-data
bootstraps, exposed for power users).

**Implementation.** `auto_k_resample.py` (shares replicate-loop scaffolding
with IDEA-3):

```python
def bootstrap_paths(cache, y, *, B, max_k, boot_mode, top_m, corr_prune,
                    random_state) -> list[np.ndarray]
def select_k_stability(paths, p_valid, config) -> tuple[int, pd.DataFrame]
```

Diagnostics: `k, phi, phi_se, mean_jaccard, selected` + a second per-feature
frame in the summary extras (`pi_at_k_hat` as a dict or Series) — per-feature
frequencies are the user-facing gold here.

**Cost.** B × (panel corr recompute + greedy). The panel correlation under
new weights is the expensive bit, and `cache.Rxx` does **not** help here
(the weights change per replicate): each replicate pays one
`top_m² × n_rows` weighted correlation. B=30, top_m=500, n=20k is
seconds-to-a-minute territory — state the honest cost table in the
docstring, and keep D9 within budget via the cache's row subsample.

**Tests.** Unit: Φ formula against the Nogueira paper's worked example
(their Example 1 table); perfect-stability toy (identical paths → Φ=1);
random-paths toy (Φ ≈ 0). Sims: D1: Φ(k) peaks within [k*−2, k*+2]. On D3
(redundant blocks), raw Φ is *expected* to dip from within-block member
swapping even at the default prune=0.95 — record the number rather than
asserting against it, and assert instead that selection frequencies
collapsed to block level (π of a block = frequency any member is in the
prefix) recover 8 stable blocks. If the raw-Φ dip breaks G1 on D3, that is a
finding about the method's scope, not a fix-blocker: `corr_prune` upstream is
the mitigation, and D3 exists to measure how well it works.

**Gates.** G1 (D1, D2, D7 primarily; D3 informational per above), G5
(B-multiple), G6 (this method should *win* G6 — that's its thesis).

---

### IDEA-10 — `changepoint`: noise-floor changepoint on the gain path (elbow 2.0)

**What.** A drop-in replacement for `select_k_elbow` with the arbitrary
`min_rel_gain` replaced by a floor estimated from the path's own noise tail
and cross-checked against the analytic max-of-χ² prediction. Zero extra
runs; strictly better calibrated than the current elbow; the natural
fallback default when no time/groups/folds are available and even B=20
permutations is too much.

**Math.** Work on `x_t = log(objective_scale·g_t + 1e−12)`, t ≤ L (path run
to a generous max_k — the tail must contain noise; see guard). For Gaussian
CEFS+, `objective_scale = n_eff`; for binary CEFS+ log-likelihood/score-test
gains, `objective_scale = 2.0` by Wilks' theorem, mirroring
`select_k_penalized_objective`. Tail window W =
`min(L - 1, max(10, ceil(floor_window·L)))`; if `L < 3` or the bounded tail
window leaves no pre-tail evaluation range, warn and fall back to the method
floor/effective max rather than returning an accidental empty curve.
Estimate `μ̂ = median(x_W)`, `σ̂ = 1.4826·MAD(x_W)`. Analytic cross-check:
under the null the tail gains are maxima over m_t survivors, so their median
is computed stably as
`x_med(m) = log(chi2.isf(-expm1(log(0.5)/m), df=1))`; if
`μ̂ > x_med(m_tail) + 3σ̂`, the tail is *not* noise (signal extends past
max_k): warn `floor_not_reached`, return effective max with the standard
saturation flags. Otherwise:

```
threshold = μ̂ + floor_z·σ̂
k̂ = last t ≤ L − |W| with x_t > threshold   (0/min_k if none)
```

with an optional 3-point median smoothing of x_t before thresholding
(`changepoint_smooth`, default on) to stop a single heavy-tail null spike
from extending k̂. Patience is implicit in "last exceedance"; `stop_patience`
is reused as the smoothing half-width if set > 2.

**Implementation.** `select_k_changepoint(objective_path, config, *,
objective_scale, n_eff, p_candidates)` in `auto_k_stop.py` (~70 lines).
Diagnostics: `k, gain, log_scaled_gain, objective_scale, floor_mu,
floor_sigma, analytic_floor_median, threshold, exceeds, selected` + extras
`{"floor_not_reached": bool, "floor_z", ...}`. Gaussian orchestrators pass
`objective_scale=n_eff`; binary orchestrators pass `objective_scale=2.0`.
Deprecation note (docs only): position as the recommended replacement for
`elbow`; keep `elbow` untouched.

**Tests.** Unit: crafted gain sequences (10 signal gains at n·g≈50 then 90
null gains sampled from max-of-m χ²(1)) → k̂ = 10 across 20 seeds;
floor_not_reached guard trips when "nulls" are actually signals; smoothing
kills a single injected spike at t=60. Sims: D1/D2/D6 — beats `elbow`
defaults on |k̂−k_oracle| (this is the direct A/B the idea exists for).

**Gates.** G1 (vs `elbow` specifically — must dominate it to merge), G5
(trivially), G6.

---

### IDEA-11 (bonus) — `consensus`: median-of-methods with disagreement diagnostics

**What.** Once ≥3 of the above exist: run a configured subset (default
`("ebic", "chi2_stop", "perm_gap", "gaussian_cv")`), share the panel/path
across them, return the median k̂ and a per-method table. Two claims worth
buying: the median is robust to any single method's blind spot (each idea
above has a documented one), and the *spread* is the missing UX signal —
`spread = max(k̂) / max(1, min(k̂))`; if spread > 2, warn that k is
ill-determined and parsimony (`smallest`) semantics are advisable.

**Implementation.** Orchestrator-level only (`filter_auto_k.py`): call the
per-method `select_k_*` functions on shared inputs; methods needing context
the caller didn't supply (e.g. no time/groups for fold methods → use
`strategy="kfold"`) participate only if runnable — record participation in
diagnostics. k̂ = median rounded to the nearest evaluated k, ties → smaller.
Diagnostics: one row per method (`method, k_hat, runtime_s, note`) + extras
`{"consensus_spread", "consensus_n_methods"}`.

**Tests.** Unit: median/tie logic; partial-participation paths. Harness:
report as its own method row; the claim to verify is **min-max robustness**:
consensus's *worst* design-level mean regret_frac across D1–D8 ≤ every
individual method's worst. If that fails, consensus is dropped without
prejudice.

---

## E. Sequencing

Ordered by information-per-effort; each PR includes its tests + DOCS.MD +
user-guide updates. Run the harness and append results to
`benchmarks/README.md` at every step.

1. **PR-0 — Phase 0 harness + §C.1 CandidatePanel refactor.** Record
   baselines (elbow, bic, evaluate/best, evaluate/one_se, fixed_50, oracle)
   on D1–D9. This table alone will clarify how bad the status quo is and
   where.
2. **PR-1 — IDEA-7 (EBIC/RIC) + IDEA-8 (posterior).** Tiny, shared code,
   immediate expected win on D7/D1; establishes the per-k penalty plumbing.
3. **PR-2 — IDEA-1 + IDEA-2 + IDEA-10** (one PR: they share
   `path_gain_pvalues` and the run/patience logic in `auto_k_stop.py`).
4. **PR-3 — IDEA-3 (perm_gap)** — introduces the structure-aware null
   utilities; cross-validates PR-2's analytic p-values (add a diagnostic
   comparing analytic vs permutation p at matched steps on D6; large gaps =
   the copula approximation is hot, prefer perm_gap defaults).
5. **PR-4 — IDEA-5 + IDEA-6** (one PR: shared fold plumbing in
   `auto_k_xfit.py`). After this lands, decide the recommended default from
   the harness table (leading candidates: `gaussian_cv/one_se` or
   `ebic/auto`).
6. **PR-5 — IDEA-4 (knockoff_path).** Depends on nothing above except the
   panel; sequenced late only because select_fdr already covers part of this
   ground.
7. **PR-6 — IDEA-9 (stability).**
8. **PR-7 — IDEA-11 (consensus) + docs finalization:** the "which auto-k
   should I use" decision table in the user guide, keyed by the final
   harness numbers, plus release-notes entry.

Kill/keep is per-idea via §B.5 gates; nothing here is all-or-nothing.

## F. References

- Chen & Chen (2008), *Extended Bayesian information criteria for model
  selection with large model spaces*, Biometrika — EBIC, γ-consistency.
- Foster & George (1994), *The risk inflation criterion for multiple
  regression*, Ann. Statist. — RIC (2·log p).
- G'Sell, Wager, Chouldechova, Tibshirani (2016), *Sequential selection
  procedures and false discovery rate control*, JRSS-B — ForwardStop.
- Barber & Candès (2015), *Controlling the false discovery rate via
  knockoffs*, Ann. Statist. — SeqStep+/knockoff counting (Thm 3).
- Tibshirani, Walther, Hastie (2001), *Estimating the number of clusters via
  the gap statistic*, JRSS-B — null-reference envelope logic.
- Li & Ji (2005), *Adjusting multiple testing in multilocus analyses*,
  Heredity — effective number of tests.
- Nogueira, Sechidis, Brown (2018), *On the stability of feature selection
  algorithms*, JMLR — chance-corrected stability index.
- Meinshausen & Bühlmann (2010), *Stability selection*, JRSS-B — π-threshold
  semantics.
- Scott & Berger (2010), *Bayes and empirical-Bayes multiplicity adjustment*,
  Ann. Statist. — model-size prior (≡ EBIC γ=1).
- Anderson (2003), *An Introduction to Multivariate Statistical Analysis* —
  partial correlation null distribution (the F(1, n−s−2) identity).
