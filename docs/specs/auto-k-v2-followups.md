# Auto-K v2 — review findings and follow-up work

Status: action list (supervisor review of the initial Auto-K v2 implementation)
Companions: [auto-k-v2.md](auto-k-v2.md), [knockoffs-followups.md](knockoffs-followups.md)

---

## Status after the second follow-up pass (supervisor re-review, final)

All parts are now complete and independently verified (suite: 710 passed /
12 skipped; program gates recomputed from the raw CSVs and matching the
recorded tables; D1 campaign rows reproduced bit-exactly via the CLI;
FIX-2b gate rerun: stability signal k̂ = 10 on 5/5 seeds, null k̂ = 0 on 5/5;
zero-config router verified end-to-end incl. binary and the weight-skew /
p>n_eff branches).

- **Part 5 done.** `penalized/ebic` is the measured default — confirmed, and
  understated by the campaign summary: it is the **only** v2 method passing
  G1 on all of D1/D2/D3/D7 (gaussian_cv fails D3 G1: median |k̂−k_oracle| 7.5
  vs baseline 2.0, regret 0.0146; chi2_stop/consensus also miss D3's
  median-k bar). knockoff_path / changepoint / stability / xfit_objective
  failed-gate labels all verified from raw data; stability's residual D3/D7
  failure is the spec-predicted block-swapping scope limit, not a bug.
- **Part 6 done.** Router routes no-config CEFS+ (and binary) `k="auto"` to
  EBIC with `auto_routing` diagnostics and an EBIC fallback on degenerate
  primaries.
- **FIX-2b done** (Φ floor 0.5, `stopped_by="stability_floor"`); Part 2 and
  Part 3 remainders done (D4 β=0.053, CatBoost transfer run recorded,
  `--quick` wired, U(0,1) KS test, EBIC/RIC arithmetic + γ-monotonicity,
  tibshirani knee, digamma drift, Nogueira Φ + jackknife, SeqStep+ hand
  computation, calibration/recovery sims, negative control).

Closeout after this review:

1. Release notes now call out the behavior change: no-config `k="auto"` on
   CEFS+ with `time`/`groups` previously ran `evaluate/time_holdout|group_cv`;
   it now routes to EBIC, and the previously-raising no-context case now works.
2. DOCS.MD, docs/API.md, and docs/ADVANCED.md now document that the router uses
   method-specific effective floors (0 for ebic/perm_gap, >=1 for gaussian_cv);
   users needing a hard floor should set an explicit `k_method`.
3. Real-data sanity check on the WNBA DPM script
   `/Users/kmedved/Dropbox/github/wnba_darko/pipeline_scripts/models/32_dpm.py`
   was run but is **censored, not completed**: CEFS+ `k="auto"` routed to
   EBIC and hit the script's `auto_max_k=125` cap on all eight targets
   (`selected_at_effective_max_k=True`). This is a lower-bound observation
   ("≥125 features carry EBIC-detectable signal"), consistent with the
   production fixed settings of ~300–360, and not evidence for or against
   the default. The DPM feature surface is a dense-signal regime (the D4
   analog, where every support-recovery method underselects by design and
   the campaign's oracle wanted essentially all real features). The decisive
   real-data experiment is still open — see "R2 closeout experiment" below.

### R2 closeout experiment (real-data validation, completed)

Run read-only against the WNBA DPM script on all eight CEFS+ targets. Outputs
were written to `/private/tmp/wnba_dpm_autok_method_sweep_final.csv`,
`/private/tmp/wnba_dpm_perm_gap_b20.csv`, and
`/private/tmp/wnba_dpm_r2_risk_summary.csv`.

1. **Uncapped EBIC stopped internally, but dense.** On/off context EBIC picked
   409/502/489/418; strict-box EBIC picked 234/234/214/221. None of the
   full-path runs selected the effective max, so the earlier 125 result was
   purely a cap-censored lower bound.
2. **Triangulation split by question.** Support/detectability methods stayed
   high: RIC ≈ 212–409, chi2_stop ≈ 191–346, forward_stop ≈ 227–430, posterior
   MAP ≈ EBIC. Predictive/null methods were much smaller: gaussian_cv/best
   picked 5–109, gaussian_cv/one_se picked 1–52, and perm_gap B=20 picked
   3–18. Consensus spread was >2x on every target, which correctly says k is
   ill-determined unless the production question is specified.
3. **Ridge-proxy risk curve resolved the production question.** A 5-fold
   GroupKFold ridge proxy on CEFS+ prefixes gave:

   | surface | target | EBIC k | risk-best k | one-SE k |
   |---|---:|---:|---:|---:|
   | on/off context | o_rapm | 409 | 150 | 30 |
   | on/off context | d_rapm | 502 | 100 | 40 |
   | on/off context | elo_o | 489 | 150 | 75 |
   | on/off context | elo_d | 418 | 75 | 50 |
   | strict box | elo_o | 234 | 100 | 100 |
   | strict box | elo_d | 234 | 40 | 40 |
   | strict box | o_rapm | 214 | 75 | 75 |
   | strict box | d_rapm | 221 | 100 | 50 |

   The risk curve is flat after an early knee and often worsens by the EBIC
   prefix. For DPM, EBIC should be read as "count of detectable features";
   predictive sufficiency is closer to the grouped risk curve / gaussian_cv
   family.
4. **Decision.** EBIC stays the zero-config CEFS+ default because it won the
   benchmark gates and remains null-honest, free, and binary-compatible. For
   dense-signal production domains like DPM, docs should recommend
   `gaussian_cv` / explicit prefix-risk scoring and should frame EBIC as a
   support-detectability criterion rather than a downstream model-size answer.

Product fix motivated by this incident is implemented: the `k="auto"` router
raises a `UserWarning` when `selected_at_effective_max_k=True` and records
`saturated=True` in `auto_routing`. A silently-capped zero-config answer is how
this censored run got mistaken for a completed validation.

Remaining reading note (no code blocker):

1. When reading the D9 table, note "selection runtime" excludes the shared
   path/cache build; gaussian_cv's G5 FAIL is per the letter of the gate
   (vs 0.5x evaluate) - its 2.1s absolute cost at 50kx2000 is small next to
   the ~15s cache build. Docs should not over-read it as "slow".

---

## Part 7 — dense-regime followups (from scoring every method against the DPM risk curve)

Every method's DPM pick was scored against the measured grouped-ridge risk
curve (regret as % of achievable signal, mean over the 8 targets / max):

| approach | mean regret | max |
|---|---:|---:|
| risk-curve one-SE pick (the instrument itself) | 0.5% | 1.3% |
| `gaussian_cv` with `selection_rule="best"` | 2.9% | 7.1% |
| `chi2_stop` | 4.4% | 8.5% |
| `penalized/ebic` (zero-config default) | 6.6% | 12.8% |
| production fixed k=300–360 | 7.0% | 17.6% |
| `gaussian_cv` with `one_se` (current default rule) | 8.6% | 15.6% |
| `perm_gap` B=20 (tibshirani knee) | 15.3% | 49.2% |

Readings: (a) the zero-config default already matches/beats hand-tuned
production on this data; (b) the selection *rule* dominates on dense data —
same gaussian_cv curves, `best` 2.9% vs `one_se` 8.6%, an inversion of the
sparse-design result (fold SEs are large relative to the shallow dense
slope, so one-SE cuts far past the knee); (c) perm_gap's knee rule is not a
sufficiency method on dense data — its 15.3% should retire it from any
dense-regime advice.

**FIX-7.1 — doc bug (do first).** The dense-regime guidance added in
`c3ff024` ("use `gaussian_cv` …") currently makes things *worse* if followed
with defaults: gaussian_cv defaults to `one_se` (8.6%) which underperforms
the ebic default (6.6%) it advises away from. Amend DOCS.MD / API.md /
ADVANCED.md: dense-regime advice is `gaussian_cv` **with
`selection_rule="best"`** or an explicit raw-space prefix-risk curve; keep
`one_se` advice for sparse regimes.

**FIX-7.2 — benchmark the `gaussian_cv/best` variant** before promoting it in
docs: add it as a method row to the campaign (D1–D8; it shares curves with
gaussian_cv, only the rule differs). Accept if it stays within G1/G6 bars on
sparse designs and beats `one_se` on D4; then FIX-7.1's wording is
data-backed on both regimes.

**FIX-7.3 — router disagreement diagnostic.** When the routed EBIC pick is
large (k̂ ≥ 100 or k̂ ≥ 0.25·max_k), optionally cross-check with
`gaussian_cv/best` (~10 s at 89k×713, measured) and warn when the two
disagree by >2×: "dense-signal regime: EBIC counts detectable features
(k=X); for downstream sizing consider gaussian_cv/best or a prefix-risk
curve (k≈Y)". Off by default if cost is a concern (`auto_dense_check` flag);
the warning text is the product.

**FIX-7.4 — investigate the copula-proxy parsimony gap.** gaussian_cv/one_se
picked 3–27 where the raw-space one-SE point is 30–100. Hypotheses: the
rank-Gauss transform compresses heavy-tailed count features (this surface is
skewed box-score data — the D6 regime), and/or fixed `xfit_ridge=1e-3` vs
RidgeCV alpha selection. Compare the copula-proxy curve to the raw-space
curve on one DPM target; if the proxy knee is systematically early on
heavy-tailed features, document it as a known limitation.

**FIX-7.5 — dense design at scale.** D4 predicted all of this but sits
outside the program bar and at n=5000. Add a D4-scaled design (n≈90k,
p≈700, dense weak signal, groups) and a gate on it, so future default
decisions are accountable to the regime the user's production data actually
lives in.

**Status after Part 7 implementation.** Done with one important scope note:
`gaussian_cv/best` is a dense-regime recommendation, not a new zero-config
default. DOCS.MD / API.md / ADVANCED.md now say dense weak-signal sufficiency
requires `gaussian_cv` with `selection_rule="best"` or an explicit prefix-risk
curve; one-SE remains the sparse/parsimony rule. The benchmark harness accepts
`gaussian_cv/best` and strategy suffixes such as `gaussian_cv/best/group_cv`.
The D1-D8 campaign row is recorded in
`benchmarks/results/auto_k_v2_main.csv` and summarized in
`auto_k_v2_summary.csv` / `auto_k_v2_gates.csv`: program mean regret
0.001022, D5 max k=3, D4 regret 0.06237 (vs gaussian_cv/one_se 0.22402),
and D3 median |k̂-k_oracle| improves from 7.5 to 4.0 but still misses the
strict G1 comparison against the best baseline (2.0). Therefore EBIC remains
the measured zero-config default.

The router has an opt-in dense diagnostic:
`AutoKConfig(k_method="auto", auto_dense_check=True)`. When EBIC returns a
large k (default: k>=100 or k>=0.25*effective_max_k), the router runs
`gaussian_cv/best` and warns if the answers differ by more than 2x. The warning
is intentionally phrased as a question split: EBIC counts detectable features;
Gaussian CV / prefix-risk curves are downstream-size diagnostics. The routing
metadata records the dense check, EBIC k, Gaussian-CV k, ratio, and whether a
warning fired.

D10 was added as the production-scale dense benchmark design: full mode is
n=90k, p=700, 685 groups, 220 weak dense signals, with a 600-feature path cap.
The recorded D10 full slice (2 seeds) is in
`benchmarks/results/auto_k_v2_d10_full.csv`: gaussian_cv/best/group_cv mean
regret 0.00538, gaussian_cv/best 0.00766, EBIC 0.01948, gaussian_cv/one_se
0.02377. This synthetic surface is less extreme than WNBA DPM (EBIC does not
overshoot by hundreds), but it does keep the dense-rule decision accountable:
best beats one-SE, and grouped best is strongest.

Copula-proxy parsimony gap: the durable evidence is the DPM point comparison,
not a full saved copula curve. On the real DPM surface, gaussian_cv/best picked
5-109 while raw grouped-ridge risk-best was 40-150 and raw one-SE was 30-100;
gaussian_cv/one_se was smaller still (1-52). The most plausible mechanism is
the intended rank-Gauss copula compression plus a fixed light ridge in a
heavy-tailed / zero-inflated count surface; use the explicit raw-space
prefix-risk curve when production model sizing matters. A future deeper
diagnostic should persist the full Gaussian CV fold curve for one DPM target
alongside the raw risk curve, but the production guidance no longer depends on
that missing artifact.

---

## Status after the first follow-up pass (supervisor re-review)

Verified against the working tree after the second follow-up pass (focused
suite: 46 auto-k v2 tests passing; FIX-2b gate re-run):

| Item | Status |
|---|---|
| Part 1 FIX-1 (changepoint demotion) | **Done** (docs-only, as specified; behavior unchanged, confirmed). |
| Part 1 FIX-2/FIX-2b (stability) | **Done.** Jackknife SE, sorted `pi_at_k_hat`, largest-k `max_one_se`, default restored to `max_one_se`, and absolute Phi floor diagnostics landed. Gate re-run: D1 k̂ = 10/10/10/10/10; D5 k̂ = 1/1/1/1/1. |
| Part 1 FIX-3 (xfit one_se + null guard) | **Done and passes its gate** (signal k̂ = 10 on 5/5 seeds; null k̂ ≤ 2 on 4/5). |
| Part 1 FIX-4 (consensus warning, degenerate folds, Cholesky, HPD docs) | **Done.** Cholesky leading-block factorization verified correct; equivalence test present. No regressions in gaussian_cv / consensus / chi2_stop (re-run: signal 10/10/10, null ≤1). |
| Part 2 harness | **Done for this pass.** `evaluate/one_se` has singleton-group 5-fold GroupKFold with `synthetic_group_cv`; D6 transforms are strictly monotone; D4 β is 0.053; optional guarded `--model catboost` risk scoring is wired; `k_dispersion_group`, exact off-grid k̂ risk, `--quick`, and D9 `--full` sizing are pinned. |
| Part 3 test debt | **Done for this pass.** Added exact-null U(0,1) KS, residual partial-correlation path checks, null/recovery sims, gaussian_cv holdout-MSE ground truth, xfit digamma debias validation, EBIC/RIC arithmetic, Nogueira Φ ground truth, knockoff SeqStep+/pair-table arithmetic, forward_stop hand-built p-sequences, Tibshirani gap knee, changepoint floor-not-reached, and D8 within-group negative control. |
| Part 4 docs | **Done** (verified: approximate/q-calibrated wording, `bic` pattern reverted, experimental caveats, config-field tables in DOCS.MD + API.md, troubleshooting heading, architecture entry, `Unreleased` release-notes section, dead test assertion fixed). |
| Part 5 benchmark campaign | **Done.** Main D1-D8 sweep (4,320 rows), D5 null calibration, D9 `--full` timing, and guarded CatBoost transfer are recorded under `benchmarks/results/`; summary and gate tables are appended to `benchmarks/README.md`. |
| Part 6 `k_method="auto"` router | **Done.** CEFS+ and binary CEFS+ no-config `k="auto"` route to measured-default EBIC; explicit `AutoKConfig(k_method="auto")` also records deterministic router metadata and fallback details. |

### FIX-2b — stability needs an absolute-Φ null guard (do this next)

Measured state after FIX-2 (same D1-like/null smoke, 5 seeds):
`pi_threshold` (current default) → signal k̂ = 23–30, null k̂ = 18–25;
`max_one_se` (largest-k) → signal k̂ = 10, 10, 10, 10, 10, null k̂ = 40–60.
Root cause is structural: bootstrap reweighting of the *same* dataset keeps
the same spuriously-correlated features "stable", so selection frequency and
Φ are not null-calibrated quantities — on a pure null, ~20 features still
exceed π ≥ 0.6 at a depth-60 prefix.

The regimes are cleanly separable on the Φ scale itself (measured): on
signal, max Φ = 1.000; on a pure null, max Φ ≈ 0.17 across all k. So:

1. Add an absolute stability floor: if `max_k Φ(k) < 0.5`, return the
   zero-capable floor (0 when `min_k = 0`) with
   `stopped_by="stability_floor"` and the max Φ in the extras. Apply the
   guard to **both** rules. (Φ is chance-corrected; below 0.5, most observed
   agreement is chance-level — document that sentence.)
2. With the guard in place, switch the default `stability_rule` back to
   `"max_one_se"` (largest-k semantics): it is exactly right on signal and
   the guard closes its null hole. Keep `pi_threshold` as an option and
   document its overselection failure mode on same-data bootstraps.
3. Gate before the campaign (rerun the 5-seed smoke): signal k̂ ∈ [6, 14]
   majority, null k̂ ≤ 3 in ≥ 4/5 seeds. Add a small seeded test pinning the
   guard (a null Φ curve → floor; the crafted stable-paths toy → k*).

Implementation note: this was completed in the second follow-up pass. The
blocking smoke was run with
`python benchmarks/bench_auto_k.py --designs D1,D5 --methods stability --seeds 5 --n-test 1000 --output /private/tmp/fix2b-stability-gate.csv`;
results were D1 `[10, 10, 10, 10, 10]` and D5 `[1, 1, 1, 1, 1]`.

---

This document records the supervisor review of the Auto-K v2 implementation
(all 11 methods + consensus, panel refactor, harness, tests, docs — currently
uncommitted in the working tree) and specifies the follow-up work. Review
inputs: line-by-line reads of every new/changed module against
[auto-k-v2.md](auto-k-v2.md), an independent review of the benchmark harness
and test suite, a docs accuracy/tone audit, and empirical smoke runs of every
method through the public API.

**Headline verdict.** The plumbing is genuinely good: dispatch, config
validation, zero-selection semantics, panel refactor (with a byte-identity
regression test), exports, and diagnostics all match the spec, and the full
suite passes (684 passed / 12 skipped). The math of the analytic family
(chi2_stop, forward_stop, EBIC/RIC, posterior, xfit debias, gaussian_cv risk
form, knockoff SeqStep+) was re-derived and matches the spec. But **three
methods are empirically broken or degenerate** (changepoint, stability, and
xfit_objective's default rule), the test suite validates plumbing rather than
statistics (every calibration/recovery sim from the spec is missing), the
harness has one design bug and one degenerate baseline, and the docs bless
unbenchmarked methods. None of this is visible from "tests pass" — which is
exactly why the harness exists.

Empirical smoke evidence (D1-like: n=2000, p=200, k*=10, β=linspace(1.5,0.5);
null: same X, pure-noise y; 5 seeds; `min_k=0`, `max_k=60`):

| method | k̂ on signal (want ≈10) | k̂ on null (want ≈0) |
|---|---|---|
| elbow (baseline) | 11 | 2 |
| penalized/bic (baseline) | 11 | 2 |
| penalized/ebic | 11 | 0 |
| penalized/ric | 11 | 0 |
| chi2_stop | 10 | 0 |
| forward_stop | 11 | 0 |
| k_posterior | 11 | 0 |
| perm_gap | 11 | 0 |
| knockoff_path | 11 | 0 |
| gaussian_cv (one_se) | 10 | 1 (floor) |
| consensus | 10 | 0 |
| **changepoint** | **47, 44, 48, 37, 47** | **48, 48, 35, 47, 38** |
| **stability** | **1, 2, 1, 3, 2** | **35, 28, 18, 27, 2** |
| **xfit_objective (best)** | **20, 47, 28, 49, 12** | **7, 54, 1, 60, 50** |
| xfit_objective (one_se) | 10, 10, 10, 10, 10 | 1, 1, 1, 21, 47 |

Nine of twelve methods work essentially as designed on first contact. The
three failures below are the priority.

---

## Part 1 — Method fixes (do these before the benchmark campaign)

### FIX-1 — `changepoint` is structurally miscalibrated (spec-level flaw)

**Evidence.** On pure-noise data it selects k̂≈35–48 every seed. Diagnostics
show why: on a greedy path, null gains are **not** a stationary noise floor.
The greedy consumes the largest spurious correlations first, so null
log-gains decline smoothly (observed: 2.44 at k=1 down to 0.04 at k=60 on a
null). The tail window estimates its "floor" from the *bottom of that slope*
(μ̂=0.09, σ̂=0.02), so every earlier null step exceeds μ̂ + 2.5σ̂ and the
"last exceedance" rule returns ≈ the whole path. The spec's premise in
IDEA-10 ("under the null the tail gains are draws from the same
max-over-remaining-candidates distribution … goes flat") is wrong for greedy
selection-order statistics; this is not an implementation bug (the
implementation follows the spec, minor deviations aside).

**Also observed:** the analytic cross-check median (`x_med(m_tail)`≈2.07) sits
far *above* the realized tail (0.09) for the same reason — the analytic
max-of-m null describes step-1 gains, not step-50 gains — so
`floor_not_reached` can never fire in the intended way either.

**Action.** Do **not** try to rescue the self-estimated-floor design; any
correct per-step null reference for greedy gains is what `chi2_stop` already
computes analytically. Instead:

1. Keep the code and config surface, but mark `changepoint` **failed-gate /
   diagnostic-only** in DOCS.MD, ADVANCED.md, and the decision table, with one
   sentence of the explanation above. Negative results are results; record its
   rows in the benchmark table like everyone else.
2. Point users who wanted "elbow 2.0" at `chi2_stop` (which is precisely the
   calibrated elbow and passes the null smoke at α=0.05).
3. Remove `changepoint` from any candidate-default consideration and from
   `consensus_methods` examples in docs (it is not in the default tuple —
   keep it that way).

### FIX-2 — `stability` has a real bug and a degenerate default rule

**Bug (code):** `phi_se` in `select_k_stability`
([auto_k_resample.py](../../sift/selection/auto_k_resample.py) ~line 286) is
computed as the running standard deviation of Φ across *k values seen so far*
(`np.nanstd(phi_values)` where `phi_values` accumulates over the k loop). The
spec requires the SE of Φ(k) at fixed k via **jackknife over the B
replicates**. The current quantity grows with k and has no sampling
interpretation; it is the tolerance used by the default `max_one_se` rule, so
k̂ is effectively arbitrary. Implement the jackknife: for each k, recompute Φ
leaving out one replicate at a time (π recomputable from counts in O(p) per
leave-out), `se = sqrt((B-1)/B · Σ_b (Φ_(b) − Φ̄_(·))²)`.

**Design flaw (spec-level):** even with a correct SE, `max_one_se` ("smallest
k with Φ(k) ≥ Φ(k*) − se") is degenerate on strong-signal data: every prefix
of a stable ordering is itself stable, so Φ(k) ≈ 1 for **all** k ≤ k* and the
rule collapses to k̂ ≈ 1 (observed: 1–3 on signal). The rule's parsimony
semantics answer "smallest stable prefix", not "how many features are
reliably selected".

**Action.**

1. Fix `phi_se` (jackknife) regardless of anything else.
2. Change the default `stability_rule` to `"pi_threshold"`
   (Meinshausen–Bühlmann: k̂ = #{j : π_j(max_k) ≥ stability_pi}), which
   answers the question users actually ask of this method and does not have
   the plateau degeneracy. Keep `max_one_se` available but re-specify it as
   the **largest** k on the Φ plateau (largest k with Φ(k) ≥ Φ(k*) − se(k*)),
   and document the semantics difference.
3. Also fix: the `pi_at_k_hat` summary extra in
   `select_gaussian_stability_path`
   ([filter_auto_k.py](../../sift/selection/filter_auto_k.py)) takes
   `np.flatnonzero(freq > 0)[:100]` — the first 100 by *column index*, not the
   top 100 by frequency. Sort by frequency descending before truncating.
4. Re-run the smoke above; gate: signal k̂ ∈ [6, 14] majority of seeds, null
   k̂ ≤ 3 majority of seeds, before it goes to the full benchmark.

### FIX-3 — `xfit_objective` default rule argmaxes a plateau; add a null guard

**Evidence.** With `selection_rule="best"` (the current default path), k̂ on
signal is 12–49 across seeds and 1–60 on null: after debiasing, the curve is
flat past k*, and argmax-of-flat-noise is a coin flip — the exact §A.4
failure mode the method was meant to escape. With `one_se` the signal side is
perfect (10 on 5/5 seeds) but the null still fires occasionally (21, 47 in
2/5 seeds) because on a no-signal curve "within one SE of the best noise
bump" is still noise.

**Action.**

1. Default `xfit_objective` to `selection_rule="one_se"` (document; the
   generic `AutoKConfig` default of `best` stays, so implement as a
   method-local effective default with a config override, mirroring how
   evaluate resolves rules — do not silently mutate shared config).
2. Add a **global null guard** before rule application: the debiased curve
   has D(0) = 0 by construction; if `max_k D(k) ≤ z · se(k_argmax)` (z = 2 by
   default, reuse `floor_z`? no — hardcode 2 and note it; `floor_z` belongs to
   changepoint) then return the zero-capable floor (0 when `min_k=0`) with
   `stopped_by="null_guard"` in the summary extras. This makes the method
   empty-capable and honest on nulls, consistent with §C.2's empty-selection
   contract.
3. Re-run the smoke; gate: signal 10±2 majority, null ≤ 2 in ≥4/5 seeds.

### FIX-4 — smaller correctness/robustness items

1. **Consensus false unused-field warning.** `validate_auto_k_config` warns
   when e.g. `alpha` is non-default while `k_method="consensus"`, but
   consensus sub-methods consume `alpha`/`perm_B`/etc. In
   `_warn_unused_method_fields`, when `k_method == "consensus"`, treat a field
   as used if any method in `config.consensus_methods` uses it.
2. **Degenerate folds kill the curve silently.** In
   [auto_k_xfit.py](../../sift/selection/auto_k_xfit.py),
   `_curve_from_fold_scores` takes `min(len(scores))` over folds, so one
   degenerate fold (empty scores) empties the whole curve and the orchestrator
   returns k=0 with no explanation. Drop degenerate folds with a `UserWarning`
   when ≥2 healthy folds remain; otherwise warn and fall back to the method
   floor with `stopped_by="degenerate_folds"`.
3. **`gaussian_cv` per-k solves are O(K⁴).** `_gaussian_cv_scores` calls
   `np.linalg.solve` from scratch for every prefix; the spec requires
   incremental (bordered) Cholesky, O(K³) total. Matters for G5 and for
   max_k ≥ 300. Implement the rank-1 border update (the codebase's standing
   pattern), keep the jitter-retry fallback, and add the
   "incremental ≡ per-k direct solves" equivalence test from the spec.
4. **`changepoint` dead branch:** `pre_tail <= 0` is unreachable
   (`tail_width ≤ effective_max − 1` guarantees `pre_tail ≥ 1`) — harmless,
   but remove or assert if touched during FIX-1's doc demotion.
5. **HPD floor semantics (design decision to confirm):** when `min_k > 0`,
   k=0 is excluded from the HPD set even when the posterior mass says
   otherwise; `p_zero` is still reported. Keep the behavior but document that
   the HPD is computed over *selectable* k, and that `p_zero` is the honest
   no-signal diagnostic. (An existing test pins this; make the docstring say
   it out loud.)

---

## Part 2 — Harness fixes (the yardstick must be right before we trust it)

All in [bench_auto_k.py](../../benchmarks/bench_auto_k.py) /
[auto_k_designs.py](../../benchmarks/auto_k_designs.py) unless noted.

1. **`evaluate/one_se` baseline is degenerate (HIGH).** It runs
   `strategy="time_holdout"` → one split → sift warns and silently falls back
   to `best`; verified: `one_se` ≡ `best` on every design/seed sampled, and
   the fallback is not recorded in the CSV. G3 compares against this baseline,
   so it is currently invalid. Fix: run the `evaluate/one_se` baseline with
   `strategy="kfold"` (n_splits=5) so an SE exists; additionally, record any
   rule fallback in the `notes` column.
2. **D6 is broken (HIGH).** `log1p(np.abs(x))` is an even function; on
   symmetric t(3) columns, 8 of 10 "signals" have zero correlation with y and
   are undetectable by any copula/correlation selector — D6 stops measuring
   copula robustness. Replace with strictly monotone transforms per the spec
   (e.g. `exp(x/2)` on standardized columns, `log(x)` on the lognormal
   columns, `x**3`), and keep tails bounded enough that Var(y) is finite
   (avoid exponentiating raw t(3); apply exp to a clipped or standardized
   version and note it).
3. **D4 is ~3.6× easier than spec'd.** β=0.12 gives ≈0.91% of Var(y) per
   signal vs the spec's ~0.25%. Set β so each signal contributes ~0.25%
   (β≈0.053 at the spec's variance bookkeeping — recompute exactly).
4. **`--model catboost` is `NotImplementedError`.** Implement the small
   fixed-config CatBoost risk-curve variant (guarded import) for D1–D3; it is
   the only GBM-transfer bridge (spec R2) and the user's features feed a GBM.
5. **`k_dispersion_group` is always empty.** Populate it in aggregation
   (std of k̂ across seeds per design×method) or compute it in the results
   summarizer that writes benchmarks/README.md — either way, G6 needs it.
6. **Exact k̂ risk.** rmse(k̂) is currently evaluated at the nearest grid
   point when k̂ > 30 is off-grid. Fit one extra RidgeCV at exactly k̂ per row
   instead (cost: one fit); keep the `notes` breadcrumb when the fallback is
   ever used.
7. **Cosmetics that will bite:** `--quick` is parsed but unused (wire or
   remove); D5's `regret_frac` is NaN by construction (denominator 0) — make
   the summarizer exclude D5 from regret means explicitly; D9 under `--full`
   should reach the spec's 50k×2000 and stay timing-only (2 seeds, no risk
   curve needed beyond what G5 requires); dead code in `_d6_test` (`old = ...;
   del old`).

---

## Part 3 — Test debt (the suite must catch a broken method)

The current `tests/test_auto_k_v2.py` runs in ~1s of its 60s budget and is
plumbing-only: of the spec's ~35 per-idea "Tests." bullets, ~21 are absent —
including **every** calibration sim and signal-recovery sim. The changepoint
and stability failures above would sail through the current suite. Add, in
priority order (all seeded, small-n, target < 45s total):

1. **Exact-null U(0,1) p-value test** (IDEA-1): iid Gaussian generated
   directly in copula space, n=500, single candidate, 200 seeds; KS test on
   `p⁽¹⁾_1` from the real `cefsplus_loop_with_objective` output (p > 0.01).
   Also a 3-step path where ν_t and m_t progressions are checked against
   hand-computed values (the existing formula test is circular — it re-derives
   with the implementation's own formula and only exercises step 1; replace or
   supplement it with regression-residual ground truth).
2. **Null-calibration sims** (the tests that define "error-controlled"):
   D5-style n=500, p=50, 5 seeds, `min_k=0` — chi2_stop k̂ ≤ 2 at α=0.05;
   perm_gap k̂ ≤ 1; knockoff_path k̂ = 0 in ≥ 80%; EBIC k̂ ≤ 2 in ≥ 90%.
   Signal-recovery sims: D1-style n=1000, p=50, 8 strong signals, 3 seeds —
   k̂ within stated per-idea bands (spec §D per-idea "Tests." blocks).
3. **gaussian_cv risk-formula ground truth**: brute-force
   `1 − 2β'r̃ + β'R̃β` via explicit lstsq on small locally standardized
   matrices; plus (with FIX-4.3) incremental-Cholesky ≡ direct solves; plus
   the >0.99 correlation check vs an actual holdout MSE of a linear model on
   rank-Gauss features (n=2000, one seed).
4. **xfit debias validation**: mean of `−log(1−ρ̂²)` over ≥2000 simulated
   null partials ≈ `digamma((ν+1)/2) − digamma(ν/2)` at a few ν values.
5. **Arithmetic units with external ground truth**: `_log_comb` vs
   `math.comb`; EBIC γ="auto" formula values; γ-monotonicity
   k̂(0) ≥ k̂(0.5) ≥ k̂(1); at least one `ric` test (currently zero);
   Nogueira Φ worked example (their Example 1) + identical-paths Φ=1 +
   random-paths Φ≈0 (after FIX-2 lands, also a jackknife-SE sanity test);
   knockoff pair-table construction on a crafted entry order and SeqStep+
   FDP̂ against hand computation; forward_stop on hand-built p-sequences at
   α ∈ {0.1, 0.2}; tibshirani gap rule on a synthetic curve with a known knee
   (the default rule is currently never exercised); changepoint
   `floor_not_reached` tripping case.
6. **D8 negative control** (small): grouped data where `null="permute"`
   overselects and `null="within_group"` does not — the spec calls this the
   test that documents why the option exists.
7. **Fix the actively-wrong test**: `test_auto_k_v2.py:574` — the set
   difference removes `corr_prune_disabled` before the subset check, so the
   assertion never tests it (dead expression). Assert the flag directly.

---

## Part 4 — Docs corrections (measured-tone pass)

1. Replace "FDR-controlled" for `knockoff_path` (DOCS.MD, ADVANCED.md) with
   the house phrasing already used for `select_fdr`: "q-calibrated
   **approximate** Gaussian-copula knockoff selection". The guarantee is
   plug-in approximate and *unmeasured for this method until the harness
   runs*.
2. Mark every v2 method **experimental — pending benchmark gates** in the
   DOCS.MD decision table and user-guide "first pass" paragraph. The table
   stays (it's useful), but hedged; per spec §E the final wording is keyed to
   harness numbers.
3. Revert the "Recommended patterns" block's silent `bic` → `ebic` switch
   until EBIC clears its gates (it likely will — but that's the point of
   gates).
4. Document the ~20 new `AutoKConfig` fields (at minimum: `alpha`,
   `knockoff_q`, `perm_B`/`perm_null`, `xfit_folds`/`xfit_ridge`,
   `ebic_gamma`, `boot_B`/`stability_pi`, `consensus_methods`, `n_eff_mode`)
   — the docs currently tell users to try `knockoff_path` without ever
   stating `knockoff_q=0.2` exists.
5. Restore exact raised-error text in the troubleshooting heading for the
   `k='auto' requires …` error (users grep for the message).
6. Add `sift/selection/panel.py` to the architecture module table and
   `auto_k_designs.py` to the benchmarks README script table.
7. Move the release-notes bullet out of the tagged `0.7.0` section into an
   `Unreleased` heading (v0.7.0 already points at `ff80d4c`).

---

## Part 5 — Benchmark campaign (after Parts 1–2 land)

Run and record (this is PR-0's promised baseline table plus the full method
table, in one campaign):

```bash
# Full sweep: all designs except D9, all methods + baselines, 30 seeds.
python benchmarks/bench_auto_k.py \
  --designs D1,D2,D3,D4,D5,D6,D7,D8 \
  --methods baselines,penalized/ebic,penalized/ric,chi2_stop,forward_stop,\
perm_gap,knockoff_path,xfit_objective,gaussian_cv,k_posterior,stability,\
changepoint,consensus \
  --seeds 30 --output benchmarks/results/auto_k_v2_main.csv

# Calibration deep-run for G2 (α/q methods, 50 seeds, min_k=0):
python benchmarks/bench_auto_k.py --designs D5 \
  --methods chi2_stop,forward_stop,perm_gap,knockoff_path,penalized/ebic \
  --seeds 50 --output benchmarks/results/auto_k_v2_null.csv

# Timing (D9, 2 seeds), and GBM transfer once FIX for --model lands:
python benchmarks/bench_auto_k.py --designs D9 --full --methods baselines,chi2_stop,\
penalized/ebic,gaussian_cv,xfit_objective,perm_gap --seeds 2 \
  --output benchmarks/results/auto_k_v2_d9.csv
python benchmarks/bench_auto_k.py --designs D1,D2,D3 --model catboost \
  --methods evaluate/one_se,gaussian_cv,penalized/ebic,chi2_stop,consensus \
  --seeds 10 --output benchmarks/results/auto_k_v2_catboost.csv
```

(Adjust method spellings to the CLI's actual names; `baselines` = the §B.4
six. If a `--methods baselines` alias doesn't exist, add it.)

Then:

1. Append the aggregated results table to `benchmarks/README.md` (house
   convention): per design × method — mean regret_frac, median |k̂ −
   k_oracle|, std k̂, support F1, runtime; plus the D5 calibration table
   (P(k̂ > 3) vs level) and the D8-vs-D2 structure-honesty deltas.
2. Evaluate every method against its §B.5 gates (G1–G6) mechanically; write
   PASS/FAIL per gate per method into the table. Methods that fail stay in
   the codebase but get the "experimental/failed-gate" label in docs — record
   *why* in one line each.
3. **Decide the default** per the program-level success bar: mean
   regret_frac ≤ 0.02 on D1–D3+D7, std(k̂)/k_oracle ≤ 0.35, G2-calibrated on
   D5. Current priors from the smoke runs (to be confirmed or overturned by
   the table, not asserted): `gaussian_cv/one_se`, `penalized/ebic`,
   `chi2_stop`, and `consensus` are the live candidates.
4. Sanity-check the winner once on a real dataset (user-supplied) before
   blessing it in DOCS.MD — synthetic sweeps are the gate, real data is the
   confirmation (spec R2/Q2).

**Verification stance for whoever reviews this campaign:** reproduce at least
one design×method row independently before trusting the table; check the D5
calibration numbers against the seeds actually run; do not accept "tests
pass" as evidence a gate passed.

Implementation note: this campaign was run on 2026-07-08 and recorded in
`benchmarks/README.md`. The measured CEFS+ default is `penalized/ebic`: it
passes the program bar (mean regret 0.001708 on D1-D3+D7,
std(k̂)/oracle 0.08048, D5 P(k̂>3)=0), and its D9 selection runtime is
0.00445x of the full CEFS+ path-build denominator. `gaussian_cv` remains a
useful fold-curve method but missed the D3 accuracy gate and D9 runtime target.
`changepoint`, `stability`, `xfit_objective`, and `knockoff_path` are kept as
diagnostics/experimental methods after failed gates.

---

## Part 6 — The single smart pathway: `k_method="auto"` router

Goal (user requirement): `select_cefsplus(X, y, k="auto")` should do the
right thing with **no config**, choosing the best method for the data rather
than making the user pick among 13 k_methods.

Design (implement only **after** Part 5's table exists — the routing rules
below are placeholders to be confirmed/replaced by measured results):

1. New `k_method="auto"` (name it `"auto"`; keep `resolve_auto_k_config`'s
   legacy no-config inference intact for backward compat, but route the
   no-config CEFS+ case to the new router once the default is blessed).
   The router is an orchestrator-level function
   (`select_gaussian_auto_path` in filter_auto_k.py) that:
   - computes cheap data facts: `n_eff` (Kish), `p_valid`, n_eff/p ratio,
     weights skew (`n_eff_kish / n`), presence of `time`/`groups`, selector
     method (cefsplus vs other), binary vs regression;
   - picks a primary method by a small, documented rule table (draft below);
   - runs it; if it returns a degenerate answer (k=0 with
     `stopped_by="max_k"`-style saturation, or curve degenerate), falls back
     to the next rule; and
   - records `{"auto_routing": {"chosen": ..., "reason": ..., "facts": ...}}`
     in the summary extras so the choice is always auditable.
2. Draft routing table (REPLACE with harness-derived rules; keep it to ≤5
   rows — a router with 20 branches is unmaintainable):

   | Condition (first match wins) | Method | Rationale to verify in Part 5 |
   |---|---|---|
   | non-CEFS+ Gaussian selector | `gaussian_cv/one_se` (kfold or context strategy) | only fold methods are eligible |
   | binary CEFS+ | `penalized/ebic` | fold/test methods not wired for binary |
   | p ≫ n (e.g. `p_valid > n_eff`) | `penalized/ebic` | EBIC territory (D7) |
   | heavy weights / suspected copula stress (weights skew > threshold) | `perm_gap/auto` | analytic nulls erode (D6/D8 measure this) |
   | otherwise | winner of Part 5 (prior: `gaussian_cv/one_se` or `consensus`) | the measured default |

3. If `consensus` wins Part 5 outright, the router can simply *be* consensus
   with context-aware membership (drop perm_gap when no structure and B×cost
   matters; drop fold methods when folds are impossible) — that is simpler
   and more robust than a rule table; decide from the numbers, considering
   runtime too (consensus ≈ sum of its members; gaussian_cv alone was ~10×
   cheaper in the smoke runs).
4. Ship with: DOCS.MD "Automatic k selection" rewritten around `k="auto"`
   (one blessed path + the full method menu for power users), the decision
   table updated with measured numbers, a `UserWarning` when the router's
   consensus-spread (if consensus) exceeds 2×, and router unit tests
   (routing determinism; every branch reachable; fallback chain; metadata
   recorded).

Implementation note: the shipped router is a small rule table, not consensus.
For CEFS+ it uses EBIC as the measured default, switches to EBIC when
`p_valid > n_eff_kish`, routes heavy weight skew to `perm_gap`, and routes
non-CEFS+ Gaussian selectors with explicit `k_method="auto"` to
`gaussian_cv/one_se`. Binary CEFS+ routes to EBIC. Each public route records
`auto_routing` metadata in `diagnostics_["auto_k"]`.

---

## Sequencing

- **PR-F1 — method + code fixes** (Part 1 + Part 4 docs corrections): small,
  self-contained, unblocks everything. Include the Part 3 unit tests that
  guard the fixes (stability jackknife/Φ ground truth, xfit null guard,
  gaussian_cv Cholesky equivalence).
- **PR-F2 — harness fixes + test debt** (Part 2 + rest of Part 3): the
  yardstick and the alarm system.
- **PR-F3 — benchmark campaign** (Part 5): results table into
  benchmarks/README.md, gate verdicts, default decision. Mostly a
  compute-and-write PR; keep code changes out of it.
- **PR-F4 — `k_method="auto"` router** (Part 6) + final docs pass keyed to
  the measured table.

Each PR: code + tests + docs in the same PR, suite clean under
`-W error::RuntimeWarning`, and the standing rule applies — verify against
gates and sims, not "tests pass".
