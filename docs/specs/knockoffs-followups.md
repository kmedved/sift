# Knockoffs Review Findings — Fix Specifications

Status: proposed (post-v1 in-depth review)
Companion to: [fdr-knockoffs.md](fdr-knockoffs.md) (the v1 spec)

This document turns every finding from the v1 review into a concrete,
implementation-ready fix: exact code changes, why they are correct, the tests
that lock them in, and acceptance criteria. Findings that were already fixed
during the review itself are recorded in §0 so this file is a complete account.

## Verified baseline (what the fixes are measured against)

- Suite: 560 passed, 17 skipped; `pytest -ra` clean; `-W error::RuntimeWarning`
  clean at 20k×1000 including `statistic="cefsplus"`.
- Performance at n=20k, p=1000: `build_cache(+Rxx)` 2.4s (was 17.1s);
  `select_fdr` X-path 2.8s (was ~17.7s); 9-draw derandomized 1.5s (was 3.1s).
- Statistics: swap-antisymmetry holds under a 40-random-multi-pair-swap stress
  test for both enabled statistics. `relevance` on the AR(1) reference design
  (n=800, p=40, 8 signals, q=0.2, 15 seeds): mean FDP 0.045, power 1.00.
  `cefsplus`: mean FDP 0.019, **power 0.475**; **0 selections** on a strong
  5-signal n=20k, p=1000 design; **10.7s** at default options (vs 0.37s for
  `relevance`).
- `mvr`: s̄ 0.354 vs equi 0.353 on AR(1) ρ=0.7 (no gain where true MVR gains
  most); s̄ 0.305 vs 0.105 on AR(0.9)⊕I (real gain on heterogeneous designs);
  fit cost 2.15s at p=1000 (64 full eigendecompositions).

## 0. Resolved during review (no action; recorded for completeness)

| Finding | Fix applied | Guard |
|---|---|---|
| `weighted_model` metadata was `True` for every unweighted run (compared mean-1 weights to `1/n`) | `bool(np.ptp(w) > 1e-9)` | asserted both directions in `test_knockoff_filter.py` |
| `KnockoffSelector` silently dropped `sample_weight` when a cache was set | explicit `ValueError` in `_fit_impl` | `test_knockoff_selector_rejects_sample_weight_with_cache` (constructor-cache and fit-cache paths) |
| RuntimeWarnings leaked from the CEFS+ update loop at 20k×1000 (two divisions outside the `errstate` block) | `errstate` extended over the full rank-1 update in both blocks | re-verified with `-W error::RuntimeWarning` at the repro scale |
| User guide claimed `mvr`/`me` "can improve power … on correlated designs" and implied `cefsplus` parity with the default | rewritten to match measurements (heuristic; helps on heterogeneous designs; `cefsplus` is a conservative, slower second opinion) | — |
| DOCS.MD "Low-Level Estimators" said `sift.estimators` exposes **three** lazy modules, omitting `sift.estimators.knockoffs` (exported since v1) | list corrected to four with the sampler API named | docs smoke test covers top-level exports only; reviewed manually |

---

## FIX-1 — CEFS+ statistic: objective-gain scoring (power) — **highest impact**

### Finding

Power 0.475 vs 1.000 for `relevance` on the reference design; 0 selections on
a strong-signal p=1000 design. Root cause is structural: entry scores are
ranks (`h = path_depth − position`), so null pairs entering mid-path carry
large ± magnitudes (deep-path entry order is noise). The knockoff+ threshold
must clear the largest negative null, which swamps the signals.

### Fix

Score each accepted entry by its **marginal objective gain**, which the
log-det recursion already computes. In
`_cefsplus_incremental_scores` ([knockoff_filter.py](../../sift/selection/knockoff_filter.py)),
replace the rank assignment at the accepted step:

```python
# before:
h[j] = float(path_depth - count)

# after:
gain = float(np.log(s1[rem_pos]) - np.log(s2[rem_pos]))
h[j] = max(gain, 0.0)          # clamp: eps floors can make it ~-1e-16
```

**Why this is the right quantity.** The greedy objective is
`logdet_S − logdet_yS = 2·I(y; S)`. Adding column `j` increments `logdet_S`
by `log s1` and `logdet_yS` by `log s2`, so
`gain = log s1 − log s2 = 2·I(y; x_j | selected) = −log(1 − ρ²_partial) ≥ 0`.
Null entries deep in the path have `ρ_partial ≈ 0` → `gain ≈ O(1/n)` instead
of a large rank — exactly the magnitude behavior the threshold needs. Signals
keep O(1) gains.

**Selection order is unchanged.** The loop's selection criterion is
`scores = lf − lc = (logdet_S − logdet_yS) + (log s1 − log s2)`, i.e. a
constant shift of the gain within an iteration. `argmax scores ==
argmax gain`, and score-ties are gain-ties, so the two-stage
pair-neutralization logic and the antisymmetry argument carry over verbatim.
Only the `h` payload changes.

**W scale note.** `W = h_orig − h_knock` moves from rank units to log-MI
units (typical null |W| ~ 1/n_effective, signal |W| ~ 0.1–1). The threshold
is scale-free, so nothing else changes; `W_draw_*` diagnostic columns just
get new units — mention in the docstring.

### Tests

- Existing antisymmetry, exact-tie, and multi-swap tests must pass unchanged
  (they exercise the neutralization logic, which is untouched).
- New power regression in `tests/test_knockoff_fdr_control.py`:

```python
def test_cefsplus_reference_calibration_and_power():
    fdps, powers = [], []
    for seed in range(15):
        X, y, truth = _reference_design(seed)
        result = select_fdr(X, y, q=0.2, statistic="cefsplus",
                            random_state=seed, verbose=False)
        fdp, power = _fdp_power(result.selected_indices, truth)
        fdps.append(fdp); powers.append(power)
    assert float(np.mean(fdps)) <= 0.30
    assert float(np.mean(powers)) >= 0.80      # was 0.475 with rank scoring
```

- Strong-signal sanity: the 5-signal n=20k, p=1000 construction from the
  review must select ≥ 4 of 5 signals at q=0.2 (single fixed seed; mark slow
  or shrink to n=5k if runtime matters).

### Acceptance

Reference-design mean power ≥ 0.80 at unchanged FDP margin; the p=1000
strong-signal case selects; suite green.

### Docs follow-up

After the numbers land, soften the user-guide passage added during review
("noticeably more conservative…") to reflect the new behavior — keep the cost
caveat until early stopping is on **by default** (FIX-2's two-stage policy),
not merely available: at the shipped `min_gain_ratio=0.0` default the
full-path cost is unchanged.

---

## FIX-2 — CEFS+ statistic: early stopping (runtime) — same PR as FIX-1

### Finding

10.7s at p=1000 with default options vs 0.37s for `relevance`, because the
greedy path walks all `path_depth = m` steps (Python loop, O(s²·n_rem) per
step) even though everything after the signal features contributes noise.

### Fix

Stop the path when the best available gain is negligible. In the main loop of
`_cefsplus_incremental_scores`, immediately after `best_score` is computed and
before tie handling:

```python
gain_best = best_score - (logdet_S - logdet_yS)
if count > 0 and min_gain_ratio > 0.0 and gain_best < min_gain_abs:
    break
if count == 0:
    first_gain = max(gain_best, eps)
    min_gain_abs = min_gain_ratio * first_gain
```

The `min_gain_ratio > 0.0` guard in the break condition is load-bearing:
without it, the "disabled" default `0.0` would still truncate the path on
epsilon-negative `gain_best` (the s1/s2 recursions are independently
eps-floored, so accumulated rounding can produce ~−1e-16 — the same reason
FIX-1 clamps `h`), silently violating the zero-behavior-risk stage of the
default policy.

- New option `min_gain_ratio` (via `statistic_options`): the stop threshold
  is `min_gain_ratio ×` the first accepted gain; `0.0` disables early
  stopping — made true by the guard above, not by convention. Registered in
  `allowed_options` (FIX-6).
- **Two-stage default policy:** land FIX-1/FIX-2 with default `0.0` — the
  power fix ships with zero behavior risk, and early stopping is a
  documented opt-in for large runs. As built, `path_depth` defaults to a
  conservative cap of 10 screened pairs while `min_gain_ratio=0.0` still truly
  disables gain-based early stopping. Flip the gain default to `1e-4` in a
  follow-up commit only after the mixed-effect-size acceptance data below
  exists; a threshold that changes selections must not ship on intuition.
- **Exchangeability is preserved**: `gain_best` is a symmetric function of
  the augmented data (it does not depend on which side of any pair produced
  it), so the stopping decision is swap-invariant. Features never entered
  keep `W = 0`, which the threshold already treats correctly (zeros are
  excluded from candidates and from the negative count — the same rule
  screened-out features use).
- `path_depth` remains as a hard cap; with FIX-1 magnitudes it is no longer
  the power knob, just a cost bound.

### Acceptance data required before `1e-4` becomes the default

The intuition (null gains concentrate at O(1/n_effective), so a detectable
signal sits well above `1e-4 ×` the strongest gain) must be demonstrated,
not asserted — and the flip gate needs a *defined* design, or it is
unfalsifiable (every other design in this plan is strong-signal). The
mixed-effect-size design, concretely:

- AR(1) ρ=0.5, n=800, p=40; strong signals `β[0:4] = linspace(1.6, 1.0, 4)`;
  weak signals `β[4:8] = [0.30, 0.20, 0.12, 0.08]` (chosen to straddle the
  detection threshold at this n — the weakest should be borderline for the
  `0.0` run itself); noise σ=1; q=0.25; 15 seeds.
- Run `statistic="cefsplus"` with `min_gain_ratio=1e-4` and `0.0` on
  identical data. Track power separately for the strong and weak subsets.

Flip the default only if mean power at `1e-4` is within 0.02 of the `0.0`
run on **both** subsets and FDP margins are unchanged. If the data shows
weak-signal loss, keep `0.0` and document the knob as a large-p performance
trade-off instead. House the sim in `tests/test_knockoff_fdr_control.py`
(or a recorded benchmark script if runtime forces it out of the suite).

### Tests / acceptance

- **"0.0 disables" is a real invariant, not rhetoric**: assert
  `statistic_options={"min_gain_ratio": 0.0}` and the option omitted produce
  byte-identical `W` on the reference design (pins the C1 guard above).
- Determinism test still passes (stopping is deterministic).
- Antisymmetry tests still pass (symmetric stopping rule).
- Runtime: with `min_gain_ratio=1e-4` opted in (and by default once the
  flip lands), the p=1000 case drops from 10.7s to ≤ ~1s (the path
  terminates shortly after the signals); assert nothing about wall time in
  tests — record it in `benchmarks/README.md` via the `bench_knockoffs.py`
  cefsplus cases instead.

---

## FIX-3 — Suite blindness: the tests could not see the power problem

### Finding

`test_select_fdr_cefsplus_smoke_and_path_depth_metadata` asserts only
metadata and finiteness; the calibration suite sims only `relevance`,
unweighted, ungrouped. A statistic can be enabled with severely degraded
power and the suite stays green.

### Fix

Extend `tests/test_knockoff_fdr_control.py` (keep total runtime < ~90s):

1. `cefsplus` calibration + power — the test in FIX-1.
2. **Weighted** calibration on a *weight-sensitive* construction ("differs
   from the unweighted run" is not a reliable invariant — strong signals can
   select identically under both weightings). Build a design where correct
   weighting is load-bearing: half the rows get weight 1.0 and follow
   `y = Xβ + ε`; the other half get weight 1e-3 and follow `y = −Xβ + ε`.
   Unweighted, the signal cancels; weighted, it is at full strength. Assert
   on the same data: weighted run has mean power ≥ 0.8 and FDP within margin;
   unweighted run has mean power ≤ 0.2; `weighted_model is True` on the
   weighted run; and the weighted `W` vector differs numerically from the
   unweighted one. This guards the importance-weighted plug-in path
   end-to-end (currently zero sim coverage) with assertions that cannot pass
   by accident.
3. **Grouped global null**: 40 features in 10 groups of 4, pure-noise `y`,
   `q=0.2`, 20 seeds; assert ≤ 40% of seeds select any group (mirrors the
   featurewise global-null bound and exercises
   `_group_knockoff_statistics` under the threshold).
4. Promote the review's ad-hoc stress test into
   `tests/test_knockoff_filter.py`: for each enabled statistic, 10 random
   swap subsets of sizes 1–3 at p=12 (deterministic seeds), asserting exact
   sign flips on swapped pairs and unchanged W elsewhere. The existing test
   swaps a single fixed pair; multi-pair swaps are what the exchangeability
   argument actually quantifies over.

### Acceptance

All four added; suite runtime increase bounded (~30–60s); each new test fails
if its guarded property is broken (verified by mutation: e.g. reverting FIX-1
must fail test 1).

---

## FIX-4 — `mvr`/`me`: implement the real optimizers (or rename)

### Finding

The shipped `"mvr"`/`"me"` are a scaled conditional-variance heuristic with
bisection feasibility. Valid (never worse than equi, by guard) but not the
Spector–Janson optimizers the names promise: measured **zero gain on AR(1)**
(s̄ 0.354 vs 0.353 at ρ=0.7) — the canonical case where true MVR shines — and
64 full eigendecompositions per fit (2.15s at p=1000, O(p³) each).

### Fix (primary): true coordinate descent

Both objectives decouple over the standard block rotation
`G = [[Σ, Σ−D], [Σ−D, Σ]] ≅ blockdiag(2Σ − D, D)`, giving

```
MVR loss:  L(s) = tr((2Σ − D)⁻¹) + Σ_j 1/s_j        (minimize)
ME  loss:  L(s) = −log det(2Σ − D) − Σ_j log s_j     (minimize)
```

Coordinate descent over `s_j` with `A = 2Σ − D` and `Ainv = A⁻¹` maintained
explicitly. Perturbing `s_j` by `δ` is a rank-1 update `A ← A − δ e_j e_jᵀ`;
with `c = Ainv[j, j]` and `v = ‖Ainv[:, j]‖²`, the coordinate-optimal steps
are **closed form** (verified numerically against brute-force minimization,
agreement to 1e-4 across coordinates and both objectives):

```
MVR:  δ* = (1 − √v · s_j) / (√v + c)
ME:   δ* = (1 − c · s_j) / (2c)
```

Both are automatically interior-feasible: `1 − δ*c > 0` and `s_j + δ* > 0`
algebraically (no clamping needed beyond a numerical slack cap
`δ ≤ (1 − 1e-6)/c`). After each step, Sherman–Morrison update:

```python
u = Ainv[:, j].copy()
Ainv += (delta / (1.0 - delta * c)) * np.outer(u, u)
s[j] += delta
```

Skeleton for `sift/estimators/knockoffs.py`:

```python
def _solve_mvr_me_s(Sigma: np.ndarray, *, objective: str, s_init: np.ndarray,
                    sweeps: int = 8, rtol: float = 1e-4) -> np.ndarray:
    p = Sigma.shape[0]
    s = np.clip(np.asarray(s_init, dtype=np.float64), 1e-8, None)
    for sweep in range(sweeps):
        # refresh from Cholesky each sweep to cap Sherman–Morrison drift
        A = 2.0 * Sigma - np.diag(s)
        try:
            cf = cho_factor(A, lower=True, check_finite=False)
        except LinAlgError:
            break  # float accumulation nudged A indefinite: keep last valid s
        Ainv = cho_solve(cf, np.eye(p), check_finite=False)
        max_rel = 0.0
        for j in range(p):
            c = Ainv[j, j]
            if objective == "mvr":
                v = float(Ainv[:, j] @ Ainv[:, j])
                delta = (1.0 - np.sqrt(v) * s[j]) / (np.sqrt(v) + c)
            else:  # "me"
                delta = (1.0 - c * s[j]) / (2.0 * c)
            delta = min(delta, (1.0 - 1e-6) / c)      # PSD slack
            delta = max(delta, 1e-8 - s[j])           # keep s_j > 0
            if abs(delta) < 1e-14:
                continue
            u = Ainv[:, j].copy()
            Ainv += (delta / (1.0 - delta * c)) * np.outer(u, u)
            s[j] += delta
            max_rel = max(max_rel, abs(delta) / max(s[j], 1e-8))
        if max_rel < rtol:
            break
    return s
```

Integration in `_solve_diagonal_s`:

- Initialize from the heuristic's *base vector* — `min(2/diag(Σ⁻¹), 1)`
  survives as a warm start even though `_feasible_scaled_s` and its bisection
  are deleted (the coordinate descent maintains feasibility itself) — or
  simply from equi; both converge, the base vector just saves a sweep or two.
- **Replace the mean-s guard with a loss guard**: the current
  `mean(s) < mean(equi) → equi` fallback would wrongly discard genuine MVR
  solutions (MVR does not maximize mean s). Compare `L(s_solved)` vs
  `L(s_equi)` under the chosen objective and keep the lower; keep the final
  `λ_min(2Σ − D) ≥ −1e-8` feasibility assertion as a hard backstop.
- Delete `_feasible_scaled_s` and its 64-eigendecomposition bisection — PD is
  maintained per coordinate, so no global feasibility search is needed.
- Do **not** cap `s_j` at 1.0: the only true constraint is `0 ≺ D ≺ 2Σ`;
  `s_j > 1` (negative original–knockoff correlation) is legitimate and is
  where much of MVR's power on correlated designs comes from. Update the
  equi-only assumption in any test that asserts `s ≤ 1`.

Complexity: O(p²) per coordinate, O(p³) per sweep, ≤ 8 sweeps → comparable to
one Cholesky-plus-inverse per sweep. Expected ≈ 1–3s at p=1000, replacing the
2.15s heuristic *and* buying real power.

### Fix (fallback, if the primary is deferred)

Rename the heuristic honestly and un-claim the literature names:

- `s_method="cvar_scaled"` for the current implementation (docstring: "scaled
  conditional-variance heuristic; matches equi on uniformly correlated
  designs, helps on heterogeneous ones").
- `"mvr"`/`"me"` raise `ValueError(... "reserved until the exact optimizers
  are implemented")`, mirroring the reserved-statistic pattern.
- Metadata `s_method` reports the honest name.

One of the two **must** happen before 0.7.0 — shipping literature names on a
heuristic that measures at zero improvement in the canonical case is the kind
of claim this library's docs discipline exists to prevent.

### Tests / acceptance (primary fix)

**Do not gate on mean s.** Running the coordinate descent above on AR(1)
ρ=0.7 (p=100) gives s̄ = 0.237 vs equi's 0.353 — *lower* mean s — while
reducing the MVR loss from 2.8×10⁶ to 739 with `λ_min(2Σ−D) = 0.12`
(equi's loss explodes because its `1 − 1e-6` slack leaves `2Σ − D`
near-singular; MVR's whole point is to buy reconstructability back). A
mean-s acceptance bar would reject a mathematically correct solver. MVR's
power comes from the loss, not from average knockoff noise. Gate on:

- Closed-form-vs-brute-force unit test at p=6: both objectives, two
  coordinates, agreement 1e-4 (numerically verified during review).
- **Loss dominance**: `L(s_mvr) ≤ L(s_equi)` and
  `L(s_mvr) ≤ L(s_heuristic)` under the MVR objective on AR(1) ρ∈{0.5, 0.7}
  and the heterogeneous block design (measured reference points:
  AR(1) ρ=0.7 → 739 vs 2.8×10⁶; heterogeneous → mvr s̄ 0.536, loss 1316).
- **Feasibility**: `λ_min(2Σ − diag(s)) ≥ −1e-8` on all tested designs.
- **Regression: lower mean-s is accepted.** Assert the AR(1) solution is
  *kept* (not replaced by the equi fallback) even though
  `mean(s_mvr) < mean(s_equi)` — this pins the loss-guard replacement of the
  old mean-s guard.
- **Power sim** (the user-visible criterion): reference design with ρ=0.7
  correlation, `relevance` statistic, `s_method="mvr"` vs `"equi"` — mean
  power strictly higher for mvr at equal FDP margin (10 seeds).
- If a shape diagnostic is wanted, record `s` quantiles or the loss ratio in
  metadata/benchmarks — never as a lower bound on `mean(s)`.
- Fit runtime at p=1000 recorded in `benchmarks/README.md` (≤ ~3s target).

---

## FIX-5 — `_feasible_scaled_s` cost (only if FIX-4 primary is deferred)

If the heuristic temporarily survives (fallback path of FIX-4), cut its cost:

- `eigh(..., eigvals_only=True)` → `eigh(..., subset_by_index=[0, 0],
  eigvals_only=True)` in the bisection probe and `_min_noise_eig` — only the
  smallest eigenvalue is consumed; computing all `p` per probe is waste.
- Replace the fixed 64 iterations with a convergence break
  (`hi − lo < 1e-4` → ~14 iterations; the remaining 50 refine noise).

Measured baseline 2.15s at p=1000; expected ~5–8× reduction. Deleted entirely
by FIX-4 primary.

---

## FIX-6 — `statistic_options` typo-blindness (confirmed live)

### Finding

`select_fdr(..., statistic_options={"path_dept": 3})` is silently accepted;
the misspelled key is ignored and the default depth used. Same
silently-ignored-typo class as the CatBoost `group_col` finding (TODO §11).

### Fix

Add per-statistic allowed keys to the registry spec and validate before use:

```python
@dataclass(frozen=True)
class KnockoffStatSpec:
    name: str
    fn: Callable[[KnockoffStatContext], np.ndarray]
    enabled: bool = True
    needs_screening: bool = True
    allowed_options: frozenset[str] = frozenset()

# registry:
"relevance": KnockoffStatSpec(..., allowed_options=frozenset()),
"cefsplus":  KnockoffStatSpec(..., allowed_options=frozenset({"path_depth", "min_gain_ratio"})),
"lcd":       KnockoffStatSpec(..., allowed_options=frozenset(
                 {"cv", "max_iter", "tol", "selection", "alphas", "eps", "n_alphas"})),
```

In `select_fdr`, validate **before** injecting the internal name key — and
stop injecting it: pass the statistic name as a proper `KnockoffStatContext`
field (`statistic_name: str = ""`, defaulted so the frozen dataclass stays
constructible for direct callers) instead of smuggling `_statistic_name`
through `options`. `_reserved_statistic` reads the context field.

**Required call-site updates (the refactor breaks them otherwise):**
`_build_context` is private but has two out-of-module callers that construct
contexts directly and one that smuggles the name —
[benchmarks/bench_knockoffs.py](../../benchmarks/bench_knockoffs.py) (~line
97, passes `options={"_statistic_name": statistic}`) and
`tests/test_knockoff_filter.py` (`_context_for_cache`, passes `options={}`).
Update both in the same commit: the benchmark passes
`statistic_name=statistic` (and drops the options smuggle — with FIX-6
validation active, `_statistic_name` would otherwise raise as an unknown
key), and the test helper forwards `statistic_name` where the reserved-error
test needs it.

```python
unknown = set(options) - stat_spec.allowed_options
if unknown:
    allowed = sorted(stat_spec.allowed_options) or ["<none>"]
    raise ValueError(
        f"Unknown statistic_options for {stat_spec.name!r}: {sorted(unknown)}; "
        f"allowed: {allowed}"
    )
```

### Tests

`{"path_dept": 3}` raises naming both the bad key and `path_depth`;
`relevance` rejects any key; each allowed key is accepted for its statistic;
the internal name never appears in error messages (regression for the
context-field refactor).

---

## FIX-7 — Metadata reports requested, not effective, `path_depth`

`selector_metadata["path_depth"]` echoes `options.get("path_depth")` (`None`
when defaulted) while `_validate_path_depth` resolves the real value. Since
`m = min(p_active, screen_pairs or p_active)` does not depend on the draw,
resolve once in `select_fdr` after the active mask is built:

```python
p_active = int(active_positions.shape[0])   # select_fdr has no p_active local today
m_pairs = p_active if screen_pairs_int is None else min(p_active, screen_pairs_int)
path_depth_effective = (
    _validate_path_depth(options.get("path_depth"), m_pairs)
    if stat_spec.name == "cefsplus" else None
)
metadata["path_depth_requested"] = options.get("path_depth")
metadata["path_depth"] = path_depth_effective
```

Pass the resolved value through options so `_stat_cefsplus` stops re-deriving
it. Tests: default run reports `path_depth == m_pairs`; a request above `2m`
reports the cap; `relevance` reports `None`.

---

## FIX-8 — `subsample` default is indistinguishable from an explicit value

`_resolve_cache` rejects `subsample` with a cache only when it differs from
`(None, 50_000)`, so an explicit `subsample=50_000` (or `None`) with a cache
passes silently. Use a module-level sentinel — but **not a plain `object()`**:
sklearn's `clone()` deep-copies estimator params (verified: `clone` on an
estimator holding an `object()` sentinel yields a *different* object, so an
`is` check fails), which would make every cloned `KnockoffSelector` inside
`cross_val_score`/`GridSearchCV` spuriously raise on the cache path. The
sentinel must survive `copy`, `deepcopy`, **and pickle** (joblib workers):

```python
class _SubsampleDefaultType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<subsample default: 50,000 rows when X is given>"

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):          # pickle round-trips to the same singleton
        return (_SubsampleDefaultType, ())


_SUBSAMPLE_DEFAULT = _SubsampleDefaultType()

def select_fdr(..., subsample=_SUBSAMPLE_DEFAULT, ...):
    ...
# in _resolve_cache:
if cache is not None and subsample is not _SUBSAMPLE_DEFAULT:
    raise ValueError("subsample cannot be passed with a prebuilt cache")
resolved_subsample = 50_000 if subsample is _SUBSAMPLE_DEFAULT else subsample
```

`KnockoffSelector` must adopt the same sentinel as its constructor default
and forward it verbatim — anything else either makes an explicit `50_000`
indistinguishable from the default or silently ignores a non-default value in
the cache branch, recreating the bug class this fix exists to remove.
Document the effective default ("50,000 rows when X is given") in the
docstring since the signature no longer shows it. Update the validation tests
to assert: explicit `subsample=50_000` and explicit `subsample=None` with a
cache both raise (function and wrapper paths); the untouched default with a
cache passes; the X path still honors explicit values; and — the regression
that motivated the singleton — `clone(KnockoffSelector(cache=...)).fit(X, y)`
and a `pickle.loads(pickle.dumps(...))` round-trip both still take the
default path (`cloned.subsample is _SUBSAMPLE_DEFAULT`).

**Alternative considered (round-4 review): split the entry point** — a
cache-first variant mirroring `select_cached` (e.g.
`select_fdr_cached(cache, y, ...)`) whose signature simply has no
`subsample`/`sample_weight`, deleting the entire "X-only kwarg with a
prebuilt cache" bug class structurally instead of detecting it, with zero
sentinel/clone/pickle machinery. Trade-off: two public names plus wrapper
branching. With no backwards-compat constraint before 0.7.0 this is cheap
now and expensive later — decide explicitly when implementing; the sentinel
spec above stands if the single-entry-point API is kept.

---

## FIX-9 — `feature_groups` length validation (code, not just docs)

Review round 2 upgraded this from a documentation note to a validation hole:
`_resolve_feature_groups` accepts **any** array whose length exceeds
`max(valid_cols)` and silently indexes it by `valid_cols` — an overlong
(stale, from a wider matrix) group vector is silently truncated-by-indexing,
and some under-specified "original-length" vectors pass whenever the omitted
labels happen to be trailing dropped columns. Tighten the contract to exact
lengths:

```python
p_valid = cache.Z.shape[1]
n_original = len(cache.feature_names) if cache.feature_names is not None else None
if groups_arr.shape[0] == p_valid:
    valid_groups = groups_arr
elif n_original is not None and groups_arr.shape[0] == n_original:
    valid_groups = groups_arr[cache.valid_cols.astype(np.int64)]
else:
    expected = f"{p_valid}" if n_original is None else f"{p_valid} or {n_original}"
    raise ValueError(
        f"feature_groups has length {groups_arr.shape[0]}; expected exactly {expected} "
        "(valid cache columns, or the original input columns)"
    )
```

(`build_cache` always populates `feature_names` — synthetic or real — with
the original column count, so `n_original` exists for every cache it builds;
only hand-constructed caches with `feature_names=None` fall back to
`p_valid`-only.) Keep the docs note: "pass one label per *original* column; a
list whose length equals the valid-column count is interpreted as
cache-aligned; when the two counts coincide, cache-aligned wins (they are the
same mapping unless columns were dropped)."

**Tests:** off-by-one lengths (`p_original + 1` and `p_original − 1`) and an
overlong array (`p_original + 3`) all raise instead of silently indexing;
exact-original length with trailing dropped columns maps through
`valid_cols`; exact-`p_valid` length is taken as cache-aligned; the error
message names both accepted lengths.

---

## FIX-10 — `weighted_corr_with_vector` is a serial numba loop

89ms per call at 20k×1000, called twice per draw; it is a matvec. Add a BLAS
path in [copula.py](../../sift/estimators/copula.py):

```python
def weighted_corr_with_vector_blas(Z, zy, w, *, batch_size=50_000):
    # 50k rows × p float64 per batch matches weighted_correlation_matrix_blas's
    # allocation ceiling; 100k × p=2000 would spike ~1.6 GB.
    w64 = np.asarray(w, dtype=np.float64)
    wy = (w64 * np.asarray(zy, dtype=np.float64))
    acc = np.zeros(Z.shape[1], dtype=np.float64)
    for start in range(0, Z.shape[0], batch_size):
        stop = min(Z.shape[0], start + batch_size)
        acc += np.asarray(Z[start:stop], dtype=np.float64).T @ wy[start:stop]
    r = acc / float(w64.sum())
    return np.clip(r, -0.999999, 0.999999).astype(np.float32)
```

Route callers (`select_cached`, knockoff filter) through a dispatcher that
uses BLAS for `n·p ≥ ~1e6` and the numba kernel below that (or just replace —
the numba kernel's only advantage is avoiding the float64 upcast on tiny
inputs). Accumulation order differs from the numba kernel at the 1e-7 level;
add an equivalence test with `atol=1e-5` (downstream tolerance is far looser).
Expected ~5–10× on this step; benefits `select_cached` too.

**Same PR: hoist the draw-invariant `r`.** `_build_context` recomputes
`r = weighted_corr_with_vector(Z, zy, w)` on every draw, and `select_fdr`
computes it once more as `r_orig` for the relevance metadata column — but `r`
depends only on the originals, not the knockoffs. Compute it once before the
draw loop, reuse it for the relevance column, and pass it into
`_build_context` (new parameter; only `rt` is per-draw). At 11 draws and
20k×1000 that removes ~1s of pure redundancy by this fix's own measurement.

**Document the standardization assumption while touching this function:**
`weighted_corr_with_vector` computes *uncentered* weighted second moments;
its validity as a correlation relies on cache columns and `zy` being
weighted-standardized by construction, and on `Z̃` inheriting weighted zero
mean only in expectation (an O(n^{−1/2}) empirical offset in `rt`). The `lcd`
statistic re-centers explicitly for exactly this reason; `relevance` and
`cefsplus` rely on it silently. One docstring sentence makes the contract
explicit.

---

## FIX-11 — Noise-draw dtype: decide and document (do not drift)

`rng.standard_normal((rows, p))` draws float64 then casts to float32 —
deliberate, to keep the draw stream stable. Passing `dtype=np.float32`
roughly halves noise-generation time but changes the stream, silently
changing every seeded result. Decision required, pre-release being the cheap
moment:

- **Option A (recommended):** switch to `dtype=np.float32` now, note the
  stream change in the 0.7.0 release notes ("knockoff draws differ from
  pre-release builds for the same seed").
- **Option B:** keep float64 draws and add a code comment stating it is a
  reproducibility choice, so a future "optimization" doesn't flip it
  silently.

Either way, add a pinned-seed regression test asserting a small known slice of
`sample_knockoffs(cache, random_state=123)` so any future stream change fails
loudly instead of silently. (Round-2 review endorsed Option A: resolving the
stream shift pre-release beats carrying the float64 generation cost forever
to protect an unreleased stream.) The pinned test doubles as the tripwire for
numpy `Generator` bit-stream changes across versions. Also add a comment on
`block_size = 8192` in the sampler noting it is provably *not* part of the
seed contract — sequential block draws consume the stream identically to one
call — so nobody "fixes" it defensively.

---

## FIX-12 — The spec's 50k×2000 target has never actually been run

Spec acceptance criterion 7 requires the single-draw default statistic at
n=50k, p=2000 under ~60s **with numbers recorded in `benchmarks/README.md`**.
The benchmark's full mode *does* already include `(50_000, 2_000)` relevance
cases (1 and 11 draws) — the actual gaps are: no `cefsplus` case at that
scale, and no recorded results anywhere (full mode has not been run for the
record). Actions:

1. Add `(50_000, 2_000, 1, "cefsplus")` to `_cases(full=True)` in
   [bench_knockoffs.py](../../benchmarks/bench_knockoffs.py) — after FIX-1/2,
   which make that case tractable (pass `min_gain_ratio=1e-4` explicitly in
   the benchmark until the default flips per FIX-2's two-stage policy).
2. Run `python benchmarks/bench_knockoffs.py --full --output ...` once
   post-FIX-1/2 and paste the table into `benchmarks/README.md`.
3. Extrapolation says the relevance target passes comfortably (cache ~8s +
   filter ~2s); if it does not, that is exactly what the criterion exists to
   catch.

---

## FIX-13 — `build_cache` weighted subsampling ignores the weights (P0, pre-existing)

### Finding (round-2 review; blocker-level, predates knockoffs)

`build_cache` ([copula.py](../../sift/estimators/copula.py), subsample block)
draws `row_idx = rng.choice(n, size=subsample, replace=False)` **uniformly
over all rows before looking at `sample_weight`**; the only guard is that the
sampled weights sum to a positive number. With many zero-weight rows and
`n > subsample`, a weighted cache either raises nondeterministically (the
guard fires) or — worse — silently builds `Z`/`Rxx` from whatever sliver of
the positive-weight population survived uniform sampling. Every weighted
X-path `select_fdr` call inherits this via `_resolve_cache`, as does every
other weighted cache consumer. TODO §7's "deterministic safe behavior" claim
currently covers only the raise path, not the effective-sample loss.

### Fix

Subsample from the positive-weight support only. **Uniformly over positive
rows, not weight-proportionally** — downstream statistics re-weight by
`ws = w[row_idx]`, so weight-proportional sampling would double-count weights
unless `ws` were also reset to uniform (a larger semantic change, explicitly
out of scope here). Zero-weight rows are semantically free to drop: every
downstream quantity multiplies by `w`.

```python
if subsample is not None and n > subsample:
    positive = np.flatnonzero(w > 0.0)
    if positive.size == n:
        # unweighted / all-positive: keep the existing call so seeded
        # unweighted caches reproduce byte-for-byte across this change
        row_idx = rng.choice(n, size=subsample, replace=False)
    elif positive.size <= subsample:
        row_idx = positive
    else:
        row_idx = rng.choice(positive, size=subsample, replace=False)
else:
    row_idx = np.arange(n)
```

Keep the existing positive-total-weight guard as a backstop (it becomes
unreachable when any positive row exists, which `ensure_weights` already
guarantees). Note in the release notes that seeded **weighted** caches select
different rows after this change; unweighted caches are unchanged by
construction.

### Tests (cache-level, e.g. `tests/test_weights.py` or a new
`tests/test_cache_subsample.py` — this touches all Gaussian selectors, not
just knockoffs)

- `n=10_000`, `subsample=100`, exactly 50 positive-weight rows scattered so
  uniform sampling would usually miss most of them: assert
  `set(row_idx) == set(positive)` (all positive rows retained) and the build
  never raises.
- `n=10_000`, `subsample=100`, 5,000 positive rows: 100 sampled rows, all
  positive.
- Unweighted seeded run: `row_idx` identical to the pre-change implementation
  (byte-compat regression, pinned expected indices).
- Weighted `select_fdr` X-path smoke on the sparse-weight construction:
  completes and calibrates (ties into FIX-3's weighted simulation).
- Update TODO §7's notes once landed so the claim is true.

## FIX-14 — `KnockoffSelector(cache=...)` never checks that `X` matches the cache

### Finding (round-2 review, hidden-assumption)

On the cache path, `feature_names_in_` comes from the `X` passed to `fit`
while `selected_features_`/`selected_indices_` come from the cache-backed
result. Nothing validates that `X` and the cache describe the same columns in
the same order, so a stale or mismatched cache silently detaches
`transform()` from the selection run — `get_support()` indexes the wrong
columns of `X`.

### Fix

In `KnockoffSelector._fit_impl`'s cache branch, after `feature_names =
_feature_names_or_default(X)`:

```python
cache_names = resolved_cache.feature_names
if cache_names is not None and not resolved_cache.feature_names_are_synthetic:
    if list(feature_names) != list(cache_names):
        raise ValueError(
            "X columns do not match cache.feature_names (names and order must "
            "be identical); fit the cache from the same matrix"
        )
elif cache_names is not None and len(feature_names) != len(cache_names):
    raise ValueError(
        f"X has {len(feature_names)} columns but the cache was built from "
        f"{len(cache_names)}"
    )
```

Synthetic-name caches (ndarray inputs) can only be width-checked — document
that column *order* remains caller responsibility in that case, consistent
with the cache-freshness caveat already in the docs.

### Tests

DataFrame with renamed or reordered columns + cache → raises; matching
DataFrame → passes; ndarray of wrong width → raises; ndarray of right width
against a synthetic-name cache → passes.

## Release checklist (0.7.0)

- [x] `__version__ = "0.7.0"`; release notes covering: the feature; the
      `build_cache` speedup (benefits **all** Gaussian selectors); the
      approximate-plug-in framing; FIX-11's decision if Option A.
- [x] Flip [fdr-knockoffs.md](fdr-knockoffs.md) from "draft for review" to
      "implemented (as-built)" with a deviations section: `cefsplus` enabled
      in v1, `feature_groups` pulled forward from Phase 2 (group
      *thresholding* over individual knockoffs, not block-S group knockoffs),
      `mvr`/`me` per FIX-4's outcome, `lcd` still reserved.
- [x] `docs/troubleshooting.md` entries: "it selected nothing" (expected
      outcome; legitimate knobs `q`/`offset=0`/`n_draws`, and that rerunning
      until it selects is not one); the shrinkage `UserWarning`
      (`gamma`/`lambda_min`, dedup advice); run-to-run variation and
      `random_state`/derandomization; multiclass one-vs-rest recipe.
- [x] Group diagnostics: store per-draw `group_W` and group thresholds in
      `diagnostics_` when `feature_groups` is active (currently only
      featurewise sets are kept, so users cannot see *why* a group survived).
- [x] Optional integer-label guard: one-time `UserWarning` when `y` is
      integer-typed with 3–20 unique values suggesting one-vs-rest (matches
      the library's loud-about-ambiguity philosophy; costs one `np.unique`).

## Phase-2 pointers (unchanged from the v1 spec, not expanded here)

True block-S group knockoffs (Dai & Barber group-equi; sampler formulas
generalize with `D → S` block-diagonal); raw-space inverse-ECDF push-back for
tree-model statistics; auto-k cross-check surface; remaining reserved
statistics (`mrmr_*`, `jmi*`) as variants of the FIX-1 wrapper once
objective-gain scoring exists.

## Suggested sequencing

1. **FIX-13** (standalone PR): pre-existing weighted-cache correctness bug;
   affects all Gaussian selectors, and FIX-3's weighted simulation depends on
   it being fixed.
2. **FIX-1 + FIX-2 + FIX-3** (one PR): makes the second statistic real and
   proves it in the suite.
3. **FIX-6/7/8/9/14** (one hardening PR).
4. **FIX-4** — primary if budgeted, fallback rename otherwise (blocking for
   0.7.0 either way).
5. **FIX-10/11/12** + release checklist, then cut 0.7.0.
