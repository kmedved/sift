# Implementation Spec: FDR-Controlled Selection via Model-X Knockoffs

Status: draft for review
Target version: 0.7.0
Depends on: `FeatureCache` / Gaussian copula machinery (`sift/estimators/copula.py`),
cached Gaussian selectors (`sift/selection/cefsplus.py`), shared validation
(`sift/_preprocess.py`), result conventions (`sift/selection/result.py`).

---

## 1. Summary

Add a knockoff filter to sift: a `select_fdr(X, y, q=...)` entry point that
returns a feature set with **provable false-discovery-rate control** — "at
most a fraction `q` of the selected features are expected to be null" —
rather than a ranked top-k list.

The implementation is **second-order Gaussian Model-X knockoffs built in
copula space**: knockoff copies are sampled from the Gaussianized feature
matrix `Z` and correlation matrix `Rxx` that `build_cache` already produces.
Feature statistics `W_j` are computed by running sift's existing cached
selectors (CEFS+, Gaussian mRMR/JMI) or a lasso path on the augmented matrix
`[Z | Z̃]`, and the knockoff+ threshold converts them into an FDR-controlled
selection set. Derandomization (multiple knockoff draws, selection-frequency
aggregation) reuses the same machinery.

No new dependencies. numpy/scipy/scikit-learn (already hard deps) suffice.

### References (for docstrings and docs)

- Candès, Fan, Janson, Lv (2018), "Panning for gold: Model-X knockoffs for
  high-dimensional controlled variable selection", JRSS-B. (Model-X, knockoff+.)
- Barber & Candès (2015), "Controlling the false discovery rate via
  knockoffs", Annals of Statistics. (Threshold, antisymmetric statistics.)
- Ren, Wei, Candès (2021), "Derandomizing knockoffs", JASA. (Multi-draw
  aggregation.)
- Spector & Janson (2020), "Powerful knockoffs via minimizing
  reconstructability". (MVR/ME constructions — Phase 2.)

---

## 2. Background (just enough to review the code)

Given features `X = (X_1..X_p)` and target `y`, a valid knockoff matrix `X̃`
satisfies:

1. **Swap exchangeability**: for any subset `S`, swapping `X_j ↔ X̃_j` for
   `j ∈ S` leaves the joint distribution of `(X, X̃)` unchanged.
2. **Nullity**: `X̃ ⊥ y | X` (guaranteed by construction — `X̃` is sampled
   without looking at `y`).

Compute any **antisymmetric statistic** per feature,
`W_j = f(score(X_j), score(X̃_j))` with `W_j > 0` meaning "original beats its
knockoff", such that swapping `X_j ↔ X̃_j` flips the sign of `W_j`. Then the
**knockoff+ threshold**

```
τ = min { t > 0 : (1 + #{j : W_j ≤ −t}) / max(1, #{j : W_j ≥ t}) ≤ q }
```

yields a selection set `Ŝ = {j : W_j ≥ τ}` with `FDR(Ŝ) ≤ q`, exactly, in
finite samples — *conditional on the knockoffs being valid*.

**Second-order Gaussian knockoffs** approximate `X ~ N(0, Σ)` and sample

```
X̃ | X ~ N( X (I − Σ⁻¹ D),  2D − D Σ⁻¹ D ),    D = diag(s),  0 ⪯ D ⪯ 2Σ
```

which makes `(X, X̃)` jointly Gaussian with covariance
`G = [[Σ, Σ−D], [Σ−D, Σ]]` — exactly swap-exchangeable *under the Gaussian
model*. sift's angle: the cache's weighted rank-Gaussian transform makes every
column marginally `N(0,1)` by construction, so the Gaussian approximation is a
**copula assumption on the dependence structure only** — a much weaker and
more defensible assumption than Gaussianity of raw features. This is the core
reason knockoffs are cheap to add here: `Z` and `Rxx = Σ` already exist.

Honest framing to carry into all docs: FDR control is exact under the
Gaussian-copula model of `X` and approximate otherwise ("second-order
knockoffs"). This is the standard practical regime in the literature.

---

## 3. Design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Knockoffs are sampled **in copula space** (`Z`), never inverse-transformed to raw feature space (v1). | Monotone per-column transforms preserve feature identity; every supported statistic operates on `Z` anyway. Raw-space push-back is only needed for tree-model statistics — Phase 2. |
| D2 | `s`-vector via the **equicorrelated** construction plus automatic correlation shrinkage. | Closed form (one smallest-eigenvalue computation), no SDP solver, no new dependency. MVR/ME (better power) are Phase 2; the API takes `s_method` so they slot in. |
| D3 | The augmented feature-feature correlation matrix is the **analytic** `G = [[Σ, Σ−D], [Σ−D, Σ]]`, not re-estimated from `[Z | Z̃]`. | Free (no `O(n·p²)` pass), lower variance, and exactly swap-invariant, which makes the greedy-path statistics' exchangeability easy to verify. Feature–target correlations are empirical (they must be — that's where `y` enters). |
| D4 | Feature statistics come from a small **registry**: `cefsplus` (default), `mrmr_diff`, `mrmr_quot`, `jmi`, `jmim` (greedy entry-order statistics reusing the existing loops), `relevance` (marginal Gaussian-MI difference), `lcd` (lasso coefficient difference). | Sift-native statistics are the differentiator and are weight-aware end to end; `lcd` is the literature-standard baseline. Registry mirrors `sift/scoring.py` conventions. |
| D5 | Screening for expensive statistics is **pair-coupled**: rank pairs by `max(|r_j|, |r̃_j|)`, keep or drop originals and knockoffs together. | Any symmetric function of the pair preserves exchangeability; screening that could split a pair would silently break the FDR guarantee. |
| D6 | `corr_prune` is **never applied** inside the augmented selection run. | Pruning a knockoff against its own original (they are highly correlated by design when `s` is small) changes entry-order semantics; disabling is the conservative, obviously-correct choice. |
| D7 | `n_draws > 1` gives **derandomized knockoffs** (per-draw knockoff+ selection at level `q`, aggregate by selection frequency `π_j ≥ eta`). | Single-draw knockoffs are randomized — two runs can select different sets. Derandomization is the standard fix; the exact-FDR guarantee formally applies to `n_draws=1` and this is documented. |
| D8 | `y` handling matches `select_cached`: `zy = weighted_rank_gauss_1d(y)`. Continuous and binary targets supported; multiclass is documented as unsupported (encode or one-vs-rest). | Consistency with every other Gaussian-path selector; no `task` parameter to invent semantics for. |
| D9 | All weighting flows through the existing conventions: `Σ` is the weighted correlation matrix, `zy`/`r` are weighted, `lcd` uses the `√w` row-rescaling trick. Knockoff noise rows are i.i.d. | There is no exact "weighted Model-X" theory; treating weights as importance weights in estimation and statistics is the pragmatic choice, stated plainly in docs. |
| D10 | An **empty selection set is a valid, meaningful result**, not an error. | "Nothing survives at q=0.1" is precisely the information the feature is for. |

---

## 4. Public API

### 4.1 `select_fdr` — main entry point

```python
# sift/selection/knockoff_filter.py

def select_fdr(
    X=None,
    y=None,
    *,
    q: float = 0.1,
    statistic: str = "cefsplus",       # registry key, see 6.3
    n_draws: int = 1,
    eta: float = 0.5,                  # derandomization frequency threshold
    offset: int = 1,                   # 1 = knockoff+ (exact FDR), 0 = knockoff (mFDR)
    s_method: str = "equi",            # "equi" only in v1; "mvr"/"me" reserved
    min_eig: float = 1e-3,             # shrinkage floor for λ_min(Σ)
    screen_pairs: int | None = 2000,   # pair-coupled screening cap; None = all pairs
    statistic_options: dict | None = None,   # e.g. {"path_depth": ..., "cv": 5}
    sample_weight=None,                # only valid with X input
    subsample: int | None = 50_000,    # forwarded to build_cache when X given
    cache: FeatureCache | None = None, # prebuilt cache alternative to X
    random_state: int = 0,
    n_jobs: int = 1,
    verbose: bool = True,
) -> KnockoffSelectionResult
```

Contract:

- Exactly one of `X` / `cache` must be provided (raise `ValueError` otherwise).
  `sample_weight`/`subsample` are rejected alongside `cache` (weights live in
  the cache), matching the `select_cached` division of labor.
- `y` is required; length must equal `cache.n_rows_original` (reuse the
  existing check pattern from `select_cached`, `cefsplus.py:376`).
- When `X` is given, internally call
  `build_cache(X, sample_weight=..., subsample=..., random_state=...,
  compute_Rxx=True)`.
- When `cache.Rxx is None`, compute
  `weighted_correlation_matrix(cache.Z, cache.sample_weight)` locally (do
  **not** mutate the caller's cache object) and note it once under `verbose`.
- Returns a `KnockoffSelectionResult` (4.3). Never returns a bare list —
  threshold, W table, and frequencies are the point of the feature.

### 4.2 Advanced building blocks (exported for power users)

```python
# sift/estimators/knockoffs.py

@dataclass(frozen=True)
class GaussianKnockoffModel:
    """Precomputed sampling operators for one (Σ, s) pair."""
    s: np.ndarray            # (p,) knockoff s-vector
    mean_op: np.ndarray      # A = I − Σ⁻¹ D, float64 (p, p)
    noise_chol: np.ndarray   # L_N with L_N @ L_N.T = 2D − D Σ⁻¹ D, float64 (p, p)
    gamma: float             # shrinkage actually applied
    lambda_min: float        # smallest eigenvalue of the *unshrunk* Rxx

def fit_gaussian_knockoffs(
    Sigma: np.ndarray, *, s_method: str = "equi", min_eig: float = 1e-3,
) -> GaussianKnockoffModel

def sample_gaussian_knockoffs(
    Z: np.ndarray, model: GaussianKnockoffModel, rng: np.random.Generator,
) -> np.ndarray            # Z̃, float32, same shape as Z


# sift/selection/knockoff_filter.py

def knockoff_threshold(W: np.ndarray, q: float, *, offset: int = 1) -> float
    # Pure function; returns np.inf when no threshold achieves the bound.

def sample_knockoffs(
    cache: FeatureCache, *, s_method: str = "equi", min_eig: float = 1e-3,
    random_state: int = 0,
) -> np.ndarray            # convenience: fit + one draw against a cache
```

### 4.3 Result object

```python
# sift/selection/knockoff_result.py (or inside knockoff_filter.py; see 5)

@dataclass(frozen=True)
class KnockoffSelectionResult:
    selected_features: List[str]          # ordered by descending W (draw-aggregated)
    selected_indices: Optional[List[int]] # original-X column indices (via cache.valid_cols)
    selector_metadata: Dict[str, Any]     # see below
    W: pd.DataFrame                       # per-feature statistics, all draws (long or wide)
    threshold: Optional[float]            # knockoff+ threshold (n_draws == 1), else None
    selection_frequency: Optional[pd.Series]  # π_j indexed by feature (n_draws > 1), else None
    diagnostics_: Optional[Dict[str, Any]] = None

    def get_feature_ranking(self) -> pd.DataFrame:
        # columns: feature, W (mean across draws), rank, selected,
        #          selection_frequency (NaN when n_draws == 1), selected_index
```

`selector_metadata` extends `build_selector_metadata`-style keys
(`selector="knockoff_fdr"`, `n_features`) with: `q`, `offset`, `statistic`,
`s_method`, `n_draws`, `eta`, `screen_pairs`, `path_depth`, `gamma`
(shrinkage applied), `lambda_min`, `s_mean` (mean of the s-vector — the
single best power proxy), `random_state`, `n_rows_used`.

`diagnostics_` holds the per-draw thresholds and per-draw selection sets.

### 4.4 Sklearn-style wrapper

`KnockoffSelector` in `sift/selectors.py`, following the `MRMRSelector`
pattern (`selectors.py:603`): constructor mirrors `select_fdr` keyword args,
`_init_selector(select_fdr, locals())` if `_BaseSelector` generalizes (it
assumes a `k` arg — verify; if it does not fit cleanly, write the small
`fit/transform/get_support` by hand rather than contorting `_BaseSelector`).
`fit` stores `selected_features_`, `result_`.

### 4.5 Export surface

- `sift/__init__.py` `__all__` additions: `select_fdr`,
  `KnockoffSelectionResult`, `KnockoffSelector`, `sample_knockoffs`.
- `sift/selection/__init__.py`: `select_fdr`, `knockoff_threshold`,
  `KnockoffSelectionResult`.
- `sift/estimators/__init__.py`: `fit_gaussian_knockoffs`,
  `sample_gaussian_knockoffs`, `GaussianKnockoffModel`.
- **Required**: DOCS.MD "Top-level exports" code block must be updated in the
  same commit — `tests/test_docs_smoke.py` asserts it matches `sift.__all__`.

---

## 5. Files

| File | Status | Contents | Est. size |
|------|--------|----------|-----------|
| `sift/estimators/knockoffs.py` | new | `fit_gaussian_knockoffs`, `sample_gaussian_knockoffs`, `GaussianKnockoffModel`, equi-s + shrinkage helpers | ~200 lines |
| `sift/selection/knockoff_filter.py` | new | statistic registry, pair screening, `knockoff_threshold`, orchestration `select_fdr`, `sample_knockoffs`, `KnockoffSelectionResult` | ~400 lines |
| `sift/selectors.py` | edit | `KnockoffSelector` wrapper | ~40 lines |
| `sift/__init__.py`, `sift/selection/__init__.py`, `sift/estimators/__init__.py` | edit | exports | small |
| `tests/test_knockoff_sampler.py` | new | construction/exchangeability/numerics | ~150 lines |
| `tests/test_knockoff_filter.py` | new | threshold, statistics, antisymmetry, validation, result object | ~250 lines |
| `tests/test_knockoff_fdr_control.py` | new | simulation-based FDR/power/null tests | ~120 lines |
| `benchmarks/bench_knockoffs.py` | new | timing harness | ~80 lines |
| `DOCS.MD`, `README.md`, `docs/user-guide.md`, `benchmarks/README.md`, `TODO.MD` | edit | see §10 | — |

Keep `KnockoffSelectionResult` in `knockoff_filter.py` unless it creates an
import cycle with `sift/selectors.py`; if it does, split into
`sift/selection/knockoff_result.py` (mirrors `selection/result.py`).

---## 6. Algorithms

### 6.1 s-vector and shrinkage (`fit_gaussian_knockoffs`)

Input `Sigma` is `cache.Rxx` (float32, unit diagonal, off-diagonals clipped to
±0.999999). All work in float64.

1. `lambda_min = scipy.linalg.eigh(Sigma, subset_by_index=[0, 0], eigvals_only=True)[0]`.
2. **Shrinkage** (numerical necessity when the cache has more features than
   rows — `Rxx` is then singular and `λ_min ≈ 0`, which would make knockoffs
   near-copies with zero power):
   - if `lambda_min >= min_eig`: `gamma = 0.0`, `Sigma_g = Sigma`.
   - else: `gamma = (min_eig - lambda_min) / (1.0 - lambda_min)`,
     `Sigma_g = (1 - gamma) * Sigma + gamma * I`, and emit a single
     `UserWarning` reporting `lambda_min` and `gamma` and pointing at the
     user-guide power caveats. `λ_min(Sigma_g) = min_eig` by construction.
   - `min_eig` must be a finite float in `(0, 1)`; reject bools (match the
     `corr_prune` validation style, `cefsplus.py:31`).
3. **Equi construction**: `s_j = min(2 * min_eig_or_lambda_min(Sigma_g), 1.0) * (1 - 1e-6)`
   for all `j` (the `1 − 1e-6` slack keeps `2Σ_g − D` strictly PD).
4. Operators, computed once and reused across draws:
   - Cholesky `L = cholesky(Sigma_g)`.
   - `V = cho_solve(L, D)` where `D = diag(s)` → `V = Σ_g⁻¹ D`.
   - `mean_op A = I − V`.
   - `N = 2D − D V`; symmetrize `N ← (N + Nᵀ)/2`.
   - `noise_chol L_N = cholesky(N)`, with escalating jitter on failure
     (`1e-12 · tr(N)/p`, ×10 up to 3 attempts), final fallback:
     `eigh`, clip negative eigenvalues to 0, `L_N = U diag(√λ)`.
5. Return `GaussianKnockoffModel(s, A, L_N, gamma, lambda_min)`.

The **shrunk** `Sigma_g` is the sampling model; the same `Sigma_g` must be
used for the analytic augmented matrix in 6.3 (consistency is what preserves
exchangeability).

### 6.2 Sampling (`sample_gaussian_knockoffs`)

```
E  = rng.standard_normal((n, p))          # float64, generated in row blocks
Z̃ = Z.astype(np.float64, per block) @ A + E @ L_N.T
return Z̃.astype(np.float32)
```

- Block over rows (e.g. 8192 rows) to bound peak float64 memory at
  `O(block · p)`.
- `rng` is a `np.random.Generator`; `select_fdr` derives per-draw generators
  via `np.random.SeedSequence(random_state).spawn(n_draws)` so draws are
  independent and the whole run is reproducible.
- Cost per draw: two `n × p × p` GEMMs — `O(n·p²)`. Documented in §8.

### 6.3 Feature statistics

All statistics receive the same inputs and must satisfy the same contract.

**Inputs**: `Z` (n×p float32), `Zt` (knockoffs, n×p float32), `zy`
(`weighted_rank_gauss_1d(y[cache.row_idx], cache.sample_weight)`), `w`
(cache weights), `Sigma_g`, `s`, `statistic_options`.

**Contract (enforced by tests, §9.2)**: `W = stat(...)` is a length-`p`
float64 vector such that swapping columns `Z[:, j] ↔ Zt[:, j]` for any single
`j` flips the sign of `W_j` and leaves `W_i (i ≠ j)` unchanged, holding all
seeds fixed. Features screened out or never entered get `W_j = 0` (zeros are
excluded from threshold candidates and can never be selected — conservative
and valid).

**Shared preamble** (orchestrator, not per-statistic):

1. `r = weighted_corr_with_vector(Z, zy, w)`; `rt = weighted_corr_with_vector(Zt, zy, w)`.
2. **Pair-coupled screening**: `pair_score_j = max(|r_j|, |rt_j|)`. Keep the
   top `m = min(p, screen_pairs)` pairs (both members). `screen_pairs=None`
   keeps all. Screening uses a symmetric function of each pair, so swap
   exchangeability of the retained set is preserved; unscreened features get
   `W_j = 0`.
3. Analytic augmented correlation over the `m` retained pairs, ordered
   `[originals..., knockoffs...]`:
   `G = [[Σ_m, Σ_m − D_m], [Σ_m − D_m, Σ_m]]` (2m×2m float64), where `Σ_m`
   is the submatrix of `Sigma_g`. Empirical target vector
   `r_aug = concat(r[kept], rt[kept])`.

**`relevance`** (cheapest): `W_j = gaussian_mi_from_corr(|r_j|) − gaussian_mi_from_corr(|rt_j|)`
computed on all `p` pairs (no screening needed — it is already O(np)).

**Greedy entry-order statistics** — `cefsplus` (default), `mrmr_diff`,
`mrmr_quot`, `jmi`, `jmim`: run the existing loop
(`cefsplus_loop`, `_gaussian_mrmr_select`, `_gaussian_jmi_select` in
`sift/selection/cefsplus.py`) on `(G, r_aug)` with:

- `k = path_depth`, default `min(2m, m)` = `m` (a path can contain at most
  `m` true features; deeper adds nothing), overridable via
  `statistic_options["path_depth"]`;
- **no `corr_prune`, no `top_m`** inside the augmented run (D5/D6 — the
  orchestrator's pair screening is the only prefilter);
- tie-break relevance = `gaussian_mi_from_corr(r_aug)` (a symmetric function
  of the data — safe).

Convert entry order to scores: a column entering at 0-based position `t` gets
`h = path_depth − t` (so first-in gets the largest score, never-entered gets
0), then `W_j = h(original_j) − h(knockoff_j)`.

**`lcd`** (lasso coefficient difference): weighted least squares via row
rescaling — `Zs = Z_aug * sqrt(w)[:, None]`, `ys = zy * sqrt(w)` — then
`sklearn.linear_model.LassoCV(cv=statistic_options.get("cv", 5),
random_state=<derived seed>, n_jobs=n_jobs)` on `(Zs, ys)`;
`W_j = |β_j| − |β_{j̃}|`. Columns are already standardized (copula transform),
so `fit_intercept=False`, no re-standardization. The `√w` trick avoids any
dependence on sklearn version support for `sample_weight` in `LassoCV`.

Registry: module-level `_KNOCKOFF_STAT_REGISTRY: dict[str, KnockoffStatSpec]`
mirroring `sift/scoring.py` (`ScoringSpec` / `get_scoring`,
`scoring.py:93-116`), with a frozen `KnockoffStatSpec(name, fn,
needs_screening: bool)` and `VALID_KNOCKOFF_STATISTICS` tuple. Unknown names
raise `ValueError` listing valid keys.

### 6.4 Threshold (`knockoff_threshold`)

```python
def knockoff_threshold(W, q, *, offset=1):
    W = np.asarray(W, dtype=np.float64)
    ts = np.unique(np.abs(W[W != 0.0]))
    for t in ts:                       # ascending; vectorize with cumsums in impl
        fdp = (offset + np.sum(W <= -t)) / max(1, np.sum(W >= t))
        if fdp <= q:
            return float(t)
    return float(np.inf)
```

- `offset=1` → knockoff+, exact FDR ≤ q. `offset=0` → modified FDR; allowed
  because it materially improves power at small p, documented as such.
- Validation: `q` float in open `(0, 1)`; `offset in {0, 1}`; reject bools.
- Implementation should be vectorized (sort + cumulative counts), but the
  spec-level semantics are the loop above; unit tests pin exact arithmetic.

### 6.5 Derandomization (`n_draws > 1`)

For draws `b = 1..B` (independent spawned seeds, optionally parallel via
`joblib.Parallel(prefer="threads")` — the work is BLAS-bound):

1. `Z̃_b = sample_gaussian_knockoffs(Z, model, rng_b)` (the
   `GaussianKnockoffModel` is fitted **once**; only noise is redrawn).
2. `W_b = stat(...)`; `τ_b = knockoff_threshold(W_b, q, offset)`;
   `S_b = {j : W_b[j] ≥ τ_b}`.
3. `π_j = (1/B) Σ_b 1[j ∈ S_b]`; final selection `{j : π_j ≥ eta}`.

Ordering of `selected_features`: descending `mean_b W_b[j]`.
`eta` validated in `(0, 1]`; `n_draws` a positive int (reject bools).
Docs state plainly: `n_draws=1` carries the exact knockoff+ guarantee;
derandomized selection trades that for run-to-run stability (Ren et al. give
guarantees in a related but not identical framework).

### 6.6 Orchestration (`select_fdr` flow)

```
validate scalars (q, offset, n_draws, eta, screen_pairs, statistic, s_method)
resolve cache:  X path → build_cache(compute_Rxx=True) | cache path → check Rxx (compute locally if None)
check y length vs cache.n_rows_original;  zy = weighted_rank_gauss_1d(y[cache.row_idx], cache.sample_weight)
model = fit_gaussian_knockoffs(cache.Rxx as float64, s_method, min_eig)
for each draw (seeded):  Z̃ → W → τ → S
aggregate (6.5 if B > 1)
map selected candidate indices → cache.valid_cols → cache.feature_names
build KnockoffSelectionResult + metadata; verbose one-line summary
   (e.g. "knockoff+ q=0.10: selected 37 features (threshold=0.0213, s̄=0.41)")
```

---

## 7. Correctness invariants and edge cases

Invariants (each maps to a test in §9):

- I1: `cov([Z, Z̃])` under the model equals `G`; empirically approximately.
- I2: Every statistic is swap-antisymmetric (contract in 6.3), verified
  programmatically per registry entry, not by code review.
- I3: The same `Sigma_g` (post-shrinkage) is used for sampling and for the
  analytic `G` in statistics.
- I4: Pair screening never separates an original from its knockoff.
- I5: `W_j = 0` features are excluded from threshold candidates and never
  selected.
- I6: Same `random_state` → byte-identical results; different draws within a
  run use independent spawned seeds.

Edge cases:

| Case | Behavior |
|------|----------|
| No `t` satisfies the bound | `threshold = inf`, empty selection, valid result object; verbose message explains that emptiness is informative |
| `p == 1` | Works (`λ_min = 1`, `s = 1`, knockoff is pure noise) |
| Constant / all-NaN columns | Already removed by `build_cache` (`valid_cols`); indices reported in original-X namespace as elsewhere |
| `cache.Rxx is None` | Computed locally, caller's cache not mutated |
| `sample_weight` degenerate | Rejected by `ensure_weights` (existing paths) |
| Binary `y` | Supported (rank-Gaussian gives a two-point `zy`; existing tests cover this transform) |
| Multiclass `y` | Not supported; documented (encode or run one-vs-rest), consistent with the Gaussian path generally |
| `k`-style arguments | None exist — `q` replaces `k`; docs draw the contrast explicitly |
| Duplicate feature names in `X` | Reject with `ValueError` (consistent with the recent duplicate-label hardening in evaluate/auto-k) |
| Internal knockoff column names | Never leak into results; diagnostics use a `"::knockoff"` suffix internally only |

---

## 8. Performance and memory budget

Dominant costs, `n` = cache rows (≤ `subsample`, default 50k), `p` = valid features, `m` = screened pairs:

| Step | Cost | Notes |
|------|------|-------|
| `λ_min`, Cholesky, `Σ⁻¹D` | `O(p³)` once | p = 2000 → well under a second; p = 5000 → seconds |
| Sampling per draw | `O(n·p²)` (two GEMMs) | The bottleneck. n = 50k, p = 2000 → ~4·10¹¹ flops ≈ tens of seconds float64; row-blocked. Users with big p should lower `subsample`; note in docs |
| `r`, `rt` | `O(n·p)` | negligible |
| Greedy statistic | `O((2m)² · path_depth)` | m = 2000, depth = 2000 → same order as one `select_cached` call |
| `lcd` | LassoCV on `n × 2m` | the slowest statistic; documented |
| Memory | `G` is `(2m)²` float64 → m = 2000 ⇒ 128 MB; `Z̃` is n×p float32 (same as `Z`) | `screen_pairs` default 2000 exists precisely to bound `G` |

Float discipline: `Z`/`Z̃` float32; all p×p algebra float64 (matches
`weighted_correlation_matrix` conventions).

Benchmark harness `benchmarks/bench_knockoffs.py`: grid over
`p ∈ {500, 2000}`, `n = 50_000`, statistics `{relevance, cefsplus}`,
`n_draws ∈ {1, 11}`; report fit/sample/stat/threshold timings, in the style
of `bench_mrmr.py`. Soft target: single-draw `cefsplus` run at
`n=50k, p=2000` completes in under ~60 s on a laptop.

---

## 9. Testing plan

### 9.1 `tests/test_knockoff_sampler.py`

1. **Moment check**: `n=20_000, p=10`, random well-conditioned Σ; empirical
   covariance of `[Z, Z̃]` matches analytic `G` entrywise within 0.05.
2. **Equi s**: hand-checked λ_min on a 3×3 matrix → exact expected `s`.
3. **Shrinkage trigger**: p > n cache → `gamma > 0`, `UserWarning` raised,
   `λ_min(Σ_g) ≈ min_eig`.
4. **Cholesky fallback**: near-singular `N` (s at its cap) still returns a
   valid `L_N` (no crash, `N ≈ L_N L_Nᵀ`).
5. **Determinism**: same seed → identical `Z̃`; spawned seeds differ across draws.
6. **Validation**: bad `min_eig` (0, 1, bool, inf), unknown `s_method` raise.

### 9.2 `tests/test_knockoff_filter.py`

1. **Threshold arithmetic**: hand-crafted `W` vectors pin exact `τ` for
   `offset ∈ {0, 1}`, including the no-valid-`t` → `inf` case and the
   all-zeros case.
2. **Antisymmetry (the load-bearing test)**: for every registry statistic,
   on a fixed small problem, swap one pair's columns and assert `W_j` flips
   sign exactly and all other entries are unchanged. Run for a null `j` and a
   signal `j`.
3. **Pair-coupled screening**: construct `r`/`rt` where naive top-`|r|`
   screening would split a pair; assert both members retained/dropped together.
4. **Entry-order scores**: never-entered → 0; first-entered gets max; `W`
   signs match construction.
5. **`corr_prune` absence**: near-duplicate original/knockoff pairs still
   both appear in the augmented path (no silent pruning).
6. **Validation**: `q ∉ (0,1)`, bool `q`, `offset=2`, `n_draws=0`,
   `eta=0`/`eta>1`, unknown `statistic`, `X` and `cache` both/neither,
   `sample_weight` with `cache`, `y` length mismatch, duplicate column names —
   all raise `ValueError` with clear messages.
7. **Result object**: metadata keys present; `get_feature_ranking()` columns
   and ordering; `selection_frequency` None vs populated by `n_draws`;
   `selected_indices` map through `valid_cols` correctly when constant
   columns were dropped.
8. **Weights**: weighted vs unweighted runs differ on a weight-sensitive
   construction; weight scaling by a constant is a no-op (existing
   normalization guarantees).
9. **Empty result semantics**: pure-noise `y` at tight `q` → empty selection,
   `threshold=inf`, no exception.
10. **`KnockoffSelector`**: fit/transform round-trip, `get_support`,
    sklearn `clone` compatibility.

### 9.3 `tests/test_knockoff_fdr_control.py` (simulation; keep < ~60 s total)

Fixed-seed synthetic Gaussian designs, `n=800, p=40`, 8 true features with
strong coefficients, AR(1) correlation ρ=0.5:

1. **FDR**: 30 seeded replications at `q=0.2`, `statistic="cefsplus"`:
   mean FDP ≤ 0.30 (generous margin over the 0.2 guarantee to kill flake).
2. **Power**: mean power ≥ 0.6 on the same runs.
3. **Global null**: y independent of X; knockoff+ implies
   `P(any selection) ≤ q`; assert ≤ 40% of 30 seeds select anything at `q=0.2`.
4. **Derandomization**: `n_draws=11` selection is a subset-ish, more stable
   set — assert Jaccard similarity between two derandomized runs (different
   `random_state`) ≥ that of two single-draw runs, on a fixed design.
5. Repeat (1) with `statistic="lcd"` at reduced replication count.

### 9.4 Existing-suite guarantees

No changes to existing behavior; full suite stays green
(`python -m pytest -ra`, currently 274 passed / 10 skipped locally, 336 in
the darko env). `test_docs_smoke.py` forces the DOCS.MD export block update.

---

## 10. Documentation

- **DOCS.MD**: add exports to the "Top-level exports" block (test-enforced);
  full API section for `select_fdr` / `KnockoffSelector` /
  `sample_knockoffs` / `knockoff_threshold`; a row in the selector support
  matrix (Gaussian-path only; weights: yes; classification: binary only;
  `k`: n/a — controlled by `q`).
- **README.md**: one row in Main Components ("FDR-controlled selection —
  `select_fdr`, `KnockoffSelector`") and a 5-line quickstart snippet.
- **docs/user-guide.md**: new section covering — what the guarantee means and
  what it is conditional on (Gaussian copula, second-order); choosing `q`;
  `n_draws`/`eta` guidance; why an empty result is an answer; the
  `s_mean`/`gamma` diagnostics and the correlated-features power caveat
  (recommend pre-pruning near-duplicates via `corr_prune`-style dedup before
  building the cache, until group knockoffs land); contrast with `k="auto"`
  ("auto-k asks how many features help prediction; knockoffs ask how many you
  can trust — use both").
- **docs/architecture.md**: one paragraph on the two new modules and the
  copula-cache dependency.
- **TODO.MD**: new tracked item with phase checklist.

---

## 11. Phasing

**Phase 1 (this spec)**: everything above — equi + shrinkage, statistics
registry (`cefsplus`, `mrmr_diff`, `mrmr_quot`, `jmi`, `jmim`, `relevance`,
`lcd`), knockoff/knockoff+ thresholds, derandomization, result object,
wrapper class, tests, benchmarks, docs.

Suggested implementation order (each step lands with its tests):
1. `estimators/knockoffs.py` + sampler tests (9.1).
2. `knockoff_threshold` + arithmetic tests.
3. Statistics registry + antisymmetry/screening tests (9.2.2–9.2.5).
4. `select_fdr` orchestration + validation/result tests.
5. Simulation tests (9.3), tuning default `screen_pairs`/`path_depth` if needed.
6. `KnockoffSelector`, exports, DOCS.MD/README/user-guide, benchmarks.

**Phase 2 (out of scope, interfaces reserved)**:
- `s_method="mvr"` / `"me"` via coordinate descent (Spector & Janson) — no
  SDP dependency; large power win on correlated designs.
- **Group knockoffs**: `groups=` parameter, cluster near-duplicates
  (reusing `Rxx`-threshold clustering), group-level FDR — the principled fix
  for the equicorrelated power collapse on collinear data.
- Raw-space knockoffs via per-column inverse empirical CDF, enabling
  tree-model statistics (`statistic="catboost"`, Boruta-style importance W).
- Auto-k cross-check: surface "auto-k chose k=40 but only 12 features survive
  q=0.2" in `FeaturePathEvaluationResult` diagnostics.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Power collapse when features are highly collinear (`λ_min ≈ 0` ⇒ `s ≈ 2·min_eig`, knockoffs ≈ copies, all `W ≈ 0`) | Shrinkage floor + `UserWarning`; `s_mean`/`gamma`/`lambda_min` surfaced in metadata; user-guide guidance to dedup near-duplicates first; group knockoffs in Phase 2. This is a *conservative* failure (too few selections), never an FDR violation. |
| Statistic silently breaks exchangeability (the classic knockoff implementation bug) | The swap-antisymmetry contract is tested programmatically per registry entry (9.2.2); D5/D6 remove the two mechanisms (screening, pruning) most likely to break it. |
| Second-order/copula approximation invalid for exotic joint dependence | Documented honestly everywhere the guarantee is mentioned; this is the standard practical regime for Model-X. |
| Runtime on very wide caches (`O(n·p²)` per draw) | Row-blocked GEMMs, one-time model fit reused across draws, `subsample` guidance, benchmark harness to keep numbers honest. |
| Simulation tests flake in CI | Fixed seeds, generous margins (FDP ≤ 0.30 vs guarantee 0.20), small designs, bounded runtime. |

---

## 13. Acceptance criteria

1. New test files pass; full suite green in both the base and darko
   environments; no new warnings under `pytest -ra`.
2. Antisymmetry test passes for **every** registered statistic.
3. Simulation: empirical mean FDP within margin at `q=0.2`, power ≥ 0.6 on
   the reference design; global-null selection rate within bound.
4. `test_docs_smoke.py` passes with the new exports (DOCS.MD updated).
5. Benchmark: single-draw `cefsplus` at `n=50k, p=2000` under the soft
   target (~60 s laptop); numbers recorded in `benchmarks/README.md`.
6. No new runtime dependencies; `pip install -e .` unchanged.
7. Deterministic: fixed `random_state` reproduces selections byte-for-byte.
