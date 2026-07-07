# Implementation Spec: FDR-Controlled Selection via Model-X Knockoffs

Status: implemented (as-built)
Target version: 0.7.0
Depends on: `FeatureCache` / Gaussian copula machinery (`sift/estimators/copula.py`),
cached Gaussian selectors (`sift/selection/cefsplus.py`), shared validation
(`sift/_preprocess.py`), result conventions (`sift/selection/result.py`).

---

## 1. Summary

Add a knockoff filter to sift: a `select_fdr(X, y, q=...)` entry point that
returns a feature set calibrated by a target false-discovery rate rather than
a ranked top-k list. In the default implementation this is an **approximate
second-order Gaussian-copula knockoff filter**: the usual knockoff+ guarantee
is exact only when the Gaussian-copula feature model used by the sampler is the
true feature distribution and the statistic is swap-antisymmetric. With
estimated `Rxx`, shrinkage, sample weights, or derandomization, report the
result as plug-in/approximate rather than exact finite-sample FDR control.

The implementation is **second-order Gaussian Model-X knockoffs built in
copula space**: knockoff copies are sampled from the Gaussianized feature
matrix `Z` and correlation matrix `Rxx` that `build_cache` already produces.
Feature statistics `W_j` are computed from the augmented matrix `[Z | Z̃]`.
The safe core statistic is marginal relevance; tie-safe CEFS+ is enabled through
a pair-aware wrapper, while lasso coefficient differences and the remaining
sift-native greedy entry-order statistics stay reserved until their
implementations satisfy the knockoff antisymmetry contract, including exact tie
cases. The knockoff+ threshold converts `W` into a q-calibrated selection set.
Derandomization (multiple knockoff draws, selection-frequency aggregation)
reuses the same machinery and is explicitly approximate in v1.

No new dependencies. numpy/scipy/scikit-learn (already hard deps) suffice.

### As-built deviations from the initial v1 plan

- `statistic="cefsplus"` shipped enabled in v1 through the tie-safe,
  pair-aware wrapper. Its W magnitudes are objective gains rather than path
  ranks, its default path depth is capped at 10 screened pairs, and it supports
  opt-in early stopping via `min_gain_ratio`.
- `feature_groups` shipped in v1 as group-level thresholding over
  feature-level knockoffs. This is not true block-S group knockoff sampling.
- `s_method="mvr"` and `"me"` use diagonal coordinate-descent optimizers for
  the MVR and maximum-entropy objectives rather than the interim scaled
  conditional-variance heuristic.
- `lcd`, `mrmr_diff`, `mrmr_quot`, `jmi`, and `jmim` remain reserved until their
  exact-tie antisymmetry contracts are proven.
- Knockoff noise generation uses NumPy float32 standard-normal draws in 0.7.0;
  seeded outputs can differ from pre-release builds.

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

Honest framing to carry into all docs: FDR control is exact only under the
Gaussian-copula model actually used to sample the knockoffs, with valid
antisymmetric statistics and no model-changing regularization. The default
cached path estimates that model from data, may shrink it, and may use sample
weights; those are useful practical choices, but their result is an approximate
plug-in second-order knockoff filter. This is the standard practical regime in
the literature, but the API and docs must not advertise it as exact Model-X
control.

---

## 3. Design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Knockoffs are sampled **in copula space** (`Z`), never inverse-transformed to raw feature space (v1). | Monotone per-column transforms preserve feature identity; every supported statistic operates on `Z` anyway. Raw-space push-back is only needed for tree-model statistics — Phase 2. |
| D2 | `s`-vector via **equicorrelated**, **MVR**, or **ME** diagonal constructions plus automatic correlation shrinkage. | `"equi"` is fastest and closed-form. `"mvr"`/`"me"` use diagonal coordinate descent for the MVR and maximum-entropy objectives and are kept only when they improve the corresponding loss over equicorrelated. Shrinkage is a numerical/modeling approximation: when `gamma > 0`, metadata and docs must say exact Model-X FDR is not claimed. |
| D3 | The augmented feature-feature correlation matrix is the **analytic** `G = [[Σ_g, Σ_g−D], [Σ_g−D, Σ_g]]`, not re-estimated from `[Z | Z̃]`. | Free (no `O(n·p²)` pass), lower variance, and swap-invariant under the chosen Gaussian model. `Σ_g` is the post-shrinkage sampling covariance stored in the knockoff model; it is not necessarily the empirical cache covariance. Feature–target correlations are empirical (they must be — that's where `y` enters). |
| D4 | Feature statistics come from a small **registry**: `relevance` (default, marginal Gaussian-MI difference), `lcd` (lasso coefficient difference after tie/non-unique-solution checks), and `cefsplus`, `mrmr_diff`, `mrmr_quot`, `jmi`, `jmim` implemented through knockoff-specific, pair-aware greedy wrappers. | Existing cached greedy loops are not reused directly: their first-occurrence tie behavior favors originals under `[originals..., knockoffs...]` ordering and breaks antisymmetry. Sift-native statistics remain the differentiator, but they ship only once exact tie cases are side-neutral. Registry mirrors `sift/scoring.py` conventions. |
| D5 | Screening for expensive statistics is **pair-coupled**: rank pairs by `max(|r_j|, |r̃_j|)`, keep or drop originals and knockoffs together. | Any symmetric function of the pair preserves exchangeability; screening that could split a pair would silently break the FDR guarantee. |
| D6 | `corr_prune` is **never applied** inside the augmented selection run. | Pruning a knockoff against its own original (they are highly correlated by design when `s` is small) changes entry-order semantics; disabling is the conservative, obviously-correct choice. |
| D7 | `n_draws > 1` gives **derandomized knockoffs** (per-draw knockoff+ selection at level `q`, aggregate by selection frequency `π_j ≥ eta`). | Single-draw knockoffs are randomized — two runs can select different sets. Derandomization is the standard fix; only `n_draws=1` is eligible for classical knockoff+ guarantees, and only under a valid model/statistic. |
| D8 | `y` handling matches `select_cached`: `zy = weighted_rank_gauss_1d(y)`. Numeric continuous and binary targets are supported; integer labels with 3-20 unique values warn because categorical multiclass users should run one-vs-rest explicitly. | Consistency with every other Gaussian-path selector; no `task` parameter to invent semantics for. |
| D9 | All weighting flows through the existing conventions: `Σ` is the weighted correlation matrix, `zy`/`r` are weighted, `lcd` uses the `√w` row-rescaling trick. Knockoff noise rows are i.i.d. Columns with zero positive-weight variance are locally dropped/zeroed before the FDR path trusts `Rxx`. | There is no exact "weighted Model-X" theory; treating weights as importance weights in estimation and statistics is the pragmatic choice, stated plainly in docs. |
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
    statistic: str = "relevance",      # registry key, see 6.3
    n_draws: int = 1,
    eta: float = 0.5,                  # derandomization frequency threshold
    offset: int = 1,                   # 1 = knockoff+, 0 = knockoff/mFDR-style
    s_method: str = "equi",            # "equi", "mvr", or "me"
    min_eig: float = 1e-3,             # shrinkage floor for λ_min(Σ)
    screen_pairs: int | None = 2000,   # pair-coupled screening cap; None = all pairs
    statistic_options: dict | None = None,   # e.g. {"path_depth": ..., "cv": 5}
    feature_groups: Sequence[Any] | None = None, # optional group-level thresholding
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
  existing check pattern from `select_cached`, `cefsplus.py:376`). Always
  convert with `y_arr = to_numpy(y, dtype=np.float32).ravel()` before slicing;
  never use `y[cache.row_idx]`, because pandas Series integer indexing can be
  label-based.
- When `X` is given, internally call
  `build_cache(X, sample_weight=..., subsample=..., random_state=...,
  compute_Rxx=True, n_jobs=n_jobs)`.
- When `cache.Rxx is None`, compute
  `weighted_correlation_matrix(cache.Z, cache.sample_weight)` locally (do
  **not** mutate the caller's cache object) and note it once under `verbose`.
- Before fitting knockoffs, compute weighted variance of `cache.Z` under
  `cache.sample_weight`. Any valid-cache column with zero positive-weight
  variance is excluded from the active knockoff model and reported with
  `W_j = 0`/unselected in the final table. If no active columns remain, raise
  `ValueError`.
- Reject duplicate non-synthetic `feature_names` for both `X` and prebuilt
  `cache` inputs. Synthetic names (`x0`, `x1`, ...) are already unique by
  construction.
- Returns a `KnockoffSelectionResult` (4.3). Never returns a bare list —
  threshold, W table, and frequencies are the point of the feature.

### 4.2 Advanced building blocks (exported for power users)

```python
# sift/estimators/knockoffs.py

@dataclass(frozen=True)
class GaussianKnockoffModel:
    """Precomputed sampling operators for one (Σ, s) pair."""
    s: np.ndarray            # (p,) knockoff s-vector
    Sigma_g: np.ndarray      # post-shrinkage sampling covariance, float64 (p, p)
    mean_op: np.ndarray      # A = I − Σ⁻¹ D, float64 (p, p)
    noise_chol: np.ndarray   # L_N with L_N @ L_N.T = 2D − D Σ⁻¹ D, float64 (p, p)
    gamma: float             # shrinkage actually applied
    lambda_min: float        # smallest eigenvalue of the *unshrunk* Rxx

def fit_gaussian_knockoffs(
    Sigma: np.ndarray, *, s_method: str = "equi", min_eig: float = 1e-3,
) -> GaussianKnockoffModel

def sample_gaussian_knockoffs(
    Z: np.ndarray, model: GaussianKnockoffModel, rng: np.random.Generator,
    *, mean: np.ndarray | None = None,
) -> np.ndarray            # Z̃, float32, same shape as Z

def gaussian_knockoff_mean(
    Z: np.ndarray, model: GaussianKnockoffModel,
) -> np.ndarray            # deterministic conditional mean for repeated draws


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
    selected_features: List[Any]          # ordered by descending W (draw-aggregated)
    selected_indices: Optional[List[int]] # original-X column indices (via cache.valid_cols)
    selector_metadata: Dict[str, Any]     # see below
    W: pd.DataFrame                       # per-feature statistics, all draws (long or wide)
    threshold: Optional[float]            # knockoff+ threshold (n_draws == 1), else None
    selection_frequency: Optional[pd.Series]  # π_j indexed by feature (n_draws > 1), else None
    diagnostics_: Optional[Dict[str, Any]] = None

    def get_feature_ranking(self) -> pd.DataFrame:
        # columns: feature, W (mean across draws), rank, selected,
        #          selection_frequency (NaN when n_draws == 1), selected_index,
        #          relevance, selector
```

`selector_metadata` extends `build_selector_metadata`-style keys
(`selector="knockoff_fdr"`, `n_features`) with: `q`, `offset`, `statistic`,
`s_method`, `n_draws`, `eta`, `screen_pairs`, `path_depth`, `gamma`
(shrinkage applied), `lambda_min`, `s_mean` (mean of the s-vector — the
single best power proxy), `random_state`, `n_rows_used`, `fdr_control`
(`"exact_modelx"` only for caller-supplied known valid models in a future mode;
`"approximate_plugin"` for v1), `validity_model` (`"gaussian_copula_plugin"`),
`weighted_model` (bool), and `n_zero_weight_variance_features`.

`diagnostics_` holds the per-draw thresholds and per-draw selection sets.

### 4.4 Sklearn-style wrapper

`KnockoffSelector` in `sift/selectors.py`: subclass `_BaseSelector` only to
reuse categorical preprocessing, `transform`, `fit_transform`, and support-mask
helpers, but override the q-based fit path directly. The existing
`_BaseSelector._fit_impl` / `_fit_selector` path assumes `self.k`, passes
`k=...`, and forces `return_result=True`; `select_fdr` is q-based and already
returns a result object. Constructor mirrors `select_fdr` keyword args plus the
standard selector-class categorical parameters (`cat_features`, `cat_encoding`,
`allow_full_data_target_encoding`, and `loo_*` where applicable). The custom
fit path calls `_fit_transform_categoricals`, disables selector-function
categorical handling on the encoded matrix, invokes `select_fdr(q=self.q, ...)`,
and stores `selected_features_`, `selected_indices_`, `result_`,
`feature_names_in_`, and `n_features_in_`. Reset/exception paths clear all of
those attributes, including `result_`.

### 4.5 Export surface

- `sift/__init__.py` `__all__` additions: `select_fdr`,
  `KnockoffSelectionResult`, `KnockoffSelector`, `sample_knockoffs`.
- `sift/api.py`: import/re-export `select_fdr`, `KnockoffSelectionResult`, and
  `sample_knockoffs` if top-level exports continue to flow through `sift.api`.
- `sift/selection/__init__.py`: `select_fdr`, `knockoff_threshold`,
  `KnockoffSelectionResult`.
- `sift/estimators/__init__.py`: add only the lazy submodule name
  `"knockoffs"` to package-level `__all__`. Do not add function/class names
  there unless the lazy `__getattr__` is also rewritten; today it treats every
  `__all__` entry as a submodule. The functions/classes are exported from
  `sift.estimators.knockoffs.__all__` and imported directly where top-level
  `sift` needs them.
- **Required**: DOCS.MD "Top-level exports" code block must be updated in the
  same commit — `tests/test_docs_smoke.py` asserts it matches `sift.__all__`.

---

## 5. Files

| File | Status | Contents | Est. size |
|------|--------|----------|-----------|
| `sift/estimators/knockoffs.py` | new | `fit_gaussian_knockoffs`, `sample_gaussian_knockoffs`, `GaussianKnockoffModel`, equi-s + shrinkage helpers | ~200 lines |
| `sift/selection/knockoff_filter.py` | new | statistic registry, pair screening, `knockoff_threshold`, orchestration `select_fdr`, `sample_knockoffs`, `KnockoffSelectionResult` | ~400 lines |
| `sift/selectors.py` | edit | `KnockoffSelector` wrapper | ~40 lines |
| `sift/api.py`, `sift/__init__.py`, `sift/selection/__init__.py`, `sift/estimators/__init__.py` | edit | exports | small |
| `tests/test_knockoff_sampler.py` | new | construction/exchangeability/numerics | ~150 lines |
| `tests/test_knockoff_filter.py` | new | threshold, statistics, antisymmetry, validation, result object | ~250 lines |
| `tests/test_knockoff_fdr_control.py` | new | simulation-based FDR/power/null tests | ~120 lines |
| `benchmarks/bench_knockoffs.py` | new | timing harness | ~80 lines |
| `DOCS.MD`, `README.md`, `docs/user-guide.md`, `benchmarks/README.md`, `TODO.MD` | edit | see §10 | — |

Keep `KnockoffSelectionResult` in `knockoff_filter.py` unless it creates an
import cycle with `sift/selectors.py`; if it does, split into
`sift/selection/knockoff_result.py` (mirrors `selection/result.py`).

---

## 6. Algorithms

### 6.1 s-vector and shrinkage (`fit_gaussian_knockoffs`)

Input `Sigma` is `cache.Rxx` (float32, unit diagonal, off-diagonals clipped to
±0.999999). Start with `Sigma = np.asarray(Sigma, dtype=np.float64)`; all
linear algebra is float64.

1. `lambda_min = scipy.linalg.eigh(Sigma, subset_by_index=[0, 0], eigvals_only=True)[0]`.
2. **Shrinkage** (numerical necessity when the cache has more features than
   rows — `Rxx` is then singular and `λ_min ≈ 0`, which would make knockoffs
   near-copies with zero power):
   - if `lambda_min >= min_eig`: `gamma = 0.0`, `Sigma_g = Sigma`.
   - else: `gamma = (min_eig - lambda_min) / (1.0 - lambda_min)`,
     `Sigma_g = (1 - gamma) * Sigma + gamma * I`, and emit a single
     `UserWarning` reporting `lambda_min` and `gamma` and pointing at the
     user-guide validity/power caveats. `λ_min(Sigma_g) = min_eig` by
     construction. The warning must say this is a plug-in approximation that
     changes the covariance model; it is not merely a conservative power fix.
   - `min_eig` must be a finite float in `(0, 1)`; reject bools (match the
     `corr_prune` validation style, `cefsplus.py:31`).
3. **`s` construction**:
   - `"equi"`: `s_j = min(2 * min_eig_or_lambda_min(Sigma_g), 1.0) * (1 - 1e-6)`
     for all `j` (the `1 − 1e-6` slack keeps `2Σ_g − D` strictly PD).
   - `"mvr"` / `"me"`: run diagonal coordinate descent on the MVR or
     maximum-entropy objective with Sherman-Morrison inverse updates, keep the
     solution only if it improves the chosen loss over equi, and fall back to
     equi if the final `2Σ_g − diag(s)` check is not feasible.
4. Operators, computed once and reused across draws:
   - Cholesky factorization `cf = cho_factor(Sigma_g, lower=True)`.
   - `V = cho_solve(cf, D)` where `D = diag(s)` → `V = Σ_g⁻¹ D`.
   - `mean_op A = I − V`.
   - `N = 2D − D V`; implement the diagonal multiplication as
     `2D - s[:, None] * V` rather than materializing a dense `D @ V`.
     Symmetrize `N ← (N + Nᵀ)/2`.
   - `noise_chol L_N = cholesky(N, lower=True)`, so
     `L_N @ L_N.T = N`. This orientation is required because sampling uses
     `E @ L_N.T`; the scipy default is upper-triangular and would give the
     wrong covariance. Use escalating jitter on failure
     (`1e-12 · tr(N)/p`, ×10 up to 3 attempts), final fallback:
     `eigh`, clip negative eigenvalues to 0, `L_N = U diag(√λ)`.
5. Return `GaussianKnockoffModel(s, Sigma_g, A, L_N, gamma, lambda_min)`.

The **shrunk** `Sigma_g` is the sampling model; the same `Sigma_g` must be
used for the analytic augmented matrix in 6.3 (consistency is what preserves
exchangeability under the chosen model). Do not rebuild `G` from the unshrunk
cache covariance after fitting the model.

### 6.2 Sampling (`sample_gaussian_knockoffs`)

```
E  = rng.standard_normal((n, p))          # generated in row blocks
Z̃ = Z.astype(np.float32, per block) @ A32 + E32 @ L_N32.T
return Z̃.astype(np.float32)
```

`L_N` must be any factor satisfying `L_N @ L_N.T = N`, not the default upper
factor from `scipy.linalg.cholesky`.

- Block over rows (e.g. 8192 rows) to bound peak working memory at
  `O(block · p)`.
- Keep model operators in float64, but cast them once per sampling call for
  float32 GEMMs because `Z` and the returned knockoffs are float32.
- For `n_draws > 1`, precompute the deterministic conditional mean
  `Z @ (I - Σ_g^-1D)` once with `gaussian_knockoff_mean` and redraw only the
  fresh noise term.
- `rng` is a `np.random.Generator`; `select_fdr` derives per-draw generators
  via `np.random.SeedSequence(random_state).spawn(n_draws)` so draws are
  independent and the whole run is reproducible.
- Cost per draw: two `n × p × p` GEMMs — `O(n·p²)`. Documented in §8.

### 6.3 Feature statistics

All statistics receive a shared context and must satisfy the same contract.

```python
@dataclass(frozen=True)
class KnockoffStatContext:
    Z: np.ndarray              # active originals, n x p_active
    Zt: np.ndarray             # active knockoffs, n x p_active
    zy: np.ndarray             # weighted_rank_gauss_1d(y_arr[cache.row_idx], w)
    w: np.ndarray
    model: GaussianKnockoffModel
    r: np.ndarray              # corr(Z_j, zy), length p_active
    rt: np.ndarray             # corr(Zt_j, zy), length p_active
    kept: np.ndarray           # active pair indices retained after screening
    G: np.ndarray              # analytic augmented correlation over kept pairs
    r_aug: np.ndarray          # concat(r[kept], rt[kept])
    options: dict
    n_jobs: int
    rng: np.random.Generator
```

**Contract (enforced by tests, §9.2)**: `W = stat(...)` is a length-`p`
float64 vector in the active-feature namespace such that swapping columns
`Z[:, j] ↔ Zt[:, j]` for any single `j` flips the sign of `W_j` and leaves
`W_i (i ≠ j)` unchanged, holding all seeds fixed. Features screened out or
never entered get `W_j = 0` (zeros are excluded from threshold candidates and
can never be selected — conservative and valid). The orchestrator scatters the
active-feature `W` back to the full valid-cache feature table; inactive
zero-weight-variance features stay at `W_j = 0`.

**Shared preamble** (orchestrator, not per-statistic):

1. Work only on the active feature mask that survived the weighted-variance
   check in §4.1. Non-active valid-cache features get `W_j = 0` in the final
   all-feature result table.
2. If `zy` has zero weighted variance, return all-zero `W` immediately for
   every statistic. This prevents order-only selection on constant targets.
3. Normalize options once: `options = statistic_options or {}`. No statistic
   should subscript `None`.
4. `r = weighted_corr_with_vector(Z, zy, w)`; `rt = weighted_corr_with_vector(Zt, zy, w)`.
5. **Pair-coupled screening**: `pair_score_j = max(|r_j|, |rt_j|)`. Keep the
   top `m = min(p, screen_pairs)` pairs (both members). `screen_pairs=None`
   keeps all. Screening uses a symmetric function of each pair, so swap
   exchangeability of the retained set is preserved; unscreened features get
   `W_j = 0`.
6. Analytic augmented correlation over the `m` retained pairs, ordered
   `[originals..., knockoffs...]`:
   `G = [[Σ_m, Σ_m − D_m], [Σ_m − D_m, Σ_m]]` (2m×2m float64), where `Σ_m`
   is the submatrix of `model.Sigma_g`. Empirical target vector
   `r_aug = concat(r[kept], rt[kept])`.
7. Build `KnockoffStatContext` with these precomputed arrays. Registry
   functions must use `context.G`, `context.r_aug`, and `context.kept` rather
   than recomputing augmented correlations or losing the scatter map.

**`relevance`** (cheapest): `W_j = gaussian_mi_from_corr(|context.r[j]|) −
gaussian_mi_from_corr(|context.rt[j]|)` computed on all active pairs (no
screening needed — it is already O(np)).

**Greedy entry-order statistics** — `cefsplus`, `mrmr_diff`, `mrmr_quot`,
`jmi`, `jmim`: use knockoff-specific wrappers around the existing scoring
logic, not the existing loops directly. The current loops break exact ties by
array order; under `[originals..., knockoffs...]` that favors originals and
can produce positive null `W` on zero-signal targets. The wrappers must:

- use pair-aware, side-neutral tie handling. If an original/knockoff pair is
  tied at the decision boundary, assign the same entry score to both sides or
  neutralize the pair's `W_j` to zero. Never let lower augmented column index
  decide original-vs-knockoff ties;
- return all-zero `W` when `r_aug` is all zero after screening;
- pass the antisymmetry test for exact ties, not only generic random data;
- use `k = path_depth`, default `m`, overridable via
  `context.options["path_depth"]`. Larger depths can change
  `h(original_j) - h(knockoff_j)` when a counterpart enters after depth `m`;
  `m` is a cost/power default, not a semantics-preserving limit;
- **no `corr_prune`, no `top_m`** inside the augmented run (D5/D6 — the
  orchestrator's pair screening is the only prefilter);
- tie-break relevance may use `gaussian_mi_from_corr(r_aug)`, but exact ties
  still require the side-neutral policy above.

Convert entry order to scores: a column entering at 0-based position `t` gets
`h = path_depth − t` (so first-in gets the largest score, never-entered gets
0), then `W_j = h(original_j) − h(knockoff_j)`.

**`lcd`** (lasso coefficient difference): coordinate descent is order-sensitive,
so one direct `LassoCV` run on `[Z_m, Zt_m]` is not acceptable. Use a
symmetrized two-run statistic:

1. Build `Z_aug_1 = [Z_m, Zt_m]` and `Z_aug_2 = [Zt_m, Z_m]` for the retained
   pairs.
2. Weighted-center `Z_aug_*` and `zy` under `w`, then row-rescale:
   `Zs = Z_aug_centered * sqrt(w)[:, None]`, `ys = zy_centered * sqrt(w)`.
   This avoids relying on generated knockoff noise having exactly zero
   empirical weighted mean.
3. Run `sklearn.linear_model.LassoCV(cv=options.get("cv", 5),
   random_state=<derived seed>, n_jobs=inner_n_jobs, fit_intercept=False)` on
   both orderings with identical CV splits / alpha grid.
4. If `β1` is from `[Z_m, Zt_m]` and `β2` is from `[Zt_m, Z_m]`, compute
   `W_kept[j] = 0.5 * (abs(β1[j]) - abs(β1[j+m])) +
                0.5 * (abs(β2[j+m]) - abs(β2[j]))`.
5. Scatter `W_kept` back through `context.kept`; unkept active features get
   zero.

`lcd` is enabled only after passing the antisymmetry tests with numerical
tolerance (`np.allclose(..., atol=1e-10)` rather than bitwise equality), because
the two symmetrized fits can differ by floating-point summation order.

Registry: module-level `_KNOCKOFF_STAT_REGISTRY: dict[str, KnockoffStatSpec]`
mirroring `sift/scoring.py` (`ScoringSpec` / `get_scoring`,
`scoring.py:93-116`), with a frozen `KnockoffStatSpec(name, fn,
needs_screening: bool)` and `VALID_KNOCKOFF_STATISTICS` tuple. Unknown names
raise `ValueError` listing valid keys.

`cefsplus` is implemented through this wrapper and is enabled. `mrmr_diff`,
`mrmr_quot`, `jmi`, and `jmim` remain reserved until their pair-aware wrappers
pass the same exact-tie antisymmetry tests.

### 6.3.1 Feature-group thresholding

`feature_groups` provides optional group-level knockoff selection. The sampler
still builds feature-level knockoffs; after each draw, feature-level `W` values
are collapsed to group statistics by the largest absolute `W` in each group,
with exact positive/negative boundary ties neutralized to zero. The knockoff
threshold is applied to the group `W` vector, and selected groups are expanded
back to active member features in the result table.

This is a group-discovery mode: it is useful for one-hot families, lags, spline
bases, embeddings, and interactions, but docs must not describe it as exact
feature-level FDR control inside a selected group.

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

- `offset=1` → knockoff+. Exact FDR ≤ q only when the knockoff model and
  statistic are valid; in the default plug-in path, report q-calibrated
  approximate control. `offset=0` → modified FDR; allowed because it materially
  improves power at small p, documented as such.
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
Treat `n_jobs` as a total parallelism budget. If outer draws are parallelized,
set inner statistic parallelism (`LassoCV.n_jobs`, joblib subcalls, and any
threadpool-limited BLAS scope available through sklearn/threadpoolctl) to 1 to
avoid oversubscription; if a statistic owns internal parallelism, run draws
serially or split the budget explicitly.
Docs state plainly: `n_draws=1` is the only mode eligible for the classical
knockoff+ guarantee, and only under a valid feature model/statistic. The v1
derandomized selection trades exactness for run-to-run stability (Ren et al.
give guarantees in a related but not identical framework).

### 6.6 Orchestration (`select_fdr` flow)

```
validate scalars (q, offset, n_draws, eta, screen_pairs, statistic, s_method)
resolve cache:  X path → build_cache(compute_Rxx=True, n_jobs=n_jobs) | cache path → check Rxx (compute locally if None)
reject duplicate non-synthetic feature names; validate Rxx shape/finiteness/symmetry/unit diagonal when provided
options = statistic_options or {}
resolve optional feature_groups against cache.valid_cols
y_arr = to_numpy(y, dtype=np.float32).ravel(); check y length vs cache.n_rows_original
zy = weighted_rank_gauss_1d(y_arr[cache.row_idx], cache.sample_weight)
build active feature mask from positive-weight variance of cache.Z; inactive features get W=0
model = fit_gaussian_knockoffs(active Rxx as float64, s_method, min_eig)
for each draw (seeded):  Z̃ → W → optional group W → τ → S
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
- I7: Exact ties never favor the original side by augmented column order.
- I8: Zero positive-weight-variance features are excluded from the active
  knockoff model and cannot be selected.
- I9: `sample_gaussian_knockoffs` uses a noise factor satisfying
  `L_N @ L_N.T = 2D - D Sigma_g^{-1} D`.
- I10: `y` slicing is positional after conversion to a NumPy array.

Edge cases:

| Case | Behavior |
|------|----------|
| No `t` satisfies the bound | `threshold = inf`, empty selection, valid result object; verbose message explains that emptiness is informative |
| `p == 1` | Works (`λ_min = 1`, `s ≈ 1` because of the numerical slack, knockoff is near-pure noise) |
| Constant / all-NaN columns | Already removed by `build_cache` (`valid_cols`); indices reported in original-X namespace as elsewhere |
| Zero positive-weight-variance feature | Locally inactive for `select_fdr`; reported with `W=0`, never selected |
| `cache.Rxx is None` | Computed locally, caller's cache not mutated |
| Provided `cache.Rxx` is malformed | Reject non-square, wrong shape, non-finite, asymmetric beyond tolerance, or non-unit diagonal; staleness relative to `cache.Z` is documented as a caller responsibility |
| `sample_weight` degenerate | Rejected by `ensure_weights` (existing paths) |
| Binary `y` | Supported (rank-Gaussian gives a two-point `zy`; existing tests cover this transform) |
| Numeric multiclass labels | Treated as continuous numeric `y`; integer labels with 3-20 unique values emit a one-time warning suggesting one-vs-rest categorical handling |
| `k`-style arguments | None exist — `q` replaces `k`; docs draw the contrast explicitly |
| Duplicate feature names in `X` or `cache.feature_names` | Reject with `ValueError` for non-synthetic names (consistent with the recent duplicate-label hardening in evaluate/auto-k) |
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
| `lcd` | Two LassoCV fits on `n × 2m` | the slowest statistic; documented separately from the default |
| Memory | `G` is `(2m)²` float64 → m = 2000 ⇒ 128 MB; `Z̃` is n×p float32 (same as `Z`) | `screen_pairs` default 2000 exists precisely to bound `G` |

Float discipline: `Z`/`Z̃` float32; all p×p algebra float64 (matches
`weighted_correlation_matrix` conventions).

Benchmark harness `benchmarks/bench_knockoffs.py`: grid over
`p ∈ {500, 2000}`, `n = 50_000`, statistics `{relevance}` plus any enabled
tie-safe expensive statistics (`cefsplus` only after the greedy wrapper passes
the tie tests), `n_draws ∈ {1, 11}`; report fit/sample/stat/threshold timings,
in the style of `bench_mrmr.py`. Soft target: single-draw default statistic at
`n=50k, p=2000` completes in under ~60 s on a laptop; record separate timings
for any enabled greedy statistic.

---

## 9. Testing plan

### 9.1 `tests/test_knockoff_sampler.py`

1. **Moment check**: `n=20_000, p=10`, random well-conditioned samples drawn
   from the same `Sigma_g` used by the model; empirical covariance of
   `[Z, Z̃]` matches analytic `G` entrywise within 0.05.
2. **Equi s**: hand-checked λ_min on a 3×3 matrix → exact expected `s`.
3. **Shrinkage trigger**: p > n cache → `gamma > 0`, `UserWarning` raised,
   `λ_min(Σ_g) ≈ min_eig`. (The corresponding `fdr_control` metadata assertion
   lives in the filter tests, 9.2.11 — the sampler layer has no metadata.)
4. **Cholesky fallback**: near-singular `N` (s at its cap) still returns a
   valid `L_N` (no crash, `N ≈ L_N L_Nᵀ`).
5. **Noise factor orientation**: `noise_chol @ noise_chol.T` equals
   `2D - D Σ_g^{-1} D`; regression test catches accidental use of scipy's
   default upper Cholesky factor with `E @ L_N.T`.
6. **Model stores `Sigma_g`**: analytic `G` construction uses
   `GaussianKnockoffModel.Sigma_g`, not the unshrunk input `Sigma`.
7. **Determinism**: same seed → identical `Z̃`; spawned seeds differ across draws.
8. **Validation**: bad `min_eig` (0, 1, bool, inf), unknown `s_method` raise.

### 9.2 `tests/test_knockoff_filter.py`

1. **Threshold arithmetic**: hand-crafted `W` vectors pin exact `τ` for
   `offset ∈ {0, 1}`, including the no-valid-`t` → `inf` case and the
   all-zeros case.
2. **Antisymmetry (the load-bearing test)**: for every enabled registry statistic,
   on a fixed small problem, swap one pair's columns and assert `W_j` flips
   sign exactly and all other entries are unchanged. Run for a null `j` and a
   signal `j`.
3. **Exact-tie antisymmetry**: construct an all-zero `zy` / all-zero `r_aug`
   case and assert every enabled statistic returns all-zero `W`; construct a
   pair tied at the greedy decision boundary and assert no original-side
   first-occurrence bias leaks into `W`.
4. **`lcd` symmetrization**: when `lcd` is enabled, assert swap antisymmetry
   with `np.allclose(atol=1e-10)` and verify the two-ordering formula is used.
5. **Pair-coupled screening**: construct `r`/`rt` where naive top-`|r|`
   screening would split a pair; assert both members retained/dropped together.
6. **Entry-order scores**: never-entered → 0; first-entered gets max; `W`
   signs match construction.
7. **`corr_prune` absence**: near-duplicate original/knockoff pairs still
   both appear in the augmented path (no silent pruning).
8. **Weighted variance mask**: construct a feature varying only on zero-weight
   rows and assert it is inactive, gets `W=0`, and cannot affect `Rxx`/`G`.
9. **Pandas target indexing**: `y` as a `pd.Series` with shuffled/non-default
   integer index slices positionally via `to_numpy`; no label-alignment
   `KeyError` or target scrambling.
10. **Validation**: `q ∉ (0,1)`, bool `q`, `offset=2`, `n_draws=0`,
   `eta=0`/`eta>1`, unknown `statistic`, `X` and `cache` both/neither,
   `sample_weight` with `cache`, `y` length mismatch, duplicate column names
   in `X` or `cache.feature_names`, malformed `cache.Rxx` — all raise
   `ValueError` with clear messages.
11. **Result object**: metadata keys present, including
   `fdr_control="approximate_plugin"` / `validity_model` asserted on a
   `gamma > 0` cache; `get_feature_ranking()` columns
   and ordering, including `relevance` and `selector` schema compatibility;
   `selection_frequency` None vs populated by `n_draws`; `selected_indices`
   map through `valid_cols` correctly when constant columns were dropped.
12. **Weights**: weighted vs unweighted runs differ on a weight-sensitive
   construction; weight scaling by a constant is a no-op (existing
   normalization guarantees).
13. **Empty result semantics**: pure-noise `y` at tight `q` → empty selection,
   `threshold=inf`, no exception.
14. **`KnockoffSelector`**: fit/transform round-trip, `get_support`,
    sklearn `clone` compatibility, categorical DataFrame preprocessing parity
    with other selector wrappers, `result_` stored and cleared on failed refit.

### 9.3 `tests/test_knockoff_fdr_control.py` (simulation; keep < ~60 s total)

Fixed-seed synthetic Gaussian designs, `n=800, p=40`, 8 true features with
strong coefficients, AR(1) correlation ρ=0.5. These are calibration/smoke
checks for the plug-in implementation, not proof of exact FDR:

1. **Default-statistic calibration**: 30 seeded replications at `q=0.2`,
   `statistic="relevance"`: mean FDP ≤ 0.30 (generous margin over q to kill
   flake).
2. **Power**: mean power ≥ 0.6 on the same runs.
3. **Global null**: y independent of X; assert ≤ 40% of 30 seeds select
   anything at `q=0.2`, and exact all-zero relevance target selects nothing.
4. **Derandomization**: `n_draws=11` selection is a subset-ish, more stable
   set — assert Jaccard similarity between two derandomized runs (different
   `random_state`) ≥ that of two single-draw runs, on a fixed design.
5. Repeat (1) with `statistic="lcd"` or a greedy statistic only after that
   statistic is enabled by exact-tie antisymmetry tests.

### 9.4 Existing-suite guarantees

No changes to existing behavior; full suite stays green
(`python -m pytest -ra`, currently 274 passed / 10 skipped locally, 336 in
the darko env). `test_docs_smoke.py` forces the DOCS.MD export block update.

---

## 10. Documentation

- **DOCS.MD**: add exports to the "Top-level exports" block (test-enforced);
  full API section for `select_fdr` / `KnockoffSelector` /
  `sample_knockoffs` / `knockoff_threshold`; a row in the selector support
  matrix (Gaussian-path only; weights: approximate/importance-weighted;
  target: numeric continuous or binary; categorical multiclass requires
  one-vs-rest; `k`: n/a — controlled by `q`).
- **README.md**: one row in Main Components ("q-calibrated knockoff selection —
  `select_fdr`, `KnockoffSelector`") and a 5-line quickstart snippet.
- **docs/user-guide.md**: new section covering — what the guarantee means and
  what it is conditional on (valid Gaussian copula, exact Model-X vs plug-in
  second-order approximation); choosing `q`; `n_draws`/`eta` guidance; why an
  empty result is an answer; `fdr_control`, `validity_model`, `weighted_model`,
  `s_mean`/`gamma` diagnostics; the correlated-features power/validity caveat
  (recommend pre-pruning near-duplicates via `corr_prune`-style dedup before
  building the cache, or using `feature_groups` for known feature families);
  contrast with `k="auto"`
  ("auto-k asks how many features help prediction; knockoffs ask how many you
  can trust — use both").
- **docs/architecture.md**: one paragraph on the two new modules and the
  copula-cache dependency.
- **TODO.MD**: new tracked item with phase checklist.

---

## 11. Phasing

**Phase 1 (safe core in this spec)**: equi/MVR/ME + shrinkage with explicit
plug-in validity metadata, `relevance` default statistic, tie-safe `cefsplus`,
feature-group thresholding, knockoff/knockoff+ thresholds, derandomization,
result object, weighted-variance masking, wrapper class, tests, benchmarks,
docs. `lcd` and the remaining greedy statistics stay reserved registry keys
until they pass exact-tie antisymmetry tests.

Suggested implementation order (each step lands with its tests):
1. `estimators/knockoffs.py` + sampler tests (9.1).
2. `knockoff_threshold` + arithmetic tests.
3. Safe `relevance` statistic + antisymmetry/screening/zero-target tests.
4. `select_fdr` orchestration + validation/result/weighted-variance tests.
5. `KnockoffSelector`, exports, DOCS.MD/README/user-guide, benchmarks.
6. Enable additional statistics only after their knockoff-specific tie handling
   passes 9.2 exact-tie tests.
7. Simulation calibration tests (9.3), tuning default `screen_pairs`/`path_depth`
   if expensive statistics are enabled.

**Phase 2 (out of scope, interfaces reserved)**:
- Raw-space knockoffs via per-column inverse empirical CDF, enabling
  tree-model statistics (`statistic="catboost"`, Boruta-style importance W).
- Auto-k cross-check: surface "auto-k chose k=40 but only 12 features survive
  q=0.2" in `FeaturePathEvaluationResult` diagnostics.

---

## 12. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Power collapse and model drift when features are highly collinear (`λ_min ≈ 0` ⇒ shrinkage / tiny `s`, knockoffs ≈ copies, all `W ≈ 0`) | Shrinkage floor + `UserWarning`; `fdr_control="approximate_plugin"`, `s_mean`/`gamma`/`lambda_min` surfaced in metadata; MVR/ME `s_method` options; optional `feature_groups` thresholding for feature families; user-guide guidance to dedup near-duplicates first. Do not claim this is "never an FDR violation"; shrinkage changes the validity model. |
| Statistic silently breaks exchangeability (the classic knockoff implementation bug) | The swap-antisymmetry contract is tested programmatically per enabled registry entry (9.2.2); exact ties and zero-target cases are mandatory; D5/D6 remove screening/pruning mechanisms likely to break it. |
| Weighted caches can contain zero positive-weight-variance columns | Local weighted-variance mask before fitting knockoffs; inactive columns get `W=0`; tests cover zero-weight-support features. |
| Second-order/copula approximation invalid for exotic joint dependence or stale caller-provided `Rxx` | Documented honestly everywhere the guarantee is mentioned; validate `Rxx` shape/finiteness/symmetry/unit diagonal, but stale cache consistency remains caller responsibility. |
| Runtime on very wide caches (`O(n·p²)` per draw) | Row-blocked GEMMs, one-time model fit reused across draws, `subsample` guidance, benchmark harness to keep numbers honest. |
| Simulation tests flake in CI | Fixed seeds, generous margins (FDP ≤ 0.30 vs q=0.20), small designs, bounded runtime; treat simulations as calibration smoke tests, not proof of exact control. |

---

## 13. Acceptance criteria

1. New test files pass; full suite green in both the base and darko
   environments; no new warnings under `pytest -ra`.
2. Antisymmetry test passes for **every enabled** registered statistic,
   including exact-tie and zero-target cases (`lcd` uses the documented
   allclose tolerance).
3. `fdr_control`, `validity_model`, `weighted_model`, `gamma`, and
   zero-weight-variance diagnostics are present and documented.
4. Sampler tests verify the noise-factor orientation
   (`noise_chol @ noise_chol.T = 2D − D Σ_g⁻¹ D`); filter tests verify
   positional pandas `y` slicing and statistic-context scatter behavior.
5. Simulation calibration: empirical mean FDP within margin at `q=0.2`, power
   ≥ 0.6 on the reference design for enabled statistics; global-null selection
   rate within bound.
6. `test_docs_smoke.py` passes with the new exports (DOCS.MD updated).
7. Benchmark: single-draw default statistic at `n=50k, p=2000` under the soft
   target (~60 s laptop); any enabled expensive statistics have separate
   numbers recorded in `benchmarks/README.md`.
8. No new runtime dependencies; `pip install -e .` unchanged.
9. Deterministic: fixed `random_state` reproduces selections byte-for-byte.
