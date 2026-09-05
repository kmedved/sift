# Feature Selection Algorithms

This guide explains the algorithms exposed by SIFT, what they optimize, and
when each one is a good fit. For exact parameters, see the
[generated API reference](reference/index.md) and the
[full manual](https://github.com/kmedved/sift/blob/main/DOCS.MD).

## Algorithm Map

The canonical [selector decision tree](choosing-a-selector.md) maps output
contracts to public entry points. This page supplies the mathematical and
algorithmic detail behind each leaf rather than maintaining another choice map.
Shared vocabulary — relevance, redundancy, knockoffs, stability frequency —
is in the [glossary](glossary.md).

## Filter Methods

Filter methods rank or greedily select features using statistics computed from
the data, without repeatedly fitting the final downstream model. They are often
the fastest way to build a credible feature panel.

### mRMR

mRMR stands for minimum Redundancy Maximum Relevance. At each step it chooses a
candidate with high target relevance and low redundancy with already selected
features.

For a candidate `f` and selected set `S`, SIFT supports:

```text
quotient:   score(f) = relevance(f, y) / mean_redundancy(f, S)
difference: score(f) = relevance(f, y) - mean_redundancy(f, S)
```

The classic estimator uses relevance scores such as F-statistics, KS statistics,
or random-forest relevance, with Pearson-style redundancy. The Gaussian
estimator uses rank-Gaussian correlation and the Gaussian mutual-information
proxy, which is fast for regression.

Use mRMR when:

- you need a quick baseline;
- feature count is large;
- pairwise redundancy is the main concern;
- you want a stable ordered path for later evaluation.

Limitations:

- It is greedy, not globally optimal.
- Pairwise redundancy can miss higher-order structure.
- Strongly interchangeable features can enter in arbitrary order.

### JMI

Joint Mutual Information scores candidates by the information they provide with
already selected features:

```text
score(f) = sum over s in S of I(f, s; y)
```

This favors features that add complementary signal rather than just marginal
signal. SIFT exposes multiple estimators, including Gaussian, binned, KSG, R2,
and `auto`.

Use JMI when interactions or complementary views matter and you can afford more
work than mRMR.

### JMIM

JMIM is a conservative variant of JMI:

```text
score(f) = min over s in S of I(f, s; y)
```

The minimum makes a candidate prove value relative to every selected feature.
This usually produces a more cautious and diverse path.

Use JMIM when you prefer false negatives over redundant additions, or when the
feature set needs to remain compact and varied.

### CEFS+

CEFS+ is SIFT's Gaussian-copula conditional information path. It transforms
features to rank-Gaussian space, uses correlations to build a conditional
information objective, and greedily adds features by incremental log-det gain.

At a high level, it asks: "How much extra information about `y` remains in this
candidate after conditioning on the selected set?"

`include=` is a true conditioning set: CEFS+ initializes the partial-Cholesky
residual state from those features before the first greedy step. `exclude=`
and `candidates=` only change who is eligible to be discovered. `k` is the
number of additional discoveries; included features are returned first in
caller order and are not counted as discoveries. The same contract applies
to classic mRMR/JMI/JMIM, Gaussian JMI/mRMR, binary log-loss CEFS+, and
cache-backed selection. Auto-k methods that cannot reuse that conditioned
path reject the keywords instead of post-filtering an unconditioned run.

Use CEFS+ when:

- the task is regression;
- conditional relationships matter;
- you want a minimal-optimal style subset;
- Gaussian-copula assumptions are acceptable.

`corr_prune` can remove near-duplicates from the path. This helps keep the
subset compact, but if correlated feature families are meaningful, review the
path before pruning too aggressively. With `store_proxies=True`,
`SelectionView.redundancy_report` and `proxy_clusters` expose those
near-duplicates from the stored copula block without retaining `X`.

Regression filters also accept `within="groups"` or `within="two_way"`. Those
subtract weighted entity means, or alternate entity and time demeaning for a
fixed five iterations, from `X` and `y` before ranks. Auto-k evaluate,
Gaussian CV, and xfit-objective fit those means on training rows only; an
entity unseen in training uses the training grand mean. Demeaning can leave
no within-entity variation, including when every group is a singleton; the
call then returns an empty selection or raises. `between_relevance` is an
entity-mean summary, not row-level evidence, and is not comparable in
magnitude to `within_relevance`. The selector still returns original column
identities, and sklearn `transform` still emits the selected raw columns.

### Binary CEFS+

Binary CEFS+ adapts the CEFS+ idea for Bernoulli-like targets. It uses a
conditional logistic or Brier-style proxy rather than the continuous
rank-Gaussian target path.

Use it for binary outcomes where logistic conditional signal is a better match
than treating labels as a continuous numeric target.

## Knockoff FDR

`select_fdr` is different from fixed-k filters. It returns a selected set at a
target false-discovery level `q`, not the first `k` entries in a path.

The 0.7.0 implementation:

1. Builds or reuses a rank-Gaussian `FeatureCache`.
2. Fits a second-order Gaussian-copula knockoff sampler.
3. Samples knockoff copies in cache space.
4. Computes antisymmetric feature statistics `W`.
5. Applies the knockoff+ threshold, or aggregates multiple draws by frequency
   or, if requested, by averaged knockoff e-values and e-BH.

Knockoff+ is discrete: metadata `min_feasible_q` is `1/min(m)` over completed
draws, a necessary count bound rather than a sufficient discovery condition.
`n_tested_per_draw` is the truthful post-screening count; `n_eligible` is the
pre-screen unit count and is not a completed tested draw. Early returns such
as a constant target set `tested_state="not_run"` instead of inventing
screened counts. `n_discoveries_offset_0` counts reported discovery features
at `offset=0` from the same `W`, which for grouped runs is the expanded
feature list, not the tested-group count. A warning is emitted for
`offset=1` draws with `m·q < 1`; an infeasible draw does not imply an empty
aggregate. Included conditioning features are not tested units.

The default statistic is `statistic="relevance"`, a fast marginal
Gaussian-information difference between each original feature and its knockoff.
`statistic="ridge"` is the analytic coefficient-difference statistic.
`statistic="lsm"` is the lasso signed-max path statistic.
`statistic="cefsplus"` enables a tie-safe greedy statistic that is slower but
redundancy-aware. The 0.9 default stays `relevance`. Whether 1.0 should flip
to `ridge` is an owner decision from the committed
[statistic bakeoff](knockoff-statistic-bakeoff.md), not from uncommitted
scratch timings. That study reports realized FDP and power; it does not
upgrade `approximate_plugin`, and it does not treat LSM or CEFS+ rows as a
sign-flip proof.

`s_method` controls the diagonal knockoff construction:

- `"equi"` is the fast equicorrelated construction.
- `"mvr"` uses a minimum-variance reconstructability objective.
- `"me"` uses a maximum-entropy objective.

The guarantee language matters. SIFT reports `fdr_control="approximate_plugin"`
and `validity_model="gaussian_copula_plugin"` because the feature model is
estimated from the same data and can be shrunk for numerical stability. Exact
finite-sample Model-X FDR requires the fitted feature model to be the true
feature distribution and the statistic to obey the swap-antisymmetry contract.

Opt-in `aggregation="evalues"` follows Ren and Barber (arXiv:2205.15461): each
draw contributes `e_j = m · 1{W_j ≥ T_q} / (1 + #{W ≤ −T_q})` with `m = |T|`
the size of the common tested universe, not the number of nonzero e-values.
Averaging preserves the aggregate null bound `∑_{j ∈ H0} E[e_j] ≤ m`; e-BH at
`q` then controls FDR at the same plug-in level as a single knockoff+ draw.
SIFT does not claim that each null's e-value has unit expectation. Validated
ungrouped e-value mode is only `relevance` and `ridge`. `relevance` is
`W_j = g(r_j) - g(r̃_j)` for Gaussian MI `g`, so swapping feature `j` with its
knockoff swaps the two correlations and negates `W_j` with no path or
stopping rule. `ridge` solves `β = (G + λI)^{-1}[r; r̃]` on the analytic
original/knockoff Gram; `G` is invariant under swapping pair `j`, which
permutes the right-hand side and therefore swaps `β_j` with `β_{j+m}`, so
`W_j = |β_j| - |β_{j+m}|` flips sign. `lsm` and `cefsplus` remain
exploratory: truncated LARS entry and adaptive/greedy CEFS+ paths can change
used depth or break ties in a way that is not a sign flip (including on
inputs that a swap leaves unchanged). A swap-invariant pair screen
(`max(|r|, |r̃|)`) that still varies across draws is recorded and zero-padded,
but that union was not fixed before the statistics, so the run is
exploratory. Grouped e-values are exploratory until a valid group statistic
exists. The per-draw threshold uses the same `q` as e-BH (the specification's
`T_q`); a smaller knockoff level would affect power only.

Use knockoffs when:

- you care about a trusted discovery set more than a fixed count;
- null-feature control is part of the workflow;
- an empty result should be allowed to mean "nothing survived q";
- you want diagnostics such as `W`, thresholds, `gamma`, `lambda_min`, and
  `s_mean`.

Compare with auto-k:

- Auto-k asks how many features help prediction.
- Knockoffs ask how many discoveries survive a target false-discovery level.

Both can be useful on the same project, but they answer different questions.

## Stability Selection

Stability selection repeatedly fits sparse linear models on resampled data and
keeps features selected often enough:

```text
frequency(f) = selected_bootstraps(f) / n_bootstrap
```

SIFT supports regression and classification, optional sample weights, group
bootstrap, block bootstrap for grouped time series, and smart sampling for large
datasets.

Use stability selection when:

- data are noisy;
- you want robustness across resamples;
- many features are weakly interchangeable;
- you want frequency diagnostics rather than a single greedy path.

The implementation is a practical heuristic inspired by stability-selection
literature. It is not the q-calibrated `select_fdr` API.

## Boruta

Boruta is an all-relevant wrapper method:

1. Create shadow features by permuting originals.
2. Fit a model on originals plus shadows.
3. Compare each original feature with the shadow baseline.
4. Confirm, reject, or keep features tentative.
5. Repeat until convergence or `max_iter`.

Use Boruta when finding all potentially relevant variables is more important
than finding the smallest subset. It can keep redundant features by design.

SIFT supports native tree importance, permutation importance, and SHAP-backed
variants through `BorutaSelector`, `select_boruta`, and `select_boruta_shap`.

## CatBoost-Based Selection

CatBoost selectors use a model wrapper around CatBoost feature importance and
validation performance. They are optional and require the `catboost` extra.

Supported algorithms include:

| Algorithm | Shape | Notes |
| --- | --- | --- |
| `forward` | Iterative importance | Fast production default |
| `forward_greedy` | Candidate-by-candidate search | Expensive, useful for small problems |
| `shap` | Recursive elimination by SHAP-like importance | Interpretability-oriented |
| `permutation` | Recursive elimination by validation loss change | Model-agnostic within the CatBoost path |
| `prediction` | Recursive elimination by prediction change | Faster approximation |

Use CatBoost selection when:

- nonlinear model behavior matters;
- categorical handling should stay native to CatBoost;
- the selected set should be validated by model performance;
- time-series or grouped CV splitters are part of the modeling workflow.

## Mutual-Information Estimators

SIFT uses several information proxies internally:

| Estimator | Idea | Strength | Tradeoff |
| --- | --- | --- | --- |
| Gaussian | `I ~= -0.5 log(1 - rho^2)` | Very fast | Best for monotone/Gaussian-copula signal |
| Binned | Histogram MI | Flexible | Sensitive to bins and sparse counts |
| KSG | Nearest-neighbor MI | Nonlinear continuous signal | More expensive |
| R2 proxy | Convert predictive R2 to MI-like score | Fast approximation | Linear-model bias |
| Logistic/Brier proxy | Binary conditional score | Binary targets | Not a multiclass generalization |

## Choosing a Method

Apply the [selector decision tree](choosing-a-selector.md) before using the
practical notes below. In particular, choose the result contract first; then
decide whether auto-k, a cache, or an sklearn wrapper belongs in the workflow.

## Practical Notes

- Fixed-k paths are ranking tools. Validate the chosen prefix downstream.
- `k="auto"` evaluates prefixes; it does not turn a filter into an unbiased
  nested feature-selection procedure unless you explicitly use a nested
  selector-class mode where supported.
- Sample weights change the estimand. For knockoffs, weighted runs are reported
  as importance-weighted plug-in approximations.
- Highly correlated features reduce knockoff power because valid knockoffs can
  become close copies. Deduplicate or use `feature_groups` when feature
  families are known.
- Empty selections are valid for `select_fdr`, strict Boruta, and high stability
  thresholds.

## References

- Peng, Long, and Ding (2005), "Feature selection based on mutual information:
  criteria of max-dependency, max-relevance, and min-redundancy."
- Yang and Moody (1999), "Data Visualization and Feature Selection: New
  Algorithms for Nongaussian Data."
- Bennasar, Hicks, and Setchi (2015), "Feature selection using Joint Mutual
  Information Maximisation."
- Brown, Pocock, Zhao, and Lujan (2012), "Conditional Likelihood Maximisation:
  A Unifying Framework for Information Theoretic Feature Selection."
- Candes, Fan, Janson, and Lv (2018), "Panning for gold: Model-X knockoffs for
  high-dimensional controlled variable selection."
- Barber and Candes (2015), "Controlling the false discovery rate via
  knockoffs."
- Ren, Wei, and Candes (2021), "Derandomizing knockoffs."
- Meinshausen and Buhlmann (2010), "Stability selection."
- Kursa and Rudnicki (2010), "Feature Selection with the Boruta Package."
- Kraskov, Stogbauer, and Grassberger (2004), "Estimating mutual information."
