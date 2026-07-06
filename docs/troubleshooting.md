# Troubleshooting

Common errors, what they mean, and how to fix them. For full API behavior see
the [API manual](../DOCS.MD); for picking a selector see the
[user guide](user-guide.md).

## Installation

### `ModuleNotFoundError: No module named 'category_encoders'`

Raised by `encode_categoricals` when `cat_encoding` is one of `"loo"`,
`"target"`, or `"james_stein"`. SIFT keeps `category_encoders` optional.

```bash
python -m pip install -e ".[categorical]"
```

Alternatively, set `cat_encoding="loo_logit"` (binary targets, no extra
dependency) or `cat_encoding="none"` after pre-encoding categoricals upstream.

### `catboost` import errors

CatBoost is loaded lazily; `import sift` does not require it. The error appears
only when you call `sift.catboost_select`, `sift.catboost_regression`,
`sift.catboost_classif`, or set `importance="shap"` on `BorutaSelector`.

```bash
python -m pip install -e ".[catboost]"
```

## Auto-k

### `ValueError: k='auto' requires time, groups, or auto_k_config with k_method='elbow' or 'penalized_objective'`

`AutoKConfig` defaults to `k_method="evaluate"` with `strategy="time_holdout"`.
Evaluate-mode auto-k always needs a held-out split, so pass either `time=...`,
`groups=...`, or build an `AutoKConfig` whose `k_method` does not require a
held-out split:

```python
from sift import AutoKConfig, select_cefsplus

config = AutoKConfig(k_method="elbow", min_k=5, max_k=80)
select_cefsplus(X, y, k="auto", auto_k_config=config)
```

### `ValueError: auto-k evaluate with strategy='time_holdout' requires time parameter`

You set `AutoKConfig(strategy="time_holdout")` but did not pass `time=...`. Same
for `strategy="group_cv"` requiring `groups=...`. Either pass the split context
or switch to an objective-only `k_method` (`elbow` or `penalized_objective`).

### `<selector> does not support k_method=<value>`

Auto-k support depends on the selector route:

| Route | Supported `k_method` |
| --- | --- |
| Classic mRMR/JMI/JMIM | `evaluate` |
| Gaussian mRMR/JMI/JMIM | `evaluate`, `elbow` |
| CEFS+ | `evaluate`, `elbow`, `penalized_objective` |
| Binary CEFS+ | `evaluate`, `elbow`, `penalized_objective` |

Pick a supported mode or switch selectors.

### `NotImplementedError: AutoKConfig(auto_k_mode='nested') is not implemented yet`

Function-style selectors only support `auto_k_mode="prefix_only"`. Drop the
`auto_k_mode="nested"` override, or use a sklearn-style selector class
(`MRMRSelector`, `CEFSPlusSelector`, etc.) where nested mode is wired through.

## Categorical Features

### `ValueError: cat_encoding='loo' fits a supervised categorical encoder on the full dataset…`

Function-style selectors block full-data target encoding by default to avoid
leakage. Two safe options:

- **Opt in explicitly** (only when leakage is handled externally):
  ```python
  select_mrmr(X, y, k=20, task="regression",
              cat_encoding="loo", allow_full_data_target_encoding=True)
  ```
- **Pre-encode in a leakage-safe pipeline**, then pass `cat_encoding="none"`.

This applies to `"target"`, `"loo"`, `"james_stein"`, and `"loo_logit"`.

### `TypeError: cat_features/cat_encoding require X to be a pandas DataFrame`

Categorical encoding needs named columns. Convert `X` to a `pandas.DataFrame`
with column names before passing `cat_features` / `cat_encoding`.

### `ValueError: Non-numeric columns found: ['cat1', 'cat2']…`

Boruta and the Gaussian cache cannot consume object/string/category columns
directly. Either:

- Pass `cat_encoding="loo"` (or another supported encoder) to BorutaSelector,
  or
- Encode categoricals upstream and pass numeric data.

For `BorutaSelector(importance_data="test")`, supervised categorical encodings
are rejected to avoid held-out target leakage. In that mode,
`cat_encoding="none"` means the categorical columns must already be numeric or
pre-encoded; raw object/category columns still raise this non-numeric-column
error.

When `BorutaSelector` is fitted with `cat_encoding != "none"`, `transform()`
re-applies the fitted categorical encoder before selecting columns. Transforming
new data therefore requires a DataFrame with the same categorical columns.

## Estimators

### `ValueError: estimator='ksg' does not support sample_weight`

The KSG mutual-information estimator is unweighted. Either drop
`sample_weight=...`, or switch to `estimator="binned"` (classification) or
`estimator="r2"`/`estimator="gaussian"` (regression) which honor weights.

### `ValueError: estimator='gaussian' is regression-only.`

Same for `"r2"` and `"ksg"`. Use `estimator="binned"` for classification, or
`estimator="auto"` to let SIFT pick based on the task.

### `ValueError: binary CEFS+ requires exactly two target classes`

`select_cefsplus_binary` validates that `y` has exactly two unique non-null
values. Drop missing rows or filter classes before calling.

## Stability Selection

### Stability returned fewer features than `max_features`

`StabilitySelector(max_features=10)` treats `max_features` as a **cap**, not a
target count. The actual selection is the set of features whose selection
frequency meets `threshold`, capped at `max_features`. If too few features clear
the threshold you may get fewer than `max_features`, including zero.

The convenience wrappers `stability_regression(..., k=10)` and
`stability_classif(..., k=10)` fill short selections from the top selection
frequencies so they return `k` features when at least `k` features exist. To do
the same manually with the estimator class:

```python
selector = StabilitySelector(task="regression", threshold=0.6, max_features=None)
selector.fit(X, y)
order = np.argsort(-selector.selection_frequencies_)
top_k = [selector.feature_names_in_[i] for i in order[:k]]
```

## Boruta

### `ValueError: BorutaSelector(importance_data='test') is not supported with importance='native'`

Native importances are read from the fitted model and do not evaluate held-out
rows; combining them with `importance_data="test"` would be misleading. Either
switch to `importance_data="train"` or pick a held-out-compatible backend
(`importance="shap"`).

## Cache

### `ValueError: y has N rows but cache was built from M rows`

A `FeatureCache` is tied to a specific `X`. Build a new cache for each X with a
different row count, or align your X to match `cache.n_rows_original`.

If you persist `FeatureCache` objects, rebuild caches created before SIFT stored
`feature_names_are_synthetic`. Those older caches cannot prove whether `x0`,
`x1`, ... were real DataFrame names or generated names for unnamed positional
input, so rebuilding avoids silent feature-name ambiguity.

### `ValueError: All features were filtered out (constant or invalid). Cannot build cache.`

Every column in `X` has near-zero standard deviation in the subsample. Pass
informative features, increase `subsample`, or check upstream preprocessing.

## Bug Reports

If you hit something not covered here, please open an issue with a minimal
reproducer. See [docs/development.md](development.md) for the test slices that
are most likely to surface the bug.
