# SIFT

SIFT is a Python feature-selection toolbox for fast filter selectors, automatic
feature-count selection, q-calibrated Gaussian-copula knockoffs, stability
selection, smart sampling, Boruta-style selection, grouped or time-aware
permutation importance, and optional CatBoost selection.

The package is a single Python library. Public entry points are exported from
`sift`, while advanced building blocks live under `sift.selection`,
`sift.estimators`, and `sift.sampling`.

## Quickstart

SIFT is not published to PyPI. Install it from a repository checkout:

```bash
python -m pip install .
```

The built distribution name is `sift-feature-selection`, while the import stays
`sift`. That import collides with Sift Science's SDK, so the two projects must
not be installed in the same Python environment.

For editable local development, install from the repository root:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[categorical]"
python -m pip install -e ".[catboost]"
python -m pip install -e ".[test]"
python -m pip install -e ".[all]"
```

Run a fixed-k selector:

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import select_mrmr, select_cefsplus

X_arr, y = make_regression(
    n_samples=500,
    n_features=30,
    n_informative=8,
    noise=0.2,
    random_state=0,
)
X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])

mrmr_features = select_mrmr(X, y, k=10, task="regression", verbose=False)
cefs_features = select_cefsplus(X, y, k=10, verbose=False)
```

Run a q-calibrated knockoff selector:

```python
import numpy as np
import pandas as pd
from sift import select_fdr

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(800, 30)), columns=[f"f{i}" for i in range(30)])
y = 3.0 * X.iloc[:, :12].sum(axis=1) + rng.normal(size=800)

result = select_fdr(X, y, q=0.2, random_state=0, verbose=False)
trusted_features = result.selected_features
```

`select_fdr` reports approximate plug-in Gaussian-copula validity metadata; see
the tutorial for the exact Model-X assumptions behind the q-calibrated result.
The default knockoff+ threshold either returns nothing or at least `1/q`
features, so a small `q` on a narrow design legitimately selects none.

For the full public API, examples, selector support matrix, and option details,
start with the [full API manual](https://github.com/kmedved/sift/blob/main/DOCS.MD).

## What's New in 0.9

0.9 is a product layer that sits beside the existing API: the surfaces below are
additive, and every legacy result type and default they touch is unchanged. A
short list of deliberate behavior changes does exist — the centered `target_cv`
encoding, `stability_regression`/`stability_classif` no longer padding a short
selection up to `k`, and `feature_names_in_` becoming sklearn's ndarray. Those,
the migration notes, and the deprecation ledger are in the
[release notes](https://github.com/kmedved/sift/blob/main/docs/release-notes.md).

- `sift.as_result(...)` and `SelectionView` give every selector family one
  normalized result view.
- `cat_encoding="target_cv"` adds leakage-safe, fold-centered target encoding
  for DataFrames with string columns, with no optional dependency.
- Selector classes gain `output_order`, `inverse_transform`, sklearn's
  `feature_names_in_` contract, `set_output(transform="pandas")`, and explicit
  sklearn 1.4+ metadata routing.
- `AutoKConfig` gains named presets and `from_groups(...)` option groups.
- `sift.experimental` stages 16 research-oriented auto-k helpers.
- `set_verbosity("info")` routes progress through the `sift` logger,
  `select_cached(..., return_result=True)` returns a full view, DataFrame
  callers may pass `groups="column"`/`time="column"`, and `k="auto"` warns when
  it selects nothing.

### Normalized result views

Ask a selector for its rich result, then adapt it. The same five accessors work
for filters, knockoffs, Boruta, feature-path evaluation, CatBoost, permutation
importance, and a fitted `StabilitySelector`.

```python
import numpy as np
import pandas as pd
import sift

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
y = 2.0 * X["f0"] + X["f3"] - 0.5 * X["f5"] + rng.normal(scale=0.1, size=200)

result = sift.select_mrmr(
    X, y, k=3, task="regression", return_result=True, verbose=False
)
view = sift.as_result(result, input_features=X.columns)

print(view.features)               # ['f0', 'f3', 'f5']
print(view.indices)                # positions in the raw input
print(view.k)                      # 3
print(view.table.head())           # one row per raw column
print(view.metadata["selector"])   # 'mrmr'
payload = view.to_dict()           # JSON-safe, schema_version "1"
```

See [Reading results](https://github.com/kmedved/sift/blob/main/docs/results.md)
for per-family table columns, curves, and partial-identity rules.

### Leakage-safe categorical encoding

`cat_encoding="target_cv"` is built into SIFT and needs no `category_encoders`
extra. It emits centered category effects: out-of-fold training rows get
`fold_encoding - fold_training_prior`, so an unseen category maps to zero rather
than to a prior that identifies its own fold.

```python
import numpy as np
import pandas as pd
from sift import select_mrmr

rng = np.random.default_rng(0)
city = rng.choice(["oslo", "lima", "cairo", "perth"], size=300)
lift = pd.Series(city).map({"oslo": 1.5, "lima": -1.0, "cairo": 0.2, "perth": -0.7})

X = pd.DataFrame(
    {"city": city, "noise_a": rng.normal(size=300), "noise_b": rng.normal(size=300)}
)
y = lift.to_numpy() + rng.normal(scale=0.3, size=300)

selected = select_mrmr(
    X,
    y,
    k=2,
    task="regression",
    cat_features=["city"],
    cat_encoding="target_cv",
    verbose=False,
)
print(selected)  # ['city', ...]
```

Centering removes the fold marker; it is not a defence against high cardinality.
A level seen two or more times inside a fold still carries its sibling rows'
targets, so a near-unique identifier remains selectable by design. Drop ID-like
columns, or pass `groups=` so an identifier's rows land in one fold.

### Sklearn-native selector classes

All eight selector classes are `SelectorMixin` transformers. `output_order`
chooses between historical selection order and ascending input order, and
`set_output(transform="pandas")` keeps a DataFrame on the way out.

```python
import numpy as np
import pandas as pd
from sift import MRMRSelector

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
y = 2.0 * X["f4"] + X["f1"] + rng.normal(scale=0.1, size=200)

legacy = MRMRSelector(k=2, task="regression", verbose=False).fit(X, y)
print(list(legacy.get_feature_names_out()))  # ['f4', 'f1'] - selection order

original = MRMRSelector(k=2, task="regression", output_order="original", verbose=False)
original.set_output(transform="pandas")
X_selected = original.fit_transform(X, y)

print(list(X_selected.columns))               # ['f1', 'f4'] - input order
print(list(original.feature_names_in_))       # all six fitted names
print(np.shape(original.inverse_transform(X_selected)))  # (200, 6), zero-filled
```

Metadata routing is opt-in and explicit. With sklearn 1.4+ routing enabled, call
`selector.set_fit_request(groups=True)` before `cross_validate(..., params=...)`;
row context requires a group-aware `k="auto"` configuration, because fixed-k
filters reject `groups`/`time`. On sklearn 1.3, pass accepted metadata straight
to `fit`.

### Auto-K presets and the experimental namespace

```python
from sift import AutoKConfig

router = AutoKConfig.default()                                    # measured router
predictive = AutoKConfig.predictive(strategy="kfold", rule="one_se", n_folds=5)
discovery = AutoKConfig.discovery(alpha=0.05)                     # calibrated stop
downstream = AutoKConfig.downstream(strategy="group_cv", metric="r2", rule="best")
```

`AutoKConfig.from_groups(...)` flattens seven immutable option groups into the
same flat config. `sift.experimental` stages 16 research-oriented auto-k helpers
whose access emits a `FutureWarning`; all 58 names in `sift.__all__` stay
importable from `sift` itself, warning-free, throughout 0.9.

## Documentation

- [Full API manual](https://github.com/kmedved/sift/blob/main/DOCS.MD)
- [Generated API reference source](https://github.com/kmedved/sift/blob/main/docs/reference/index.md)
  (render locally with `mkdocs serve`)
- [Selector decision tree](https://github.com/kmedved/sift/blob/main/docs/choosing-a-selector.md)
- [Runtime and scaling](https://github.com/kmedved/sift/blob/main/docs/runtime-scaling.md)
- [Data-type support matrix](https://github.com/kmedved/sift/blob/main/docs/data-type-support.md)
- [Glossary](https://github.com/kmedved/sift/blob/main/docs/glossary.md)
- [Algorithm guide](https://github.com/kmedved/sift/blob/main/docs/ALGORITHMS.md)
- [Advanced workflow guide](https://github.com/kmedved/sift/blob/main/docs/ADVANCED.md)
- [Tutorial](https://github.com/kmedved/sift/blob/main/docs/user-guide.md)
- [Reading results](https://github.com/kmedved/sift/blob/main/docs/results.md)
- [Troubleshooting](https://github.com/kmedved/sift/blob/main/docs/troubleshooting.md)
- [Architecture and module boundaries](https://github.com/kmedved/sift/blob/main/docs/architecture.md)
- [Development guide](https://github.com/kmedved/sift/blob/main/docs/development.md)
- [Benchmarks](https://github.com/kmedved/sift/blob/main/benchmarks/README.md)
- [Release notes](https://github.com/kmedved/sift/blob/main/docs/release-notes.md)
- [Contributing guide](https://github.com/kmedved/sift/blob/main/CONTRIBUTING.md)

## Main Components

| Area | Entry points |
| --- | --- |
| Core filters | `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, `select_cefsplus_binary` |
| q-calibrated knockoffs | `select_fdr`, `KnockoffSelector`, `sample_knockoffs` |
| Automatic `k` | `k="auto"` for measured CEFS+ auto-routing, `AutoKConfig` and its presets, `select_k_auto`, `select_k_elbow`, `select_k_penalized_objective`, `select_k_chi2_stop`, `select_k_perm_gap`, `select_k_gaussian_cv` |
| Result objects and views | `as_result`, `SelectionView`, `FilterSelectionResult`, `KnockoffSelectionResult`, `BorutaResult`, `FeaturePathEvaluationResult` |
| Selector classes | `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector`, `KnockoffSelector`, `BorutaSelector`, `StabilitySelector`, `Stabilized` |
| Cache-backed Gaussian paths | `build_cache`, `select_cached`, `FeatureCache` |
| Sampling and stability | `smart_sample`, `SmartSamplerConfig`, `StabilitySelector`, `stability_regression`, `stability_classif`, `Stabilized` |
| Model-based importance | `permutation_importance`, `BorutaSelector`, `select_boruta`, `select_boruta_shap`, CatBoost helpers |
| Diagnostics | `set_verbosity`, `evaluate_feature_path`, `sift.experimental` |

## Choosing a Selector

Start with the canonical
[selector decision tree](https://github.com/kmedved/sift/blob/main/docs/choosing-a-selector.md).
It separates fixed-size rankings, all-relevant searches, resampling diagnostics,
model-specific importance, and q-calibrated knockoff sets before recommending
an entry point.

## Development

Install test dependencies and run the suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See the [development guide](https://github.com/kmedved/sift/blob/main/docs/development.md) for focused test slices,
benchmarks, documentation checks, and release notes.

## License

SIFT is released under the [MIT License](https://github.com/kmedved/sift/blob/main/LICENSE).
