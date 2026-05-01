# SIFT

SIFT is a Python feature-selection toolbox for fast filter selectors, automatic
feature-count selection, stability selection, smart sampling, Boruta-style
selection, grouped or time-aware permutation importance, and optional CatBoost
selection.

The package is a single Python library. Public entry points are exported from
`sift`, while advanced building blocks live under `sift.selection`,
`sift.estimators`, and `sift.sampling`.

## Quickstart

Install from the repository root:

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

For the full public API, examples, and option details, start with
[DOCS.MD](DOCS.MD).

## Documentation

- [Full API manual](DOCS.MD)
- [User guide](docs/user-guide.md)
- [Architecture](docs/architecture.md)
- [Development guide](docs/development.md)
- [Release tracker](TODO.MD)

## Main Components

| Area | Entry points |
| --- | --- |
| Core filters | `select_mrmr`, `select_jmi`, `select_jmim`, `select_cefsplus`, `select_cefsplus_binary` |
| Automatic `k` | `AutoKConfig`, `k="auto"`, `select_k_auto`, `select_k_elbow`, `select_k_penalized_objective` |
| Result objects and wrappers | `FilterSelectionResult`, `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector` |
| Cache-backed Gaussian paths | `build_cache`, `select_cached`, `FeatureCache` |
| Sampling and stability | `smart_sample`, `SmartSamplerConfig`, `StabilitySelector`, `stability_regression`, `stability_classif` |
| Model-based importance | `permutation_importance`, `BorutaSelector`, `select_boruta`, `select_boruta_shap`, CatBoost helpers |

## Development

Install test dependencies and run the suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See [docs/development.md](docs/development.md) for focused test slices,
benchmarks, documentation checks, and release notes.

## License

SIFT is released under the [MIT License](LICENSE).
