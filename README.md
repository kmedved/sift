# SIFT

SIFT is a Python feature-selection toolbox for fast filter selectors, automatic
feature-count selection, q-calibrated Gaussian-copula knockoffs, stability
selection, smart sampling, Boruta-style selection, grouped or time-aware
permutation importance, and optional CatBoost selection.

The package is a single Python library. Public entry points are exported from
`sift`, while advanced building blocks live under `sift.selection`,
`sift.estimators`, and `sift.sampling`.

## Quickstart

Install the feature-selection distribution from PyPI (the import remains
`sift`):

```bash
python -m pip install sift-feature-selection
```

The distribution name is distinct from Sift Science's occupied `Sift` project.
For source compatibility this project still imports as `sift`, so the two
projects must not be installed in the same Python environment.

For local development, install from the repository root:

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
from sift import select_fdr

result = select_fdr(X, y, q=0.1, verbose=False)
trusted_features = result.selected_features
```

`select_fdr` reports approximate plug-in Gaussian-copula validity metadata; see
the user guide for the exact Model-X assumptions behind the q-calibrated result.

For the full public API, examples, selector support matrix, and option details,
start with the [full API manual](https://github.com/kmedved/sift/blob/main/DOCS.MD).

## Documentation

- [Full API manual](https://github.com/kmedved/sift/blob/main/DOCS.MD)
- [Standalone API reference](https://github.com/kmedved/sift/blob/main/docs/API.md)
- [Algorithm guide](https://github.com/kmedved/sift/blob/main/docs/ALGORITHMS.md)
- [Advanced workflow guide](https://github.com/kmedved/sift/blob/main/docs/ADVANCED.md)
- [User guide](https://github.com/kmedved/sift/blob/main/docs/user-guide.md)
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
| Automatic `k` | `k="auto"` for measured CEFS+ auto-routing, `AutoKConfig`, `select_k_auto`, `select_k_elbow`, `select_k_penalized_objective`, `select_k_chi2_stop`, `select_k_perm_gap`, `select_k_gaussian_cv` |
| Result objects and wrappers | `FilterSelectionResult`, `KnockoffSelectionResult`, `MRMRSelector`, `JMISelector`, `JMIMSelector`, `CEFSPlusSelector`, `CEFSPlusBinarySelector`, `KnockoffSelector` |
| Cache-backed Gaussian paths | `build_cache`, `select_cached`, `FeatureCache` |
| Sampling and stability | `smart_sample`, `SmartSamplerConfig`, `StabilitySelector`, `stability_regression`, `stability_classif` |
| Model-based importance | `permutation_importance`, `BorutaSelector`, `select_boruta`, `select_boruta_shap`, CatBoost helpers |

## Choosing a Selector

| Goal | Start with |
| --- | --- |
| Fast relevance/redundancy baseline | `select_mrmr` |
| Complementary information path | `select_jmi` or `select_jmim` |
| Compact regression subset | `select_cefsplus` |
| Binary-target conditional path | `select_cefsplus_binary` |
| q-calibrated trusted discoveries | `select_fdr` or `KnockoffSelector` |
| Robustness across resamples | `StabilitySelector` |
| All-relevant feature discovery | `BorutaSelector` |
| Model-aware nonlinear selection | `catboost_select` |

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
