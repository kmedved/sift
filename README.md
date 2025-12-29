<p align="center">
<img src="https://raw.githubusercontent.com/smazzanti/mrmr/main/docs/img/mrmr_logo_white_bck.png" alt="sift logo" width="450"/>
</p>

# sift: feature selection toolbox

`sift` is a feature selection toolbox that brings together minimal-optimal and stability-focused selectors.
It includes **mRMR**, **JMI/JMIM**, **CEFS+** (and related Gaussian-copula variants), and **Stability Selection**.

## Supported selectors

- **mRMR / JMI / JMIM** (classification & regression; pandas, with a polars mRMR path)
- **CEFS+** (plus `mrmr_fcd` / `mrmr_fcq` variants)
- **Stability Selection** (regression & classification)

## Installation

```bash
pip install sift
```

### Extras

```bash
pip install "sift[all]"
pip install "sift[polars]"
pip install "sift[categorical]"
pip install "sift[numba]"
pip install "sift[test]"
```

## Quick examples

### mRMR / JMI / JMIM (pandas)

```python
import pandas as pd
from sklearn.datasets import make_classification
from sift import mrmr_classif

X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=10,
    n_redundant=40,
)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y)

# mRMR (default)
selected_mrmr = mrmr_classif(X=X, y=y, K=10)

# JMI / JMIM
selected_jmi = mrmr_classif(X=X, y=y, K=10, method="jmi")
selected_jmim = mrmr_classif(X=X, y=y, K=10, method="jmim")
```

### CEFS+ (regression)

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import cefsplus_regression

X, y = make_regression(n_samples=500, n_features=30, n_informative=8, noise=0.1)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
y = pd.Series(y)

selected_cefs = cefsplus_regression(X, y, K=8)
```

### Stability Selection

```python
import pandas as pd
from sklearn.datasets import make_regression
from sift import stability_regression

X, y = make_regression(n_samples=300, n_features=25, n_informative=6, noise=0.2)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

selected_stable = stability_regression(
    X,
    y,
    K=10,
    n_bootstrap=30,
    threshold=0.3,
    random_state=42,
    verbose=False,
)
```

### mRMR with Polars

```python
import polars as pl
import sift

data = [
    (1.0, 1.0, 1.0, 7.0, 1.5, -2.3),
    (2.0, None, 2.0, 7.0, 8.5, 6.7),
    (2.0, None, 3.0, 7.0, -2.3, 4.4),
    (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
    (4.0, 5.0, 4.0, 7.0, 12.1, -5.2),
]
columns = ["target", "some_null", "feature", "constant", "other_feature", "another_feature"]

df_polars = pl.DataFrame(data=data, schema=columns)
selected = sift.polars.mrmr_regression(df=df_polars, target_column="target", K=2)
```

## Reference

For an easy-going introduction to *mRMR*, read my article on **Towards Data Science**:
[“MRMR” Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b).

Also, this article describes an example of *mRMR* used on the **MNIST** dataset:
[Feature Selection: How To Throw Away 95% of Your Data and Get 95% Accuracy](https://towardsdatascience.com/feature-selection-how-to-throw-away-95-of-your-data-and-get-95-accuracy-ad41ca016877).

*mRMR* was born in **2003**, this is the original paper:
[Minimum Redundancy Feature Selection From Microarray Gene Expression Data](https://www.researchgate.net/publication/4033100_Minimum_Redundancy_Feature_Selection_From_Microarray_Gene_Expression_Data).

Since then, it has been used in many practical applications, due to its simplicity and effectiveness.
For instance, in **2019**, **Uber** engineers published a paper describing how they implemented MRMR in their marketing machine learning platform:
[Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/).
