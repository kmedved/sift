# Contributing to SIFT

Thank you for your interest in contributing to SIFT! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/sift.git
   cd sift
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/kmedved/sift.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all,test]"
```

### Dependencies

Core dependencies:
- `numpy>=1.18.1`
- `pandas>=1.0.3`
- `scikit-learn`
- `scipy`
- `numba`
- `joblib`

Optional dependencies:
- `catboost` - For CatBoost-based selection
- `category_encoders` - For advanced categorical encoding
- `pytest` - For running tests

## Code Style

### General Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep lines under 100 characters when possible
- Use descriptive variable names

### Type Hints

```python
from typing import List, Optional, Union
import numpy as np
import pandas as pd

def select_features(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    k: int,
    *,
    task: str = "regression",
    verbose: bool = True,
) -> List[str]:
    """Select features using the algorithm.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix with shape (n_samples, n_features).
    y : Series or ndarray
        Target variable with shape (n_samples,).
    k : int
        Number of features to select.
    task : str, default="regression"
        Task type: "regression" or "classification".
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    List[str]
        Selected feature names.
    """
    pass
```

### Docstring Format

We use NumPy-style docstrings:

```python
def function_name(param1, param2):
    """Short description of the function.

    Longer description if needed. Can span multiple
    paragraphs.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2. Default is None.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        When param1 is invalid.

    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3

    Notes
    -----
    Additional implementation notes.

    References
    ----------
    .. [1] Author, "Paper Title", Journal, Year.
    """
    pass
```

### Import Order

```python
# Standard library imports
from __future__ import annotations
import warnings
from typing import List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Local imports
from sift._preprocess import validate_inputs
from sift.estimators import relevance
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_smoke.py

# Run specific test function
pytest tests/test_smoke.py::test_mrmr_regression

# Run with coverage
pip install pytest-cov
pytest --cov=sift --cov-report=html
```

### Writing Tests

Tests are located in the `tests/` directory. Follow these guidelines:

1. **Test file naming**: `test_*.py`
2. **Test function naming**: `test_*`
3. **Use descriptive names**: `test_mrmr_regression_with_weights`
4. **One assertion per test when possible**
5. **Use fixtures for common setup**

Example test:

```python
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from sift import select_mrmr


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    return X, y


def test_mrmr_regression_returns_correct_count(regression_data):
    """Test that mRMR returns the requested number of features."""
    X, y = regression_data
    k = 5

    selected = select_mrmr(X, y, k=k, task="regression", verbose=False)

    assert len(selected) == k
    assert all(f in X.columns for f in selected)


def test_mrmr_regression_with_weights(regression_data):
    """Test that mRMR works with sample weights."""
    X, y = regression_data
    weights = np.ones(len(y))
    weights[:50] = 2.0

    selected = select_mrmr(
        X, y, k=5,
        task="regression",
        sample_weight=weights,
        verbose=False
    )

    assert len(selected) == 5


@pytest.mark.parametrize("k", [1, 5, 10])
def test_mrmr_regression_various_k(regression_data, k):
    """Test mRMR with various k values."""
    X, y = regression_data

    selected = select_mrmr(X, y, k=k, task="regression", verbose=False)

    assert len(selected) == k
```

### Test Categories

- **Smoke tests** (`test_smoke.py`): Basic API functionality
- **Unit tests**: Individual function/method tests
- **Integration tests**: End-to-end workflows
- **Performance tests**: Timing and memory benchmarks

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code
   - Add tests
   - Update documentation if needed

3. **Run tests locally**:
   ```bash
   pytest
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:
   - Go to GitHub and create a PR from your branch
   - Fill in the PR template
   - Link any related issues

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance tasks

Examples:
```
feat(mrmr): add gaussian estimator for faster computation
fix(stability): handle edge case with single feature
docs(readme): add example for time series data
test(catboost): add tests for group-aware bootstrap
```

### PR Checklist

Before submitting:

- [ ] Code follows the project's style guidelines
- [ ] Tests pass locally (`pytest`)
- [ ] New functionality includes tests
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, package versions

Example:

```markdown
## Bug Description
`select_mrmr` raises an error when using sample weights with classification.

## Steps to Reproduce
```python
from sift import select_mrmr
import numpy as np
import pandas as pd

X = pd.DataFrame(np.random.randn(100, 10))
y = np.random.randint(0, 2, 100)
weights = np.random.rand(100)

selected = select_mrmr(X, y, k=5, task="classification", sample_weight=weights)
```

## Expected Behavior
Should return 5 selected features.

## Actual Behavior
Raises `ValueError: ...`

## Environment
- Python 3.10.4
- sift 0.6.0
- numpy 1.24.0
- pandas 2.0.0
- OS: Ubuntu 22.04
```

## Feature Requests

When requesting a feature:

1. **Use case**: Describe the problem you're trying to solve
2. **Proposed solution**: How you'd like it to work
3. **Alternatives**: Other approaches you've considered
4. **Examples**: Code examples showing desired API

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

If you have questions about contributing:

1. Check existing issues and documentation
2. Open a new issue with the "question" label
3. Be specific about what you need help with

Thank you for contributing to SIFT!
