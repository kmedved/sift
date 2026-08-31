"""Small, deterministic fixtures for the legacy public-contract spine."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from sift import build_cache


@dataclass(frozen=True)
class ContractData:
    X: pd.DataFrame
    X_array: np.ndarray
    y: np.ndarray
    y_binary: np.ndarray
    sample_weight: np.ndarray
    cache: object


@pytest.fixture(scope="function")
def contract_data() -> ContractData:
    """The one fixed fixture used by all B0 compatibility tests."""
    rng = np.random.default_rng(907)
    n = 96
    signal = np.linspace(-2.0, 2.0, n)
    proxy = signal + rng.normal(0.0, 0.08, n)
    weak = rng.normal(size=n)
    noise = rng.normal(size=n)
    X = pd.DataFrame(
        {"signal": signal, "proxy": proxy, "weak": weak, "noise": noise}
    )
    y = 2.0 * signal + 0.25 * weak + rng.normal(0.0, 0.1, n)
    y_binary = (y >= np.median(y)).astype(np.int64)
    sample_weight = np.linspace(0.5, 1.5, n)
    return ContractData(
        X=X,
        X_array=X.to_numpy().copy(),
        y=y,
        y_binary=y_binary,
        sample_weight=sample_weight,
        cache=build_cache(X, subsample=None),
    )
