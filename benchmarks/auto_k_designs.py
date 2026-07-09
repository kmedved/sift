"""Ground-truth synthetic designs for Auto-K benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AutoKDesign:
    """One benchmark design with paired train/test samplers."""

    design_id: str
    make: Callable[[int, bool], tuple[pd.DataFrame, np.ndarray, dict]]
    sample_test: Callable[[int, int, bool], tuple[pd.DataFrame, np.ndarray]]


def _feature_frame(X: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _ar1_cov(p: int, rho: float) -> np.ndarray:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _linear_y(X: np.ndarray, beta: np.ndarray, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    return X @ beta + rng.normal(scale=sigma, size=X.shape[0])


def _d1_params(full: bool) -> tuple[int, int, np.ndarray]:
    n, p = ((4000, 400) if full else (2000, 200))
    beta = np.zeros(p)
    beta[:10] = np.linspace(1.5, 0.5, 10)
    return n, p, beta


def _d1(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p, beta = _d1_params(full)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = _linear_y(X, beta, rng)
    return _feature_frame(X), y, {"true_support": list(range(10)), "k_star": 10}


def _d1_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p, beta = _d1_params(full)
    rng = np.random.default_rng(seed + 10_000)
    X = rng.normal(size=(n_test, p))
    return _feature_frame(X), _linear_y(X, beta, rng)


def _d2_params(full: bool) -> tuple[int, int, np.ndarray, np.ndarray]:
    n, p = ((4000, 400) if full else (2000, 200))
    beta = np.zeros(p)
    support = np.array([0, 20, 40, 60, 80, 100, 120, 121, 140, 160], dtype=int)
    beta[support] = np.linspace(1.5, 0.5, support.size)
    return n, p, beta, support


def _d2(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p, beta, support = _d2_params(full)
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(p), _ar1_cov(p, 0.6), size=n)
    y = _linear_y(X, beta, rng)
    return _feature_frame(X), y, {"true_support": support.tolist(), "k_star": int(support.size)}


def _d2_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p, beta, _ = _d2_params(full)
    rng = np.random.default_rng(seed + 10_000)
    X = rng.multivariate_normal(np.zeros(p), _ar1_cov(p, 0.6), size=n_test)
    return _feature_frame(X), _linear_y(X, beta, rng)


def _d3_params(full: bool) -> tuple[int, int, np.ndarray, list[list[int]]]:
    n = 4000 if full else 2000
    n_blocks = 8
    block_size = 5
    pure_noise = 320 if full else 160
    p = n_blocks * block_size + pure_noise
    beta = np.zeros(n_blocks)
    beta[:] = np.linspace(1.4, 0.7, n_blocks)
    blocks = [list(range(i * block_size, (i + 1) * block_size)) for i in range(n_blocks)]
    return n, p, beta, blocks


def _sample_d3(seed: int, n: int, full: bool) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    _, p, beta, blocks = _d3_params(full)
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, len(blocks)))
    X = rng.normal(size=(n, p))
    for b, block in enumerate(blocks):
        X[:, block] = latent[:, [b]] + 0.23 * rng.normal(size=(n, len(block)))
    y = latent @ beta + rng.normal(size=n)
    return X, y, blocks


def _d3(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, _, _, _ = _d3_params(full)
    X, y, blocks = _sample_d3(seed, n, full)
    return _feature_frame(X), y, {"true_support": blocks, "k_star": len(blocks), "support_type": "blocks"}


def _d3_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    X, y, _ = _sample_d3(seed + 10_000, n_test, full)
    return _feature_frame(X), y


def _d4_params(full: bool) -> tuple[int, int, np.ndarray]:
    n, p = ((10_000, 800) if full else (5000, 400))
    beta = np.zeros(p)
    beta[:40] = 0.053
    return n, p, beta


def _d4(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p, beta = _d4_params(full)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = _linear_y(X, beta, rng)
    return _feature_frame(X), y, {"true_support": list(range(40)), "k_star": None}


def _d4_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p, beta = _d4_params(full)
    rng = np.random.default_rng(seed + 10_000)
    X = rng.normal(size=(n_test, p))
    return _feature_frame(X), _linear_y(X, beta, rng)


def _d5(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p = ((4000, 400) if full else (2000, 200))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    return _feature_frame(X), y, {"true_support": [], "k_star": 0}


def _d5_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p = ((4000, 400) if full else (2000, 200))
    rng = np.random.default_rng(seed + 10_000)
    return _feature_frame(rng.normal(size=(n_test, p))), rng.normal(size=n_test)


def _d6_monotone_transform(x: np.ndarray, j: int) -> np.ndarray:
    if j % 4 == 0:
        return np.sign(x) * np.log1p(np.abs(x))
    if j % 4 == 1:
        return np.arctan(x)
    if j % 4 == 2:
        return x / np.sqrt(1.0 + x * x)
    return np.tanh(0.5 * x)


def _d6(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p, beta = _d1_params(full)
    rng = np.random.default_rng(seed)
    X = rng.lognormal(mean=0.0, sigma=0.8, size=(n, p))
    heavy = rng.standard_t(df=3, size=(n, p // 5))
    X[:, : heavy.shape[1]] = heavy
    signal = np.zeros(n)
    for j in range(10):
        f = _d6_monotone_transform(X[:, j], j)
        signal += beta[j] * (f - np.mean(f)) / (np.std(f) + 1e-12)
    y = signal + rng.normal(size=n)
    return _feature_frame(X), y, {"true_support": list(range(10)), "k_star": 10}


def _d6_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    # Reuse the same monotone DGP by temporarily sampling at the requested size.
    _n, p, beta = _d1_params(full)
    rng = np.random.default_rng(seed + 10_000)
    X = rng.lognormal(mean=0.0, sigma=0.8, size=(n_test, p))
    heavy = rng.standard_t(df=3, size=(n_test, p // 5))
    X[:, : heavy.shape[1]] = heavy
    signal = np.zeros(n_test)
    for j in range(10):
        f = _d6_monotone_transform(X[:, j], j)
        signal += beta[j] * (f - np.mean(f)) / (np.std(f) + 1e-12)
    return _feature_frame(X), signal + rng.normal(size=n_test)


def _d7(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p = ((600, 4000) if full else (300, 2000))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:8] = np.linspace(1.8, 0.8, 8)
    y = _linear_y(X, beta, rng)
    return _feature_frame(X), y, {"true_support": list(range(8)), "k_star": 8}


def _d7_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p = ((600, 4000) if full else (300, 2000))
    rng = np.random.default_rng(seed + 10_000)
    X = rng.normal(size=(n_test, p))
    beta = np.zeros(p)
    beta[:8] = np.linspace(1.8, 0.8, 8)
    return _feature_frame(X), _linear_y(X, beta, rng)


def _d8(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p, beta, support = _d2_params(full)
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(p), _ar1_cov(p, 0.6), size=n)
    group_size = 10
    groups = np.repeat(np.arange(int(np.ceil(n / group_size))), group_size)[:n]
    group_effects = rng.normal(scale=1.0, size=int(groups.max()) + 1)
    time = np.arange(n)
    drift = 0.8 * np.sin(2.0 * np.pi * time / max(20, n // 4))
    y = X @ beta + group_effects[groups] + drift + rng.normal(size=n)
    return (
        _feature_frame(X),
        y,
        {
            "true_support": support.tolist(),
            "k_star": int(support.size),
            "groups": groups,
            "time": time,
        },
    )


def _d8_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p, beta, _support = _d2_params(full)
    rng = np.random.default_rng(seed + 10_000)
    X = rng.multivariate_normal(np.zeros(p), _ar1_cov(p, 0.6), size=n_test)
    group_size = 10
    groups = np.repeat(np.arange(int(np.ceil(n_test / group_size))), group_size)[:n_test]
    group_effects = rng.normal(scale=1.0, size=int(groups.max()) + 1)
    time = np.arange(n_test)
    drift = 0.8 * np.sin(2.0 * np.pi * time / max(20, n_test // 4))
    y = X @ beta + group_effects[groups] + drift + rng.normal(size=n_test)
    return _feature_frame(X), y


def _d9(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, p = ((50_000, 2000) if full else (10_000, 500))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:15] = np.linspace(1.5, 0.5, 15)
    y = _linear_y(X, beta, rng)
    return _feature_frame(X), y, {"true_support": list(range(15)), "k_star": 15}


def _d9_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    _, p = ((50_000, 2000) if full else (10_000, 500))
    rng = np.random.default_rng(seed + 10_000)
    X = rng.normal(size=(n_test, p))
    beta = np.zeros(p)
    beta[:15] = np.linspace(1.5, 0.5, 15)
    return _feature_frame(X), _linear_y(X, beta, rng)


def _d10_params(full: bool) -> tuple[int, int, int, int, int]:
    if full:
        return 90_000, 700, 685, 220, 600
    return 12_000, 350, 180, 120, 250


def _sample_d10(seed: int, n: int, full: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    _n, p, n_groups, n_signal, _max_k = _d10_params(full)
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, size=n, endpoint=False)
    X = rng.normal(size=(n, p)).astype(np.float32)

    heavy_width = min(p, max(20, p // 5))
    X[:, :heavy_width] = rng.standard_t(df=3, size=(n, heavy_width)).astype(np.float32)
    count_start = heavy_width
    count_stop = min(p, heavy_width + max(20, p // 4))
    counts = rng.negative_binomial(n=2, p=0.35, size=(n, count_stop - count_start))
    zero_mask = rng.random(size=counts.shape) < 0.35
    X[:, count_start:count_stop] = np.where(zero_mask, 0, counts).astype(np.float32)

    block_width = min(p - count_stop, max(0, p // 5))
    if block_width:
        latent = rng.normal(size=(n, 12)).astype(np.float32)
        loadings = rng.normal(scale=0.45, size=(12, block_width)).astype(np.float32)
        X[:, count_stop : count_stop + block_width] = (
            latent @ loadings + 0.6 * rng.normal(size=(n, block_width))
        ).astype(np.float32)

    signal = np.zeros(n, dtype=np.float64)
    beta = np.linspace(0.06, 0.012, n_signal)
    for j, b in enumerate(beta):
        col = X[:, j].astype(np.float64)
        if j % 3 == 0:
            col = np.sign(col) * np.log1p(np.abs(col))
        elif j % 3 == 1:
            col = np.arctan(col)
        col = (col - np.mean(col)) / (np.std(col) + 1e-12)
        signal += b * col
    group_effects = rng.normal(scale=0.25, size=n_groups)
    y = signal + group_effects[groups] + rng.normal(scale=1.0, size=n)
    return X, y, groups, list(range(n_signal))


def _d10(seed: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray, dict]:
    n, _p, _n_groups, _n_signal, max_k = _d10_params(full)
    X, y, groups, support = _sample_d10(seed, n, full)
    return (
        _feature_frame(X),
        y,
        {
            "true_support": support,
            "k_star": None,
            "groups": groups,
            "benchmark_max_k": max_k,
            "regime": "production_scale_dense_weak_grouped",
        },
    )


def _d10_test(seed: int, n_test: int, full: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    X, y, _groups, _support = _sample_d10(seed + 10_000, n_test, full)
    return _feature_frame(X), y


DESIGNS: dict[str, AutoKDesign] = {
    "D1": AutoKDesign("D1", _d1, _d1_test),
    "D2": AutoKDesign("D2", _d2, _d2_test),
    "D3": AutoKDesign("D3", _d3, _d3_test),
    "D4": AutoKDesign("D4", _d4, _d4_test),
    "D5": AutoKDesign("D5", _d5, _d5_test),
    "D6": AutoKDesign("D6", _d6, _d6_test),
    "D7": AutoKDesign("D7", _d7, _d7_test),
    "D8": AutoKDesign("D8", _d8, _d8_test),
    "D9": AutoKDesign("D9", _d9, _d9_test),
    "D10": AutoKDesign("D10", _d10, _d10_test),
}


def score_support(selected: list[int], meta: dict) -> tuple[float, float, float]:
    """Return precision, recall, F1 under point or block support semantics."""
    truth = meta.get("true_support", [])
    selected_set = {int(i) for i in selected}
    if not truth:
        precision = 1.0 if not selected_set else 0.0
        return precision, 1.0, 2.0 * precision / max(precision + 1.0, 1e-12)

    if meta.get("support_type") == "blocks":
        blocks = [set(map(int, block)) for block in truth]
        block_members = set().union(*blocks)
        tp_blocks = sum(1 for block in blocks if selected_set & block)
        fp = len(selected_set - block_members)
        precision = tp_blocks / max(1, tp_blocks + fp)
        recall = tp_blocks / len(blocks)
    else:
        truth_set = {int(i) for i in truth}
        tp = len(selected_set & truth_set)
        fp = len(selected_set - truth_set)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, len(truth_set))
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)
