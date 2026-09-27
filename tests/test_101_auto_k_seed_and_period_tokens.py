"""1.0.1: auto-k seed validation and pandas Period row metadata in manifests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sift import AutoKConfig, MRMRSelector, compare, select_cefsplus, select_mrmr
from sift.selection.view import _label_token


def _regression_data(n: int = 240):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.5, size=n)
    return X, y


def _seed_message(k_method: str, seed, route: str = "") -> str:
    return (
        "AutoKConfig.random_state must be a non-negative integer for "
        f"k_method={k_method!r}{route}, which draws seeded resamples or "
        f"shuffled folds; got {seed!r}"
    )


@pytest.mark.parametrize("seed", [None, 1.5, -1, "3", True])
@pytest.mark.parametrize(
    ("k_method", "strategy"),
    [
        ("perm_gap", "time_holdout"),
        ("knockoff_path", "time_holdout"),
        ("stability", "time_holdout"),
        ("consensus", "time_holdout"),
        ("gaussian_cv", "kfold"),
        ("xfit_objective", "kfold"),
    ],
)
def test_seeded_auto_k_rules_reject_a_seed_they_cannot_use(k_method, strategy, seed):
    X, y = _regression_data()
    config = AutoKConfig(k_method=k_method, strategy=strategy, random_state=seed)
    with pytest.raises(ValueError) as excinfo:
        select_cefsplus(X, y, k="auto", time=np.arange(len(y)), auto_k_config=config)
    assert str(excinfo.value) == _seed_message(k_method, seed)


def test_auto_router_names_the_seeded_rule_it_chose():
    X, y = _regression_data()
    config = AutoKConfig(k_method="auto", random_state=None)
    with pytest.raises(ValueError) as excinfo:
        select_mrmr(X, y, k="auto", task="regression", estimator="gaussian", auto_k_config=config)
    assert str(excinfo.value) == _seed_message(
        "gaussian_cv", None, " (chosen by k_method='auto': non_cefsplus_gaussian_selector)"
    )

    weights = np.r_[np.full(10, 200.0), np.ones(len(y) - 10)]
    with pytest.raises(ValueError) as excinfo:
        select_cefsplus(X, y, k="auto", sample_weight=weights, auto_k_config=config)
    assert str(excinfo.value) == _seed_message(
        "perm_gap", None, " (chosen by k_method='auto': heavy_weight_skew)"
    )


@pytest.mark.parametrize(
    ("k_method", "strategy"),
    [("evaluate", "time_holdout"), ("elbow", "time_holdout"), ("gaussian_cv", "time_holdout")],
)
def test_rules_that_never_read_the_seed_still_accept_none(k_method, strategy):
    X, y = _regression_data()
    time = np.arange(len(y))
    seeded = select_cefsplus(
        X, y, k="auto", time=time,
        auto_k_config=AutoKConfig(k_method=k_method, strategy=strategy),
    )
    with warnings.catch_warnings():
        # A non-default seed on a rule that ignores it keeps its advisory.
        warnings.filterwarnings("ignore", message="AutoKConfig.random_state is set")
        unseeded = select_cefsplus(
            X, y, k="auto", time=time,
            auto_k_config=AutoKConfig(k_method=k_method, strategy=strategy, random_state=None),
        )
    assert unseeded == seeded


def test_numpy_integer_seed_matches_the_python_int_seed():
    X, y = _regression_data()
    time = np.arange(len(y))
    as_int = select_cefsplus(
        X, y, k="auto", time=time, auto_k_config=AutoKConfig(k_method="perm_gap", random_state=7)
    )
    as_numpy = select_cefsplus(
        X, y, k="auto", time=time,
        auto_k_config=AutoKConfig(k_method="perm_gap", random_state=np.int64(7)),
    )
    assert as_numpy == as_int


def test_period_labels_have_an_exact_deterministic_token():
    token = _label_token(pd.Period("2000-01", freq="M"))
    assert token["type"].endswith(".Period")
    assert token["value"] == {"freq": "M", "ordinal": 360}
    # The same ordinal at another frequency is a different period.
    assert _label_token(pd.Period(ordinal=360, freq="D"))["value"] == {
        "freq": "D",
        "ordinal": 360,
    }


@pytest.mark.parametrize("wrap", [pd.PeriodIndex, np.asarray, pd.Series])
def test_compare_accepts_a_period_time_axis_and_digests_it(wrap):
    X, y = _regression_data()
    periods = pd.period_range("2000-01", periods=len(y), freq="M")
    factories = {"mrmr": lambda: MRMRSelector(k=2, task="regression")}
    first = compare(factories, X, y, time=wrap(periods))
    second = compare(factories, X, y, time=wrap(periods))
    shifted = compare(factories, X, y, time=wrap(periods + 1))
    digest = first.diagnostics["split"]["time_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert second.diagnostics["split"]["time_sha256"] == digest
    assert shifted.diagnostics["split"]["time_sha256"] != digest


def test_compare_row_metadata_without_a_token_is_recorded_as_opaque():
    class Stamp:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            return self.value < other.value

    X, y = _regression_data()
    stamps = np.array([Stamp(i) for i in range(len(y))], dtype=object)
    result = compare({"mrmr": lambda: MRMRSelector(k=2, task="regression")}, X, y, time=stamps)
    assert result.diagnostics["split"]["time_sha256"] == {
        "status": "opaque",
        "reason": "no_deterministic_token",
    }
