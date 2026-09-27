"""1.0.1: auto-k seed validation and row metadata (Period, one-column time) in manifests."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import enum
from fractions import Fraction
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from sift import (
    AutoKConfig,
    MRMRSelector,
    PurgedTimeSeriesSplit,
    build_cache,
    compare,
    evaluate_feature_path,
    gaussian_cv_curves,
    select_cefsplus,
    select_cefsplus_binary,
    select_jmi,
    select_k_auto,
    select_k_chi2_stop,
    select_k_gaussian_cv,
    select_k_knockoff_path,
    select_k_perm_gap,
    select_k_stability,
    select_mrmr,
)
from sift.selection.view import _label_token


def _regression_data(n: int = 240):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.5, size=n)
    return X, y


def _resample_message(k_method: str, seed, route: str = "") -> str:
    return (
        "AutoKConfig.random_state must be a non-negative integer for "
        f"k_method={k_method!r}{route}, which draws seeded resamples; got {seed!r}"
    )


def _kfold_message(k_method: str, seed, route: str = "") -> str:
    return (
        "AutoKConfig.random_state must be an integer in [0, 2**32 - 1] for "
        f"k_method={k_method!r} with strategy='kfold'{route}, which seeds the "
        f"shuffled folds; got {seed!r}"
    )


def _consensus_message(members, seed) -> str:
    return (
        "AutoKConfig.random_state must be an integer for k_method='consensus', "
        f"whose members {list(members)!r} derive their seeds from it; got {seed!r}"
    )


_SEEDED_RULES = {
    "perm_gap": ("time_holdout", _resample_message),
    "knockoff_path": ("time_holdout", _resample_message),
    "stability": ("time_holdout", _resample_message),
    "gaussian_cv": ("kfold", _kfold_message),
    "xfit_objective": ("kfold", _kfold_message),
}


_NON_INTEGER_SEEDS = pytest.mark.parametrize(
    "seed",
    [None, 1.5, "3", Decimal("7.5"), Fraction(15, 2), float("nan"), np.array([7])],
    ids=["none", "float", "str", "decimal", "fraction", "nan", "1-d-array"],
)


@_NON_INTEGER_SEEDS
@pytest.mark.parametrize("k_method", sorted(_SEEDED_RULES))
def test_seeded_auto_k_rules_reject_a_non_integer_seed(k_method, seed):
    X, y = _regression_data()
    strategy, message = _SEEDED_RULES[k_method]
    config = AutoKConfig(k_method=k_method, strategy=strategy, random_state=seed)
    with pytest.raises(ValueError) as excinfo:
        select_cefsplus(X, y, k="auto", time=np.arange(len(y)), auto_k_config=config)
    assert str(excinfo.value) == message(k_method, seed)


@pytest.mark.parametrize(
    ("k_method", "seed"),
    [
        ("perm_gap", -1),
        ("knockoff_path", -1),
        ("stability", np.int64(-3)),
        ("gaussian_cv", -1),
        ("gaussian_cv", 2**32),
        ("xfit_objective", 2**40),
    ],
)
def test_seeded_auto_k_rules_name_the_seed_range_they_need(k_method, seed):
    # These raised numpy's own "expected non-negative integer" / "Seed must
    # be between 0 and 2**32 - 1" before; the rule is unchanged, the message
    # now names AutoKConfig.random_state.
    X, y = _regression_data()
    strategy, message = _SEEDED_RULES[k_method]
    config = AutoKConfig(k_method=k_method, strategy=strategy, random_state=seed)
    with pytest.raises(ValueError) as excinfo:
        select_cefsplus(X, y, k="auto", time=np.arange(len(y)), auto_k_config=config)
    assert str(excinfo.value) == message(k_method, seed)


def test_a_bool_seed_is_the_integer_it_equals():
    X, y = _regression_data()
    time = np.arange(len(y))
    for k_method, (strategy, _message) in _SEEDED_RULES.items():
        as_bool = select_cefsplus(
            X, y, k="auto", time=time,
            auto_k_config=AutoKConfig(k_method=k_method, strategy=strategy, random_state=True),
        )
        as_int = select_cefsplus(
            X, y, k="auto", time=time,
            auto_k_config=AutoKConfig(k_method=k_method, strategy=strategy, random_state=1),
        )
        as_numpy_bool = select_cefsplus(
            X, y, k="auto", time=time,
            auto_k_config=AutoKConfig(
                k_method=k_method, strategy=strategy, random_state=np.bool_(True)
            ),
        )
        assert as_bool == as_int == as_numpy_bool, k_method


class _Seed(enum.IntEnum):
    SEVEN = 7


# Values int() has always turned into 7, and that are exactly 7.
_INTEGRAL_SEEDS = {
    "float": 7.0,
    "numpy-float": np.float64(7.0),
    "decimal": Decimal("7.0"),
    "fraction": Fraction(14, 2),
    "0-d-array": np.array(7),
    "0-d-float-array": np.array(7.0),
    "int-enum": _Seed.SEVEN,
}


@pytest.mark.parametrize("k_method", sorted(_SEEDED_RULES))
def test_an_integral_seed_is_the_integer_it_equals(k_method):
    # These worked through int() in 1.0.0 and must name the same stream.
    X, y = _regression_data()
    strategy, _message = _SEEDED_RULES[k_method]

    def run(seed):
        config = AutoKConfig(k_method=k_method, strategy=strategy, random_state=seed)
        result = select_cefsplus(
            X, y, k="auto", time=np.arange(len(y)), auto_k_config=config, return_result=True
        )
        return result.selected_features, result.diagnostics_["auto_k"]["selected_k"]

    expected = run(7)
    for name, seed in _INTEGRAL_SEEDS.items():
        assert run(seed) == expected, name


_CONSENSUS_MEMBER_LISTS = {
    "default": (("ebic", "chi2_stop", "perm_gap", "gaussian_cv"), ["perm_gap", "gaussian_cv"]),
    "gaussian_cv": (("ebic", "gaussian_cv"), ["gaussian_cv"]),
    "xfit_objective": (("ebic", "xfit_objective"), ["xfit_objective"]),
    "stability": (("ebic", "stability"), ["stability"]),
    "perm_gap": (("ebic", "perm_gap"), ["perm_gap"]),
}


@pytest.mark.parametrize("seed", [None, 1.5, "3"], ids=["none", "float", "str"])
@pytest.mark.parametrize("with_time", [False, True], ids=["no-time", "time"])
@pytest.mark.parametrize("members", sorted(_CONSENSUS_MEMBER_LISTS))
def test_every_consensus_that_reads_the_seed_rejects_a_non_integer(members, with_time, seed):
    # gaussian_cv / xfit_objective members read the seed whatever the
    # strategy, and fall back to shuffled kfold without time.
    X, y = _regression_data()
    methods, seeded = _CONSENSUS_MEMBER_LISTS[members]
    config = AutoKConfig(k_method="consensus", consensus_methods=methods, random_state=seed)
    time = np.arange(len(y)) if with_time else None
    with pytest.raises(ValueError) as excinfo:
        select_cefsplus(X, y, k="auto", time=time, auto_k_config=config)
    assert str(excinfo.value) == _consensus_message(seeded, seed)


@pytest.mark.filterwarnings(
    # Four members on a two-signal toy target disagree about k; incidental.
    "ignore:consensus auto-k methods disagree by more than 2x:UserWarning"
)
def test_consensus_accepts_any_integer_seed_as_before():
    # Each member's seed is derived from random_state % 2**32, so a negative
    # or oversized integer always worked and names the same streams.
    X, y = _regression_data()
    methods = ("ebic", "perm_gap", "gaussian_cv", "stability")

    def run(seed):
        config = AutoKConfig(
            k_method="consensus", consensus_methods=methods, random_state=seed,
            perm_B=6, boot_B=6,
        )
        result = select_cefsplus(X, y, k="auto", auto_k_config=config, return_result=True)
        votes = result.diagnostics_["auto_k_diagnostics"]
        return result.selected_features, votes[["method", "k_hat"]].to_dict("list")

    assert run(-1) == run(2**32 - 1)
    assert run(2**32 + 7) == run(7)
    assert run(True) == run(1)
    assert run(np.uint64(7)) == run(7)
    assert run(Decimal(7)) == run(7.0) == run(np.array(7)) == run(7)


def test_consensus_without_a_seeded_member_ignores_the_seed():
    X, y = _regression_data()
    methods = ("ebic", "chi2_stop")
    seeded = select_cefsplus(
        X, y, k="auto", auto_k_config=AutoKConfig(k_method="consensus", consensus_methods=methods)
    )
    with warnings.catch_warnings():
        # A non-default seed on members that ignore it keeps its advisory.
        warnings.filterwarnings("ignore", message="AutoKConfig.random_state is set")
        unseeded = select_cefsplus(
            X, y, k="auto",
            auto_k_config=AutoKConfig(
                k_method="consensus", consensus_methods=methods, random_state=None
            ),
        )
    assert unseeded == seeded


def test_consensus_xfit_member_uses_the_seed_so_setting_it_is_not_flagged_as_unused():
    # The member falls back to shuffled kfold here (no time), seeded from
    # random_state, so "random_state is set but ... does not use it" was wrong.
    X, y = _regression_data()
    config = AutoKConfig(
        k_method="consensus", consensus_methods=("ebic", "gaussian_cv"), random_state=7
    )
    with pytest.warns(UserWarning) as record:
        select_cefsplus(X, y, k="auto", auto_k_config=config)
    # Only the (incidental) disagreement advisory; no unused-field warning.
    assert [str(w.message) for w in record] == [
        "consensus auto-k methods disagree by more than 2x; k is ill-determined."
    ]


def _dense_check_config(seed) -> AutoKConfig:
    # auto_dense_min_frac=0 makes every EBIC pick "large", so the check runs.
    return AutoKConfig(
        k_method="auto", auto_dense_check=True, auto_dense_min_frac=0.0, random_state=seed
    )


_DENSE_CHECK_SKIPPED = (
    "Auto-K dense check could not run gaussian_cv/best; selected k is unchanged. Reason: "
)


def _dense_check_route(result) -> dict:
    return result.diagnostics_["auto_k"]["auto_routing"]["dense_check"]


@_NON_INTEGER_SEEDS
def test_auto_dense_check_skips_for_a_seed_it_cannot_use(seed):
    # The cross-check is a diagnostic: as in 1.0.0, a seed its shuffled
    # k-fold cannot use skips it with a warning and the EBIC k stands. 1.0.0
    # gave int()'s TypeError as the reason for None (and truncated 1.5); the
    # reason now names random_state.
    X, y = _regression_data()
    with pytest.warns(UserWarning) as record:
        result = select_cefsplus(
            X, y, k="auto", auto_k_config=_dense_check_config(seed), return_result=True
        )
    reason = "ValueError: " + _kfold_message("gaussian_cv", seed)
    assert [str(w.message) for w in record] == [_DENSE_CHECK_SKIPPED + reason]
    assert result.selected_features == ["f0", "f1"]
    route = _dense_check_route(result)
    assert (route["ran"], route["reason"], route["error"]) == (False, "gaussian_cv_failed", reason)


def test_auto_dense_check_runs_with_an_integral_seed_and_skips_an_out_of_range_one():
    X, y = _regression_data()
    with pytest.warns(UserWarning) as record:
        checked = select_cefsplus(
            X, y, k="auto", auto_k_config=_dense_check_config(1), return_result=True
        )
    ran = [str(w.message) for w in record]
    assert checked.selected_features == ["f0", "f1"]
    assert any(message.startswith("Auto-K dense-signal diagnostic") for message in ran)
    assert _dense_check_route(checked)["ran"] is True
    # Booleans and integral values run the same check on the same folds.
    for seed in (True, np.bool_(True), 1.0, Decimal(1), Fraction(1), np.array(1)):
        with pytest.warns(UserWarning) as seed_record:
            same = select_cefsplus(
                X, y, k="auto", auto_k_config=_dense_check_config(seed), return_result=True
            )
        assert same.selected_features == checked.selected_features
        assert [str(w.message) for w in seed_record] == ran
        assert _dense_check_route(same) == _dense_check_route(checked)

    # An integer KFold cannot use still only skips the diagnostic, as before.
    for seed in (-1, 2**40):
        with pytest.warns(UserWarning) as skip_record:
            skipped = select_cefsplus(X, y, k="auto", auto_k_config=_dense_check_config(seed))
        assert skipped == checked.selected_features
        assert [str(w.message) for w in skip_record] == [
            _DENSE_CHECK_SKIPPED + "ValueError: " + _kfold_message("gaussian_cv", seed)
        ]


def test_auto_dense_check_does_not_read_the_seed_on_a_time_holdout():
    X, y = _regression_data()
    time = np.arange(len(y))
    with pytest.warns(UserWarning) as seeded_record:
        seeded = select_cefsplus(X, y, k="auto", time=time, auto_k_config=_dense_check_config(42))
    with pytest.warns(UserWarning) as unseeded_record:
        unseeded = select_cefsplus(
            X, y, k="auto", time=time, auto_k_config=_dense_check_config(None)
        )
    assert unseeded == seeded
    assert [str(w.message) for w in unseeded_record] == [
        str(w.message) for w in seeded_record
    ]


def test_auto_router_names_the_seeded_rule_it_chose():
    X, y = _regression_data()
    config = AutoKConfig(k_method="auto", random_state=None)
    with pytest.raises(ValueError) as excinfo:
        select_mrmr(X, y, k="auto", task="regression", estimator="gaussian", auto_k_config=config)
    assert str(excinfo.value) == _kfold_message(
        "gaussian_cv", None, " (chosen by k_method='auto': non_cefsplus_gaussian_selector)"
    )

    weights = np.r_[np.full(10, 200.0), np.ones(len(y) - 10)]
    with pytest.raises(ValueError) as excinfo:
        select_cefsplus(X, y, k="auto", sample_weight=weights, auto_k_config=config)
    assert str(excinfo.value) == _resample_message(
        "perm_gap", None, " (chosen by k_method='auto': heavy_weight_skew)"
    )


def _classification_target(y):
    return (y > np.median(y)).astype(int)


_UNSUPPORTED_ROUTES = {
    "binary_perm_gap": (
        lambda X, y, c: select_cefsplus_binary(X, _classification_target(y), k="auto", auto_k_config=c),
        {"k_method": "perm_gap"},
        "CEFS+ binary does not support k_method='perm_gap'",
    ),
    "binary_gaussian_cv_kfold": (
        lambda X, y, c: select_cefsplus_binary(X, _classification_target(y), k="auto", auto_k_config=c),
        {"k_method": "gaussian_cv", "strategy": "kfold"},
        "CEFS+ binary does not support k_method='gaussian_cv'",
    ),
    "classic_mrmr_stability": (
        lambda X, y, c: select_mrmr(X, y, k="auto", task="regression", auto_k_config=c),
        {"k_method": "stability"},
        "mRMR does not support k_method='stability'",
    ),
    "classic_mrmr_consensus": (
        lambda X, y, c: select_mrmr(X, y, k="auto", task="regression", auto_k_config=c),
        {"k_method": "consensus"},
        "mRMR does not support k_method='consensus'",
    ),
    "gaussian_mrmr_perm_gap": (
        lambda X, y, c: select_mrmr(
            X, y, k="auto", task="regression", estimator="gaussian", auto_k_config=c
        ),
        {"k_method": "perm_gap"},
        "mRMR does not support k_method='perm_gap'",
    ),
    "gaussian_jmi_knockoff_path": (
        lambda X, y, c: select_jmi(
            X, y, k="auto", task="regression", estimator="gaussian", auto_k_config=c
        ),
        {"k_method": "knockoff_path"},
        "JMI does not support k_method='knockoff_path'",
    ),
    "MRMRSelector_consensus": (
        lambda X, y, c: MRMRSelector(
            k="auto", task="regression", estimator="gaussian", auto_k_config=c
        ).fit(X, y),
        {"k_method": "consensus"},
        "mRMR does not support k_method='consensus'",
    ),
    "select_k_auto_perm_gap": (
        lambda X, y, c: select_k_auto(X, y, list(X.columns), c, task="regression"),
        {"k_method": "perm_gap"},
        "select_k_auto supports only AutoKConfig(k_method='evaluate'). Use "
        "select_k_elbow(...) or a selector path that explicitly supports "
        "objective-path auto-k.",
    ),
    "select_k_chi2_stop_stability": (
        lambda X, y, c: select_k_chi2_stop(np.array([5.0, 2.0]), c, n_eff=240.0, p_candidates=6),
        {"k_method": "stability"},
        "select_k_chi2_stop requires AutoKConfig(k_method='chi2_stop')",
    ),
}


@pytest.mark.parametrize("route", sorted(_UNSUPPORTED_ROUTES))
def test_an_unsupported_k_method_is_reported_before_the_seed(route):
    X, y = _regression_data()
    call, fields, message = _UNSUPPORTED_ROUTES[route]
    for seed in (None, 42):
        with pytest.raises(ValueError) as excinfo:
            call(X, y, AutoKConfig(random_state=seed, **fields))
        assert str(excinfo.value) == message, seed


_INPUT_ERRORS_BEFORE_THE_SEED = {
    "ordinal_stability": (
        dict(cat_features=["level"], cat_encoding="ordinal"),
        "stability",
        "cat_encoding in {'ordinal', 'frequency'} is not supported with "
        "k_method='stability'",
    ),
    "blocks_perm_gap": (
        dict(feature_blocks={"pair": ["f0", "f1"]}),
        "perm_gap",
        "feature_blocks is not supported with k_method='perm_gap'",
    ),
}


@pytest.mark.parametrize("case", sorted(_INPUT_ERRORS_BEFORE_THE_SEED))
def test_a_rule_specific_input_error_is_reported_before_the_seed(case):
    X, y = _regression_data()
    X = X.assign(level=np.where(X["f2"] > 0.0, "hi", "lo"))
    kwargs, k_method, prefix = _INPUT_ERRORS_BEFORE_THE_SEED[case]
    errors = []
    for seed in (None, 42):
        with pytest.raises(ValueError) as excinfo:
            select_cefsplus(
                X, y, k="auto",
                auto_k_config=AutoKConfig(k_method=k_method, random_state=seed),
                **kwargs,
            )
        errors.append(str(excinfo.value))
    assert errors[0] == errors[1]
    assert errors[0].startswith(prefix)


def test_public_auto_k_helpers_check_the_seed_only_where_they_read_it():
    X, y = _regression_data()
    y_arr = y.to_numpy()
    cache = build_cache(X)

    def config(k_method, seed, **extra):
        return AutoKConfig(k_method=k_method, random_state=seed, min_k=0, max_k=6, **extra)

    # Precomputed nulls, bootstrap paths and fold curves carry their own
    # randomness: the selection rule applied to them never reads the seed.
    objective = np.array([5.0, 7.0, 7.1])
    nulls = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.35]])
    assert select_k_perm_gap(objective, nulls, config("perm_gap", None))[0] == (
        select_k_perm_gap(objective, nulls, config("perm_gap", 42))[0]
    )
    paths = [np.array([0, 1, 2]), np.array([0, 1, 3])]
    assert select_k_stability(paths, 6, config("stability", None))[0] == (
        select_k_stability(paths, 6, config("stability", 42))[0]
    )
    kfold = config("gaussian_cv", 42, strategy="kfold")
    curves = gaussian_cv_curves(
        cache, y_arr, config=kfold, top_m=6, corr_prune=None, method="cefsplus"
    )
    assert select_k_gaussian_cv(curves, replace(kfold, random_state=None))[0] == (
        select_k_gaussian_cv(curves, kfold)[0]
    )

    # The helpers that draw with the seed check it.
    with pytest.raises(ValueError) as excinfo:
        gaussian_cv_curves(
            cache, y_arr, config=replace(kfold, random_state=None), top_m=6,
            corr_prune=None, method="cefsplus",
        )
    assert str(excinfo.value) == _kfold_message("gaussian_cv", None)
    with pytest.raises(ValueError) as excinfo:
        select_k_knockoff_path(cache, y_arr, config("knockoff_path", -1), top_m=6)
    assert str(excinfo.value) == _resample_message("knockoff_path", -1)


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


@pytest.mark.parametrize("shape", ["column", "frame", "datetime-frame"])
@pytest.mark.parametrize("splitter", ["default", "time_series", "purged"])
def test_compare_digests_a_single_column_time_like_its_1d_values(shape, splitter):
    X, y = _regression_data()
    # Shuffled time: a digest that sorted the column, or read it in any order
    # but the rows', would differ from the 1-D digest.
    order = np.random.default_rng(3).permutation(len(y))
    flat = np.arange(len(y))[order]
    if shape == "datetime-frame":
        flat = pd.date_range("2020-01-01", periods=len(y), freq="D").to_numpy()[order]
    assert not pd.Index(flat).is_monotonic_increasing
    wrapped = flat.reshape(-1, 1) if shape == "column" else pd.DataFrame({"date": flat})
    cv = {
        "default": None,
        "time_series": TimeSeriesSplit(3),
        "purged": PurgedTimeSeriesSplit(3),
    }[splitter]
    factories = {"mrmr": lambda: MRMRSelector(k=2, task="regression")}
    one_d = compare(factories, X, y, time=flat, cv=cv)
    single_column = compare(factories, X, y, time=wrapped, cv=cv)
    digest = one_d.diagnostics["split"]["time_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert single_column.diagnostics["split"]["time_sha256"] == digest
    assert single_column.fold_bookkeeping == one_d.fold_bookkeeping
    # evaluate_feature_path, which flattens time before splitting, agrees.
    path = evaluate_feature_path(
        X, y, ["f0", "f1"], [1, 2], time=wrapped, splitter=PurgedTimeSeriesSplit(3)
    )
    splitter_record = path.reproducibility_()["configuration"]["configured"]["splitter"]
    assert splitter_record["time_sha256"] == digest


def test_compare_still_rejects_a_multi_column_time():
    X, y = _regression_data()
    time = np.column_stack([np.arange(len(y)), np.arange(len(y))])
    with pytest.raises(ValueError) as excinfo:
        compare({"mrmr": lambda: MRMRSelector(k=2, task="regression")}, X, y, time=time)
    assert str(excinfo.value) == "time must be 1-D when supplied for hashing"


def test_path_row_metadata_without_a_token_is_recorded_as_opaque():
    X, y = _regression_data()
    result = evaluate_feature_path(
        X, y, ["f0", "f1"], [1, 2], time=pd.interval_range(0, len(y))
    )
    splitter_record = result.reproducibility_()["configuration"]["configured"]["splitter"]
    assert splitter_record["time_sha256"] == {
        "status": "opaque",
        "reason": "no_deterministic_token",
    }
    assert splitter_record["event_end_sha256"] is None
