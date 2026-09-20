"""Closure round: sklearn wrappers reject one-shot conditioning iterables.

The function APIs materialize ``include``/``exclude``/``candidates`` once per
call, so a generator is safe there.  The wrapper classes keep the constructor
value on ``self`` (sklearn forbids rewriting it in ``__init__``), so a
generator used to give a correct first fit, a silently unconditioned refit and
a ``clone`` that died on pickling.  They now refuse a one-shot iterator at
``fit`` while it is still unconsumed.

The rest of the file pins the factual claims the docstrings of
``sift/selectors.py`` and ``sift/selection/filter_api.py`` now make: the
sklearn parameter round trip, ``selector_metadata_``, the raw-namespace
``get_support()`` under one-hot, integer positions being DataFrame labels
only, prebuilt caches refusing every encoding, and ``eta`` still driving the
offset-zero counterfactual under e-value aggregation.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sift import (
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    JMIMSelector,
    JMISelector,
    KnockoffSelector,
    MRMRSelector,
    build_cache,
    build_classic_cache,
    select_cefsplus,
)

CONDITIONING_PARAMS = ("include", "exclude", "candidates")

#: ``name -> factory`` for objects that are their own iterator.
ONE_SHOT = {
    "generator": lambda values: (v for v in values),
    "map": lambda values: map(str, values),
    "filter": lambda values: filter(None, values),
    "iter": lambda values: iter(list(values)),
}


def _frame(n: int = 160, p: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = 2.0 * X["f0"] + 1.5 * X["f2"] + 0.3 * rng.normal(size=n)
    return X, y


def _estimator(cls, **kwargs):
    """Build one wrapper of each kind with a small, quiet configuration."""
    common = dict(verbose=False, **kwargs)
    if cls in (MRMRSelector, JMISelector, JMIMSelector):
        return cls(k=2, task="regression", estimator="gaussian", **common)
    if cls in (CEFSPlusSelector, CEFSPlusBinarySelector):
        return cls(k=2, **common)
    return cls(q=0.5, random_state=0, include_provenance="prespecified", **common)


FILTER_WRAPPERS = (
    MRMRSelector,
    JMISelector,
    JMIMSelector,
    CEFSPlusSelector,
    CEFSPlusBinarySelector,
)
ALL_WRAPPERS = FILTER_WRAPPERS + (KnockoffSelector,)


def _target(cls, y):
    return (y.to_numpy() > 0).astype(int) if cls is CEFSPlusBinarySelector else y


# --------------------------------------------------------------------------
# 1. one-shot iterators
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_WRAPPERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("param", CONDITIONING_PARAMS)
@pytest.mark.parametrize("kind", sorted(ONE_SHOT))
def test_one_shot_conditioning_iterable_is_refused_at_fit(cls, param, kind):
    X, y = _frame()
    values = ["f1"] if param != "candidates" else ["f1", "f2", "f3"]
    est = _estimator(cls, **{param: ONE_SHOT[kind](values)})

    with pytest.raises(TypeError, match=param):
        est.fit(X, _target(cls, y))

    # The estimator stays unfitted, and nothing was consumed on the way out.
    assert not hasattr(est, "selected_features_")
    assert list(getattr(est, param)) == (
        [str(v) for v in values] if kind == "map" else list(values)
    )


@pytest.mark.parametrize("cls", ALL_WRAPPERS, ids=lambda c: c.__name__)
def test_one_shot_error_names_the_parameter_and_the_remedy(cls):
    X, y = _frame()
    est = _estimator(cls, exclude=(name for name in ["f1"]))

    with pytest.raises(TypeError) as excinfo:
        est.fit(X, _target(cls, y))

    message = str(excinfo.value)
    assert "exclude" in message
    assert cls.__name__ in message
    assert "list" in message and "tuple" in message


@pytest.mark.parametrize(
    "container",
    [
        ["f1"],
        ("f1",),
        np.array(["f1"], dtype=object),
        pd.Index(["f1"]),
        pd.Series(["f1"]),
    ],
    ids=["list", "tuple", "ndarray", "Index", "Series"],
)
def test_reusable_containers_keep_working(container):
    """Only objects that are their own iterator are refused."""
    X, y = _frame()
    est = CEFSPlusSelector(k=2, exclude=container, verbose=False)

    first = list(est.fit(X, y).selected_features_)

    assert "f1" not in first
    # Oracle: excluding a column must equal selecting on the frame without it.
    assert first == select_cefsplus(X.drop(columns=["f1"]), y, 2, verbose=False)


def test_range_is_accepted_positionally_on_ndarray_input():
    """``range`` is a reusable sequence, not a one-shot iterator."""
    X, y = _frame()
    est = CEFSPlusSelector(k=2, exclude=range(1, 2), verbose=False)

    selected = list(est.fit(X.to_numpy(), y).selected_features_)

    assert selected == ["x0", "x2"]


def test_scalar_conditioning_reference_is_not_treated_as_an_iterator():
    """A bare label is not iterable-as-itself and must still be accepted."""
    X, y = _frame()

    est = CEFSPlusSelector(k=1, include="f0", verbose=False).fit(X, y)

    assert est.selected_features_[0] == "f0"


def test_refit_with_a_list_is_stable_and_clone_reproduces_it():
    X, y = _frame()
    est = CEFSPlusSelector(k=2, exclude=["f2"], verbose=False)

    first = list(est.fit(X, y).selected_features_)
    second = list(est.fit(X, y).selected_features_)
    cloned = list(clone(est).fit(X, y).selected_features_)

    assert first == second == cloned
    assert "f2" not in first
    # Oracle: the same answer as selecting on the frame without that column.
    assert first == select_cefsplus(X.drop(columns=["f2"]), y, 2, verbose=False)


def test_function_api_still_materializes_a_one_shot_iterable():
    X, y = _frame()

    from_generator = select_cefsplus(
        X, y, 2, verbose=False, exclude=(name for name in ["f2"])
    )
    from_list = select_cefsplus(X, y, 2, verbose=False, exclude=["f2"])

    assert from_generator == from_list
    assert "f2" not in from_generator


# --------------------------------------------------------------------------
# 2. sklearn parameter round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_WRAPPERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("encoding", ["onehot", "ordinal", "frequency"])
def test_conditioning_and_encoding_params_round_trip(cls, encoding):
    if cls is KnockoffSelector and encoding == "onehot":
        pytest.skip("KnockoffSelector rejects cat_encoding='onehot' by contract")
    settings = dict(
        include=["f0"],
        exclude=["f1"],
        candidates=["f2", "f3"],
        feature_blocks={"blk": ["f2", "f3"]},
        cat_encoding=encoding,
    )
    est = _estimator(cls, **settings)

    params = est.get_params()
    for key, value in settings.items():
        assert params[key] == value, key

    cloned = clone(est)
    assert cloned.get_params() == params
    for key, value in settings.items():
        assert getattr(cloned, key) == value, key

    # set_params round-trips the same values through a default-built estimator.
    fresh = _estimator(cls).set_params(**settings)
    assert {key: fresh.get_params()[key] for key in settings} == settings


# --------------------------------------------------------------------------
# 3. documented fitted attributes and the one-hot namespace split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", FILTER_WRAPPERS, ids=lambda c: c.__name__)
def test_filter_wrappers_publish_documented_selector_metadata(cls):
    X, y = _frame()
    est = _estimator(cls).fit(X, _target(cls, y))

    metadata = est.selector_metadata_
    assert isinstance(metadata, dict)
    for key in ("configured_options", "n_rows_original", "n_rows_used",
                "random_state", "subsample"):
        assert key in metadata, key
    assert metadata["n_rows_original"] == len(X)
    assert isinstance(metadata["configured_options"], dict)
    for key in ("include", "exclude", "candidates", "feature_blocks"):
        assert key in metadata["configured_options"], key

    documented = inspect.getdoc(cls)
    assert "selector_metadata_ : dict" in documented


def test_knockoff_wrapper_publishes_its_result_metadata():
    X, y = _frame()
    est = KnockoffSelector(q=0.5, random_state=0, verbose=False).fit(X, y)

    assert est.selector_metadata_ == est.result_.selector_metadata
    assert "configured_options" not in est.selector_metadata_
    assert est.selector_metadata_["n_rows_original"] == len(X)
    assert "selector_metadata_ : dict" in inspect.getdoc(KnockoffSelector)


def test_get_support_is_raw_while_transform_is_expanded_under_onehot():
    rng = np.random.default_rng(3)
    n = 200
    levels = rng.choice(list("abcd"), size=n)
    X = pd.DataFrame(
        {
            "cat": levels,
            "n1": rng.normal(size=n),
            "n2": rng.normal(size=n),
        }
    )
    y = 2.0 * (levels == "a") + X["n1"].to_numpy() + 0.1 * rng.normal(size=n)

    est = MRMRSelector(
        k=2, task="regression", cat_features=["cat"], cat_encoding="onehot",
        verbose=False,
    ).fit(X, y)

    mask = est.get_support()
    assert mask.shape == (X.shape[1],)
    assert "cat" in est.selected_features_
    # Oracle: one raw column per selected feature, one dummy per observed level.
    assert int(mask.sum()) == len(est.selected_features_)
    expected_width = len(est.selected_features_) - 1 + len(set(levels))
    assert est.transform(X).shape[1] == expected_width
    assert len(est.get_feature_names_out()) == expected_width
    assert int(mask.sum()) < est.transform(X).shape[1]
    assert set(est.get_support(indices=True)) <= set(range(X.shape[1]))


# --------------------------------------------------------------------------
# 4. conditioning references, caches and knockoff eta
# --------------------------------------------------------------------------


@pytest.mark.parametrize("param", CONDITIONING_PARAMS)
def test_integer_entries_are_labels_on_a_dataframe_and_positions_on_ndarray(param):
    X, y = _frame()
    values = [0] if param != "candidates" else [0, 1]

    with pytest.raises(ValueError, match="unknown feature"):
        CEFSPlusSelector(k=1, verbose=False, **{param: values}).fit(X, y)

    positional = CEFSPlusSelector(k=1, verbose=False, **{param: values})
    positional.fit(X.to_numpy(), y)
    assert positional.selected_features_  # positions are fine for an ndarray

    labelled = pd.DataFrame(X.to_numpy(), columns=[0, 1, 2, 3, 4])
    by_label = CEFSPlusSelector(k=1, verbose=False, **{param: values}).fit(labelled, y)
    assert by_label.selected_features_  # integers that ARE labels resolve


@pytest.mark.parametrize("encoding", ["onehot", "ordinal", "frequency"])
def test_prebuilt_caches_reject_every_target_blind_encoding(encoding):
    X, y = _frame()

    gaussian = build_cache(X)
    with pytest.raises(ValueError, match="prebuilt cache"):
        CEFSPlusSelector(k=2, verbose=False, cat_encoding=encoding, cache=gaussian).fit(
            X, y
        )

    classic = build_classic_cache(X)
    with pytest.raises(ValueError, match="prebuilt"):
        MRMRSelector(
            k=2, task="regression", estimator="classic", verbose=False,
            cat_encoding=encoding, cache=classic,
        ).fit(X, y)


def test_eta_drives_the_offset_zero_counterfactual_under_evalue_aggregation():
    """Documented ``eta`` contract: no effect on e-BH, still sets the diagnostic."""
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    y = 2.0 * X["f0"].to_numpy() + X["f1"].to_numpy() + 0.5 * rng.normal(size=len(X))

    runs = {}
    for eta in (0.1, 0.9):
        est = KnockoffSelector(
            q=0.5, n_draws=3, eta=eta, aggregation="evalues",
            random_state=2, verbose=False,
        ).fit(X, y)
        runs[eta] = (
            list(est.selected_features_),
            est.selector_metadata_["n_discoveries_offset_0"],
            list(est.selector_metadata_["n_discoveries_offset_0_per_draw"]),
        )

    low, high = runs[0.1], runs[0.9]
    assert low[0] == high[0], "e-BH selection must not depend on eta"
    assert low[2] == high[2], "the per-draw record is the same W in both runs"
    # Oracle: the offset-zero vote counts features chosen in >= eta of draws.
    per_draw = low[2]
    assert low[1] >= high[1]
    assert low[1] == max(per_draw)
    assert low[1] != high[1], "eta still moves the frequency-vote counterfactual"
