"""Recorded randomness and single-pass wrapped configuration snapshots."""

import json

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectorMixin, VarianceThreshold

from sift import Stabilized, StabilitySelector, as_result


def test_stability_none_seed_can_be_replayed_without_changing_configured_none():
    rng = np.random.default_rng(170)
    X = rng.normal(size=(80, 6))
    y = X[:, 0] + 0.5 * rng.normal(size=80)
    base = StabilitySelector(n_bootstrap=5, n_jobs=1, verbose=False, store_coefs=True)
    with pytest.warns(FutureWarning, match="random_state=None"):
        fitted = base.fit(X, y)
    assert fitted.get_params()["random_state"] is None
    realized = fitted._actual_random_state_
    assert isinstance(realized, int)
    replay = clone(base).set_params(random_state=realized).fit(X, y)
    np.testing.assert_array_equal(fitted.selection_frequencies_, replay.selection_frequencies_)
    np.testing.assert_array_equal(fitted.coef_bootstrap_, replay.coef_bootstrap_)
    assert fitted.alpha_ == replay.alpha_
    view = as_result(fitted)
    assert view.metadata["random_state"] is None
    assert view.metadata["realized_random_state"] == realized
    assert view.metadata["configured_options"]["random_state"] is None
    assert view.metadata["n_rows_original"] == len(X)
    # Changing constructor options after fit does not rewrite its recorded seed.
    fitted.set_params(random_state=123)
    assert as_result(fitted).metadata["random_state"] is None
    assert as_result(fitted).metadata["realized_random_state"] == realized


class _SeedRecorder(SelectorMixin, BaseEstimator):
    records = []

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        values = np.asarray(X)
        type(self).records.append((self.random_state, values[:, 0].copy()))
        self.n_features_in_ = values.shape[1]
        self.support_ = np.random.default_rng(self.random_state).random(values.shape[1]) > 0.4
        return self

    def _get_support_mask(self):
        return self.support_


@pytest.mark.parametrize("base_seed", [None, 79])
def test_stabilized_seeds_unset_base_parameters_without_changing_draws(base_seed):
    X = np.column_stack([np.arange(50), np.sin(np.arange(50)), np.cos(np.arange(50))])
    base = _SeedRecorder(random_state=base_seed)
    _SeedRecorder.records = []
    fitted = Stabilized(base, n_resamples=3, random_state=9, verbose=False).fit(X)
    first_records = list(_SeedRecorder.records)
    replay = clone(fitted).fit(X)
    np.testing.assert_array_equal(fitted.selection_frequencies_, replay.selection_frequencies_)
    for i, ((seed, rows), child) in enumerate(zip(
        first_records, np.random.SeedSequence(9).spawn(3), strict=True
    )):
        expected_rows = np.random.default_rng(child).choice(50, size=25, replace=False)
        np.testing.assert_array_equal(rows, expected_rows)
        expected_seed = (
            int(np.random.SeedSequence(9, spawn_key=(i, 1)).generate_state(1)[0])
            if base_seed is None else base_seed
        )
        assert seed == expected_seed
    assert base.random_state == base_seed
    control = as_result(fitted).metadata["base_seed_control"]
    assert control["derived_parameters"] == (["random_state"] if base_seed is None else [])
    assert control["preserved_integer_parameters"] == ([] if base_seed is None else ["random_state"])


def test_nested_unseeded_base_repeats_and_snapshot_keeps_explicit_model_seed():
    rng = np.random.default_rng(171)
    X = rng.normal(size=(80, 6))
    y = X[:, 0] + rng.normal(size=80)
    base = SelectFromModel(RandomForestRegressor(n_estimators=5, max_depth=3))
    fitted = Stabilized(base, n_resamples=3, random_state=12, verbose=False).fit(X, y)
    replay = clone(fitted).fit(X, y)
    np.testing.assert_array_equal(fitted.selection_frequencies_, replay.selection_frequencies_)
    assert base.estimator.random_state is None
    assert as_result(fitted).metadata["base_seed_control"]["derived_parameters"] == [
        "estimator__random_state"
    ]
    explicit = clone(fitted).set_params(selector__estimator__random_state=17).fit(X, y)
    snapshot = as_result(explicit).metadata["configured_options"]["base_selector"]
    assert snapshot["params"]["estimator"]["params"]["random_state"] == 17
    json.dumps(snapshot)


def test_base_without_seed_parameters_is_not_claimed_seeded():
    X = np.arange(60).reshape(20, 3)
    fitted = Stabilized(VarianceThreshold(), n_resamples=1, verbose=False).fit(X)
    control = as_result(fitted).metadata["base_seed_control"]
    assert control["status"] == "not_declared"
    assert control["derived_parameters"] == []
