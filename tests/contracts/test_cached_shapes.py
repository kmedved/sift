"""Legacy tuple-shape contracts for named Gaussian caches."""

import warnings

import numpy as np
import pytest

import sift


def test_select_cached_omitted_defaults_match_explicit_current_defaults(contract_data):
    cache = sift.build_cache(contract_data.X.copy(), subsample=None)

    omitted = sift.select_cached(cache, contract_data.y.copy(), k=2)
    explicit = sift.select_cached(
        cache,
        contract_data.y.copy(),
        k=2,
        method="cefsplus",
        top_m=None,
        corr_prune="auto",
        return_objective=False,
        return_indices=False,
        warn_noise_floor=True,
        callback=None,
        return_result=False,
    )

    assert type(omitted) is type(explicit) is list
    assert omitted == explicit == ["signal", "weak"]


@pytest.mark.parametrize(
    "method",
    ("cefsplus", "jmi", "jmim", "mrmr_quot", "mrmr_diff"),
)
def test_select_cached_four_legacy_shapes(contract_data, method):
    expected_features = ["signal", "weak"]
    expected_indices = [0, 2]
    for return_objective, return_indices in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        # Build a new named cache for each shape so this contract also guards
        # against accidental mutation of cache-owned arrays between calls.
        cache = sift.build_cache(contract_data.X.copy(), subsample=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = sift.select_cached(
                cache,
                contract_data.y,
                k=2,
                method=method,
                return_objective=return_objective,
                return_indices=return_indices,
            )
        if method.startswith("mrmr_"):
            assert len(caught) == 1
            assert caught[0].category is UserWarning
            assert "noise floor" in str(caught[0].message)
        else:
            assert caught == []
        if not return_objective and not return_indices:
            assert type(value) is list
            assert value == expected_features
        elif return_objective and not return_indices:
            assert type(value) is tuple and len(value) == 2
            features, objective = value
            assert type(features) is list and features == expected_features
            assert type(objective) is np.ndarray and objective.shape == (2,)
            assert np.isfinite(objective).all()
        elif not return_objective and return_indices:
            assert type(value) is tuple and len(value) == 2
            features, indices = value
            assert type(features) is list and features == expected_features
            assert type(indices) is list and indices == expected_indices
        else:
            assert type(value) is tuple and len(value) == 3
            features, indices, objective = value
            assert type(features) is list and features == expected_features
            assert type(indices) is list and indices == expected_indices
            assert type(objective) is np.ndarray and objective.shape == (2,)
            assert np.isfinite(objective).all()
