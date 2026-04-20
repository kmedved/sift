import numpy as np
import pytest

from sift._permute import build_group_info, permute_matrix, resolve_permutation_method


def test_resolve_auto_with_time_only_uses_circular_shift():
    time = np.array([3, 1, 4, 2, 0])

    assert resolve_permutation_method("auto", groups=None, time=time) == "circular_shift"


def test_build_group_info_time_only_uses_implicit_single_group():
    time = np.array([30, 10, 20, 40])

    group_info = build_group_info(None, time, n_samples=time.size)

    assert list(group_info) == [0]
    assert group_info[0].tolist() == [1, 2, 0, 3]


def test_time_only_circular_shift_is_deterministic_and_shape_safe():
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    time = np.array([5, 1, 4, 2, 3, 0])
    seed = 7

    permuted = permute_matrix(
        X,
        method="circular_shift",
        groups=None,
        time=time,
        block_size="auto",
        seed=seed,
        axis="rows",
    )

    order = np.argsort(time, kind="mergesort")
    expected_shift = np.random.default_rng(seed).integers(1, X.shape[0])
    expected = X.copy()
    expected[order] = np.roll(X[order], expected_shift, axis=0)

    assert permuted.shape == X.shape
    assert np.array_equal(permuted, expected)
    assert not np.array_equal(permuted, X)


def test_time_only_block_is_deterministic_and_permutation_preserving():
    X = np.arange(18, dtype=np.float64).reshape(6, 3)
    time = np.array([5, 1, 4, 2, 3, 0])
    seed = 11

    permuted = permute_matrix(
        X,
        method="block",
        groups=None,
        time=time,
        block_size=2,
        seed=seed,
        axis="rows",
    )
    permuted_again = permute_matrix(
        X,
        method="block",
        groups=None,
        time=time,
        block_size=2,
        seed=seed,
        axis="rows",
    )

    assert permuted.shape == X.shape
    assert np.array_equal(permuted, permuted_again)
    assert sorted(map(tuple, permuted.tolist())) == sorted(map(tuple, X.tolist()))


def test_permute_matrix_rejects_invalid_method_and_axis():
    X = np.arange(6, dtype=np.float64).reshape(3, 2)

    with pytest.raises(ValueError, match="Unknown permutation method"):
        permute_matrix(
            X,
            method="bogus",  # type: ignore[arg-type]
            groups=None,
            time=None,
            block_size="auto",
            seed=0,
            axis="rows",
        )

    with pytest.raises(ValueError, match="Unknown permutation axis"):
        permute_matrix(
            X,
            method="global",
            groups=None,
            time=None,
            block_size="auto",
            seed=0,
            axis="diagonal",  # type: ignore[arg-type]
        )


def test_build_group_info_validates_lengths():
    groups = np.array([0, 0, 1])
    time = np.array([2, 1])

    with pytest.raises(ValueError, match="groups has 3 elements but expected 2"):
        build_group_info(groups, None, n_samples=2)

    with pytest.raises(ValueError, match="time has 2 elements but expected 3"):
        build_group_info(groups, time, n_samples=3)
