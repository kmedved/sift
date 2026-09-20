"""Closure fixes: ordinal level order, cross-dtype transform, tiny weights."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from sift import MRMRSelector, select_cefsplus
from sift._preprocess import CatEncoding, OneHotBlockEncoder, validate_inputs
from sift._unsupervised_cat import UnsupervisedCatEncoder


def _codes(values, *, col="c", weights=None):
    """Ordinal codes for a raw column, as a plain list of floats."""
    frame = pd.DataFrame({col: values})
    enc = UnsupervisedCatEncoder([col], method="ordinal").fit(frame, sample_weight=weights)
    return enc.transform(frame)[col].to_numpy().tolist()


def test_integer_levels_code_in_ascending_numeric_order():
    levels = list(range(1, 13))
    codes = _codes(levels * 2)
    # 1..12 are their own rank; the old repr order gave 1, 10, 11, 12, 2, ...
    assert codes == [float(v - 1) for v in levels] * 2


def test_numeric_levels_order_by_value_across_int_and_float():
    assert _codes([-2, -10, 0, 1]) == [1.0, 0.0, 2.0, 3.0]
    assert _codes([-2.5, -10.0, 0.0, 3.5, 1.0]) == [1.0, 0.0, 2.0, 4.0, 3.0]
    # Numerically equal int and float stay distinct levels, integer first.
    mixed = pd.Series([1.0, 1, 2], dtype=object)
    assert _codes(mixed) == [1.0, 0.0, 2.0]


def test_strings_keep_string_order_and_never_mix_with_numbers():
    assert _codes(["007", "10", "9"]) == [0.0, 1.0, 2.0]
    # Numbers rank before strings, each group in its own natural order.
    assert _codes([10, 9, "10", "9"]) == [1.0, 0.0, 2.0, 3.0]
    assert _codes(["1", 1, True]) == [2.0, 1.0, 0.0]


def test_datetime_bytes_and_other_kinds_order_by_value():
    stamps = [datetime(2021, 1, 2), datetime(2020, 6, 1), datetime(2021, 1, 1)]
    assert _codes(stamps) == [2.0, 0.0, 1.0]
    assert _codes([timedelta(days=3), timedelta(hours=1)]) == [1.0, 0.0]
    assert _codes([b"z", b"a"]) == [1.0, 0.0]


def test_ordered_categorical_uses_declared_order():
    ordered = pd.Categorical(
        ["mid", "low", "high", "mid"],
        categories=["low", "mid", "high"],
        ordered=True,
    )
    assert _codes(ordered) == [1.0, 0.0, 2.0, 1.0]

    unordered = pd.Categorical(
        ["mid", "low", "high", "mid"],
        categories=["low", "mid", "high"],
        ordered=False,
    )
    assert _codes(unordered) == [2.0, 1.0, 0.0, 2.0]

    # Declared order wins over numeric order, and codes stay dense when a
    # declared category is never observed.
    numeric = pd.Categorical([1, 10, 2], categories=[10, 2, 1, 7], ordered=True)
    assert _codes(numeric) == [2.0, 0.0, 1.0]


def test_missing_level_always_takes_the_last_code():
    assert _codes(["b", None, "a"]) == [1.0, 2.0, 0.0]
    assert _codes([5, np.nan, 1]) == [1.0, 2.0, 0.0]
    ordered = pd.Categorical(
        ["high", None, "low"], categories=["low", "high"], ordered=True
    )
    assert _codes(ordered) == [1.0, 2.0, 0.0]


def test_level_order_is_independent_of_row_order():
    values = [7, "b", True, 2.5, None, b"x", "a", 7, datetime(2000, 1, 1)]
    baseline = _codes(values)
    rng = np.random.default_rng(0)
    for _ in range(3):
        order = rng.permutation(len(values))
        permuted = [values[i] for i in order]
        assert _codes(permuted) == [baseline[i] for i in order]


def test_mixed_kinds_order_by_kind_then_value():
    values = ["b", "b", "b", 1, 1, 2.0, True, None]
    # bool < numeric < str < missing, so True=0, 1=1, 2.0=2, "b"=3, missing=4.
    assert _codes(values) == [3.0, 3.0, 3.0, 1.0, 1.0, 2.0, 0.0, 4.0]


def test_frequency_values_and_onehot_vocabulary_are_unchanged():
    values = ["b", "b", "b", 1, 1, 2.0, True, None]
    frame = pd.DataFrame({"c": values})
    freq = UnsupervisedCatEncoder(["c"], method="frequency").fit(frame)
    shares = freq.transform(frame)["c"].to_numpy().tolist()
    assert shares == [3 / 8, 3 / 8, 3 / 8, 2 / 8, 2 / 8, 1 / 8, 1 / 8, 1 / 8]

    onehot = OneHotBlockEncoder(["c"]).fit(frame)
    # Descending mass, ties broken by repr of the identity: ('bool', True) <
    # ('float', 2.0) < ('missing',).
    assert list(onehot.transform(frame).columns) == [
        "c__b",
        "c__1",
        "c__True",
        "c__2.0",
        "c__missing",
    ]


def test_monotone_ordinal_feature_beats_a_weaker_decoy():
    rng = np.random.default_rng(7)
    n = 200
    level = rng.integers(1, 13, size=n)
    y = level + rng.normal(0, 1.0, size=n)
    X = pd.DataFrame({"level": level, "decoy": 0.35 * y + rng.normal(size=n)})
    # The decoy is genuinely the weaker feature on the raw levels.
    assert abs(np.corrcoef(X["decoy"], y)[0, 1]) < abs(np.corrcoef(level, y)[0, 1])
    picked = select_cefsplus(
        X, y, k=1, cat_features=["level"], cat_encoding="ordinal",
        subsample=None, verbose=False,
    )
    assert list(picked) == ["level"]


def _cross_dtype_frames():
    train_float = pd.DataFrame({"g": [1.0, 2.0, 1.0, np.nan, 2.0, 1.0] * 4})
    score_int = pd.DataFrame({"g": [1, 2, 1, 2] * 2})
    return train_float, score_int


@pytest.mark.parametrize("method", ["ordinal", "frequency"])
def test_encoder_fitted_on_floats_recognises_an_integer_batch(method):
    train_float, score_int = _cross_dtype_frames()
    enc = UnsupervisedCatEncoder(["g"], method=method).fit(train_float)
    got = enc.transform(score_int)["g"].to_numpy()
    matched = enc.transform(score_int.astype(float))["g"].to_numpy()
    np.testing.assert_allclose(got, matched)
    # Hand-computed: 3 of 6 training rows are 1.0, 2 are 2.0, 1 is missing, and
    # the scoring batch alternates 1, 2.
    expected = {"ordinal": [0.0, 1.0], "frequency": [3 / 6, 2 / 6]}[method]
    np.testing.assert_allclose(got, expected * (len(score_int) // 2))
    assert enc.vocabulary_["g"]["identities"] == (("float", 1.0), ("float", 2.0), ("missing",))


@pytest.mark.parametrize("method", ["ordinal", "frequency"])
def test_encoder_fitted_on_integers_recognises_a_float_batch(method):
    train_int = pd.DataFrame({"g": [1, 2, 1, 2, 1, 3]})
    score_float = pd.DataFrame({"g": [1.0, 3.0, 2.0]})
    enc = UnsupervisedCatEncoder(["g"], method=method).fit(train_int)
    got = enc.transform(score_float)["g"].to_numpy()
    matched = enc.transform(pd.DataFrame({"g": [1, 3, 2]}))["g"].to_numpy()
    np.testing.assert_allclose(got, matched)
    expected = {"ordinal": [0.0, 2.0, 1.0], "frequency": [3 / 6, 1 / 6, 2 / 6]}[method]
    np.testing.assert_allclose(got, expected)
    assert enc.vocabulary_["g"]["identities"] == (("int", 1), ("int", 2), ("int", 3))


def test_onehot_fitted_on_floats_recognises_an_integer_batch():
    train_float, score_int = _cross_dtype_frames()
    enc = OneHotBlockEncoder(["g"]).fit(train_float)
    got = enc.transform(score_int)
    matched = enc.transform(score_int.astype(float))
    # Fitted dummy names stay in the float vocabulary; only lookup changes.
    assert list(got.columns) == ["g__1.0", "g__2.0", "g__missing"]
    np.testing.assert_allclose(got.to_numpy(), matched.to_numpy())
    assert got.to_numpy().sum() == len(score_int)


def test_cross_dtype_fallback_never_crosses_other_kinds():
    train = pd.DataFrame({"g": [1.0, 2.0, 1.0, 2.5]})
    enc = UnsupervisedCatEncoder(["g"], method="ordinal").fit(train)
    probe = pd.DataFrame({"g": ["1", True, 3, 2.5, 1]})
    got = enc.transform(probe)["g"].to_numpy().tolist()
    # str "1", bool True and the unfitted 3 stay unknown; 2.5 and int 1 match.
    assert got == [-1.0, -1.0, -1.0, 2.0, 0.0]
    big = pd.DataFrame({"g": [float(2**53)]})
    enc_big = UnsupervisedCatEncoder(["g"], method="ordinal").fit(big)
    # 2**53 + 1 is not representable as that float, so it must stay unknown.
    assert enc_big.transform(pd.DataFrame({"g": [2**53 + 1]}))["g"].tolist() == [-1.0]
    assert enc_big.transform(pd.DataFrame({"g": [2**53]}))["g"].tolist() == [0.0]


@pytest.mark.parametrize("encoding", ["ordinal", "frequency", "onehot"])
def test_selector_wrapper_transform_handles_an_integer_batch(encoding):
    train_float, score_int = _cross_dtype_frames()
    train = train_float.assign(noise=np.linspace(-1.0, 1.0, len(train_float)))
    score = score_int.assign(noise=np.linspace(-1.0, 1.0, len(score_int)))
    y = np.arange(len(train), dtype=float)
    selector = MRMRSelector(
        k=2, task="regression", cat_features=["g"], cat_encoding=encoding,
        subsample=None, verbose=False,
    ).fit(train, y)
    got = np.asarray(selector.transform(score), dtype=np.float64)
    matched = np.asarray(selector.transform(score.astype(float)), dtype=np.float64)
    np.testing.assert_allclose(got, matched)
    names = [str(name) for name in selector.get_feature_names_out()]
    columns = [i for i, name in enumerate(names) if name == "g" or name.startswith("g__")]
    assert columns
    block = got[:, columns]
    if encoding == "onehot":
        np.testing.assert_allclose(block.sum(axis=1), np.ones(len(score)))
    else:
        assert not np.any(block == (-1.0 if encoding == "ordinal" else 0.0))


def test_non_numeric_error_lists_every_accepted_encoding():
    X = pd.DataFrame({"c": ["a", "b"], "n": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Non-numeric columns found") as exc:
        validate_inputs(X, np.array([0.0, 1.0]), "regression")
    message = str(exc.value)
    assert "'ordinal'" in message and "'frequency'" in message
    # Every accepted value is offered; 'none' would leave the columns raw.
    for name in get_args(CatEncoding):
        assert (repr(name) in message) == (name != "none")


def test_tiny_positive_weights_keep_their_levels():
    frame = pd.DataFrame({"c": ["a", "b", "c"]})
    weights = np.array([1e308, 1e-300, 1e-300])
    ordinal = UnsupervisedCatEncoder(["c"], method="ordinal").fit(frame, sample_weight=weights)
    assert ordinal.vocabulary_["c"]["identities"] == (
        ("str", "a"),
        ("str", "b"),
        ("str", "c"),
    )
    assert ordinal.transform(frame)["c"].to_numpy().tolist() == [0.0, 1.0, 2.0]

    freq = UnsupervisedCatEncoder(["c"], method="frequency").fit(frame, sample_weight=weights)
    mapping = freq.vocabulary_["c"]["mapping"]
    assert set(mapping) == {("str", "a"), ("str", "b"), ("str", "c")}
    assert mapping[("str", "a")] == pytest.approx(1.0)
    # The shares of the tiny levels underflow, but they are fitted levels.
    assert mapping[("str", "b")] == 0.0
    # Zero-weight rows are still dropped from the vocabulary.
    zero = UnsupervisedCatEncoder(["c"], method="ordinal").fit(
        frame, sample_weight=np.array([1.0, 0.0, 1.0])
    )
    assert ("str", "b") not in zero.vocabulary_["c"]["mapping"]
    assert zero.transform(frame)["c"].to_numpy().tolist() == [0.0, -1.0, 1.0]
