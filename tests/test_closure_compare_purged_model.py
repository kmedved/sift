"""Closure regressions for compare, the purged splitters, and ModelSelector.

Every behavioural assertion here is checked against an oracle that does not
reuse the implementation: brute-force pairwise interval overlap for the purge,
hand-written embargo/cap rules for the splitters, ``hashlib`` recomputation of
the fold fingerprints, and ``collections.Counter`` for fold class coverage.
"""

from __future__ import annotations

import hashlib
import inspect
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from sift import (
    GroupPurgedTimeSeriesSplit,
    ModelSelector,
    PurgedTimeSeriesSplit,
    compare,
)
from sift.selection.compare import CompareResult, PREFIX_COLUMNS
from sift.selection.path_eval import evaluate_feature_path


# --------------------------------------------------------------------------
# shared fixtures / helpers
# --------------------------------------------------------------------------


def _frame(n=90, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(X["f0"] * 2.0 - X["f1"] + rng.normal(scale=0.3, size=n))
    return X, y


def _kbest(k=2):
    return {"kb": lambda: SelectKBest(f_regression, k=k)}


def _index_sha(idx) -> str:
    """Recompute compare's fold fingerprint without importing its helper."""
    arr = np.ascontiguousarray(np.asarray(idx, dtype=np.int64).reshape(-1))
    return hashlib.sha256(arr.tobytes()).hexdigest()


# --------------------------------------------------------------------------
# brute-force purged-split oracle (items 9 and 13)
# --------------------------------------------------------------------------


def _oracle_folds(
    time,
    event_end,
    *,
    n_splits,
    test_size,
    embargo,
    mode,
    max_train_size=None,
    groups=None,
):
    """Recompute the folds from the documented rules, independently.

    Overlap is decided by true pairwise closed-interval intersection against
    every validation row (the implementation uses a min/max shortcut), the
    embargo is plain float arithmetic on the timeline, and the
    ``max_train_size`` cap is applied to the post-purge/embargo survivors.
    Returns ``None`` for a configuration the documented rules cannot satisfy,
    which the splitter is then required to reject.
    """
    start = [float(v) for v in np.asarray(time)]
    end = start if event_end is None else [float(v) for v in np.asarray(event_end)]
    n = len(start)
    uniq = sorted(set(start))
    position = {value: index for index, value in enumerate(uniq)}
    tid = [position[value] for value in start]
    n_unique = len(uniq)

    if mode == "forward":
        width = test_size if test_size is not None else n_unique // (n_splits + 1)
        if width < 1 or n_unique - width * n_splits <= 0:
            return None
        blocks = [
            (n_unique - (n_splits - i) * width, width) for i in range(n_splits)
        ]
    else:
        if n_unique < n_splits:
            return None
        if test_size is None:
            base, extra = divmod(n_unique, n_splits)
            if base < 1:
                return None
            blocks = []
            cursor = 0
            for i in range(n_splits):
                width = base + (1 if i < extra else 0)
                if cursor == 0 and width == n_unique:
                    return None
                blocks.append((cursor, width))
                cursor += width
        else:
            if test_size * n_splits > n_unique:
                return None
            blocks = [(i * test_size, test_size) for i in range(n_splits)]

    folds = []
    for block_start, block_width in blocks:
        val_ids = set(range(block_start, block_start + block_width))
        val = [i for i in range(n) if tid[i] in val_ids]
        if not val:
            return None
        if mode == "forward":
            candidate_ids = set(range(block_start))
        else:
            candidate_ids = set(range(n_unique)) - val_ids
        keep = [
            i
            for i in range(n)
            if tid[i] in candidate_ids
            and all(
                not (start[i] <= end[v] and start[v] <= end[i]) for v in val
            )
        ]
        if embargo:
            val_start_min = min(start[v] for v in val)
            val_end_max = max(end[v] for v in val)
            before = {i for i in keep if end[i] < val_start_min - embargo}
            if mode == "forward":
                keep = sorted(before)
            else:
                after = {i for i in keep if start[i] > val_end_max + embargo}
                keep = sorted(before | after)
        if groups is not None:
            held_out = {groups[v] for v in val}
            keep = [i for i in keep if groups[i] not in held_out]
        if not keep:
            return None
        if max_train_size is not None:
            remaining = sorted({tid[i] for i in keep})
            if len(remaining) > max_train_size:
                if mode == "forward":
                    kept_ids = set(remaining[-max_train_size:])
                else:
                    low = block_start
                    high = block_start + block_width - 1
                    ordered = sorted(
                        remaining,
                        key=lambda j: (min(abs(j - low), abs(j - high)), j),
                    )
                    kept_ids = set(ordered[:max_train_size])
                # Tied timestamps are one unit: every row at a kept time id
                # stays, the cap counts distinct times.
                keep = [i for i in keep if tid[i] in kept_ids]
            if not keep:
                return None
        if max(start[i] for i in keep) >= min(start[v] for v in val) and mode == "forward":
            return None
        folds.append((np.asarray(keep, dtype=np.int64), np.asarray(val, dtype=np.int64)))
    return folds


def _random_timeline(rng, n, kind):
    if kind == "dense":
        base = np.arange(n)
    elif kind == "tied":
        base = np.sort(rng.integers(0, max(3, n // 3), size=n))
    else:
        base = np.cumsum(rng.integers(1, 5, size=n))
    return base


@pytest.mark.parametrize("mode", ["forward", "purged_kfold"])
def test_purged_folds_match_brute_force_overlap_oracle_on_mixed_dtypes(mode):
    """Embargo rules follow ``time``'s dtype, not ``event_end``'s (item 9)."""
    rng = np.random.default_rng(20260919)
    compared = 0
    horizon_cases = 0
    float_time_int_end = 0
    rejected = 0
    for _ in range(180):
        n = int(rng.integers(18, 55))
        base = _random_timeline(rng, n, rng.choice(["dense", "tied", "gapped"]))
        time_is_float = bool(rng.random() < 0.5)
        end_is_float = bool(rng.random() < 0.5)
        time = base.astype(np.float64) if time_is_float else base.astype(np.int64)
        horizon = int(rng.integers(0, 6))
        if horizon:
            raw_end = base + horizon + (0.5 if end_is_float else 0)
            event_end = (
                raw_end.astype(np.float64) if end_is_float else raw_end.astype(np.int64)
            )
            horizon_cases += 1
            if time_is_float and not end_is_float:
                float_time_int_end += 1
        else:
            event_end = None
        if time_is_float:
            embargo = float(rng.integers(0, 5)) + (0.5 if rng.random() < 0.5 else 0.0)
        else:
            embargo = int(rng.integers(0, 5))
        n_splits = int(rng.integers(2, 5))
        test_size = None if rng.random() < 0.6 else int(rng.integers(1, 4))
        cap = None if rng.random() < 0.5 else int(rng.integers(1, 9))
        splitter = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            embargo=embargo,
            mode=mode,
            max_train_size=cap,
        )
        expected = _oracle_folds(
            time,
            event_end,
            n_splits=n_splits,
            test_size=test_size,
            embargo=embargo,
            mode=mode,
            max_train_size=cap,
        )
        X = np.zeros((n, 1))
        if expected is None:
            with pytest.raises(ValueError):
                list(splitter.split(X, time=time, event_end=event_end))
            rejected += 1
            continue
        produced = list(splitter.split(X, time=time, event_end=event_end))
        assert len(produced) == len(expected)
        for (train, val), (exp_train, exp_val) in zip(produced, expected):
            assert np.array_equal(np.sort(train), exp_train)
            assert np.array_equal(np.sort(val), exp_val)
        compared += 1
    assert compared >= 80
    assert horizon_cases >= 40
    # The defect this pins: a float timeline with integer horizons used to
    # raise "integer timestamps require a ... integer embargo".
    assert float_time_int_end >= 10
    assert rejected >= 1


def test_float_time_with_integer_event_end_accepts_fractional_embargo():
    """Direct, hand-computed case for the dtype-dispatch defect (item 9)."""
    n = 24
    time = np.arange(n, dtype=np.float64)
    event_end = np.arange(n, dtype=np.int64) + 2
    splitter = PurgedTimeSeriesSplit(n_splits=3, embargo=0.5)
    produced = list(splitter.split(np.zeros((n, 1)), time=time, event_end=event_end))
    # Fold 0 validates unique times 6..11 (n_unique=24, test_size=24//4=6).
    # A training row survives when event_end < 6 - 0.5, i.e. row + 2 < 5.5,
    # i.e. row <= 3.
    assert produced[0][0].tolist() == [0, 1, 2, 3]
    assert produced[0][1].tolist() == list(range(6, 12))
    all_float = list(
        PurgedTimeSeriesSplit(n_splits=3, embargo=0.5).split(
            np.zeros((n, 1)), time=time, event_end=event_end.astype(np.float64)
        )
    )
    for (a_tr, a_va), (b_tr, b_va) in zip(produced, all_float):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)


def test_grouped_purged_cap_matches_oracle_on_staggered_panels():
    """``max_train_size`` after purge/embargo/group exclusion (item 13)."""
    rng = np.random.default_rng(4242)
    compared = 0
    for _ in range(60):
        entities = int(rng.integers(4, 8))
        periods = int(rng.integers(10, 20))
        groups = np.tile(np.arange(entities), periods)
        time = np.repeat(np.arange(periods, dtype=np.int64), entities)
        # Stagger entity lifespans so group exclusion cannot empty training.
        keep = rng.random(groups.size) < 0.7
        keep[: entities * 2] = groups[: entities * 2] % 2 == 0
        keep[-entities * 2 :] = groups[-entities * 2 :] % 2 == 1
        groups, time = groups[keep], time[keep]
        n = groups.size
        cap = int(rng.integers(1, 7))
        embargo = int(rng.integers(0, 3))
        n_splits = int(rng.integers(2, 4))
        splitter = GroupPurgedTimeSeriesSplit(
            n_splits=n_splits, embargo=embargo, max_train_size=cap
        )
        expected = _oracle_folds(
            time,
            None,
            n_splits=n_splits,
            test_size=None,
            embargo=embargo,
            mode="forward",
            max_train_size=cap,
            groups=list(groups),
        )
        X = np.zeros((n, 1))
        if expected is None:
            with pytest.raises(ValueError):
                list(splitter.split(X, groups=groups, time=time))
            continue
        produced = list(splitter.split(X, groups=groups, time=time))
        assert len(produced) == len(expected)
        for (train, val), (exp_train, exp_val) in zip(produced, expected):
            assert np.array_equal(np.sort(train), exp_train)
            assert np.array_equal(np.sort(val), exp_val)
        compared += 1
    assert compared >= 20


def test_max_train_size_cap_is_exact_and_keeps_tied_timestamps_together():
    time = np.repeat(np.arange(12, dtype=np.int64), 2)
    splitter = PurgedTimeSeriesSplit(n_splits=3, max_train_size=2)
    folds = list(splitter.split(np.zeros((24, 1)), time=time))
    # Fold 1 validates unique times 6..8; the two most recent surviving
    # training times are 4 and 5, each holding two tied rows.
    train, val = folds[1]
    assert sorted(set(time[train].tolist())) == [4, 5]
    assert train.tolist() == [8, 9, 10, 11]
    assert val.tolist() == [12, 13, 14, 15, 16, 17]


# --------------------------------------------------------------------------
# time / event_end dtype validation (item 10)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        np.array([f"2024-01-{i % 28 + 1:02d}" for i in range(12)], dtype=object),
        np.array([f"t{i:02d}" for i in range(12)]),
        np.arange(12, dtype=np.complex128),
        np.array([i % 2 == 0 for i in range(12)]),
        np.array([b"x"] * 12),
        pd.Categorical([f"p{i // 3}" for i in range(12)], ordered=True),
    ],
)
def test_unsupported_time_dtypes_are_rejected_by_name(values):
    with pytest.raises(ValueError, match=r"^time must be integer, unsigned integer"):
        list(PurgedTimeSeriesSplit(n_splits=2).split(np.zeros((12, 1)), time=values))


def test_unsupported_event_end_dtype_names_event_end():
    time = np.arange(12, dtype=np.int64)
    bad = np.array([f"t{i}" for i in range(12)])
    with pytest.raises(ValueError, match=r"^event_end must be integer"):
        list(
            PurgedTimeSeriesSplit(n_splits=2).split(
                np.zeros((12, 1)), time=time, event_end=bad
            )
        )


def test_supported_time_dtypes_still_build_folds():
    n = 12
    stamps = pd.date_range("2024-01-01", periods=n, freq="D")
    reference = list(
        PurgedTimeSeriesSplit(n_splits=2).split(
            np.zeros((n, 1)), time=stamps.to_numpy()
        )
    )
    # An object array of pandas Timestamps converts losslessly and is kept.
    as_objects = np.empty(n, dtype=object)
    as_objects[:] = list(stamps)
    produced = list(
        PurgedTimeSeriesSplit(n_splits=2).split(np.zeros((n, 1)), time=as_objects)
    )
    for (a_tr, a_va), (b_tr, b_va) in zip(reference, produced):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)
    deltas = (stamps - stamps[0]).to_numpy()
    assert len(list(PurgedTimeSeriesSplit(n_splits=2).split(np.zeros((n, 1)), time=deltas))) == 2


# --------------------------------------------------------------------------
# balanced panels (item 11) and the documented deviation (item 12)
# --------------------------------------------------------------------------


def test_balanced_panel_names_group_exclusion_and_offers_an_alternative():
    entities, periods = 5, 8
    groups = np.tile(np.arange(entities), periods)
    time = np.repeat(np.arange(periods, dtype=np.int64), entities)
    X = np.zeros((entities * periods, 1))
    with pytest.raises(ValueError, match="belongs to a group that also appears"):
        list(GroupPurgedTimeSeriesSplit(n_splits=3).split(X, groups=groups, time=time))
    with pytest.raises(ValueError, match=r"cv=PurgedTimeSeriesSplit"):
        list(GroupPurgedTimeSeriesSplit(n_splits=3).split(X, groups=groups, time=time))
    # The suggested route works on the same panel.
    folds = list(PurgedTimeSeriesSplit(n_splits=3).split(X, time=time))
    assert all(len(train) and len(val) for train, val in folds)


def test_empty_train_from_purge_alone_keeps_the_original_message():
    time = np.arange(9, dtype=np.int64)
    groups = np.arange(9)
    with pytest.raises(ValueError, match="after purge, embargo, or group exclusion"):
        list(
            GroupPurgedTimeSeriesSplit(n_splits=2, embargo=100).split(
                np.zeros((9, 1)), groups=groups, time=time
            )
        )


def test_model_selector_default_group_purged_split_surfaces_guidance():
    rng = np.random.default_rng(7)
    entities, periods = 5, 20
    groups = np.tile(np.arange(entities), periods)
    time = np.repeat(np.arange(periods, dtype=np.int64), entities)
    X = pd.DataFrame(rng.normal(size=(entities * periods, 3)), columns=list("abc"))
    y = pd.Series(X["a"] * 2 + rng.normal(size=entities * periods) * 0.2)
    with pytest.raises(ValueError, match="defaulted to GroupPurgedTimeSeriesSplit"):
        ModelSelector(Ridge(), n_features_to_select=None).fit(
            X, y, groups=groups, time=time
        )
    fitted = ModelSelector(
        Ridge(), n_features_to_select=None, cv=PurgedTimeSeriesSplit(n_splits=3)
    ).fit(X, y, time=time)
    assert fitted.selected_features_


def test_lopez_de_prado_embargo_deviation_is_named_in_the_docstring():
    doc = PurgedTimeSeriesSplit.__doc__
    assert "López de Prado" in doc
    assert "past side" in doc
    assert "after" in doc
    assert "both sides" in doc
    assert "López de Prado" in GroupPurgedTimeSeriesSplit.__doc__


# --------------------------------------------------------------------------
# compare: event_end passthrough (item 1)
# --------------------------------------------------------------------------


def test_compare_event_end_folds_equal_the_splitters_own_and_differ_from_points():
    X, y = _frame(n=80, p=3, seed=11)
    time = np.arange(80, dtype=np.int64)
    event_end = time + 6
    splitter = PurgedTimeSeriesSplit(n_splits=3, embargo=2)

    with_horizon = compare(_kbest(), X, y, cv=splitter, time=time, event_end=event_end)
    without = compare(_kbest(), X, y, cv=splitter, time=time)

    own_horizon = list(
        PurgedTimeSeriesSplit(n_splits=3, embargo=2).split(
            np.zeros((80, 1)), np.asarray(y), time=time, event_end=event_end
        )
    )
    own_points = list(
        PurgedTimeSeriesSplit(n_splits=3, embargo=2).split(
            np.zeros((80, 1)), np.asarray(y), time=time
        )
    )
    assert with_horizon.folds["train_index_sha256"].tolist() == [
        _index_sha(train) for train, _ in own_horizon
    ]
    assert with_horizon.folds["val_index_sha256"].tolist() == [
        _index_sha(val) for _, val in own_horizon
    ]
    assert without.folds["train_index_sha256"].tolist() == [
        _index_sha(train) for train, _ in own_points
    ]
    # The horizon purge is what makes event_end worth passing.
    assert with_horizon.folds["n_train"].tolist() != without.folds["n_train"].tolist()
    assert all(
        a < b
        for a, b in zip(
            with_horizon.folds["n_train"].tolist(), without.folds["n_train"].tolist()
        )
    )
    # Recorded as a digest only; the timestamps themselves are not retained.
    digest = with_horizon.diagnostics["split"]["event_end_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert without.diagnostics["split"]["event_end_sha256"] is None
    assert isinstance(with_horizon.diagnostics["split"]["time_sha256"], str)


def test_compare_event_end_accepts_the_time_column_shorthand():
    X, y = _frame(n=60, p=3, seed=12)
    time = np.arange(60, dtype=np.int64)
    framed = X.copy()
    framed["ts"] = time
    framed["te"] = time + 5
    splitter = PurgedTimeSeriesSplit(n_splits=2)
    via_columns = compare(_kbest(), framed, y, cv=splitter, time="ts", event_end="te")
    via_arrays = compare(_kbest(), X, y, cv=splitter, time=time, event_end=time + 5)
    assert (
        via_columns.folds["train_index_sha256"].tolist()
        == via_arrays.folds["train_index_sha256"].tolist()
    )
    # Both metadata columns are dropped from the design matrix.
    assert via_columns.diagnostics["n_features"] == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(cv=PurgedTimeSeriesSplit(n_splits=2)), "event_end requires time"),
        (dict(cv=KFold(n_splits=2), time=np.arange(60)), "KFold.split cannot use it"),
        (dict(time=np.arange(60)), "cannot use it"),
        (
            dict(cv=[(np.arange(40), np.arange(40, 60))], time=np.arange(60)),
            "precomputed split indices cannot use it",
        ),
    ],
)
def test_compare_rejects_event_end_the_split_route_cannot_use(kwargs, message):
    X, y = _frame(n=60, p=3, seed=13)
    with pytest.raises(ValueError, match=message):
        compare(_kbest(), X, y, event_end=np.arange(60) + 3, **kwargs)


def test_compare_rejects_misaligned_event_end():
    X, y = _frame(n=60, p=3, seed=14)
    with pytest.raises(ValueError, match="event_end has 10 rows but expected 60"):
        compare(
            _kbest(),
            X,
            y,
            cv=PurgedTimeSeriesSplit(n_splits=2),
            time=np.arange(60),
            event_end=np.arange(10),
        )


def test_evaluate_feature_path_event_end_changes_folds_and_is_keyword_only():
    X, y = _frame(n=70, p=3, seed=15)
    time = np.arange(70, dtype=np.int64)
    splitter = PurgedTimeSeriesSplit(n_splits=2, embargo=1)
    path = ["f0", "f1", "f2"]
    with_horizon = evaluate_feature_path(
        X, y.to_numpy(), path, [1, 2], estimator=Ridge(),
        splitter=splitter, time=time, event_end=time + 8,
    )
    without = evaluate_feature_path(
        X, y.to_numpy(), path, [1, 2], estimator=Ridge(),
        splitter=splitter, time=time,
    )
    assert with_horizon.scores != without.scores
    with pytest.raises(ValueError, match="default random holdout split cannot use it"):
        evaluate_feature_path(
            X, y.to_numpy(), path, [1, 2], estimator=Ridge(),
            time=time, event_end=time + 8,
        )
    parameters = inspect.signature(evaluate_feature_path).parameters
    assert parameters["event_end"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["event_end"].default is None
    # Additive: event_end is appended after the pre-existing keyword-only
    # parameters, so no existing keyword position moves.
    keyword_only = [
        name
        for name, value in parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert keyword_only[-1] == "event_end"
    assert keyword_only[:-1] == [
        "estimator",
        "estimator_factory",
        "scoring",
        "splitter",
        "val_frac",
        "random_state",
        "sample_weight",
        "groups",
        "time",
    ]


def test_compare_event_end_is_appended_after_existing_keyword_only_parameters():
    parameters = inspect.signature(compare).parameters
    keyword_only = [
        name
        for name, value in parameters.items()
        if value.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert keyword_only[-1] == "event_end"
    assert keyword_only[:-1] == [
        "estimator",
        "estimator_factory",
        "cv",
        "scoring",
        "groups",
        "time",
        "sample_weight",
        "mode",
        "task",
        "random_state",
        "val_frac",
    ]
    assert parameters["event_end"].default is None


# --------------------------------------------------------------------------
# compare: val_frac tolerance (item 2)
# --------------------------------------------------------------------------


def test_val_frac_accepts_equivalent_floats_and_still_rejects_real_changes():
    X, y = _frame(n=60, p=3, seed=16)
    baseline = compare(_kbest(), X, y, cv=3)
    for value in (0.2, np.float32(0.2), np.float64(0.2), 1 / 5):
        produced = compare(_kbest(), X, y, cv=3, val_frac=value)
        assert (
            produced.folds["train_index_sha256"].tolist()
            == baseline.folds["train_index_sha256"].tolist()
        )
    for value in (0.3, 0.19, np.float32(0.25)):
        with pytest.raises(ValueError, match="compare does not use val_frac"):
            compare(_kbest(), X, y, cv=3, val_frac=value)
    with pytest.raises(ValueError, match=r"val_frac must be a finite number"):
        compare(_kbest(), X, y, cv=3, val_frac=1.5)


# --------------------------------------------------------------------------
# compare: __repr__ (item 3)
# --------------------------------------------------------------------------


def test_compare_result_repr_is_compact_and_omits_the_tables():
    X, y = _frame(n=60, p=4, seed=17)
    result = compare(
        {
            "kb1": lambda: SelectKBest(f_regression, k=1),
            "kb2": lambda: SelectKBest(f_regression, k=2),
        },
        X,
        y,
        cv=3,
    )
    text = repr(result)
    assert text.count("\n") + 1 <= 8
    assert len(text) < 600
    assert text.startswith("CompareResult(selectors=['kb1', 'kb2'], n_folds=3")
    assert "mode='cv'" in text and "scoring='r2'" in text
    assert "kb1: score_mean=" in text and "kb2: score_mean=" in text
    # None of the seven frames are rendered in full.
    assert "train_index_sha256" not in text
    assert "selection_identity" not in text
    assert "__repr__" in CompareResult.__dict__


def test_compare_result_repr_truncates_many_selectors():
    X, y = _frame(n=60, p=8, seed=18)
    selectors = {
        f"kb{k}": (lambda k=k: SelectKBest(f_regression, k=k)) for k in range(1, 8)
    }
    text = repr(compare(selectors, X, y, cv=2))
    assert "... 2 more selector(s)" in text
    assert len(text) < 900


# --------------------------------------------------------------------------
# compare: stratified classification default (item 4)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cv", [None, 4])
def test_classification_default_cv_is_stratified_and_keeps_every_class(cv):
    rng = np.random.default_rng(19)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = np.zeros(n, dtype=int)
    y[:8] = 1  # 10% minority class
    result = compare(
        {"kb": lambda: SelectKBest(f_classif, k=2)},
        X,
        y,
        task="classification",
        cv=cv,
    )
    described = result.diagnostics["split"]
    assert described["type"].endswith("StratifiedKFold")
    assert described["params"]["shuffle"] is True
    assert described["params"]["random_state"] == 0

    n_splits = 5 if cv is None else cv
    oracle = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(
            np.empty((n, 1)), y
        )
    )
    assert result.folds["val_index_sha256"].tolist() == [
        _index_sha(val) for _, val in oracle
    ]
    # Independent check that the fix does what it is for: every validation
    # fold sees both classes.
    for _, val in oracle:
        counts = Counter(y[val].tolist())
        assert set(counts) == {0, 1}, counts
    # An unstratified shuffled KFold on the same data loses a class.
    plain = list(
        KFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.empty((n, 1)))
    )
    assert any(len(set(y[val].tolist())) < 2 for _, val in plain)


def test_regression_and_grouped_and_time_routes_keep_the_unstratified_default():
    X, y = _frame(n=90, p=4, seed=20)
    groups = np.repeat(np.arange(5), 18)
    time = np.arange(90, dtype=np.int64)

    regression = compare(_kbest(), X, y, cv=None)
    assert regression.diagnostics["split"]["type"].endswith("KFold")
    assert not regression.diagnostics["split"]["type"].endswith("StratifiedKFold")
    assert regression.folds["val_index_sha256"].tolist() == [
        _index_sha(val)
        for _, val in KFold(n_splits=5, shuffle=True, random_state=0).split(
            np.empty((90, 1))
        )
    ]

    grouped = compare(_kbest(), X, y, groups=groups, task="classification")
    assert grouped.diagnostics["split"]["type"].endswith("GroupKFold")
    assert grouped.folds["val_index_sha256"].tolist() == [
        _index_sha(val)
        for _, val in GroupKFold(n_splits=5).split(np.empty((90, 1)), None, groups)
    ]

    y_cls = (np.asarray(y) > np.median(y)).astype(int)
    timed = compare(
        {"kb": lambda: SelectKBest(f_classif, k=2)},
        X,
        y_cls,
        time=time,
        task="classification",
        cv=3,
    )
    assert timed.diagnostics["split"]["type"].endswith("KFold")
    assert not timed.diagnostics["split"]["type"].endswith("StratifiedKFold")


# --------------------------------------------------------------------------
# compare: generator splits (item 5) and empty-frame dtypes (item 6)
# --------------------------------------------------------------------------


def test_generator_of_split_pairs_is_materialized_once():
    X, y = _frame(n=60, p=3, seed=21)
    pairs = [
        (np.arange(0, 40), np.arange(40, 60)),
        (np.arange(20, 60), np.arange(0, 20)),
    ]
    from_list = compare(_kbest(), X, y, cv=list(pairs))
    from_generator = compare(_kbest(), X, y, cv=(pair for pair in pairs))
    from_map = compare(_kbest(), X, y, cv=map(tuple, pairs))
    for produced in (from_generator, from_map):
        assert (
            produced.folds["train_index_sha256"].tolist()
            == from_list.folds["train_index_sha256"].tolist()
        )
        assert produced.scores["score"].tolist() == from_list.scores["score"].tolist()
    with pytest.raises(TypeError, match="splitter must be None"):
        compare(_kbest(), X, y, cv=iter([1, 2, 3]))
    with pytest.raises(ValueError, match="at least one split"):
        compare(_kbest(), X, y, cv=iter([]))


def test_empty_prefix_scores_carries_the_populated_dtypes():
    X, y = _frame(n=60, p=3, seed=22)
    cv_result = compare(_kbest(), X, y, cv=3)
    path_result = compare(_kbest(), X, y, cv=3, mode="in_sample_path")
    assert cv_result.prefix_scores.empty
    assert not path_result.prefix_scores.empty
    assert list(cv_result.prefix_scores.columns) == list(PREFIX_COLUMNS)
    assert cv_result.prefix_scores.dtypes.to_dict() == {
        name: path_result.prefix_scores[name].dtype for name in PREFIX_COLUMNS
    }
    assert cv_result.prefix_scores["score"].dtype == np.dtype("float64")
    assert cv_result.prefix_scores["k"].dtype == np.dtype("int64")
    assert cv_result.prefix_scores["in_sample"].dtype == np.dtype("bool")
    # Concatenating an empty frame onto a populated one keeps the dtypes.
    stacked = pd.concat([cv_result.prefix_scores, path_result.prefix_scores])
    assert stacked["k"].dtype == np.dtype("int64")
    # The same rule reaches the single-selector overlap table.
    assert cv_result.overlap.empty
    assert cv_result.overlap["mean_jaccard"].dtype == np.dtype("float64")


# --------------------------------------------------------------------------
# compare: 2-D y (item 7) and docstring contracts (item 8)
# --------------------------------------------------------------------------


def test_two_dimensional_y_is_rejected_with_the_shared_wording():
    X, y = _frame(n=60, p=3, seed=23)
    wide = np.column_stack([np.asarray(y), np.asarray(y) * 2.0])
    with pytest.raises(ValueError) as excinfo:
        compare(_kbest(), X, wide, cv=3)
    message = str(excinfo.value)
    assert message.startswith(
        "2-D y is only supported for select_cefsplus / CEFSPlusSelector"
    )
    assert "select_cached(method='cefsplus')" in message
    assert "200 rows" not in message
    single_column = compare(_kbest(), X, np.asarray(y).reshape(-1, 1), cv=3)
    flat = compare(_kbest(), X, np.asarray(y), cv=3)
    assert (
        single_column.scores["score"].tolist() == flat.scores["score"].tolist()
    )


def test_compare_docstrings_state_the_paired_fold_and_winner_contracts():
    text = f"{compare.__doc__}\n{CompareResult.__doc__}"
    assert "same folds" in text or "same* folds" in text
    assert "paired by construction" in text
    assert "score_std" in text
    assert "not a standard error" in text or "It is not a" in text
    assert "outer holdout" in text
    assert "nested" in text
    assert "does not impute" in text or "no imputation" in text
    path_doc = evaluate_feature_path.__doc__
    assert "mean-imputed per training fold" in path_doc


# --------------------------------------------------------------------------
# ModelSelector (items 14, 15, 16)
# --------------------------------------------------------------------------


def test_threshold_and_n_resamples_docstrings_say_rejected_not_ignored():
    doc = ModelSelector.__doc__
    assert "Ignored for RFE/forward" not in doc
    assert "Rejected unless ``method='stability'``" in doc
    rng = np.random.default_rng(24)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = pd.Series(X["a"] * 2 + rng.normal(size=40) * 0.2)
    for method in ("rfe", "forward"):
        with pytest.raises(ValueError, match="threshold is only used"):
            ModelSelector(
                Ridge(), method=method, n_features_to_select=2, threshold=0.5
            ).fit(X, y)
        with pytest.raises(ValueError, match="n_resamples is only used"):
            ModelSelector(
                Ridge(), method=method, n_features_to_select=2, n_resamples=5
            ).fit(X, y)


def test_forward_rejects_a_non_default_importance_instead_of_ignoring_it():
    rng = np.random.default_rng(25)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = pd.Series(X["a"] * 3 + X["b"] + rng.normal(size=40) * 0.2)
    calls = {"n": 0}

    def spy(estimator):
        calls["n"] += 1
        return np.abs(np.asarray(estimator.coef_, dtype=np.float64))

    with pytest.raises(ValueError, match="importance is only used with method='rfe'"):
        ModelSelector(
            Ridge(), method="forward", n_features_to_select=2, importance=spy
        ).fit(X, y)
    assert calls["n"] == 0
    with pytest.raises(ValueError, match="importance is only used"):
        ModelSelector(
            Ridge(),
            method="forward",
            n_features_to_select=2,
            importance="permutation",
        ).fit(X, y)
    # The default stays accepted, and the other methods still consume it.
    assert ModelSelector(
        Ridge(), method="forward", n_features_to_select=2, importance="auto"
    ).fit(X, y).selected_features_ == ["a", "b"]
    assert ModelSelector(
        Ridge(), method="rfe", n_features_to_select=2, importance=spy
    ).fit(X, y).selected_features_
    assert calls["n"] > 0


def test_cv_provenance_records_the_effective_splitter_beside_the_request():
    rng = np.random.default_rng(26)
    n = 90
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(X["a"] * 3 + X["b"] + rng.normal(size=n) * 0.2)
    groups = np.repeat(np.arange(3), n // 3)
    time = np.arange(n, dtype=np.int64)

    grouped = ModelSelector(Ridge(), n_features_to_select=None, cv=5).fit(
        X, y, groups=groups
    )
    options = grouped.result_view().metadata["configured_options"]
    assert options["cv"] == 5  # the request is untouched
    effective = options["cv_effective"]
    assert effective["type"].endswith("GroupKFold")
    assert effective["params"]["n_splits"] == 3
    # GroupKFold with three groups can only run three folds, whatever cv said.
    assert effective["n_splits_effective"] == GroupKFold(n_splits=3).get_n_splits()
    assert effective["source"] == "resolved"

    timed = ModelSelector(Ridge(), n_features_to_select=None).fit(X, y, time=time)
    timed_options = timed.result_view().metadata["configured_options"]
    plain = ModelSelector(Ridge(), n_features_to_select=None).fit(X, y)
    plain_options = plain.result_view().metadata["configured_options"]
    assert timed_options["cv"] is None and plain_options["cv"] is None
    assert timed_options["cv_effective"]["type"].endswith("PurgedTimeSeriesSplit")
    assert timed_options["cv_effective"]["params"]["mode"] == "forward"
    assert plain_options["cv_effective"]["type"].endswith("KFold")
    assert plain_options["cv_effective"]["params"]["shuffle"] is True
    # The two runs that snapshot `cv: null` are now distinguishable.
    assert timed_options["cv_effective"] != plain_options["cv_effective"]

    caller = ModelSelector(
        Ridge(), n_features_to_select=None, cv=KFold(n_splits=2, shuffle=False)
    ).fit(X, y)
    caller_effective = caller.result_view().metadata["configured_options"]["cv_effective"]
    assert caller_effective["source"] == "caller"
    assert caller_effective["n_splits_effective"] == 2

    explicit = ModelSelector(Ridge(), n_features_to_select=2).fit(X, y)
    assert (
        explicit.result_view().metadata["configured_options"]["cv_effective"] is None
    )
    assert "cv_effective" in str(grouped.result_view().reproducibility_())
