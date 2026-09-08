"""Focused regressions for the accepted Stage 2 panel/proxy corrections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sift
from sift.selection.proxies import reject_unavailable_proxy_positions
from sift.selection.within import (
    fit_within_transform,
    require_seen_within_validation_levels,
)


def _view(block: pd.DataFrame, selected: list[int], names: list[str]) -> sift.SelectionView:
    path_rank = pd.array(
        [selected.index(i) + 1 if i in selected else pd.NA for i in range(len(names))],
        dtype="Int64",
    )
    table = pd.DataFrame(
        {
            "feature": names,
            "selected_index": pd.array(range(len(names)), dtype="Int64"),
            "path_rank": path_rank,
            "selected": [i in selected for i in range(len(names))],
        }
    )
    return sift.SelectionView(
        features=[names[i] for i in selected],
        indices=selected,
        raw_features=names,
        n_raw_features=len(names),
        raw_table=table,
        metadata={"table_complete": True},
        proxy_correlations=block,
    )


def test_within_transform_does_not_mutate_one_dimensional_input():
    X = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.0, 2.0, 2.0])
    groups = np.array([0, 0, 1, 1])
    fitted = fit_within_transform("groups", X.reshape(-1, 1), y, groups, None, np.ones(4))
    before = X.copy()

    transformed, _ = fitted.transform(X, y, groups)

    np.testing.assert_array_equal(X, before)
    np.testing.assert_allclose(transformed[:, 0], [-0.5, 0.5, -0.5, 0.5])


def test_within_validation_guard_preserves_direct_unseen_level_fallback():
    X = np.array([[0.0], [2.0]])
    y = np.array([0.0, 2.0])
    groups = np.array([0, 0])
    fitted = fit_within_transform("groups", X, y, groups, None, np.ones(2))

    with pytest.raises(ValueError, match="no validation entity can be demeaned"):
        require_seen_within_validation_levels(fitted, np.array([1, 1]))
    transformed, _ = fitted.transform(np.array([[100.0]]), np.array([100.0]), np.array([1]))
    np.testing.assert_allclose(transformed, [[99.0,]])


def test_empty_redundancy_report_preserves_declared_schema():
    block = pd.DataFrame(
        np.array([[1.0], [0.1]], dtype=np.float32),
        index=pd.Index([0, 1], name="selected_index"),
        columns=pd.Index([0], name="selected_index"),
    )
    report = _view(block, [0], ["a", "b"]).redundancy_report(0.9)

    assert report.empty
    assert report.dtypes.to_dict() == {
        "selected_feature": np.dtype(object),
        "selected_index": np.dtype("int64"),
        "feature": np.dtype(object),
        "candidate_index": np.dtype("int64"),
        "correlation": np.dtype("float64"),
    }


def test_r_min_boundary_matches_float32_storage_precision():
    block = pd.DataFrame(
        np.array([[1.0], [0.9]], dtype=np.float64),
        index=pd.Index([0, 1], name="selected_index"),
        columns=pd.Index([0], name="selected_index"),
    )
    view = _view(block, [0], ["a", "b"])

    assert float(view._proxy_correlations.loc[1, 0]) < 0.9
    assert len(view.proxies_at(0, 0.9)) == 1
    assert len(view.redundancy_report(0.9)) == 1
    assert len(view.proxy_clusters(0.9)) == 2


def test_unavailable_proxy_helper_names_selected_constants():
    with pytest.raises(ValueError, match="selected constant features"):
        reject_unavailable_proxy_positions(
            [0], available_original=[1], feature_names=["constant", "varying"]
        )


def test_stale_proxy_guidance_explains_threshold_expansion():
    rng = np.random.default_rng(19)
    n = 200
    signal = rng.normal(size=n)
    X = pd.DataFrame(
        {
            "a": signal,
            "b": signal + 0.02 * rng.normal(size=n),
            "c": rng.normal(size=n),
            "d": rng.normal(size=n),
        }
    )
    y = signal + 0.3 * rng.normal(size=n)
    selector = sift.StabilitySelector(
        n_bootstrap=10,
        threshold=0.9,
        store_proxies=True,
        store_coefs=False,
        random_state=0,
        verbose=False,
        n_jobs=1,
    ).fit(X, y)
    view = selector.set_threshold(0.05).result_view_

    with pytest.raises(NotImplementedError, match="threshold change added"):
        view.redundancy_report(0.8)


def _public_panel():
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(6), 5)
    time = np.tile(np.arange(5), 6)
    X = pd.DataFrame(rng.normal(size=(len(groups), 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() + 0.1 * rng.normal(size=len(groups))
    return X, y, groups, time


@pytest.mark.parametrize("method", ["gaussian_cv", "xfit_objective"])
def test_public_xfit_methods_reject_group_cv_with_unseen_entities(method):
    X, y, groups, _time = _public_panel()
    config = sift.AutoKConfig(
        k_method=method,
        strategy="group_cv",
        xfit_folds=3,
        min_k=1,
        max_k=2,
    )

    with pytest.raises(ValueError, match="Use overlapping-level splits"):
        sift.select_cefsplus(
            X,
            y,
            k="auto",
            groups=groups,
            within="groups",
            auto_k_config=config,
            verbose=False,
            subsample=None,
        )


@pytest.mark.parametrize("method", ["gaussian_cv", "xfit_objective"])
def test_public_xfit_methods_reject_two_way_time_holdout_without_seen_time(method):
    X, y, groups, time = _public_panel()
    config = sift.AutoKConfig(
        k_method=method,
        strategy="time_holdout",
        val_frac=0.2,
        min_k=1,
        max_k=2,
    )

    with pytest.raises(ValueError, match="no validation time can be demeaned"):
        sift.select_cefsplus(
            X,
            y,
            k="auto",
            groups=groups,
            time=time,
            within="two_way",
            auto_k_config=config,
            verbose=False,
            subsample=None,
        )
