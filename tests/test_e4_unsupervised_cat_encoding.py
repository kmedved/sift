"""E4 unsupervised ordinal/frequency encoding: 1:1 maps, train-only inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sift import (
    AutoKConfig,
    CEFSPlusBinarySelector,
    CEFSPlusSelector,
    KnockoffSelector,
    MRMRSelector,
    as_result,
    build_cache,
    build_classic_cache,
    select_cefsplus,
    select_cefsplus_binary,
    select_k_auto,
    select_mrmr,
)
from sift._preprocess import _onehot_level_identity
from sift._unsupervised_cat import UnsupervisedCatEncoder


def _frame(n=40, seed=3):
    rng = np.random.default_rng(seed)
    city = np.array(["NY", "LA", "CHI", "NY"] * (n // 4 + 1), dtype=object)[:n]
    X = pd.DataFrame({"city": city, "x0": rng.normal(size=n)})
    y = (X["city"].eq("NY").astype(float) * 1.8 + 0.4 * X["x0"] + 0.1 * rng.normal(size=n))
    return X, y


def test_known_ordinal_and_frequency_mappings():
    X = pd.DataFrame({"city": ["LA", "NY", "CHI", "NY"], "x0": [0.0, 1.0, 2.0, 3.0]})
    w = np.array([1.0, 2.0, 1.0, 1.0])
    ordinal = UnsupervisedCatEncoder(["city"], method="ordinal").fit(X, y=np.arange(4), sample_weight=w)
    identities = ordinal.vocabulary_["city"]["identities"]
    assert identities == tuple(sorted(identities, key=repr))
    assert set(identities) == {("str", "CHI"), ("str", "LA"), ("str", "NY")}
    Xt = ordinal.transform(X)
    codes = {ident: i for i, ident in enumerate(identities)}
    expected = [codes[_onehot_level_identity(v)] for v in X["city"]]
    assert Xt["city"].to_numpy().tolist() == expected
    assert Xt["city"].dtype == np.float64
    assert list(Xt.columns) == ["city", "x0"]

    freq = UnsupervisedCatEncoder(["city"], method="frequency").fit(X, sample_weight=w)
    total = 5.0
    mapping = freq.vocabulary_["city"]["mapping"]
    assert mapping[("str", "NY")] == pytest.approx(3.0 / total)
    assert mapping[("str", "LA")] == pytest.approx(1.0 / total)
    assert mapping[("str", "CHI")] == pytest.approx(1.0 / total)
    assert freq.transform(X)["city"].to_numpy()[1] == pytest.approx(3.0 / total)


def test_zero_weight_rescaling_and_replication_equivalence():
    X = pd.DataFrame({"city": ["NY", "LA", "CHI", "NY"], "x0": [1.0, 2.0, 3.0, 4.0]})
    w = np.array([2.0, 0.0, 1.0, 1.0])
    freq = UnsupervisedCatEncoder(["city"], method="frequency").fit(X, sample_weight=w)
    assert ("str", "LA") not in freq.vocabulary_["city"]["mapping"]
    Xt = freq.transform(X)
    assert float(Xt.loc[1, "city"]) == 0.0
    scaled = UnsupervisedCatEncoder(["city"], method="frequency").fit(X, sample_weight=w * 3.0)
    assert freq.transform(X)["city"].to_numpy() == pytest.approx(scaled.transform(X)["city"].to_numpy())

    repeated_rows = pd.concat([X.iloc[[0]], X.iloc[[0]], X.iloc[[2]], X.iloc[[3]]], ignore_index=True)
    unit = np.ones(len(repeated_rows))
    replicated = UnsupervisedCatEncoder(["city"], method="frequency").fit(repeated_rows, sample_weight=unit)
    orig = freq.transform(X.iloc[[0, 2, 3]]).reset_index(drop=True)
    got = replicated.transform(repeated_rows.iloc[[0, 2, 3]]).reset_index(drop=True)
    assert orig["city"].to_numpy() == pytest.approx(got["city"].to_numpy())

    with pytest.raises(ValueError, match="positive"):
        UnsupervisedCatEncoder(["city"], method="ordinal").fit(X, sample_weight=np.zeros(len(X)))


def test_y_independence_and_declared_levels_ignored():
    X = pd.DataFrame(
        {
            "city": pd.Categorical(
                ["NY", "LA", "NY"],
                categories=["NY", "LA", "SEA"],
                ordered=True,
            ),
            "x0": [0.0, 1.0, 2.0],
        }
    )
    enc_a = UnsupervisedCatEncoder(["city"], method="ordinal").fit(X, y=np.array([0, 1, 0]))
    enc_b = UnsupervisedCatEncoder(["city"], method="ordinal").fit(X, y=np.array([9, 8, 7]))
    assert enc_a.vocabulary_["city"]["mapping"] == enc_b.vocabulary_["city"]["mapping"]
    assert ("str", "SEA") not in enc_a.vocabulary_["city"]["mapping"]
    perm = X.iloc[::-1].reset_index(drop=True)
    enc_rev = UnsupervisedCatEncoder(["city"], method="ordinal").fit(perm)
    assert enc_a.vocabulary_["city"]["identities"] == enc_rev.vocabulary_["city"]["identities"]

    selected_a = select_mrmr(
        X, np.array([0.0, 1.0, 0.0]), k=1, task="regression",
        cat_encoding="ordinal", subsample=None, verbose=False,
    )
    selected_b = select_mrmr(
        X, np.array([3.0, -1.0, 8.0]), k=1, task="regression",
        cat_encoding="ordinal", subsample=None, verbose=False,
    )
    assert selected_a == selected_b


def test_unseen_missing_and_transform_does_not_refit():
    X = pd.DataFrame({"city": ["NY", "LA", None], "x0": [0.0, 1.0, 2.0]})
    enc_o = UnsupervisedCatEncoder(["city"], method="ordinal").fit(X)
    enc_f = UnsupervisedCatEncoder(["city"], method="frequency").fit(X)
    assert ("missing",) in enc_o.vocabulary_["city"]["mapping"]
    Xt = enc_o.transform(pd.DataFrame({"city": ["BOS", None, "NY"], "x0": [0.0, 1.0, 2.0]}))
    assert float(Xt.loc[0, "city"]) == -1.0
    assert float(Xt.loc[1, "city"]) == enc_o.vocabulary_["city"]["mapping"][("missing",)]
    Xf = enc_f.transform(pd.DataFrame({"city": ["BOS", None], "x0": [0.0, 1.0]}))
    assert float(Xf.loc[0, "city"]) == 0.0

    train = pd.DataFrame({"city": ["NY", "LA"], "x0": [0.0, 1.0]})
    enc = UnsupervisedCatEncoder(["city"], method="frequency").fit(train)
    only_missing = UnsupervisedCatEncoder(["city"], method="frequency").fit(train)
    new = pd.DataFrame({"city": [None, "SEA"], "x0": [0.0, 1.0]})
    out = enc.transform(new)
    assert out["city"].to_numpy().tolist() == [0.0, 0.0]
    assert ("missing",) not in only_missing.vocabulary_["city"]["mapping"]


def test_class_weight_does_not_enter_unsupervised_maps(monkeypatch):
    rng = np.random.default_rng(4)
    n = 60
    X = pd.DataFrame(
        {
            "city": np.array(["a"] * 50 + ["b"] * 10, dtype=object),
            "x0": rng.normal(size=n),
        }
    )
    y = np.array([0] * 50 + [1] * 10)
    seen = []
    orig = UnsupervisedCatEncoder.fit

    def spy(self, X_fit, y=None, sample_weight=None):
        seen.append(None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy())
        return orig(self, X_fit, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(UnsupervisedCatEncoder, "fit", spy)
    select_cefsplus_binary(
        X, y, k=1, cat_encoding="frequency", subsample=None, verbose=False,
    )
    select_cefsplus_binary(
        X, y, k=1, cat_encoding="frequency", class_weight="balanced",
        subsample=None, verbose=False,
    )
    assert seen
    for weights in seen:
        if weights is None:
            continue
        assert np.allclose(weights, weights[0]) or np.all(weights == 1.0)
    mixed = UnsupervisedCatEncoder(["city"], method="frequency").fit(
        X, sample_weight=np.where(y == 1, 5.0, 1.0),
    )
    plain = UnsupervisedCatEncoder(["city"], method="frequency").fit(X)
    assert mixed.vocabulary_["city"]["mapping"] != plain.vocabulary_["city"]["mapping"]


def test_function_wrapper_parity_and_raw_1to1_transform():
    X, y = _frame()
    fn = select_mrmr(
        X, y, k=1, task="regression", cat_encoding="ordinal", subsample=None, verbose=False,
    )
    sel = MRMRSelector(k=1, task="regression", cat_encoding="ordinal", subsample=None, verbose=False)
    sel.fit(X, y)
    assert list(sel.selected_features_) == fn
    Xt = sel.transform(X)
    assert Xt.shape[1] == 1
    assert list(sel.get_feature_names_out()) == fn
    view = as_result(
        select_cefsplus(
            X, y, k=1, cat_encoding="frequency", subsample=None, verbose=False, return_result=True,
        ),
        input_features=list(X.columns),
    )
    assert set(view.features) <= set(X.columns)
    new = X.copy()
    new.loc[0, "city"] = "UNSEEN"
    transformed = sel.transform(new)
    assert transformed.shape == (len(X), 1)
    enc = sel.categorical_encoder_
    assert isinstance(enc, UnsupervisedCatEncoder)
    assert ("str", "UNSEEN") not in enc.vocabulary_["city"]["mapping"]


def test_defaults_and_numeric_none_unchanged():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x0": rng.normal(size=30), "x1": rng.normal(size=30)})
    y = X["x0"] + 0.1 * rng.normal(size=30)
    assert select_mrmr(X, y, k=1, task="regression", subsample=None, verbose=False) == select_mrmr(
        X, y, k=1, task="regression", cat_encoding="none", subsample=None, verbose=False,
    )


def test_cache_wall_and_preencoded_numeric_cache():
    X, y = _frame()
    cache = build_cache(X[["x0"]], subsample=None)
    with pytest.raises(ValueError, match="encoding provenance"):
        MRMRSelector(
            k=1, task="regression", estimator="gaussian",
            cat_encoding="ordinal", cache=cache, verbose=False,
        ).fit(X, y)
    classic = build_classic_cache(X[["x0"]], subsample=None)
    with pytest.raises(ValueError, match="encoding provenance"):
        MRMRSelector(
            k=1, task="regression", estimator="classic",
            cat_encoding="frequency", cache=classic, verbose=False,
        ).fit(X, y)
    numeric = UnsupervisedCatEncoder(["city"], method="ordinal").fit_transform(X)
    out = select_mrmr(
        numeric, y, k=1, task="regression", subsample=None, verbose=False,
    )
    assert out


def test_full_data_hatch_rejected():
    X, y = _frame()
    with pytest.raises(ValueError, match="allow_full_data_target_encoding=True"):
        select_mrmr(
            X, y, k=1, task="regression", cat_encoding="ordinal",
            allow_full_data_target_encoding=True, subsample=None, verbose=False,
        )


def test_knockoff_accepts_without_fdr_upgrade():
    rng = np.random.default_rng(5)
    n = 120
    X = pd.DataFrame(
        {
            "city": np.array(["NY", "LA", "CHI"] * (n // 3), dtype=object),
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = 2.0 * X["x0"] + 0.1 * rng.normal(size=n)
    sel = KnockoffSelector(q=0.5, cat_encoding="ordinal", subsample=None, verbose=False)
    sel.fit(X, y)
    md = dict(sel.selector_metadata_ or {})
    assert "validity_note" not in md


def test_boruta_train_ok_test_importance_rejected():
    from sift import select_boruta

    X, y = _frame(n=50)
    selected = select_boruta(
        X, y, task="regression", cat_encoding="frequency",
        max_iter=3, random_state=0, verbose=False,
    )
    assert set(selected) <= set(X.columns)
    with pytest.raises(ValueError, match="importance_data='test'") as exc:
        select_boruta(
            X, y, task="regression", cat_encoding="ordinal",
            importance_data="test", importance="shap",
            max_iter=2, random_state=0, verbose=False,
        )
    assert "supervised" not in str(exc.value).lower()


def test_resampled_auto_k_rejected():
    X, y = _frame()
    with pytest.raises(ValueError, match="stability"):
        select_cefsplus(
            X, y, k="auto", cat_encoding="frequency",
            auto_k_config=AutoKConfig(k_method="stability", min_k=1, max_k=2),
            subsample=None, verbose=False,
        )


def _spy_unsupervised_fits(monkeypatch):
    fits = []
    orig = UnsupervisedCatEncoder.fit

    def spy(self, X_fit, y=None, sample_weight=None):
        out = orig(self, X_fit, y=y, sample_weight=sample_weight)
        col = self.cols[0] if self.cols else None
        spec = self.vocabulary_.get(col, {}) if col is not None else {}
        fits.append(
            {
                "n": len(X_fit),
                "ids": tuple(spec.get("identities", ())),
                "mapping": dict(spec.get("mapping", {})),
                "method": self.method,
                "weight": None if sample_weight is None else np.asarray(sample_weight, dtype=float).copy(),
            }
        )
        return out

    monkeypatch.setattr(UnsupervisedCatEncoder, "fit", spy)
    return fits


def test_evaluate_and_gaussian_cv_fit_train_rows_only(monkeypatch):
    rng = np.random.default_rng(12)
    n = 90
    groups = np.repeat(np.arange(3), n // 3)
    cat = np.array(["a", "b"] * (n // 2), dtype=object)
    cat = cat.copy()
    cat[groups == 2] = "fold_only"
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    y = 0.3 * (X["cat"] == "a").astype(float) + X["noise"]
    fits = _spy_unsupervised_fits(monkeypatch)
    result = select_cefsplus(
        X,
        y,
        k="auto",
        cat_encoding="ordinal",
        auto_k_config=AutoKConfig(
            k_method="evaluate",
            strategy="group_cv",
            n_splits=3,
            min_k=1,
            max_k=2,
            selection_rule="best",
        ),
        groups=groups,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert result.selected_features
    holdout = ("str", "fold_only")
    fold_fits = [row for row in fits if row["n"] < n]
    assert fold_fits
    assert any(holdout not in row["ids"] for row in fold_fits)

    fits.clear()
    cv = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        cat_encoding="frequency",
        auto_k_config=AutoKConfig(
            k_method="gaussian_cv",
            strategy="group_cv",
            xfit_folds=3,
            min_k=1,
            max_k=2,
        ),
        groups=groups,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert cv.selected_features
    fold_fits = [row for row in fits if row["n"] < n]
    assert fold_fits
    assert any(holdout not in row["ids"] for row in fold_fits)


def test_time_holdout_path_map_is_train_only(monkeypatch):
    rng = np.random.default_rng(8)
    n = 40
    time = np.arange(n)
    cat = np.array(["a", "b"] * (n // 2), dtype=object)
    cat = cat.copy()
    cat[-10:] = "future"
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    y = X["noise"].to_numpy()
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        val_frac=0.25,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    fits = _spy_unsupervised_fits(monkeypatch)
    select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        cat_encoding="frequency",
        auto_k_config=cfg,
        time=time,
        subsample=None,
        verbose=False,
    )
    future = ("str", "future")
    assert fits
    assert all(future not in row["ids"] for row in fits)
    train_maps = [row["mapping"][("str", "a")] for row in fits if ("str", "a") in row["mapping"]]
    assert train_maps
    baseline_a = train_maps[0]

    mutated = X.copy()
    mutated.loc[time >= n * 0.75, "cat"] = "mutated_future"
    fits.clear()
    select_mrmr(
        mutated,
        y,
        k="auto",
        task="regression",
        cat_encoding="frequency",
        auto_k_config=cfg,
        time=time,
        subsample=None,
        verbose=False,
    )
    mutated_id = ("str", "mutated_future")
    assert fits
    assert all(mutated_id not in row["ids"] for row in fits)
    assert all(
        row["mapping"][("str", "a")] == pytest.approx(baseline_a)
        for row in fits
        if ("str", "a") in row["mapping"]
    )


def test_wrapper_nested_evaluate_and_inference_map_fixed():
    rng = np.random.default_rng(9)
    n = 60
    groups = np.repeat(np.arange(3), n // 3)
    cat = np.array(["a", "b", "c"] * (n // 3), dtype=object)
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    y = 2.0 * (X["cat"] == "a").astype(float) + 0.1 * X["noise"]
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selector = CEFSPlusSelector(
        k="auto", auto_k_config=cfg, cat_encoding="ordinal", subsample=None, verbose=False,
    )
    selector.fit(X, y, groups=groups)
    Xt = selector.transform(X)
    assert Xt.shape[1] == len(selector.selected_features_)
    future = X.copy()
    future["cat"] = "brand_new"
    Xf = selector.transform(future)
    assert Xf.shape == Xt.shape
    assert selector.categorical_encoder_ is not None


def test_multi_target_cefsplus_accepts_unsupervised_maps():
    X, y = _frame(n=48, seed=11)
    noise = np.random.default_rng(12).normal(size=len(X))
    Y = np.column_stack([np.asarray(y), 0.4 * X["x0"].to_numpy() + noise])
    selected = select_cefsplus(
        X, Y, k=1, cat_encoding="ordinal", subsample=None, verbose=False,
    )
    assert selected
    assert set(selected) <= set(X.columns)


def test_deferred_auto_k_wrapper_inference_is_numeric_and_fixed():
    rng = np.random.default_rng(14)
    n = 90
    X = pd.DataFrame(
        {"cat": np.array(["a"] * 60 + ["b"] * 20 + ["c"] * 10), "noise": rng.normal(size=n)}
    )
    yr = (X["cat"] == "c").astype(float) * 4 + 0.05 * X["noise"]
    yb = (X["cat"] == "c").astype(int).to_numpy()
    groups = np.arange(n) % 3
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    for cls, y in ((CEFSPlusSelector, yr), (CEFSPlusBinarySelector, yb)):
        selector = cls(
            k="auto", cat_encoding="ordinal", auto_k_config=cfg, subsample=None, verbose=False,
        )
        selector.fit(X, y, groups=groups)
        assert isinstance(selector.categorical_encoder_, UnsupervisedCatEncoder)
        trained = selector.fit_transform(X, y, groups=groups)
        inferred = selector.transform(X)
        np.testing.assert_allclose(
            np.asarray(trained, dtype=np.float64),
            np.asarray(inferred, dtype=np.float64),
        )
        unseen = X.copy()
        unseen["cat"] = "UNSEEN"
        uout = selector.transform(unseen)
        if "cat" in list(selector.selected_features_):
            col = list(selector.selected_features_).index("cat")
            np.testing.assert_array_equal(
                np.asarray(uout, dtype=np.float64)[:, col],
                np.full(len(X), -1.0),
            )

    late = CEFSPlusSelector(k="auto", cat_encoding="ordinal", subsample=None, verbose=False)
    late.fit(X, yr, groups=groups, auto_k_config=cfg)
    assert isinstance(late.categorical_encoder_, UnsupervisedCatEncoder)
    np.testing.assert_allclose(
        np.asarray(late.fit_transform(X, yr, groups=groups, auto_k_config=cfg), dtype=np.float64),
        np.asarray(late.transform(X), dtype=np.float64),
    )


def test_class_weight_does_not_enter_brier_or_logloss_evaluate_maps(monkeypatch):
    rng = np.random.default_rng(14)
    n = 90
    X = pd.DataFrame(
        {"cat": np.array(["a"] * 60 + ["b"] * 20 + ["c"] * 10), "noise": rng.normal(size=n)}
    )
    yb = (X["cat"] == "c").astype(int).to_numpy()
    groups = np.arange(n) % 3
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    expected = {
        ("str", "a"): 60 / 90,
        ("str", "b"): 20 / 90,
        ("str", "c"): 10 / 90,
    }
    fits = _spy_unsupervised_fits(monkeypatch)
    select_cefsplus_binary(
        X, yb, k=1, loss="brier", class_weight="balanced",
        cat_encoding="frequency", subsample=None, verbose=False,
    )
    assert fits
    path_map = fits[0]["mapping"]
    for ident, share in expected.items():
        assert path_map[ident] == pytest.approx(share)

    fits.clear()
    select_cefsplus_binary(
        X, yb, k="auto", loss="logloss", class_weight="balanced",
        cat_encoding="frequency", subsample=None, verbose=False,
        auto_k_config=cfg, groups=groups,
    )
    fold_maps = [row["mapping"] for row in fits if row["n"] < n]
    assert fold_maps
    for mapping in fold_maps:
        if ("str", "c") in mapping:
            assert mapping[("str", "c")] < 0.3


def test_gaussian_cv_composes_within_on_original_target(monkeypatch):
    import sift.selection.within as within_mod

    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(8), 12)
    X = pd.DataFrame(
        {
            "between": groups.astype(float),
            "signal": rng.normal(size=96),
            "cat": np.array(["a", "b", "c"] * 32),
        }
    )
    y = groups * 10 + X["signal"].to_numpy()
    cfg = AutoKConfig(
        k_method="gaussian_cv",
        strategy="group_cv",
        xfit_folds=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    orig = within_mod.fit_within_transform
    calls = []

    def spy(*args, **kwargs):
        calls.append(len(args[1]))
        return orig(*args, **kwargs)

    monkeypatch.setattr(within_mod, "fit_within_transform", spy)
    result = select_cefsplus(
        X, y, k="auto", groups=groups, within="groups", cat_encoding="ordinal",
        auto_k_config=cfg, subsample=None, verbose=False, return_result=True,
    )
    assert result.selected_features
    assert len(calls) >= 4
    assert 96 in calls
    assert any(size < 96 for size in calls)


def test_gaussian_cv_block_alignment_drops_constant_before_joint_block(monkeypatch):
    import sift.selection.auto_k_xfit as xfit

    rng = np.random.default_rng(2)
    n = 90
    X = pd.DataFrame(
        {
            "constant": np.ones(n),
            "cat": ["a", "b", "c"] * 30,
            "signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = X["signal"].to_numpy() + np.tile([0.0, 1.0, 2.0], 30)
    cfg = AutoKConfig(
        k_method="gaussian_cv",
        strategy="kfold",
        xfit_folds=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    orig = xfit.local_corr_panel
    seen = []

    def spy(Z, *args, **kwargs):
        seen.append(
            {
                "shape": tuple(np.asarray(Z).shape),
                "std": np.std(np.asarray(Z), axis=0),
                "block_members": kwargs.get("block_members"),
            }
        )
        return orig(Z, *args, **kwargs)

    monkeypatch.setattr(xfit, "local_corr_panel", spy)
    result = select_cefsplus(
        X, y, k="auto", cat_encoding="ordinal",
        feature_blocks={"joint": ["cat", "signal"]},
        auto_k_config=cfg, subsample=None, verbose=False, return_result=True,
    )
    assert result.selected_features
    fold_seen = [row for row in seen if row["shape"][0] < n]
    assert fold_seen
    for row in fold_seen:
        assert row["shape"][1] == 3
        assert np.all(row["std"] > 1e-12)
        members = [np.asarray(m).tolist() for m in (row["block_members"] or [])]
        assert [0, 1] in members
        assert [2] in members


def test_frequency_survives_large_finite_weights():
    X, y = _frame(n=90, seed=3)
    ones = select_mrmr(
        X, y, k=1, task="regression", cat_encoding="frequency",
        sample_weight=np.ones(len(X)), subsample=None, verbose=False,
    )
    huge = select_mrmr(
        X, y, k=1, task="regression", cat_encoding="frequency",
        sample_weight=np.full(len(X), 1e308), subsample=None, verbose=False,
    )
    assert ones == huge
    enc = UnsupervisedCatEncoder(["city"], method="frequency").fit(
        X, sample_weight=np.full(len(X), 1e308)
    )
    assert sum(enc.vocabulary_["city"]["mapping"].values()) == pytest.approx(1.0)


def test_fixed_k_and_nondeferred_fit_transform_matches_transform():
    rng = np.random.default_rng(1)
    n = 70
    frames = [
        pd.DataFrame(
            {
                "cat": np.array(["a"] * 40 + ["b"] * 20 + ["c"] * 10, dtype=object),
                "noise": rng.normal(size=n),
            }
        ),
        pd.DataFrame(
            {
                "cat": np.array([1] * 40 + [2] * 20 + [3] * 10),
                "noise": rng.normal(size=n),
            }
        ),
    ]
    elbow = AutoKConfig(k_method="elbow", min_k=1, max_k=2)
    for X in frames:
        y = X["cat"].isin(["c", 3]).astype(float)
        extra = {} if X["cat"].dtype == object else {"cat_features": ["cat"]}
        for cls in (CEFSPlusSelector, CEFSPlusBinarySelector):
            for mode in ("ordinal", "frequency"):
                for k, cfg in ((1, None), ("auto", elbow)):
                    selector = cls(
                        k=k,
                        cat_encoding=mode,
                        subsample=None,
                        verbose=False,
                        auto_k_config=cfg,
                        **extra,
                    )
                    training = selector.fit_transform(X, y)
                    inference = selector.transform(X)
                    assert "cat" in list(selector.selected_features_)
                    np.testing.assert_allclose(
                        np.asarray(training, dtype=np.float64),
                        np.asarray(inference, dtype=np.float64),
                    )
                    cat_col = list(selector.selected_features_).index("cat")
                    values = np.asarray(training, dtype=np.float64)[:, cat_col]
                    assert np.unique(values).size == 3
                    if mode == "ordinal":
                        np.testing.assert_array_equal(np.sort(np.unique(values)), [0.0, 1.0, 2.0])
                    else:
                        expected = UnsupervisedCatEncoder(["cat"], method="frequency").fit(X)
                        np.testing.assert_allclose(
                            values,
                            expected.transform(X)["cat"].to_numpy(),
                        )


def test_select_k_auto_encoding_sample_weight_does_not_override_target_cv():
    rng = np.random.default_rng(6)
    n = 60
    X = pd.DataFrame(
        {
            "cat": np.array(["a", "b", "c"] * 20, dtype=object),
            "noise": rng.normal(size=n),
        }
    )
    y = (X["cat"] == "a").astype(float) + 0.2 * X["noise"]
    groups = np.arange(n) % 4
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="group_cv",
        n_splits=3,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    ones = np.ones(n)
    common = dict(
        feature_path=["cat", "noise"],
        config=cfg,
        task="regression",
        groups=groups,
        cat_encoding="target_cv",
        sample_weight=ones,
    )
    default = select_k_auto(X, y, **common)
    explicit_none = select_k_auto(X, y, encoding_sample_weight=None, **common)
    override = select_k_auto(X, y, encoding_sample_weight=np.arange(1, n + 1, dtype=float), **common)
    assert default[0] == explicit_none[0] == override[0]
    assert default[1] == explicit_none[1] == override[1]
    np.testing.assert_allclose(
        default[2]["score_mean"].to_numpy(),
        explicit_none[2]["score_mean"].to_numpy(),
    )
    np.testing.assert_allclose(
        default[2]["score_mean"].to_numpy(),
        override[2]["score_mean"].to_numpy(),
    )
