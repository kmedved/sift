"""E4 one-hot encoding: atomic categories, cap/pooling, raw/encoded identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sift import (
    AutoKConfig,
    CEFSPlusSelector,
    as_result,
    select_cefsplus,
    select_cefsplus_binary,
    select_k_auto,
    select_mrmr,
)
from sift._preprocess import (
    ONEHOT_MAX_LEVELS_DEFAULT,
    OneHotBlockEncoder,
    _onehot_level_identity,
)


def _onehot_frame(n=80, seed=3):
    rng = np.random.default_rng(seed)
    city = np.array(["NY", "LA", "CHI", "NY", "LA"] * (n // 5 + 1))[:n]
    X = pd.DataFrame(
        {
            "city": city,
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
        }
    )
    y = (X["city"].eq("NY").astype(float) * 2.2 + 0.4 * X["x0"] + 0.15 * rng.normal(size=n))
    return X, y


def test_onehot_selects_raw_category_as_one_block():
    X, y = _onehot_frame()
    result = select_cefsplus(
        X,
        y,
        k=1,
        cat_encoding="onehot",
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert result.selected_features == ["city"]
    md = result.selector_metadata
    assert md["k"] == 1
    assert md["n_columns_selected"] == 1
    assert md["n_encoded_columns_selected"] >= 2
    assert md["onehot"] is True
    view = as_result(result, input_features=list(X.columns))
    assert view.features == ["city"]
    assert view.k == 1


def test_onehot_cap_pools_remainder_deterministically():
    rng = np.random.default_rng(1)
    n = 60
    levels = np.array([f"c{i}" for i in range(10)])
    city = rng.choice(levels, size=n, p=np.linspace(3, 1, 10) / np.linspace(3, 1, 10).sum())
    X = pd.DataFrame({"city": city, "x0": rng.normal(size=n)})
    y = rng.normal(size=n)
    encoder = OneHotBlockEncoder(["city"], max_levels=3)
    encoder.fit(X)
    spec = encoder.vocabulary_["city"]
    assert len(spec["retained"]) == 3
    assert spec["other_name"] == "city__other"
    assert spec["pooled"]
    Xt = encoder.transform(X)
    dummy_cols = [c for c in Xt.columns if str(c).startswith("city__")]
    assert dummy_cols[-1] == "city__other"
    assert Xt[dummy_cols].sum(axis=1).max() == pytest.approx(1.0)


def test_onehot_missing_unknown_and_zero_weight_cap():
    X = pd.DataFrame(
        {
            "city": ["NY", "LA", "NY", None, "CHI", "SEA"],
            "x0": np.arange(6, dtype=float),
        }
    )
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    encoder = OneHotBlockEncoder(["city"], max_levels=8)
    encoder.fit(X, sample_weight=w)
    retained = encoder.vocabulary_["city"]["retained"]
    dummy_names = encoder.vocabulary_["city"]["dummy_names"]
    assert ("str", "CHI") not in retained
    assert ("str", "SEA") not in retained
    assert ("missing",) in retained
    assert "city__missing" in dummy_names
    Xt = encoder.transform(pd.DataFrame({"city": ["BOS", None, "NY"], "x0": [0.0, 1.0, 2.0]}))
    assert "city__other" in Xt.columns or "city__missing" in Xt.columns
    if "city__other" in Xt.columns:
        assert float(Xt.loc[0, "city__other"]) == 1.0
    assert float(Xt.loc[1, "city__missing"]) == 1.0


def test_onehot_name_collision_raises():
    X = pd.DataFrame({"city": ["NY", "LA"], "city__NY": [0.0, 1.0]})
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="collides"):
        select_mrmr(
            X,
            y,
            k=1,
            task="regression",
            cat_encoding="onehot",
            subsample=None,
            verbose=False,
        )


def test_onehot_include_is_atomic_and_wrapper_transform_matches_encoded_width():
    X, y = _onehot_frame()
    result = select_cefsplus(
        X,
        y,
        k=1,
        include=["city"],
        cat_encoding="onehot",
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert "city" in result.selected_features
    md = result.selector_metadata
    assert md["k"] == md["n_blocks_selected"]
    selector = CEFSPlusSelector(
        k=1, cat_encoding="onehot", subsample=None, verbose=False
    )
    Xt = selector.fit_transform(X, y)
    names = selector.get_feature_names_out()
    assert Xt.shape[1] == len(names)
    assert set(selector.selected_features_) <= set(X.columns)
    assert len(selector.get_support()) == X.shape[1]
    assert selector.get_support().sum() == len(selector.selected_features_)
    assert all("__" in str(name) or name in X.columns for name in names)


def test_onehot_rejects_cache_within_and_knockoff():
    X, y = _onehot_frame()
    from sift import KnockoffSelector

    with pytest.raises(ValueError, match="onehot"):
        KnockoffSelector(cat_encoding="onehot").fit(X.assign(z=y), y)


def test_onehot_nested_evaluate_fits_vocab_on_train_folds():
    rng = np.random.default_rng(8)
    n = 96
    groups = np.repeat(np.arange(8), n // 8)
    city = np.array(["NY", "LA", "CHI"] * (n // 3 + 1), dtype=object)[:n]
    city = city.copy()
    city[groups == 7] = "ONLYFOLD"
    assert "ONLYFOLD" in set(city)
    X = pd.DataFrame({"city": city, "x0": rng.normal(size=n)})
    y = X["x0"] + rng.normal(size=n) * 0.1
    cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=4,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selector = CEFSPlusSelector(
        k="auto",
        auto_k_config=cfg,
        cat_encoding="onehot",
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=groups)
    names = list(selector.get_feature_names_out())
    assert selector.k_ in {1, 2}
    Xt = selector.transform(X)
    assert Xt.shape[1] == len(names)
    assert Xt.shape[1] >= len(selector.selected_features_)


def test_onehot_binary_and_default_cap():
    X, y = _onehot_frame()
    yb = (y > np.median(y)).astype(int)
    selected = select_cefsplus_binary(
        X, yb, k=1, cat_encoding="onehot", subsample=None, verbose=False
    )
    assert selected
    assert set(selected) <= set(X.columns)
    assert ONEHOT_MAX_LEVELS_DEFAULT == 32


def test_onehot_does_not_change_target_cv_or_numeric_defaults():
    X, y = _onehot_frame()
    numeric = X[["x0", "x1"]]
    assert select_cefsplus(numeric, y, k=1, subsample=None, verbose=False) == select_cefsplus(
        numeric, y, k=1, subsample=None, verbose=False, cat_encoding="none"
    )
    encoded = select_mrmr(
        X,
        (y > np.median(y)).astype(int),
        k=2,
        task="classification",
        cat_encoding="target_cv",
        subsample=None,
        verbose=False,
    )
    assert set(encoded) <= set(X.columns)


def test_onehot_preserves_distinct_category_identities():
    cases = [
        ["missing", "level_missing"],
        ["other", "level_other"],
        ["", "empty"],
        [1, "1"],
    ]
    for levels in cases:
        X = pd.DataFrame(
            {
                "cat": pd.Series(list(levels) * 20, dtype=object),
                "noise": np.arange(40, dtype=float),
            }
        )
        Xt = OneHotBlockEncoder(["cat"]).fit_transform(X)
        dummy_cols = [c for c in Xt.columns if str(c).startswith("cat__")]
        assert len(dummy_cols) == 2, levels
        assert not np.array_equal(Xt[dummy_cols[0]].to_numpy(), Xt[dummy_cols[1]].to_numpy()), levels
        y = np.array([0.0, 10.0] * 20)
        selected = select_cefsplus(X, y, k=1, cat_encoding="onehot", subsample=None, verbose=False)
        renamed = X.copy()
        mapping = {levels[0]: "left", levels[1]: "right"}
        renamed["cat"] = [mapping[v] for v in X["cat"]]
        selected_renamed = select_cefsplus(
            renamed, y, k=1, cat_encoding="onehot", subsample=None, verbose=False
        )
        assert selected == selected_renamed == ["cat"], levels


def test_onehot_keeps_integer_and_mixed_raw_labels():
    rng = np.random.default_rng(4)
    n = 80
    X = pd.DataFrame(
        {
            7: np.array(["a", "b"] * (n // 2)),
            8: rng.normal(size=n),
        }
    )
    y = (X[7].eq("a").astype(float) * 2.0) + 0.1 * X[8]
    result = select_cefsplus(
        X, y, k=1, cat_encoding="onehot", subsample=None, verbose=False, return_result=True
    )
    assert result.selected_features == [7]
    mixed = pd.DataFrame({"cat": ["a", "b"] * (n // 2), 1: rng.normal(size=n)})
    y2 = mixed["cat"].eq("a").astype(float) + 0.05 * mixed[1]
    out = select_cefsplus(
        mixed, y2, k=1, cat_encoding="onehot", subsample=None, verbose=False
    )
    assert out[0] in {"cat", 1}


def test_onehot_view_exposes_encoded_identity():
    X, y = _onehot_frame()
    result = select_cefsplus(
        X, y, k=1, cat_encoding="onehot", subsample=None, verbose=False, return_result=True
    )
    view = as_result(result, input_features=list(X.columns))
    assert view.encoded_features
    assert view.encoded_indices
    assert view.encoded_support_ is not None
    assert view.encoded_output is not None
    assert view.encoded_table is not None
    assert view.encoded_output["n_features"] == len(view.encoded_features)
    assert view.encoded_support_.sum() == len(view.encoded_indices)
    assert len(view.encoded_indices) >= len(view.features)
    assert "raw_feature" in view.encoded_table.columns
    assert "block_id" in view.encoded_table.columns
    assert len(view.encoded_table) == len(view.encoded_features)
    assert set(view.encoded_table["raw_feature"]) == set(X.columns)


def test_onehot_store_proxies_rejected():
    X, y = _onehot_frame()
    with pytest.raises(ValueError, match="store_proxies"):
        select_cefsplus(
            X,
            y,
            k=1,
            cat_encoding="onehot",
            store_proxies=True,
            return_result=True,
            subsample=None,
            verbose=False,
        )
    y_numeric = X["x0"] + 0.05 * np.asarray(y)
    with pytest.raises(ValueError, match="store_proxies"):
        select_cefsplus(
            X,
            y_numeric,
            k=2,
            cat_encoding="onehot",
            store_proxies=True,
            return_result=True,
            subsample=None,
            verbose=False,
        )


def test_onehot_nested_diagnostics_use_raw_block_path():
    rng = np.random.default_rng(9)
    n = 120
    cat = np.array(["a", "b", "c"] * (n // 3))
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    y = 3.0 * (X["cat"] == "a").astype(float) + 0.1 * X["noise"]
    groups = np.repeat(np.arange(3), n // 3)
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
        k="auto", auto_k_config=cfg, cat_encoding="onehot", subsample=None, verbose=False
    )
    selector.fit(X, y, groups=groups)
    diag = selector.nested_auto_k_diagnostics_
    rows = diag["folds"]
    assert isinstance(rows, pd.DataFrame)
    assert "path" in rows.columns
    k1 = rows.loc[rows["k"] == 1, "path"].iloc[0]
    assert tuple(k1) == ("cat",)
    k2 = rows.loc[rows["k"] == 2, "path"].iloc[0]
    assert tuple(k2) == ("cat", "noise")


def _spy_onehot_fits(monkeypatch):
    fits = []
    retained = []
    orig_fit = OneHotBlockEncoder.fit

    def spy(self, X_fit, y=None, sample_weight=None):
        out = orig_fit(self, X_fit, y=y, sample_weight=sample_weight)
        fits.append(len(X_fit))
        seen = {}
        for col, spec in getattr(self, "vocabulary_", {}).items():
            values = X_fit[col].to_numpy(dtype=object, copy=False) if col in X_fit.columns else []
            seen[col] = {
                "retained": tuple(spec["retained"]),
                "pooled": tuple(spec["pooled"]),
                "fit_ids": tuple(
                    dict.fromkeys(_onehot_level_identity(value) for value in values)
                ),
            }
        retained.append(seen)
        return out

    monkeypatch.setattr(OneHotBlockEncoder, "fit", spy)
    return fits, retained


def test_onehot_auto_routing_fits_fold_local_vocabulary(monkeypatch):
    rng = np.random.default_rng(11)
    n = 120
    time = np.arange(n)
    cat = np.array(["a", "b"] * (n // 2), dtype=object)
    cat = cat.copy()
    cat[-20:] = "holdout_only"
    assert "holdout_only" in set(cat)
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    y = X["noise"] + 0.2 * (X["cat"] == "a").astype(float)
    fits, retained = _spy_onehot_fits(monkeypatch)
    result = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        cat_encoding="onehot",
        onehot_max_levels=1,
        auto_k_config=AutoKConfig(
            k_method="auto",
            strategy="time_holdout",
            min_k=1,
            max_k=2,
            val_frac=0.2,
        ),
        time=time,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert fits
    fold_fits = [n_rows for n_rows in fits if n_rows < n]
    assert fold_fits
    holdout = ("str", "holdout_only")
    for n_rows, vocab in zip(fits, retained):
        if n_rows < n and "cat" in vocab:
            spec = vocab["cat"]
            known = set(spec["retained"]) | set(spec["pooled"])
            saw = holdout in spec["fit_ids"]
            assert (holdout in known) == saw
    routed = result.diagnostics_["auto_k"].get("auto_routing", {})
    assert routed.get("chosen") == "gaussian_cv"


def test_onehot_evaluate_and_gaussian_cv_are_fold_local(monkeypatch):
    rng = np.random.default_rng(12)
    n = 90
    groups = np.repeat(np.arange(3), n // 3)
    cat = np.array(["a", "b"] * (n // 2), dtype=object)
    cat = cat.copy()
    cat[groups == 2] = "fold_only"
    assert "fold_only" in set(cat)
    X = pd.DataFrame({"cat": cat, "noise": rng.normal(size=n)})
    y = 0.3 * (X["cat"] == "a").astype(float) + X["noise"]
    fits, retained = _spy_onehot_fits(monkeypatch)
    result = select_cefsplus(
        X,
        y,
        k="auto",
        cat_encoding="onehot",
        onehot_max_levels=1,
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
    fold_vocabs = [
        vocab["cat"]
        for n_rows, vocab in zip(fits, retained)
        if n_rows < n and "cat" in vocab
    ]
    assert fold_vocabs
    holdout = ("str", "fold_only")
    for spec in fold_vocabs:
        known = set(spec["retained"]) | set(spec["pooled"])
        saw = holdout in spec["fit_ids"]
        assert (holdout in known) == saw
    assert any(holdout not in spec["fit_ids"] for spec in fold_vocabs)

    fits.clear()
    retained.clear()
    cv = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        cat_encoding="onehot",
        onehot_max_levels=1,
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
    fold_vocabs = [
        vocab["cat"]
        for n_rows, vocab in zip(fits, retained)
        if n_rows < n and "cat" in vocab
    ]
    assert fold_vocabs
    for spec in fold_vocabs:
        known = set(spec["retained"]) | set(spec["pooled"])
        saw = holdout in spec["fit_ids"]
        assert (holdout in known) == saw
    assert any(holdout not in spec["fit_ids"] for spec in fold_vocabs)


def test_onehot_boruta_and_select_k_auto_dispatch():
    X, y = _onehot_frame()
    from sift import select_boruta

    with pytest.raises(ValueError, match="onehot") as exc:
        select_boruta(X, y, task="regression", cat_encoding="onehot")
    assert "allow_full_data_target_encoding" not in str(exc.value)
    assert "supervised" not in str(exc.value).lower()
    with pytest.raises(ValueError, match="filter-selector") as exc_opt:
        select_boruta(
            X,
            y,
            task="regression",
            cat_encoding="onehot",
            allow_full_data_target_encoding=True,
        )
    assert "allow_full_data_target_encoding" not in str(exc_opt.value)
    _k, names, _diag = select_k_auto(
        X,
        np.asarray(y),
        ["city", "x0"],
        AutoKConfig(
            k_method="evaluate",
            strategy="time_holdout",
            min_k=1,
            max_k=2,
            val_frac=0.3,
        ),
        time=np.arange(len(X)),
        cat_encoding="onehot",
    )
    assert names
    assert set(names) <= {"city", "x0"}


def test_onehot_compound_blocks_stay_atomic_under_evaluate():
    rng = np.random.default_rng(123)
    n = 200
    X = pd.DataFrame(
        {
            "cat": np.array(["a", "b"] * 100, dtype=object),
            "num": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    y = 3.0 * (X["cat"] == "a").astype(float) + 0.01 * rng.normal(size=n)
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=1,
        selection_rule="best",
    )
    blocks = {"bundle": ["cat", "num"]}
    common = dict(
        k="auto",
        auto_k_config=cfg,
        feature_blocks=blocks,
        cat_encoding="onehot",
        time=np.arange(n),
        subsample=None,
        verbose=False,
        return_result=True,
    )
    gauss = select_cefsplus(X, y, **common)
    assert gauss.selected_features == ["cat", "num"]
    assert gauss.selector_metadata["selected_blocks"] == ["bundle"]
    classic = select_mrmr(X, y, estimator="classic", task="regression", **common)
    assert classic.selected_features == ["cat", "num"]
    yb = (y > np.median(y)).astype(int)
    binary = select_cefsplus_binary(X, yb, **common)
    assert binary.selected_features == ["cat", "num"]

    nested_cfg = AutoKConfig(
        k_method="evaluate",
        auto_k_mode="nested",
        strategy="group_cv",
        n_splits=4,
        min_k=1,
        max_k=2,
        selection_rule="best",
    )
    selector = CEFSPlusSelector(
        k="auto",
        auto_k_config=nested_cfg,
        feature_blocks=blocks,
        cat_encoding="onehot",
        subsample=None,
        verbose=False,
    )
    selector.fit(X, y, groups=np.repeat(np.arange(4), n // 4))
    rows = selector.nested_auto_k_diagnostics_["folds"]
    k1 = tuple(rows.loc[rows["k"] == 1, "path"].iloc[0])
    assert set(k1) == {"cat", "num"}


def test_onehot_select_k_auto_base_features_use_full_encoded_width(monkeypatch):
    rng = np.random.default_rng(123)
    n = 120
    X = pd.DataFrame(
        {
            "cat": np.tile(np.array(["a", "b", "c"], dtype=object), 40),
            "num": rng.random(n),
        }
    )
    y = 3.0 * (X["cat"] == "a").astype(float) + X["num"] + 0.05 * rng.normal(size=n)
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=1,
        val_frac=0.3,
        selection_rule="best",
    )
    time = np.arange(n)
    captured: list[tuple[list, list]] = []
    from sift.selection import auto_k as auto_k_mod

    orig = auto_k_mod.evaluate_numeric_prefixes

    def spy(Xtr, Xva, *args, **kwargs):
        captured.append((list(Xtr.columns), list(kwargs["k_grid"])))
        return orig(Xtr, Xva, *args, **kwargs)

    monkeypatch.setattr(auto_k_mod, "evaluate_numeric_prefixes", spy)
    select_k_auto(
        X,
        np.asarray(y),
        ["num"],
        cfg,
        time=time,
        cat_encoding="onehot",
        base_features=["cat"],
    )
    assert captured
    cols, grid = captured[-1]
    assert len(cols) == 4
    assert grid == [4]
    captured.clear()
    select_k_auto(
        X,
        np.asarray(y),
        ["cat"],
        cfg,
        time=time,
        cat_encoding="onehot",
        base_features=["num"],
    )
    cols, grid = captured[-1]
    assert len(cols) == 4
    assert grid == [4]


def test_onehot_typed_raw_relevance_keeps_integer_and_string_keys():
    rng = np.random.default_rng(123)
    n = 200
    X = pd.DataFrame(
        {
            "cat": np.array(["a", "b"] * 100, dtype=object),
            1: rng.normal(size=n),
            "1": rng.normal(size=n),
        }
    )
    y = X[1] + 0.05 * rng.normal(size=n)
    result = select_cefsplus(
        X, y, k=1, cat_encoding="onehot", subsample=None, verbose=False, return_result=True
    )
    assert result.selected_features == [1]
    ranking = result.ranking_
    rel_int = float(ranking.loc[ranking["feature"].map(lambda v: v == 1 and not isinstance(v, str)), "relevance"].iloc[0])
    rel_str = float(ranking.loc[ranking["feature"].map(lambda v: v == "1"), "relevance"].iloc[0])
    assert rel_int != pytest.approx(rel_str)
    encoded = result.diagnostics_["encoded_ranking"]
    enc_int = float(
        encoded.loc[encoded["feature"].map(lambda v: v == 1 and not isinstance(v, str)), "relevance"].iloc[0]
    )
    assert rel_int == pytest.approx(enc_int)


def test_onehot_select_k_auto_positional_weights_match_keyword():
    rng = np.random.default_rng(13)
    n = 120
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = X["a"] + 0.1 * rng.normal(size=n)
    weights = np.linspace(0.2, 2.0, n)
    cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        min_k=1,
        max_k=2,
        val_frac=0.3,
        selection_rule="best",
    )
    time = np.arange(n)
    path = ["a", "b", "c"]
    k_pos, names_pos, diag_pos = select_k_auto(
        X, np.asarray(y), path, cfg, None, time, "regression", "none", None, weights
    )
    k_kw, names_kw, diag_kw = select_k_auto(
        X,
        np.asarray(y),
        path,
        cfg,
        groups=None,
        time=time,
        task="regression",
        cat_encoding="none",
        cat_features=None,
        sample_weight=weights,
    )
    assert k_pos == k_kw
    assert names_pos == names_kw
    np.testing.assert_allclose(
        np.asarray(diag_pos["score"], dtype=float),
        np.asarray(diag_kw["score"], dtype=float),
    )
