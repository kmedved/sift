import numpy as np
import pandas as pd

from sift import select_cefsplus, select_jmi, select_jmim, select_mrmr
from sift.selection.auto_k import AutoKConfig
from sift.selection.result import FilterSelectionResult


def _make_regression_data(n: int = 180, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] * 2.0 + 0.5 * X["f1"] + rng.normal(size=n) * 0.1
    return X, y


def test_filter_selectors_default_return_list():
    X, y = _make_regression_data()

    assert isinstance(select_mrmr(X, y, k=4, task="regression", verbose=False), list)
    assert isinstance(select_jmi(X, y, k=4, task="regression", verbose=False), list)
    assert isinstance(select_jmim(X, y, k=4, task="regression", verbose=False), list)
    assert isinstance(select_cefsplus(X, y, k=4, verbose=False), list)


def test_filter_selectors_return_result_has_indices_and_metadata_fixed_k():
    X, y = _make_regression_data()
    feature_names = list(X.columns)

    result_mrmr = select_mrmr(
        X, y, k=4, task="regression", estimator="classic", verbose=False, return_result=True
    )
    result_jmi = select_jmi(
        X, y, k=4, task="regression", estimator="r2", verbose=False, return_result=True
    )
    result_jmim = select_jmim(
        X, y, k=4, task="regression", estimator="r2", verbose=False, return_result=True
    )
    result_cefsplus = select_cefsplus(X, y, k=4, verbose=False, return_result=True)

    for result in (result_mrmr, result_jmi, result_jmim, result_cefsplus):
        assert isinstance(result, FilterSelectionResult)
        assert isinstance(result.selected_features, list)
        assert isinstance(result.selected_indices, list)
        assert len(result.selected_features) == len(result.selected_indices)
        assert all(0 <= i < X.shape[1] for i in result.selected_indices)
        assert [feature_names[i] for i in result.selected_indices] == result.selected_features
        assert "selector" in result.selector_metadata
        assert "auto_k" in result.selector_metadata
        assert result.selector_metadata["auto_k"] is False
        assert result.selector_metadata["k"] == len(result.selected_features)
        assert result.selector_metadata["k_requested"] == 4

        ranking = result.get_feature_ranking()
        assert list(ranking.columns) == [
            "feature",
            "rank",
            "selected",
            "selected_index",
            "relevance",
            "selector",
        ]
        assert ranking["selected"].all()
        assert ranking["rank"].tolist() == list(range(1, len(result.selected_features) + 1))


def test_gaussian_return_result_duplicate_names_keep_positional_indices():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["dup", "dup", "noise"])
    y = X.iloc[:, 0].to_numpy() + 0.01 * rng.normal(size=len(X))

    result = select_cefsplus(X, y, k=1, verbose=False, return_result=True)

    assert result.selected_features == ["dup"]
    assert result.selected_indices == [0]


def test_filter_selectors_auto_k_return_result():
    X, y = _make_regression_data(seed=1)
    feature_names = list(X.columns)
    eval_cfg = AutoKConfig(
        k_method="evaluate",
        strategy="time_holdout",
        max_k=6,
        min_k=1,
        val_frac=0.25,
    )
    elbow_cfg = AutoKConfig(k_method="elbow", max_k=6, min_k=1)

    result_mrmr = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="classic",
        auto_k_config=eval_cfg,
        time=np.arange(len(X)),
        verbose=False,
        return_result=True,
    )
    result_jmi = select_jmi(
        X,
        y,
        k="auto",
        task="regression",
        auto_k_config=eval_cfg,
        time=np.arange(len(X)),
        verbose=False,
        return_result=True,
    )
    result_jmim = select_jmim(
        X,
        y,
        k="auto",
        task="regression",
        auto_k_config=eval_cfg,
        time=np.arange(len(X)),
        verbose=False,
        return_result=True,
    )
    result_gaussian_mrmr = select_mrmr(
        X,
        y,
        k="auto",
        task="regression",
        estimator="gaussian",
        auto_k_config=elbow_cfg,
        verbose=False,
        return_result=True,
    )
    result_cefsplus = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=elbow_cfg,
        verbose=False,
        return_result=True,
    )

    for result in (result_mrmr, result_jmi, result_jmim):
        assert isinstance(result, FilterSelectionResult)
        assert result.selector_metadata["auto_k"] is True
        assert result.selector_metadata["auto_k_mode"] == "prefix_only"
        assert result.selector_metadata["k_method"] == "evaluate"
        assert 1 <= len(result.selected_features) <= 6
        assert result.selector_metadata["k"] == len(result.selected_features)
        assert result.selected_indices is None or len(result.selected_indices) == len(result.selected_features)
        if result.selected_indices is not None:
            assert [feature_names[i] for i in result.selected_indices] == result.selected_features
        assert result.selector_metadata["k_requested"] == "auto"
        assert result.selector_metadata["n_features"] == X.shape[1]

    for result in (result_gaussian_mrmr, result_cefsplus):
        assert isinstance(result, FilterSelectionResult)
        assert result.selector_metadata["auto_k"] is True
        assert result.selector_metadata["auto_k_mode"] == "prefix_only"
        assert result.selector_metadata["k_method"] == "elbow"
        assert 1 <= len(result.selected_features) <= 6
        assert result.selector_metadata["k"] == len(result.selected_features)
        assert result.selected_indices is None or len(result.selected_indices) == len(result.selected_features)
        if result.selected_indices is not None:
            assert [feature_names[i] for i in result.selected_indices] == result.selected_features
        assert result.selector_metadata["k_requested"] == "auto"
        assert result.selector_metadata["n_features"] == X.shape[1]

    assert result_cefsplus.diagnostics_["auto_k"]["method"] == "elbow"
    assert not result_cefsplus.diagnostics_["auto_k_diagnostics"].empty


def test_cefsplus_penalized_objective_return_result_diagnostics():
    X, y = _make_regression_data(seed=2)
    cfg = AutoKConfig(k_method="penalized_objective", min_k=1, max_k=6)

    result = select_cefsplus(
        X,
        y,
        k="auto",
        auto_k_config=cfg,
        verbose=False,
        return_result=True,
    )

    assert isinstance(result, FilterSelectionResult)
    assert 1 <= len(result.selected_features) <= 6
    assert result.selector_metadata["k_method"] == "penalized_objective"
    assert result.selector_metadata["objective_penalty"] == "bic"
    assert result.diagnostics_["auto_k"]["objective_penalty"] == "bic"
    assert result.diagnostics_["auto_k"]["objective_scale"] == "gaussian_2mi"
    assert result.diagnostics_["auto_k_diagnostics"]["penalized_score"].notna().all()
