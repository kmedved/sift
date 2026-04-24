import numpy as np
import pandas as pd
import pytest

from sift import CEFSPlusBinarySelector, select_cefsplus, select_cefsplus_binary
from sift._preprocess import LeaveOneOutLogitEncoder
from sift.api import _resolve_binary_weights, _validate_binary_target
import sift.selection.cefsplus_binary as cefsplus_binary_module
from sift.selection.cefsplus_binary import (
    compute_logistic_block_gram,
    intercept_only_prob,
    logistic_score_test_scores,
    logistic_score_test_scores_from_gram,
    weighted_standardize,
)
from sift.selection.auto_k import AutoKConfig
from sift.selection.result import FilterSelectionResult


def _classification_frame(seed=0, n=220, p=8):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    logits = 3.0 * X["f0"].to_numpy() + 0.5 * X["f1"].to_numpy()
    y = (logits + rng.normal(scale=0.8, size=n) > 0.0).astype(int)
    return X, y


def test_binary_first_step_matches_univariate_score_test():
    X, y = _classification_frame(seed=1)
    w = np.ones(len(y))
    Z, valid, _, _ = weighted_standardize(X.to_numpy(dtype=float), w)
    p0 = intercept_only_prob(y.astype(float), w)
    scores, _, _ = logistic_score_test_scores(Z, y.astype(float), w, p0)
    expected = X.columns[np.flatnonzero(valid)[int(np.argmax(scores))]]

    selected = select_cefsplus_binary(
        X,
        y,
        k=1,
        corr_prune=None,
        subsample=None,
        verbose=False,
    )

    assert selected == [expected]


def test_score_test_conditions_on_intercept_after_refit():
    rng = np.random.default_rng(101)
    Z = rng.normal(size=(80, 3))
    w = rng.uniform(0.5, 2.0, size=80)
    y = (rng.normal(size=80) + 1.2 * Z[:, 0] > 0.0).astype(float)
    p = 1.0 / (1.0 + np.exp(-(0.7 + 1.5 * Z[:, 0])))
    ridge = 1e-4

    scores, _, _ = logistic_score_test_scores(
        Z[:, 1:],
        y,
        w,
        p,
        Z_selected=Z[:, :1],
        ridge=ridge,
    )

    W = w * p * (1.0 - p)
    base = np.column_stack([np.ones(len(y)), Z[:, :1]])
    A = base.T @ (base * W[:, None])
    A[1:, 1:] += np.eye(1) * ridge
    B = base.T @ (Z[:, 1:] * W[:, None])
    solved = np.linalg.solve(A, B)
    cond = (
        np.sum(W[:, None] * Z[:, 1:] * Z[:, 1:], axis=0)
        + ridge
        - np.sum(B * solved, axis=0)
    )
    U = Z[:, 1:].T @ (w * (y - p))
    expected = 0.5 * U * U / cond

    np.testing.assert_allclose(scores, expected)


def test_score_test_adjusts_stale_selected_feature_score():
    rng = np.random.default_rng(102)
    Z = rng.normal(size=(90, 3))
    w = rng.uniform(0.5, 2.0, size=90)
    y = (rng.normal(size=90) + 1.1 * Z[:, 0] + 0.4 * Z[:, 1] > 0.0).astype(float)
    p = np.full(len(y), np.average(y, weights=w), dtype=float)
    ridge = 1e-4

    scores, _, _ = logistic_score_test_scores(
        Z[:, 1:],
        y,
        w,
        p,
        Z_selected=Z[:, :1],
        ridge=ridge,
        adjust_score=True,
        nuisance_penalty_gradient=np.zeros(2),
    )

    W = w * p * (1.0 - p)
    base = np.column_stack([np.ones(len(y)), Z[:, :1]])
    A = base.T @ (base * W[:, None])
    A[1:, 1:] += np.eye(1) * ridge
    B = base.T @ (Z[:, 1:] * W[:, None])
    solved = np.linalg.solve(A, B)
    cond = (
        np.sum(W[:, None] * Z[:, 1:] * Z[:, 1:], axis=0)
        + ridge
        - np.sum(B * solved, axis=0)
    )
    U = Z[:, 1:].T @ (w * (y - p))
    U_base = base.T @ (w * (y - p))
    U_eff = U - B.T @ np.linalg.solve(A, U_base)
    expected = 0.5 * U_eff * U_eff / cond

    np.testing.assert_allclose(scores, expected)


def test_block_gram_scores_match_row_level_scores():
    rng = np.random.default_rng(103)
    Z = rng.normal(size=(100, 5))
    w = rng.uniform(0.5, 1.5, size=100)
    y = (rng.normal(size=100) + 0.9 * Z[:, 0] > 0.0).astype(float)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 0.8 * Z[:, 0])))
    selected = [0, 2]
    candidates = np.array([1, 3, 4])
    penalty = np.array([0.0, 1e-4 * 0.3, 0.0])

    row_scores, row_failures, row_invalid = logistic_score_test_scores(
        Z[:, candidates],
        y,
        w,
        p,
        Z_selected=Z[:, selected],
        ridge=1e-4,
        adjust_score=True,
        nuisance_penalty_gradient=penalty,
    )
    block = compute_logistic_block_gram(Z, y, w, p)
    gram_scores, gram_failures, gram_invalid = logistic_score_test_scores_from_gram(
        block,
        candidates,
        selected=selected,
        ridge=1e-4,
        adjust_score=True,
        nuisance_penalty_gradient=penalty,
    )

    assert gram_failures == row_failures
    assert gram_invalid == row_invalid
    np.testing.assert_allclose(gram_scores, row_scores, rtol=1e-10, atol=1e-10)


def test_binary_logloss_selects_true_signal():
    X, y = _classification_frame(seed=2)

    selected = select_cefsplus_binary(X, y, k=3, subsample=None, verbose=False)

    assert len(selected) == 3
    assert selected[0] == "f0"


def test_binary_logloss_refit_every_block_mode_runs_with_adjusted_scores():
    X, y = _classification_frame(seed=22)

    result = select_cefsplus_binary(
        X,
        y,
        k=5,
        refit_every=3,
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert len(result.selected_features) == 5
    assert result.selector_metadata["refit_every"] == 3
    assert result.selected_features[0] == "f0"
    assert result.diagnostics_["n_gram_blocks"] == 2
    assert result.diagnostics_["n_logistic_refits"] == 1


def test_exact_path_adjusts_score_after_selected_feature_refit(monkeypatch):
    X, y = _classification_frame(seed=24)
    calls = []
    original = cefsplus_binary_module.logistic_score_test_scores

    def wrapped(*args, **kwargs):
        z_selected = kwargs.get("Z_selected")
        has_selected = z_selected is not None and z_selected.shape[1] > 0
        calls.append((has_selected, bool(kwargs.get("adjust_score", False))))
        return original(*args, **kwargs)

    monkeypatch.setattr(cefsplus_binary_module, "logistic_score_test_scores", wrapped)

    select_cefsplus_binary(
        X,
        y,
        k=3,
        corr_prune=None,
        refit_every=1,
        subsample=None,
        verbose=False,
    )

    selected_calls = [adjust for has_selected, adjust in calls if has_selected]
    assert selected_calls
    assert all(selected_calls)


def test_block_gram_uses_post_screen_candidate_matrix(monkeypatch):
    X, y = _classification_frame(seed=23, n=180, p=12)
    widths = []
    original = cefsplus_binary_module.compute_logistic_block_gram

    def wrapped(Z, y_arg, w_arg, p_arg):
        widths.append(Z.shape[1])
        return original(Z, y_arg, w_arg, p_arg)

    monkeypatch.setattr(cefsplus_binary_module, "compute_logistic_block_gram", wrapped)

    result = select_cefsplus_binary(
        X,
        y,
        k=4,
        top_m=5,
        corr_prune=None,
        refit_every=2,
        subsample=None,
        verbose=False,
        return_result=True,
    )

    candidate_count = len(result.diagnostics_["candidate_indices"])
    assert X.shape[1] > candidate_count
    assert candidate_count == 5
    assert widths
    assert all(width == candidate_count for width in widths)


def test_weighted_logloss_alias_matches_logloss_with_weights():
    X, y = _classification_frame(seed=3)
    w = np.linspace(0.5, 2.0, len(y))

    selected_logloss = select_cefsplus_binary(
        X,
        y,
        k=4,
        loss="logloss",
        sample_weight=w,
        subsample=None,
        verbose=False,
    )
    selected_alias = select_cefsplus_binary(
        X,
        y,
        k=4,
        loss="weighted_logloss",
        sample_weight=w,
        subsample=None,
        verbose=False,
    )

    assert selected_alias == selected_logloss


def test_weighted_logloss_requires_effective_weights():
    X, y = _classification_frame(seed=4)

    with pytest.raises(ValueError, match="requires sample_weight or class_weight"):
        select_cefsplus_binary(X, y, k=2, loss="weighted_logloss", verbose=False)


def test_sample_weights_can_change_binary_selection():
    rng = np.random.default_rng(5)
    n = 160
    y = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    rng.shuffle(y)
    rare = np.arange(30)
    X = pd.DataFrame(
        {
            "global_signal": y + rng.normal(scale=0.3, size=n),
            "rare_signal": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    X.loc[rare, "global_signal"] = rng.normal(size=len(rare))
    X.loc[rare, "rare_signal"] = y[rare] + rng.normal(scale=0.05, size=len(rare))
    w = np.ones(n)
    w[rare] = 200.0

    unweighted = select_cefsplus_binary(X, y, k=1, subsample=None, verbose=False)
    weighted = select_cefsplus_binary(X, y, k=1, sample_weight=w, subsample=None, verbose=False)

    assert unweighted == ["global_signal"]
    assert weighted == ["rare_signal"]


def test_class_weight_balanced_runs_and_marks_weighted_result():
    X, y = _classification_frame(seed=6)
    y[:160] = 0
    y[160:] = 1

    result = select_cefsplus_binary(
        X,
        y,
        k=2,
        class_weight="balanced",
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert isinstance(result, FilterSelectionResult)
    assert result.selector_metadata["weighted"] is True
    assert result.selector_metadata["class_weight"] == "balanced"


@pytest.mark.parametrize(
    ("bad_y", "match"),
    [
        (np.zeros(20, dtype=int), "exactly two"),
        (np.arange(20) % 3, "exactly two"),
        (np.r_[np.zeros(19), np.nan], "Missing"),
    ],
)
def test_binary_target_validation_errors(bad_y, match):
    X = np.random.default_rng(7).normal(size=(20, 3))

    with pytest.raises(ValueError, match=match):
        select_cefsplus_binary(X, bad_y, k=2, verbose=False)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_weight": np.r_[np.ones(19), -1.0]}, "negative"),
        ({"sample_weight": np.zeros(20)}, "at least one positive"),
        ({"ridge": 0.0}, "ridge"),
        ({"ridge": None}, "ridge"),
        ({"refit_every": 0}, "refit_every"),
        ({"refit_every": 1.5}, "refit_every"),
        ({"refit_every": "2"}, "refit_every"),
        ({"refit_every": True}, "refit_every"),
        ({"loo_smoothing": None}, "finite numeric"),
        ({"loo_smoothing": 0.0}, "loo_smoothing"),
        ({"loo_clip_min": 0.5, "loo_clip_max": 0.5}, "loo_clip"),
        ({"class_weight": np.ones(2)}, "class_weight"),
        ({"class_weight": {0: 1.0, 1: None}}, "class_weight values"),
        ({"cat_encoding": "bogus"}, "cat_encoding"),
        ({"corr_prune": 0.0}, "corr_prune"),
        ({"corr_prune": 1.5}, "corr_prune"),
        ({"corr_prune": "bad"}, "corr_prune"),
        ({"top_m": 0}, "top_m"),
        ({"subsample": 0}, "subsample"),
        ({"subsample": 1.5}, "subsample"),
    ],
)
def test_binary_parameter_and_weight_validation_errors(kwargs, match):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(20, 3))
    y = np.arange(20) % 2

    with pytest.raises(ValueError, match=match):
        select_cefsplus_binary(X, y, k=2, verbose=False, **kwargs)


def test_binary_selector_rejects_1d_feature_matrix_cleanly():
    y = np.arange(20) % 2

    with pytest.raises(ValueError, match="2D"):
        select_cefsplus_binary(np.arange(20), y, k=1, verbose=False)


def test_zero_effective_class_weight_errors():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 4))
    y = np.arange(40) % 2

    with pytest.raises(ValueError, match="positive effective weight"):
        select_cefsplus_binary(
            X,
            y,
            k=2,
            sample_weight=(y == 0).astype(float),
            verbose=False,
        )


def test_class_weight_dict_supports_non_numeric_labels():
    X, y_num = _classification_frame(seed=91)
    y = np.where(y_num == 1, "yes", "no")

    result = select_cefsplus_binary(
        X,
        y,
        k=2,
        class_weight={"no": 1.0, "yes": 3.0},
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert len(result.selected_features) == 2
    assert result.selector_metadata["weighted"] is True
    assert result.selector_metadata["class_weight_scope"] == "pre_subsample"


def test_class_weight_dict_rejects_encoded_keys_for_non_numeric_labels():
    X, y_num = _classification_frame(seed=910)
    y = np.where(y_num == 1, "yes", "no")

    with pytest.raises(ValueError, match="raw binary class labels"):
        select_cefsplus_binary(
            X,
            y,
            k=2,
            class_weight={0: 1.0, 1: 3.0},
            subsample=None,
            verbose=False,
        )


def test_class_weight_dict_supports_numeric_non_01_raw_labels():
    X, y_num = _classification_frame(seed=911)
    y = np.where(y_num == 1, 2, 1)

    result = select_cefsplus_binary(
        X,
        y,
        k=2,
        class_weight={1: 1.0, 2: 3.0},
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert len(result.selected_features) == 2
    assert result.selector_metadata["weighted"] is True


def test_class_weight_dict_rejects_encoded_keys_for_numeric_non_01_labels():
    X, y_num = _classification_frame(seed=912)
    y = np.where(y_num == 1, 2, 1)

    with pytest.raises(ValueError, match="raw binary class labels"):
        select_cefsplus_binary(
            X,
            y,
            k=2,
            class_weight={0: 10.0, 1: 1.0},
            subsample=None,
            verbose=False,
        )


def test_mixed_object_binary_labels_are_handled_without_numpy_sorting_error():
    X, y_num = _classification_frame(seed=92)
    y = np.empty(len(y_num), dtype=object)
    y[y_num == 1] = "yes"
    y[y_num == 0] = 0

    result = select_cefsplus_binary(
        X,
        y,
        k=2,
        class_weight={0: 1.0, "yes": 2.0},
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert len(result.selected_features) == 2
    assert result.selector_metadata["weighted"] is True


def test_mixed_object_multiclass_target_raises_clean_value_error():
    X = np.random.default_rng(93).normal(size=(30, 3))
    y = np.array([0, "yes", "no"] * 10, dtype=object)

    with pytest.raises(ValueError, match="exactly two"):
        select_cefsplus_binary(X, y, k=2, verbose=False)


def test_constant_and_duplicate_features_are_handled_deterministically():
    rng = np.random.default_rng(10)
    n = 120
    signal = rng.normal(size=n)
    y = (signal > 0.0).astype(int)
    X = pd.DataFrame(
        {
            "constant": 1.0,
            "signal_a": signal,
            "signal_b": signal.copy(),
            "noise": rng.normal(size=n),
        }
    )

    result = select_cefsplus_binary(
        X,
        y,
        k=3,
        top_m=4,
        corr_prune=0.99,
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert result.selected_features[0] == "signal_a"
    assert "signal_b" not in result.selected_features
    assert "constant" not in result.selected_features
    assert result.diagnostics_["dropped_features"]["constant"] == "constant_or_nonfinite"
    assert result.diagnostics_["dropped_features"]["signal_b"] == "corr_pruned"
    assert result.diagnostics_["valid_indices"] == [1, 2, 3]
    assert len(result.diagnostics_["univariate_scores"]) == X.shape[1]
    assert result.diagnostics_["n_constant_or_nonfinite"] == 1
    assert result.diagnostics_["n_corr_pruned"] == 1


def test_near_separation_does_not_crash():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"x": np.r_[np.linspace(-3, -1, 50), np.linspace(1, 3, 50)]})
    X["noise"] = rng.normal(size=len(X))
    y = (X["x"] > 0).astype(int).to_numpy()

    selected = select_cefsplus_binary(X, y, k=2, subsample=None, verbose=False)

    assert selected[0] == "x"


def test_dataframe_numpy_result_metadata_and_selector_transform():
    X, y = _classification_frame(seed=12)

    result = select_cefsplus_binary(
        X,
        y,
        k=3,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert isinstance(result, FilterSelectionResult)
    assert result.selector_metadata["selector"] == "cefsplus_binary"
    assert [X.columns[i] for i in result.selected_indices] == result.selected_features
    assert result.get_feature_ranking()["score"].notna().all()
    assert result.diagnostics_["subsample_row_idx"] is None

    np_result = select_cefsplus_binary(
        X.to_numpy(),
        y,
        k=2,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    assert np_result.selected_features[0] == "x0"
    assert np_result.selected_indices[0] == 0

    selector = CEFSPlusBinarySelector(k=3, verbose=False)
    X_out = selector.fit_transform(X, y)
    assert isinstance(X_out, pd.DataFrame)
    assert list(X_out.columns) == selector.selected_features_
    assert list(selector.transform(X).columns) == selector.selected_features_


def test_binary_selector_class_duplicate_columns_transform_selected_position():
    rng = np.random.default_rng(120)
    n = 140
    constant_dup = np.ones(n)
    signal_dup = rng.normal(size=n)
    y = (signal_dup > 0.0).astype(int)
    X = pd.DataFrame(
        np.column_stack([constant_dup, signal_dup, rng.normal(size=n)]),
        columns=["dup", "dup", "noise"],
    )

    selector = CEFSPlusBinarySelector(k=1, verbose=False).fit(X, y)
    transformed = selector.transform(X)

    assert selector.selected_indices_.tolist() == [1]
    assert selector.selected_features_ == ["dup"]
    np.testing.assert_allclose(transformed.iloc[:, 0].to_numpy(), signal_dup)


def test_binary_selector_class_ignores_irrelevant_split_metadata_for_fixed_k():
    X, y = _classification_frame(seed=121)

    selector = CEFSPlusBinarySelector(k=2, verbose=False)
    selector.fit(
        X,
        y,
        groups=np.arange(len(y)) % 5,
        time=np.arange(len(y)),
    )

    assert len(selector.selected_features_) == 2


def test_binary_selector_class_owns_return_result_fit_param():
    X, y = _classification_frame(seed=124)

    selector = CEFSPlusBinarySelector(k=2, verbose=False)
    selector.fit(X, y, return_result=False)

    assert len(selector.selected_features_) == 2
    assert selector.selected_indices_.shape == (2,)


@pytest.mark.parametrize("cat_encoding", ["target", "loo"])
def test_binary_selector_class_encodes_categoricals_with_binary_target_for_string_labels(
    cat_encoding,
):
    pytest.importorskip("category_encoders")
    rng = np.random.default_rng(125)
    n = 120
    team = np.where(np.arange(n) % 3 == 0, "a", "b")
    y_num = (team == "a").astype(int)
    y = np.where(y_num == 1, "yes", "no")
    X = pd.DataFrame({"team": team, "noise": rng.normal(size=n)})

    expected = select_cefsplus_binary(
        X,
        y,
        k=1,
        cat_features=["team"],
        cat_encoding=cat_encoding,
        allow_full_data_target_encoding=True,
        subsample=None,
        verbose=False,
    )
    selector = CEFSPlusBinarySelector(
        k=1,
        cat_features=["team"],
        cat_encoding=cat_encoding,
        verbose=False,
    ).fit(X, y)

    assert selector.selected_features_ == expected


def test_binary_selector_class_rejects_preprocessing_fit_time_overrides():
    X = pd.DataFrame(
        {
            "team": ["a", "a", "b", "b", "a", "b"],
            "x": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    selector = CEFSPlusBinarySelector(
        k=1,
        cat_features=["team"],
        cat_encoding="loo_logit",
        verbose=False,
    )
    with pytest.raises(ValueError, match="fit-time overrides"):
        selector.fit(X, y, loss="brier")
    with pytest.raises(ValueError, match="fit-time overrides"):
        selector.fit(X, y, class_weight="balanced")


def test_binary_selector_class_rejects_unsupported_auto_k_and_cache_cleanly():
    X, y = _classification_frame(seed=122)
    cfg = AutoKConfig(k_method="evaluate", strategy="time_holdout", max_k=3)

    with pytest.raises(ValueError, match="k='auto' is not supported"):
        CEFSPlusBinarySelector(k="auto", verbose=False).fit(
            X,
            y,
            time=np.arange(len(y)),
        )

    with pytest.raises(ValueError, match="does not support auto_k_config"):
        CEFSPlusBinarySelector(k=2, verbose=False).fit(
            X,
            y,
            auto_k_config=cfg,
        )

    with pytest.raises(ValueError, match="does not support prebuilt caches"):
        CEFSPlusBinarySelector(k=2, verbose=False).fit(X, y, cache=object())


def test_top_m_uses_binary_univariate_screen():
    X, y = _classification_frame(seed=13)
    w = np.ones(len(y))
    Z, valid, _, _ = weighted_standardize(X.to_numpy(dtype=float), w)
    scores, _, _ = logistic_score_test_scores(Z, y.astype(float), w, intercept_only_prob(y, w))
    top2_local = np.lexsort((np.flatnonzero(valid), -scores))[:2]
    expected_indices = np.flatnonzero(valid)[top2_local].tolist()

    result = select_cefsplus_binary(
        X,
        y,
        k=1,
        top_m=2,
        corr_prune=None,
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert result.diagnostics_["candidate_indices"] == expected_indices


def test_brier_mode_matches_existing_cefsplus_without_categoricals():
    X, y = _classification_frame(seed=14)

    expected = select_cefsplus(
        X,
        y.astype(float),
        k=4,
        top_m=None,
        corr_prune=0.95,
        subsample=None,
        verbose=False,
    )
    selected = select_cefsplus_binary(
        X,
        y,
        k=4,
        loss="brier",
        cat_encoding="none",
        top_m=None,
        corr_prune=0.95,
        subsample=None,
        verbose=False,
    )

    assert selected == expected


def test_brier_mode_return_result_preserves_delegate_indices_and_metadata():
    X, y = _classification_frame(seed=141)

    expected = select_cefsplus(
        X,
        y.astype(float),
        k=3,
        subsample=None,
        verbose=False,
        return_result=True,
    )
    result = select_cefsplus_binary(
        X,
        y,
        k=3,
        loss="brier",
        cat_encoding="none",
        subsample=None,
        verbose=False,
        return_result=True,
    )

    assert result.selected_features == expected.selected_features
    assert result.selected_indices == expected.selected_indices
    assert result.selector_metadata["selector"] == "cefsplus_binary"
    assert result.selector_metadata["delegate_selector"] == "cefsplus"


def test_weighted_brier_mode_matches_resolved_weighted_cefsplus():
    X, y = _classification_frame(seed=142)
    sample_weight = np.linspace(0.5, 3.0, len(y))
    y01, raw_y, _ = _validate_binary_target(y)
    weights, _ = _resolve_binary_weights(
        y01,
        raw_y,
        sample_weight=sample_weight,
        class_weight="balanced",
    )

    expected = select_cefsplus(
        X,
        y01.astype(float),
        k=3,
        sample_weight=weights,
        subsample=None,
        verbose=False,
    )
    selected = select_cefsplus_binary(
        X,
        y,
        k=3,
        loss="brier",
        sample_weight=sample_weight,
        class_weight="balanced",
        cat_encoding="none",
        subsample=None,
        verbose=False,
    )

    assert selected == expected


def test_brier_mode_matches_existing_cefsplus_with_loo_categoricals():
    pytest.importorskip("category_encoders")
    rng = np.random.default_rng(15)
    n = 100
    team = np.where(np.arange(n) % 2 == 0, "a", "b")
    y = (team == "a").astype(int)
    X = pd.DataFrame({"team": team, "noise": rng.normal(size=n)})

    expected = select_cefsplus(
        X,
        y.astype(float),
        k=1,
        cat_features=["team"],
        cat_encoding="loo",
        allow_full_data_target_encoding=True,
        subsample=None,
        verbose=False,
    )
    selected = select_cefsplus_binary(
        X,
        y,
        k=1,
        loss="brier",
        cat_features=["team"],
        cat_encoding="loo",
        allow_full_data_target_encoding=True,
        subsample=None,
        verbose=False,
    )

    assert selected == expected


def test_binary_selector_rejects_brier_loo_logit_categorical_ambiguity():
    X = pd.DataFrame(
        {
            "team": ["a", "a", "b", "b", "a", "b"],
            "x": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    selector = CEFSPlusBinarySelector(
        k=1,
        loss="brier",
        cat_features=["team"],
        cat_encoding="loo_logit",
        verbose=False,
    )
    with pytest.raises(ValueError, match="cat_encoding='loo'"):
        selector.fit(X, y)


def test_loo_logit_encoder_smooths_singletons_and_unknowns():
    X = pd.DataFrame({"team": ["a", "a", "b", "c"]})
    y = np.array([1, 0, 1, 0])
    encoder = LeaveOneOutLogitEncoder(["team"], smoothing=2.0)

    encoded = encoder.fit_transform(X, y)
    transformed = encoder.transform(pd.DataFrame({"team": ["unknown", "b"]}))

    assert np.isfinite(encoded["team"]).all()
    assert encoded.loc[2, "team"] == pytest.approx(0.0)
    assert encoded.loc[3, "team"] == pytest.approx(0.0)
    assert transformed.loc[0, "team"] == pytest.approx(0.0)


def test_loo_logit_encoder_honors_sample_weight():
    X = pd.DataFrame({"team": ["a", "a", "b", "b"]})
    y = np.array([1, 0, 0, 0])
    sample_weight = np.array([10.0, 1.0, 1.0, 1.0])

    unweighted = LeaveOneOutLogitEncoder(["team"], smoothing=1.0).fit_transform(X, y)
    weighted = LeaveOneOutLogitEncoder(["team"], smoothing=1.0).fit_transform(
        X,
        y,
        sample_weight=sample_weight,
    )

    assert weighted.loc[1, "team"] > unweighted.loc[1, "team"]


def test_loo_logit_encoder_handles_mixed_object_binary_labels_cleanly():
    X = pd.DataFrame({"team": ["a", "a", "b", "b"]})
    y = np.array([0, "yes", 0, "yes"], dtype=object)

    encoded = LeaveOneOutLogitEncoder(["team"], smoothing=1.0).fit_transform(X, y)

    assert np.isfinite(encoded["team"]).all()


def test_loo_logit_encoder_bad_numeric_params_raise_clean_value_error():
    with pytest.raises(ValueError, match="finite numeric"):
        LeaveOneOutLogitEncoder(["team"], smoothing=None)


def test_loo_logit_encoder_rejects_duplicate_encoded_column_names_cleanly():
    X = pd.DataFrame(
        [["a", "b"], ["a", "b"], ["c", "d"], ["c", "d"]],
        columns=["team", "team"],
    )
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="unique DataFrame column names"):
        LeaveOneOutLogitEncoder(["team"]).fit_transform(X, y)


def test_loo_logit_function_selector_requires_explicit_opt_in():
    X = pd.DataFrame({"team": ["a", "a", "b", "b"], "x": [0.0, 1.0, 0.0, 1.0]})
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="allow_full_data_target_encoding=True"):
        select_cefsplus_binary(
            X,
            y,
            k=1,
            cat_features=["team"],
            cat_encoding="loo_logit",
            verbose=False,
        )

    selected = select_cefsplus_binary(
        X,
        y,
        k=1,
        cat_features=["team"],
        cat_encoding="loo_logit",
        allow_full_data_target_encoding=True,
        verbose=False,
    )
    assert len(selected) == 1


def test_absent_binary_cat_features_are_noop_without_encoder_dependency():
    X, y = _classification_frame(seed=143)

    selected = select_cefsplus_binary(
        X,
        y,
        k=2,
        cat_features=["missing_category"],
        cat_encoding="loo",
        subsample=None,
        verbose=False,
    )

    assert len(selected) == 2
