import numpy as np

from sift.estimators import joint_mi as jmi
from sift.selection.loops import jmi_select


def _legacy_r2_jmi_select(
    X,
    y,
    k,
    relevance,
    *,
    aggregation,
    top_m,
    sample_weight,
):
    valid_mask = relevance > 0
    valid_idx = np.where(valid_mask)[0]
    X_valid = X[:, valid_idx]
    rel_valid = relevance[valid_idx]

    if top_m is not None and top_m < len(valid_idx):
        top_local = np.argpartition(rel_valid, -top_m)[-top_m:]
        X_cand = X_valid[:, top_local]
        rel_cand = rel_valid[top_local]
        idx_map = valid_idx[top_local]
    else:
        X_cand = X_valid
        rel_cand = rel_valid
        idx_map = valid_idx

    m = X_cand.shape[1]
    k = min(k, m)
    scores = np.zeros(m, dtype=np.float64)
    if aggregation == "min":
        scores.fill(np.inf)

    is_selected = np.zeros(m, dtype=bool)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(np.argmax(rel_cand))
    is_selected[selected[0]] = True
    count = 1

    for t in range(1, k):
        last = int(selected[t - 1])
        cand_indices = np.where(~is_selected)[0]
        if cand_indices.size == 0:
            break

        mi_values = jmi.r2_joint_mi_indexed(
            X_cand,
            cand_indices.astype(np.int64, copy=False),
            X_cand[:, last],
            y,
            sample_weight,
        )

        for i, idx in enumerate(cand_indices):
            if aggregation == "sum":
                scores[idx] += mi_values[i]
            else:
                scores[idx] = min(scores[idx], mi_values[i])

        best_score = -np.inf
        best_idx = -1
        for idx in cand_indices:
            score = scores[idx] if np.isfinite(scores[idx]) else rel_cand[idx]
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        selected[t] = best_idx
        is_selected[best_idx] = True
        count += 1

    return idx_map[selected[:count]]


def _r2_case(weighted):
    rng = np.random.default_rng(20260420)
    n, p = 180, 18
    X = rng.normal(size=(n, p)).astype(np.float64)
    y = (
        1.8 * X[:, 0]
        - 1.2 * X[:, 3]
        + 0.8 * X[:, 5]
        + 0.35 * X[:, 7]
        + rng.normal(scale=0.2, size=n)
    ).astype(np.float64)
    relevance = np.abs(X.T @ y) / n + np.linspace(1e-5, 2e-5, p)
    if weighted:
        w = rng.uniform(0.25, 2.5, size=n).astype(np.float64)
    else:
        w = np.ones(n, dtype=np.float64)
    return X, y, relevance, w


def test_r2_precomputed_scores_match_indexed_weighted_and_unweighted():
    for weighted in (False, True):
        X, y, _, w = _r2_case(weighted)
        cand_idx = np.array([0, 2, 4, 8, 11, 15], dtype=np.int64)
        selected_idx = 3

        expected = jmi.r2_joint_mi_indexed(X, cand_idx, X[:, selected_idx], y, w)
        Z, r_y, w_state, w_sum = jmi._prepare_r2_joint_mi_state(X, y, w)
        actual = jmi._r2_joint_mi_indexed_from_state(
            Z,
            r_y,
            cand_idx,
            selected_idx,
            w_state,
            w_sum,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_r2_jmi_select_matches_legacy_indexed_path_weighted_and_unweighted():
    for weighted in (False, True):
        X, y, relevance, w = _r2_case(weighted)
        for aggregation in ("sum", "min"):
            expected = _legacy_r2_jmi_select(
                X,
                y,
                7,
                relevance,
                aggregation=aggregation,
                top_m=14,
                sample_weight=w,
            )
            actual = jmi_select(
                X,
                y,
                7,
                relevance,
                mi_estimator="r2",
                aggregation=aggregation,
                top_m=14,
                sample_weight=w,
            )

            np.testing.assert_array_equal(actual, expected)


def test_binned_jmi_select_reuses_prebinned_selected_columns(monkeypatch):
    rng = np.random.default_rng(12345)
    n, p = 96, 11
    X = rng.normal(size=(n, p)).astype(np.float64)
    y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n)
    relevance = np.linspace(1.0, 0.2, p, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)

    call_count = 0
    original_quantile_bin = jmi._quantile_bin

    def wrapped_quantile_bin(x, n_bins):
        nonlocal call_count
        call_count += 1
        return original_quantile_bin(x, n_bins)

    monkeypatch.setattr(jmi, "_quantile_bin", wrapped_quantile_bin)

    selected = jmi_select(
        X,
        y,
        5,
        relevance,
        mi_estimator="binned",
        aggregation="sum",
        y_kind="continuous",
        sample_weight=w,
    )

    assert selected.size == 5
    assert call_count == p + 1
