"""Binary CEFS+ selection via logistic score-test updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sift._progress import ProgressCallback, report_progress


@dataclass
class BinaryCEFSPlusPath:
    selected_original: list[int]
    selected_features: list[str]
    path_scores: list[float]
    univariate_scores: np.ndarray
    valid_original: list[int]
    candidate_original: list[int]
    dropped_features: dict[str, str]
    numerical_failures: int
    invalid_conditional_information: int
    n_valid_features: int
    n_screened_features: int
    n_gram_blocks: int
    n_logistic_refits: int


@dataclass
class LogisticBlockGram:
    gram: np.ndarray
    score: np.ndarray


def _empty_path(
    *,
    univariate_scores: np.ndarray,
    valid_original: list[int] | np.ndarray,
    dropped_features: dict[str, str],
    numerical_failures: int = 0,
    invalid_conditional_information: int = 0,
    n_valid_features: int = 0,
    n_screened_features: int = 0,
    n_gram_blocks: int = 0,
    n_logistic_refits: int = 0,
) -> BinaryCEFSPlusPath:
    return BinaryCEFSPlusPath(
        selected_original=[],
        selected_features=[],
        path_scores=[],
        univariate_scores=univariate_scores,
        valid_original=[int(i) for i in valid_original],
        candidate_original=[],
        dropped_features=dropped_features,
        numerical_failures=numerical_failures,
        invalid_conditional_information=invalid_conditional_information,
        n_valid_features=n_valid_features,
        n_screened_features=n_screened_features,
        n_gram_blocks=n_gram_blocks,
        n_logistic_refits=n_logistic_refits,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def weighted_standardize(
    X: np.ndarray,
    w: np.ndarray,
    *,
    min_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-standardize columns, dropping constant ones.

    A column is kept when its weighted standard deviation exceeds ``min_std``,
    which defaults to exact constancy so units do not affect validity.
    """
    if not np.isfinite(min_std) or min_std < 0.0:
        raise ValueError("min_std must be finite and non-negative")
    w_sum = float(np.sum(w))
    means = (w @ X) / w_sum
    centered = X - means
    variances = (w @ (centered * centered)) / w_sum
    stds = np.sqrt(np.maximum(variances, 0.0))
    valid = np.isfinite(stds) & (stds > min_std)
    Z = centered[:, valid] / stds[valid]
    return Z.astype(np.float64, copy=False), valid, means, variances


def weighted_corr_matrix(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    w_sum = float(np.sum(w))
    R = (Z.T @ (Z * w[:, None])) / w_sum
    np.clip(R, -1.0, 1.0, out=R)
    return R


def _solve_spd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    jitter = 0.0
    last_error: Exception | None = None
    for _ in range(5):
        try:
            A_eff = A if jitter == 0.0 else A + np.eye(A.shape[0]) * jitter
            L = np.linalg.cholesky(A_eff)
            return np.linalg.solve(L.T, np.linalg.solve(L, B))
        except np.linalg.LinAlgError as exc:
            last_error = exc
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    if last_error is not None:
        raise last_error
    raise np.linalg.LinAlgError("Failed to solve SPD system")


def _penalized_neg_logloss(
    Xd: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    beta: np.ndarray,
    *,
    ridge: float,
) -> float:
    eta = Xd @ beta
    data_loss = np.sum(w * (np.logaddexp(0.0, eta) - y * eta))
    penalty = 0.5 * ridge * float(beta[1:] @ beta[1:])
    return float(data_loss + penalty)


def fit_logistic_ridge(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    ridge: float,
    max_iter: int = 50,
    tol: float = 1e-8,
    obj_rtol: float = 1e-10,
    beta_init: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted ridge-penalized logistic regression by damped Newton steps.

    ``beta_init`` warm-starts the solve (intercept first, then one coefficient
    per column of ``X``; a shorter vector is zero-padded, which is the natural
    start when one feature was appended to a previously fitted prefix). The
    iteration stops when the step is tiny (``tol``) or when a full Newton
    sweep no longer reduces the penalized objective by more than ``obj_rtol``
    relative to its magnitude, which avoids creeping through dozens of
    line-searched micro-steps on quasi-separable data.
    """
    n = len(y)
    Xd = np.ones((n, X.shape[1] + 1), dtype=np.float64)
    if X.shape[1]:
        Xd[:, 1:] = X

    beta = np.zeros(Xd.shape[1], dtype=np.float64)
    if beta_init is not None:
        init = np.asarray(beta_init, dtype=np.float64).ravel()
        if init.shape[0] > beta.shape[0] or not np.isfinite(init).all():
            init = None
        else:
            beta[: init.shape[0]] = init
    if beta_init is None or init is None:
        p0 = np.clip(float(np.sum(w * y) / np.sum(w)), 1e-6, 1.0 - 1e-6)
        beta[0] = np.log(p0 / (1.0 - p0))
    penalty = np.zeros(Xd.shape[1], dtype=np.float64)
    penalty[1:] = ridge

    old_obj = _penalized_neg_logloss(Xd, y, w, beta, ridge=ridge)
    for _ in range(max_iter):
        eta = Xd @ beta
        p = np.clip(sigmoid(eta), 1e-6, 1.0 - 1e-6)
        W = w * p * (1.0 - p)
        grad = Xd.T @ (w * (y - p)) - penalty * beta
        H = Xd.T @ (Xd * W[:, None])
        if X.shape[1]:
            H[1:, 1:] += np.eye(X.shape[1]) * ridge
        try:
            step = _solve_spd(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H + np.eye(H.shape[0]) * 1e-8, grad, rcond=None)[0]
        if not np.isfinite(step).all():
            raise FloatingPointError("logistic ridge Newton step was non-finite")

        accepted = False
        scaled_step = step
        new_obj = old_obj
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate = beta + scale * step
            if not np.isfinite(candidate).all():
                continue
            new_obj = _penalized_neg_logloss(Xd, y, w, candidate, ridge=ridge)
            if np.isfinite(new_obj) and new_obj <= old_obj + 1e-10:
                beta = candidate
                scaled_step = scale * step
                accepted = True
                break
        if not accepted:
            break

        decrease = old_obj - new_obj
        old_obj = new_obj
        if float(np.max(np.abs(scaled_step))) < tol:
            break
        if decrease <= obj_rtol * max(abs(new_obj), 1.0):
            break
    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    eta = np.full(X.shape[0], beta[0], dtype=np.float64)
    if X.shape[1]:
        eta += X @ beta[1:]
    return np.clip(sigmoid(eta), 1e-6, 1.0 - 1e-6)


def intercept_only_prob(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    p0 = np.clip(float(np.sum(w * y) / np.sum(w)), 1e-6, 1.0 - 1e-6)
    return np.full(len(y), p0, dtype=np.float64)


def compute_logistic_block_gram(
    Z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
) -> LogisticBlockGram:
    residual = y - p
    fisher_weight = w * p * (1.0 - p)
    n_features = Z.shape[1]
    gram = np.empty((n_features + 1, n_features + 1), dtype=np.float64)
    gram[0, 0] = float(np.sum(fisher_weight))
    if n_features:
        intercept_cross = Z.T @ fisher_weight
        gram[0, 1:] = intercept_cross
        gram[1:, 0] = intercept_cross
        gram[1:, 1:] = Z.T @ (Z * fisher_weight[:, None])

    score = np.empty(n_features + 1, dtype=np.float64)
    weighted_residual = w * residual
    score[0] = float(np.sum(weighted_residual))
    if n_features:
        score[1:] = Z.T @ weighted_residual
    return LogisticBlockGram(gram=gram, score=score)


def logistic_score_test_scores(
    Z_candidates: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    *,
    Z_selected: np.ndarray | None = None,
    ridge: float = 1e-4,
    eps: float = 1e-12,
    adjust_score: bool = False,
    nuisance_penalty_gradient: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    if Z_candidates.shape[1] == 0:
        return np.empty(0, dtype=np.float64), 0, 0

    residual = y - p
    W = w * p * (1.0 - p)
    U = Z_candidates.T @ (w * residual)
    I_diag = np.sum(W[:, None] * Z_candidates * Z_candidates, axis=0)
    cond = I_diag + ridge

    if Z_selected is None or Z_selected.shape[1] == 0:
        Z_base = np.ones((len(y), 1), dtype=np.float64)
        penalty_start = 1
    else:
        Z_base = np.column_stack([np.ones(len(y), dtype=np.float64), Z_selected])
        penalty_start = 1

    if Z_base.shape[1] > 0:
        A = Z_base.T @ (Z_base * W[:, None])
        if Z_base.shape[1] > penalty_start:
            A[penalty_start:, penalty_start:] += (
                np.eye(Z_base.shape[1] - penalty_start, dtype=np.float64) * ridge
            )
        B = Z_base.T @ (Z_candidates * W[:, None])
        try:
            solved = _solve_spd(A, B)
            cond -= np.sum(B * solved, axis=0)
            if adjust_score:
                U_base = Z_base.T @ (w * residual)
                if nuisance_penalty_gradient is not None:
                    penalty = np.asarray(nuisance_penalty_gradient, dtype=np.float64).reshape(-1)
                    if penalty.shape != U_base.shape:
                        raise ValueError(
                            "nuisance_penalty_gradient must match the selected-feature block"
                        )
                    U_base = U_base - penalty
                solved_score = _solve_spd(A, U_base)
                U = U - B.T @ solved_score
        except np.linalg.LinAlgError:
            scores = np.full(Z_candidates.shape[1], -np.inf, dtype=np.float64)
            return scores, Z_candidates.shape[1], 0

    scores = np.full(Z_candidates.shape[1], -np.inf, dtype=np.float64)
    valid = np.isfinite(cond) & (cond > eps)
    scores[valid] = 0.5 * U[valid] * U[valid] / cond[valid]
    invalid_conditional_information = int(np.sum(~valid))
    return scores, 0, invalid_conditional_information


def logistic_score_test_scores_from_gram(
    block: LogisticBlockGram,
    candidates: np.ndarray,
    *,
    selected: list[int] | np.ndarray | None = None,
    ridge: float = 1e-4,
    eps: float = 1e-12,
    adjust_score: bool = False,
    nuisance_penalty_gradient: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float64), 0, 0

    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_cols = candidates + 1
    selected_arr = np.asarray(selected if selected is not None else [], dtype=np.int64)
    base_cols = np.concatenate(
        [np.array([0], dtype=np.int64), selected_arr + 1],
    )

    gram = block.gram
    score_vec = block.score
    U = score_vec[candidate_cols].copy()
    cond = np.diag(gram)[candidate_cols] + ridge

    A = gram[np.ix_(base_cols, base_cols)].copy()
    if len(base_cols) > 1:
        A[1:, 1:] += np.eye(len(base_cols) - 1, dtype=np.float64) * ridge
    B = gram[np.ix_(base_cols, candidate_cols)]

    try:
        solved = _solve_spd(A, B)
        cond -= np.sum(B * solved, axis=0)
        if adjust_score:
            U_base = score_vec[base_cols].copy()
            if nuisance_penalty_gradient is not None:
                penalty = np.asarray(nuisance_penalty_gradient, dtype=np.float64).reshape(-1)
                if penalty.shape != U_base.shape:
                    raise ValueError(
                        "nuisance_penalty_gradient must match the selected-feature block"
                    )
                U_base -= penalty
            solved_score = _solve_spd(A, U_base)
            U -= B.T @ solved_score
    except np.linalg.LinAlgError:
        scores = np.full(len(candidates), -np.inf, dtype=np.float64)
        return scores, len(candidates), 0

    scores = np.full(len(candidates), -np.inf, dtype=np.float64)
    valid = np.isfinite(cond) & (cond > eps)
    scores[valid] = 0.5 * U[valid] * U[valid] / cond[valid]
    invalid_conditional_information = int(np.sum(~valid))
    return scores, 0, invalid_conditional_information


def _rank_desc(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    order = np.lexsort((indices, -scores))
    return indices[order]


def _corr_prune_candidates(
    Z: np.ndarray,
    w: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    threshold: float | None,
    tie_break_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, set[int]]:
    if threshold is None or len(candidates) <= 1:
        return candidates, set()
    if threshold <= 0.0:
        raise ValueError("corr_prune threshold must be positive when provided")
    tie_break = candidates if tie_break_indices is None else tie_break_indices
    ordered_local = np.lexsort((tie_break, -scores[candidates]))
    active = np.ones(len(ordered_local), dtype=bool)
    keep_local: list[int] = []
    w_sum = float(np.sum(w))
    block_size = 256
    for pos, local_idx in enumerate(ordered_local):
        if not active[pos]:
            continue
        keep_local.append(int(local_idx))
        later_positions = np.flatnonzero(active[pos + 1 :]) + pos + 1
        if later_positions.size:
            kept_col = int(candidates[local_idx])
            weighted_kept = w * Z[:, kept_col]
            for start in range(0, later_positions.size, block_size):
                positions = later_positions[start : start + block_size]
                later_local = ordered_local[positions]
                later_cols = candidates[later_local]
                correlations = Z[:, later_cols].T @ weighted_kept / w_sum
                active[positions] &= ~(np.abs(correlations) >= threshold)
    pruned_positions = ordered_local[~active]
    keep = candidates[np.asarray(keep_local, dtype=np.int64)]
    pruned = {int(candidates[i]) for i in pruned_positions}
    return keep, pruned


def select_binary_logistic_path(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    feature_names: list[str],
    *,
    k: int,
    top_m: int | None,
    corr_prune: float | None,
    ridge: float,
    refit_every: int,
    callback: ProgressCallback | None = None,
    include_idx: np.ndarray | None = None,
    candidate_idx: np.ndarray | None = None,
) -> BinaryCEFSPlusPath:
    Z, valid_mask, _, _ = weighted_standardize(X, w)
    valid_original = np.flatnonzero(valid_mask)
    dropped = {
        feature_names[i]: "constant_or_nonfinite"
        for i in np.flatnonzero(~valid_mask)
    }
    univariate_scores_original = np.full(len(feature_names), -np.inf, dtype=np.float64)
    invalid_information = 0
    include_arr = (
        np.empty(0, dtype=np.int64)
        if include_idx is None
        else np.asarray(include_idx, dtype=np.int64)
    )
    include_valid = []
    for orig in include_arr:
        matches = np.flatnonzero(valid_original == int(orig))
        if matches.size == 0:
            raise ValueError(
                f"include feature {feature_names[int(orig)]!r} is not a valid "
                "non-constant column for binary CEFS+"
            )
        include_valid.append(int(matches[0]))
    include_valid_arr = np.asarray(include_valid, dtype=np.int64)
    include_valid_set = set(include_valid)
    discovery_original = None if candidate_idx is None else set(int(i) for i in candidate_idx)
    if Z.shape[1] == 0:
        return _empty_path(
            univariate_scores=univariate_scores_original,
            valid_original=[],
            dropped_features=dropped,
        )

    p = intercept_only_prob(y, w)
    block_refit_count = 0
    use_block_gram = refit_every > 1
    block: LogisticBlockGram | None = None
    n_gram_blocks = 0
    n_logistic_refits = 0
    univariate_scores, failures, invalid = logistic_score_test_scores(
        Z,
        y,
        w,
        p,
        ridge=ridge,
    )
    invalid_information += invalid
    univariate_scores_original[valid_original] = univariate_scores
    finite = np.flatnonzero(np.isfinite(univariate_scores))
    if discovery_original is not None:
        finite = np.array(
            [
                int(i)
                for i in finite
                if int(valid_original[int(i)]) in discovery_original
                and int(i) not in include_valid_set
            ],
            dtype=np.int64,
        )
    else:
        finite = np.array(
            [int(i) for i in finite if int(i) not in include_valid_set],
            dtype=np.int64,
        )
    if candidate_idx is not None and finite.size == 0:
        raise ValueError(
            "candidates contains no usable discovery columns after "
            "validity/score screening"
        )
    if finite.size == 0 and include_valid_arr.size == 0:
        return _empty_path(
            univariate_scores=univariate_scores_original,
            valid_original=valid_original,
            dropped_features=dropped,
            numerical_failures=failures,
            invalid_conditional_information=invalid_information,
            n_valid_features=Z.shape[1],
            n_screened_features=0,
            n_gram_blocks=n_gram_blocks,
            n_logistic_refits=n_logistic_refits,
        )

    if top_m is None:
        top_m_eff = len(finite)
    else:
        top_m_eff = max(int(top_m), int(k))
        top_m_eff = min(top_m_eff, len(finite))
    ranked = _rank_desc(univariate_scores[finite], finite) if finite.size else np.empty(0, dtype=np.int64)
    screened = ranked[:top_m_eff]
    for local_idx in ranked[top_m_eff:]:
        dropped[feature_names[int(valid_original[local_idx])]] = "outside_top_m"

    candidates, pruned = _corr_prune_candidates(
        Z,
        w,
        screened,
        univariate_scores,
        corr_prune,
        tie_break_indices=valid_original[screened] if screened.size else screened,
    )
    for local_idx in pruned:
        dropped[feature_names[int(valid_original[local_idx])]] = "corr_pruned"

    candidate_valid = np.asarray(candidates, dtype=np.int64)
    if include_valid_arr.size:
        protect = [int(i) for i in include_valid_arr if int(i) not in set(int(x) for x in candidate_valid)]
        if protect:
            candidate_valid = np.concatenate(
                [include_valid_arr, candidate_valid]
            ) if candidate_valid.size else include_valid_arr
        else:
            # Keep include first in the working matrix.
            rest = np.array([int(i) for i in candidate_valid if int(i) not in include_valid_set], dtype=np.int64)
            candidate_valid = np.concatenate([include_valid_arr, rest]) if rest.size else include_valid_arr
    Z_work = np.ascontiguousarray(Z[:, candidate_valid], dtype=np.float64)
    work_to_original = valid_original[candidate_valid]
    n_forced = int(include_valid_arr.size)

    selected: list[int] = []
    selected_mask = np.zeros(Z_work.shape[1], dtype=bool)
    path_scores: list[float] = []
    beta: np.ndarray | None = None
    if n_forced:
        selected = list(range(n_forced))
        selected_mask[:n_forced] = True
        try:
            beta = fit_logistic_ridge(
                Z_work[:, selected], y, w, ridge=ridge
            )
            p = predict_logistic(Z_work[:, selected], beta)
            n_logistic_refits += 1
            block_refit_count = n_forced
            if use_block_gram and Z_work.shape[1]:
                block = compute_logistic_block_gram(Z_work, y, w, p)
                n_gram_blocks += 1
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError(
                "include features could not be fit in the binary logistic path; "
                "they cannot initialize exact conditioning"
            ) from exc
    elif use_block_gram and Z_work.shape[1]:
        block = compute_logistic_block_gram(Z_work, y, w, p)
        n_gram_blocks += 1

    target_count = min(int(k) + n_forced, Z_work.shape[1])
    while len(selected) < target_count:
        if selected and len(selected) - block_refit_count >= refit_every:
            try:
                # Warm-start from the previous refit (new coefficients start at 0).
                beta = fit_logistic_ridge(
                    Z_work[:, selected], y, w, ridge=ridge, beta_init=beta
                )
                p = predict_logistic(Z_work[:, selected], beta)
                block_refit_count = len(selected)
                n_logistic_refits += 1
                if use_block_gram:
                    block = compute_logistic_block_gram(Z_work, y, w, p)
                    n_gram_blocks += 1
            except (np.linalg.LinAlgError, FloatingPointError):
                failures += 1
                break

        remaining = np.flatnonzero(~selected_mask)
        if remaining.size == 0:
            break
        adjust_score = bool(selected)
        nuisance_penalty_gradient = None
        if adjust_score:
            nuisance_penalty_gradient = np.zeros(len(selected) + 1, dtype=np.float64)
            if beta is not None and block_refit_count > 0:
                n_penalized = min(block_refit_count, len(selected), len(beta) - 1)
                nuisance_penalty_gradient[1 : 1 + n_penalized] = (
                    ridge * beta[1 : 1 + n_penalized]
                )
        if use_block_gram:
            if block is None:
                failures += 1
                break
            scores, score_failures, invalid = logistic_score_test_scores_from_gram(
                block,
                remaining,
                selected=selected,
                ridge=ridge,
                adjust_score=adjust_score,
                nuisance_penalty_gradient=nuisance_penalty_gradient,
            )
        else:
            Z_selected = Z_work[:, selected] if selected else None
            scores, score_failures, invalid = logistic_score_test_scores(
                Z_work[:, remaining],
                y,
                w,
                p,
                Z_selected=Z_selected,
                ridge=ridge,
                adjust_score=adjust_score,
                nuisance_penalty_gradient=nuisance_penalty_gradient,
            )
        failures += score_failures
        invalid_information += invalid
        if not np.isfinite(scores).any():
            for work_idx in remaining:
                dropped[feature_names[int(work_to_original[work_idx])]] = "nonpositive_fisher"
            break
        best_pos = int(np.lexsort((work_to_original[remaining], -scores))[0])
        best_local = int(remaining[best_pos])
        selected.append(best_local)
        selected_mask[best_local] = True
        path_scores.append(float(scores[best_pos]))
        if callback is not None:
            discovery_step = len(selected) - n_forced
            report_progress(
                callback,
                discovery_step,
                int(k),
                stage="path",
                selector="cefsplus_binary",
            )

    discovered = selected[n_forced:]
    selected_original = [int(work_to_original[i]) for i in discovered]
    selected_features = [feature_names[i] for i in selected_original]
    candidate_original = [int(i) for i in work_to_original[n_forced:]]
    return BinaryCEFSPlusPath(
        selected_original=selected_original,
        selected_features=selected_features,
        path_scores=path_scores,
        univariate_scores=univariate_scores_original,
        valid_original=valid_original.astype(int).tolist(),
        candidate_original=candidate_original,
        dropped_features=dropped,
        numerical_failures=failures,
        invalid_conditional_information=invalid_information,
        n_valid_features=Z.shape[1],
        n_screened_features=len(candidate_original),
        n_gram_blocks=n_gram_blocks,
        n_logistic_refits=n_logistic_refits,
    )


def validate_corr_prune(corr_prune: float | None) -> float | None:
    if corr_prune is None:
        return None
    if isinstance(corr_prune, (bool, np.bool_)):
        raise ValueError("corr_prune must be None or a finite float in (0, 1]")
    try:
        value = float(corr_prune)
    except (TypeError, ValueError) as exc:
        raise ValueError("corr_prune must be None or a finite float in (0, 1]") from exc
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("corr_prune must be None or a finite float in (0, 1]")
    return value


def make_diagnostics(path: BinaryCEFSPlusPath) -> dict[str, Any]:
    return {
        "path_scores": list(path.path_scores),
        "candidate_indices": list(path.candidate_original),
        "valid_indices": list(path.valid_original),
        "univariate_scores": path.univariate_scores.tolist(),
        "dropped_features": dict(path.dropped_features),
        "numerical_failures": int(path.numerical_failures),
        "invalid_conditional_information": int(path.invalid_conditional_information),
        "n_valid_features": int(path.n_valid_features),
        "n_screened_features": int(path.n_screened_features),
        "n_gram_blocks": int(path.n_gram_blocks),
        "n_logistic_refits": int(path.n_logistic_refits),
        "n_constant_or_nonfinite": sum(
            reason == "constant_or_nonfinite" for reason in path.dropped_features.values()
        ),
        "n_corr_pruned": sum(reason == "corr_pruned" for reason in path.dropped_features.values()),
        "n_outside_top_m": sum(
            reason == "outside_top_m" for reason in path.dropped_features.values()
        ),
    }
