import pandas as pd
import numpy as np
from tqdm import tqdm

# Floor for numerical stability in correlation-based redundancy
# Using 1e-6 instead of 0.001 to avoid inflating scores
FLOOR = 1e-6


def groupstats2fstat(avg, var, n):
    """Compute F-statistic of some variables across groups

    Compute F-statistic of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.

    Parameters
    ----------
    avg: pandas.DataFrame of shape (n_groups, n_variables)
        Simple average of variables within groups. Each row is a group, each column is a variable.

    var: pandas.DataFrame of shape (n_groups, n_variables)
        Variance of variables within groups. Each row is a group, each column is a variable.

    n: pandas.DataFrame of shape (n_groups, n_variables)
        Count of instances for whom variable is not null. Each row is a group, each column is a variable.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic of each variable, based on group statistics.

    Reference
    ---------
    https://en.wikipedia.org/wiki/F-test
    """
    avg_global = (avg * n).sum() / n.sum()  # global average of each variable
    numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
    denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
    f = numerator / denominator
    return f.fillna(0.0)


def mrmr_base(K, relevance_func, redundancy_func,
              relevance_args=None, redundancy_args=None,
              denominator_func=np.mean, only_same_domain=False,
              return_scores=False, show_progress=True):
    """General function for mRMR algorithm.

    Uses O(P) memory via incremental redundancy accumulation instead of O(P²) matrix.

    Parameters
    ----------
    K: int
        Maximum number of features to select. The length of the output is *at most* equal to K

    relevance_func: callable
        Function for computing Relevance.
        It must return a pandas.Series containing the relevance (a number between 0 and +Inf)
        for each feature. The index of the Series must consist of feature names.

    redundancy_func: callable
        Function for computing Redundancy.
        It must return a pandas.Series containing the redundancy (a number between -1 and 1,
        but note that negative numbers will be taken in absolute value) of some features (called features)
        with respect to a variable (called target_column).
        It must have *at least* two parameters: "target_column" and "features".
        The index of the Series must consist of feature names.

    relevance_args: dict (optional, default=None)
        Optional arguments for relevance_func.

    redundancy_args: dict (optional, default=None)
        Optional arguments for redundancy_func.

    denominator_func: callable (optional, default=numpy.mean)
        Synthesis function to apply to the denominator of MRMR score.
        It must take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary redundancy coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.

    Notes
    -----
    When return_scores=True, returns (selected_features, relevance, redundancy).
    The redundancy matrix is recomputed for the selected features only to keep
    the selection path O(P) in memory.
    """

    relevance_args = {} if relevance_args is None else relevance_args
    redundancy_args = {} if redundancy_args is None else redundancy_args

    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    K = min(K, len(features))

    # Use numpy arrays for O(P) memory instead of O(P²) DataFrame
    rel_values = relevance.values.astype(np.float64)
    n_features = len(features)

    # Incremental accumulators - O(P) memory
    # For mean: store sum of redundancies, divide by count
    # For max: store running max
    if denominator_func in (np.mean, np.nanmean):
        denom_mode = "mean"
        redundancy_sum = np.zeros(n_features, dtype=np.float64)
        redundancy_cnt = np.zeros(n_features, dtype=np.int32)
    elif denominator_func in (np.max, np.nanmax):
        denom_mode = "max"
        redundancy_max = np.zeros(n_features, dtype=np.float64)
        redundancy_cnt = np.zeros(n_features, dtype=np.int32)
    else:
        raise NotImplementedError(
            "mrmr_base supports denominator_func in {np.mean, np.max} for O(P) memory mode. "
            "Use a custom selector or store redundancy explicitly for other callables."
        )

    is_candidate = np.ones(n_features, dtype=bool)
    selected_features = []

    # Pre-compute domains if needed
    if only_same_domain:
        domains = np.array([f.split('_')[0] for f in features])

    for i in tqdm(range(K), disable=not show_progress):

        # Compute scores
        if i == 0:
            scores = rel_values.copy()
        else:
            if denom_mode == "mean":
                denom = np.where(redundancy_cnt > 0, redundancy_sum / redundancy_cnt, 1.0)
            else:
                denom = np.where(redundancy_cnt > 0, redundancy_max, 1.0)
            denom = np.maximum(denom, FLOOR)
            scores = rel_values / denom

        # Mask out selected features
        scores = np.where(is_candidate, scores, -np.inf)

        # Select best
        best_idx = np.argmax(scores)
        best_feature = features[best_idx]
        selected_features.append(best_feature)
        is_candidate[best_idx] = False

        if len(selected_features) >= K:
            break

        # Update redundancy accumulators
        candidate_indices = np.where(is_candidate)[0]
        if len(candidate_indices) == 0:
            break

        candidate_names = [features[j] for j in candidate_indices]

        if only_same_domain:
            # Only update candidates in same domain
            last_domain = domains[best_idx]
            same_domain_mask = (domains == last_domain) & is_candidate
            update_indices = np.where(same_domain_mask)[0]
            if len(update_indices) > 0:
                update_names = [features[j] for j in update_indices]
                new_red = redundancy_func(
                    target_column=best_feature,
                    features=update_names,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                new_vals = new_red.reindex(update_names).fillna(FLOOR).to_numpy(dtype=np.float64)
                if denom_mode == "mean":
                    redundancy_sum[update_indices] += new_vals
                    redundancy_cnt[update_indices] += 1
                else:
                    redundancy_max[update_indices] = np.maximum(redundancy_max[update_indices], new_vals)
                    redundancy_cnt[update_indices] = 1
        else:
            # Update all candidates
            new_red = redundancy_func(
                target_column=best_feature,
                features=candidate_names,
                **redundancy_args
            ).fillna(FLOOR).abs().clip(FLOOR)
            new_vals = new_red.reindex(candidate_names).fillna(FLOOR).to_numpy(dtype=np.float64)
            if denom_mode == "mean":
                redundancy_sum[candidate_indices] += new_vals
                redundancy_cnt[candidate_indices] += 1
            else:
                redundancy_max[candidate_indices] = np.maximum(redundancy_max[candidate_indices], new_vals)
                redundancy_cnt[candidate_indices] = 1

    if not return_scores:
        return selected_features

    redundancy = pd.DataFrame(index=features, columns=selected_features, dtype=float)
    if only_same_domain:
        feat_domains = {f: f.split("_")[0] for f in features}
        for selected in selected_features:
            domain = feat_domains[selected]
            domain_features = [f for f in features if feat_domains[f] == domain]
            col = redundancy_func(
                target_column=selected,
                features=domain_features,
                **redundancy_args
            )
            redundancy.loc[domain_features, selected] = (
                col.reindex(domain_features).fillna(0.0).abs().to_numpy()
            )
        redundancy = redundancy.fillna(0.0)
    else:
        for selected in selected_features:
            col = redundancy_func(
                target_column=selected,
                features=features,
                **redundancy_args
            )
            redundancy[selected] = col.reindex(features).fillna(0.0).abs().to_numpy()

    return (selected_features, relevance, redundancy)


def jmi_base(K, relevance_func, joint_mi_func,
             relevance_args=None, joint_mi_args=None,
             method='jmi', only_same_domain=False,
             return_scores=False, show_progress=True):
    """General function for JMI and JMIM algorithms.

    Uses O(P) memory via incremental score accumulation.

    JMI (Joint Mutual Information):
        score(f) = sum_{s in S} I(f, s; y)
        Rewards features that are jointly informative with all selected features.

    JMIM (JMI Maximization):
        score(f) = min_{s in S} I(f, s; y)
        More conservative - feature must be jointly informative with its worst partner.

    Parameters
    ----------
    K: int
        Maximum number of features to select.

    relevance_func: callable
        Function for computing Relevance I(f; y).
        Used only for selecting the first feature.
        Must return a pandas.Series with feature names as index.

    joint_mi_func: callable
        Function for computing Joint Mutual Information I(f, s; y).
        Must accept 'target_column' (the selected feature s) and 'features' (candidates f).
        Must return a pandas.Series with feature names as index.

    relevance_args: dict (optional, default=None)
        Optional arguments for relevance_func.

    joint_mi_args: dict (optional, default=None)
        Optional arguments for joint_mi_func.

    method: str (optional, default='jmi')
        Either 'jmi' (sum aggregation) or 'jmim' (min aggregation).

    only_same_domain: bool (optional, default=False)
        If True, only features belonging to the same domain are compared.
        Note: For JMI/JMIM this is applied per-iteration only. Features outside the
        active domain are scored by relevance until they receive a joint-MI update.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, joint_mi_matrix) is returned.
        The joint_mi_matrix is recomputed for selected features and returned with full feature columns
        (shape: P x P) for backward compatibility.

    show_progress: bool (optional, default=True)
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    relevance_args = {} if relevance_args is None else relevance_args
    joint_mi_args = {} if joint_mi_args is None else joint_mi_args

    # Compute initial relevance for first feature selection
    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    K = min(K, len(features))

    n_features = len(features)
    feature_to_idx = {f: i for i, f in enumerate(features)}

    # O(P) accumulators
    if method == 'jmi':
        # Sum of joint MI with all selected features
        score_acc = np.zeros(n_features, dtype=np.float64)
        updated_cnt = np.zeros(n_features, dtype=np.int32)
    elif method == 'jmim':
        # Min of joint MI with all selected features
        score_acc = np.full(n_features, np.inf, dtype=np.float64)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jmi' or 'jmim'.")

    is_candidate = np.ones(n_features, dtype=bool)
    selected_features = []

    # Pre-compute domains if needed
    if only_same_domain:
        domains = np.array([f.split('_')[0] for f in features])

    for i in tqdm(range(K), disable=not show_progress):

        if i == 0:
            # First feature: select by relevance alone
            scores = relevance.values.copy()
        else:
            last_selected = selected_features[-1]
            last_idx = feature_to_idx[last_selected]

            # Determine which candidates to compute joint MI for
            candidate_indices = np.where(is_candidate)[0]
            if len(candidate_indices) == 0:
                break

            if only_same_domain:
                last_domain = domains[last_idx]
                same_domain_mask = domains[candidate_indices] == last_domain
                update_indices = candidate_indices[same_domain_mask]
                update_names = [features[j] for j in update_indices]
            else:
                update_indices = candidate_indices
                update_names = [features[j] for j in update_indices]

            if len(update_names) > 0:
                # Compute joint MI for candidates with last selected feature
                new_jmi = joint_mi_func(
                    target_column=last_selected,
                    features=update_names,
                    **joint_mi_args
                ).fillna(FLOOR).clip(FLOOR)

                new_vals = new_jmi.reindex(update_names).fillna(FLOOR).to_numpy(dtype=np.float64)

                if method == 'jmi':
                    score_acc[update_indices] += new_vals
                    updated_cnt[update_indices] += 1
                else:  # jmim
                    score_acc[update_indices] = np.minimum(score_acc[update_indices], new_vals)

            # Compute final scores
            if method == 'jmi':
                scores = np.where(updated_cnt > 0, score_acc, relevance.values)
            else:  # jmim
                # For features not yet updated, score_acc is inf; use relevance as fallback
                scores = np.where(np.isinf(score_acc), relevance.values, score_acc)

        # Mask out already-selected features
        scores = np.where(is_candidate, scores, -np.inf)

        # Select best
        best_idx = int(np.argmax(scores))
        best_feature = features[best_idx]
        selected_features.append(best_feature)
        is_candidate[best_idx] = False

    if not return_scores:
        return selected_features

    # Recompute joint MI for selected features while returning a full matrix (for backward compat)
    joint_mi_matrix = pd.DataFrame(0.0, index=features, columns=features, dtype=float)
    for selected in selected_features:
        if only_same_domain:
            sel_domain = selected.split('_')[0]
            domain_features = [f for f in features if f.split('_')[0] == sel_domain]
        else:
            domain_features = features

        domain_features = [f for f in domain_features if f != selected]
        col = joint_mi_func(
            target_column=selected,
            features=domain_features,
            **joint_mi_args
        )
        if domain_features:
            joint_mi_matrix.loc[domain_features, selected] = (
                col.reindex(domain_features).fillna(FLOOR).clip(FLOOR).to_numpy()
            )
        joint_mi_matrix.loc[selected, selected] = 0.0

    return (selected_features, relevance, joint_mi_matrix)
