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
              relevance_args={}, redundancy_args={},
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
        with respect to a variable (called target_variable).
        It must have *at least* two parameters: "target_variable" and "features".
        The index of the Series must consist of feature names.

    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.

    redundancy_args: dict (optional, default={])
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
    """

    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    K = min(K, len(features))

    # Use numpy arrays for O(P) memory instead of O(P²) DataFrame
    feature_to_idx = {f: i for i, f in enumerate(features)}
    rel_values = relevance.values.astype(np.float64)
    n_features = len(features)

    # Incremental accumulators - O(P) memory
    # For mean: store sum of redundancies, divide by count
    # For max: store running max
    if denominator_func == np.mean:
        redundancy_sum = np.zeros(n_features, dtype=np.float64)
    else:  # max
        redundancy_max = np.full(n_features, FLOOR, dtype=np.float64)

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
            if denominator_func == np.mean:
                denom = redundancy_sum / i
            else:
                denom = redundancy_max
            denom = np.where(denom < FLOOR, FLOOR, denom)
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

                for j, fname in zip(update_indices, update_names):
                    val = new_red.get(fname, FLOOR)
                    if denominator_func == np.mean:
                        redundancy_sum[j] += val
                    else:
                        redundancy_max[j] = max(redundancy_max[j], val)
        else:
            # Update all candidates
            new_red = redundancy_func(
                target_column=best_feature,
                features=candidate_names,
                **redundancy_args
            ).fillna(FLOOR).abs().clip(FLOOR)

            for j, fname in zip(candidate_indices, candidate_names):
                val = new_red.get(fname, FLOOR)
                if denominator_func == np.mean:
                    redundancy_sum[j] += val
                else:
                    redundancy_max[j] = max(redundancy_max[j], val)

    if not return_scores:
        return selected_features
    else:
        # For backward compatibility, return relevance (redundancy matrix no longer stored)
        return (selected_features, relevance, None)


def jmi_base(K, relevance_func, joint_mi_func,
             relevance_args={}, joint_mi_args={},
             method='jmi', only_same_domain=False,
             return_scores=False, show_progress=True):
    """General function for JMI and JMIM algorithms.

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

    relevance_args: dict (optional, default={})
        Optional arguments for relevance_func.

    joint_mi_args: dict (optional, default={})
        Optional arguments for joint_mi_func.

    method: str (optional, default='jmi')
        Either 'jmi' (sum aggregation) or 'jmim' (min aggregation).

    only_same_domain: bool (optional, default=False)
        If True, only features belonging to the same domain are compared.

    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, joint_mi_matrix) is returned.

    show_progress: bool (optional, default=True)
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    # Compute initial relevance for first feature selection
    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    K = min(K, len(features))

    selected_features = []
    not_selected_features = features.copy()

    # Store joint MI values for each (candidate, selected) pair
    # We'll compute lazily and cache
    joint_mi_cache = pd.DataFrame(FLOOR, index=features, columns=features)

    for i in tqdm(range(K), disable=not show_progress):

        if i == 0:
            # First feature: select by relevance alone (same as MRMR)
            score = relevance.loc[not_selected_features]
        else:
            last_selected_feature = selected_features[-1]

            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if
                                             c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features

            # Compute joint MI for candidates with the last selected feature
            if not_selected_features_sub:
                joint_mi_cache.loc[not_selected_features_sub, last_selected_feature] = joint_mi_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **joint_mi_args
                ).fillna(FLOOR).clip(FLOOR)

            # Get joint MI scores for all candidates with all selected features
            scores_matrix = joint_mi_cache.loc[not_selected_features, selected_features]

            # Aggregate based on method
            if method == 'jmi':
                score = scores_matrix.sum(axis=1)
            elif method == 'jmim':
                score = scores_matrix.min(axis=1)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'jmi' or 'jmim'.")

        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    if not return_scores:
        return selected_features
    else:
        return (selected_features, relevance, joint_mi_cache)
