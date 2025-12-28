import sift
import numpy as np
import pandas as pd

variables = ['drop', 'second', 'third', 'first']

relevance = pd.Series(
    [0, 1, 0.5, 2], index=variables)

redundancy = pd.DataFrame([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.2, 0.9],
    [0.0, 0.2, 1.0, 0.1],
    [0.0, 0.9, 0.1, 1.0]], index=variables, columns=variables)

def relevance_func():
    return relevance

def redundancy_func(target_column, features):
    return redundancy.loc[features, target_column]

def test_mrmr_base_without_scores():
    selected_features = sift.mrmr_base(
        K=100, relevance_func=relevance_func, redundancy_func=redundancy_func,
        relevance_args={}, redundancy_args={},
        denominator_func=np.mean, only_same_domain=False,
        return_scores=False, show_progress=True)

    assert selected_features == ['first', 'third', 'second']

def test_mrmr_base_with_scores():
    selected_features, relevance_out, redundancy_out = sift.mrmr_base(
        K=100, relevance_func=relevance_func, redundancy_func=redundancy_func,
        relevance_args={}, redundancy_args={},
        denominator_func=np.mean, only_same_domain=False,
        return_scores=True, show_progress=True)

    assert selected_features == ['first', 'third', 'second']
    assert isinstance(relevance_out, pd.Series)
    assert isinstance(redundancy_out, pd.DataFrame)


def test_jmi_base_only_same_domain_relevance_fallback():
    features = ["a_1", "a_2", "b_1", "b_2"]
    relevance_local = pd.Series([0.9, 0.05, 0.2, 0.01], index=features)

    def relevance_func_local():
        return relevance_local

    def joint_mi_func_local(target_column, features):
        return pd.Series(0.1, index=features)

    selected_features = sift.jmi_base(
        K=3,
        relevance_func=relevance_func_local,
        joint_mi_func=joint_mi_func_local,
        relevance_args={},
        joint_mi_args={},
        method="jmi",
        only_same_domain=True,
        return_scores=False,
        show_progress=False,
    )

    assert selected_features[:2] == ["a_1", "b_1"]
