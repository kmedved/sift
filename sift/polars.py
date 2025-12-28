import numpy as np
import pandas as pd
import polars as pl
from .main import groupstats2fstat, mrmr_base


def get_numeric_features(df, target_column):
    """Get all numeric column names from a Polars DataFrame

    Parameters
    ----------
    df: polars.DataFrame
        Polars DataFrame.

    target_column: str
        Name of target column.

    Returns
    -------
    numeric_features : list of str
        List of numeric column names.
    """
    numeric_features = []
    numeric_dtypes = {
        pl.Float32,
        pl.Float64,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
    for col in df.columns:
        if col == target_column:
            continue
        dtype = df.schema[col]
        is_num = False
        if hasattr(dtype, 'is_numeric'):
            attr = dtype.is_numeric
            is_num = attr() if callable(attr) else attr
        else:
            try:
                is_num = dtype in numeric_dtypes
            except TypeError:
                is_num = False
        if is_num:
            numeric_features.append(col)
    return numeric_features


def correlation(target_column, features, df):
    if len(features) == 0:
        return pd.Series(dtype=float)
    corrs = {}
    for f in features:
        filtered = df.filter(pl.col(f).is_not_null() & pl.col(target_column).is_not_null())
        if filtered.height >= 2:
            val = filtered.select(pl.corr(f, target_column))[0, 0]
            corrs[f] = val if (val is not None and np.isfinite(val)) else 0.0
        else:
            corrs[f] = 0.0
    return pd.Series(corrs, dtype=float)


def notna(target_column, features, df):
    if len(features) == 0:
        return pd.Series(dtype=float)
    counts = df.select([
        ((pl.col(f).is_not_null()) & (pl.col(target_column).is_not_null())).sum().alias(f)
        for f in features
    ]).to_pandas().iloc[0]
    return counts.astype(float)


def f_regression(target_column, features, df):
    """F-statistic between one numeric target column and many numeric columns of a Polars DataFrame

    F-statistic is actually obtained from the Pearson's correlation coefficient through the following formula:
    corr_coef ** 2 / (1 - corr_coef ** 2) * degrees_of_freedom
    where degrees_of_freedom = n_instances - 1.

    Parameters
    ----------
    target_column: str
        Name of target column.

    features: list of str
        List of numeric column names.

    df: polars.DataFrame
        Polars DataFrame.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic between each column and the target column.
    """

    corr_coef = correlation(target_column=target_column, features=features, df=df)
    n = notna(target_column=target_column, features=features, df=df)

    deg_of_freedom = n - 2
    corr_coef_squared = corr_coef ** 2
    f = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom

    return f


def f_classif(target_column, features, df):
    groupby_method = getattr(df, "group_by", None) or df.groupby
    try:
        groupby = groupby_method(target_column, maintain_order=True)
    except TypeError:
        groupby = groupby_method(target_column)
    
    avg = groupby \
        .agg([pl.col(feature).mean().alias(feature) for feature in features]) \
        .to_pandas() \
        .set_index(target_column)
    
    n = groupby \
        .agg([(pl.col(feature).is_not_null().sum()).alias(feature) for feature in features]) \
        .to_pandas() \
        .set_index(target_column)
    
    var = groupby \
        .agg([pl.col(feature).var().alias(feature) for feature in features]) \
        .to_pandas() \
        .set_index(target_column) \
        * (n-1) / n

    f = groupstats2fstat(avg=avg, var=var, n=n)
    f.name = target_column

    return f


def mrmr_classif(df, K, target_column, features=None, denominator='mean', only_same_domain=False,
                 return_scores=False, show_progress=True):
    """MRMR feature selection for a classification task.

    Parameters
    ----------
    df: polars.DataFrame
        Polars DataFrame.

    K: int
        Number of features to select.

    target_column: str
        Name of target column.

    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.

    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
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

    if features is None:
        features = get_numeric_features(df=df, target_column=target_column)

    if type(denominator) == str and denominator == 'mean':
        denominator_func = np.mean
    elif type(denominator) == str and denominator == 'max':
        denominator_func = np.max
    elif type(denominator) == str:
        raise ValueError("Invalid denominator function. It should be one of ['mean', 'max'].")
    else:
        denominator_func = denominator

    relevance_args = {'target_column': target_column, 'features': features, 'df': df}
    redundancy_args = {'df': df}

    return mrmr_base(K=K, relevance_func=f_classif, redundancy_func=correlation,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)


def mrmr_regression(df, target_column, K, features=None, denominator='mean', only_same_domain=False,
                    return_scores=False, show_progress=True):
    """MRMR feature selection for a regression task.

    Parameters
    ----------
    df: polars.DataFrame
        Polars DataFrame.

    target_column: str
        Name of target column.

    K: int
        Number of features to select.

    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.

    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
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
    selected: list of str
        List of selected features.
    """

    if features is None:
        features = get_numeric_features(df=df, target_column=target_column)

    if type(denominator) == str and denominator == 'mean':
        denominator_func = np.mean
    elif type(denominator) == str and denominator == 'max':
        denominator_func = np.max
    elif type(denominator) == str:
        raise ValueError("Invalid denominator function. It should be one of ['mean', 'max'].")
    else:
        denominator_func = denominator

    relevance_args = {'target_column': target_column, 'features': features, 'df': df}
    redundancy_args = {'df': df}

    return mrmr_base(K=K, relevance_func=f_regression, redundancy_func=correlation,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)
