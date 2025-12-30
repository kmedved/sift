import numpy as np


def subsample_xy(X, y, subsample, random_state, warn_subsample=False):
    """Subsample X and y to at most `subsample` rows."""
    if isinstance(y, list):
        y = np.array(y)
    if subsample is None:
        return X, y
    n = len(X)
    if n <= subsample:
        return X, y
    if warn_subsample:
        import warnings
        warnings.warn(
            f"Subsampling from {n} to {subsample} rows.",
            RuntimeWarning,
            stacklevel=3,
        )
    rng = np.random.default_rng(random_state)
    row_idx = rng.choice(n, size=subsample, replace=False)
    X_sub = X.iloc[row_idx] if hasattr(X, "iloc") else X[row_idx]
    y_sub = y.iloc[row_idx] if hasattr(y, "iloc") else y[row_idx]
    return X_sub, y_sub
