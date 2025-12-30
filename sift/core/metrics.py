import numpy as np

HIGHER_IS_BETTER_METRICS = frozenset({
    'AUC', 'ACCURACY', 'R2', 'NDCG', 'PFOUND', 'MAP', 'PRAUC',
    'RECALLAT', 'PRECISIONAT', 'F1', 'TOTALF1', 'MCC',
    'BALANCEDACCURACY', 'KAPPA', 'WKAPPA', 'AVERAGEGAIN',
})

LOWER_IS_BETTER_METRICS = frozenset({
    'RMSE', 'MAE', 'MAPE', 'SMAPE', 'MEDIANABSOLUTEERROR',
    'LOGLOSS', 'CROSSENTROPY', 'MULTICLASS', 'MULTICLASSONEVSALL',
    'QUANTILE', 'EXPECTILE', 'POISSON', 'TWEEDIE', 'HUBER',
    'LOGLINQUANTILE', 'MSLE', 'QUERYRMSE', 'QUERYSOFTMAX',
})


def infer_higher_is_better(metric):
    """Infer score direction from metric name."""
    if metric is None:
        return False
    metric_upper = metric.upper()
    for m in HIGHER_IS_BETTER_METRICS:
        if m in metric_upper:
            return True
    for m in LOWER_IS_BETTER_METRICS:
        if m in metric_upper:
            return False
    return False


def best_score_from_dict(scores, higher_is_better):
    """Return (best_k, best_score) from scores dict, filtering invalid values."""
    if not scores:
        return 0, float('nan')
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        return 0, float('nan')
    if higher_is_better:
        best_k = max(valid, key=valid.get)
    else:
        best_k = min(valid, key=valid.get)
    return best_k, valid[best_k]
