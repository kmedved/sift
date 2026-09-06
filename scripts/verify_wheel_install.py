"""Verify the installed wheel contract from outside the source tree."""

from __future__ import annotations

from dataclasses import fields
import inspect
from importlib.metadata import distribution
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import sift
import sift.experimental as experimental
from sift.importance import ImportanceResult
from sift.selection.auto_k_options import AutoKCVOptions


class _ArrayPredictor:
    """Minimal predictor that preserves duplicate-label inputs for the smoke test."""

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)[:, 0]


def main() -> None:
    dist = distribution("sift-feature-selection")
    package_dir = Path(sift.__file__).resolve().parent
    marker = package_dir / "py.typed"
    if not marker.is_file():
        raise SystemExit(f"missing installed typed-package marker: {marker}")

    metadata = dist.metadata
    if metadata.get("License-Expression") != "MIT":
        raise SystemExit(
            "wheel metadata must contain the SPDX license expression "
            f"'MIT'; got {metadata.get('License-Expression')!r}"
        )

    files = tuple(dist.files or ())
    paths = tuple(Path(str(item)) for item in files)
    if not any(path.name == "LICENSE" and "licenses" in path.parts for path in paths):
        raise SystemExit("wheel metadata does not include LICENSE under .dist-info/licenses")

    forbidden = {"benchmarks", "docs", "examples", "tests"}
    leaked = sorted({path.parts[0] for path in paths if path.parts} & forbidden)
    if leaked:
        raise SystemExit(f"wheel leaked repository-only top-level packages: {leaked}")
    if len(sift.__all__) != 64 or "ImportanceResult" in sift.__all__:
        raise SystemExit("installed wheel changed the pinned 64-name top-level surface")
    if "Stabilized" not in sift.__all__:
        raise SystemExit("installed wheel is missing the Stabilized export")
    if "PurgedTimeSeriesSplit" not in sift.__all__ or "GroupPurgedTimeSeriesSplit" not in sift.__all__:
        raise SystemExit("installed wheel is missing the purged-split exports")
    if "ModelSelector" not in sift.__all__:
        raise SystemExit("installed wheel is missing the ModelSelector export")
    if hasattr(sift, "ImportanceResult"):
        raise SystemExit("ImportanceResult must remain module-only")
    if inspect.signature(sift.select_cached).parameters["return_result"].default is not False:
        raise SystemExit("installed select_cached changed return_result default")
    if inspect.signature(sift.StabilitySelector).parameters["penalty"].default is not None:
        raise SystemExit("installed StabilitySelector changed penalty default")
    if len(fields(sift.AutoKConfig)) != 49:
        raise SystemExit("installed AutoKConfig changed its pinned 49-field storage")
    predictive = sift.AutoKConfig.predictive(
        strategy="group_cv",
        rule="one_se",
        n_folds=7,
    )
    if predictive != sift.AutoKConfig(
        k_method="gaussian_cv",
        strategy="group_cv",
        selection_rule="one_se",
        xfit_folds=7,
    ):
        raise SystemExit("installed AutoKConfig predictive preset changed flat semantics")
    grouped = sift.AutoKConfig.from_groups(
        k_method="gaussian_cv",
        cv=AutoKCVOptions(strategy="kfold", xfit_folds=3),
    )
    if grouped.cv != AutoKCVOptions(strategy="kfold", xfit_folds=3):
        raise SystemExit("installed AutoKConfig option-group view changed flat semantics")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        experimental_posterior = experimental.select_k_posterior
    if experimental_posterior is not sift.select_k_posterior:
        raise SystemExit("installed experimental namespace changed object identity")
    if len(caught) != 1 or caught[0].category is not FutureWarning:
        raise SystemExit("installed experimental access must emit one FutureWarning")

    X = np.arange(60, dtype=np.float64).reshape(20, 3)
    y = 2.0 * X[:, 0] - X[:, 1]
    selector = sift.StabilitySelector(
        n_bootstrap=2,
        alpha=0.1,
        store_coefs=False,
        n_jobs=1,
        random_state=0,
        verbose=False,
    ).fit(X, y)
    view = selector.result_view_
    if view.indices != selector.selected_features_.tolist():
        raise SystemExit("installed StabilitySelector result view changed selected positions")
    if view.transform(X).shape != (len(X), view.k):
        raise SystemExit("installed StabilitySelector result view transform has wrong shape")

    importance_model = LinearRegression().fit(X, y)
    legacy_importance = sift.permutation_importance(
        importance_model,
        X,
        y,
        n_repeats=2,
        n_jobs=1,
        random_state=0,
    )
    if list(legacy_importance.columns) != [
        "feature",
        "importance_mean",
        "importance_std",
        "baseline_score",
    ]:
        raise SystemExit("installed permutation_importance changed its default DataFrame")

    X_named = pd.DataFrame(X, columns=["dup", "dup", "noise"])
    importance = sift.permutation_importance(
        _ArrayPredictor(),
        X_named,
        y,
        n_repeats=2,
        n_jobs=1,
        random_state=0,
        return_result=True,
    )
    importance_view = sift.as_result(importance)
    if type(importance) is not ImportanceResult:
        raise SystemExit("installed return_result=True did not return ImportanceResult")
    if importance.importances_.shape != (X.shape[1], 2):
        raise SystemExit("installed ImportanceResult repeat matrix has wrong shape")
    if importance_view.indices != importance.ranking_indices:
        raise SystemExit("installed ImportanceResult view changed ranking positions")
    if importance_view.metadata.get("selection_semantics") != "ranking_only":
        raise SystemExit("installed ImportanceResult view lost ranking-only semantics")
    if importance_view.table["feature"].tolist() != ["dup", "dup", "noise"]:
        raise SystemExit("installed ImportanceResult view collapsed duplicate labels")

    X_cache = pd.DataFrame(X, columns=["signal", "proxy", "noise"])
    cache = sift.build_cache(X_cache, subsample=None)
    cached_view = sift.select_cached(
        cache,
        y,
        k=2,
        return_result=True,
        store_proxies=True,
        warn_noise_floor=False,
    )
    if type(cached_view) is not sift.SelectionView or cached_view.k != 2:
        raise SystemExit("installed cached rich-result contract is unavailable")
    if cached_view.metadata.get("cache_backed") is not True:
        raise SystemExit("installed cached result lost cache provenance")
    if cached_view.metadata.get("proxy_correlations_stored") is not True:
        raise SystemExit("installed cached result lost opt-in proxy correlations")
    proxy_rows = cached_view.proxies_at(cached_view.indices[0], r_min=0.0)
    if proxy_rows.empty:
        raise SystemExit("installed proxy lookup lost the remaining candidate")
    if set(proxy_rows["selected_index"]).intersection(cached_view.indices):
        raise SystemExit("installed proxy lookup returned an already-selected feature")

    X_context = X_cache.assign(group=np.repeat(np.arange(5), 4))
    contextual_importance = sift.permutation_importance(
        LinearRegression().fit(X_cache, y),
        X_context,
        y,
        groups="group",
        n_repeats=1,
        n_jobs=1,
        random_state=0,
    )
    if set(contextual_importance["feature"]) != set(X_cache.columns):
        raise SystemExit("installed metadata-column shorthand leaked into features")

    print(f"verified installed SIFT wheel {dist.version} at {package_dir}")


if __name__ == "__main__":
    main()
