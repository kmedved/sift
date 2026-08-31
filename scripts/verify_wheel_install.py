"""Verify the installed wheel contract from outside the source tree."""

from __future__ import annotations

from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import sift


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

    print(f"verified installed SIFT wheel {dist.version} at {package_dir}")


if __name__ == "__main__":
    main()
