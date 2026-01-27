import numpy as np
import pandas as pd

from sift.boruta import select_boruta


def test_select_boruta_filters_cat_features_after_drop():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "group": np.repeat(["a", "b"], n // 2),
            "time": np.tile(np.arange(n // 2), 2),
            "cat_col": rng.integers(0, 3, size=n),
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
        }
    )
    y = rng.normal(size=n)

    selected = select_boruta(
        df,
        y,
        task="regression",
        group_col="group",
        time_col="time",
        cat_features=["group", "cat_col"],
        cat_encoding="none",
        importance_data="train",
        max_iter=2,
        random_state=0,
        verbose=False,
    )

    assert isinstance(selected, list)
