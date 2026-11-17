"""Acceptance benchmark for ``sift.sample_rows`` outside the DARKO panel.

Two questions, answered on a synthetic regression panel, the CDC diabetes
survey (binary classification) and California housing (small-n regression):

1. **Training.** Does a model fitted on a weighted smart sample land closer to
   the full-data model than one fitted on a uniform sample of the same size?
   Learners: HistGradientBoosting (fixed 300 iterations, no early stopping) and
   a linear model (Ridge / LogisticRegression).
2. **Feature selection.** Does ``select_mrmr`` on a weighted smart sample pick
   features as well as on a uniform sample? Scored by agreement with the
   full-data selection, recall of the true features (synthetic only), and the
   test score of an HGB model trained on all rows with the selected features.

Rule fixed before running (see docs/ADVANCED.md for the outcome):
  training passes if, at 20% of rows, smart minus uniform (log test loss,
  paired by seed, mean over dataset x learner cells) has a 95% seed-bootstrap
  interval entirely below 0, and no single cell has smart significantly worse
  than uniform; selection passes if smart is not significantly worse than
  uniform on downstream score in any dataset at 10% or 20%.

Usage: python benchmarks/bench_row_sampling.py OUT_DIR [--seeds N] [--quick] [--cdc PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sift  # noqa: E402

# UCI 891 "CDC Diabetes Health Indicators" CSV; pass --cdc PATH (skipped when absent).
CDC_DEFAULT = Path.home() / "code/darkofit/.cache/r21-binary-fresh-reserve-v2-20260807/cdc_diabetes_health_indicators.csv"
FRACS = (0.1, 0.2)


def synth(seed: int = 0, n: int = 400_000, p: int = 60):
    """Panel-like regression: 12 informative features, proxies, heavy-tailed heteroscedastic noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    for j in range(12, 24):  # noisy proxies of informative features
        X[:, j] = 0.7 * X[:, j - 12] + 0.7 * X[:, j]
    signal = (X[:, :6] @ [1.0, -0.8, 0.6, 0.5, -0.4, 0.3] + np.sin(2 * X[:, 6]) + 0.5 * X[:, 7] ** 2
              + 0.6 * X[:, 8] * X[:, 9] + 0.15 * X[:, 10] + 0.1 * X[:, 11])
    y = signal + rng.standard_t(3, size=n) * (0.5 + np.abs(X[:, 0]))
    groups = rng.integers(0, 5_000, n)
    time_ = rng.integers(0, 400, n)
    cols = [f"x{j}" for j in range(p)]
    df = pd.DataFrame(X, columns=cols)
    test = rng.random(n) < 0.25
    return dict(name="synth_reg", task="regression", X=df, y=y, groups=groups, time=time_,
                test=test, true=set(cols[:12]), k=12)


def cdc(path):
    df = pd.read_csv(path).drop(columns=["ID"])
    y = df.pop("Diabetes_binary").to_numpy()
    test = np.random.default_rng(1).random(len(df)) < 0.2
    return dict(name="cdc_diabetes", task="classification", X=df.astype(np.float64), y=y,
                groups=None, time=None, test=test, true=None, k=8)


def calif():
    d = fetch_california_housing(as_frame=True)
    test = np.random.default_rng(2).random(len(d.frame)) < 0.2
    return dict(name="california", task="regression", X=d.data, y=d.target.to_numpy(), groups=None,
                time=None, test=test, true=None, k=None)


def learners(task):
    if task == "regression":
        return {"hgb": lambda: HistGradientBoostingRegressor(max_iter=300, early_stopping=False, random_state=0),
                "linear": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))}
    return {"hgb": lambda: HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=0),
            "linear": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))}


def fit_score(make, task, Xtr, ytr, Xte, yte, w=None):
    m = make()
    kw = {}
    if w is not None:
        kw = {f"{m.steps[-1][0]}__sample_weight" if hasattr(m, "steps") else "sample_weight": w}
    m.fit(Xtr, ytr, **kw)
    if task == "regression":
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    return float(log_loss(yte, m.predict_proba(Xte)[:, 1]))


def draw(ds, arm, frac, seed, Xtr, ytr, gtr, ttr):
    if arm == "uniform":
        n = len(ytr)
        k = int(np.floor(frac * n))
        return np.sort(np.random.default_rng([seed, 7]).choice(n, k, replace=False)), None
    s = sift.sample_rows(Xtr, ytr, frac, groups=gtr, time=ttr, task=ds["task"], random_state=seed)
    return s.indices, s.weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cdc", type=Path, default=CDC_DEFAULT)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "rows.jsonl", "a")
    datasets = [synth(n=60_000) if a.quick else synth(), calif()]
    if a.cdc.exists():
        datasets.insert(1, cdc(a.cdc))
    else:
        print(f"skipping CDC diabetes: {a.cdc} not found", flush=True)
    for ds in datasets:
        tr, te = ~ds["test"], ds["test"]
        Xtr, Xte = ds["X"][tr].reset_index(drop=True), ds["X"][te].reset_index(drop=True)
        ytr, yte = ds["y"][tr], ds["y"][te]
        gtr = None if ds["groups"] is None else ds["groups"][tr]
        ttr = None if ds["time"] is None else ds["time"][tr]
        L = learners(ds["task"])

        def emit(**row):
            row.update(dataset=ds["name"], sift=sift.__version__)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(row, flush=True)

        # full-data references
        for lname, make in L.items():
            t0 = time.monotonic()
            emit(kind="train", arm="full", frac=1.0, seed=0, learner=lname,
                 loss=fit_score(make, ds["task"], Xtr, ytr, Xte, yte), secs=time.monotonic() - t0)
        full_sel = None
        if ds["k"]:
            full_sel = sift.select_mrmr(Xtr, ytr, ds["k"], task=ds["task"], subsample=None)
            emit(kind="select", arm="full", frac=1.0, seed=0, selected=list(full_sel),
                 downstream=fit_score(L["hgb"], ds["task"], Xtr[full_sel], ytr, Xte[full_sel], yte))
        for seed in range(a.seeds):
            for frac in FRACS:
                for arm in ("uniform", "smart"):
                    t0 = time.monotonic()
                    idx, w = draw(ds, arm, frac, seed, Xtr, ytr, gtr, ttr)
                    dsecs = time.monotonic() - t0
                    Xs, ys = Xtr.iloc[idx], ytr[idx]
                    for lname, make in L.items():
                        t0 = time.monotonic()
                        emit(kind="train", arm=arm, frac=frac, seed=seed, learner=lname, rows=int(idx.size),
                             loss=fit_score(make, ds["task"], Xs, ys, Xte, yte, w),
                             secs=time.monotonic() - t0 + dsecs)
                    if ds["k"]:
                        sel = sift.select_mrmr(Xs.reset_index(drop=True), ys, ds["k"], task=ds["task"],
                                               subsample=None, sample_weight=w)
                        emit(kind="select", arm=arm, frac=frac, seed=seed, selected=list(sel),
                             jaccard=len(set(sel) & set(full_sel)) / len(set(sel) | set(full_sel)),
                             recall=(len(set(sel) & ds["true"]) / len(ds["true"])) if ds["true"] else None,
                             downstream=fit_score(L["hgb"], ds["task"], Xtr[sel], ytr, Xte[sel], yte))


if __name__ == "__main__":
    main()
