"""LightGBM experiments scored with MCC, the competition metric.

Two validation schemes per feature set:
- stratified 5-fold CV over all train parts (what the leaderboard resembles,
  since test parts interleave with train parts in time);
- a temporal holdout: fit on the earliest 80% of parts by start time and score
  the latest 20%, which is what a model deployed on the line would face.

Early stopping uses an inner 10% split of each training fold, so the scored
fold never influences the number of trees. The decision threshold is chosen on
out-of-fold probabilities, never on the data it is scored on twice.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import load

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
              feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=63, metric="auc", verbose=-1, num_threads=12)
GROUPS = {
    "numeric": ("numeric",),
    "numeric+process": ("numeric", "process"),
    "numeric+process+sequence": ("numeric", "process", "sequence"),
}


def mcc_curve(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MCC for every distinct cut-off, predicting positive for the top-k scores."""
    order = np.argsort(-p, kind="stable")
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys)
    fp = np.arange(1, len(ys) + 1) - tp
    pos = ys.sum()
    neg = len(ys) - pos
    fn = pos - tp
    tn = neg - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    return mcc, p[order]


def best_threshold(y, p) -> tuple[float, float]:
    mcc, thr = mcc_curve(y, p)
    i = int(np.argmax(mcc))
    return float(thr[i]), float(mcc[i])


def mcc_at(y, p, t) -> float:
    pred = p >= t
    tp = float(np.sum(pred & (y == 1))); fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1))); tn = float(np.sum(~pred & (y == 0)))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den else 0.0


def fit(X, y, seed=0):
    Xtr, Xes, ytr, yes = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    model = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(Xtr, ytr), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(Xes, yes)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def run(name: str, out: Path, folds: int = 5) -> dict:
    df = load("train", GROUPS[name])
    labels = pl.read_parquet("data/features/labels.parquet").sort("Id")
    assert (df["Id"] == labels["Id"]).all()
    y = labels["Response"].to_numpy().astype(np.int8)
    feats = [c for c in df.columns if c != "Id"]
    X = df.select(feats).to_numpy().astype(np.float32)
    del df

    t0 = time.time()
    oof = np.zeros(len(y))
    rounds = []
    for k, (tr, va) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y)):
        m = fit(X[tr], y[tr], seed=k)
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        rounds.append(m.best_iteration)
        print(f"  {name} fold {k}: {m.best_iteration} trees", flush=True)
    thr, cv_mcc = best_threshold(y, oof)
    fold_mcc = [mcc_at(y[va], oof[va], thr)
                for _, va in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y)]

    # Temporal holdout: threshold picked on the training period's OOF scores.
    start = pl.read_parquet("data/features/process_sequence.parquet").filter(pl.col("is_train")) \
        .sort("Id")["start"].fill_null(np.inf).to_numpy()
    cut = np.quantile(start[np.isfinite(start)], 0.8)
    early, late = start <= cut, start > cut
    m = fit(X[early], y[early])
    p_late = m.predict(X[late], num_iteration=m.best_iteration)
    t_thr, _ = best_threshold(y[early], oof[early])
    temporal = {"mcc_at_train_threshold": mcc_at(y[late], p_late, t_thr),
                "mcc_oracle_threshold": best_threshold(y[late], p_late)[1],
                "late_parts": int(late.sum()), "late_failures": int(y[late].sum())}

    imp = dict(zip(feats, m.feature_importance("gain")))
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:25]
    report = {"name": name, "n_features": len(feats), "cv_mcc": cv_mcc, "threshold": thr,
              "fold_mcc": fold_mcc, "trees": rounds, "temporal": temporal,
              "top_gain": [[f, float(g)] for f, g in top], "minutes": (time.time() - t0) / 60}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2))
    np.save(out / f"{name}_oof.npy", oof)
    print(json.dumps({k: report[k] for k in ("name", "cv_mcc", "threshold", "temporal")}), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", default=list(GROUPS))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    for name in args.sets:
        run(name, args.out)


if __name__ == "__main__":
    main()
