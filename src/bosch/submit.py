"""Fit the chosen feature set on all training parts and write submission.csv.

The number of trees is the median early-stopping point from cross-validation
(scaled up 10% because the final model sees all the data), and the decision
threshold is the one that maximised out-of-fold MCC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .features import load
from .model import GROUPS, PARAMS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="numeric+process+sequence")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = p.parse_args()

    report = json.loads((Path("reports") / f"{args.set}.json").read_text())
    rounds = int(np.median(report["trees"]) * 1.1)
    tr = load("train", GROUPS[args.set])
    y = pl.read_parquet("data/features/labels.parquet").sort("Id")["Response"].to_numpy()
    feats = [c for c in tr.columns if c != "Id"]
    X = tr.select(feats).to_numpy().astype(np.float32)
    del tr
    models = [lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, y), num_boost_round=rounds)
              for s in range(args.seeds)]
    del X
    te = load("test", GROUPS[args.set])
    Xt = te.select(feats).to_numpy().astype(np.float32)
    prob = np.mean([m.predict(Xt) for m in models], axis=0)
    pred = (prob >= report["threshold"]).astype(np.int8)
    pl.DataFrame({"Id": te["Id"], "Response": pred}).write_csv(args.out)
    np.save(Path("reports") / "test_prob.npy", prob)
    print(f"{rounds} trees x {args.seeds} seeds; flagged {pred.sum():,} of {len(pred):,} test parts "
          f"({100 * pred.mean():.3f}%), train failure rate {100 * y.mean():.3f}%")


if __name__ == "__main__":
    main()
