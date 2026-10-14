"""Turn out-of-fold scores into inspection decisions, and draw the README figures.

A quality engineer does not act on MCC. The question is: if we pull the riskiest
X% of parts for manual inspection, what share of the failures do we stop, and
at what ratio between the cost of an escaped failure and the cost of one
inspection does the model pay off?
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SETS = {"numeric": "measurements only", "numeric+process": "+ process",
        "numeric+process+sequence": "+ sequence"}
OUT = Path("docs/img")


def capture_curve(y: np.ndarray, p: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    order = np.argsort(-p, kind="stable")
    caught = np.cumsum(y[order]) / y.sum()
    idx = np.clip((fractions * len(y)).astype(int) - 1, 0, len(y) - 1)
    return caught[idx]


def best_policy(y: np.ndarray, p: np.ndarray, ratio: float) -> tuple[float, float]:
    """Inspect the top-k parts minimising cost = k * 1 + escaped_failures * ratio.

    Returns (fraction inspected, cost relative to inspecting nothing).
    """
    order = np.argsort(-p, kind="stable")
    caught = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    cost = k + (y.sum() - caught) * ratio
    i = int(np.argmin(cost))
    nothing = y.sum() * ratio
    if cost[i] >= nothing:
        return 0.0, 1.0
    return (i + 1) / len(y), float(cost[i] / nothing)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y = pl.read_parquet("data/features/labels.parquet").sort("Id")["Response"].to_numpy()
    oof = {k: np.load(f"reports/{k}_oof.npy") for k in SETS}
    reports = {k: json.loads(Path(f"reports/{k}.json").read_text()) for k in SETS}

    frac = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    table = {k: capture_curve(y, oof[k], frac).round(4).tolist() for k in SETS}
    ratios = [10, 50, 100, 200, 500, 1000]
    policy = {k: {r: best_policy(y, oof[k], r) for r in ratios} for k in SETS}
    summary = {"fractions": frac.tolist(), "capture": table,
               "policy": {k: {str(r): {"inspect_frac": round(f, 5), "relative_cost": round(c, 4)}
                              for r, (f, c) in v.items()} for k, v in policy.items()}}
    Path("reports/inspection.json").write_text(json.dumps(summary, indent=2))

    # 1) validation: CV vs temporal holdout
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(SETS))
    cv = [reports[k]["cv_mcc"] for k in SETS]
    tp = [reports[k]["temporal"]["mcc_at_train_threshold"] for k in SETS]
    ax.bar(x - 0.2, cv, 0.4, label="5-fold CV (parts mixed in time)", color="#4C78A8")
    ax.bar(x + 0.2, tp, 0.4, label="temporal holdout (latest 20%)", color="#F58518")
    for xi, a, b in zip(x, cv, tp):
        ax.text(xi - 0.2, a + 0.008, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, b + 0.008, f"{b:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, list(SETS.values()))
    ax.set_ylabel("MCC")
    ax.set_title("Same model, two validation schemes")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "validation.png", dpi=130)
    plt.close(fig)

    # 2) capture curve
    grid = np.geomspace(1e-4, 0.2, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, lab in SETS.items():
        ax.plot(grid * 100, capture_curve(y, oof[k], grid) * 100, label=lab, lw=1.8)
    ax.plot(grid * 100, grid * 100, "k:", lw=1, label="random inspection")
    ax.set_xscale("log")
    ax.set_xlabel("% of parts pulled for inspection (riskiest first)")
    ax.set_ylabel("% of failures caught")
    ax.set_title("Failures caught vs inspection effort (out-of-fold)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "capture.png", dpi=130)
    plt.close(fig)

    best = "numeric+process+sequence"
    for f, c in zip(frac, table[best]):
        print(f"inspect {100 * f:5.1f}% -> catch {100 * c:5.1f}% of failures")
    for r in ratios:
        f, c = policy[best][r]
        print(f"escape/inspection cost {r:5d}: inspect {100 * f:.2f}%, cost {100 * c:.1f}% of doing nothing")



def temporal_capture(name: str = "numeric+process+sequence") -> dict:
    """Capture curve on the latest 20% of parts with a model fit on the earliest 80%."""
    from .features import load
    from .model import GROUPS, fit

    df = load("train", GROUPS[name])
    y = pl.read_parquet("data/features/labels.parquet").sort("Id")["Response"].to_numpy()
    X = df.drop("Id").to_numpy().astype(np.float32)
    start = pl.read_parquet("data/features/process_sequence.parquet").filter(pl.col("is_train")) \
        .sort("Id")["start"].fill_null(np.inf).to_numpy()
    cut = np.quantile(start[np.isfinite(start)], 0.8)
    early, late = start <= cut, start > cut
    m = fit(X[early], y[early])
    p = m.predict(X[late], num_iteration=m.best_iteration)
    frac = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10])
    out = {"fractions": frac.tolist(), "capture": capture_curve(y[late], p, frac).round(4).tolist(),
           "policy_ratio_100": best_policy(y[late], p, 100)}
    Path("reports/inspection_temporal.json").write_text(json.dumps(out, indent=2))
    for f, c in zip(frac, out["capture"]):
        print(f"[temporal] inspect {100 * f:5.1f}% -> catch {100 * c:5.1f}% of failures")
    print("[temporal] ratio 100 policy:", out["policy_ratio_100"])
    return out


if __name__ == "__main__":
    main()
    temporal_capture()
