"""Run the classifier evaluation, active learning comparison, kappa, and charts.

Reads real labels from MongoDB and writes results/*.json, *.csv, *.png.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import active, agreement, config, model, queries
from .db import get_db


def plot_curves(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Accuracy", "Macro F1"]):
        for strat, g in df.groupby("strategy"):
            ax.plot(g["n_labeled"], g[f"{metric}_mean"], marker="o", label=strat)
            ax.fill_between(
                g["n_labeled"],
                g[f"{metric}_mean"] - g[f"{metric}_std"],
                g[f"{metric}_mean"] + g[f"{metric}_std"],
                alpha=0.15,
            )
        ax.set_xlabel("labeled examples")
        ax.set_ylabel(title)
        ax.set_title(f"{title} on held-out test set")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_ratings(db, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        [{"annotator": r["_id"]["annotator"], "rating": r["_id"]["rating"], "n": r["n"]} for r in queries.rating_distribution(db)]
    )
    pivot = df.pivot(index="rating", columns="annotator", values="n").reindex(range(1, 6)).fillna(0)
    ax = pivot.plot(kind="bar", figsize=(6, 4))
    ax.set_ylabel("items")
    ax.set_title("Rating distribution")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--second", default=None, help="second annotator name, for kappa")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    db = get_db()
    rows = model.load_labeled(db)
    if len(rows) < 40:
        raise SystemExit(f"only {len(rows)} labels; label more items first")
    texts = [r["text"] for r in rows]
    y = model.binarize([r["rating"] for r in rows])
    report = {"n_labels": len(rows), "positive_rate": sum(y) / len(y)}
    report["classifier"] = model.evaluate(texts, y)
    curves = active.summarize(active.compare(texts, y, repeats=args.repeats))
    pd.DataFrame(curves).to_csv(out / "active_learning.csv", index=False)
    plot_curves(curves, out / "active_learning.png")
    plot_ratings(db, out / "rating_distribution.png")
    if args.second:
        report["agreement"] = agreement.kappa_report(
            agreement.paired_ratings(db, config.PRIMARY_ANNOTATOR, args.second)
        )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
