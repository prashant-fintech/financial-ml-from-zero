"""Save the tables and figures used in the post; no display server needed."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from risk_model import probability_metrics, reliability_table, review_metrics


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_report(directory, label, y, predictions, thresholds=None):
    metrics = []
    tables = []
    for name, p in predictions.items():
        metrics.append({"model": name, **probability_metrics(y, p)})
        table = reliability_table(y, p)
        table.insert(0, "model", name)
        tables.append(table)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(directory / f"{label}-metrics.csv", index=False)
    calibration = pd.concat(tables, ignore_index=True)
    calibration.to_csv(directory / f"{label}-calibration.csv", index=False)
    print(f"\n{label.upper()}\n{metrics.to_string(index=False, float_format=lambda v: f'{v:.6f}')}")

    # Two ranges, so the rare-event part of the curve stays readable.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    colors = ["#155e75", "#b45309", "#7c3aed", "#be123c", "#15803d"]
    for ax, limit in zip(axes, (1.0, .20)):
        ax.plot([0, limit], [0, limit], "--", color="#94a3b8", label="Perfect calibration")
        for (name, table), color in zip(calibration.groupby("model", sort=False), colors):
            ax.plot(table["mean_prediction"], table["event_rate"], "o-", label=name, color=color, markersize=4)
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Mean predicted 12-month probability",
               ylabel="Observed event rate", title="All probabilities" if limit == 1 else "Zoom: 0–20%")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{label.capitalize()}: synthetic mortgage loan-months", fontsize=14)
    fig.savefig(directory / f"{label}-calibration.png", dpi=170)
    plt.close(fig)

    if thresholds is not None:
        review = pd.DataFrame([
            {"model": name, **review_metrics(y, p, thresholds[name])}
            for name, p in predictions.items()
        ])
        review.to_csv(directory / f"{label}-review.csv", index=False)
        print(f"\nFIXED REVIEW POLICY\n{review.to_string(index=False)}")
    return metrics
