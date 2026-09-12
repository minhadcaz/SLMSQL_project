"""Create three standalone charts from ablation.csv."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    suffix = " — MOCK FIXTURES, NOT MODEL QUALITY" if (frame.backend == "mock").any() else ""
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for row in frame.itertuples():
        ax.scatter(row.average_latency_ms, row.execution_accuracy * 100,
                   label=row.method, s=65)
    ax.set(xlabel="Average latency (ms)", ylabel="Execution accuracy (%)", title="Accuracy / latency" + suffix)
    ax.set_ylim(-3, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(args.csv.parent / "accuracy_latency.png", dpi=160)
    plt.close(fig)
    for column, name, ylabel in [
        ("average_candidates", "average_candidates", "Average original candidates"),
        ("valid_sql_rate", "valid_rate", "Executable candidate rate (0–1)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.barh(frame.method, frame[column], color="#2563eb")
        ax.set(xlabel=ylabel, title=ylabel + suffix)
        fig.savefig(args.csv.parent / f"{name}.png", dpi=160)
        plt.close(fig)
    print(f"Charts saved in {args.csv.parent}")


if __name__ == "__main__":
    main()
