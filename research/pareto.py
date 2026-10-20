"""Pareto analysis: aggregate experiment results, plot EX vs cost curves."""

import json
import math
from pathlib import Path
from typing import Any


def load_experiment_results(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load all experiment result JSON files from a directory."""
    results_dir = Path(results_dir)
    experiments = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
            data["_file"] = f.name
            experiments.append(data)
    return experiments


def build_pareto_table(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a table with EX, avg_samples, avg_tokens, avg_seconds per experiment."""
    rows = []
    for experiment in experiments:
        if experiment.get("accuracy_basis") == "sc_proxy_when_csc_requested":
            continue  # CSC has not been executed or costed in these replay results.
        rows.append({
            "name": experiment.get("name", experiment.get("_file", "unknown")),
            "ex": experiment.get("ex_mean", experiment.get("ex")),
            "avg_samples": experiment.get("avg_samples"),
            "avg_tokens": experiment.get("avg_tokens"),
            "avg_seconds": experiment.get("avg_seconds"),
            "seconds_basis": experiment.get("seconds_basis", "unavailable"),
        })
    return rows


def pareto_frontier(rows: list[dict[str, Any]], x_key: str = "avg_samples", y_key: str = "ex") -> list[dict[str, Any]]:
    """Extract Pareto-optimal points (maximize y, minimize x)."""
    valid = [r for r in rows if all(isinstance(r.get(key), (int, float)) and math.isfinite(r[key]) for key in (x_key, y_key))]
    # Sort ties by descending accuracy; lower-accuracy equal-cost rows are dominated.
    sorted_rows = sorted(valid, key=lambda r: (r[x_key], -r[y_key]))
    frontier = []
    best_y = -1.0
    for row in sorted_rows:
        y = row.get(y_key, 0.0)
        if y > best_y:
            frontier.append(row)
            best_y = y
    return frontier


def save_pareto_report(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    x_keys: list[str] | None = None,
) -> None:
    """Save Pareto analysis as JSON with frontier for each cost metric."""
    if x_keys is None:
        x_keys = ["avg_samples", "avg_tokens", "avg_seconds"]

    report = {
        "all_experiments": rows,
        "frontiers": {},
    }
    for x_key in x_keys:
        report["frontiers"][x_key] = pareto_frontier(rows, x_key)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def plot_pareto(
    rows: list[dict[str, Any]],
    x_key: str = "avg_samples",
    y_key: str = "ex",
    output_path: str | Path | None = None,
) -> None:
    """Plot EX vs cost with Pareto frontier highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if all(isinstance(r.get(key), (int, float)) and math.isfinite(r[key]) for key in (x_key, y_key))]
    xs = [r[x_key] for r in rows]
    ys = [r[y_key] for r in rows]
    names = [r.get("name", "") for r in rows]

    frontier = pareto_frontier(rows, x_key, y_key)
    fx = [r.get(x_key, 0) for r in frontier]
    fy = [r.get(y_key, 0) for r in frontier]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, c="steelblue", s=60, zorder=3)
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.plot(fx, fy, "r--o", markersize=8, label="Pareto frontier", zorder=4)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Pareto: {y_key} vs {x_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
