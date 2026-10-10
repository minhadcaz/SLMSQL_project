"""Summarize benchmark datasets and execution-accuracy results."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

def summarize_dataset(records: list[dict]) -> dict:
    """Count questions, databases, and dataset distributions."""
    difficulties = Counter(str(record.get("difficulty") or "unknown") for record in records)
    databases = Counter(str(record.get("db_id") or "unknown") for record in records)
    return {
        "questions": len(records),
        "databases": len(databases),
        "by_difficulty": dict(sorted(difficulties.items())),
        "by_db_id": dict(sorted(databases.items())),
    }


def compare_results(results: list[dict], group_by: str = "difficulty") -> dict:
    """Return EX accuracy and counts grouped by a result field."""
    groups: dict[str, dict[str, Any]] = {}
    for result in results:
        key = str(result.get(group_by) or "unknown")
        stats = groups.setdefault(key, {"accuracy": 0.0, "total": 0, "correct": 0})
        stats["total"] += 1
        matched = result.get("correct", result.get("ex", result.get("EX", False)))
        stats["correct"] += int(matched is True or matched == 1)

    for stats in groups.values():
        stats["accuracy"] = stats["correct"] / stats["total"]
    return dict(sorted(groups.items()))


def print_summary(summary: dict) -> None:
    """Print dataset counts as a plain-text table."""
    rows = [("Questions", summary.get("questions", 0)),
            ("Databases", summary.get("databases", 0))]
    rows.extend((f"Difficulty: {name}", count)
                for name, count in summary.get("by_difficulty", {}).items())
    rows.extend((f"Database: {name}", count)
                for name, count in summary.get("by_db_id", {}).items())
    label_width = max(len("Statistic"), *(len(str(label)) for label, _ in rows))
    value_width = max(len("Count"), *(len(str(value)) for _, value in rows))
    border = f"+-{'-' * (label_width + 2)}-+-{'-' * (value_width + 2)}-+"
    print(border)
    print(f"| {'Statistic'.ljust(label_width)} | {'Count'.rjust(value_width)} |")
    print(border)
    for label, value in rows:
        print(f"| {str(label).ljust(label_width)} | {str(value).rjust(value_width)} |")
    print(border)


def _dataset_file(data_dir: Path, dataset: str) -> Path:
    """Find the requested development JSON in common benchmark layouts."""
    names = ("bird_dev.json", "dev.json") if dataset == "bird_dev" else (
        "spider_dev.json", "dev.json",
    )
    directories = (data_dir, data_dir / "bird", data_dir / "BIRD") if dataset == "bird_dev" else (
        data_dir, data_dir / "spider", data_dir / "Spider",
    )
    for directory in directories:
        for name in names:
            path = directory / name
            if path.is_file():
                return path
    expected = ", ".join(str(directory / name) for directory in directories for name in names)
    raise FileNotFoundError(f"Could not find {dataset} JSON; checked: {expected}")


def _load_dataset(data_dir: Path, dataset: str) -> list[dict]:
    """Load a BIRD or Spider development split."""
    path = _dataset_file(data_dir, dataset)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError(f"Expected a JSON array of objects in {path}.")
    return records


def main() -> None:
    """Print statistics for a local development dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("bird_dev", "spider_dev"), required=True)
    args = parser.parse_args()
    try:
        records = _load_dataset(args.data_dir, args.dataset)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print_summary(summarize_dataset(records))


if __name__ == "__main__":
    main()
