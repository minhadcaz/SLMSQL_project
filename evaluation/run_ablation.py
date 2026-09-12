"""Run six configurations on identical items and seeds; model loading is shared."""
import argparse
from copy import deepcopy
import csv
from pathlib import Path
import time

from core.config import ROOT, load_config
from core.model_backend import create_backend
from databases.create_student_db import DEFAULT_DB
from evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "huggingface"], default="mock")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/evaluation_questions.json")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = load_config(args.config)
    base["model"]["backend"] = args.backend
    backend = create_backend(base)
    output = args.output or ROOT / "results" / ("ablation-" + time.strftime("%Y%m%d-%H%M%S"))
    summaries = []
    settings = [
        ("greedy", "greedy", 1, False, False),
        ("fixed_sc8", "fixed_sc", 8, False, False),
        ("fixed_sc16", "fixed_sc", 16, False, False),
        ("adaptive", "adaptive_sc", 8, False, False),
        ("adaptive_validation", "adaptive_sc", 8, True, False),
        ("adaptive_validation_correction", "adaptive_sc", 8, True, True),
    ]
    for label, mode, k, validation, correction in settings:
        config = deepcopy(base)
        config["sampling"].update(mode=mode, fixed_k=k)
        config["validation"]["schema"] = validation
        config["correction"]["enabled"] = correction
        summary = evaluate(config, args.dataset, args.db, output / label, args.limit, backend)
        summaries.append({"method": label, **summary})
    with (output / "ablation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved: {output / 'ablation.csv'}")


if __name__ == "__main__":
    main()
