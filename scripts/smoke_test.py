"""Run a complete query against the demo database."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import ROOT, load_config
from core.pipeline import Pipeline
from databases.create_student_db import create_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mock", "huggingface"], default="mock")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model-id")
    parser.add_argument("--mode", choices=["greedy", "fixed_sc", "adaptive_sc"], default="adaptive_sc")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    config = load_config()
    config["model"].update(backend=args.backend, device=args.device)
    if args.model_id:
        config["model"]["model_id"] = args.model_id
    config["sampling"]["mode"] = args.mode
    config["generation"]["max_new_tokens"] = args.max_new_tokens
    db_path = create_database()
    result = Pipeline(config).run(db_path, "Có bao nhiêu sinh viên?")
    target = ROOT / "results" / f"smoke_{args.backend}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result.public_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.selected:
        raise RuntimeError("Smoke test failed: no executable SQL; inspect results JSON.")
    assert result.selected.execution.rows == [[16]], "Expected the demo student count to be 16"
    print(f"PASS backend={args.backend} mode={args.mode} candidates={len(result.candidates)}")
    print(result.selected.sql)


if __name__ == "__main__":
    main()
