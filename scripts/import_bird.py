"""Convert a locally downloaded BIRD JSON/SQLite package; no automatic downloads."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = json.loads(args.questions.read_text(encoding="utf-8"))
    root = args.database_root.resolve(strict=True)
    records = []
    for index, item in enumerate(items):
        db_id = item["db_id"]
        path = (root / db_id / (db_id + ".sqlite")).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing SQLite database for item {index}.")
        gold = item.get("SQL") or item.get("sql")
        if not isinstance(gold, str) or not gold.strip():
            raise ValueError(f"Missing gold SQL for item {index}.")
        records.append({"id": f"bird_{item.get('question_id', index)}", "db_id": db_id,
                        "db_path": os.path.relpath(path, args.output.resolve().parent),
                        "question": item["question"], "evidence": item.get("evidence", ""),
                        "gold_sql": gold, "difficulty": item.get("difficulty", "unknown"),
                        "tags": [], "source": "BIRD (user supplied)", "language": "en"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Imported {len(records)} questions; database files were not modified.")


if __name__ == "__main__":
    main()
