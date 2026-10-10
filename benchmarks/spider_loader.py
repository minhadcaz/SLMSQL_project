"""Load local Spider dev/test splits and inspect their SQLite schemas."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from core.schema_inspector import SchemaInspector


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array in {path}.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object in {path}, record {index}.")
        question = item.get("question")
        gold_sql = item.get("query")
        db_id = item.get("db_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question in {path}, record {index}.")
        if not isinstance(gold_sql, str) or not gold_sql.strip():
            raise ValueError(f"Missing gold SQL in {path}, record {index}.")
        if not isinstance(db_id, str) or not db_id.strip():
            raise ValueError(f"Missing db_id in {path}, record {index}.")
        records.append({
            "question": question,
            "gold_sql": gold_sql,
            "db_id": db_id,
            "difficulty": "unknown",
        })
    return records


def load_spider_dev(data_dir: str | Path) -> list[dict[str, Any]]:
    """Read and normalize Spider's dev.json split."""
    return _load_records(Path(data_dir) / "dev.json")


def load_spider_test(data_dir: str | Path) -> list[dict[str, Any]]:
    """Read and normalize Spider's test.json split."""
    return _load_records(Path(data_dir) / "test.json")


def get_db_path(data_dir: str | Path, db_id: str) -> Path:
    """Return the SQLite path for a Spider database."""
    return Path(data_dir) / "database" / db_id / f"{db_id}.sqlite"


def get_schema(data_dir: str | Path, db_id: str) -> dict[str, Any]:
    """Inspect a Spider database through SchemaInspector's read-only connection."""
    inspector = SchemaInspector(sample_rows=2, max_tables=100, max_columns=300)
    return inspector.inspect(get_db_path(data_dir, db_id))


def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return basic counts for a normalized Spider split."""
    return {
        "questions": len(records),
        "databases": len(Counter(record["db_id"] for record in records)),
        "by_database": dict(sorted(Counter(record["db_id"] for record in records).items())),
        "by_difficulty": dict(sorted(Counter(
            record["difficulty"] for record in records
        ).items())),
    }


def main() -> None:
    """Print normalized Spider records or split statistics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("dev", "test", "stats"), required=True)
    args = parser.parse_args()
    try:
        if args.action == "dev":
            result: Any = load_spider_dev(args.data_dir)
        elif args.action == "test":
            result = load_spider_test(args.data_dir)
        else:
            result = {}
            for split in ("dev", "test"):
                split_path = args.data_dir / f"{split}.json"
                if not split_path.is_file():
                    continue
                records = load_spider_dev(args.data_dir) if split == "dev" else load_spider_test(
                    args.data_dir
                )
                result[split] = _stats(records)
            if not result:
                raise FileNotFoundError(f"No dev.json or test.json found in {args.data_dir}.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
