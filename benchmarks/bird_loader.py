"""Load local BIRD splits and inspect their SQLite schemas."""

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.schema_inspector import SchemaInspector


def _load_records(path: Path, db_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    selected = None if db_ids is None else set(db_ids)
    with path.open(encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise ValueError(f"Expected a JSON array in {path}.")
    records = []
    for index, item in enumerate(items):
        if selected is not None and item["db_id"] not in selected:
            continue
        gold_sql = item.get("SQL") or item.get("sql") or item.get("gold_sql")
        if not isinstance(gold_sql, str) or not gold_sql.strip():
            raise ValueError(f"Missing gold SQL in {path}, record {index}.")
        records.append({
            "question": item["question"],
            "gold_sql": gold_sql,
            "db_id": item["db_id"],
            "difficulty": item.get("difficulty") or "unknown",
            "evidence": item.get("evidence") or "",
        })
    return records


def load_bird_dev(data_dir: str | Path) -> list[dict[str, Any]]:
    """Read and normalize the local dev.json split."""
    return _load_records(Path(data_dir) / "dev.json")


def load_bird_train_subset(
    data_dir: str | Path, db_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Read train.json, optionally keeping only the given database IDs."""
    return _load_records(Path(data_dir) / "train.json", db_ids)


def get_db_path(data_dir: str | Path, db_id: str) -> Path:
    """Return the SQLite path for a BIRD development database."""
    return Path(data_dir) / "dev_databases" / db_id / f"{db_id}.sqlite"


def get_schema(data_dir: str | Path, db_id: str) -> dict[str, Any]:
    """Inspect a database through SchemaInspector's read-only connection."""
    inspector = SchemaInspector(sample_rows=2, max_tables=100, max_columns=300)
    return inspector.inspect(get_db_path(data_dir, db_id))


def main() -> None:
    """Print normalized records or split statistics as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("dev", "train", "stats"), required=True)
    parser.add_argument("--db-ids", nargs="+", help="Database IDs to include from train.json")
    args = parser.parse_args()
    if args.db_ids is not None and args.action == "dev":
        parser.error("--db-ids applies only to train or stats.")
    try:
        if args.action == "dev":
            result = load_bird_dev(args.data_dir)
        elif args.action == "train":
            result = load_bird_train_subset(args.data_dir, args.db_ids)
        else:
            result = {}
            for split in ("dev", "train"):
                if not (args.data_dir / f"{split}.json").is_file():
                    continue
                records = (
                    load_bird_dev(args.data_dir) if split == "dev"
                    else load_bird_train_subset(args.data_dir, args.db_ids)
                )
                by_database = Counter(record["db_id"] for record in records)
                result[split] = {
                    "questions": len(records),
                    "databases": len(by_database),
                    "by_database": dict(sorted(by_database.items())),
                    "by_difficulty": dict(sorted(Counter(
                        record["difficulty"] for record in records
                    ).items())),
                }
            if not result:
                raise FileNotFoundError(f"No dev.json or train.json found in {args.data_dir}.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
