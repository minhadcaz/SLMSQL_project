"""Database-disjoint adaptation splits; final BIRD dev never creates training pairs."""

from __future__ import annotations

import random
import sqlite3
from contextlib import closing
from pathlib import Path

from core.schema_inspector import open_readonly, quote_identifier
from research.slm_sql.artifacts import (
    locked_stage,
    atomic_json,
    digest,
    file_hash,
    read_json,
    read_jsonl,
    write_jsonl,
)

SPLITS = ("train", "validation", "evaluation")


def _load(root: Path, name: str) -> list[dict]:
    source = root / f"{name}.json"
    items = read_json(source)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Expected nonempty benchmark records: {source}")
    rows = []
    seen_source_ids = set()
    for index, item in enumerate(items):
        db_id = item["db_id"]
        if not isinstance(db_id, str) or Path(db_id).name != db_id or db_id in ("", ".", ".."):
            raise ValueError("Invalid BIRD database ID.")
        sql = item.get("SQL") or item.get("sql") or item.get("gold_sql")
        question = item.get("question")
        if (
            not isinstance(sql, str)
            or not sql.strip()
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError(f"Missing question/gold SQL in {source}")
        question_id = item.get("question_id")
        if question_id is None:
            source_id = ["content", question.strip(), sql.strip()]
        else:
            if (
                isinstance(question_id, bool)
                or not isinstance(question_id, (int, str))
                or (isinstance(question_id, str) and not question_id.strip())
            ):
                raise ValueError(f"Invalid question_id in {source}: {question_id!r}")
            source_id = ["question_id", type(question_id).__name__, question_id]
        source_key = (db_id, *source_id)
        if source_key in seen_source_ids:
            raise ValueError(f"Duplicate question_id in {source}: {question_id!r}")
        seen_source_ids.add(source_key)
        identity = digest([db_id, source_id, question.strip(), sql.strip()])
        path = root / f"{name}_databases" / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            raise FileNotFoundError(f"Missing SQLite database: {path}")
        rows.append(
            {
                "id": identity,
                "question": question,
                "db_id": db_id,
                "gold_sql": sql,
                "difficulty": item.get("difficulty") or "unknown",
                "evidence": item.get("evidence") or "",
                "db_path": str(path.resolve()),
            }
        )
    return rows


def schema_text(path: str | Path, sample_rows: int = 3) -> str:
    parts = []
    with closing(open_readonly(path)) as conn:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for name, ddl in tables:
            if not ddl or "VIRTUAL TABLE" in ddl.upper():
                continue
            parts.append(ddl.rstrip(";") + ";")
            columns = conn.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
            if sample_rows and columns:
                projection = ",".join(
                    f"substr(CAST({quote_identifier(c[1])} AS TEXT),1,80)" for c in columns
                )
                samples = conn.execute(
                    f"SELECT {projection} FROM {quote_identifier(name)} LIMIT ?", (sample_rows,)
                ).fetchall()
                parts.append(f"-- {name} example rows (column order as declared): {samples!r}")
    if not parts:
        raise ValueError(f"No ordinary tables in {path}")
    return "\n".join(parts)


@locked_stage("prepare")
def prepare(cfg: dict, *, evaluation_only: bool = False) -> dict:
    folder = Path(cfg["experiment_dir"]) / "data"
    manifest_path = folder / "manifest.json"
    settings = cfg["data"]
    train_root, eval_root = Path(settings["train_dir"]), Path(settings["evaluation_dir"])
    inputs = {"evaluation": file_hash(eval_root / "dev.json")}
    if not evaluation_only:
        inputs["train"] = file_hash(train_root / "train.json")
    evaluation_keys = ("evaluation_dir", "evaluation_limit", "schema_sample_rows")
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous and previous.get("format_version") != 3:
        raise ValueError("Prepared data format changed. Choose a new experiment_dir.")
    signature = digest({"settings": settings, "inputs": inputs, "version": 3})
    if previous:
        if previous["inputs"]["evaluation"] != inputs["evaluation"] or any(
            previous["settings"][key] != settings[key] for key in evaluation_keys
        ):
            raise ValueError("Evaluation source/settings changed. Choose a new experiment_dir.")
        if evaluation_only:
            load_split(cfg, "evaluation")
            return {**previous, "reused": True}
        if "train" in previous["splits"]:
            if previous["signature"] != signature:
                raise ValueError(
                    "Split settings or source JSON changed. Choose a new experiment_dir."
                )
            for split in SPLITS:
                load_split(cfg, split)
            return {**previous, "reused": True}
        # Extend an evaluation-only run without invalidating its generator/CSC caches.
        load_split(cfg, "evaluation")

    evaluation = _load(eval_root, "dev")
    partitions = {"evaluation": evaluation}
    rng = random.Random(settings["seed"])
    if not evaluation_only:
        train = _load(train_root, "train")
        db_ids = sorted({r["db_id"] for r in train})
        eval_ids = {r["db_id"] for r in evaluation}
        if set(db_ids) & eval_ids:
            raise ValueError("Train and final evaluation share database IDs.")
        if len(db_ids) < 2:
            raise ValueError(
                "At least two training databases are required for held-out validation."
            )
        rng.shuffle(db_ids)
        n_validation = max(
            1, min(len(db_ids) - 1, round(len(db_ids) * settings["validation_fraction"]))
        )
        validation_ids = set(db_ids[:n_validation])
        partitions = {
            "train": [r for r in train if r["db_id"] not in validation_ids],
            "validation": [r for r in train if r["db_id"] in validation_ids],
            "evaluation": evaluation,
        }
    databases = {}
    database_owners = {}
    for split, rows in partitions.items():
        if split != "evaluation":
            rng.shuffle(rows)
        limit = settings[f"{split}_limit"]
        rows = rows[:limit] if limit else rows
        if not rows:
            raise ValueError(f"Empty {split} split.")
        for row in rows:
            path = Path(row["db_path"])
            if row["db_id"] not in databases:
                stat = path.stat()
                content_hash = file_hash(path)
                previous_owner = database_owners.get(content_hash)
                if previous_owner and previous_owner != split:
                    raise ValueError(
                        "Identical SQLite content appears across partitions, possibly under renamed IDs."
                    )
                database_owners[content_hash] = split
                databases[row["db_id"]] = {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": content_hash,
                    "split": split,
                }
                schema_path = folder / "schemas" / f"{row['db_id']}.json"
                atomic_json(
                    schema_path, {"text": schema_text(path, settings["schema_sample_rows"])}
                )
                databases[row["db_id"]]["schema_sha256"] = file_hash(schema_path)
            # Compile gold before GPU work; this does not execute its scan/joins.
            try:
                with closing(open_readonly(path)) as conn:
                    conn.execute("EXPLAIN " + row["gold_sql"]).fetchone()
            except sqlite3.Error as exc:
                raise ValueError(f"Gold SQL does not compile for {row['id']}: {exc}") from exc
            row["source_split"] = split
            row["database_sha256"] = databases[row["db_id"]]["sha256"]
        partitions[split] = rows
        write_jsonl(folder / f"{split}.jsonl", rows)
    manifest = {
        "format_version": 3,
        "signature": signature,
        "settings": settings,
        "inputs": inputs,
        "databases": databases,
        "splits": {
            split: {
                "count": len(rows),
                "sha256": file_hash(folder / f"{split}.jsonl"),
                "db_ids": sorted({r["db_id"] for r in rows}),
            }
            for split, rows in partitions.items()
        },
    }
    for split, info in manifest["splits"].items():
        relevant = (
            {key: settings[key] for key in evaluation_keys} if split == "evaluation" else settings
        )
        info["contract"] = digest(
            {
                "settings": relevant,
                "sha256": info["sha256"],
                "databases": {
                    db: {
                        key: value
                        for key, value in databases[db].items()
                        if key in ("sha256", "schema_sha256")
                    }
                    for db in info["db_ids"]
                },
            }
        )
    if (
        previous
        and previous["splits"]["evaluation"]["contract"]
        != manifest["splits"]["evaluation"]["contract"]
    ):
        raise ValueError("Evaluation data changed while adding adaptation splits.")
    atomic_json(manifest_path, manifest)
    return {**manifest, "reused": False}


def load_split(cfg: dict, split: str, *, verify_databases: bool = True) -> tuple[list[dict], dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    folder = Path(cfg["experiment_dir"]) / "data"
    manifest = read_json(folder / "manifest.json")
    keys = (
        ("evaluation_dir", "evaluation_limit", "schema_sample_rows")
        if split == "evaluation"
        else cfg["data"].keys()
    )
    if any(manifest["settings"][key] != cfg["data"][key] for key in keys):
        raise ValueError("Prepared split configuration differs from the current experiment.")
    if split not in manifest["splits"]:
        raise ValueError(
            f"{split} is not prepared. Run prepare with train and evaluation data to add adaptation splits."
        )
    path = folder / f"{split}.jsonl"
    if file_hash(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Prepared {split} data was modified.")
    for db_id in manifest["splits"][split]["db_ids"]:
        info = manifest["databases"][db_id]
        if file_hash(folder / "schemas" / f"{db_id}.json") != info["schema_sha256"]:
            raise ValueError("Prepared schema was modified.")
        if not verify_databases:
            continue
        stat = Path(info["path"]).stat()
        if stat.st_size != info["size"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise ValueError(f"Database changed since preparation: {info['path']}")
    return read_jsonl(path), manifest


def get_schema(cfg: dict, db_id: str) -> str:
    path = Path(cfg["experiment_dir"]) / "data" / "schemas" / f"{db_id}.json"
    return read_json(path)["text"]
