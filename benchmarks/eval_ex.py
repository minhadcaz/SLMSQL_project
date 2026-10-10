"""Evaluate SQLite execution accuracy from JSONL predictions.

Each record supplies ``pred_sql``, ``gold_sql``, ``db_id`` and optionally
``difficulty``. Accuracy is a fraction in [0, 1]; errors count records that
could not be evaluated, including invalid gold SQL, rather than result mismatches.
"""

import argparse
import json
import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot.errors import SqlglotError

from benchmarks.bird_loader import get_db_path
from core.result_normalizer import fingerprint, requires_order
from core.schema_inspector import open_readonly


_EVALUATION_ERRORS = (sqlite3.Error, OSError, ValueError, TypeError, KeyError, SqlglotError)


def _validate_timeout(timeout: int) -> None:
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be a positive integer (seconds).")


def _execute_fingerprint(
    sql: str, db_path: str | Path, timeout: int, ordered: bool,
) -> str:
    """Read the complete result, interrupting SQLite after a per-query deadline."""
    deadline = time.monotonic() + timeout

    def expired() -> bool:
        return time.monotonic() >= deadline

    def check_deadline() -> None:
        if expired():
            raise TimeoutError(f"SQL execution exceeded {timeout} seconds.")

    with closing(open_readonly(db_path)) as connection:
        check_deadline()
        # Lock waits do not invoke SQLite's progress callback.
        remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
        connection.execute(f"PRAGMA busy_timeout={remaining_ms}").close()
        connection.set_progress_handler(lambda: int(expired()), 1000)
        with closing(connection.execute(sql)) as cursor:
            if cursor.description is None:
                raise ValueError("SQL must return a result set.")
            column_count = len(cursor.description)
            rows: list[tuple[Any, ...]] = []
            while True:
                check_deadline()
                batch = cursor.fetchmany(1024)
                check_deadline()
                if not batch:
                    break
                rows.extend(batch)
    return fingerprint(rows, column_count, ordered)


def _compare(
    pred_sql: str, gold_sql: str, db_path: str | Path, timeout: int,
) -> bool:
    for name, sql in (("pred_sql", pred_sql), ("gold_sql", gold_sql)):
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"{name} must be a nonempty SQL string.")

    ordered = requires_order(gold_sql)
    if ordered:
        # The shared helper also flags LIMIT/OFFSET. EX preserves order only
        # for an outer ORDER BY; nested clauses do not order the final result.
        ordered = bool(sqlglot.parse_one(gold_sql, read="sqlite").args.get("order"))
    gold_result = _execute_fingerprint(gold_sql, db_path, timeout, ordered)
    pred_result = _execute_fingerprint(pred_sql, db_path, timeout, ordered)
    return pred_result == gold_result


def execution_match(
    pred_sql: str, gold_sql: str, db_path: str | Path, timeout: int = 30,
) -> bool:
    """Compare full results by hash, preserving duplicates and column count.

    Gold's outer ORDER BY selects ordered comparison. Otherwise rows are a
    multiset, even if the prediction orders them or gold has LIMIT/OFFSET.
    Execution errors, missing databases and timeouts in either query return
    False. A nonpositive or noninteger timeout raises ValueError.
    """
    _validate_timeout(timeout)
    try:
        return _compare(pred_sql, gold_sql, db_path, timeout)
    except _EVALUATION_ERRORS:
        return False


def batch_evaluate(
    predictions: list[dict[str, Any]],
    db_path_fn: Callable[[str], str | Path],
    timeout: int = 30,
) -> dict[str, Any]:
    """Return overall and per-difficulty counts with fractional accuracy.

    Errors remain in the denominator and count once per record. Missing or
    empty difficulty is grouped as ``unknown``. An empty batch scores 0.0.
    """
    _validate_timeout(timeout)
    correct = errors = 0
    per_difficulty: dict[str, dict[str, Any]] = {}
    for item in predictions:
        difficulty = str(item.get("difficulty") or "unknown") if isinstance(item, dict) else "unknown"
        stats = per_difficulty.setdefault(
            difficulty, {"accuracy": 0.0, "total": 0, "correct": 0, "errors": 0},
        )
        stats["total"] += 1
        try:
            if not isinstance(item, dict):
                raise TypeError("Each prediction must be a JSON object.")
            db_id = item["db_id"]
            if not isinstance(db_id, str) or not db_id.strip():
                raise ValueError("db_id must be a nonempty string.")
            matched = _compare(
                item["pred_sql"], item["gold_sql"], db_path_fn(db_id), timeout,
            )
        except _EVALUATION_ERRORS:
            errors += 1
            stats["errors"] += 1
        else:
            correct += int(matched)
            stats["correct"] += int(matched)

    for stats in per_difficulty.values():
        stats["accuracy"] = stats["correct"] / stats["total"]
    total = len(predictions)
    return {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "correct": correct,
        "errors": errors,
        "per_difficulty": per_difficulty,
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            records.append(record)
    return records


def main() -> None:
    """Print EX statistics for BIRD's dev_databases/<db_id>/<db_id>.sqlite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL file")
    parser.add_argument("--timeout", type=int, default=30, help="Seconds allowed per SQL query")
    args = parser.parse_args()
    try:
        _validate_timeout(args.timeout)
        predictions = _load_predictions(args.predictions)
        result = batch_evaluate(
            predictions, lambda db_id: get_db_path(args.data_dir, db_id), args.timeout,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
