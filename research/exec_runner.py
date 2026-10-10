"""Batch-execute candidate SQL and compute result fingerprints."""

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from core.result_normalizer import fingerprint, requires_order
from core.schema_inspector import open_readonly

logger = logging.getLogger(__name__)


def _exec_one(sql: str, db_path: str, timeout: int = 30) -> dict[str, Any]:
    """Execute a single SQL and return fingerprint + metadata."""
    import sqlite3
    import time as _time

    result: dict[str, Any] = {
        "sql": sql,
        "fingerprint": None,
        "error": None,
        "is_empty": False,
        "elapsed_ms": 0.0,
    }

    if not sql or not sql.strip():
        result["error"] = "empty_sql"
        return result

    start = _time.perf_counter()
    try:
        conn = open_readonly(db_path)
        deadline = _time.monotonic() + timeout
        conn.set_progress_handler(lambda: int(_time.monotonic() >= deadline), 1000)
        cursor = conn.execute(sql)
        if cursor.description is None:
            result["error"] = "no_result_set"
            conn.close()
            return result
        rows = cursor.fetchall()
        col_count = len(cursor.description)
        conn.close()

        ordered = requires_order(sql)
        result["fingerprint"] = fingerprint(rows, col_count, ordered)
        result["is_empty"] = len(rows) == 0
    except Exception as e:
        result["error"] = type(e).__name__
    finally:
        result["elapsed_ms"] = round((_time.perf_counter() - start) * 1000, 2)

    return result


def execute_candidates(
    candidates_path: str | Path,
    db_path_fn,
    output_path: str | Path,
    timeout: int = 30,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Execute all candidates and save results as JSONL."""
    from research.candidate_generator import load_candidates

    records = load_candidates(candidates_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_candidates = 0
    total_errors = 0
    total_empty = 0
    start = time.perf_counter()

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            db_path = str(db_path_fn(record["db_id"]))
            exec_results = []
            for cand in record["candidates"]:
                er = _exec_one(cand["sql"], db_path, timeout)
                exec_results.append(er)
                total_candidates += 1
                if er["error"]:
                    total_errors += 1
                if er["is_empty"]:
                    total_empty += 1

            out_record = {
                "question": record["question"],
                "db_id": record["db_id"],
                "gold_sql": record.get("gold_sql", ""),
                "difficulty": record.get("difficulty", "unknown"),
                "candidates": [
                    {**cand, **er}
                    for cand, er in zip(record["candidates"], exec_results)
                ],
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - start
    stats = {
        "total_questions": len(records),
        "total_candidates": total_candidates,
        "total_errors": total_errors,
        "total_empty": total_empty,
        "elapsed_s": round(elapsed, 1),
    }
    logger.info("exec_runner: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    from benchmarks.bird_loader import get_db_path
    stats = execute_candidates(
        args.candidates,
        lambda db_id: get_db_path(args.data_dir, db_id),
        args.output,
        args.timeout,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
