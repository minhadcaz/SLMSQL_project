"""Batch-execute candidate SQL and compute result fingerprints."""

import argparse
import errno
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _exec_one(sql: str, db_path: str, timeout: int = 30) -> dict[str, Any]:
    """Use the same complete-result, timeout-bounded BIRD scorer as 09–11."""
    from research.slm_sql.execution import EXECUTION_VERSION, execute

    result = execute(sql, db_path, timeout=timeout, preview_rows=0)
    return {
        "sql": sql,
        "fingerprint": result["fingerprint"],
        "error": result["error"],
        "is_empty": result["error"] is None and result["row_count"] == 0,
        "row_count": result["row_count"],
        "elapsed_ms": round(result["elapsed_s"] * 1000, 2),
        "execution_protocol": EXECUTION_VERSION,
    }


def _exec_job(job):
    return _exec_one(*job)


def execute_candidates(
    candidates_path: str | Path,
    db_path_fn,
    output_path: str | Path,
    timeout: int = 30,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Execute all candidates and save results as JSONL."""
    from research.candidate_generator import load_candidates

    if timeout < 1 or max_workers < 1:
        raise ValueError("timeout and max_workers must be positive.")
    from research.slm_sql.artifacts import Journal, digest, file_hash
    from research.slm_sql.execution import EXECUTION_VERSION

    records = load_candidates(candidates_path)
    output_path = Path(output_path)
    database_paths = {r["db_id"]: str(db_path_fn(r["db_id"])) for r in records}
    spec = {
        "version": 2, "execution": EXECUTION_VERSION, "timeout": timeout,
        "source_sha256": file_hash(candidates_path),
        "databases": {db: file_hash(path) for db, path in database_paths.items()},
    }
    ids = [r.get("id") or digest([r["db_id"], r["question"], r.get("gold_sql")]) for r in records]
    started = time.perf_counter()

    def run_jobs(jobs):
        if max_workers == 1:
            return [_exec_job(job) for job in jobs]
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(_exec_job, jobs))
        except BrokenProcessPool as exc:
            # Python 3.14 may select forkserver, which is unavailable in some
            # sandboxes. Preserve correctness with a deterministic CPU fallback.
            logger.warning("Parallel candidate execution unavailable; using sequential fallback: %s", exc)
            return [_exec_job(job) for job in jobs]
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise
            logger.warning("Parallel candidate execution unavailable; using sequential fallback: %s", exc)
            return [_exec_job(job) for job in jobs]

    with Journal(output_path, spec, ids) as journal:
        if not journal.complete:
            for index in range(len(journal.records), len(records)):
                record = records[index]
                path = database_paths[record["db_id"]]
                # Evaluate identical SQL only once; retain every original vote.
                sqls = list(dict.fromkeys(
                    [record.get("gold_sql", "")] + [c["sql"] for c in record["candidates"]]
                ))
                jobs = [(sql, path, timeout) for sql in sqls]
                results = run_jobs(jobs)
                by_sql = dict(zip(sqls, results))
                gold = by_sql[record.get("gold_sql", "")]
                if gold["fingerprint"] is None:
                    raise ValueError(f"Gold execution failed for {ids[index]}: {gold['error']}")
                journal.append({
                    **record, "id": ids[index], "gold_execution": gold,
                    "candidates": [{**c, **by_sql[c["sql"]]} for c in record["candidates"]],
                })
            journal.finish()
        executed = journal.records
    candidates = [c for r in executed for c in r["candidates"]]
    return {
        "total_questions": len(executed), "total_candidates": len(candidates),
        "total_errors": sum(bool(c["error"]) for c in candidates),
        "total_empty": sum(c["is_empty"] for c in candidates),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "execution_protocol": EXECUTION_VERSION,
    }


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
