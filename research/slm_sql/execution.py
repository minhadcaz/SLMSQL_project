"""BIRD EX semantics: unordered sets of complete SQLite rows.

Reference: AlibabaResearch/DAMO-ConvAI bird/llm/src/evaluation.py.
Unlike the app's stricter scorer, duplicate rows and row ordering are ignored.
Gold execution failures are never silently counted as model errors.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import sqlglot
from sqlglot import exp

from core.schema_inspector import open_readonly
from research.slm_sql.artifacts import digest

EXECUTION_VERSION = "bird-set-v1"


def _value_key(value):
    if value is None:
        return ["null"]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ["number", str(value)]
        numerator, denominator = value.as_integer_ratio()
        return ["number", numerator, denominator]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    return ["text", value]


def result_fingerprint(rows) -> str:
    keys = {
        hashlib.sha256(
            json.dumps(
                [_value_key(v) for v in row],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).digest()
        for row in rows
    }
    return hashlib.sha256(b"".join(sorted(keys))).hexdigest()


def _preview(row):
    return [
        v.hex()[:160]
        if isinstance(v, bytes)
        else str(v)[:160]
        if isinstance(v, float) and not math.isfinite(v)
        else v[:160]
        if isinstance(v, str)
        else v
        for v in row
    ]


def execute(sql: str, db_path: str | Path, *, timeout=30, max_rows=1000000, preview_rows=5) -> dict:
    started = time.perf_counter()
    result = {"fingerprint": None, "error": None, "row_count": 0, "preview": [], "column_count": 0}
    deadline = time.monotonic() + timeout
    try:
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("empty_sql")
        statements = sqlglot.parse(sql, read="sqlite")
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            raise ValueError("expected_single_read_query")
        with closing(open_readonly(db_path)) as conn:
            conn.execute(f"PRAGMA busy_timeout={max(1, int(timeout * 1000))}")
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            with closing(conn.execute(sql)) as cursor:
                result["column_count"] = len(cursor.description)
                keys = set()
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("sql_timeout")
                    rows = cursor.fetchmany(512)
                    if not rows:
                        break
                    result["row_count"] += len(rows)
                    if result["row_count"] > max_rows:
                        raise ValueError("result_row_limit")
                    for row in rows:
                        if len(result["preview"]) < preview_rows:
                            result["preview"].append(_preview(row))
                        key = json.dumps(
                            [_value_key(v) for v in row], ensure_ascii=False, separators=(",", ":")
                        ).encode()
                        keys.add(hashlib.sha256(key).digest())
                if time.monotonic() >= deadline:
                    raise TimeoutError("sql_timeout")
                result["fingerprint"] = hashlib.sha256(b"".join(sorted(keys))).hexdigest()
    except (sqlite3.Error, ValueError, OSError, TimeoutError, sqlglot.errors.SqlglotError) as exc:
        result["error"] = (
            "sql_timeout" if time.monotonic() >= deadline else f"{type(exc).__name__}: {exc}"
        )
    result["elapsed_s"] = time.perf_counter() - started
    return result


class ExecutionCache:
    def __init__(self, path: str | Path, settings: dict, preview_rows: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.settings = {**settings, "preview_rows": preview_rows}

    def run(self, sql: str, record: dict) -> dict:
        key = digest([EXECUTION_VERSION, record["database_sha256"], sql, self.settings])
        cached = self.conn.execute("SELECT payload FROM results WHERE key=?", (key,)).fetchone()
        if cached:
            return json.loads(cached[0])
        value = execute(sql, record["db_path"], **self.settings)
        self.conn.execute(
            "INSERT INTO results VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False))
        )
        return value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.conn.commit()
        self.conn.close()

    def commit(self):
        self.conn.commit()
