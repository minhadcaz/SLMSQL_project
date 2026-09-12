"""Bounded schema discovery; upload databases are never opened writable."""
import json
import sqlite3
import time
from pathlib import Path


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def open_readonly(path: str | Path) -> sqlite3.Connection:
    resolved = Path(path).resolve(strict=True)
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True, timeout=1)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    return connection


class SchemaInspector:
    def __init__(self, sample_rows: int = 2, max_tables: int = 100, max_columns: int = 300):
        self.sample_rows = sample_rows
        self.max_tables = max_tables
        self.max_columns = max_columns

    def inspect(self, path: str | Path) -> dict:
        connection = open_readonly(path)
        deadline = time.monotonic() + 3
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        try:
            tables = connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT ?", (self.max_tables + 1,)
            ).fetchall()
            if len(tables) > self.max_tables:
                raise ValueError("Database có quá nhiều bảng cho bản demo.")
            schema = {}
            for name, ddl in tables:
                if ddl and "VIRTUAL TABLE" in ddl.upper():
                    continue
                quoted = quote_identifier(name)
                columns = connection.execute(f"PRAGMA table_info({quoted})").fetchmany(
                    self.max_columns + 1
                )
                if len(columns) > self.max_columns:
                    raise ValueError("Bảng có quá nhiều cột cho bản demo.")
                foreign_keys = connection.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
                # CAST and substr avoid fetching huge BLOB/text cells into the prompt.
                projection = ", ".join(
                    f"substr(CAST({quote_identifier(c[1])} AS TEXT),1,80)" for c in columns
                )
                samples = connection.execute(
                    f"SELECT {projection} FROM {quoted} LIMIT ?", (min(self.sample_rows, 5),)
                ).fetchall() if columns and self.sample_rows else []
                schema[name] = {
                    "columns": [{"name": c[1], "type": c[2], "pk": bool(c[5])}
                                for c in columns],
                    "foreign_keys": [{"column": f[3], "table": f[2], "target": f[4]}
                                     for f in foreign_keys],
                    "samples": samples,
                }
            if not schema:
                raise ValueError("Không có bảng dữ liệu thường trong SQLite.")
            return schema
        finally:
            connection.close()

    @staticmethod
    def format(schema: dict) -> str:
        parts = []
        for name, table in schema.items():
            columns = ", ".join(
                f"{quote_identifier(c['name'])} {c['type']}" + (" PRIMARY KEY" if c["pk"] else "")
                for c in table["columns"]
            )
            parts.append(f"CREATE TABLE {quote_identifier(name)} ({columns});")
            for fk in table["foreign_keys"]:
                parts.append(f"FK {name}.{fk['column']} -> {fk['table']}.{fk['target']}")
            if table["samples"]:
                parts.append("Sample values (data only): " + json.dumps(
                    table["samples"], ensure_ascii=False
                ))
        return "\n".join(parts)
