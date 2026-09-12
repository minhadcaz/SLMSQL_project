"""Execute only gated queries, with complete voting results and bounded UI previews."""
import sqlite3
import time

from core.result_normalizer import display_value, fingerprint, requires_order
from core.schema_inspector import open_readonly
from core.sql_validator import SQLValidator, authorizer, sanitize_error
from core.types import ExecutionResult


class SQLExecutor:
    def __init__(self, config: dict):
        self.config = config

    def execute(self, db_path, sql: str) -> ExecutionResult:
        start = time.perf_counter()
        result = ExecutionResult()
        gate = SQLValidator().validate(sql, check_schema=False)
        if not gate.is_safe or not gate.is_valid:
            result.error = "; ".join(gate.errors)
            result.error_type = gate.error_type
            return result
        connection = None
        steps = 0

        def progress():
            nonlocal steps
            steps += 1000
            return int(steps > self.config["max_steps"] or
                       time.perf_counter() - start > self.config["timeout_seconds"])

        try:
            connection = open_readonly(db_path)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.config["max_result_bytes"])
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 50000)
            connection.set_authorizer(authorizer)
            connection.set_progress_handler(progress, 1000)
            cursor = connection.execute(sql)
            result.columns = [column[0] for column in cursor.description]
            rows = []
            size = 0
            while True:
                batch = cursor.fetchmany(128)
                if not batch:
                    break
                rows.extend(batch)
                size += sum(sum(len(v) if isinstance(v, bytes) else len(str(v).encode("utf-8"))
                                for v in row) for row in batch)
                if len(rows) > self.config["max_result_rows"] or size > self.config["max_result_bytes"]:
                    result.error_type = "result_limit"
                    result.error = "Kết quả vượt giới hạn; không dùng kết quả cắt ngắn để bỏ phiếu."
                    return result
                if progress():
                    result.error_type = "timeout"
                    result.error = "Truy vấn vượt thời gian hoặc số bước cho phép."
                    return result
            result.row_count = len(rows)
            result.rows = [[display_value(v) for v in row]
                           for row in rows[:self.config["max_rows"]]]
            result.truncated = len(rows) > self.config["max_rows"]
            result.normalized_result = fingerprint(rows, len(result.columns), requires_order(sql))
            result.unordered_result = fingerprint(rows, len(result.columns), False)
            result.ordered_result = fingerprint(rows, len(result.columns), True)
            result.success = True
        except (sqlite3.Error, OSError) as error:
            result.error = sanitize_error(error)
            message = str(error).lower()
            result.error_type = "timeout" if "interrupted" in message else "execution"
            if "authorized" in message or "prohibited" in message:
                result.error_type = "safety"
        finally:
            if connection:
                connection.close()
            result.elapsed_ms = (time.perf_counter() - start) * 1000
        return result
