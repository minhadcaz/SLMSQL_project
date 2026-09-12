"""AST gate plus SQLite preparation for scoped CTE/alias/column validation."""
import re
import sqlite3

import sqlglot
from sqlglot import exp

from core.schema_inspector import open_readonly
from core.types import ValidationResult

# Defense in depth: a read-only connection alone still allows ATTACH and extension functions.
SAFE_FUNCTIONS = set("""
abs avg char coalesce concat concat_ws count date datetime format glob group_concat hex
ifnull iif instr julianday length like likelihood likely lower ltrim max min nullif printf
quote replace round rtrim sign strftime string_agg substr substring sum time total trim
typeof unicode unixepoch unlikely upper zeroblob ceiling ceil floor mod pow power sqrt
exp ln log log10 log2 sin cos tan asin acos atan atan2 degrees radians pi trunc
row_number rank dense_rank percent_rank cume_dist ntile lag lead first_value last_value
nth_value json json_array json_array_length json_extract json_object json_quote json_type
json_valid json_group_array json_group_object
""".split())
FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
             "ATTACH", "DETACH", "PRAGMA", "VACUUM", "INTO", "TRANSACTION", "COMMIT"}


def authorizer(action, arg1, arg2, database, source):
    if action == sqlite3.SQLITE_READ:
        if (arg1 or "").lower().startswith("sqlite_"):
            return sqlite3.SQLITE_DENY
        # SQLite reports database=None for table-level reads (COUNT(*) / covering index).
        return sqlite3.SQLITE_OK if database == "main" or (
            database is None and arg2 == ""
        ) else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in SAFE_FUNCTIONS else sqlite3.SQLITE_DENY
    if action in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE}:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def sanitize_error(error: Exception | str) -> str:
    text = str(error)
    text = re.sub(r"[A-Za-z]:[\\/][^\n]*|/(?:[^\s/]+/)+[^\s]*", "[path]", text)
    return " ".join(text.split())[:500]


class SQLValidator:
    def validate(self, sql: str, db_path=None, check_schema: bool = True) -> ValidationResult:
        result = ValidationResult()
        if not sql or len(sql) > 50000:
            result.errors = ["SQL rỗng hoặc quá dài."]
            result.error_type = "syntax"
            return result
        try:
            tokens = sqlglot.tokenize(sql, read="sqlite")
            # Keywords in strings/quoted identifiers are data, not commands.
            if any(t.text.upper() in FORBIDDEN and t.token_type.name not in {"STRING", "IDENTIFIER"}
                   for t in tokens):
                result.errors = ["SQL chứa thao tác bị chặn."]
                result.error_type = "safety"
                return result
            trees = [t for t in sqlglot.parse(sql, read="sqlite") if t is not None]
            if len(trees) != 1 or not isinstance(trees[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
                result.errors = ["Chỉ cho phép một SELECT hoặc WITH ... SELECT."]
                result.error_type = "safety"
                return result
            tree = trees[0]
            if any(isinstance(n, (exp.Command, exp.DDL, exp.DML, exp.Into)) for n in tree.walk()):
                result.errors = ["AST chứa thao tác ghi hoặc command."]
                result.error_type = "safety"
                return result
            for table in tree.find_all(exp.Table):
                if table.catalog or table.db or table.name.lower().startswith("sqlite_"):
                    result.errors = ["Không cho phép bảng hệ thống hoặc database ngoài."]
                    result.error_type = "safety"
                    return result
            # SQLite authorizer checks all function calls again at preparation and execution.
            for func in tree.find_all(exp.Anonymous):
                if func.name.lower() not in SAFE_FUNCTIONS:
                    result.errors = ["Hàm SQL chưa nằm trong danh sách cho phép."]
                    result.error_type = "safety"
                    return result
            result.is_safe = True
            if any(n.comments for n in tree.walk()):
                result.warnings.append("SQL có comment; AST và SQLite vẫn được kiểm tra.")
            if any(True for _ in tree.find_all(exp.Join)):
                result.warnings.append("JOIN cần đúng nghiệp vụ; thực thi thành công chưa chứng minh đúng.")
            if check_schema and db_path:
                connection = open_readonly(db_path)
                try:
                    connection.set_authorizer(authorizer)
                    connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
                finally:
                    connection.close()
            result.is_valid = True
            result.schema_score = 1.0 if check_schema and db_path else 0.5
        except sqlglot.errors.SqlglotError as error:
            # A malformed read query may be corrected; nothing executes before a fresh gate.
            result.is_safe = bool(re.match(r"^\s*(SELECT|WITH)\b", sql, re.I))
            result.errors = [sanitize_error(error)]
            result.error_type = "syntax"
        except sqlite3.Error as error:
            result.errors = [sanitize_error(error)]
            text = str(error).lower()
            result.error_type = "schema" if any(x in text for x in (
                "no such", "ambiguous", "has no column"
            )) else "syntax"
            if "authorized" in text or "prohibited" in text:
                result.is_safe = False
                result.error_type = "safety"
        return result
