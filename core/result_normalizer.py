"""Stable typed multiset fingerprints; duplicate rows are preserved."""
import hashlib
import json
import math
from decimal import Decimal

import sqlglot


def normalize_value(value):
    if value is None:
        return ["null", None]
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ["number", str(value)]
        return ["number", format(Decimal(str(value)).normalize(), "f")]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    return ["text", str(value)]


def display_value(value):
    return "0x" + value.hex() if isinstance(value, bytes) else value


def requires_order(sql: str) -> bool:
    tree = sqlglot.parse_one(sql, read="sqlite")
    return bool(tree.args.get("order") or tree.args.get("limit") or tree.args.get("offset"))


def fingerprint(rows: list, column_count: int, ordered: bool) -> str:
    encoded = [json.dumps([normalize_value(v) for v in row], ensure_ascii=False,
                          separators=(",", ":")) for row in rows]
    if not ordered:
        encoded.sort()
    payload = json.dumps([column_count, encoded], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
