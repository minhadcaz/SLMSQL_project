"""Filter SynSQL data and produce stratified SFT subsets.

Filtering rules from SLM-SQL §3.1:
- Must contain SELECT (read-only queries only)
- No duplicate SQL strings
- No SQL comments (--)
- Wrap in <think>/<answer> tags
- Truncate by token-proxy length
"""

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _question_text(record: dict[str, Any]) -> str:
    return str(
        record.get("question")
        or record.get("sql_prompt")
        or record.get("prompt")
        or ""
    )


def _schema_text(record: dict[str, Any]) -> str:
    return str(
        record.get("schema")
        or record.get("db_schema")
        or record.get("database_schema")
        or record.get("sql_context")
        or record.get("context")
        or ""
    )


def _reasoning_text(record: dict[str, Any]) -> str:
    return str(
        record.get("reasoning")
        or record.get("sql_explanation")
        or record.get("explanation")
        or ""
    )


def _sql_hash(sql: str) -> str:
    normalized = " ".join(sql.split()).strip().rstrip(";").lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def filter_synsql(raw_path: str | Path) -> list[dict[str, Any]]:
    """Load raw SynSQL JSONL and apply §3.1 filters.

    Returns deduplicated samples where the SQL contains SELECT,
    has no line comments, and is non-empty.
    """
    raw_path = Path(raw_path)
    seen_hashes: set[str] = set()
    accepted: list[dict[str, Any]] = []
    skipped = {"no_select": 0, "has_comment": 0, "duplicate": 0, "empty": 0, "parse_error": 0}

    with raw_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped["parse_error"] += 1
                continue

            sql = record.get("sql") or record.get("query") or record.get("gold_sql") or ""
            if not isinstance(sql, str) or not sql.strip():
                skipped["empty"] += 1
                continue

            sql = sql.strip()

            if "SELECT" not in sql.upper():
                skipped["no_select"] += 1
                continue

            if "--" in sql:
                skipped["has_comment"] += 1
                continue

            h = _sql_hash(sql)
            if h in seen_hashes:
                skipped["duplicate"] += 1
                continue
            seen_hashes.add(h)

            normalized = dict(record)
            normalized.setdefault("question", _question_text(record))
            normalized.setdefault("schema", _schema_text(record))
            normalized.setdefault("reasoning", _reasoning_text(record))
            accepted.append(normalized)

    logger.info(
        "filter_synsql: %d accepted, skipped=%s (from %s)",
        len(accepted), skipped, raw_path.name,
    )
    return accepted


def truncate_by_length(
    samples: list[dict[str, Any]], max_chars: int = 8192,
) -> list[dict[str, Any]]:
    """Drop samples whose combined prompt+response text exceeds a character limit.

    Uses character count as a proxy for tokens (~4 chars/token for English).
    The default 8192 chars ≈ 2048 tokens.
    """
    kept = []
    for sample in samples:
        text_fields = [
            _question_text(sample),
            _schema_text(sample),
            str(sample.get("sql", sample.get("query", sample.get("gold_sql", "")))),
            _reasoning_text(sample),
        ]
        total_len = sum(len(f) for f in text_fields)
        if total_len <= max_chars:
            kept.append(sample)
    logger.info("truncate_by_length: %d → %d (max_chars=%d)", len(samples), len(kept), max_chars)
    return kept


def stratified_sample(
    samples: list[dict[str, Any]],
    size: int,
    seed: int = 42,
    stratify_key: str = "db_id",
) -> list[dict[str, Any]]:
    """Sample *size* records, stratified by *stratify_key* where possible.

    Falls back to uniform random if the key is missing from records.
    """
    if size >= len(samples):
        return list(samples)

    rng = random.Random(seed)

    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        key = str(sample.get(stratify_key, "unknown"))
        groups.setdefault(key, []).append(sample)

    per_group = max(1, size // len(groups))
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for group_samples in groups.values():
        rng.shuffle(group_samples)
        selected.extend(group_samples[:per_group])
        remaining.extend(group_samples[per_group:])

    if len(selected) < size:
        rng.shuffle(remaining)
        selected.extend(remaining[: size - len(selected)])
    elif len(selected) > size:
        rng.shuffle(selected)
        selected = selected[:size]

    rng.shuffle(selected)
    logger.info(
        "stratified_sample: %d → %d (groups=%d, key=%s)",
        len(samples), len(selected), len(groups), stratify_key,
    )
    return selected


def prepare_subsets(
    raw_path: str | Path,
    output_dir: str | Path,
    sizes: tuple[int, ...] = (10_000, 30_000, 50_000),
    max_chars: int = 8192,
    seed: int = 42,
) -> dict[str, Any]:
    """Full pipeline: filter → truncate → produce sized subsets as JSONL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered = filter_synsql(raw_path)
    filtered = truncate_by_length(filtered, max_chars)

    stats: dict[str, Any] = {"filtered_total": len(filtered), "subsets": {}}

    for size in sizes:
        subset = stratified_sample(filtered, size, seed)
        out_path = output_dir / f"sft_{size}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in subset:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        lengths = []
        for r in subset:
            sql = str(r.get("sql", r.get("query", r.get("gold_sql", ""))))
            lengths.append(len(sql))

        stats["subsets"][str(size)] = {
            "actual_size": len(subset),
            "output_path": str(out_path),
            "sql_length_mean": sum(lengths) / len(lengths) if lengths else 0,
            "sql_length_max": max(lengths) if lengths else 0,
        }

    stats_path = output_dir / "prep_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d subsets + stats to %s", len(sizes), output_dir)
    return stats


def main() -> None:
    """CLI: filter SynSQL and produce stratified SFT subsets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Raw SynSQL JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 30_000, 50_000])
    parser.add_argument("--max-chars", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = prepare_subsets(args.raw, args.output_dir, tuple(args.sizes), args.max_chars, args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
