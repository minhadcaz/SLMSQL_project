"""Filter SynSQL data and produce stratified SFT subsets.

Filtering rules from SLM-SQL §3.1:
- Must contain SELECT (read-only queries only)
- No repeated SQL in reasoning; deduplicate within the same schema
- No SQL comments (--)
- Wrap in <think>/<answer> tags
- Filter complete chat examples by actual tokenizer length when supplied
"""

import argparse
import hashlib
import json
import logging
import random
import re

import sqlglot
from sqlglot import exp
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _author_prompt_parts(prompt: str) -> tuple[str, str]:
    """Parse the public SynsQL-Think ``input_seq`` prompt into schema/question."""
    if not prompt:
        return "", ""
    schema_marker = "Database Schema:\n"
    question_marker = "\nQuestion:\n"
    if schema_marker not in prompt or question_marker not in prompt:
        return "", ""
    schema = prompt.split(schema_marker, 1)[1].split("\nThis schema describes", 1)[0].strip()
    question = prompt.split(question_marker, 1)[1].split("\n\nInstructions:", 1)[0].strip()
    return schema, question


def _question_text(record: dict[str, Any]) -> str:
    explicit = record.get("question") or record.get("sql_prompt") or record.get("prompt")
    if explicit:
        return str(explicit)
    _, question = _author_prompt_parts(str(record.get("input_seq") or ""))
    return question


def _schema_text(record: dict[str, Any]) -> str:
    explicit = (record.get("schema") or record.get("db_schema")
                or record.get("database_schema") or record.get("sql_context")
                or record.get("context"))
    if explicit:
        return str(explicit)
    schema, _ = _author_prompt_parts(str(record.get("input_seq") or ""))
    return schema


def _reasoning_text(record: dict[str, Any]) -> str:
    return str(
        record.get("reasoning")
        or record.get("sql_explanation")
        or record.get("explanation")
        or record.get("cot")
        or record.get("output")
        or record.get("output_seq")
        or ""
    )


def _sql_from_record(record: dict[str, Any]) -> str:
    sql = record.get("sql") or record.get("query") or record.get("gold_sql")
    if sql:
        return str(sql)
    output = record.get("output") or record.get("output_seq") or ""
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", str(output), re.S | re.I)
    if answer:
        sql = answer.group(1).strip()
    else:
        # Compatibility with raw examples that only put SQL in a code block.
        code = re.search(r"```sql\s*(.*?)```", str(output), re.S | re.I)
        sql = code.group(1).strip() if code else ""
    return sql


def _sql_hash(sql: str, schema: str = "") -> str:
    # Normalize keywords/identifiers without changing case-sensitive literals.
    normalized = sqlglot.parse_one(sql, read="sqlite").sql(normalize=True, comments=False)
    return hashlib.sha256((schema + "\0" + normalized).encode("utf-8")).hexdigest()


def _clean_reasoning(text: str, sql: str) -> str | None:
    think = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    text = think.group(1) if think else text
    pattern = r"\s*".join(re.escape(part) for part in sql.rstrip(";").split())
    matches = list(re.finditer(pattern, text, flags=re.I))
    if len(matches) > 1:
        return None
    if matches:
        # SynSQL output commonly contains reasoning followed by the final SQL.
        text = text[:matches[0].start()]
    return re.sub(r"```(?:sql)?\s*$", "", text.strip(), flags=re.I).strip()


def _iter_raw_records(raw_path: Path):
    """Yield JSONL records and the public author JSON-array exports."""
    if raw_path.suffix.lower() == ".json":
        with raw_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            yield 1, payload
        elif isinstance(payload, list):
            for line_no, record in enumerate(payload, 1):
                yield line_no, record
        else:
            yield 1, payload
        return
    with raw_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                yield line_no, None


def filter_synsql(raw_path: str | Path) -> list[dict[str, Any]]:
    """Load raw SynSQL JSONL and apply §3.1 filters.

    Returns deduplicated samples where the SQL contains SELECT,
    has no line comments, and is non-empty.
    """
    raw_path = Path(raw_path)
    seen_hashes: set[str] = set()
    accepted: list[dict[str, Any]] = []
    skipped = dict.fromkeys(("no_select", "has_comment", "duplicate", "empty", "parse_error", "repeated_sql_in_cot"), 0)

    for _line_no, record in _iter_raw_records(raw_path):
        if record is None:
            skipped["parse_error"] += 1
            continue

        if not isinstance(record, dict):
            skipped["parse_error"] += 1
            continue
        sql = _sql_from_record(record)
        if not isinstance(sql, str) or not sql.strip():
            skipped["empty"] += 1
            continue

        sql = sql.strip()

        try:
            trees = sqlglot.parse(sql, read="sqlite")
            readonly_select = (
                len(trees) == 1 and isinstance(trees[0], exp.Query)
                and trees[0].find(exp.Select) is not None
                and trees[0].find(exp.Into) is None
            )
        except sqlglot.errors.SqlglotError:
            readonly_select = False
        if not readonly_select:
            skipped["no_select"] += 1
            continue

        if "--" in sql:
            skipped["has_comment"] += 1
            continue

        reasoning = _clean_reasoning(_reasoning_text(record), sql)
        if reasoning is None:
            skipped["repeated_sql_in_cot"] += 1
            continue
        h = _sql_hash(sql, _schema_text(record))
        if h in seen_hashes:
            skipped["duplicate"] += 1
            continue
        seen_hashes.add(h)

        normalized = dict(record)
        normalized.update(
            sql=sql, question=_question_text(record), schema=_schema_text(record),
            reasoning=reasoning,
        )
        # Legacy SynSQL exports may omit question/schema fields. Preserve those
        # records for backwards compatibility; author-format records are
        # normalized above from input_seq. Downstream SFT validation can reject
        # incomplete prompts when it has a tokenizer and template available.
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


def filter_by_token_length(samples: list[dict], tokenizer, max_tokens=2048) -> list[dict]:
    """Count the full rendered chat including tags, evidence and template tokens."""
    from training.prompt_template import format_sft_messages

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")
    kept = []
    for sample in samples:
        if not all((_question_text(sample), _schema_text(sample), _sql_from_record(sample))):
            continue
        messages = format_sft_messages(
            _question_text(sample), _schema_text(sample), sample.get("sql", sample.get("query", "")),
            _reasoning_text(sample), sample.get("evidence", ""),
        )
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if len(ids) <= max_tokens:
            kept.append({**sample, "sft_tokens": len(ids)})
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
    *, tokenizer=None, max_tokens: int = 2048,
) -> dict[str, Any]:
    """Full pipeline: filter → truncate → produce sized subsets as JSONL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered = filter_synsql(raw_path)
    filtered = filter_by_token_length(filtered, tokenizer, max_tokens) if tokenizer else truncate_by_length(filtered, max_chars)

    stats: dict[str, Any] = {
        "filtered_total": len(filtered), "subsets": {},
        "length_filter": "full_chat_tokenizer" if tokenizer else "character_proxy_unverified",
        "max_tokens": max_tokens if tokenizer else None,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
    }

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
    parser.add_argument("--tokenizer", help="Qwen tokenizer ID/path for exact length filtering")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    stats = prepare_subsets(
        args.raw, args.output_dir, tuple(args.sizes), args.max_chars, args.seed,
        tokenizer=tokenizer, max_tokens=args.max_tokens,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
