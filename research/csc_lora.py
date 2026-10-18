"""CSC (Correct via Self-Consistency) module using LoRA.

Builds merge-correct training data from top-2 fingerprint groups,
trains a LoRA adapter on the merge task, and provides inference.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_csc_training_data(
    exec_records: list[dict[str, Any]],
    db_path_fn,
    output_path: str | Path,
) -> dict[str, int]:
    """Build CSC training data: pairs of top-2 candidate SQLs → gold SQL."""
    if any(r.get("source_split") != "train" for r in exec_records):
        raise ValueError(
            "CSC training pairs must come from a declared training split; dev/evaluation labels "
            "are forbidden. Use python -m research.slm_sql pairs for database-disjoint data."
        )
    from research.exec_runner import _exec_one
    from core.schema_inspector import SchemaInspector

    pairs = []
    inspector = SchemaInspector(sample_rows=2, max_tables=100, max_columns=300)
    schema_cache: dict[str, dict[str, Any]] = {}
    for record in exec_records:
        gold_sql = record.get("gold_sql", "")
        if not gold_sql:
            continue
        db_path = str(db_path_fn(record["db_id"]))
        gold_result = _exec_one(gold_sql, db_path)
        gold_fp = gold_result["fingerprint"]
        if gold_fp is None:
            continue

        schema = record.get("schema")
        if not schema:
            if db_path not in schema_cache:
                schema_cache[db_path] = inspector.inspect(db_path)
            schema = schema_cache[db_path]

        cands = record["candidates"]
        fps_to_sql: dict[str, list[str]] = {}
        for c in cands:
            fp = c.get("fingerprint")
            if fp:
                fps_to_sql.setdefault(fp, []).append(c.get("sql", ""))

        vote = Counter(c.get("fingerprint") for c in cands if c.get("fingerprint"))
        top2 = [fp for fp, _ in vote.most_common(2)]
        if len(top2) < 2:
            continue

        sql_a = fps_to_sql[top2[0]][0]
        sql_b = fps_to_sql[top2[1]][0]

        pairs.append({
            "question": record["question"],
            "db_id": record["db_id"],
            "schema": schema,
            "sql_a": sql_a,
            "sql_b": sql_b,
            "gold_sql": gold_sql,
            "gold_in_top2": gold_fp in top2,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    stats = {
        "total_pairs": len(pairs),
        "gold_in_top2": sum(1 for p in pairs if p["gold_in_top2"]),
    }
    logger.info("CSC training data: %s", stats)
    return stats


def format_csc_prompt(question: str, schema: str, sql_a: str, sql_b: str) -> str:
    """Format a CSC merge-correct prompt."""
    from core.schema_inspector import SchemaInspector

    schema_text = SchemaInspector.format(schema) if isinstance(schema, dict) else str(schema)
    return (
        f"Given the question and database schema, two candidate SQL queries are provided. "
        f"Analyze both candidates and produce the correct SQL query.\n\n"
        f"Question: {question}\n\n"
        f"Schema:\n{schema_text}\n\n"
        f"Candidate A:\n```sql\n{sql_a}\n```\n\n"
        f"Candidate B:\n```sql\n{sql_b}\n```\n\n"
        f"Produce the correct SQL query."
    )


def train_csc_lora(
    training_data_path: str | Path,
    base_adapter_path: str,
    output_path: str | Path,
    max_seq_len: int = 2048,
    lora_rank: int = 16,
    epochs: int = 3,
    lr: float = 2e-4,
) -> dict[str, Any]:
    """Legacy training entry point; use the provenance-checked author-merger workflow."""
    raise RuntimeError(
        "Legacy CSC training is disabled: it could train a generator adapter on dev gold SQL. "
        "Use python -m research.slm_sql --config configs/csc_experiment.yaml train. "
        "That command adapts the author's CSC checkpoint using declared train/validation pairs."
    )


def run_csc_inference(
    model_path: str,
    question: str,
    schema: str,
    sql_a: str,
    sql_b: str,
    max_tokens: int = 512,
) -> str:
    """Run CSC inference to merge two candidate SQLs."""
    from vllm import LLM, SamplingParams

    prompt = format_csc_prompt(question, schema, sql_a, sql_b)
    llm = LLM(model=model_path, dtype="float16", max_model_len=2048, gpu_memory_utilization=0.4)
    tokenizer = llm.get_tokenizer()

    messages = [
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate([formatted], params)

    from core.sql_extractor import extract_sql
    try:
        return extract_sql(outputs[0].outputs[0].text)
    except ValueError:
        return outputs[0].outputs[0].text.strip()
