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
    from research.exec_runner import _exec_one

    pairs = []
    for record in exec_records:
        gold_sql = record.get("gold_sql", "")
        if not gold_sql:
            continue
        gold_result = _exec_one(gold_sql, str(db_path_fn(record["db_id"])))
        gold_fp = gold_result["fingerprint"]
        if gold_fp is None:
            continue

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
            "schema": record.get("schema", {}),
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
    """Train a CSC LoRA adapter from merge-correct data."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    pairs = []
    with open(training_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    texts = []
    for p in pairs:
        prompt = format_csc_prompt(
            p["question"], p.get("schema", ""), p["sql_a"], p["sql_b"]
        )
        texts.append({"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{p['gold_sql']}<|im_end|>"})

    dataset = Dataset.from_list(texts)

    model, tokenizer = FastLanguageModel.from_pretrained(
        base_adapter_path,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_rank, lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none", use_rslora=True,
    )

    output_path = Path(output_path)
    config_kwargs = {
        "output_dir": str(output_path),
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "fp16": True,
        "bf16": False,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "dataset_text_field": "text",
    }
    try:
        config = SFTConfig(max_seq_length=max_seq_len, **config_kwargs)
    except TypeError as exc:
        if "max_seq_length" not in str(exc):
            raise
        config = SFTConfig(max_length=max_seq_len, **config_kwargs)

    trainer_kwargs = {"model": model, "train_dataset": dataset, "args": config}
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "tokenizer" not in str(exc):
            raise
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    trainer.train()
    model.save_pretrained(str(output_path / "final"))
    tokenizer.save_pretrained(str(output_path / "final"))

    return {"n_pairs": len(pairs), "output": str(output_path / "final")}


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
