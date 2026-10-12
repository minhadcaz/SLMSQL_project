"""Unsloth + TRL SFT training loop with checkpoint resume for Kaggle sessions."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.lora_config import LoRAExperiment
from training.prompt_template import format_sft_messages

logger = logging.getLogger(__name__)


def load_sft_dataset(data_path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL SFT data into list of dicts."""
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def train(
    experiment: LoRAExperiment,
    train_data_path: str | Path,
    output_dir: str | Path,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Run SFT with Unsloth FastLanguageModel + TRL SFTTrainer.

    This function must run on a GPU environment (Kaggle/Colab).
    Returns training stats dict.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_in_4bit = experiment.quantization == "4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=experiment.model_id,
        max_seq_length=experiment.max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=experiment.rank,
        target_modules=experiment.target_modules,
        lora_alpha=experiment.lora_alpha,
        lora_dropout=experiment.lora_dropout,
        use_rslora=experiment.rslora,
        use_gradient_checkpointing="unsloth" if experiment.gradient_checkpointing else False,
    )

    raw_data = load_sft_dataset(train_data_path)

    def format_record(record):
        question = record.get("question") or record.get("sql_prompt") or record.get("prompt", "")
        sql = record.get("sql") or record.get("query") or record.get("gold_sql", "")
        reasoning = record.get("reasoning") or record.get("sql_explanation") or record.get("explanation", "")
        schema = (
            record.get("schema")
            or record.get("db_schema")
            or record.get("database_schema")
            or record.get("sql_context")
            or record.get("context")
            or ""
        )
        evidence = record.get("evidence", "")
        messages = format_sft_messages(
            question=question,
            schema=schema,
            gold_sql=sql,
            reasoning=reasoning,
            evidence=evidence,
        )
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
        return {"text": text}

    dataset = Dataset.from_list([format_record(r) for r in raw_data])

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": experiment.per_device_train_batch_size,
        "gradient_accumulation_steps": experiment.gradient_accumulation_steps,
        "warmup_ratio": experiment.warmup_ratio,
        "num_train_epochs": experiment.num_train_epochs,
        "learning_rate": experiment.learning_rate,
        "fp16": experiment.dtype == "float16",
        "bf16": experiment.dtype == "bfloat16",
        "logging_steps": experiment.logging_steps,
        "save_steps": experiment.save_steps,
        "save_total_limit": 3,
        "seed": experiment.seed,
        "weight_decay": experiment.weight_decay,
        "dataset_text_field": "text",
        "report_to": "none",
    }
    # TRL 0.12 used max_seq_length; newer TRL releases renamed it max_length.
    try:
        training_args = SFTConfig(max_seq_length=experiment.max_seq_len, **training_kwargs)
    except TypeError as exc:
        if "max_seq_length" not in str(exc):
            raise
        training_args = SFTConfig(max_length=experiment.max_seq_len, **training_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": training_args,
    }
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError as exc:
        if "tokenizer" not in str(exc):
            raise
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)

    if resume_from and Path(resume_from).exists():
        logger.info("Resuming from checkpoint: %s", resume_from)
        trainer.train(resume_from_checkpoint=str(resume_from))
    else:
        trainer.train()

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    final_loss = None
    for event in reversed(getattr(trainer.state, "log_history", [])):
        if event.get("loss") is not None:
            final_loss = float(event["loss"])
            break

    stats = {
        "experiment": experiment.name,
        "model": experiment.model_id,
        "train_samples": len(dataset),
        "output_dir": str(output_dir),
        "adapter_path": str(adapter_dir),
        "final_loss": final_loss,
        "config": asdict(experiment),
    }
    with open(output_dir / "train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="configs/lora_grid.yaml")
    parser.add_argument("--model-key", default="small", choices=("small", "medium"))
    parser.add_argument("--experiment-index", type=int, default=0)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    from training.lora_config import generate_experiments
    experiments = generate_experiments(args.config, args.model_key)
    exp = experiments[args.experiment_index]
    logger.info("Training experiment: %s", exp.name)
    stats = train(exp, args.train_data, args.output_dir, args.resume_from)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
