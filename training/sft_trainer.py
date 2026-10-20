"""Unsloth LoRA SFT with completion loss, checked resume and measured resources."""

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.slm_sql.artifacts import atomic_json, digest, file_hash, read_json
from research.slm_sql.adaptation import tokenize_example
from training.lora_config import LoRAExperiment
from training.prompt_template import format_sft_messages

logger = logging.getLogger(__name__)


def load_sft_dataset(data_path: str | Path) -> list[dict[str, Any]]:
    with open(data_path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_sft_records(records, tokenizer, max_length, *, cot_mode="full_cot", short_cot_tokens=128):
    """Filter full chats first so CoT variants retain identical source examples."""
    if cot_mode not in ("full_cot", "answer_only", "short_cot"):
        raise ValueError("Unknown CoT variant.")
    encoded, kept, skipped = [], [], []
    for index, record in enumerate(records):
        question = record.get("question") or record.get("sql_prompt") or record.get("prompt", "")
        sql = record.get("sql") or record.get("query") or record.get("gold_sql", "")
        reasoning = record.get("reasoning") or record.get("sql_explanation") or record.get("explanation", "")
        schema = record.get("schema") or record.get("sql_context") or record.get("db_schema") or ""
        messages = format_sft_messages(question, schema, sql, reasoning, record.get("evidence", ""))
        row = {"messages": messages[:-1], "target": messages[-1]["content"]}
        full = tokenize_example(tokenizer, row, max_length)
        if full is None:
            skipped.append(index)
            continue
        if cot_mode != "full_cot":
            if cot_mode == "answer_only":
                reasoning = ""
            elif len(tokenizer.encode(reasoning, add_special_tokens=False)) > short_cot_tokens:
                # Use the same examples; select a short prefix of the provided
                # rationale, never generate a rationale using evaluation gold.
                reasoning = tokenizer.decode(tokenizer.encode(reasoning, add_special_tokens=False)[:short_cot_tokens])
            variant = format_sft_messages(question, schema, sql, reasoning, record.get("evidence", ""))
            full = tokenize_example(tokenizer, {"messages": variant[:-1], "target": variant[-1]["content"]}, max_length)
            if full is None:
                raise ValueError("CoT variant exceeds an already validated full-chat budget.")
        encoded.append(full)
        kept.append(index)
    return encoded, kept, skipped


def train(
    experiment: LoRAExperiment, train_data_path: str | Path, output_dir: str | Path,
    resume_from: str | Path | None = None, *, max_samples: int | None = None,
    cot_mode: str = "full_cot",
) -> dict[str, Any]:
    # Import Unsloth before transformers, as required by its patching mechanism.
    from unsloth import FastLanguageModel
    import torch
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from research.reproducibility import capture_runtime

    if not torch.cuda.is_available():
        raise RuntimeError("Legacy SFT requires a CUDA session.")
    if experiment.dtype != "float16":
        raise ValueError("This T4/P100 training profile requires explicit float16.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 3, "config": asdict(experiment), "data_sha256": file_hash(train_data_path),
        "implementation": {name: file_hash(Path(__file__).resolve().parents[1] / name)
                           for name in ("training/sft_trainer.py", "training/prompt_template.py",
                                        "research/slm_sql/adaptation.py", "research/slm_sql/prompts.py")},
        "max_samples": max_samples, "cot_mode": cot_mode, "loss": "completion_only",
    }
    manifest_path = output_dir / "training_manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous and previous["signature"] != digest(spec):
        raise ValueError("Training config/data changed; use another output directory.")
    if previous and previous.get("status") == "complete":
        if any(file_hash(output_dir / "final_adapter" / name) != value for name, value in previous["adapter_files"].items()):
            raise ValueError("Completed adapter weights changed.")
        return read_json(output_dir / "train_stats.json")
    runtime = capture_runtime(output_dir, "training")
    if previous and previous.get("packages") != runtime["packages"]:
        raise ValueError("Training environment changed during resume.")
    set_seed(experiment.seed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=experiment.model_id, max_seq_length=experiment.max_seq_len,
        dtype=torch.float16, load_in_4bit=experiment.quantization == "4bit",
    )
    model = FastLanguageModel.get_peft_model(
        model, r=experiment.rank, target_modules=experiment.target_modules,
        lora_alpha=experiment.lora_alpha, lora_dropout=experiment.lora_dropout,
        use_rslora=experiment.rslora,
        use_gradient_checkpointing="unsloth" if experiment.gradient_checkpointing else False,
        random_state=experiment.seed,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_sft_dataset(train_data_path)
    raw = raw[:max_samples] if max_samples else raw
    dataset, kept, skipped = encode_sft_records(raw, tokenizer, experiment.max_seq_len, cot_mode=cot_mode)
    if not dataset:
        raise ValueError("No complete training examples fit the token budget.")
    manifest = {
        "signature": digest(spec), "spec": spec, "status": "training", "packages": runtime["packages"],
        "kept_source_indices": kept, "skipped_context_indices": skipped,
        "source_subset_sha256": digest([raw[i] for i in kept]),
        "model_revision": getattr(model.config, "_commit_hash", None),
    }
    if previous and previous.get("model_revision") != manifest["model_revision"]:
        raise ValueError("Model revision changed during resume.")
    atomic_json(manifest_path, manifest)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=experiment.per_device_train_batch_size,
        gradient_accumulation_steps=experiment.gradient_accumulation_steps,
        warmup_ratio=experiment.warmup_ratio, num_train_epochs=experiment.num_train_epochs,
        learning_rate=experiment.learning_rate, fp16=True, bf16=False,
        logging_steps=experiment.logging_steps, save_steps=experiment.save_steps,
        save_total_limit=3, seed=experiment.seed, data_seed=experiment.seed,
        weight_decay=experiment.weight_decay, report_to="none", remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    if resume_from and not Path(resume_from).is_dir():
        raise FileNotFoundError(resume_from)
    checkpoint = str(resume_from) if resume_from else get_last_checkpoint(str(output_dir / "checkpoints")) if (output_dir / "checkpoints").exists() else None
    prior_steps = read_json(Path(checkpoint) / "trainer_state.json")["global_step"] if checkpoint else 0
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.perf_counter() - started
    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trained_steps = result.global_step - prior_steps
    stats = {
        "experiment": experiment.name, "model": experiment.model_id,
        "train_samples": len(dataset), "skipped_context_examples": len(skipped),
        "output_dir": str(output_dir), "adapter_path": str(adapter_dir),
        "final_loss": result.training_loss, "config": asdict(experiment), "cot_mode": cot_mode,
        "source_subset_sha256": manifest["source_subset_sha256"],
        "wall_time_s": elapsed, "gpu_hours": elapsed / 3600,
        "steps_this_session": trained_steps,
        "seconds_per_step": elapsed / trained_steps if trained_steps else None,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "metrics": result.metrics,
    }
    atomic_json(output_dir / "train_stats.json", stats)
    manifest.update(status="complete", adapter_files={p.name: file_hash(p) for p in adapter_dir.iterdir() if p.name.startswith("adapter_")})
    atomic_json(manifest_path, manifest)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default="configs/lora_grid.yaml")
    parser.add_argument("--model-key", default="small", choices=("small", "medium"))
    parser.add_argument("--experiment-index", type=int, default=0)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cot-mode", choices=("full_cot", "answer_only", "short_cot"), default="full_cot")
    args = parser.parse_args()
    from training.lora_config import generate_experiments
    experiment = generate_experiments(args.config, args.model_key)[args.experiment_index]
    print(json.dumps(train(experiment, args.train_data, args.output_dir, args.resume_from, max_samples=args.max_samples, cot_mode=args.cot_mode), indent=2))


if __name__ == "__main__":
    main()
