"""Build train-only disagreement pairs and adapt the author's RL-trained merger."""

from __future__ import annotations

import inspect
import importlib.metadata
import random
from pathlib import Path

import sqlglot
from sqlglot import exp

from research.slm_sql.artifacts import (
    locked_stage,
    atomic_json,
    completed_artifact,
    digest,
    file_hash,
    read_json,
    read_jsonl,
    write_jsonl,
)
from research.slm_sql.config import artifact_path
from research.slm_sql.data import get_schema, load_split
from research.slm_sql.pipeline import load_execution
from research.slm_sql.prompts import merge_messages, top_groups


def focus_tags(sql_a: str, sql_b: str) -> list[str]:
    """Structural heuristics, not assertions of the semantic cause of an error."""
    try:
        trees = [sqlglot.parse_one(sql, read="sqlite") for sql in (sql_a, sql_b)]
    except sqlglot.errors.SqlglotError:
        return []
    tags = []
    joins = [
        sorted(node.sql(dialect="sqlite", normalize=True) for node in tree.find_all(exp.Join))
        for tree in trees
    ]
    aggregates = [
        sorted(
            node.sql(dialect="sqlite", normalize=True)
            for kind in (exp.AggFunc, exp.Group, exp.Having, exp.Distinct)
            for node in tree.find_all(kind)
        )
        for tree in trees
    ]
    if joins[0] != joins[1]:
        tags.append("join_disagreement")
    if aggregates[0] != aggregates[1]:
        tags.append("aggregation_disagreement")
    return tags


@locked_stage("build_pairs")
def build_pairs(cfg: dict, seed=42) -> dict:
    folder = Path(cfg["experiment_dir"]) / "adaptation"
    sources, partitions, databases = {}, {}, {}
    for split in ("train", "validation"):
        prepared, _ = load_split(cfg, split)
        records, manifest = load_execution(cfg, split, seed)
        sources[split] = {"sha256": manifest["sha256"], "signature": manifest["signature"]}
        databases[split] = {r["database_sha256"] for r in prepared}
        partitions[split] = records
    if databases["train"] & databases["validation"]:
        raise ValueError("Training and validation databases overlap.")
    evaluation, data_manifest = load_split(cfg, "evaluation")
    eval_dbs = {r["database_sha256"] for r in evaluation}
    if eval_dbs & (databases["train"] | databases["validation"]):
        raise ValueError("Adaptation data overlaps final evaluation databases.")
    spec = {
        "version": 1,
        "seed": seed,
        "sources": sources,
        "prepared": data_manifest["signature"],
        "augment_swap": cfg["training"]["augment_swap"],
        "focus": "join_aggregate_structural_disagreement",
        "target": "verified_correct_draft_answer_only",
        "implementation": file_hash(Path(__file__)),
    }
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["signature"] != digest(spec):
            raise ValueError(
                "Pair configuration or source artifacts changed. Choose a new experiment_dir."
            )
        for split in ("train", "validation"):
            if file_hash(folder / f"{split}.jsonl") != previous["files"][split]:
                raise ValueError("Adaptation pairs were modified.")
        return {**previous, "reused": True}
    stats, files = {}, {}
    rng = random.Random(cfg["training"]["seed"])
    for split, records in partitions.items():
        pairs = []
        counts = {
            "questions": len(records),
            "disagreements": 0,
            "focused": 0,
            "gold_outside_top2": 0,
            "leader_correct": 0,
            "runner_up_correct": 0,
        }
        for record in records:
            if record["source_split"] != split:
                raise ValueError("Pair source split is inconsistent.")
            groups = top_groups(record["candidates"])
            if len(groups) < 2:
                continue
            counts["disagreements"] += 1
            tags = focus_tags(groups[0]["sql"], groups[1]["sql"])
            if not tags:
                continue
            counts["focused"] += 1
            gold_fp = record["gold_execution"]["fingerprint"]
            correct_index = next((i for i in range(2) if groups[i]["fingerprint"] == gold_fp), None)
            if correct_index is None:
                counts["gold_outside_top2"] += 1
                continue
            counts["leader_correct" if correct_index == 0 else "runner_up_correct"] += 1
            target = "<answer>" + groups[correct_index]["sql"] + "</answer>"
            orders = (
                [(0, 1), (1, 0)]
                if split == "train" and cfg["training"]["augment_swap"]
                else [(0, 1)]
            )
            for order in orders:
                pairs.append(
                    {
                        "id": digest([record["id"], order]),
                        "source_id": record["id"],
                        "source_split": split,
                        "db_id": record["db_id"],
                        "database_sha256": record["database_sha256"],
                        "tags": tags,
                        "leader_was_correct": correct_index == 0,
                        "messages": merge_messages(
                            record,
                            get_schema(cfg, record["db_id"]),
                            [groups[order[0]], groups[order[1]]],
                        ),
                        "target": target,
                    }
                )
        if not pairs:
            raise ValueError(
                f"No usable {split} JOIN/aggregation pairs. Increase the training limit or samples; "
                "final evaluation data must not be substituted."
            )
        rng.shuffle(pairs)
        write_jsonl(folder / f"{split}.jsonl", pairs)
        stats[split] = {**counts, "examples": len(pairs)}
        files[split] = file_hash(folder / f"{split}.jsonl")
    manifest = {
        "signature": digest(spec),
        "spec": spec,
        "files": files,
        "stats": stats,
        "database_hashes": {key: sorted(value) for key, value in databases.items()},
    }
    atomic_json(manifest_path, manifest)
    return {**manifest, "reused": False}


def tokenize_example(tokenizer, row: dict, max_length: int) -> dict | None:
    prefix = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        row["messages"] + [{"role": "assistant", "content": row["target"]}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full[: len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Tokenizer chat template does not preserve the assistant prefix.")
    if len(full) > max_length:
        return None  # Reported by ID; never truncate a target or schema silently.
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prefix) + full[len(prefix) :],
    }


def _pairs_for_training(cfg, *, verify_databases=True):
    folder = Path(cfg["experiment_dir"]) / "adaptation"
    manifest = read_json(folder / "manifest.json")
    _, data_manifest = load_split(cfg, "evaluation")
    if manifest["spec"]["prepared"] != data_manifest["signature"]:
        raise ValueError("Pair manifest differs from the prepared benchmark splits.")
    sets = []
    for split in ("train", "validation"):
        prepared, _ = load_split(cfg, split, verify_databases=verify_databases)
        prepared_ids = {r["id"] for r in prepared}
        path = folder / f"{split}.jsonl"
        if file_hash(path) != manifest["files"][split]:
            raise ValueError("Adaptation pairs were modified.")
        rows = read_jsonl(path)
        if not rows or any(r["source_split"] != split for r in rows):
            raise ValueError(
                "Adaptation pairs must be nonempty and tagged with the correct source split."
            )
        if any(r["source_id"] not in prepared_ids for r in rows):
            raise ValueError("A pair does not belong to its declared training/validation split.")
        sets.append(rows)
    if {r["database_sha256"] for r in sets[0]} & {r["database_sha256"] for r in sets[1]}:
        raise ValueError("Training and validation leakage detected.")
    return sets, manifest


@locked_stage("train_adapter")
def train_adapter(cfg: dict, seed=42) -> dict:
    datasets, pair_manifest = _pairs_for_training(cfg)
    # Establish the author's end-to-end baseline before spending GPU on an adapter.
    k = cfg["generation"]["evaluation_samples"]
    _, execution_manifest = load_execution(cfg, "evaluation", seed)
    _, author = completed_artifact(artifact_path(cfg, "evaluation", f"author.csc-k{k}", seed))
    if author["spec"]["execution"] != execution_manifest["sha256"]:
        raise ValueError("The author baseline belongs to different evaluation candidates.")
    base_identity = author.get("model_identity")
    if not base_identity:
        raise ValueError(
            "Author baseline made no CSC calls; establish a representative CSC baseline first."
        )
    if (
        author["spec"]["merger"] != cfg["merger"]
        or cfg["merger"]["model"] == cfg["generation"]["model"]
    ):
        raise ValueError(
            "Training must start from the configured author CSC merger, not the generation model."
        )
    root = Path(cfg["experiment_dir"]) / "adapter"
    metadata_path = root / "training_manifest.json"
    spec = {
        "training": cfg["training"],
        "base": base_identity,
        "pairs": pair_manifest["signature"],
        "files": pair_manifest["files"],
        "target_mode": "completion_only_answer",
        "version": 1,
    }
    if metadata_path.exists():
        previous = read_json(metadata_path)
        if previous["signature"] != digest(spec):
            raise ValueError(
                "Adapter training configuration or data changed. Choose a new experiment_dir."
            )
        if previous["status"] == "complete":
            _check_weights(root / "final", previous["adapter_files"])
            return {**previous, "reused": True}
    prior_packages = previous.get("packages") if metadata_path.exists() else None
    metadata = {
        "signature": digest(spec),
        "spec": spec,
        "status": "training",
        "base_identity": base_identity,
        "pair_manifest": pair_manifest,
    }
    if prior_packages:
        metadata["packages"] = prior_packages
    atomic_json(metadata_path, metadata)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CSC adapter training requires CUDA. No CPU fallback is started.")
    settings = cfg["training"]
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate")
    }
    if settings["quantization"] == "4bit":
        packages["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    if prior_packages and prior_packages != packages:
        raise ValueError(
            "Training package versions changed during resume; use the original environment."
        )
    metadata["packages"] = packages
    atomic_json(metadata_path, metadata)
    if base_identity["revision"] == "local":
        from research.slm_sql.backend import model_identity

        if model_identity(base_identity["model"], "local") != base_identity:
            raise ValueError("Local merger weights changed after baseline evaluation.")
    set_seed(settings["seed"])
    revision = None if base_identity["revision"] == "local" else base_identity["revision"]
    tokenizer = AutoTokenizer.from_pretrained(base_identity["model"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded, skipped = [], {}
    for split, rows in zip(("train", "validation"), datasets):
        items, dropped = [], []
        for row in rows:
            item = tokenize_example(tokenizer, row, settings["max_seq_len"])
            if item is None:
                dropped.append(row["id"])
            else:
                items.append(item)
        if not items:
            raise ValueError(f"All {split} examples exceed max_seq_len; increase context length.")
        encoded.append(items)
        skipped[split] = dropped
    metadata["skipped_context_ids"] = skipped
    metadata["used_examples"] = dict(zip(("train", "validation"), map(len, encoded)))
    atomic_json(metadata_path, metadata)
    quantization = None
    if settings["quantization"] == "4bit":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_identity["model"],
        revision=revision,
        torch_dtype=torch.float16,
        quantization_config=quantization,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=settings["rank"],
            lora_alpha=settings["alpha"],
            lora_dropout=settings["dropout"],
            target_modules=settings["target_modules"],
            bias="none",
        ),
    )
    args = TrainingArguments(
        output_dir=str(root / "checkpoints"),
        num_train_epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        warmup_ratio=settings["warmup_ratio"],
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=settings["seed"],
        data_seed=settings["seed"],
        optim="adamw_torch",
    )
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": encoded[0],
        "eval_dataset": encoded[1],
        "data_collator": DataCollatorForSeq2Seq(
            tokenizer, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100
        ),
    }
    tokenizer_argument = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer).parameters
        else "tokenizer"
    )
    trainer = Trainer(**kwargs, **{tokenizer_argument: tokenizer})
    checkpoint = (
        get_last_checkpoint(str(root / "checkpoints")) if (root / "checkpoints").exists() else None
    )
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(root / "final"))
    tokenizer.save_pretrained(root / "final")
    metadata.update(
        status="complete",
        metrics=result.metrics,
        best_validation_checkpoint=trainer.state.best_model_checkpoint,
        adapter_files=_weight_hashes(root / "final"),
    )
    atomic_json(metadata_path, metadata)
    return {**metadata, "reused": False}


def _weight_hashes(folder):
    paths = [
        p
        for p in Path(folder).iterdir()
        if p.name == "adapter_config.json" or p.name.startswith("adapter_model.")
    ]
    if not any(p.name.startswith("adapter_model.") for p in paths):
        raise ValueError("Adapter weights are missing.")
    return {p.name: file_hash(p) for p in paths}


def _check_weights(folder, expected):
    if _weight_hashes(folder) != expected:
        raise ValueError("Adapter weights or adapter configuration were modified.")


def validate_adapter(cfg, split, seed, k):
    _pairs_for_training(cfg, verify_databases=False)
    root = Path(cfg["experiment_dir"]) / "adapter"
    metadata = read_json(root / "training_manifest.json")
    if metadata["status"] != "complete" or metadata["spec"]["training"] != cfg["training"]:
        raise ValueError("Adapter training is incomplete or settings changed.")
    current_pairs = read_json(Path(cfg["experiment_dir"]) / "adaptation" / "manifest.json")
    if metadata["spec"]["pairs"] != current_pairs["signature"]:
        raise ValueError("Adapter was trained on different pairs.")
    _check_weights(root / "final", metadata["adapter_files"])
    _, baseline = completed_artifact(artifact_path(cfg, split, f"author.csc-k{k}", seed))
    if baseline.get("model_identity") != metadata["base_identity"]:
        raise ValueError("Author baseline and LoRA must have identical merger base weights.")
    adapter_config = read_json(root / "final" / "adapter_config.json")
    base = adapter_config.get("base_model_name_or_path")
    if base not in (cfg["merger"]["model"], metadata["base_identity"]["model"]):
        raise ValueError("This adapter belongs to a different base checkpoint.")
    return (
        str(root / "final"),
        {"signature": metadata["signature"], "files": metadata["adapter_files"]},
        metadata["base_identity"],
    )
