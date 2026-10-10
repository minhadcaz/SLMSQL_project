"""Load LoRA grid config and generate experiment configurations."""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoRAExperiment:
    name: str
    model_id: str
    rank: int
    target_modules: list[str]
    learning_rate: float
    rslora: bool
    lora_alpha: int
    lora_dropout: float = 0.0
    max_seq_len: int = 2048
    gradient_checkpointing: bool = True
    dtype: str = "float16"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42
    quantization: str = "none"


def load_grid(config_path: str | Path = "configs/lora_grid.yaml") -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_experiments(
    config_path: str | Path = "configs/lora_grid.yaml",
    model_key: str = "small",
) -> list[LoRAExperiment]:
    """Generate all LoRA experiment configs from the grid YAML."""
    cfg = load_grid(config_path)
    grid = cfg["grid"]
    fixed = cfg["fixed"]
    model_id = cfg["models"][model_key]
    quant = cfg["quantization"].get(model_key, "none")

    experiments = []
    for rank, (mod_name, mod_list), lr, rslora in itertools.product(
        grid["ranks"],
        grid["target_modules"].items(),
        grid["learning_rates"],
        grid["rslora"],
    ):
        name = f"r{rank}_{mod_name}_lr{lr}_rslora{rslora}"
        exp = LoRAExperiment(
            name=name,
            model_id=model_id,
            rank=rank,
            target_modules=mod_list,
            learning_rate=lr,
            rslora=rslora,
            lora_alpha=rank * fixed["lora_alpha_ratio"],
            lora_dropout=fixed["lora_dropout"],
            max_seq_len=fixed["max_seq_len"],
            gradient_checkpointing=fixed["gradient_checkpointing"],
            dtype=fixed["dtype"],
            warmup_ratio=fixed["warmup_ratio"],
            weight_decay=fixed["weight_decay"],
            num_train_epochs=fixed["num_train_epochs"],
            per_device_train_batch_size=fixed["per_device_train_batch_size"],
            gradient_accumulation_steps=fixed["gradient_accumulation_steps"],
            save_steps=fixed["save_steps"],
            logging_steps=fixed["logging_steps"],
            seed=fixed["seed"],
            quantization=quant,
        )
        experiments.append(exp)

    return experiments
