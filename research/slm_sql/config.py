"""Validated experiment configuration; loading this module never loads a model."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "csc_experiment.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    import yaml

    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    path = Path(path)
    text = os.path.expandvars(path.read_text(encoding="utf-8"))
    overrides = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(overrides, dict):
        raise ValueError("Experiment config must be a mapping.")

    def merge(base, changes, prefix=""):
        result = copy.deepcopy(base)
        for key, value in changes.items():
            if key not in base:
                raise ValueError(f"Unknown config key: {prefix}{key}")
            if isinstance(base[key], dict):
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a mapping: {prefix}{key}")
                result[key] = merge(base[key], value, f"{prefix}{key}.")
            else:
                result[key] = value
        return result

    cfg = merge(defaults, overrides)
    if "sampling_k_values" not in overrides.get("evaluation", {}):
        cfg["evaluation"]["sampling_k_values"] = [
            k for k in cfg["evaluation"]["sampling_k_values"]
            if k <= cfg["generation"]["evaluation_samples"]
        ] or [cfg["generation"]["evaluation_samples"]]
    if "checkpoints" not in overrides.get("adaptive", {}):
        cfg["adaptive"]["checkpoints"] = [
            k for k in cfg["adaptive"].get("checkpoints", [])
            if k <= cfg["generation"]["evaluation_samples"]
        ] or [cfg["generation"]["evaluation_samples"]]
    if "max_k" not in overrides.get("adaptive", {}):
        cfg["adaptive"]["max_k"] = min(
            cfg["adaptive"]["max_k"], cfg["generation"]["evaluation_samples"]
        )
    if "min_k" not in overrides.get("adaptive", {}):
        cfg["adaptive"]["min_k"] = min(cfg["adaptive"]["min_k"], cfg["adaptive"]["max_k"])
    for section, keys in {
        "generation": [
            "train_samples",
            "validation_samples",
            "evaluation_samples",
            "max_tokens",
            "max_model_len",
            "batch_size",
        ],
        "merger": ["samples", "max_tokens", "max_model_len", "batch_size", "result_preview_rows"],
        "execution": ["timeout", "max_rows"],
        "training": [
            "rank",
            "alpha",
            "epochs",
            "max_seq_len",
            "batch_size",
            "gradient_accumulation_steps",
        ],
        "evaluation": ["bootstrap_samples"],
    }.items():
        for key in keys:
            value = cfg[section][key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{section}.{key} must be a positive integer.")
    for key in (
        "train_limit",
        "validation_limit",
        "evaluation_limit",
        "schema_sample_rows",
        "seed",
    ):
        value = cfg["data"][key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"data.{key} must be a nonnegative integer.")
    if not 0 < cfg["data"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    for section in ("generation", "merger"):
        values = cfg[section]
        if values["max_tokens"] >= values["max_model_len"]:
            raise ValueError(f"{section}: output tokens must be smaller than the context window.")
        if not 0 < values["top_p"] <= 1 or not 0 < values["temperature"] <= 2:
            raise ValueError(f"{section}: sampling needs temperature > 0 and top_p in (0, 1].")
        if not 0 < values["gpu_memory_utilization"] < 1:
            raise ValueError(f"{section}: gpu_memory_utilization must be in (0, 1).")
        if not values["model"] or not values["revision"]:
            raise ValueError(f"{section}: model and revision are required.")
    if cfg["training"]["quantization"] not in ("none", "4bit"):
        raise ValueError("training.quantization must be none or 4bit.")
    if not 0 <= cfg["training"]["dropout"] < 1 or cfg["training"]["learning_rate"] <= 0:
        raise ValueError("Invalid LoRA dropout or learning rate.")
    if cfg["training"]["rank"] not in (8, 16, 32, 64, 128, 256):
        raise ValueError("LoRA rank must be supported by vLLM: 8/16/32/64/128/256.")
    if not 0 <= cfg["training"]["warmup_ratio"] < 1:
        raise ValueError("training.warmup_ratio must be in [0, 1).")
    if not isinstance(cfg["training"]["seed"], int) or cfg["training"]["seed"] < 0:
        raise ValueError("training.seed must be a nonnegative integer.")
    if cfg["generation"]["model"] == cfg["merger"]["model"]:
        raise ValueError("Generation and CSC must use their respective author checkpoints.")
    k_values = cfg["evaluation"].get("sampling_k_values")
    if (
        not isinstance(k_values, list)
        or not k_values
        or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in k_values)
        or len(k_values) != len(set(k_values))
    ):
        raise ValueError("evaluation.sampling_k_values must be unique positive integers.")
    if max(k_values) > cfg["generation"]["evaluation_samples"]:
        raise ValueError("evaluation.sampling_k_values cannot exceed evaluation_samples.")
    adaptive = cfg["adaptive"]
    if adaptive["policy"] not in ("beta", "margin", "router"):
        raise ValueError("adaptive.policy must be beta, margin or router.")
    checkpoints = adaptive.get("checkpoints")
    if (not isinstance(checkpoints, list) or not checkpoints
        or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in checkpoints)
        or len(checkpoints) != len(set(checkpoints))):
        raise ValueError("adaptive.checkpoints must be unique positive integers.")
    if max(checkpoints) > cfg["generation"]["evaluation_samples"]:
        raise ValueError("adaptive.checkpoints cannot exceed evaluation_samples.")
    for key in ("min_k", "max_k", "batch_size", "margin"):
        value = adaptive[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"adaptive.{key} must be a positive integer.")
    if adaptive["min_k"] > adaptive["max_k"]:
        raise ValueError("adaptive.min_k cannot exceed adaptive.max_k.")
    if adaptive["max_k"] > cfg["generation"]["evaluation_samples"]:
        raise ValueError("adaptive.max_k cannot exceed evaluation_samples.")
    if (not isinstance(adaptive["enabled"], bool)
        or not isinstance(adaptive["threshold"], (int, float))
        or isinstance(adaptive["threshold"], bool)
        or not 0 < adaptive["threshold"] <= 1):
        raise ValueError("adaptive.enabled must be boolean and adaptive.threshold must be in (0, 1].")
    if (not isinstance(adaptive.get("stop_threshold", 0.9), (int, float))
        or isinstance(adaptive.get("stop_threshold", 0.9), bool)
        or not 0 < adaptive.get("stop_threshold", 0.9) <= 1
        or not isinstance(adaptive.get("csc_threshold", 0.7), (int, float))
        or isinstance(adaptive.get("csc_threshold", 0.7), bool)
        or not 0 < adaptive.get("csc_threshold", 0.7) <= 1
        or not isinstance(adaptive.get("call_csc_at_budget", False), bool)):
        raise ValueError("adaptive router thresholds must be in (0, 1] and call_csc_at_budget must be boolean.")
    if (not isinstance(adaptive["entropy_threshold"], (int, float))
        or isinstance(adaptive["entropy_threshold"], bool)
        or not 0 <= adaptive["entropy_threshold"] < float("inf")):
        raise ValueError("adaptive.entropy_threshold must be finite and nonnegative.")
    if not cfg["training"]["target_modules"] or not isinstance(
        cfg["training"]["augment_swap"], bool
    ):
        raise ValueError("LoRA target_modules and boolean augment_swap are required.")
    for key in ("experiment_dir",):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    for key in ("train_dir", "evaluation_dir"):
        cfg["data"][key] = str(Path(cfg["data"][key]).expanduser().resolve())
    return cfg


def artifact_path(cfg: dict, split: str, stage: str, seed: int) -> Path:
    if split not in ("train", "validation", "evaluation"):
        raise ValueError(f"Unknown split: {split}")
    return Path(cfg["experiment_dir"]) / f"seed{seed}" / f"{split}.{stage}.jsonl"


def merge_artifact_path(cfg, split, seed, variant, k=None, *, source="fixed"):
    if source not in ("fixed", "adaptive"):
        raise ValueError("CSC source must be fixed or adaptive.")
    stage = f"{variant}.csc-k{k}" if source == "fixed" else f"{variant}.adaptive-csc"
    return artifact_path(cfg, split, stage, seed)
