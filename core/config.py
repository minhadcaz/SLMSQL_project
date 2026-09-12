"""Load and validate project configuration."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return deepcopy(config)


def validate_config(config: dict) -> None:
    sample = config["sampling"]
    if sample["mode"] not in {"greedy", "fixed_sc", "adaptive_sc"}:
        raise ValueError("Chế độ sinh SQL không hợp lệ.")
    for key in ("fixed_k", "min_valid_candidates", "max_candidates"):
        if not 1 <= sample[key] <= 128:
            raise ValueError(f"sampling.{key} phải từ 1 đến 128.")
    if not sample["rounds"] or any(not isinstance(n, int) or n < 1 for n in sample["rounds"]):
        raise ValueError("rounds phải là danh sách số nguyên dương.")
    for key in ("confidence_threshold", "margin_threshold"):
        if not 0 <= sample[key] <= 1:
            raise ValueError(f"{key} phải nằm trong [0, 1].")
    if sample["time_budget_seconds"] <= 0:
        raise ValueError("Time budget phải lớn hơn 0.")
    gen = config["generation"]
    if gen["temperature"] < 0 or not 0 < gen["top_p"] <= 1:
        raise ValueError("temperature/top_p không hợp lệ.")
    if any(gen[k] <= 0 for k in ("batch_size", "max_new_tokens", "max_input_tokens")):
        raise ValueError("Giới hạn generation phải dương.")
    if any(v <= 0 for v in config["execution"].values()):
        raise ValueError("Giới hạn execution phải dương.")
    if config["correction"]["per_candidate"] not in (0, 1, 2):
        raise ValueError("Chỉ hỗ trợ tối đa 2 vòng correction.")
    if config["correction"]["total_budget"] < 0:
        raise ValueError("Correction budget không được âm.")
    if config["model"]["backend"] not in {"mock", "huggingface"}:
        raise ValueError("Backend không hợp lệ.")
    if config["model"]["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device không hợp lệ.")
    if str(config["model"]["quantization"]) not in {"none", "4bit", "8bit"}:
        raise ValueError("Quantization không hợp lệ.")
