"""Author-style sampling sweep for the canonical SLM-SQL workflow.

The sweep keeps one stochastic generator cache and reports, on the same paired
questions, greedy, pass@k, execution-group voting (SC/SLM-SQL@k), author CSC@k,
LoRA CSC@k when available, and an execution-aware adaptive generator run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.final_eval import bootstrap_mean_ci
from research.slm_sql import PROTOCOL_VERSION
from research.slm_sql.artifacts import atomic_json, completed_artifact
from research.slm_sql.config import artifact_path
from research.slm_sql.execution import EXECUTION_VERSION
from research.slm_sql.pipeline import load_adaptive, load_execution
from research.slm_sql.prompts import PROMPT_VERSION, top_groups


def _accuracy(flags: list[bool], bootstrap_samples: int, seed: int) -> dict[str, Any]:
    values = [bool(flag) for flag in flags]
    return {
        "ex": sum(values) / len(values) if values else 0.0,
        "questions": len(values),
        "accuracy_ci95": bootstrap_mean_ci(values, bootstrap_samples, seed) if values else None,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _load_csc_if_present(cfg, split, seed, variant, k):
    path = artifact_path(cfg, split, f"{variant}.csc-k{k}", seed)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.exists() or not manifest_path.exists():
        return None, None
    return completed_artifact(path)


def _check_ids(expected: list[dict], actual: list[dict], label: str) -> None:
    if [row["id"] for row in expected] != [row["id"] for row in actual]:
        raise ValueError(f"{label} does not contain exactly the prepared questions in order.")


def evaluate_suite(cfg: dict, split: str = "evaluation", seed: int = 42, *, k: int | None = None) -> dict:
    """Score all available author-style sampling variants without loading a model."""
    generated, generation_manifest = load_execution(cfg, split, seed)
    greedy, greedy_manifest = load_execution(cfg, split, seed, mode="greedy")
    _check_ids(generated, greedy, "Greedy cache")
    adaptive = None
    adaptive_manifest = None
    adaptive_path = artifact_path(cfg, split, "adaptive", seed)
    if adaptive_path.exists() and adaptive_path.with_suffix(adaptive_path.suffix + ".manifest.json").exists():
        adaptive, adaptive_manifest = load_adaptive(cfg, split, seed)
        _check_ids(generated, adaptive, "Adaptive cache")

    configured = list(cfg["evaluation"]["sampling_k_values"])
    k_values = [k] if k is not None else configured
    if not k_values or any(k_value not in configured for k_value in k_values):
        raise ValueError("Requested k must be one of evaluation.sampling_k_values.")
    max_cached = cfg["generation"][f"{split}_samples"]
    if any(k_value > max_cached for k_value in k_values):
        raise ValueError("Requested k exceeds the cached stochastic generator samples.")

    bootstrap_samples = cfg["evaluation"]["bootstrap_samples"]
    records = []
    fixed_summary = {}
    for k_value in k_values:
        author_csc, author_manifest = _load_csc_if_present(cfg, split, seed, "author", k_value)
        lora_csc, lora_manifest = _load_csc_if_present(cfg, split, seed, "lora", k_value)
        if author_csc is not None:
            _check_ids(generated, author_csc, f"Author CSC@{k_value}")
        if lora_csc is not None:
            _check_ids(generated, lora_csc, f"LoRA CSC@{k_value}")

        rows = []
        for index, record in enumerate(generated):
            gold_fp = record["gold_execution"].get("fingerprint")
            if gold_fp is None:
                raise ValueError(f"Gold execution is missing for {record['id']}.")
            groups = top_groups(record["candidates"], k_value)
            sc_fp = groups[0]["fingerprint"] if groups else None
            row = {
                "id": record["id"],
                "db_id": record["db_id"],
                "gold_fp": gold_fp,
                "pass@k": gold_fp in [group["fingerprint"] for group in groups],
                "sc@k": sc_fp == gold_fp,
                "slm_sql@k": sc_fp == gold_fp,
                "top2_coverage@k": gold_fp in [group["fingerprint"] for group in groups[:2]],
                "n_samples": k_value,
                "n_tokens": sum(candidate.get("n_tokens", 0) for candidate in record["candidates"][:k_value]),
                "author_csc@k": None,
                "lora_csc@k": None,
            }
            if author_csc is not None:
                row["author_csc@k"] = author_csc[index]["selected_fp"] == gold_fp
            if lora_csc is not None:
                row["lora_csc@k"] = lora_csc[index]["selected_fp"] == gold_fp
            rows.append(row)

        metrics = {
            "pass@k": _accuracy([row["pass@k"] for row in rows], bootstrap_samples, seed),
            "sc@k": _accuracy([row["sc@k"] for row in rows], bootstrap_samples, seed),
            "slm_sql@k": _accuracy([row["slm_sql@k"] for row in rows], bootstrap_samples, seed),
            "top2_coverage@k": sum(row["top2_coverage@k"] for row in rows) / len(rows),
            "mean_samples": k_value,
            "mean_tokens": _mean([row["n_tokens"] for row in rows]),
            "author_csc@k": None,
            "lora_csc@k": None,
            "author_csc_calls": None,
            "lora_csc_calls": None,
        }
        if author_csc is not None:
            metrics["author_csc@k"] = _accuracy(
                [row["author_csc@k"] for row in rows], bootstrap_samples, seed
            )
            metrics["author_csc_calls"] = sum(bool(row["csc_called"]) for row in author_csc)
        if lora_csc is not None:
            metrics["lora_csc@k"] = _accuracy(
                [row["lora_csc@k"] for row in rows], bootstrap_samples, seed
            )
            metrics["lora_csc_calls"] = sum(bool(row["csc_called"]) for row in lora_csc)
        fixed_summary[str(k_value)] = metrics
        for row in rows:
            records.append({**row, "k": k_value})

    greedy_flags = []
    for index, row in enumerate(greedy):
        candidates = row.get("candidates", [])
        if len(candidates) != 1:
            raise ValueError("Greedy cache must contain exactly one candidate per question.")
        greedy_flags.append(candidates[0]["execution"].get("fingerprint") == row["gold_execution"].get("fingerprint"))
    greedy_summary = _accuracy(greedy_flags, bootstrap_samples, seed)
    greedy_summary.update({"mean_samples": 1, "mean_tokens": _mean([r["candidates"][0].get("n_tokens", 0) for r in greedy])})

    adaptive_summary = None
    if adaptive is not None:
        flags = [bool(row["correct"]) for row in adaptive]
        adaptive_summary = _accuracy(flags, bootstrap_samples, seed)
        adaptive_summary.update(
            {
                "policy": cfg["adaptive"]["policy"],
                "mean_samples": _mean([row["n_samples_used"] for row in adaptive]),
                "mean_tokens": _mean(
                    [sum(candidate.get("n_tokens", 0) for candidate in row["candidates"]) for row in adaptive]
                ),
                "mean_generation_seconds": _mean(
                    [row.get("amortized_generation_s", 0.0) for row in adaptive]
                ),
                "actions": {
                    action: sum(row.get("adaptive_action") == action for row in adaptive)
                    for action in ("STOP", "SAMPLE_MORE", "CALL_CSC")
                },
                "accuracy_basis": "execution_aware_online_generation",
            }
        )

    provenance = {
        "generation": generation_manifest,
        "greedy": greedy_manifest,
        "adaptive": adaptive_manifest,
        "csc": {},
    }
    for k_value in k_values:
        for variant in ("author", "lora"):
            _, manifest = _load_csc_if_present(cfg, split, seed, variant, k_value)
            if manifest is not None:
                provenance["csc"][f"{variant}@{k_value}"] = manifest

    report = {
        "protocol": PROTOCOL_VERSION,
        "evaluator": EXECUTION_VERSION,
        "prompt": PROMPT_VERSION,
        "split": split,
        "seed": seed,
        "k_values": k_values,
        "summary": {"greedy": greedy_summary, "fixed": fixed_summary, "adaptive": adaptive_summary},
        "provenance": provenance,
        "records": records,
        "notes": [
            "pass@k is the oracle metric: the gold execution group appears in the first k cached drafts.",
            "SC@k and SLM-SQL@k are execution-group majority voting with first-generated tie breaking.",
            "CSC@k is reported only when the corresponding author or LoRA merger artifact exists.",
            "Adaptive is generated online in execution-aware rounds; it is not an offline permutation proxy.",
            "The SLM-SQL paper's SLM-SQL@k terminology is kept alongside the SC@k alias; no new SFT run is implied.",
        ],
    }
    path = Path(cfg["experiment_dir"]) / "reports" / f"{split}-seed{seed}.sampling.json"
    atomic_json(path, report)
    return {"path": str(path), "summary": report["summary"]}
