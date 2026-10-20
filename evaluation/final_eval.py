"""Reusable final-evaluation utilities with explicit provenance."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def sha256_path(path: str | Path) -> str:
    """Return a content hash for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(
    values: Iterable[float | bool],
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Estimate a percentile bootstrap CI for a mean with a fixed seed."""
    samples = [float(value) for value in values]
    if not samples:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "n": 0, "seed": seed}
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence must be in (0, 1).")
    rng = random.Random(seed)
    means = [
        sum(rng.choice(samples) for _ in samples) / len(samples)
        for _ in range(n_bootstrap)
    ]
    means.sort()
    alpha = (1.0 - confidence) / 2.0

    def percentile(p: float) -> float:
        index = (len(means) - 1) * p
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return means[low]
        return means[low] + (means[high] - means[low]) * (index - low)

    return {
        "mean": sum(samples) / len(samples),
        "lower": percentile(alpha),
        "upper": percentile(1.0 - alpha),
        "n": len(samples),
        "seed": seed,
    }


def mcnemar_test(
    adaptive_correct: Iterable[bool],
    fixed_correct: Iterable[bool],
) -> dict[str, float | int | None]:
    """Return paired discordant counts and an exact two-sided p-value."""
    adaptive = [bool(value) for value in adaptive_correct]
    fixed = [bool(value) for value in fixed_correct]
    if len(adaptive) != len(fixed):
        raise ValueError("McNemar inputs must have equal length.")
    b = sum(a and not f for a, f in zip(adaptive, fixed))
    c = sum(f and not a for a, f in zip(adaptive, fixed))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(b, c)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "n": len(adaptive),
        "adaptive_only": b,
        "fixed_only": c,
        "discordant": discordant,
        "exact_p_value": p_value,
    }


def _sql_features(sql: str) -> dict[str, set[str] | bool]:
    """Extract coarse post-hoc SQL features for error labeling."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="sqlite")
    tables = {str(node.name).lower() for node in tree.find_all(exp.Table)}
    functions = {
        str(getattr(node, "key", type(node).__name__)).lower()
        for node in tree.find_all(exp.Func)
    }
    aggregates = {
        "count", "sum", "avg", "min", "max", "groupconcat", "arrayagg",
    } & functions
    literals = {str(node.this).lower() for node in tree.find_all(exp.Literal)}
    return {
        "tables": tables,
        "joins": any(tree.find_all(exp.Join)),
        "aggregates": aggregates,
        "literals": literals,
    }


def classify_error(
    question: str,
    gold_sql: str,
    predicted_sql: str | None,
    execution_error: str | None = None,
) -> str:
    """Assign one transparent, post-hoc error class to a failed prediction."""
    del question  # Kept in the signature for future language-aware classifiers.
    if execution_error:
        return "execution"
    if not predicted_sql or not predicted_sql.strip():
        return "format"
    try:
        gold = _sql_features(gold_sql)
        predicted = _sql_features(predicted_sql)
    except Exception:
        return "format"
    if gold["tables"] - predicted["tables"]:
        return "schema_linking"
    if bool(gold["joins"]) and not bool(predicted["joins"]):
        return "join"
    if gold["aggregates"] and not gold["aggregates"].issubset(predicted["aggregates"]):
        return "aggregate"
    if gold["literals"] != predicted["literals"]:
        return "value"
    return "ambiguity"


def load_json(path: str | Path) -> Any:
    """Load JSON UTF-8 data."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with stable UTF-8 formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_pipeline_records(
    records: list[dict[str, Any]],
    db_path_fn: Callable[[str], str | Path],
    config: dict[str, Any],
    *,
    variant_name: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the local Pipeline and save per-question results when requested."""
    from core.model_backend import create_backend
    from core.pipeline import Pipeline
    from core.sql_executor import SQLExecutor
    from evaluation.metrics import execution_match

    backend = create_backend(config)
    pipeline = Pipeline(config, backend)
    executor = SQLExecutor(config["execution"])
    results: list[dict[str, Any]] = []
    for record in records:
        db_path = db_path_fn(record["db_id"])
        gold = executor.execute(db_path, record["gold_sql"])
        if not gold.success:
            raise ValueError(f"Gold SQL failed for {record.get('id', record['question'])}.")
        prediction = pipeline.run(db_path, record["question"], record.get("evidence", ""))
        selected = prediction.selected
        selected_execution = selected.execution if selected else None
        correct = execution_match(selected_execution, gold, record["gold_sql"])
        execution_error = None
        if selected and selected.execution and not selected.execution.success:
            execution_error = selected.execution.error_type or selected.execution.error
        results.append({
            "id": record.get("id", f"{record['db_id']}::{record['question']}"),
            "question": record["question"],
            "db_id": record["db_id"],
            "difficulty": record.get("difficulty", "unknown"),
            "language": record.get("language", "unknown"),
            "gold_sql": record["gold_sql"],
            "predicted_sql": selected.sql if selected else None,
            "correct": bool(correct),
            "error_type": None if correct else classify_error(
                record["question"], record["gold_sql"],
                selected.sql if selected else None, execution_error,
            ),
            "candidate_count": len(prediction.candidates),
            "valid_count": sum(
                bool(item.execution and item.execution.success) for item in prediction.candidates
            ),
            "elapsed_ms": prediction.elapsed_ms,
            "decision": prediction.decision.reason,
            "prediction": prediction.public_dict(),
        })
    flags = [item["correct"] for item in results]
    summary = {
        "variant": variant_name,
        "questions": len(results),
        "execution_accuracy": sum(flags) / len(flags) if flags else 0.0,
        "accuracy_bootstrap_95": bootstrap_mean_ci(flags, seed=config["seed"]),
        "average_candidates": sum(item["candidate_count"] for item in results) / len(results) if results else 0.0,
        "average_latency_ms": sum(item["elapsed_ms"] for item in results) / len(results) if results else 0.0,
        "seed": config["seed"],
        "config": config,
        "records": results,
    }
    if output_path is not None:
        write_json(output_path, summary)
    return summary


def paired_report_test(adaptive: dict, fixed: dict) -> dict:
    """Refuse intersections, duplicates and mismatched benchmark questions."""
    left, right = adaptive.get("records", []), fixed.get("records", [])
    left_ids, right_ids = [r["id"] for r in left], [r["id"] for r in right]
    if not left or len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        raise ValueError("Paired reports need nonempty unique question IDs.")
    if set(left_ids) != set(right_ids) or adaptive.get("seed") != fixed.get("seed"):
        raise ValueError("Paired reports must contain exactly the same questions and seed.")
    left_protocol = adaptive.get("provenance", {}).get("protocol_signature")
    right_protocol = fixed.get("provenance", {}).get("protocol_signature")
    if left_protocol or right_protocol:
        if left_protocol != right_protocol:
            raise ValueError("Paired reports must come from the same prospective protocol.")
    left_model = adaptive.get("config", {}).get("model", {})
    right_model = fixed.get("config", {}).get("model", {})
    for key in ("backend", "model_id", "device", "dtype", "quantization"):
        if key in left_model or key in right_model:
            if left_model.get(key) != right_model.get(key):
                raise ValueError("Paired reports must use the same model configuration.")
    lookup = {r["id"]: r for r in right}
    paired = [(a, lookup[a["id"]]) for a in left]
    for a, b in paired:
        if any(a.get(key) != b.get(key) for key in ("db_id", "question", "gold_sql")):
            raise ValueError("Paired question content differs.")
    differences = [int(a["correct"]) - int(b["correct"]) for a, b in paired]
    return {
        "paired_ids": left_ids,
        "mcnemar": mcnemar_test([a["correct"] for a, _ in paired], [b["correct"] for _, b in paired]),
        "paired_delta_ci95": bootstrap_mean_ci(differences),
        "adaptive_avg_samples": adaptive.get("average_candidates"),
        "fixed_avg_samples": fixed.get("average_candidates"),
        "budget_note": "Report observed costs; a fixed-16 comparator alone does not establish equal compute budgets.",
    }
