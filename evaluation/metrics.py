"""Evaluation metrics with explicit denominators and no truncated EX comparisons."""
from collections import Counter
import statistics

from core.result_normalizer import requires_order


def execution_match(prediction, gold, gold_sql: str) -> bool:
    if not prediction or not prediction.success or not gold.success:
        return False
    if requires_order(gold_sql):
        return prediction.ordered_result == gold.ordered_result
    return prediction.unordered_result == gold.unordered_result


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(records: list[dict]) -> dict:
    count = len(records)
    latency = [record["elapsed_ms"] for record in records]
    candidates = sum(record["candidate_count"] for record in records)
    attempts = sum(record["correction_attempts"] for record in records)
    return {
        "questions": count,
        "execution_accuracy": sum(r["correct"] for r in records) / count if count else 0,
        "exact_match": sum(r["exact_match"] for r in records) / count if count else 0,
        "valid_sql_rate": sum(r["valid_count"] for r in records) / candidates if candidates else 0,
        "average_candidates": candidates / count if count else 0,
        "average_latency_ms": statistics.mean(latency) if latency else 0,
        "p50_latency_ms": percentile(latency, 0.5),
        "p95_latency_ms": percentile(latency, 0.95),
        "early_stop_rate": sum(r["early_stop"] for r in records) / count if count else 0,
        "correction_attempts": attempts,
        "correction_success_rate": sum(r["correction_successes"] for r in records) / attempts if attempts else 0,
        "average_total_generations": (candidates + attempts) / count if count else 0,
        "error_breakdown": dict(Counter(r["error_type"] for r in records if r["error_type"])),
    }
