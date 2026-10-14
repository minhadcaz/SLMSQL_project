"""Offline policy simulator: replay candidates with random permutations.

Input: candidates JSONL with execution fingerprints (from exec_runner).
For each policy, replay permutations and report EX and candidate costs.
"""

import json
import random
import math
from pathlib import Path
from typing import Any, Protocol


class StoppingPolicy(Protocol):
    """Interface for all stopping policies."""
    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        """Return (should_stop, action) where action is 'STOP', 'SAMPLE_MORE', or 'CALL_CSC'."""
        ...

    def select(self, candidates: list[dict]) -> str | None:
        """Select the best SQL from candidates. Returns fingerprint or None."""
        ...


def _numeric_candidate_value(candidate: dict[str, Any], names: tuple[str, ...]) -> float | None:
    """Return the first non-negative numeric candidate metadata value."""
    for name in names:
        value = candidate.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    return None


def _candidate_token_count(candidate: dict[str, Any]) -> float:
    """Count generated tokens; fall back to a word count for old artifacts."""
    value = _numeric_candidate_value(
        candidate, ("n_tokens", "token_count", "output_tokens", "num_tokens")
    )
    if value is not None:
        return value

    for name in ("token_ids", "tokens"):
        value = candidate.get(name)
        if isinstance(value, (list, tuple)):
            return float(len(value))

    text = candidate.get("raw_text") or candidate.get("sql") or ""
    return float(len(str(text).split()))


def _candidate_seconds(candidate: dict[str, Any]) -> tuple[float | None, str]:
    """Use generation timing if available; use SQL execution as a lower bound."""
    generation = _numeric_candidate_value(
        candidate,
        (
            "generation_seconds", "generation_time_s", "generation_time",
            "latency_seconds", "elapsed_seconds", "time_seconds", "seconds",
        ),
    )
    if generation is None:
        generation_ms = _numeric_candidate_value(
            candidate, ("generation_ms", "generation_time_ms", "latency_ms")
        )
        if generation_ms is not None:
            generation = generation_ms / 1000.0

    if generation is not None:
        return generation, "generation_only"

    execution_ms = _numeric_candidate_value(candidate, ("elapsed_ms",))
    if execution_ms is not None:
        return execution_ms / 1000.0, "execution_lower_bound"
    return None, "unavailable"


def simulate_one(
    record: dict[str, Any],
    policy: StoppingPolicy,
    gold_fp: str | None,
    permutation: list[int],
    batch_size: int = 4,
) -> dict[str, Any]:
    """Simulate a policy on one question with a given candidate permutation.

    Returns correctness, used candidates, cost metadata, and selected action.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    candidates = record["candidates"]
    seen: list[dict] = []
    n_used = 0
    stop = False
    action = "SAMPLE_MORE"

    for start in range(0, len(permutation), batch_size):
        batch_indices = permutation[start:start + batch_size]
        for idx in batch_indices:
            seen.append(candidates[idx])
            n_used += 1

        stop, action = policy.should_stop(seen, n_used)
        if stop or n_used >= len(permutation):
            break

    timing = [_candidate_seconds(candidate) for candidate in seen]
    measured = [seconds or 0.0 for seconds, _ in timing]
    seconds_used = sum(measured)
    timing_sources = {source for _, source in timing}
    seconds_basis = (timing_sources.pop() if len(timing_sources) == 1 else "mixed")

    selected_fp = policy.select(seen)
    correct = selected_fp == gold_fp if gold_fp and selected_fp else False

    return {
        "correct": correct,
        "n_samples_used": n_used,
        "n_tokens_used": sum(_candidate_token_count(c) for c in seen),
        "seconds_used": seconds_used,
        "seconds_basis": seconds_basis,
        "action": action,
        "csc_called": action == "CALL_CSC",
    }


def simulate_policy(
    records: list[dict[str, Any]],
    policy: StoppingPolicy,
    db_path_fn,
    n_permutations: int = 100,
    batch_size: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full simulation across all questions with multiple permutations."""
    from research.exec_runner import _exec_one

    if n_permutations < 1 or batch_size < 1:
        raise ValueError("n_permutations and batch_size must be positive.")
    rng = random.Random(seed)
    all_correct = []
    all_samples = []
    all_tokens = []
    all_seconds = []
    seconds_bases: set[str] = set()
    total_csc_calls = 0

    for record in records:
        gold_sql = record.get("gold_sql", "")
        if not gold_sql:
            continue
        gold_result = _exec_one(gold_sql, str(db_path_fn(record["db_id"])))
        gold_fp = gold_result["fingerprint"]

        n_cands = len(record["candidates"])
        question_correct = 0
        question_samples = 0
        question_tokens = 0.0
        question_seconds = 0.0
        for _ in range(n_permutations):
            perm = list(range(n_cands))
            rng.shuffle(perm)
            result = simulate_one(record, policy, gold_fp, perm, batch_size)
            question_correct += int(result["correct"])
            question_samples += result["n_samples_used"]
            question_tokens += result["n_tokens_used"]
            question_seconds += result["seconds_used"]
            seconds_bases.add(result["seconds_basis"])
            total_csc_calls += int(result["csc_called"])

        all_correct.append(question_correct / n_permutations)
        all_samples.append(question_samples / n_permutations)
        all_tokens.append(question_tokens / n_permutations)
        all_seconds.append(question_seconds / n_permutations)

    n = len(all_correct)
    if n == 0:
        return {
            "ex_mean": 0.0, "ex_ci95": 0.0, "avg_samples": 0.0,
            "avg_tokens": 0.0, "avg_seconds": 0.0, "seconds_basis": "unavailable",
            "csc_calls": 0, "n_questions": 0, "n_permutations": n_permutations,
            "seed": seed, "batch_size": batch_size,
        }

    ex_mean = sum(all_correct) / n
    ex_std = math.sqrt(sum((x - ex_mean) ** 2 for x in all_correct) / n)
    ex_ci95 = 1.96 * ex_std / math.sqrt(n)

    avg_samples = sum(all_samples) / n
    avg_tokens = sum(all_tokens) / n
    avg_seconds = sum(all_seconds) / n
    seconds_basis = next(iter(seconds_bases)) if len(seconds_bases) == 1 else "mixed"

    return {
        "ex_mean": round(ex_mean, 4),
        "ex_ci95": round(ex_ci95, 4),
        "avg_samples": round(avg_samples, 2),
        "avg_tokens": round(avg_tokens, 2),
        "avg_seconds": round(avg_seconds, 6),
        "seconds_basis": seconds_basis,
        "seed": seed,
        "batch_size": batch_size,
        "csc_calls": total_csc_calls,
        "n_questions": n,
        "n_permutations": n_permutations,
    }


def simulate_multiple_policies(
    records: list[dict[str, Any]],
    policies: dict[str, StoppingPolicy],
    db_path_fn,
    n_permutations: int = 100,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Compare multiple policies on the same dataset."""
    results = {}
    for name, policy in policies.items():
        results[name] = simulate_policy(records, policy, db_path_fn, n_permutations, seed=seed)
    return results


def main() -> None:
    import argparse
    import logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec-results", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    from research.candidate_generator import load_candidates

    records = load_candidates(args.exec_results)
    from benchmarks.bird_loader import get_db_path

    from research.policies.fixed_k import FixedKPolicy

    policies = {
        f"fixed_{k}": FixedKPolicy(k) for k in [4, 8, 16, 32]
    }

    results = simulate_multiple_policies(
        records, policies,
        lambda db_id: get_db_path(args.data_dir, db_id),
        args.n_permutations, args.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
