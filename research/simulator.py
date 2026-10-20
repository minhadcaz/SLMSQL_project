"""Offline policy simulator: replay candidates with random permutations.

Input: candidates JSONL with execution fingerprints (from exec_runner).
For each policy, replay permutations and report EX and candidate costs.
"""

import json
import random
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

    requested = getattr(policy, "k", getattr(policy, "max_k", len(permutation)))
    if requested > len(permutation):
        raise ValueError(f"Policy requests {requested} candidates but cache has {len(permutation)}.")
    budget = min(len(permutation), requested)
    for start in range(0, budget, batch_size):
        batch_indices = permutation[start:min(start + batch_size, budget)]
        for idx in batch_indices:
            seen.append(candidates[idx])
            n_used += 1

        stop, action = policy.should_stop(seen, n_used)
        if stop or n_used >= len(permutation):
            break

    timing = [_candidate_seconds(candidate) for candidate in seen]
    seconds_used = sum(seconds for seconds, _ in timing) if all(
        seconds is not None for seconds, _ in timing
    ) else None
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
        "csc_executed": False,
        "accuracy_basis": "sc_proxy_when_csc_requested" if action == "CALL_CSC" else "sc_vote",
    }


def summarize_questions(questions, *, n_permutations, seed, batch_size):
    """Bootstrap questions, not correlated permutations of the same question."""
    from evaluation.final_eval import bootstrap_mean_ci

    def average(key):
        values = [q[key] for q in questions]
        return sum(values) / len(values) if values and all(v is not None for v in values) else None

    ci = bootstrap_mean_ci([q["ex"] for q in questions], seed=seed)
    sample_ci = bootstrap_mean_ci([q["avg_samples"] for q in questions], seed=seed)
    bases = {q["seconds_basis"] for q in questions}
    csc = sum(q["csc_calls"] for q in questions)
    return {
        "ex_mean": round(ci["mean"], 4),
        "ex_ci95": round((ci["upper"] - ci["lower"]) / 2, 4),
        "ex_bootstrap_95": ci,
        "avg_samples_bootstrap_95": sample_ci,
        "avg_samples": round(average("avg_samples") or 0.0, 2),
        "avg_tokens": round(average("avg_tokens") or 0.0, 2),
        "avg_seconds": average("avg_seconds"),
        "seconds_basis": next(iter(bases)) if len(bases) == 1 else "mixed" if bases else "unavailable",
        "seed": seed, "batch_size": batch_size,
        "csc_calls": csc, "csc_executed": False,
        "accuracy_basis": "sc_proxy_when_csc_requested" if csc else "sc_vote",
        "n_questions": len(questions), "n_permutations": n_permutations,
        "per_question": questions,
    }


def simulate_policy(
    records: list[dict[str, Any]], policy: StoppingPolicy, db_path_fn,
    n_permutations: int = 100, batch_size: int = 4, seed: int = 42,
) -> dict[str, Any]:
    """Replay candidates and report question-level bootstrap confidence intervals."""
    from research.exec_runner import _exec_one

    if n_permutations < 1 or batch_size < 1:
        raise ValueError("n_permutations and batch_size must be positive.")
    rng = random.Random(seed)
    questions = []
    for record in records:
        gold = record.get("gold_execution") or _exec_one(record.get("gold_sql", ""), str(db_path_fn(record["db_id"])))
        if gold["fingerprint"] is None:
            raise ValueError(f"Gold execution failed for {record['db_id']}: {gold.get('error')}")
        trials = []
        for _ in range(n_permutations):
            permutation = list(range(len(record["candidates"])))
            rng.shuffle(permutation)
            trials.append(simulate_one(record, policy, gold["fingerprint"], permutation, batch_size))
        bases = {t["seconds_basis"] for t in trials}
        timed = all(t["seconds_used"] is not None for t in trials)
        questions.append({
            "id": record.get("id", record["db_id"] + "::" + record["question"]),
            "db_id": record["db_id"],
            "ex": sum(t["correct"] for t in trials) / n_permutations,
            "avg_samples": sum(t["n_samples_used"] for t in trials) / n_permutations,
            "avg_tokens": sum(t["n_tokens_used"] for t in trials) / n_permutations,
            "avg_seconds": sum(t["seconds_used"] for t in trials) / n_permutations if timed else None,
            "seconds_basis": next(iter(bases)) if len(bases) == 1 else "mixed",
            "csc_calls": sum(t["csc_called"] for t in trials),
        })
    return summarize_questions(questions, n_permutations=n_permutations, seed=seed, batch_size=batch_size)


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
