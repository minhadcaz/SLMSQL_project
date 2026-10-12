"""Offline policy simulator: replay candidates with random permutations.

Input: candidates JSONL with execution fingerprints (from exec_runner).
For each policy, simulate ≥100 permutations and report EX mean±CI, avg samples.
"""

import json
import random
import math
from collections import Counter
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


def simulate_one(
    record: dict[str, Any],
    policy: StoppingPolicy,
    gold_fp: str | None,
    permutation: list[int],
    batch_size: int = 4,
) -> dict[str, Any]:
    """Simulate a policy on one question with a given candidate permutation.

    Returns dict with: correct, n_samples_used, action, and csc_called.
    """
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

    selected_fp = policy.select(seen)
    correct = selected_fp == gold_fp if gold_fp and selected_fp else False

    return {
        "correct": correct,
        "n_samples_used": n_used,
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

    rng = random.Random(seed)
    all_correct = []
    all_samples = []
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

        for _ in range(n_permutations):
            perm = list(range(n_cands))
            rng.shuffle(perm)
            result = simulate_one(record, policy, gold_fp, perm, batch_size)
            question_correct += int(result["correct"])
            question_samples += result["n_samples_used"]
            total_csc_calls += int(result["csc_called"])

        all_correct.append(question_correct / n_permutations)
        all_samples.append(question_samples / n_permutations)

    n = len(all_correct)
    if n == 0:
        return {
            "ex_mean": 0.0, "ex_ci95": 0.0, "avg_samples": 0.0,
            "csc_calls": 0, "n_questions": 0,
        }

    ex_mean = sum(all_correct) / n
    ex_std = math.sqrt(sum((x - ex_mean) ** 2 for x in all_correct) / n)
    ex_ci95 = 1.96 * ex_std / math.sqrt(n)

    avg_samples = sum(all_samples) / n

    return {
        "ex_mean": round(ex_mean, 4),
        "ex_ci95": round(ex_ci95, 4),
        "avg_samples": round(avg_samples, 2),
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
