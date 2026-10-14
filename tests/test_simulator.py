"""Regression tests for offline adaptive-policy simulation."""

import random
from collections import Counter

from research.policies.beta_stopping import BetaStoppingPolicy
from research.policies.fixed_k import FixedKPolicy
from research.simulator import simulate_policy


def _records() -> list[dict]:
    candidates = []
    for index in range(32):
        candidates.append({
            "sql": f"SELECT {index}",
            "fingerprint": "gold" if index < 18 else "wrong",
            "n_tokens": index + 1,
            "generation_ms": 10,
        })
    return [{
        "question": "q",
        "db_id": "db",
        "gold_sql": "SELECT 1",
        "candidates": candidates,
    }]


def _manual_sc_at_k(records: list[dict], k: int, n_permutations: int, seed: int) -> float:
    rng = random.Random(seed)
    question_scores = []
    for record in records:
        correct = 0
        for _ in range(n_permutations):
            permutation = list(range(len(record["candidates"])))
            rng.shuffle(permutation)
            fingerprints = [record["candidates"][i]["fingerprint"] for i in permutation[:k]]
            if Counter(fingerprints).most_common(1)[0][0] == "gold":
                correct += 1
        question_scores.append(correct / n_permutations)
    return sum(question_scores) / len(question_scores)


def test_fixed_k_16_matches_manual_sc_at_16(monkeypatch):
    """Fixed-k simulation must match manual SC@16 under the same permutations."""
    from research import exec_runner

    monkeypatch.setattr(exec_runner, "_exec_one", lambda sql, db_path: {"fingerprint": "gold"})
    records = _records()
    seed = 42
    n_permutations = 100

    result = simulate_policy(
        records,
        FixedKPolicy(k=16),
        lambda _: "unused",
        n_permutations=n_permutations,
        batch_size=4,
        seed=seed,
    )
    manual = _manual_sc_at_k(records, 16, n_permutations, seed)

    assert result["ex_mean"] == round(manual, 4)
    assert result["avg_samples"] == 16.0
    assert result["avg_tokens"] > 0
    assert result["avg_seconds"] == 0.16
    assert result["n_permutations"] == n_permutations


def test_simulation_seed_is_reproducible(monkeypatch):
    from research import exec_runner

    monkeypatch.setattr(exec_runner, "_exec_one", lambda sql, db_path: {"fingerprint": "gold"})
    records = _records()

    first = simulate_policy(records, BetaStoppingPolicy(max_k=16), lambda _: "unused", seed=7)
    second = simulate_policy(records, BetaStoppingPolicy(max_k=16), lambda _: "unused", seed=7)

    assert first == second
