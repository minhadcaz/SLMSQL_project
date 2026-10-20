"""Oracle analysis: pass@k, SC@k, top-2 coverage@k by difficulty and DB."""

import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_exec_results(path: str | Path) -> list[dict[str, Any]]:
    from research.candidate_generator import load_candidates
    return load_candidates(path)


def _gold_fingerprint(record: dict, db_path_fn) -> str | None:
    from research.exec_runner import _exec_one
    result = record.get("gold_execution")
    if result is None:
        result = _exec_one(record.get("gold_sql", ""), str(db_path_fn(record["db_id"])))
    if result.get("fingerprint") is None:
        raise ValueError(f"Gold execution failed for {record['db_id']}: {result.get('error')}")
    return result["fingerprint"]


def pass_at_k(records: list[dict], db_path_fn, k: int, n_permutations: int = 100, seed: int = 42) -> float:
    """Fraction of questions where at least one of the first k candidates matches gold."""
    if k < 1 or n_permutations < 1 or any(len(r["candidates"]) < k for r in records):
        raise ValueError("Positive k/permutations and at least k cached candidates per question are required.")
    rng = random.Random(seed)
    hits_sum = 0
    valid_questions = 0
    for record in records:
        gold_fp = _gold_fingerprint(record, db_path_fn)
        if gold_fp is None:
            continue
        valid_questions += 1
        cands = record["candidates"]
        fps = [c.get("fingerprint") for c in cands]
        question_hits = 0
        for _ in range(n_permutations):
            perm = list(range(len(fps)))
            rng.shuffle(perm)
            subset = perm[:k]
            if any(fps[i] == gold_fp for i in subset):
                question_hits += 1
        hits_sum += question_hits / n_permutations
    return hits_sum / valid_questions if valid_questions else 0.0


def sc_at_k(records: list[dict], db_path_fn, k: int, n_permutations: int = 100, seed: int = 42) -> float:
    """Self-consistency accuracy: majority-vote fingerprint among first k candidates."""
    if k < 1 or n_permutations < 1 or any(len(r["candidates"]) < k for r in records):
        raise ValueError("Positive k/permutations and at least k cached candidates per question are required.")
    rng = random.Random(seed)
    correct_sum = 0
    valid_questions = 0
    for record in records:
        gold_fp = _gold_fingerprint(record, db_path_fn)
        if gold_fp is None:
            continue
        valid_questions += 1
        cands = record["candidates"]
        fps = [c.get("fingerprint") for c in cands]
        question_correct = 0
        for _ in range(n_permutations):
            perm = list(range(len(fps)))
            rng.shuffle(perm)
            subset_fps = [fps[i] for i in perm[:k] if fps[i] is not None]
            if not subset_fps:
                continue
            vote = Counter(subset_fps)
            best_fp = vote.most_common(1)[0][0]
            if best_fp == gold_fp:
                question_correct += 1
        correct_sum += question_correct / n_permutations
    return correct_sum / valid_questions if valid_questions else 0.0


def top2_coverage_at_k(records: list[dict], db_path_fn, k: int, n_permutations: int = 100, seed: int = 42) -> float:
    """Fraction of questions where gold is in the top-2 fingerprint groups at step k."""
    if k < 1 or n_permutations < 1 or any(len(r["candidates"]) < k for r in records):
        raise ValueError("Positive k/permutations and at least k cached candidates per question are required.")
    rng = random.Random(seed)
    covered_sum = 0
    valid_questions = 0
    for record in records:
        gold_fp = _gold_fingerprint(record, db_path_fn)
        if gold_fp is None:
            continue
        valid_questions += 1
        cands = record["candidates"]
        fps = [c.get("fingerprint") for c in cands]
        question_covered = 0
        for _ in range(n_permutations):
            perm = list(range(len(fps)))
            rng.shuffle(perm)
            subset_fps = [fps[i] for i in perm[:k] if fps[i] is not None]
            if not subset_fps:
                continue
            vote = Counter(subset_fps)
            top2 = {fp for fp, _ in vote.most_common(2)}
            if gold_fp in top2:
                question_covered += 1
        covered_sum += question_covered / n_permutations
    return covered_sum / valid_questions if valid_questions else 0.0


def analyze_by_group(
    records: list[dict], db_path_fn, k_values: list[int],
    group_key: str = "difficulty", n_permutations: int = 100, seed: int = 42,
) -> dict[str, Any]:
    """Compute pass@k, SC@k, top2_coverage@k grouped by a field."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        key = str(r.get(group_key, "unknown"))
        groups.setdefault(key, []).append(r)

    results = {}
    for group_name, group_records in sorted(groups.items()):
        group_results = {}
        for k in k_values:
            group_results[f"pass@{k}"] = round(pass_at_k(group_records, db_path_fn, k, n_permutations, seed), 4)
            group_results[f"sc@{k}"] = round(sc_at_k(group_records, db_path_fn, k, n_permutations, seed), 4)
            group_results[f"top2@{k}"] = round(top2_coverage_at_k(group_records, db_path_fn, k, n_permutations, seed), 4)
        group_results["count"] = len(group_records)
        results[group_name] = group_results

    return results


def serial_dependence_diagnostic(records, db_path_fn, *, n_permutations=100, seed=42):
    """Test lag-1 correctness correlation against shuffles; not proof of independence."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    sequences = []
    for record in records:
        gold = _gold_fingerprint(record, db_path_fn)
        sequence = [int(c.get("fingerprint") == gold) for c in record["candidates"]]
        if len(sequence) >= 3 and len(set(sequence)) > 1:
            sequences.append(sequence)

    def statistic(items):
        numerator = denominator = 0.0
        for values in items:
            mean = sum(values) / len(values)
            centered = [value - mean for value in values]
            numerator += sum(a * b for a, b in zip(centered, centered[1:]))
            denominator += sum(value**2 for value in centered)
        return numerator / denominator if denominator else None

    observed = statistic(sequences)
    rng = random.Random(seed)
    null = []
    for _ in range(n_permutations):
        shuffled = [rng.sample(values, len(values)) for values in sequences]
        value = statistic(shuffled)
        if value is not None:
            null.append(value)
    null_center = sum(null) / len(null) if null else None
    p_value = (1 + sum(abs(v - null_center) >= abs(observed - null_center) for v in null)) / (1 + len(null)) if null else None
    return {
        "lag1_correctness_correlation": observed, "permutation_p_value": p_value,
        "informative_questions": len(sequences), "questions": len(records),
        "n_permutations": n_permutations, "seed": seed,
        "interpretation": "Exchangeability remains an assumption; constant sequences provide no dependence evidence.",
    }
