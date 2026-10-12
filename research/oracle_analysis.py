"""Oracle analysis: pass@k, SC@k, top-2 coverage@k by difficulty and DB."""

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_exec_results(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _gold_fingerprint(record: dict, db_path_fn) -> str | None:
    from research.exec_runner import _exec_one
    gold = record.get("gold_sql", "")
    if not gold:
        return None
    result = _exec_one(gold, str(db_path_fn(record["db_id"])))
    return result["fingerprint"]


def pass_at_k(records: list[dict], db_path_fn, k: int, n_permutations: int = 100, seed: int = 42) -> float:
    """Fraction of questions where at least one of the first k candidates matches gold."""
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
