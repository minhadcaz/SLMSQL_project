"""Calibration analysis: accuracy vs consensus features, AUC ranking."""

import math
from collections import Counter
from typing import Any


def compute_features(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Only inference-time signals; no gold SQL or labels enter the features."""
    fps = [c["fingerprint"] for c in candidates if c.get("fingerprint") and not c.get("error")]
    n_total, n_valid = len(candidates), len(fps)
    votes = Counter(fps)
    logprobs = [float(c["avg_logprob"]) for c in candidates if c.get("avg_logprob") is not None]
    lengths = [len(c.get("sql", "")) for c in candidates]
    schema_sizes = [float(c["schema_size"]) for c in candidates if c.get("schema_size") is not None]
    return {
        "leader_ratio": max(votes.values(), default=0) / n_valid if n_valid else 0.0,
        "n_groups": len(votes),
        "entropy": -sum((v / n_valid) * math.log2(v / n_valid) for v in votes.values()),
        "error_ratio": sum(bool(c.get("error")) for c in candidates) / n_total if n_total else 0.0,
        "empty_ratio": sum(bool(c.get("is_empty")) for c in candidates) / n_total if n_total else 0.0,
        "mean_logprob": sum(logprobs) / len(logprobs) if logprobs else 0.0,
        "mean_sql_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "schema_size": sum(schema_sizes) / len(schema_sizes) if schema_sizes else 0.0,
        "n_valid": n_valid,
        "n_total": n_total,
    }


def compute_calibration_data(
    exec_records: list[dict[str, Any]],
    db_path_fn,
    k_values: list[int] | None = None,
    n_permutations: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """For each question, compute features at various k and whether SC@k is correct."""
    from research.exec_runner import _exec_one

    if k_values is None:
        k_values = [4, 8, 12, 16, 24, 32]

    import random

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    rng = random.Random(seed)
    rows = []
    for record in exec_records:
        gold_sql = record.get("gold_sql", "")
        gold = record.get("gold_execution") or _exec_one(gold_sql, str(db_path_fn(record["db_id"])))
        gold_fp = gold["fingerprint"]
        if gold_fp is None:
            raise ValueError(f"Gold execution failed for {record['db_id']}: {gold.get('error')}")
        for permutation in range(n_permutations):
            candidates = list(record["candidates"])
            if n_permutations > 1:
                rng.shuffle(candidates)
            for k in k_values:
                if k > len(candidates):
                    continue
                subset = candidates[:k]
                features = compute_features(subset)
                vote = Counter(c["fingerprint"] for c in subset if c.get("fingerprint"))
                leaders = [fp for fp, _ in vote.most_common(2)]
                rows.append({
                    "question": record["question"],
                    "db_id": record["db_id"],
                    "difficulty": record.get("difficulty", "unknown"),
                    "k": k,
                    "permutation": permutation,
                    "sc_correct": bool(leaders and leaders[0] == gold_fp),
                    "top2_correct": gold_fp in leaders,
                    **features,
                })
    return rows


def feature_auc(calibration_data: list[dict[str, Any]], feature_name: str) -> float:
    """Compute AUC for a single feature as a predictor of SC correctness.

    Higher AUC means the feature better separates correct from incorrect SC votes.
    """
    pairs = [
        (row[feature_name], row["sc_correct"])
        for row in calibration_data
        if feature_name in row
    ]
    if not pairs:
        return 0.5

    # Mann–Whitney rank statistic with average ranks for tied scores.
    # A constant feature has AUC 0.5 regardless of input row order.
    pairs.sort(key=lambda item: item[0])
    positives = sum(bool(label) for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return 0.5
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(bool(label) for _, label in pairs[start:end])
        start = end
    return round((rank_sum - positives * (positives + 1) / 2) / (positives * negatives), 4)


def rank_features(calibration_data: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Rank features by AUC for predicting SC correctness."""
    feature_names = [
        "leader_ratio", "n_groups", "entropy",
        "error_ratio", "empty_ratio", "mean_logprob", "mean_sql_length", "schema_size",
    ]
    aucs = [(name, feature_auc(calibration_data, name)) for name in feature_names]
    aucs.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
    return aucs


def cross_validate_stopper(
    records: list[dict], db_path_fn, *, model_type="logistic", threshold=0.85,
    n_splits=5, n_permutations=100, seed=42,
) -> dict:
    """Evaluate once per held-out DB. Threshold is declared before CV, not tuned on test folds."""
    from sklearn.model_selection import GroupKFold
    from research.policies.learned_stopper import LearnedStopperPolicy
    from research.simulator import simulate_policy, summarize_questions

    db_ids = [r["db_id"] for r in records]
    count = min(n_splits, len(set(db_ids)))
    if count < 2:
        raise ValueError("DB-disjoint CV requires at least two databases.")
    folds, questions = [], []
    for index, (train_indices, test_indices) in enumerate(
        GroupKFold(n_splits=count).split(records, groups=db_ids)
    ):
        train_records = [records[i] for i in train_indices]
        test_records = [records[i] for i in test_indices]
        train_dbs = sorted({r["db_id"] for r in train_records})
        test_dbs = sorted({r["db_id"] for r in test_records})
        assert not set(train_dbs) & set(test_dbs)
        training_rows = compute_calibration_data(
            train_records, db_path_fn, n_permutations=10, seed=seed
        )
        policy = LearnedStopperPolicy(model_type=model_type, threshold=threshold)
        training = policy.train(training_rows)
        result = simulate_policy(
            test_records, policy, db_path_fn, n_permutations=n_permutations, seed=seed
        )
        questions.extend(result["per_question"])
        folds.append({
            "fold": index, "train_db_ids": train_dbs, "test_db_ids": test_dbs,
            "training": training, "test_ex": result["ex_mean"],
        })
    summary = summarize_questions(questions, n_permutations=n_permutations, seed=seed, batch_size=4)
    return {**summary, "evaluation": "out_of_fold_by_database", "threshold": threshold, "folds": folds}
