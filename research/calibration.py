"""Calibration analysis: accuracy vs consensus features, AUC ranking."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def compute_features(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Extract stopping-decision features from a set of executed candidates."""
    fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
    errors = [c for c in candidates if c.get("error")]
    empties = [c for c in candidates if c.get("is_empty")]
    logprobs = [c.get("avg_logprob", 0.0) for c in candidates if c.get("avg_logprob")]
    sql_lengths = [len(c.get("sql", "")) for c in candidates]

    n_total = len(candidates)
    n_valid = len(fps)

    if not fps:
        return {
            "leader_ratio": 0.0,
            "n_groups": 0,
            "entropy": 0.0,
            "error_ratio": len(errors) / n_total if n_total else 0.0,
            "empty_ratio": len(empties) / n_total if n_total else 0.0,
            "mean_logprob": 0.0,
            "mean_sql_length": 0.0,
            "n_valid": 0,
            "n_total": n_total,
        }

    vote = Counter(fps)
    groups = len(vote)
    leader_count = vote.most_common(1)[0][1]
    leader_ratio = leader_count / n_valid

    entropy = 0.0
    for count in vote.values():
        p = count / n_valid
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "leader_ratio": round(leader_ratio, 4),
        "n_groups": groups,
        "entropy": round(entropy, 4),
        "error_ratio": round(len(errors) / n_total, 4) if n_total else 0.0,
        "empty_ratio": round(len(empties) / n_total, 4) if n_total else 0.0,
        "mean_logprob": round(sum(logprobs) / len(logprobs), 4) if logprobs else 0.0,
        "mean_sql_length": round(sum(sql_lengths) / len(sql_lengths), 1) if sql_lengths else 0.0,
        "n_valid": n_valid,
        "n_total": n_total,
    }


def compute_calibration_data(
    exec_records: list[dict[str, Any]],
    db_path_fn,
    k_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """For each question, compute features at various k and whether SC@k is correct."""
    from research.exec_runner import _exec_one

    if k_values is None:
        k_values = [4, 8, 12, 16, 24, 32]

    rows = []
    for record in exec_records:
        gold_sql = record.get("gold_sql", "")
        if not gold_sql:
            continue
        gold_result = _exec_one(gold_sql, str(db_path_fn(record["db_id"])))
        gold_fp = gold_result["fingerprint"]
        if gold_fp is None:
            continue

        cands = record["candidates"]
        for k in k_values:
            subset = cands[:k]
            features = compute_features(subset)

            fps = [c.get("fingerprint") for c in subset if c.get("fingerprint")]
            if fps:
                vote = Counter(fps)
                sc_fp = vote.most_common(1)[0][0]
                sc_correct = sc_fp == gold_fp
            else:
                sc_correct = False

            rows.append({
                "question": record["question"],
                "db_id": record["db_id"],
                "difficulty": record.get("difficulty", "unknown"),
                "k": k,
                "sc_correct": sc_correct,
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

    pairs.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    total_pos = sum(1 for _, label in pairs if label)
    total_neg = len(pairs) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    for value, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp

    return round(auc / (total_pos * total_neg), 4)


def rank_features(calibration_data: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Rank features by AUC for predicting SC correctness."""
    feature_names = [
        "leader_ratio", "n_groups", "entropy",
        "error_ratio", "empty_ratio", "mean_logprob", "mean_sql_length",
    ]
    aucs = [(name, feature_auc(calibration_data, name)) for name in feature_names]
    aucs.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
    return aucs
