"""Candidate quality improvements: error filtering, logprob weighting, prompt diversity."""

import math
from collections import Counter
from typing import Any


def filter_errors(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove candidates with execution errors or empty results."""
    return [c for c in candidates if not c.get("error") and not c.get("is_empty")]


def weighted_vote(candidates: list[dict[str, Any]], weight_key: str = "avg_logprob") -> str | None:
    """Majority vote weighted by exp(logprob)."""
    scores: dict[str, float] = {}
    for c in candidates:
        fp = c.get("fingerprint")
        if not fp:
            continue
        w = math.exp(c.get(weight_key, 0.0))
        scores[fp] = scores.get(fp, 0.0) + w
    if not scores:
        return None
    return max(scores, key=scores.get)


def diverse_vote(candidates: list[dict[str, Any]], max_per_sql: int = 3) -> str | None:
    """Deduplicate near-identical SQL before voting to increase diversity."""
    seen_sql: dict[str, int] = {}
    filtered = []
    for c in candidates:
        sql = c.get("sql", "").strip()
        count = seen_sql.get(sql, 0)
        if count < max_per_sql:
            filtered.append(c)
            seen_sql[sql] = count + 1

    fps = [c.get("fingerprint") for c in filtered if c.get("fingerprint")]
    if not fps:
        return None
    vote = Counter(fps)
    return vote.most_common(1)[0][0]


def ablation_vote(
    candidates: list[dict[str, Any]],
    use_filter: bool = True,
    use_weighted: bool = False,
    use_diverse: bool = False,
    max_per_sql: int = 3,
) -> str | None:
    """Configurable voting for ablation studies."""
    pool = candidates
    if use_filter:
        pool = filter_errors(pool)
    if not pool:
        pool = candidates

    if use_diverse:
        seen_sql: dict[str, int] = {}
        deduped = []
        for c in pool:
            sql = c.get("sql", "").strip()
            count = seen_sql.get(sql, 0)
            if count < max_per_sql:
                deduped.append(c)
                seen_sql[sql] = count + 1
        pool = deduped if deduped else pool

    if use_weighted:
        return weighted_vote(pool)

    fps = [c.get("fingerprint") for c in pool if c.get("fingerprint")]
    if not fps:
        return None
    vote = Counter(fps)
    return vote.most_common(1)[0][0]


def run_ablation(
    records: list[dict[str, Any]],
    db_path_fn,
    configs: list[dict[str, bool]] | None = None,
) -> dict[str, dict[str, float]]:
    """Run ablation across filter/weighted/diverse combinations."""
    from research.exec_runner import _exec_one

    if configs is None:
        configs = [
            {"use_filter": False, "use_weighted": False, "use_diverse": False},
            {"use_filter": True, "use_weighted": False, "use_diverse": False},
            {"use_filter": True, "use_weighted": True, "use_diverse": False},
            {"use_filter": True, "use_weighted": False, "use_diverse": True},
            {"use_filter": True, "use_weighted": True, "use_diverse": True},
        ]

    results = {}
    for cfg in configs:
        label = "+".join(k.replace("use_", "") for k, v in cfg.items() if v) or "baseline"
        correct = 0
        total = 0
        for record in records:
            gold_sql = record.get("gold_sql", "")
            if not gold_sql:
                continue
            gold_result = _exec_one(gold_sql, str(db_path_fn(record["db_id"])))
            gold_fp = gold_result["fingerprint"]
            if gold_fp is None:
                continue

            selected = ablation_vote(record["candidates"], **cfg)
            if selected == gold_fp:
                correct += 1
            total += 1

        results[label] = {
            "ex": round(correct / total, 4) if total else 0.0,
            "n": total,
        }

    return results
