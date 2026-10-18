"""Semantic reconstruction of SLM-SQL Appendix E, shared by both CSC variants.

Not an assertion of byte-for-byte reproduction of unpublished inference templates.
Schema DDL, evidence and execution previews are always benchmark input, never gold.
"""

import json

PROMPT_VERSION = "appendix-e-semantic-v1"
SYSTEM = "You are an expert in translating questions into correct SQLite queries."


def generation_messages(record: dict, schema: str) -> list[dict]:
    request = json.dumps(
        {"question": record["question"], "evidence": record.get("evidence", "")}, ensure_ascii=False
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Database engine: SQLite\nSchema:\n{schema}\n\nRequest:\n{request}\n\n"
                "Analyze the schema, relevant values, joins and operations needed to answer the question. "
                "Write analysis inside <think>...</think> and exactly one final SQL query inside "
                "<answer>...</answer>."
            ),
        },
    ]


def merge_messages(record: dict, schema: str, groups: list[dict]) -> list[dict]:
    if len(groups) != 2 or groups[0]["fingerprint"] == groups[1]["fingerprint"]:
        raise ValueError("CSC requires two distinct executable result groups.")
    request = json.dumps(
        {"question": record["question"], "evidence": record.get("evidence", "")}, ensure_ascii=False
    )
    drafts = [
        {
            "draft": label,
            "sql": group["sql"],
            "result_sample": group["execution"]["preview"],
            "result_rows": group["execution"]["row_count"],
        }
        for label, group in zip(("A", "B"), groups)
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Database engine: SQLite\nSchema:\n{schema}\n\nRequest:\n{request}\n\n"
                f"Candidate SQL queries and execution samples:\n{json.dumps(drafts, ensure_ascii=False)}\n\n"
                "Compare both drafts against the question, schema and evidence. Check joins, filters, "
                "duplicates and aggregation. Choose one of the two SQL drafts; do not combine or rewrite them. "
                "Put your comparison in <think>...</think> and one final SQL in <answer>...</answer>."
            ),
        },
    ]


def top_groups(candidates: list[dict], k: int | None = None) -> list[dict]:
    groups = {}
    for candidate in candidates[:k]:
        execution = candidate["execution"]
        fp = execution.get("fingerprint")
        if fp is None:
            continue
        if fp not in groups:
            groups[fp] = {
                "fingerprint": fp,
                "sql": candidate["sql"],
                "execution": execution,
                "votes": 0,
            }
        groups[fp]["votes"] += 1
    # Python's stable sort gives deterministic first-generated tie breaking.
    return sorted(groups.values(), key=lambda group: -group["votes"])
