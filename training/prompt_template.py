"""Format SFT samples as chat messages with <think>/<answer> tags.

Wraps core.prompt_builder for the Qwen2.5-Coder chat template used by
SLM-SQL Appendix E: system prompt → user (schema + question) → assistant
(<think>reasoning</think><answer>SQL</answer>).
"""

import logging
from typing import Any

from core.prompt_builder import build_prompt
from core.sql_extractor import extract_sql

logger = logging.getLogger(__name__)

SYSTEM_MESSAGE = (
    "You are a helpful SQL assistant. Given a database schema and a natural "
    "language question, generate a correct SQLite query. Wrap your reasoning "
    "in <think>...</think> and your final SQL in <answer>...</answer>."
)


def format_sft_messages(
    question: str,
    schema: dict[str, Any] | str,
    gold_sql: str,
    reasoning: str = "",
    evidence: str = "",
) -> list[dict[str, str]]:
    """Build a chat-format training sample (system/user/assistant).

    If *reasoning* is empty, the assistant turn contains only the answer tag
    (answer-only mode for CoT ablation).
    """
    user_content = build_prompt(question, schema, evidence)
    if reasoning:
        assistant_content = f"<think>{reasoning}</think>\n<answer>{gold_sql}</answer>"
    else:
        assistant_content = f"<answer>{gold_sql}</answer>"
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def validate_output(text: str) -> str | None:
    """Try to extract SQL from model output; return None on parse failure."""
    try:
        return extract_sql(text)
    except ValueError:
        return None


def batch_format_stats(samples: list[dict[str, Any]]) -> dict[str, int]:
    """Count format errors in a batch of raw model outputs (for logging)."""
    total = len(samples)
    errors = 0
    for sample in samples:
        raw = sample.get("raw_text", sample.get("output", ""))
        if validate_output(raw) is None:
            errors += 1
    rate = errors / total if total else 0.0
    logger.info("Format error rate: %d/%d (%.1f%%)", errors, total, rate * 100)
    return {"total": total, "format_errors": errors}
