"""Format SFT samples as chat messages with <think>/<answer> tags.

Uses the shared research prompt, a semantic reconstruction of
SLM-SQL Appendix E: system prompt → user (schema + question) → assistant
(<think>reasoning</think><answer>SQL</answer>).
"""

import logging
from typing import Any

from research.slm_sql.prompts import SYSTEM, generation_messages
from core.sql_extractor import extract_sql

logger = logging.getLogger(__name__)

# Kept as an import-compatible alias for older notebook cells.
SYSTEM_MESSAGE = SYSTEM


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
    if not isinstance(schema, str):
        import json
        schema = json.dumps(schema, ensure_ascii=False)
    messages = generation_messages({"question": question, "evidence": evidence}, schema)
    if reasoning:
        assistant_content = f"<think>{reasoning}</think>\n<answer>{gold_sql}</answer>"
    else:
        assistant_content = f"<answer>{gold_sql}</answer>"
    return messages + [{"role": "assistant", "content": assistant_content}]


def validate_output(text: str) -> str | None:
    """Try to extract SQL from model output; return None on parse failure."""
    import re
    if not isinstance(text, str) or len(re.findall(r"<answer>", text, re.I)) != 1:
        return None
    if re.search(r"<answer>\s*.+?\s*</answer>", text, re.S | re.I) is None:
        return None
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
