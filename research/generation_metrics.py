"""Completion metadata shared by offline and online sampling (no GPU imports)."""

import re


def completion_logprob(completion):
    """Average the emitted tokens, excluding alternative top-logprob tokens."""
    token_ids = getattr(completion, "token_ids", [])
    cumulative = getattr(completion, "cumulative_logprob", None)
    if cumulative is not None and token_ids:
        return float(cumulative) / len(token_ids)
    probabilities = []
    for token_id, alternatives in zip(token_ids, getattr(completion, "logprobs", None) or []):
        item = (alternatives or {}).get(token_id)
        if item is not None:
            probabilities.append(float(item.logprob))
    return sum(probabilities) / len(probabilities) if probabilities else None


def has_answer_format(text):
    return bool(
        isinstance(text, str)
        and len(re.findall(r"<answer>", text, re.I)) == 1
        and len(re.findall(r"</answer>", text, re.I)) == 1
        and re.search(r"<answer>\s*.+?\s*</answer>", text, re.S | re.I)
    )
