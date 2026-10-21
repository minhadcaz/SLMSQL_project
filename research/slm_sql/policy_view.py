"""Normalize research candidates before passing them to adaptive policies."""

from __future__ import annotations

from typing import Any


def policy_view(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a flat, backward-compatible view of candidate inference signals.

    New SLM-SQL artifacts keep execution metadata under ``candidate["execution"]``.
    Older research artifacts already store these fields at the candidate root. The
    policy view accepts both layouts and exposes only inference-time signals; it
    never copies gold labels or the enclosing benchmark record.
    """
    flattened = []
    for candidate in candidates:
        execution = candidate.get("execution")
        execution = execution if isinstance(execution, dict) else {}
        fingerprint = candidate.get("fingerprint")
        if fingerprint is None:
            fingerprint = execution.get("fingerprint")
        execution_error = execution.get("error")
        parse_error = candidate.get("parse_error")
        error = execution_error or parse_error or candidate.get("error")
        row_count = execution.get("row_count")
        if row_count is None:
            row_count = candidate.get("row_count")
        is_empty = candidate.get("is_empty")
        if is_empty is None and row_count is not None:
            is_empty = not error and row_count == 0
        flattened.append(
            {
                **candidate,
                "fingerprint": fingerprint,
                "error": error,
                "is_empty": bool(is_empty) if is_empty is not None else False,
                "sql": candidate.get("sql", ""),
                "avg_logprob": candidate.get("avg_logprob"),
                "n_tokens": candidate.get("n_tokens"),
            }
        )
    return flattened
