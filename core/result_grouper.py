"""Group final candidates by full execution result, never by SQL text."""
from collections import defaultdict


def group_candidates(candidates: list) -> dict:
    groups = defaultdict(list)
    for candidate in candidates:
        candidate.group = None
        candidate.vote_score = 0.0
        if candidate.execution and candidate.execution.success:
            key = candidate.execution.normalized_result
            groups[key].append(candidate)
            candidate.group = key
    total = sum(len(group) for group in groups.values())
    for group in groups.values():
        for candidate in group:
            candidate.vote_score = len(group) / total
    return dict(groups)
