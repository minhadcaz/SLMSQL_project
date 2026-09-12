"""Configurable heuristic; neutral generation score when log-probabilities unavailable."""


def rank_candidates(candidates: list, weights: dict):
    successful = []
    for candidate in candidates:
        validation = candidate.validation
        executed = bool(candidate.execution and candidate.execution.success)
        candidate.score = (
            weights["vote"] * candidate.vote_score
            + weights["schema"] * (validation.schema_score if validation else 0)
            + weights["execution"] * executed
            + weights["generation"] * candidate.generation_score
            - weights["warning_penalty"] * len(validation.warnings if validation else [])
        )
        if executed:
            successful.append(candidate)
    return max(successful, key=lambda c: (
        c.score, c.vote_score, c.validation.schema_score,
        -len(c.validation.warnings), -len(c.sql), -c.id
    ), default=None)
