"""Generation followed by strict answer extraction."""
from core.sql_extractor import extract_sql


def generate_candidates(backend, prompt: str, n: int, parameters: dict,
                        start_id: int = 0, deadline=None) -> list:
    candidates = backend.generate(prompt, n, parameters, deadline)
    for index, candidate in enumerate(candidates):
        candidate.id = start_id + index
        if candidate.error:
            continue
        try:
            candidate.sql = extract_sql(candidate.raw_text)
        except ValueError as error:
            candidate.error = str(error)
        # Raw model output remains internal; exports/UI only include the extracted SQL.
    return candidates
