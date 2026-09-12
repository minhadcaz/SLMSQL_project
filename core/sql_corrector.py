"""One repair attempt; callers always validate and execute again."""
from core.prompt_builder import correction_prompt
from core.sql_generator import generate_candidates


def correct_candidate(backend, candidate, question, schema, evidence, parameters, deadline):
    errors = candidate.validation.errors if candidate.validation else []
    if candidate.execution and candidate.execution.error:
        errors = errors + [candidate.execution.error]
    prompt = correction_prompt(question, schema, candidate.sql, "; ".join(errors), evidence)
    settings = dict(parameters, temperature=0.0)
    generated = generate_candidates(backend, prompt, 1, settings, candidate.id, deadline)
    return generated[0] if generated else None
