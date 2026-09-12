"""Budget and consensus stopping rules."""
from core.types import StoppingDecision


def decide(groups: dict, candidates_used: int, elapsed_seconds: float,
           config: dict) -> StoppingDecision:
    sizes = sorted((len(group) for group in groups.values()), reverse=True)
    valid = sum(sizes)
    confidence = sizes[0] / valid if valid else 0.0
    margin = (sizes[0] - (sizes[1] if len(sizes) > 1 else 0)) / valid if valid else 0.0
    reason = "continue"
    if elapsed_seconds >= config["time_budget_seconds"]:
        reason = "time_budget"
    elif candidates_used >= config["max_candidates"]:
        reason = "max_candidates"
    elif (valid >= config["min_valid_candidates"]
          and confidence >= config["confidence_threshold"]
          and margin >= config["margin_threshold"]):
        reason = "consensus"
    return StoppingDecision(reason != "continue", reason, confidence, margin, candidates_used)
