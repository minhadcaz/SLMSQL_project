"""Fixed-k stopping policy: always use exactly k candidates, then majority vote."""

from collections import Counter


class FixedKPolicy:
    def __init__(self, k: int = 16):
        self.k = k

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        if step >= self.k:
            return True, "STOP"
        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if not fps:
            return None
        vote = Counter(fps)
        return vote.most_common(1)[0][0]
