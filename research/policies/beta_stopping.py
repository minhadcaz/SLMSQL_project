"""Beta stopping (Adaptive-Consistency, Aggarwal 2023).

Uses the exact top-two Beta posterior from Aggarwal et al. (2023).
This estimates stability of the leading group, not semantic correctness.
"""

import math
from collections import Counter

from research.slm_sql.policy_view import policy_view


class BetaStoppingPolicy:
    def __init__(self, threshold: float = 0.95, min_k: int = 4, max_k: int = 32):
        self.threshold = threshold
        self.min_k = min_k
        self.max_k = max_k

    def _posterior_confidence(self, candidates: list[dict]) -> float:
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if len(fps) < 2:
            return 0.0
        vote = Counter(fps)
        counts = [count for _, count in vote.most_common(2)]
        leader, runner_up = counts[0], counts[1] if len(counts) > 1 else 0
        # For integer a,b, 1-I_0.5(a,b) is a binomial CDF. This is exact
        # without scipy, including unanimous samples and ties.
        a, b = leader + 1, runner_up + 1
        trials = a + b - 1
        return sum(math.comb(trials, j) for j in range(a)) / 2**trials

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        candidates_so_far = policy_view(candidates_so_far)
        if step < self.min_k:
            return False, "SAMPLE_MORE"
        conf = self._posterior_confidence(candidates_so_far)
        if conf >= self.threshold:
            return True, "STOP"
        if step >= self.max_k:
            return False, "SAMPLE_MORE"
        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        candidates = policy_view(candidates)
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if not fps:
            return None
        vote = Counter(fps)
        return vote.most_common(1)[0][0]
