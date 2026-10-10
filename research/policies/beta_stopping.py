"""Beta stopping (Adaptive-Consistency, Aggarwal 2023).

Models the posterior probability that the majority answer is correct using
a Beta distribution over execution-fingerprint agreement. Stops when
P(correct) >= threshold or budget exhausted.
"""

import math
from collections import Counter
from typing import Any


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
        leader_count = vote.most_common(1)[0][1]
        n = len(fps)
        alpha = leader_count + 1
        beta = n - leader_count + 1
        # P(p > 0.5) via Beta CDF complement — use regularized incomplete beta approx
        # For tractability, use the normal approximation to the Beta
        mean = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        if var <= 0:
            return 1.0 if mean > 0.5 else 0.0
        std = math.sqrt(var)
        z = (mean - 0.5) / std
        # Approximate Phi(z) via logistic function
        confidence = 1.0 / (1.0 + math.exp(-1.7 * z))
        return confidence

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        if step < self.min_k:
            return False, "SAMPLE_MORE"
        if step >= self.max_k:
            return True, "STOP"
        conf = self._posterior_confidence(candidates_so_far)
        if conf >= self.threshold:
            return True, "STOP"
        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if not fps:
            return None
        vote = Counter(fps)
        return vote.most_common(1)[0][0]
