"""Margin + entropy stopping: stop when leader margin >= m votes and entropy <= threshold."""

import math
from collections import Counter
from typing import Any


class MarginEntropyPolicy:
    def __init__(
        self,
        margin: int = 3,
        entropy_threshold: float = 1.0,
        min_k: int = 4,
        max_k: int = 32,
    ):
        self.margin = margin
        self.entropy_threshold = entropy_threshold
        self.min_k = min_k
        self.max_k = max_k

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        if step < self.min_k:
            return False, "SAMPLE_MORE"
        if step >= self.max_k:
            return True, "STOP"

        fps = [c.get("fingerprint") for c in candidates_so_far if c.get("fingerprint")]
        if len(fps) < 2:
            return False, "SAMPLE_MORE"

        vote = Counter(fps)
        counts = vote.most_common()
        leader_count = counts[0][1]
        second_count = counts[1][1] if len(counts) > 1 else 0

        if leader_count - second_count < self.margin:
            return False, "SAMPLE_MORE"

        n = len(fps)
        entropy = 0.0
        for count in vote.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        if entropy <= self.entropy_threshold:
            return True, "STOP"
        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if not fps:
            return None
        vote = Counter(fps)
        return vote.most_common(1)[0][0]
