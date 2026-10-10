"""Three-action router (G2): STOP / SAMPLE_MORE / CALL_CSC.

Main contribution — extends 2-action stopping to 3-action routing by
estimating P(gold in top-2 fingerprint groups). When top-2 coverage is
high but leader confidence is low, routes to CSC (merge-correct) instead
of sampling more.
"""

import math
from collections import Counter
from typing import Any

from research.calibration import compute_features


class ThreeActionRouterPolicy:
    def __init__(
        self,
        stop_threshold: float = 0.90,
        csc_threshold: float = 0.70,
        min_k: int = 4,
        max_k: int = 32,
        model_path: str | None = None,
    ):
        """
        stop_threshold: confidence above which we STOP
        csc_threshold: confidence above which (if not stopping) we CALL_CSC
        """
        self.stop_threshold = stop_threshold
        self.csc_threshold = csc_threshold
        self.min_k = min_k
        self.max_k = max_k
        self._model = None
        self._feature_names = [
            "leader_ratio", "n_groups", "entropy",
            "error_ratio", "empty_ratio", "mean_logprob", "mean_sql_length",
        ]
        if model_path:
            self.load(model_path)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    def _estimate_confidence(self, candidates: list[dict]) -> tuple[float, float]:
        """Return (leader_confidence, top2_coverage_estimate)."""
        features = compute_features(candidates)
        leader_ratio = features["leader_ratio"]
        n_groups = features["n_groups"]
        entropy = features["entropy"]

        if self._model is not None:
            x = [[features.get(f, 0.0) for f in self._feature_names]]
            proba = self._model.predict_proba(x)[0]
            leader_conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            leader_conf = leader_ratio

        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if len(fps) >= 2:
            vote = Counter(fps)
            top2_counts = sum(c for _, c in vote.most_common(2))
            top2_coverage = top2_counts / len(fps)
        else:
            top2_coverage = leader_ratio

        return leader_conf, top2_coverage

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        if step < self.min_k:
            return False, "SAMPLE_MORE"
        if step >= self.max_k:
            return True, "STOP"

        leader_conf, top2_cov = self._estimate_confidence(candidates_so_far)

        if leader_conf >= self.stop_threshold:
            return True, "STOP"

        if top2_cov >= self.csc_threshold and leader_conf < self.stop_threshold:
            return True, "CALL_CSC"

        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if not fps:
            return None
        vote = Counter(fps)
        return vote.most_common(1)[0][0]

    def select_csc_inputs(self, candidates: list[dict]) -> list[str]:
        """Select top-2 fingerprint group representatives for CSC merge."""
        fps_to_sql: dict[str, list[str]] = {}
        for c in candidates:
            fp = c.get("fingerprint")
            if fp:
                fps_to_sql.setdefault(fp, []).append(c.get("sql", ""))

        vote = Counter(c.get("fingerprint") for c in candidates if c.get("fingerprint"))
        top2_fps = [fp for fp, _ in vote.most_common(2)]

        result = []
        for fp in top2_fps:
            sqls = fps_to_sql.get(fp, [])
            if sqls:
                result.append(sqls[0])
        return result
