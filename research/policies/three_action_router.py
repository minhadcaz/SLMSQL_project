"""STOP / SAMPLE_MORE / CALL_CSC research router.

Two optional classifiers estimate leader correctness and top-2 correctness from
calibration data only. Without fitting, vote fractions are explicit heuristics,
not calibrated probabilities. Offline CALL_CSC is a routing decision; EX requires
an actual merger run and its additional token/time budget.
"""

from collections import Counter

from research.calibration import compute_features
from research.slm_sql.policy_view import policy_view
from research.policies.learned_stopper import LearnedStopperPolicy


class ThreeActionRouterPolicy:
    def __init__(
        self, stop_threshold: float = 0.90, csc_threshold: float = 0.70,
        min_k: int = 4, max_k: int = 32, model_path: str | None = None,
        top2_model_path: str | None = None,
    ):
        self.stop_threshold, self.csc_threshold = stop_threshold, csc_threshold
        self.min_k, self.max_k = min_k, max_k
        self.leader_estimator = LearnedStopperPolicy(model_path=model_path)
        self.top2_estimator = LearnedStopperPolicy(model_path=top2_model_path)

    @property
    def confidence_basis(self):
        return "learned_calibration_estimates" if self.top2_estimator._model is not None else "vote_fraction_heuristic"

    def load(self, path: str, *, top2_path: str | None = None) -> None:
        self.leader_estimator.load(path)
        if top2_path:
            self.top2_estimator.load(top2_path)

    def train(self, calibration_data: list[dict], model_type="logistic") -> dict:
        self.leader_estimator.model_type = model_type
        self.top2_estimator.model_type = model_type
        return {
            "leader": self.leader_estimator.train(calibration_data),
            "top2": self.top2_estimator.train(calibration_data, target="top2_correct"),
            "training_db_ids": sorted({row["db_id"] for row in calibration_data}),
        }

    def _estimate_confidence(self, candidates: list[dict]) -> tuple[float, float]:
        candidates = policy_view(candidates)
        features = compute_features(candidates)
        leader = (self.leader_estimator._predict_proba(candidates)
                  if self.leader_estimator._model is not None else features["leader_ratio"])
        votes = Counter(c["fingerprint"] for c in candidates if c.get("fingerprint") and not c.get("error"))
        top2 = (self.top2_estimator._predict_proba(candidates)
                if self.top2_estimator._model is not None
                else sum(n for _, n in votes.most_common(2)) / sum(votes.values()) if votes else 0.0)
        return leader, top2

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        if step < self.min_k:
            return False, "SAMPLE_MORE"
        leader, top2 = self._estimate_confidence(candidates_so_far)
        if leader >= self.stop_threshold:
            return True, "STOP"
        if top2 >= self.csc_threshold and len(self.select_csc_inputs(candidates_so_far)) == 2:
            return True, "CALL_CSC"
        if step >= self.max_k:
            return False, "SAMPLE_MORE"
        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        candidates = policy_view(candidates)
        votes = Counter(c["fingerprint"] for c in candidates if c.get("fingerprint") and not c.get("error"))
        return votes.most_common(1)[0][0] if votes else None

    def select_csc_inputs(self, candidates: list[dict]) -> list[str]:
        candidates = policy_view(candidates)
        votes = Counter(c["fingerprint"] for c in candidates if c.get("fingerprint") and not c.get("error"))
        return [next(c["sql"] for c in candidates if c.get("fingerprint") == fp and not c.get("error"))
                for fp, _ in votes.most_common(2)]
