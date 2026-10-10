"""Learned stopper: Logistic regression or XGBoost on consensus features.

Trained on calibration data (research/calibration.py output), predicts
P(SC@k correct) from features at each step. Stops when confidence >= threshold.
"""

import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from research.calibration import compute_features


class LearnedStopperPolicy:
    def __init__(
        self,
        model_path: str | Path | None = None,
        model_type: str = "logistic",
        threshold: float = 0.85,
        min_k: int = 4,
        max_k: int = 32,
    ):
        self.threshold = threshold
        self.min_k = min_k
        self.max_k = max_k
        self.model_type = model_type
        self._model = None
        self._feature_names = [
            "leader_ratio", "n_groups", "entropy",
            "error_ratio", "empty_ratio", "mean_logprob", "mean_sql_length",
        ]
        if model_path:
            self.load(model_path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def train(self, calibration_data: list[dict[str, Any]]) -> dict[str, float]:
        """Train the stopper model on calibration data."""
        X = [[row.get(f, 0.0) for f in self._feature_names] for row in calibration_data]
        y = [int(row["sc_correct"]) for row in calibration_data]

        if self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(max_iter=1000, random_state=42)
            self._model.fit(X, y)
        elif self.model_type == "xgboost":
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, use_label_encoder=False, eval_metric="logloss",
            )
            self._model.fit(X, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        preds = self._model.predict(X)
        acc = sum(p == t for p, t in zip(preds, y)) / len(y) if y else 0.0
        return {"train_accuracy": round(acc, 4), "n_samples": len(y)}

    def _predict_proba(self, candidates: list[dict]) -> float:
        if self._model is None:
            return 0.0
        features = compute_features(candidates)
        x = [[features.get(f, 0.0) for f in self._feature_names]]
        proba = self._model.predict_proba(x)[0]
        return float(proba[1]) if len(proba) > 1 else float(proba[0])

    def should_stop(self, candidates_so_far: list[dict], step: int) -> tuple[bool, str]:
        if step < self.min_k:
            return False, "SAMPLE_MORE"
        if step >= self.max_k:
            return True, "STOP"
        conf = self._predict_proba(candidates_so_far)
        if conf >= self.threshold:
            return True, "STOP"
        return False, "SAMPLE_MORE"

    def select(self, candidates: list[dict]) -> str | None:
        fps = [c.get("fingerprint") for c in candidates if c.get("fingerprint")]
        if not fps:
            return None
        vote = Counter(fps)
        return vote.most_common(1)[0][0]
