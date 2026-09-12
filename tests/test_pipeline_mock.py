import json

import pytest

from core.config import ROOT
from core.model_backend import ModelBackend
from core.pipeline import Pipeline
from core.types import Candidate
from evaluation.evaluate import evaluate


@pytest.mark.parametrize("mode,count", [("greedy", 1), ("fixed_sc", 8), ("adaptive_sc", 4)])
def test_modes(db, config, mode, count):
    config["sampling"]["mode"] = mode
    result = Pipeline(config).run(db, "Có bao nhiêu sinh viên?")
    assert result.selected.execution.rows == [[16]]
    assert len(result.candidates) == count
    assert "raw_text" not in json.dumps(result.public_dict())
    if mode == "fixed_sc":
        assert result.correction_successes == 1
        assert result.decision.reason == "fixed_complete"


def test_all_demo_questions(db, config):
    pipeline = Pipeline(config)
    for item in json.loads((ROOT / "data/demo_questions.json").read_text(encoding="utf-8")):
        result = pipeline.run(db, item["question"])
        assert result.selected, (item["id"], result.public_dict())


class UnsafeBackend(ModelBackend):
    name = "unsafe_fixture"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, n, parameters, deadline=None):
        self.calls += 1
        return [Candidate(i, raw_text="<answer>DROP TABLE students</answer>") for i in range(n)]


def test_unsafe_never_corrected(db, config):
    config["sampling"]["mode"] = "greedy"
    backend = UnsafeBackend()
    result = Pipeline(config, backend).run(db, "test")
    assert not result.selected
    assert result.correction_attempts == 0
    assert backend.calls == 1


def test_unknown_mock_question_fails_honestly(db, config):
    result = Pipeline(config).run(db, "A held-out question")
    assert not result.selected


def test_evaluation_writes_artifacts(db, config, tmp_path):
    summary = evaluate(config, ROOT / "data/demo_questions.json", db, tmp_path, limit=2)
    assert summary["execution_accuracy"] == 1
    assert summary["mock_notice"]
    assert (tmp_path / "summary.csv").exists()
    assert len((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_executor_safety_failure_never_corrected(db, config):
    class FunctionBackend(UnsafeBackend):
        def generate(self, prompt, n, parameters, deadline=None):
            self.calls += 1
            return [Candidate(i, raw_text="SELECT random()") for i in range(n)]

    config["sampling"]["mode"] = "greedy"
    config["validation"]["schema"] = False
    backend = FunctionBackend()
    result = Pipeline(config, backend).run(db, "test")
    assert not result.selected
    assert result.correction_attempts == 0
    assert backend.calls == 1
