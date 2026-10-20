"""Scientific regressions found in the 47-task audit; no downloaded models."""

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from evaluation.final_eval import paired_report_test
from evaluation.legacy_protocol import freeze_legacy_protocol
from research.calibration import compute_features, feature_auc, cross_validate_stopper
from research.candidate_generator import generate_candidates, load_candidates
from research.exec_runner import _exec_one, execute_candidates
from research.generation_metrics import completion_logprob, has_answer_format
from research.legacy_protocol import split_legacy_dev
from research.pareto import build_pareto_table, pareto_frontier
from research.policies.beta_stopping import BetaStoppingPolicy
from research.policies.fixed_k import FixedKPolicy
from research.policies.learned_stopper import LearnedStopperPolicy
from research.simulator import simulate_one, simulate_policy
from research.slm_sql.execution import execute
from training.data_prep import filter_synsql, filter_by_token_length
from training.sft_trainer import encode_sft_records


class CharacterTokenizer:
    name_or_path = "test-character-tokenizer"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(m["role"] + ":" + m["content"] + "\n" for m in messages)
        if add_generation_prompt:
            text += "assistant:"
        return list(map(ord, text)) if tokenize else text

    def encode(self, text, **_):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def test_beta_uses_exact_top_two_not_all_losing_groups():
    candidates = [{"fingerprint": fp} for fp in "aaaabcde"]
    assert BetaStoppingPolicy()._posterior_confidence(candidates) == pytest.approx(57 / 64)
    assert BetaStoppingPolicy()._posterior_confidence([{"fingerprint": "a"}] * 4) == pytest.approx(31 / 32)
    assert BetaStoppingPolicy()._posterior_confidence([{"fingerprint": f} for f in "aabb"]) == 0.5


def test_auc_ties_are_half_credit_independent_of_row_order():
    from sklearn.metrics import roc_auc_score
    rows = [{"x": x, "sc_correct": y} for x, y in [(0, 1), (0, 0), (1, 1), (1, 0), (2, 1)]]
    expected = round(roc_auc_score([r["sc_correct"] for r in rows], [r["x"] for r in rows]), 4)
    assert feature_auc(rows, "x") == expected
    assert feature_auc(list(reversed(rows)), "x") == expected
    assert feature_auc([{"x": 1, "sc_correct": y} for y in (1, 0, 1, 0)], "x") == 0.5


def test_zero_logprob_and_failed_sql_lengths_are_observed_features():
    features = compute_features([
        {"sql": "SELECT 1", "avg_logprob": 0.0, "error": "syntax", "schema_size": 15},
        {"sql": "SELECT 2", "avg_logprob": -2.0, "fingerprint": "a", "schema_size": 15},
    ])
    assert features["mean_logprob"] == -1.0
    assert features["mean_sql_length"] == 8
    assert features["schema_size"] == 15
    assert features["error_ratio"] == 0.5


def test_single_negative_class_stopper_never_becomes_confident():
    policy = LearnedStopperPolicy()
    policy.train([{"sc_correct": False, "leader_ratio": 1.0}] * 3)
    assert policy._predict_proba([{"fingerprint": "a"}] * 4) == 0.0


def test_stopper_cv_never_trains_on_held_out_database():
    records = []
    for index in range(4):
        records.append({
            "db_id": str(index), "question": str(index), "gold_sql": "SELECT 1",
            "gold_execution": {"fingerprint": "gold"},
            "candidates": [{"sql": "SELECT 1", "fingerprint": "gold", "n_tokens": 1}] * 32,
        })
    result = cross_validate_stopper(records, lambda _: "unused", n_splits=2, n_permutations=2)
    assert result["n_questions"] == 4
    assert result["ex_mean"] == 1.0
    for fold in result["folds"]:
        assert not set(fold["train_db_ids"]) & set(fold["test_db_ids"])
    assert {r["db_id"] for r in result["per_question"]} == {"0", "1", "2", "3"}


def test_legacy_split_is_stable_disjoint_and_detects_changed_sources(tmp_path):
    records = [{"db_id": f"d{i}", "question": f"q{i}", "gold_sql": "SELECT 1"} for i in range(10)]
    path = tmp_path / "split.json"
    split = split_legacy_dev(records, path)
    assert not {r["db_id"] for r in split["calibration"]} & {r["db_id"] for r in split["evaluation"]}
    assert split == split_legacy_dev(records, path)
    with pytest.raises(ValueError, match="split changed"):
        split_legacy_dev(records[:-1], path)


def test_sampled_logprob_excludes_alternative_top_token():
    completion = SimpleNamespace(token_ids=[7, 8], logprobs=[
        {7: SimpleNamespace(logprob=-2), 9: SimpleNamespace(logprob=-0.1)},
        {8: SimpleNamespace(logprob=-4), 9: SimpleNamespace(logprob=-0.2)},
    ])
    assert completion_logprob(completion) == -3
    assert completion_logprob(SimpleNamespace(token_ids=[1], logprobs=None)) is None
    assert not has_answer_format("SELECT 1")
    assert has_answer_format("<think>x</think><answer>SELECT 1</answer>")
    assert not has_answer_format("<answer>SELECT 1</answer><answer>SELECT 2</answer>")


def test_synsql_keeps_literal_case_and_removes_repeated_sql_cot(tmp_path):
    records = [
        {"sql": "SELECT 'Paris'", "output": "Think. SELECT 'Paris'"},
        {"sql": "SELECT 'paris'", "reasoning": "Think."},
        {"sql": "SELECT 2", "reasoning": "SELECT 2, again SELECT 2"},
        {"sql": "DELETE FROM x WHERE value IN (SELECT 1)"},
        {"sql": "SELECT 1; DROP TABLE x"},
        {"output": "<think>Count rows.</think><answer>SELECT 3</answer>"},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    accepted = filter_synsql(path)
    assert [r["sql"] for r in accepted] == ["SELECT 'Paris'", "SELECT 'paris'", "SELECT 3"]
    assert "SELECT" not in accepted[0]["reasoning"]


def test_actual_token_length_includes_template_and_completion_loss_is_masked():
    tokenizer = CharacterTokenizer()
    records = [{"question": "q", "schema": "CREATE TABLE x (id INT)", "sql": "SELECT id FROM x", "reasoning": "r" * 50}]
    full, kept, skipped = encode_sft_records(records, tokenizer, 4000)
    assert kept == [0] and skipped == []
    length = len(full[0]["input_ids"])
    assert filter_by_token_length(records, tokenizer, length - 1) == []
    assert len(filter_by_token_length(records, tokenizer, length)) == 1
    nonmasked = [token for token in full[0]["labels"] if token != -100]
    assert tokenizer.decode(nonmasked).startswith("<think>")
    for mode in ("answer_only", "short_cot"):
        variant, same_kept, same_skipped = encode_sft_records(records, tokenizer, 4000, cot_mode=mode, short_cot_tokens=5)
        assert (same_kept, same_skipped) == (kept, skipped)
        assert len(variant[0]["input_ids"]) < length
    assert encode_sft_records(records, tokenizer, length - 1)[0] == []
    incomplete = [{"question": "q", "sql": "SELECT 1", "reasoning": "r"}]
    assert filter_by_token_length(incomplete, tokenizer, 4000) == []


def test_execution_deadline_is_recorded_not_raised(db, monkeypatch):
    from research.slm_sql import execution
    calls = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(calls, 2.0))
    result = execute("SELECT 1", db, timeout=1)
    assert result["fingerprint"] is None
    assert result["error"] == "sql_timeout"


def test_candidate_executor_parallel_cache_and_gold_parity(tmp_path, db):
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps({
        "question": "constant", "db_id": "demo", "gold_sql": "SELECT 1", "evidence": "keep me",
        "candidates": [{"sql": sql} for sql in ("SELECT 1", "SELECT 1 UNION ALL SELECT 1", "SELECT 2", "")],
    }) + "\n")
    parallel = tmp_path / "parallel.jsonl"
    execute_candidates(source, lambda _: db, parallel, max_workers=2)
    record = load_candidates(parallel)[0]
    assert record["evidence"] == "keep me"
    assert record["candidates"][0]["fingerprint"] == record["candidates"][1]["fingerprint"]
    assert record["gold_execution"]["fingerprint"] == _exec_one("SELECT 1", str(db))["fingerprint"]
    assert record["candidates"][-1]["error"]
    assert execute_candidates(source, lambda _: db, parallel, max_workers=1)["total_questions"] == 1


def test_simulation_bootstraps_questions_and_respects_uneven_batch_budget():
    record = {
        "question": "q", "db_id": "d", "gold_sql": "SELECT 1", "gold_execution": {"fingerprint": "a"},
        "candidates": [{"fingerprint": "a", "n_tokens": 1}] * 8,
    }
    result = simulate_one(record, FixedKPolicy(5), "a", list(range(8)), batch_size=4)
    assert result["n_samples_used"] == 5
    assert result["seconds_used"] is None
    summary = simulate_policy([record], FixedKPolicy(4), lambda _: "unused", n_permutations=3)
    assert summary["ex_bootstrap_95"]["n"] == 1
    assert summary["avg_samples_bootstrap_95"]["mean"] == 4
    with pytest.raises(ValueError, match="cache has"):
        simulate_one(record, FixedKPolicy(16), "a", list(range(8)))


def test_pareto_rejects_missing_costs_proxy_csc_and_equal_cost_dominated_points():
    rows = [{"name": "low", "avg_samples": 4, "ex": 0.5}, {"name": "high", "avg_samples": 4, "ex": 0.7}, {"name": "missing", "ex": 1.0}]
    assert [r["name"] for r in pareto_frontier(rows)] == ["high"]
    assert build_pareto_table([{"name": "fake_csc", "accuracy_basis": "sc_proxy_when_csc_requested"}]) == []


def test_paired_test_refuses_silent_id_intersection():
    row = {"id": "q", "db_id": "d", "question": "q", "gold_sql": "SELECT 1", "correct": True}
    with pytest.raises(ValueError, match="same questions"):
        paired_report_test({"records": [row], "seed": 42}, {"records": [{**row, "id": "other"}], "seed": 42})


class TinyBackend:
    calls = 0

    def __init__(self, settings, *, frozen_identity=None):
        type(self).calls += 1
        self.identity = frozen_identity or {"model": settings["model"], "revision": "test-sha"}
        self.versions = {"test_backend": "1"}

    def generate(self, messages, *, n, seeds):
        assert all("SECRET_GOLD" not in json.dumps(m) for m in messages)
        return [[{"raw_text": "<answer>SELECT 1</answer>", "n_tokens": 4, "finish_reason": "stop"}] * n for _ in messages]


def test_legacy_generation_cache_is_reused_without_gpu_and_config_changes_fail(tmp_path):
    prompts = [{"question": "q", "db_id": "d", "gold_sql": "SECRET_GOLD", "schema": "CREATE TABLE x (id INT)"}]
    path = tmp_path / "cache.jsonl"
    result = generate_candidates("test-model", prompts, n_candidates=4, cache_path=path, backend_factory=TinyBackend)
    assert result[0]["candidates"][0]["seed"] >= 0
    calls = TinyBackend.calls
    assert generate_candidates("test-model", prompts, n_candidates=4, cache_path=path, backend_factory=TinyBackend) == result
    assert TinyBackend.calls == calls
    with pytest.raises(ValueError, match="configuration changed"):
        generate_candidates("test-model", prompts, n_candidates=8, cache_path=path, backend_factory=TinyBackend)


def test_author_synsql_think_input_output_seq_is_normalized(tmp_path):
    source = json.loads(Path('/tmp/slm-sql-audit-sources/think_sample.json').read_text()) if Path('/tmp/slm-sql-audit-sources/think_sample.json').exists() else {
        'input_seq': 'Database Schema:\nCREATE TABLE t (id INT);\nThis schema describes the database.\n\nQuestion:\nHow many?\n\nInstructions:',
        'output_seq': '<think>Count.</think><answer>SELECT COUNT(*) FROM t</answer>',
    }
    path = tmp_path / 'author.jsonl'
    path.write_text(json.dumps(source) + '\n', encoding='utf-8')
    rows = filter_synsql(path)
    assert len(rows) == 1
    assert rows[0]['question'].startswith('released in the year') or rows[0]['question'] == 'How many?'
    assert rows[0]['schema'].startswith('CREATE TABLE')
    assert rows[0]['sql'].upper().startswith('SELECT')


def test_author_synsql_think_json_array_export_is_normalized(tmp_path):
    record = {
        'input_seq': (
            'Database Schema:\nCREATE TABLE t (id INT);\n'
            'This schema describes the database.\n\nQuestion:\nHow many?\n\nInstructions:'
        ),
        'output_seq': '<think>Count rows.</think><answer>SELECT COUNT(*) FROM t</answer>',
    }
    path = tmp_path / 'author.json'
    path.write_text(json.dumps([record]) + '\n', encoding='utf-8')
    rows = filter_synsql(path)
    assert len(rows) == 1
    assert rows[0]['question'] == 'How many?'
    assert rows[0]['schema'].startswith('CREATE TABLE')
    assert rows[0]['sql'] == 'SELECT COUNT(*) FROM t'


def test_legacy_protocol_freeze_reuses_and_rejects_changes(tmp_path):
    path = tmp_path / 'legacy_protocol.json'
    datasets = {'bird_dev': [{'id': 'q1', 'db_id': 'db1', 'question': 'q', 'gold_sql': 'SELECT 1'}]}
    configs = {'bird_dev_fixed': {'seed': 42, 'mode': 'fixed_sc'}}
    first = freeze_legacy_protocol(path, configs=configs, seeds=[42], k_values=[16], datasets=datasets)
    assert first['reused'] is False
    second = freeze_legacy_protocol(path, configs=configs, seeds=[42], k_values=[16], datasets=datasets)
    assert second['reused'] is True
    with pytest.raises(ValueError, match='protocol changed'):
        freeze_legacy_protocol(path, configs={**configs, 'bird_dev_fixed': {'seed': 43}}, seeds=[42], k_values=[16], datasets=datasets)
