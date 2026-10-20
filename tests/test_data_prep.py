"""Tests for training.data_prep — SynSQL filtering and stratified sampling."""

import json

import pytest

from training.data_prep import filter_synsql, stratified_sample, truncate_by_length


@pytest.fixture()
def raw_jsonl(tmp_path):
    """Create a small JSONL file mimicking SynSQL format."""
    records = [
        {"question": "How many users?", "sql": "SELECT COUNT(*) FROM users", "db_id": "db1"},
        {"question": "All names", "sql": "SELECT name FROM users", "db_id": "db1"},
        {"question": "Insert user", "sql": "INSERT INTO users VALUES (1, 'x')", "db_id": "db2"},
        {"question": "With comment", "sql": "SELECT * FROM t -- hack", "db_id": "db2"},
        {"question": "Duplicate", "sql": "SELECT COUNT(*) FROM users", "db_id": "db3"},
        {"question": "Empty", "sql": "", "db_id": "db3"},
        {"question": "No sql key", "db_id": "db3"},
    ]
    path = tmp_path / "raw.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


class TestFilterSynsql:
    def test_filters_non_select(self, raw_jsonl):
        result = filter_synsql(raw_jsonl)
        sqls = [r["sql"] for r in result]
        assert all("SELECT" in s.upper() for s in sqls)

    def test_filters_comments(self, raw_jsonl):
        result = filter_synsql(raw_jsonl)
        assert not any("--" in r["sql"] for r in result)

    def test_deduplicates(self, raw_jsonl):
        result = filter_synsql(raw_jsonl)
        sqls = [r["sql"] for r in result]
        assert len(sqls) == len(set(sqls))

    def test_filters_empty(self, raw_jsonl):
        result = filter_synsql(raw_jsonl)
        assert all(r["sql"].strip() for r in result)

    def test_expected_count(self, raw_jsonl):
        result = filter_synsql(raw_jsonl)
        assert len(result) == 2

    def test_normalizes_gretel_field_names(self, tmp_path):
        path = tmp_path / "gretel.jsonl"
        path.write_text(json.dumps({
            "sql_prompt": "How many users?",
            "sql_context": "CREATE TABLE users (id INTEGER);",
            "sql_explanation": "Count the rows.",
            "sql": "SELECT COUNT(*) FROM users",
        }) + "\n")
        result = filter_synsql(path)
        assert result[0]["question"] == "How many users?"
        assert "CREATE TABLE users" in result[0]["schema"]
        assert result[0]["reasoning"] == "Count the rows."


class TestTruncateByLength:
    def test_keeps_short(self):
        samples = [{"question": "hi", "sql": "SELECT 1"}]
        assert len(truncate_by_length(samples, max_chars=1000)) == 1

    def test_drops_long(self):
        samples = [{"question": "x" * 5000, "sql": "SELECT " + "x" * 5000}]
        assert len(truncate_by_length(samples, max_chars=100)) == 0


class TestStratifiedSample:
    def test_size_respected(self):
        samples = [{"db_id": f"db{i % 3}", "sql": f"SELECT {i}"} for i in range(100)]
        result = stratified_sample(samples, 20)
        assert len(result) == 20

    def test_returns_all_if_size_exceeds(self):
        samples = [{"db_id": "db1", "sql": "SELECT 1"}]
        result = stratified_sample(samples, 100)
        assert len(result) == 1

    def test_deterministic(self):
        samples = [{"db_id": f"db{i % 5}", "sql": f"SELECT {i}"} for i in range(50)]
        r1 = stratified_sample(samples, 10, seed=42)
        r2 = stratified_sample(samples, 10, seed=42)
        assert r1 == r2

    def test_different_seeds(self):
        samples = [{"db_id": f"db{i % 5}", "sql": f"SELECT {i}"} for i in range(50)]
        r1 = stratified_sample(samples, 10, seed=42)
        r2 = stratified_sample(samples, 10, seed=99)
        assert r1 != r2
