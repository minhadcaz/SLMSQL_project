"""Tests for benchmarks.eval_ex — execution accuracy evaluator."""

import sqlite3

import pytest

from benchmarks.eval_ex import batch_evaluate, execution_match


@pytest.fixture()
def simple_db(tmp_path):
    """Create a minimal SQLite DB with one table for testing."""
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, value REAL)")
    conn.execute("INSERT INTO t VALUES (1, 'alice', 10.0)")
    conn.execute("INSERT INTO t VALUES (2, 'bob', 20.0)")
    conn.execute("INSERT INTO t VALUES (3, 'charlie', 30.0)")
    conn.commit()
    conn.close()
    return db_path


class TestExecutionMatch:
    def test_identical_queries(self, simple_db):
        sql = "SELECT * FROM t"
        assert execution_match(sql, sql, simple_db) is True

    def test_semantically_equivalent(self, simple_db):
        pred = "SELECT id, name, value FROM t WHERE value > 5"
        gold = "SELECT id, name, value FROM t WHERE value > 5.0"
        assert execution_match(pred, gold, simple_db) is True

    def test_different_results(self, simple_db):
        pred = "SELECT * FROM t WHERE id = 1"
        gold = "SELECT * FROM t WHERE id = 2"
        assert execution_match(pred, gold, simple_db) is False

    def test_unordered_match(self, simple_db):
        pred = "SELECT name FROM t ORDER BY name DESC"
        gold = "SELECT name FROM t"
        assert execution_match(pred, gold, simple_db) is True

    def test_ordered_mismatch(self, simple_db):
        pred = "SELECT name FROM t ORDER BY name DESC"
        gold = "SELECT name FROM t ORDER BY name ASC"
        assert execution_match(pred, gold, simple_db) is False

    def test_pred_error_returns_false(self, simple_db):
        assert execution_match("SELECT * FROM nonexistent", "SELECT * FROM t", simple_db) is False

    def test_empty_sql_raises(self, simple_db):
        with pytest.raises(ValueError):
            execution_match("", "SELECT * FROM t", simple_db, timeout=0)

    def test_invalid_timeout(self, simple_db):
        with pytest.raises(ValueError):
            execution_match("SELECT 1", "SELECT 1", simple_db, timeout=-1)

    def test_missing_db(self, tmp_path):
        fake = tmp_path / "no.sqlite"
        assert execution_match("SELECT 1", "SELECT 1", fake) is False


class TestBatchEvaluate:
    def test_all_correct(self, simple_db):
        preds = [
            {"pred_sql": "SELECT * FROM t", "gold_sql": "SELECT * FROM t",
             "db_id": "test", "difficulty": "simple"},
        ]
        result = batch_evaluate(preds, lambda _: simple_db)
        assert result["accuracy"] == 1.0
        assert result["correct"] == 1
        assert result["total"] == 1

    def test_mixed(self, simple_db):
        preds = [
            {"pred_sql": "SELECT * FROM t", "gold_sql": "SELECT * FROM t",
             "db_id": "test", "difficulty": "simple"},
            {"pred_sql": "SELECT 999", "gold_sql": "SELECT * FROM t",
             "db_id": "test", "difficulty": "hard"},
        ]
        result = batch_evaluate(preds, lambda _: simple_db)
        assert result["accuracy"] == 0.5
        assert result["correct"] == 1
        assert result["total"] == 2
        assert "simple" in result["per_difficulty"]
        assert "hard" in result["per_difficulty"]

    def test_empty_batch(self, simple_db):
        result = batch_evaluate([], lambda _: simple_db)
        assert result["accuracy"] == 0.0
        assert result["total"] == 0
