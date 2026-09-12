import hashlib

from core.sql_executor import SQLExecutor
from evaluation.metrics import execution_match


def test_executor_is_independent_safety_boundary(db, config):
    executor = SQLExecutor(config["execution"])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    for sql in ["DROP TABLE students", "ATTACH 'x' AS x", "SELECT load_extension('x')"]:
        assert not executor.execute(db, sql).success
    assert executor.execute(db, "SELECT COUNT(*) FROM students").rows == [[16]]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_preview_not_used_for_ex(db, config):
    config["execution"]["max_rows"] = 1
    executor = SQLExecutor(config["execution"])
    first = executor.execute(db, "SELECT id FROM students ORDER BY id")
    second = executor.execute(db, "SELECT id FROM students WHERE id<3 ORDER BY id")
    assert first.rows == second.rows == [[1]]
    assert first.truncated
    assert not execution_match(second, first, "SELECT id FROM students ORDER BY id")


def test_result_limit_rejects_incomplete_vote(db, config):
    config["execution"]["max_result_rows"] = 2
    result = SQLExecutor(config["execution"]).execute(db, "SELECT * FROM students")
    assert not result.success and result.error_type == "result_limit"


def test_step_timeout(db, config):
    config["execution"]["max_steps"] = 1000
    result = SQLExecutor(config["execution"]).execute(
        db, "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x"
    )
    assert not result.success and result.error_type == "timeout"


def test_order_follows_gold_for_evaluation(db, config):
    executor = SQLExecutor(config["execution"])
    gold = executor.execute(db, "SELECT id FROM students ORDER BY id DESC")
    prediction = executor.execute(db, "SELECT id FROM students")
    assert not execution_match(prediction, gold, "SELECT id FROM students ORDER BY id DESC")
    assert execution_match(prediction, gold, "SELECT id FROM students")
