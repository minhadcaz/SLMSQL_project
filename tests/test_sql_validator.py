import pytest

from core.sql_validator import SQLValidator


@pytest.mark.parametrize("sql", [
    "DROP TABLE students", "DELETE FROM students", "PRAGMA user_version",
    "ATTACH DATABASE 'other.db' AS other", "SELECT 1; SELECT 2",
    "SELECT 1; -- comment\nDELETE FROM students", "UPDATE students SET full_name='x'",
    "CREATE TABLE x(a)", "SELECT load_extension('x')", "SELECT writefile('x', 'y')",
    "SELECT * FROM sqlite_master", "SELECT * FROM other.students",
    "WITH x AS (SELECT 1) DELETE FROM students", "VACUUM", "SELECT randomblob(99999999)",
])
def test_safety(sql, db):
    result = SQLValidator().validate(sql, db)
    assert not result.is_safe


@pytest.mark.parametrize("sql", [
    "SELECT * FROM students", "SELECT 'DROP' AS label",
    "SELECT s.full_name, c.name FROM students s JOIN classes c ON s.class_id=c.id",
    "WITH x AS (SELECT class_id, COUNT(*) AS n FROM students GROUP BY class_id) SELECT n FROM x",
    "SELECT class_id, COUNT(*) AS n FROM students GROUP BY class_id ORDER BY n",
    "SELECT full_name FROM students s WHERE EXISTS (SELECT 1 FROM enrollments e WHERE e.student_id=s.id)",
    "SELECT full_name FROM students UNION SELECT name FROM classes",
    "SELECT 1 /* benign comment */",
])
def test_valid_scoped_queries(sql, db):
    result = SQLValidator().validate(sql, db)
    assert result.is_safe and result.is_valid, result.errors


@pytest.mark.parametrize("sql", [
    "SELECT missing FROM students", "SELECT * FROM missing_table",
    "SELECT wrong.full_name FROM students s", "SELECT id FROM students s JOIN classes c ON c.id=s.class_id",
])
def test_schema_errors(sql, db):
    result = SQLValidator().validate(sql, db)
    assert result.is_safe and not result.is_valid
    assert result.error_type == "schema"


def test_schema_ablation(db):
    result = SQLValidator().validate("SELECT missing FROM students", db, check_schema=False)
    assert result.is_valid
