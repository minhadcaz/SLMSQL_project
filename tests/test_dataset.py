import json
import sqlite3

import sqlglot

from core.config import ROOT
from core.sql_validator import SQLValidator
from databases.create_student_db import verify_gold


def test_gold_semantics_and_separate_questions(db):
    demo = json.loads((ROOT / "data/demo_questions.json").read_text(encoding="utf-8"))
    heldout = json.loads((ROOT / "data/evaluation_questions.json").read_text(encoding="utf-8"))
    assert not {q["question"] for q in demo} & {q["question"] for q in heldout}
    assert len(demo) == len(heldout) == 100
    assert verify_gold(db) == 200
    all_rows = demo + heldout
    assert len({r['id'] for r in all_rows}) == 200
    assert len({r['question'].strip().casefold() for r in all_rows}) == 200
    assert len({sqlglot.parse_one(r['gold_sql'], read='sqlite').sql(normalize=True)
                for r in all_rows}) == 200
    for row in all_rows:
        assert row['difficulty'] in {'easy', 'medium', 'hard'}
        assert row['language'] == 'vi'
        assert row['db_id'] == 'student_management'
        assert row['tags']
        validation = SQLValidator().validate(row['gold_sql'], db)
        assert validation.is_valid, (row['id'], validation.errors)
    with sqlite3.connect(db) as connection:
        assert connection.execute(demo[1]["gold_sql"]).fetchall() == [(16,)]
        assert connection.execute(demo[9]["gold_sql"]).fetchall() == [(27,)]
        assert connection.execute(demo[21]["gold_sql"]).fetchall() == [("Phan Tuấn Kiệt",)]
        assert set(connection.execute(demo[22]["gold_sql"]).fetchall()) == {
            ("Thiết kế cơ bản",), ("Học máy",), ("Đồ án tốt nghiệp",)
        }
        assert connection.execute(heldout[7]["gold_sql"]).fetchall() == [("Thiết kế",)]
        assert connection.execute(heldout[14]["gold_sql"]).fetchall() == [(2,)]


def test_expanded_gold_handles_repeat_enrollments_and_nulls(db):
    """Check meanings that a successful SQL execution alone cannot establish."""
    demo = json.loads((ROOT / 'data/demo_questions.json').read_text(encoding='utf-8'))
    evaluation = json.loads((ROOT / 'data/evaluation_questions.json').read_text(encoding='utf-8'))
    by_id = {r['id']: r['gold_sql'] for r in demo + evaluation}
    with sqlite3.connect(db) as connection:
        connection.execute("INSERT INTO enrollments VALUES (100, 1, 1, '2025-1', 2)")
        connection.execute("INSERT INTO enrollments VALUES (101, 16, 8, '2025-1', NULL)")
        enrollments = connection.execute(
            'SELECT student_id, subject_id, semester, score FROM enrollments'
        ).fetchall()
        credits = dict(connection.execute('SELECT id, credits FROM subjects'))
        # Repeating a subject adds credits to per-registration totals, but not
        # to credits for distinct passed subjects. NULL is never a zero grade.
        expected_weighted = {}
        expected_passed = {}
        for student_id in range(1, 17):
            graded = [e for e in enrollments if e[0] == student_id and e[3] is not None]
            if graded:
                expected_weighted[student_id] = (
                    sum(e[3] * credits[e[1]] for e in graded)
                    / sum(credits[e[1]] for e in graded)
                )
            passed = {e[1] for e in graded if e[3] >= 5}
            if passed:
                expected_passed[student_id] = sum(credits[s] for s in passed)
        assert dict(connection.execute(by_id['demo_093'])) == expected_weighted
        assert dict(connection.execute(by_id['eval_087'])) == expected_passed
        # Student 16 has a registration but still no grade.
        assert connection.execute(by_id['demo_082']).fetchall() == [(16, 'Phan Tuấn Kiệt')]
        # An unregistered subject must have zero ungraded registrations, not one
        # synthetic NULL row from the LEFT JOIN.
        ungraded = {row[0]: row[2] for row in connection.execute(by_id['eval_068'])}
        assert ungraded[6] == ungraded[7] == 0
        assert ungraded[8] == 1
