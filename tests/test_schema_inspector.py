import hashlib

from core.schema_inspector import SchemaInspector
from core.schema_selector import SchemaSelector


def test_schema_and_fk(db):
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    schema = SchemaInspector().inspect(db)
    assert set(schema) == {"students", "classes", "departments", "subjects", "enrollments"}
    assert schema["students"]["foreign_keys"][0]["table"] == "classes"
    assert len(schema["students"]["samples"]) == 2
    assert "PRIMARY KEY" in SchemaInspector.format(schema)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_selector_retains_bridges(db):
    schema = SchemaInspector().inspect(db)
    assert "classes" in SchemaSelector().select("sinh viên khoa nào", schema)
    assert SchemaSelector().select("unrecognized vocabulary", schema) == schema
