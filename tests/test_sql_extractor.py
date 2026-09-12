import pytest

from core.sql_extractor import extract_sql


@pytest.mark.parametrize("text", [
    "SELECT 1", "<answer>SELECT 1</answer>", "```sql\nSELECT 1\n```",
    "<think>SELECT secret</think><answer>SELECT 1</answer>",
    "<answer>```sql\nSELECT 1\n```</answer>",
])
def test_formats(text):
    assert extract_sql(text) == "SELECT 1"


@pytest.mark.parametrize("text", ["", "<think>SELECT 1", "<answer>SELECT 1</answer><answer>SELECT 2</answer>",
                                 "Let me reason about this. SELECT 1"])
def test_reject_incomplete_or_ambiguous(text):
    with pytest.raises(ValueError):
        extract_sql(text)


def test_preserves_multiple_statements_for_safety_gate():
    assert "DROP" in extract_sql("<answer>SELECT 1; DROP TABLE students;</answer>")
