from research.slm_sql.policy_view import policy_view


def test_policy_view_flattens_nested_success_and_empty_execution():
    rows = policy_view([
        {"sql": "SELECT 1", "execution": {"fingerprint": "fp", "row_count": 1}},
        {"sql": "SELECT 2", "execution": {"fingerprint": "empty", "row_count": 0}},
    ])
    assert rows[0]["fingerprint"] == "fp"
    assert rows[0]["error"] is None
    assert not rows[0]["is_empty"]
    assert rows[1]["fingerprint"] == "empty"
    assert rows[1]["is_empty"]


def test_policy_view_preserves_nested_errors_parse_errors_and_legacy_fields():
    rows = policy_view([
        {"sql": "SELECT", "execution": {"fingerprint": None, "error": "syntax"}},
        {"sql": "bad output", "parse_error": "missing answer"},
        {"sql": "SELECT 1", "fingerprint": "legacy", "row_count": 1},
    ])
    assert rows[0]["error"] == "syntax"
    assert rows[0]["fingerprint"] is None
    assert rows[1]["error"] == "missing answer"
    assert rows[1]["fingerprint"] is None
    assert rows[2]["fingerprint"] == "legacy"
    assert rows[2]["error"] is None
