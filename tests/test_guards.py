"""The SQL guard is the trust boundary between model output and the database."""

from __future__ import annotations

import pytest

from app.query.guards import SQLValidationError, validate_sql

MAX = 100


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE tickets",
        "DELETE FROM tickets WHERE 1=1",
        "UPDATE tickets SET status = 'Resolved'",
        "INSERT INTO tickets VALUES ('x')",
        "PRAGMA table_info(tickets)",
        "ATTACH DATABASE '/etc/passwd' AS leak",
        "SELECT * FROM tickets; DROP TABLE tickets",
        "SELECT name FROM sqlite_master",
        "SELECT * FROM users",
    ],
)
def test_rejects_unsafe_sql(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql, max_rows=MAX)


def test_rejects_empty():
    with pytest.raises(SQLValidationError):
        validate_sql("   ", max_rows=MAX)


def test_strips_markdown_fences():
    cleaned = validate_sql("```sql\nSELECT COUNT(*) FROM tickets\n```", max_rows=MAX)
    assert cleaned.startswith("SELECT COUNT(*)")
    assert "```" not in cleaned


def test_appends_limit_when_missing():
    assert validate_sql("SELECT * FROM tickets", max_rows=MAX).endswith("LIMIT 100")


def test_keeps_existing_limit():
    cleaned = validate_sql("SELECT * FROM tickets LIMIT 5", max_rows=MAX)
    assert cleaned.endswith("LIMIT 5")
    assert cleaned.count("LIMIT") == 1


def test_allows_cte():
    sql = (
        "WITH per_agent AS (SELECT agent_id, AVG(customer_rating) r FROM tickets "
        "GROUP BY agent_id) SELECT * FROM per_agent ORDER BY r ASC"
    )
    assert validate_sql(sql, max_rows=MAX).startswith("WITH")


def test_keyword_inside_string_literal_is_allowed():
    """A ticket summary containing 'delete' must not trip the keyword scan."""
    sql = "SELECT * FROM tickets WHERE issue_summary = 'Request for account delete'"
    assert "issue_summary" in validate_sql(sql, max_rows=MAX)


def test_comment_hiding_second_statement_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1 FROM tickets -- x\n; DROP TABLE tickets", max_rows=MAX)
