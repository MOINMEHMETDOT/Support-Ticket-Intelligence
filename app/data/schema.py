"""Single source of truth for the ticket schema.

Both the SQL-generation prompt and the anomaly detectors read from here, so the
model's picture of the data and the code's picture cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

TABLE_NAME = "tickets"


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str
    description: str


COLUMNS: tuple[Column, ...] = (
    Column("ticket_id", "TEXT", "Unique ticket identifier, e.g. 'TKT-001'."),
    Column(
        "created_at",
        "TEXT",
        "Ticket creation timestamp as 'YYYY-MM-DD HH:MM:SS'. Compare with SQLite "
        "date functions, e.g. date(created_at) >= date('2024-03-01').",
    ),
    Column("category", "TEXT", "One of: Billing, Technical, General."),
    Column("priority", "TEXT", "One of: Low, Medium, High, Critical."),
    Column("status", "TEXT", "One of: Open, Resolved, Escalated."),
    Column("response_time_hrs", "REAL", "Hours from creation to first agent response. Never NULL."),
    Column(
        "resolution_time_hrs",
        "REAL",
        "Hours from creation to resolution. NULL for every Open and Escalated "
        "ticket — a NULL means 'not resolved yet', not 'missing data'.",
    ),
    Column("agent_id", "TEXT", "Assigned agent, 'AGT-01' through 'AGT-12'."),
    Column(
        "customer_rating",
        "INTEGER",
        "Post-resolution satisfaction, 1-5. NULL for every Open and Escalated "
        "ticket, for the same reason as resolution_time_hrs.",
    ),
    Column("issue_summary", "TEXT", "Short free-text description of the issue."),
    Column(
        "age_hrs",
        "REAL",
        "Derived: hours between created_at and the reference date (see the "
        "REFERENCE DATE note). Use this for questions about how long a ticket "
        "has been sitting unresolved.",
    ),
)

# Values the LLM may legitimately filter on. Used to spot typos / hallucinated
# categories before they silently return an empty result set.
CATEGORIES = ("Billing", "Technical", "General")
PRIORITIES = ("Low", "Medium", "High", "Critical")
STATUSES = ("Open", "Resolved", "Escalated")

UNRESOLVED_STATUSES = ("Open", "Escalated")

# Internal SLA targets in hours, by priority. Not supplied with the dataset —
# these are our stated assumption, and every SLA-based anomaly cites them.
SLA_HOURS: dict[str, float] = {
    "Critical": 4.0,
    "High": 12.0,
    "Medium": 24.0,
    "Low": 72.0,
}


def create_table_sql() -> str:
    cols = ",\n  ".join(f"{c.name} {c.sql_type}" for c in COLUMNS)
    return f"CREATE TABLE {TABLE_NAME} (\n  {cols}\n);"


def schema_for_prompt() -> str:
    """Human-readable schema block injected into the SQL-generation prompt."""
    lines = [f"TABLE {TABLE_NAME}("]
    for c in COLUMNS:
        lines.append(f"  {c.name} {c.sql_type}  -- {c.description}")
    lines.append(")")
    return "\n".join(lines)
