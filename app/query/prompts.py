"""Prompt construction for the two LLM steps: question -> SQL, result -> prose.

Both prompts are built from app.data.schema, so adding a column updates the
model's view of the data automatically.
"""

from __future__ import annotations

from datetime import datetime

from app.data.schema import (
    CATEGORIES,
    PRIORITIES,
    SLA_HOURS,
    STATUSES,
    TABLE_NAME,
    schema_for_prompt,
)

# Few-shot examples chosen to cover the failure modes seen while testing:
# null handling on unresolved tickets, relative dates, ranking with a tie-break,
# and "not resolved within N hours" which reads as a filter but is a comparison
# against a nullable column.
_EXAMPLES = """\
Q: How many tickets are currently open?
{"sql": "SELECT COUNT(*) AS open_tickets FROM tickets WHERE status = 'Open'", "explanation": "Counts tickets in the Open status.", "confidence": "high"}

Q: What is the average customer rating for Technical category tickets?
{"sql": "SELECT ROUND(AVG(customer_rating), 2) AS avg_rating, COUNT(customer_rating) AS rated_tickets FROM tickets WHERE category = 'Technical'", "explanation": "AVG skips NULL ratings, so this averages only the resolved Technical tickets; rated_tickets shows how many that was.", "confidence": "high"}

Q: Which agent has the lowest average customer rating?
{"sql": "SELECT agent_id, ROUND(AVG(customer_rating), 2) AS avg_rating, COUNT(customer_rating) AS rated_tickets FROM tickets WHERE customer_rating IS NOT NULL GROUP BY agent_id ORDER BY avg_rating ASC, rated_tickets DESC LIMIT 5", "explanation": "Ranks agents by mean rating over rated tickets; returns the top 5 so the margin over the runner-up is visible.", "confidence": "high"}

Q: Show me all Critical tickets not resolved within 12 hours.
{"sql": "SELECT ticket_id, created_at, status, resolution_time_hrs, age_hrs, agent_id FROM tickets WHERE priority = 'Critical' AND (resolution_time_hrs > 12 OR (resolution_time_hrs IS NULL AND age_hrs > 12)) ORDER BY age_hrs DESC", "explanation": "Covers both senses of 'not resolved within 12 hours': resolved but slower than 12h, and still unresolved after 12h.", "confidence": "high"}

Q: Which agent resolved the most tickets this month?
{"sql": "SELECT agent_id, COUNT(*) AS resolved FROM tickets WHERE status = 'Resolved' AND strftime('%Y-%m', created_at) = strftime('%Y-%m', 'REFERENCE_DATE') GROUP BY agent_id ORDER BY resolved DESC LIMIT 5", "explanation": "'This month' is the month of the reference date, not the real-world current month.", "confidence": "medium"}

Q: What is the customer's phone number for TKT-004?
{"error": "The dataset has no contact details - only ticket_id, timestamps, category, priority, status, response and resolution times, agent_id, customer_rating and issue_summary."}
"""


def sql_system_prompt(as_of: datetime) -> str:
    reference = as_of.strftime("%Y-%m-%d %H:%M:%S")
    sla = ", ".join(f"{k} {v:g}h" for k, v in SLA_HOURS.items())
    return f"""\
You translate questions about a customer-support ticket dataset into a single \
SQLite SELECT query.

SCHEMA
{schema_for_prompt()}

LEGAL VALUES (match exactly, they are case-sensitive)
  category: {', '.join(CATEGORIES)}
  priority: {', '.join(PRIORITIES)}
  status:   {', '.join(STATUSES)}

REFERENCE DATE
  Treat '{reference}' as "now". The dataset is historical, so "today",
  "this week", "this month" and "recently" are all relative to that instant,
  never to the real-world clock. In SQL, write the literal '{reference}'
  wherever you need the current time. The age_hrs column is already computed
  against it.

  Relative day windows use CALENDAR-DAY boundaries, not rolling hours:
    "the last 7 days"  ->  date(created_at) >= date('{reference}', '-7 days')
  Not datetime(...,'-7 days'), which measures a rolling 168 hours from the
  current time of day and answers the same question with a different number.
  Pick the calendar reading every time so the answer is reproducible.

CRITICAL SEMANTICS
  - resolution_time_hrs and customer_rating are NULL for every Open and
    Escalated ticket. NULL means "not resolved yet", not "unknown".
  - Because of that, AVG(customer_rating) and AVG(resolution_time_hrs) silently
    describe resolved tickets only. That is usually correct, but always return
    a COUNT alongside the average so the answer can state its own sample size.
  - "Unresolved" means status IN ('Open','Escalated').
  - "Not resolved within N hours" has two halves: resolved but slower than N,
    and still unresolved after N. Include both unless the question rules one out.
  - Internal SLA targets, used when a question says "overdue" or "breached
    SLA": {sla}.

RULES
  - One SELECT statement. Never INSERT, UPDATE, DELETE, DROP, ATTACH or PRAGMA.
  - Query only the {TABLE_NAME} table.
  - Alias every computed column with a readable name.
  - ROUND averages and times to 2 decimal places.
  - Add LIMIT 200 to queries that return rows rather than aggregates.
  - If the question cannot be answered from these columns, do not invent a
    query - return the error form instead.

OUTPUT
  Return raw JSON and nothing else. No prose, no markdown fences.
  Success: {{"sql": "...", "explanation": "one sentence on what the query does and any assumption you made", "confidence": "high" | "medium" | "low"}}
  Unanswerable: {{"error": "one sentence on what is missing from the data"}}
  Use "medium" or "low" confidence when the question is ambiguous, and say why
  in the explanation.

EXAMPLES
{_EXAMPLES.replace('REFERENCE_DATE', reference)}"""


def answer_system_prompt(as_of: datetime) -> str:
    return f"""\
You explain the result of a database query to a support operations manager.

You are given the user's question, the SQL that ran, and the rows it returned.
Write the answer as 1-3 plain sentences.

RULES
  - Use only numbers that appear in the result rows. Never estimate, extrapolate
    or recall figures from anywhere else.
  - Lead with the direct answer, then add context only if it changes the reading.
  - If a rate or average covers only part of the dataset (for example resolved
    tickets, because unresolved ones have no rating), say so.
  - If there are no rows, say plainly that nothing matched and, in one clause,
    what that means.
  - If the query returned a ranking, name the top entry and its margin over the
    next one.
  - The reference date for anything time-relative is {as_of:%Y-%m-%d}.
  - No markdown, no bullet points, no preamble like "Based on the data".
"""


def answer_user_prompt(question: str, sql: str, table: str, truncated: bool) -> str:
    note = (
        "\n\nNOTE: the result was truncated to the first rows; "
        "do not state a total row count.\n"
        if truncated
        else ""
    )
    return f"QUESTION\n{question}\n\nSQL\n{sql}\n\nRESULT\n{table}{note}"
