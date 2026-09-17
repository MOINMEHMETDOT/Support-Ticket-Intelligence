"""Validation for LLM-generated SQL.

Treat model output as untrusted input. Three layers protect the database:
this module (string validation), a read-only SQLite connection, and a sqlite3
authorizer callback (see data/loader.py). Any one of them would stop a write;
all three are cheap.
"""

from __future__ import annotations

import re

from app.data.schema import TABLE_NAME

_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "attach", "detach", "pragma", "vacuum", "reindex",
    "sqlite_master", "load_extension",
)

_STRING_LITERAL = re.compile(r"'[^']*'")
_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)
_LIMIT_TAIL = re.compile(r"\blimit\s+\d+\s*(?:offset\s+\d+\s*)?$", re.IGNORECASE)
_IDENTIFIER = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)


class SQLValidationError(ValueError):
    """Generated SQL was unsafe or malformed."""


def validate_sql(raw: str, *, max_rows: int) -> str:
    """Return runnable SQL, or raise SQLValidationError.

    Also appends a LIMIT when the model omitted one, so a `SELECT *` cannot
    pull the whole table into a prompt.
    """
    sql = _FENCE.sub("", (raw or "").strip()).strip()
    if not sql:
        raise SQLValidationError("The model returned an empty query.")

    sql = sql.rstrip(";").strip()

    # Comments can hide a second statement; strip them before any other check.
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL).strip()

    if ";" in sql:
        raise SQLValidationError("Only a single SQL statement is allowed.")

    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise SQLValidationError("Only SELECT queries are allowed.")

    # Scan for dangerous keywords with string literals blanked out, so a ticket
    # summary containing the word "update" doesn't trip the guard.
    scannable = _STRING_LITERAL.sub("''", sql).lower()
    for keyword in _FORBIDDEN:
        if re.search(rf"\b{keyword}\b", scannable):
            raise SQLValidationError(f"Disallowed keyword in generated SQL: {keyword!r}.")

    referenced = {name.lower() for name in _IDENTIFIER.findall(scannable)}
    # CTE names are legal targets too; collect them before rejecting unknowns.
    cte_names = {n.lower() for n in re.findall(r"\b([a-zA-Z_]\w*)\s+as\s*\(", scannable)}
    unknown = referenced - {TABLE_NAME.lower()} - cte_names
    if unknown:
        raise SQLValidationError(
            f"Query references unknown table(s): {', '.join(sorted(unknown))}. "
            f"Only {TABLE_NAME!r} exists."
        )

    if not _LIMIT_TAIL.search(sql):
        sql = f"{sql} LIMIT {max_rows}"

    return sql
