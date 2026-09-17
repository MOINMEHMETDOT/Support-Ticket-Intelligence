"""CSV ingestion into a queryable, read-only SQLite store.

Why SQLite rather than a vector store: every question in the brief is an
aggregate over structured columns ("how many", "average rating", "lowest").
Retrieving semantically similar rows and asking an LLM to count them gives
confidently wrong numbers. Translating to SQL and letting the database compute
the answer is exact and auditable.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import PROJECT_ROOT, Settings
from app.data.schema import COLUMNS, TABLE_NAME, create_table_sql

DB_PATH = PROJECT_ROOT / "data" / "tickets.db"

# sqlite3 authorizer actions that a SELECT legitimately needs. Anything else —
# inserts, updates, drops, ATTACH, pragma writes — is denied at the C level, so
# a malicious or confused LLM cannot mutate the database even if the string
# validation in query/guards.py were bypassed.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


class QueryExecutionError(RuntimeError):
    """Raised when generated SQL fails to execute or is blocked by the guard."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_markdown(self, limit: int = 20) -> str:
        """Compact rendering used to show the LLM what the query returned."""
        if not self.rows:
            return "(no rows)"
        head = self.rows[:limit]
        widths = [
            max(len(str(c)), *(len(str(r[i])) for r in head))
            for i, c in enumerate(self.columns)
        ]
        fmt = " | ".join(f"{{:<{w}}}" for w in widths)
        lines = [fmt.format(*self.columns), "-+-".join("-" * w for w in widths)]
        lines += [fmt.format(*("" if v is None else str(v) for v in r)) for r in head]
        if self.row_count > limit:
            lines.append(f"... ({self.row_count - limit} more rows)")
        return "\n".join(lines)


class TicketStore:
    """Owns the loaded dataset: a DataFrame for analytics, SQLite for NL queries."""

    def __init__(self, frame: pd.DataFrame, as_of: datetime, db_path: Path, settings: Settings):
        self.frame = frame
        self.as_of = as_of
        self.db_path = db_path
        self._settings = settings

    # -- construction ----------------------------------------------------

    @classmethod
    def from_csv(cls, settings: Settings, db_path: Path = DB_PATH) -> TicketStore:
        if not settings.data_csv.exists():
            raise FileNotFoundError(
                f"Dataset not found at {settings.data_csv}. "
                "Set DATA_CSV in .env or place support_tickets.csv under data/."
            )

        frame = pd.read_csv(settings.data_csv, parse_dates=["created_at"])
        _validate_columns(frame)

        # The dataset ends in March 2024, so datetime.now() would make every
        # relative-date question ("this week") return nothing. Anchor to the
        # newest ticket unless the operator pins a date explicitly.
        as_of = settings.as_of_date or frame["created_at"].max().to_pydatetime()

        frame = frame.copy()
        frame["age_hrs"] = (
            (pd.Timestamp(as_of) - frame["created_at"]).dt.total_seconds() / 3600.0
        ).round(2)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        cls._build_db(frame, db_path)
        return cls(frame=frame, as_of=as_of, db_path=db_path, settings=settings)

    @staticmethod
    def _build_db(frame: pd.DataFrame, db_path: Path) -> None:
        if db_path.exists():
            db_path.unlink()
        sql_frame = frame.copy()
        sql_frame["created_at"] = sql_frame["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(db_path) as conn:
            conn.execute(create_table_sql())
            ordered = [c.name for c in COLUMNS]
            sql_frame[ordered].to_sql(TABLE_NAME, conn, if_exists="append", index=False)
            conn.execute(f"CREATE INDEX idx_status ON {TABLE_NAME}(status)")
            conn.execute(f"CREATE INDEX idx_priority ON {TABLE_NAME}(priority)")
            conn.execute(f"CREATE INDEX idx_agent ON {TABLE_NAME}(agent_id)")

    # -- querying --------------------------------------------------------

    def execute_readonly(self, sql: str) -> QueryResult:
        """Run already-validated SQL against a read-only connection.

        Callers must pass SQL through query.guards.validate_sql first; the
        authorizer here is the second line of defence, not the first.
        """
        limit = self._settings.sql_max_rows
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            timeout=self._settings.sql_timeout_seconds,
        )
        try:
            conn.set_authorizer(_authorizer)
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(limit + 1)
            columns = [d[0] for d in cursor.description or []]
        except sqlite3.DatabaseError as exc:
            raise QueryExecutionError(str(exc)) from exc
        finally:
            conn.close()

        truncated = len(rows) > limit
        return QueryResult(columns=columns, rows=rows[:limit], truncated=truncated)

    # -- descriptive helpers ---------------------------------------------

    def profile(self) -> dict[str, Any]:
        """Dataset facts surfaced by /health and shown in the UI sidebar."""
        df = self.frame
        return {
            "rows": int(len(df)),
            "date_range": {
                "start": df["created_at"].min().strftime("%Y-%m-%d %H:%M"),
                "end": df["created_at"].max().strftime("%Y-%m-%d %H:%M"),
            },
            "as_of": self.as_of.strftime("%Y-%m-%d %H:%M"),
            "status_counts": df["status"].value_counts().to_dict(),
            "priority_counts": df["priority"].value_counts().to_dict(),
            "category_counts": df["category"].value_counts().to_dict(),
            "agents": int(df["agent_id"].nunique()),
            "unresolved": int(df["resolution_time_hrs"].isna().sum()),
        }


def _validate_columns(frame: pd.DataFrame) -> None:
    expected = {c.name for c in COLUMNS} - {"age_hrs"}  # age_hrs is derived
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Found: {', '.join(frame.columns)}"
        )
