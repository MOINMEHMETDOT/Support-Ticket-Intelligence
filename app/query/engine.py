"""The natural-language query pipeline.

    question -> LLM -> SQL -> guard -> SQLite -> rows -> LLM -> answer

The LLM appears twice and computes nothing in between. It chooses *what* to
ask the database and *how to phrase* what came back; every number in the final
answer was produced by SQLite. That split is the whole design: language models
are unreliable at arithmetic over hundreds of rows and good at translation, so
they only ever translate here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.data.loader import QueryExecutionError, TicketStore
from app.llm.base import LLMError, LLMProvider
from app.query import prompts
from app.query.guards import SQLValidationError, validate_sql

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class UnanswerableQuestion(ValueError):
    """The model determined the dataset cannot answer this question."""


@dataclass
class QueryAnswer:
    question: str
    answer: str
    sql: str
    explanation: str
    confidence: str
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    repaired: bool = False
    warnings: list[str] = field(default_factory=list)


class QueryEngine:
    def __init__(self, store: TicketStore, llm: LLMProvider, settings: Settings):
        self._store = store
        self._llm = llm
        self._settings = settings

    @property
    def llm(self) -> LLMProvider:
        """The configured provider, for callers that need it directly (health
        probes, the anomaly narrative) without going through ask()."""
        return self._llm

    def ask(self, question: str) -> QueryAnswer:
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if len(question) > 1000:
            raise ValueError("Question is too long (max 1000 characters).")

        plan = self._generate_sql(question)
        warnings: list[str] = []

        sql = validate_sql(plan["sql"], max_rows=self._settings.sql_max_rows)
        repaired = False

        try:
            result = self._store.execute_readonly(sql)
        except QueryExecutionError as first_error:
            # One repair attempt: show the model its own broken SQL and the
            # database's complaint. Most failures are a misremembered column
            # name, which the model fixes reliably when told what went wrong.
            logger.warning("Generated SQL failed, attempting repair: %s", first_error)
            plan = self._generate_sql(question, failed_sql=sql, error=str(first_error))
            sql = validate_sql(plan["sql"], max_rows=self._settings.sql_max_rows)
            result = self._store.execute_readonly(sql)
            repaired = True
            warnings.append("The first generated query failed and was automatically repaired.")

        if result.truncated:
            warnings.append(
                f"Showing the first {self._settings.sql_max_rows} rows; more matched."
            )

        answer = self._narrate(question, sql, result)

        return QueryAnswer(
            question=question,
            answer=answer,
            sql=sql,
            explanation=plan.get("explanation", ""),
            confidence=plan.get("confidence", "unknown"),
            rows=result.to_records(),
            row_count=result.row_count,
            truncated=result.truncated,
            repaired=repaired,
            warnings=warnings,
        )

    # -- steps -----------------------------------------------------------

    def _generate_sql(
        self, question: str, *, failed_sql: str | None = None, error: str | None = None
    ) -> dict[str, Any]:
        user = question
        if failed_sql:
            user = (
                f"{question}\n\n"
                f"Your previous query failed. Return a corrected query.\n"
                f"PREVIOUS SQL:\n{failed_sql}\n"
                f"DATABASE ERROR:\n{error}"
            )

        raw = self._llm.complete(prompts.sql_system_prompt(self._store.as_of), user)
        payload = _extract_json(raw)

        if "error" in payload and "sql" not in payload:
            raise UnanswerableQuestion(str(payload["error"]))
        if not payload.get("sql"):
            raise SQLValidationError(
                f"The model did not return a query. It said: {raw[:200]}"
            )
        return payload

    def _narrate(self, question: str, sql: str, result: Any) -> str:
        """Turn rows into a sentence. Falls back to a plain rendering if the LLM fails."""
        try:
            return self._llm.complete(
                prompts.answer_system_prompt(self._store.as_of),
                prompts.answer_user_prompt(
                    question, sql, result.to_markdown(), result.truncated
                ),
            ).strip()
        except LLMError as exc:
            # The numbers are already correct; only the prose step failed. Degrade
            # rather than discard a valid result.
            logger.warning("Narration step failed, returning raw result: %s", exc)
            return (
                f"(The summarisation step was unavailable: {exc}) "
                f"The query returned {result.row_count} row(s):\n{result.to_markdown()}"
            )


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull a JSON object out of a model response that may carry fences or prose."""
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise SQLValidationError(
        f"Could not parse the model's response as JSON. It returned: {text[:300]}"
    )
