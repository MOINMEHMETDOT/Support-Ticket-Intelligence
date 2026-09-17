"""The NL pipeline, driven by a scripted fake LLM — no network, no flakiness."""

from __future__ import annotations

import json

import pytest

from app.query.engine import QueryEngine, UnanswerableQuestion
from app.query.guards import SQLValidationError
from tests.conftest import FakeLLM, sql_reply


def engine_with(store, settings, responses):
    return QueryEngine(store, FakeLLM(responses), settings)


def test_happy_path_returns_computed_numbers(store, settings):
    engine = engine_with(
        store,
        settings,
        [sql_reply("SELECT COUNT(*) AS open FROM tickets WHERE status = 'Open'"), "There are 111 open tickets."],
    )
    result = engine.ask("How many tickets are open?")
    assert result.rows == [{"open": 111}]
    assert result.confidence == "high"
    assert result.repaired is False


def test_broken_sql_is_repaired_once(store, settings):
    engine = engine_with(
        store,
        settings,
        [
            sql_reply("SELECT COUNT(*) FROM tickets WHERE stat = 'Open'"),  # bad column
            sql_reply("SELECT COUNT(*) AS n FROM tickets WHERE status = 'Open'"),
            "There are 111 open tickets.",
        ],
    )
    result = engine.ask("How many tickets are open?")
    assert result.repaired is True
    assert result.rows == [{"n": 111}]
    assert any("repaired" in w for w in result.warnings)


def test_repair_prompt_includes_the_database_error(store, settings):
    llm = FakeLLM(
        [
            sql_reply("SELECT nope FROM tickets"),
            sql_reply("SELECT COUNT(*) AS n FROM tickets"),
            "500 tickets.",
        ]
    )
    QueryEngine(store, llm, settings).ask("How many tickets?")
    repair_prompt = llm.calls[1][1]
    assert "DATABASE ERROR" in repair_prompt
    assert "nope" in repair_prompt


def test_unanswerable_question_is_surfaced_not_guessed(store, settings):
    engine = engine_with(
        store, settings, [json.dumps({"error": "No contact details in this dataset."})]
    )
    with pytest.raises(UnanswerableQuestion, match="contact details"):
        engine.ask("What is the customer's phone number?")


def test_unsafe_sql_from_model_is_blocked(store, settings):
    engine = engine_with(store, settings, [sql_reply("DROP TABLE tickets")])
    with pytest.raises(SQLValidationError):
        engine.ask("Delete everything")


def test_unparseable_model_output_raises_clearly(store, settings):
    engine = engine_with(store, settings, ["I'm not going to answer that."])
    with pytest.raises(SQLValidationError, match="Could not parse"):
        engine.ask("How many tickets?")


def test_json_wrapped_in_prose_is_still_parsed(store, settings):
    engine = engine_with(
        store,
        settings,
        [
            "Sure! ```json\n" + sql_reply("SELECT COUNT(*) AS n FROM tickets") + "\n``` Hope that helps!",
            "500 tickets.",
        ],
    )
    assert engine.ask("How many tickets?").rows == [{"n": 500}]


def test_narration_failure_still_returns_the_numbers(store, settings):
    """If the prose step dies, a correct result must not be thrown away."""

    class HalfBrokenLLM(FakeLLM):
        def complete(self, system, user, *, temperature=0.0):
            if "explain the result" in system:
                from app.llm.base import LLMError

                raise LLMError("rate limited")
            return super().complete(system, user)

    engine = QueryEngine(
        store,
        HalfBrokenLLM([sql_reply("SELECT COUNT(*) AS n FROM tickets")]),
        settings,
    )
    result = engine.ask("How many tickets?")
    assert result.rows == [{"n": 500}]
    assert "rate limited" in result.answer


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_rejected(store, settings, question):
    with pytest.raises(ValueError, match="must not be empty"):
        engine_with(store, settings, []).ask(question)


def test_overlong_question_rejected(store, settings):
    with pytest.raises(ValueError, match="too long"):
        engine_with(store, settings, []).ask("x" * 1001)


def test_reference_date_is_injected_into_the_prompt(store, settings):
    llm = FakeLLM([sql_reply("SELECT COUNT(*) AS n FROM tickets"), "500."])
    QueryEngine(store, llm, settings).ask("How many tickets this month?")
    system_prompt = llm.calls[0][0]
    assert store.as_of.strftime("%Y-%m-%d") in system_prompt
    assert "REFERENCE DATE" in system_prompt
