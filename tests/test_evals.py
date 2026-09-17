"""Tests for the eval harness itself.

A scorer that mis-grades is worse than no scorer — it produces a number people
trust. These run offline against synthetic outcomes.
"""

from __future__ import annotations

import pytest

from evals.cases import CASES, EvalCase, Expect
from evals.runner import Outcome, grade, run


def case(expect: Expect, **kw) -> EvalCase:
    return EvalCase(kw.pop("id", "t1"), kw.pop("question", "q?"), "test", expect, **kw)


def answered(rows, answer="", **kw) -> Outcome:
    return Outcome("answered", answer=answer, rows=rows, row_count=len(rows), **kw)


# --- value matching -----------------------------------------------------


def test_integer_values_must_match_exactly():
    assert grade(case(Expect(values=(111,))), answered([{"n": 111}])) == []
    assert grade(case(Expect(values=(111,))), answered([{"n": 112}])) != []


def test_float_values_honour_tolerance():
    expect = Expect(values=(19.16,), tolerance=0.05)
    assert grade(case(expect), answered([{"avg": 19.158410}])) == []
    assert grade(case(expect), answered([{"avg": 19.9}])) != []


def test_string_values_match_case_insensitive_substrings():
    assert grade(case(Expect(values=("technical",))), answered([{"c": "Technical"}])) == []


def test_values_are_searched_across_every_cell_and_row():
    outcome = answered([{"agent": "AGT-01", "n": 5}, {"agent": "AGT-12", "n": 37}])
    assert grade(case(Expect(values=(37, "AGT-12"))), outcome) == []


def test_missing_value_is_reported_with_the_expected_figure():
    problems = grade(case(Expect(values=(500,))), answered([{"n": 111}]))
    assert len(problems) == 1 and "500" in problems[0]


# --- row counts ---------------------------------------------------------


def test_exact_row_count():
    assert grade(case(Expect(rows=2)), answered([{"a": 1}, {"a": 2}])) == []
    assert grade(case(Expect(rows=3)), answered([{"a": 1}, {"a": 2}])) != []


def test_min_rows():
    assert grade(case(Expect(min_rows=2)), answered([{"a": 1}, {"a": 2}])) == []
    assert grade(case(Expect(min_rows=5)), answered([{"a": 1}])) != []


# --- prose assertions ---------------------------------------------------


def test_answer_mentions_are_case_insensitive():
    outcome = answered([{"n": 1}], answer="Agent AGT-08 has the lowest rating.")
    assert grade(case(Expect(answer_mentions=("agt-08",))), outcome) == []


def test_missing_mention_fails_even_when_the_numbers_are_right():
    """The tie case depends on this: correct value, incomplete prose."""
    outcome = answered([{"r": 3.48}], answer="AGT-08 has the lowest rating at 3.48.")
    problems = grade(case(Expect(values=(3.48,), answer_mentions=("AGT-11",))), outcome)
    assert len(problems) == 1 and "AGT-11" in problems[0]


# --- refusals -----------------------------------------------------------


def test_expected_refusal_passes_only_when_refused():
    refused = Outcome("refused", detail="no such column")
    assert grade(case(Expect(refuses=True)), refused) == []
    assert grade(case(Expect(refuses=True)), answered([{"n": 1}])) != []


def test_unexpected_refusal_is_a_failure():
    problems = grade(case(Expect(values=(1,))), Outcome("refused", detail="nope"))
    assert problems and "unexpectedly refused" in problems[0]


def test_refuses_or_empty_accepts_both_shapes():
    expect = Expect(refuses_or_empty=True)
    assert grade(case(expect), Outcome("refused")) == []
    assert grade(case(expect), answered([])) == []
    assert grade(case(expect), answered([{"n": 0}])) == [], "a zero aggregate is empty"
    assert grade(case(expect), answered([{"n": 42}])) != []


def test_empty_accepts_both_correct_result_shapes():
    """'Which tickets match X' with no matches is either 0 rows or a COUNT of 0."""
    expect = Expect(empty=True)
    assert grade(case(expect), answered([])) == []
    assert grade(case(expect), answered([{"n": 0}])) == []
    assert grade(case(expect), answered([{"ticket_id": "TKT-001"}])) != []


def test_empty_still_requires_an_answer_not_a_refusal():
    problems = grade(case(Expect(empty=True)), Outcome("refused", detail="nope"))
    assert problems and "unexpectedly refused" in problems[0]


def test_transport_error_fails_every_expectation():
    for expect in (Expect(refuses=True), Expect(values=(1,)), Expect(refuses_or_empty=True)):
        assert grade(case(expect), Outcome("error", detail="connection refused")) != []


# --- aggregation over trials -------------------------------------------


def test_flaky_case_is_neither_passed_nor_failed_silently():
    outcomes = iter([answered([{"n": 111}]), answered([{"n": 999}])])
    report = run(lambda _q: next(outcomes), (case(Expect(values=(111,))),), trials=2)

    result = report.results[0]
    assert result.pass_rate == 0.5
    assert result.flaky is True
    assert result.passed is False


def test_known_gaps_are_excluded_from_the_gate_but_not_the_headline():
    cases = (
        case(Expect(values=(1,)), id="good"),
        case(Expect(values=(1,)), id="gap", known_gap=True),
    )
    outcomes = iter([answered([{"n": 1}]), answered([{"n": 2}])])
    report = run(lambda _q: next(outcomes), cases)

    assert report.accuracy == 0.5, "headline accuracy counts the gap honestly"
    assert report.accuracy_excluding_known_gaps == 1.0, "the gate ignores it"


def test_report_serialises_for_diffing():
    report = run(lambda _q: answered([{"n": 1}]), (case(Expect(values=(1,))),))
    payload = report.to_dict()
    assert payload["accuracy"] == 1.0
    assert payload["cases"][0]["id"] == "t1"
    assert "by_category" in payload and "latency_p95_s" in payload


# --- the golden set itself ----------------------------------------------


def test_case_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_asserts_something():
    """A case with an empty Expect would pass unconditionally and inflate the score."""
    for c in CASES:
        e = c.expect
        assert (
            e.refuses or e.refuses_or_empty or e.empty or e.values or e.answer_mentions
            or e.rows is not None or e.min_rows is not None
        ), f"{c.id} asserts nothing"


@pytest.mark.parametrize("c", CASES, ids=lambda c: c.id)
def test_refusal_cases_do_not_also_assert_values(c):
    """Contradictory expectations can never pass; catch them at author time."""
    if c.expect.refuses:
        assert not c.expect.values and not c.expect.answer_mentions
