"""Execution and scoring for the golden set.

The runner takes an `ask` callable rather than building an engine itself, so the
same scoring logic evaluates the live HTTP API, an in-process engine, or a fake
in unit tests. Nothing here knows which.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from evals.cases import CASES, EvalCase, Expect

REPORT_DIR = Path(__file__).resolve().parent / "reports"

Status = Literal["answered", "refused", "error"]


@dataclass
class Outcome:
    """What the system did with one question."""

    status: Status
    answer: str = ""
    sql: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    detail: str = ""
    confidence: str = ""
    repaired: bool = False
    latency_s: float = 0.0


Ask = Callable[[str], Outcome]


# --- scoring ------------------------------------------------------------


def _cells(rows: list[dict[str, Any]]) -> list[Any]:
    return [value for row in rows for value in row.values()]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches(expected: Any, cells: list[Any], tolerance: float) -> bool:
    """int -> exact; float -> within tolerance; str -> case-insensitive substring."""
    if isinstance(expected, bool):
        return any(cell is expected for cell in cells)
    if isinstance(expected, int):
        return any(_is_number(c) and float(c) == float(expected) for c in cells)
    if isinstance(expected, float):
        return any(_is_number(c) and abs(float(c) - expected) <= tolerance for c in cells)
    return any(str(expected).lower() in str(c).lower() for c in cells)


def _looks_empty(outcome: Outcome) -> bool:
    """No rows, or a single aggregate that came back zero."""
    if outcome.row_count == 0:
        return True
    cells = _cells(outcome.rows)
    return bool(cells) and all(_is_number(c) and float(c) == 0.0 for c in cells)


def grade(case: EvalCase, outcome: Outcome) -> list[str]:
    """Return the reasons this case failed. Empty list means it passed."""
    expect: Expect = case.expect

    if outcome.status == "error":
        return [f"request errored: {outcome.detail[:200]}"]

    if expect.refuses:
        if outcome.status == "refused":
            return []
        return [f"expected a refusal, got an answer: {outcome.answer[:140]}"]

    if expect.refuses_or_empty:
        if outcome.status == "refused" or _looks_empty(outcome):
            return []
        return [f"expected a refusal or an empty result, got {outcome.row_count} row(s)"]

    if outcome.status == "refused":
        return [f"unexpectedly refused: {outcome.detail[:200]}"]

    if expect.empty:
        if _looks_empty(outcome):
            return []
        return [f"expected an empty result, got {outcome.row_count} row(s): {outcome.answer[:120]}"]

    problems: list[str] = []

    if expect.rows is not None and outcome.row_count != expect.rows:
        problems.append(f"expected {expect.rows} row(s), got {outcome.row_count}")
    if expect.min_rows is not None and outcome.row_count < expect.min_rows:
        problems.append(f"expected at least {expect.min_rows} row(s), got {outcome.row_count}")

    cells = _cells(outcome.rows)
    for expected in expect.values:
        if not _matches(expected, cells, expect.tolerance):
            problems.append(f"expected value {expected!r} absent from the result")

    lowered = outcome.answer.lower()
    for mention in expect.answer_mentions:
        if mention.lower() not in lowered:
            problems.append(f"answer never mentions {mention!r}")

    return problems


# --- results ------------------------------------------------------------


@dataclass
class Trial:
    passed: bool
    problems: list[str]
    latency_s: float
    sql: str = ""
    answer: str = ""


@dataclass
class CaseResult:
    case: EvalCase
    trials: list[Trial]

    @property
    def pass_rate(self) -> float:
        return sum(t.passed for t in self.trials) / len(self.trials)

    @property
    def passed(self) -> bool:
        return self.pass_rate == 1.0

    @property
    def flaky(self) -> bool:
        """Passed sometimes. Worth separating: an intermittent failure in an LLM
        system is a different problem from a consistent one, and averaging them
        into a single accuracy number hides it."""
        return 0.0 < self.pass_rate < 1.0

    @property
    def latency_s(self) -> float:
        return statistics.median(t.latency_s for t in self.trials)

    @property
    def problems(self) -> list[str]:
        for trial in self.trials:
            if not trial.passed:
                return trial.problems
        return []


@dataclass
class Report:
    results: list[CaseResult]
    provider: str
    model: str
    trials: int
    duration_s: float
    started_at: str

    # -- headline numbers ------------------------------------------------

    @property
    def scored(self) -> list[CaseResult]:
        """Cases excluding documented known gaps — what CI gates on."""
        return [r for r in self.results if not r.case.known_gap]

    @property
    def accuracy(self) -> float:
        return _ratio(sum(r.passed for r in self.results), len(self.results))

    @property
    def accuracy_excluding_known_gaps(self) -> float:
        scored = self.scored
        return _ratio(sum(r.passed for r in scored), len(scored))

    @property
    def flaky(self) -> list[CaseResult]:
        return [r for r in self.results if r.flaky]

    def by_category(self) -> dict[str, tuple[int, int]]:
        table: dict[str, tuple[int, int]] = {}
        for result in self.results:
            done, total = table.get(result.case.category, (0, 0))
            table[result.case.category] = (done + int(result.passed), total + 1)
        return table

    def latency(self, percentile: float) -> float:
        values = sorted(r.latency_s for r in self.results)
        if not values:
            return 0.0
        index = min(int(percentile * len(values)), len(values) - 1)
        return values[index]

    # -- output ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "provider": self.provider,
            "model": self.model,
            "trials_per_case": self.trials,
            "duration_s": round(self.duration_s, 1),
            "accuracy": round(self.accuracy, 4),
            "accuracy_excluding_known_gaps": round(self.accuracy_excluding_known_gaps, 4),
            "latency_p50_s": round(self.latency(0.50), 2),
            "latency_p95_s": round(self.latency(0.95), 2),
            "by_category": {
                name: {"passed": p, "total": t, "accuracy": round(_ratio(p, t), 4)}
                for name, (p, t) in sorted(self.by_category().items())
            },
            "cases": [
                {
                    "id": r.case.id,
                    "category": r.case.category,
                    "question": r.case.question,
                    "passed": r.passed,
                    "pass_rate": round(r.pass_rate, 3),
                    "flaky": r.flaky,
                    "known_gap": r.case.known_gap,
                    "latency_s": round(r.latency_s, 2),
                    "problems": r.problems,
                    "sql": r.trials[0].sql,
                    "answer": r.trials[0].answer,
                }
                for r in self.results
            ],
        }

    def save(self, directory: Path = REPORT_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "").replace("T", "-")[:15]
        path = directory / f"eval-{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


# --- execution ----------------------------------------------------------


def run(
    ask: Ask,
    cases: tuple[EvalCase, ...] = CASES,
    *,
    trials: int = 1,
    provider: str = "unknown",
    model: str = "unknown",
    delay_s: float = 0.0,
    on_case: Callable[[CaseResult], None] | None = None,
) -> Report:
    """Run every case `trials` times and score the outcomes.

    Cases run sequentially on purpose: these suites are usually pointed at a
    free-tier key, and a burst of parallel requests buys a 429 instead of a
    faster result.
    """
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    clock = time.perf_counter()
    results: list[CaseResult] = []

    for case in cases:
        attempts: list[Trial] = []
        for _ in range(trials):
            outcome = ask(case.question)
            problems = grade(case, outcome)
            attempts.append(
                Trial(
                    passed=not problems,
                    problems=problems,
                    latency_s=outcome.latency_s,
                    sql=outcome.sql,
                    answer=outcome.answer or outcome.detail,
                )
            )
            if delay_s:
                time.sleep(delay_s)

        result = CaseResult(case=case, trials=attempts)
        results.append(result)
        if on_case:
            on_case(result)

    return Report(
        results=results,
        provider=provider,
        model=model,
        trials=trials,
        duration_s=time.perf_counter() - clock,
        started_at=started,
    )


__all__ = ["Outcome", "Report", "CaseResult", "Trial", "grade", "run", "asdict"]
