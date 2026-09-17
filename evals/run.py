"""CLI for the evaluation suite.

    python -m evals.run                      # against the running API
    python -m evals.run --direct             # in-process, reads the key from .env
    python -m evals.run --trials 3           # measure consistency, not just accuracy
    python -m evals.run --category ranking   # iterate on one weak spot

API mode is the default because the key may have been supplied at runtime
through the UI, in which case it lives in the server's memory and not in .env.
It also evaluates the surface a real consumer uses.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import httpx

from evals.cases import CASES, EvalCase
from evals.runner import Ask, CaseResult, Outcome, Report, run

# The engine's own wording when it declines a question, used to tell an honest
# refusal apart from a 422 caused by a malformed or rejected query.
REFUSAL_MARKER = "cannot be answered from this dataset"


# --- adapters -----------------------------------------------------------


def http_ask(base_url: str, timeout: float = 120.0) -> Ask:
    base = base_url.rstrip("/")

    def ask(question: str) -> Outcome:
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{base}/query", json={"question": question}, timeout=timeout
            )
        except httpx.HTTPError as exc:
            return Outcome("error", detail=str(exc), latency_s=time.perf_counter() - started)

        elapsed = time.perf_counter() - started
        detail = _detail(response)

        if response.status_code == 200:
            body = response.json()
            return Outcome(
                "answered",
                answer=body["answer"],
                sql=body["sql"],
                rows=body["rows"],
                row_count=body["row_count"],
                confidence=body["confidence"],
                repaired=body["repaired"],
                latency_s=elapsed,
            )
        if response.status_code == 422 and REFUSAL_MARKER in detail:
            return Outcome("refused", detail=detail, latency_s=elapsed)
        return Outcome("error", detail=f"HTTP {response.status_code}: {detail}", latency_s=elapsed)

    return ask


def direct_ask() -> tuple[Ask, str, str]:
    """Build the engine in-process. Requires a key in .env."""
    from app.config import settings
    from app.data.loader import TicketStore
    from app.llm.factory import build_provider
    from app.query.engine import QueryEngine, UnanswerableQuestion

    store = TicketStore.from_csv(settings)
    provider = build_provider(settings)
    engine = QueryEngine(store, provider, settings)

    def ask(question: str) -> Outcome:
        started = time.perf_counter()
        try:
            result = engine.ask(question)
        except UnanswerableQuestion as exc:
            return Outcome("refused", detail=str(exc), latency_s=time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 — an eval run reports, it doesn't crash
            return Outcome(
                "error",
                detail=f"{type(exc).__name__}: {exc}",
                latency_s=time.perf_counter() - started,
            )
        return Outcome(
            "answered",
            answer=result.answer,
            sql=result.sql,
            rows=result.rows,
            row_count=result.row_count,
            confidence=result.confidence,
            repaired=result.repaired,
            latency_s=time.perf_counter() - started,
        )

    return ask, provider.name, provider.model


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


def _probe(base_url: str) -> tuple[str, str]:
    """Read provider and model off /health so the report records what was tested."""
    try:
        body = httpx.get(f"{base_url.rstrip('/')}/health", timeout=60).json()
    except (httpx.HTTPError, ValueError):
        return "unknown", "unknown"
    return body.get("llm_provider", "unknown"), body.get("llm_model", "unknown")


# --- output -------------------------------------------------------------

TICK, CROSS, WAVE = "PASS", "FAIL", "FLAKY"


def _print_case(result: CaseResult) -> None:
    mark = TICK if result.passed else (WAVE if result.flaky else CROSS)
    gap = "  [known gap]" if result.case.known_gap else ""
    rate = "" if len(result.trials) == 1 else f" {result.pass_rate:.0%}"
    # flush: a run takes minutes, and Python buffers stdout when redirected to a
    # file or a CI log, which would hide progress until the very end.
    print(f"  {mark:<5}{rate:<5} {result.case.id:<26} {result.latency_s:5.1f}s  "
          f"{result.case.question[:60]}{gap}", flush=True)
    for problem in result.problems:
        print(f"           -> {problem}", flush=True)


def _print_summary(report: Report) -> None:
    print("\n" + "=" * 78)
    print(f"{report.provider} / {report.model}   "
          f"{len(report.results)} cases x {report.trials} trial(s)   "
          f"{report.duration_s:.0f}s")
    print("=" * 78)

    print("\nBy category")
    for name, (passed, total) in sorted(report.by_category().items()):
        bar = "#" * int(20 * passed / total) if total else ""
        print(f"  {name:<18} {passed:>2}/{total:<3} {_ratio_str(passed, total):>6}  {bar}")

    failures = [r for r in report.results if not r.passed]
    known = [r for r in failures if r.case.known_gap]
    real = [r for r in failures if not r.case.known_gap]

    print(f"\nAccuracy              {report.accuracy:.1%}  ({len(report.results) - len(failures)}"
          f"/{len(report.results)})")
    print(f"Excluding known gaps  {report.accuracy_excluding_known_gaps:.1%}  "
          f"({len(report.scored) - len(real)}/{len(report.scored)})")
    print(f"Latency               p50 {report.latency(0.50):.1f}s   p95 {report.latency(0.95):.1f}s")

    if report.flaky:
        print(f"\nFlaky ({len(report.flaky)}) — passed on some trials but not all:")
        for result in report.flaky:
            print(f"  {result.case.id}  {result.pass_rate:.0%}")

    if real:
        print(f"\nFailures ({len(real)}):")
        for result in real:
            print(f"  {result.case.id:<26} {result.case.question[:52]}")
            for problem in result.problems:
                print(f"    {problem}")

    if known:
        print(f"\nKnown gaps ({len(known)}) — tracked, not counted against the gate:")
        for result in known:
            print(f"  {result.case.id}: {result.case.why}")


def _ratio_str(passed: int, total: int) -> str:
    return f"{passed / total:.0%}" if total else "n/a"


# --- entry point --------------------------------------------------------


def select(cases: tuple[EvalCase, ...], category: str | None, case_id: str | None):
    chosen = cases
    if category:
        chosen = tuple(c for c in chosen if c.category == category)
    if case_id:
        chosen = tuple(c for c in chosen if c.id == case_id)
    if not chosen:
        raise SystemExit(f"No cases matched (category={category!r}, id={case_id!r}).")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ticket-system eval suite.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--direct", action="store_true", help="Bypass HTTP; needs a key in .env.")
    parser.add_argument("--trials", type=int, default=1, help="Runs per case; >1 measures consistency.")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds between calls (rate limits).")
    parser.add_argument("--category", help="Only run one category.")
    parser.add_argument("--case", dest="case_id", help="Only run one case id.")
    parser.add_argument("--min-accuracy", type=float, default=0.90,
                        help="Exit non-zero below this, ignoring known gaps.")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the JSON report.")
    args = parser.parse_args(argv)

    cases = select(CASES, args.category, args.case_id)

    if args.direct:
        try:
            ask, provider, model = direct_ask()
        except Exception as exc:  # noqa: BLE001 — surface setup failures plainly
            print(f"Could not build the engine: {exc}", file=sys.stderr)
            return 2
    else:
        ask = http_ask(args.api_url)
        provider, model = _probe(args.api_url)
        if provider == "unknown":
            print(f"Could not reach the API at {args.api_url}. "
                  f"Start it with ./run.sh api, or use --direct.", file=sys.stderr)
            return 2

    print(f"Running {len(cases)} case(s) against {provider}/{model} "
          f"x{args.trials} trial(s)\n")

    report = run(
        ask, cases, trials=args.trials, provider=provider, model=model,
        delay_s=args.delay, on_case=_print_case,
    )
    _print_summary(report)

    if not args.no_save:
        print(f"\nReport saved to {report.save()}")

    if report.accuracy_excluding_known_gaps < args.min_accuracy:
        print(f"\nFAILED: {report.accuracy_excluding_known_gaps:.1%} is below the "
              f"{args.min_accuracy:.0%} threshold.")
        return 1
    print(f"\nPASSED: {report.accuracy_excluding_known_gaps:.1%} "
          f"(threshold {args.min_accuracy:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
