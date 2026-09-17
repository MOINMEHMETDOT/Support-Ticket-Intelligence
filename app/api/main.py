"""FastAPI application exposing NL query, anomaly detection and health checks.

Startup is deliberately tolerant: the dataset must load or the app is useless,
but a missing or misconfigured LLM key only disables /query. Anomaly detection
and /health keep working and report the problem, which is what you want when an
evaluator runs this without a key of their own.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from app.anomaly.detectors import SEVERITY_RANK, detect, summarise_for_llm
from app.api.models import (
    AnomalyResponse,
    ComponentHealth,
    HealthResponse,
    LLMConfigRequest,
    LLMConfigResponse,
    QueryRequest,
    QueryResponse,
)
from app.config import Settings, settings
from app.data.loader import QueryExecutionError, TicketStore
from app.llm.base import LLMError
from app.llm.factory import build_provider
from app.query.engine import QueryEngine, UnanswerableQuestion
from app.query.guards import SQLValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets this month?",
    "Show me all Critical tickets not resolved within 12 hours.",
    "What is the average customer rating for Technical category tickets?",
    "Which category has the worst average resolution time?",
    "How many Critical tickets are still unresolved?",
]


class AppState:
    """Process-wide singletons. Built at startup; the LLM half is replaceable at
    runtime via POST /config/llm so a key can be supplied from the UI."""

    store: TicketStore | None = None
    engine: QueryEngine | None = None
    llm_error: str | None = None
    settings: Settings = settings


state = AppState()


def _redact(message: str, secret: str | None) -> str:
    """Strip a key out of an error string before it leaves the process.

    Provider SDKs sometimes embed the key in a request URL inside their error
    messages; this makes sure one never reaches a response body or a log line.
    """
    if secret and len(secret) > 4:
        return message.replace(secret, "***redacted***")
    return message


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.settings = settings
    state.store = TicketStore.from_csv(settings)
    logger.info(
        "Loaded %d tickets from %s (reference date %s)",
        len(state.store.frame),
        settings.data_csv.name,
        state.store.as_of,
    )
    try:
        provider = build_provider(settings)
        state.engine = QueryEngine(state.store, provider, settings)
        logger.info("LLM provider ready: %s / %s", provider.name, provider.model)
    except LLMError as exc:
        state.llm_error = str(exc)
        logger.warning("LLM unavailable, /query is disabled: %s", exc)
    yield


app = FastAPI(
    title="Support Ticket Intelligence API",
    description=(
        "Natural-language querying and anomaly detection over a customer support "
        "ticket dataset. Questions are translated to SQL by an LLM and executed by "
        "SQLite, so every figure in an answer is computed, not generated."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_store() -> TicketStore:
    if state.store is None:  # pragma: no cover - only if lifespan failed
        raise HTTPException(503, "Dataset is not loaded.")
    return state.store


def get_engine() -> QueryEngine:
    if state.engine is None:
        raise HTTPException(
            503,
            f"Natural-language querying is unavailable: {state.llm_error} "
            "Anomaly detection at GET /anomalies does not need an LLM and still works.",
        )
    return state.engine


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(store: TicketStore = Depends(get_store)) -> HealthResponse:
    """Liveness plus a real round-trip to the configured LLM."""
    components: dict[str, ComponentHealth] = {
        "dataset": ComponentHealth(
            ok=True, detail=f"{len(store.frame)} rows from {settings.data_csv.name}"
        ),
        "database": ComponentHealth(ok=store.db_path.exists(), detail=str(store.db_path.name)),
    }

    if state.engine is None:
        components["llm"] = ComponentHealth(ok=False, detail=state.llm_error or "not configured")
    else:
        ok, detail = state.engine.llm.health()
        components["llm"] = ComponentHealth(ok=ok, detail=detail)

    return HealthResponse(
        status="ok" if all(c.ok for c in components.values()) else "degraded",
        llm_provider=state.settings.provider,
        llm_model=state.settings.model_name,
        components=components,
        dataset=store.profile(),
    )


@app.post("/config/llm", response_model=LLMConfigResponse, tags=["ops"])
def configure_llm(
    request: LLMConfigRequest, store: TicketStore = Depends(get_store)
) -> LLMConfigResponse:
    """Set the LLM provider and key at runtime, so the UI can supply one.

    The key is validated with a real round-trip before being accepted, held in
    process memory only, and never persisted or logged. On failure the previous
    working provider is left untouched.
    """
    key = (request.api_key or "").strip()
    if request.provider in ("gemini", "groq") and not key:
        raise HTTPException(400, f"{request.provider} requires an API key.")

    overrides: dict[str, Any] = {"provider": request.provider}
    if request.provider == "gemini":
        overrides["gemini_api_key"] = key
        if request.model:
            overrides["gemini_model"] = request.model
    elif request.provider == "groq":
        overrides["groq_api_key"] = key
        if request.model:
            overrides["groq_model"] = request.model
    elif request.model:
        overrides["ollama_model"] = request.model

    candidate = replace(state.settings, **overrides)

    try:
        provider = build_provider(candidate)
    except LLMError as exc:
        raise HTTPException(400, _redact(str(exc), key))

    ok, detail = provider.health()
    detail = _redact(detail, key)
    if not ok:
        raise HTTPException(400, f"Could not reach {request.provider}: {detail}")

    state.settings = candidate
    state.engine = QueryEngine(store, provider, candidate)
    state.llm_error = None
    # Provider and model only — never the key.
    logger.info("LLM reconfigured at runtime: %s / %s", provider.name, provider.model)

    return LLMConfigResponse(
        ok=True, provider=provider.name, model=provider.model, detail=detail
    )


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest, engine: QueryEngine = Depends(get_engine)) -> QueryResponse:
    """Answer a natural-language question about the tickets."""
    try:
        result = engine.ask(request.question)
    except UnanswerableQuestion as exc:
        raise HTTPException(422, f"That question cannot be answered from this dataset. {exc}")
    except SQLValidationError as exc:
        raise HTTPException(422, f"The generated query was rejected: {exc}")
    except QueryExecutionError as exc:
        raise HTTPException(500, f"The query failed to execute: {exc}")
    except LLMError as exc:
        raise HTTPException(503, f"The language model is unavailable: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sql=result.sql,
        explanation=result.explanation,
        confidence=result.confidence,
        row_count=result.row_count,
        truncated=result.truncated,
        repaired=result.repaired,
        rows=result.rows if request.include_rows else [],
        warnings=result.warnings,
    )


@app.get("/anomalies", response_model=AnomalyResponse, tags=["anomalies"])
def anomalies(
    store: TicketStore = Depends(get_store),
    severity: str | None = Query(
        default=None, description="Filter to one severity: critical, high, medium or low."
    ),
    types: list[str] | None = Query(
        default=None, description="Restrict to named detectors, e.g. types=aging_high_priority."
    ),
    summarize: bool = Query(
        default=False, description="Add an LLM-written narrative over the findings."
    ),
) -> AnomalyResponse:
    """Flag anomalous tickets using deterministic rules and statistics."""
    try:
        found = detect(store, types)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if severity:
        severity = severity.lower()
        if severity not in SEVERITY_RANK:
            raise HTTPException(
                400, f"Unknown severity {severity!r}. Use: {', '.join(SEVERITY_RANK)}."
            )
        found = [a for a in found if a.severity == severity]

    summary: str | None = None
    if summarize and found:
        if state.engine is None:
            summary = f"(Narrative unavailable: {state.llm_error})"
        else:
            try:
                summary = state.engine.llm.complete(
                    "You are a support operations analyst. Given a list of detected "
                    "anomalies, write 2-4 sentences for a team lead: what matters most, "
                    "what it likely means, what to do first. Use only the facts given. "
                    "No markdown, no bullet points.",
                    summarise_for_llm(found),
                ).strip()
            except LLMError as exc:
                summary = f"(Narrative unavailable: {exc})"

    flagged = {tid for a in found for tid in a.ticket_ids}
    counts: dict[str, int] = {}
    for a in found:
        counts[a.severity] = counts.get(a.severity, 0) + 1

    return AnomalyResponse(
        as_of=store.as_of.strftime("%Y-%m-%d %H:%M"),
        total=len(found),
        tickets_flagged=len(flagged),
        severity_counts=counts,
        anomalies=[a.to_dict() for a in found],
        summary=summary,
    )


@app.get("/examples", tags=["query"])
def examples() -> dict[str, Any]:
    """Sample questions, used to seed the UI."""
    return {"questions": EXAMPLE_QUESTIONS}


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "Support Ticket Intelligence API",
            "docs": "/docs",
            "endpoints": ["/health", "/query", "/anomalies", "/examples"],
        }
    )
