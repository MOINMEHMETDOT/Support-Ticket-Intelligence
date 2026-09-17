"""Request and response schemas for the REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="A natural-language question about the ticket dataset.",
        examples=["Which agent has the lowest average customer rating?"],
    )
    include_rows: bool = Field(
        default=True,
        description="Return the raw result rows alongside the prose answer.",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str = Field(description="Plain-language answer derived only from the SQL result.")
    sql: str = Field(description="The query that produced the answer, shown for auditability.")
    explanation: str = Field(description="Why the model wrote the query that way.")
    confidence: str
    row_count: int
    truncated: bool
    repaired: bool = Field(description="True if the first query failed and was auto-corrected.")
    rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnomalyModel(BaseModel):
    type: str
    severity: Literal["critical", "high", "medium", "low"]
    title: str
    description: str
    count: int
    ticket_ids: list[str]
    ticket_ids_truncated: bool
    evidence: dict[str, Any]


class AnomalyResponse(BaseModel):
    as_of: str = Field(description="Reference timestamp used for all age calculations.")
    total: int
    tickets_flagged: int = Field(description="Distinct tickets appearing in at least one anomaly.")
    severity_counts: dict[str, int]
    anomalies: list[AnomalyModel]
    summary: str | None = Field(
        default=None, description="LLM-written narrative, present only when summarize=true."
    )


class ComponentHealth(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_provider: str
    llm_model: str
    components: dict[str, ComponentHealth]
    dataset: dict[str, Any]


class LLMConfigRequest(BaseModel):
    """Configure the LLM at runtime, so a key can be supplied from the UI.

    The key is held in process memory only — never written to disk, never
    logged, and never echoed back in a response.
    """

    provider: Literal["gemini", "groq", "ollama"]
    api_key: str | None = Field(
        default=None,
        description="Required for gemini and groq; ignored for ollama.",
        repr=False,  # keep it out of tracebacks and __repr__ output
    )
    model: str | None = Field(
        default=None, description="Model id. Falls back to the configured default."
    )


class LLMConfigResponse(BaseModel):
    ok: bool
    provider: str
    model: str
    detail: str = Field(description="Probe result, or why the key was rejected.")


class ErrorResponse(BaseModel):
    error: str
    detail: str
