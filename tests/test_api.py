"""API contract tests. The LLM is swapped out at the app-state level."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, state
from app.query.engine import QueryEngine
from tests.conftest import FakeLLM, sql_reply


@pytest.fixture
def client(store, settings):
    with TestClient(app) as test_client:
        # The real provider may or may not be configured in this environment;
        # pin a scripted one so these tests never touch the network.
        state.store = store
        state.engine = QueryEngine(store, FakeLLM([]), settings)
        state.settings = settings
        try:
            yield test_client
        finally:
            # /config/llm mutates process state; don't leak it into other tests.
            state.settings = settings
            state.llm_error = None


def scripted(store, settings, responses):
    state.engine = QueryEngine(store, FakeLLM(responses), settings)


def test_health_reports_components_and_dataset(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["dataset"]["rows"] == 500
    assert {"dataset", "database", "llm"} <= set(body["components"])


def test_query_returns_answer_sql_and_rows(client, store, settings):
    scripted(
        store,
        settings,
        [sql_reply("SELECT COUNT(*) AS n FROM tickets WHERE status = 'Open'"), "111 open."],
    )
    body = client.post("/query", json={"question": "How many are open?"}).json()
    assert body["rows"] == [{"n": 111}]
    assert body["sql"].startswith("SELECT")
    assert body["answer"] == "111 open."


def test_query_can_omit_rows(client, store, settings):
    scripted(store, settings, [sql_reply("SELECT COUNT(*) AS n FROM tickets"), "500."])
    body = client.post(
        "/query", json={"question": "How many tickets?", "include_rows": False}
    ).json()
    assert body["rows"] == []
    assert body["row_count"] == 1


def test_unanswerable_question_returns_422(client, store, settings):
    scripted(store, settings, ['{"error": "No contact details in this dataset."}'])
    response = client.post("/query", json={"question": "Give me the phone number"})
    assert response.status_code == 422
    assert "cannot be answered" in response.json()["detail"]


def test_empty_question_returns_422_from_validation(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_anomalies_need_no_llm(client):
    state.engine = None
    try:
        body = client.get("/anomalies").json()
    finally:
        state.engine = None
    assert body["total"] >= 1
    assert body["tickets_flagged"] >= 1
    assert body["summary"] is None


def test_query_without_llm_returns_503_and_points_at_anomalies(client):
    state.engine = None
    state.llm_error = "GEMINI_API_KEY is not set."
    response = client.post("/query", json={"question": "How many tickets?"})
    assert response.status_code == 503
    assert "/anomalies" in response.json()["detail"]


def test_anomalies_severity_filter(client):
    body = client.get("/anomalies", params={"severity": "critical"}).json()
    assert all(a["severity"] == "critical" for a in body["anomalies"])


def test_anomalies_rejects_unknown_severity(client):
    assert client.get("/anomalies", params={"severity": "urgent"}).status_code == 400


def test_anomalies_rejects_unknown_detector(client):
    assert client.get("/anomalies", params={"types": ["nope"]}).status_code == 400


def test_examples_endpoint_seeds_the_ui(client):
    assert len(client.get("/examples").json()["questions"]) >= 5


# --- POST /config/llm ---------------------------------------------------


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_config_requires_a_key_for_hosted_providers(client, provider):
    """Rejected before any network call is attempted."""
    response = client.post("/config/llm", json={"provider": provider})
    assert response.status_code == 400
    assert "requires an API key" in response.json()["detail"]


def test_config_rejects_unknown_provider(client):
    assert client.post("/config/llm", json={"provider": "openai"}).status_code == 422


def test_config_accepts_a_working_provider_and_swaps_the_engine(client, monkeypatch):
    monkeypatch.setattr("app.api.main.build_provider", lambda s: FakeLLM([]))

    body = client.post(
        "/config/llm", json={"provider": "gemini", "api_key": "k-123", "model": "some-model"}
    ).json()

    assert body["ok"] is True
    assert body["provider"] == "fake"
    # The swap is visible to the rest of the app.
    assert state.engine is not None
    assert state.settings.gemini_api_key == "k-123"
    assert state.settings.gemini_model == "some-model"


def test_config_failure_leaves_the_previous_provider_in_place(client, monkeypatch):
    working = state.engine

    def boom(_settings):
        from app.llm.base import LLMError

        raise LLMError("upstream rejected the request")

    monkeypatch.setattr("app.api.main.build_provider", boom)
    response = client.post("/config/llm", json={"provider": "gemini", "api_key": "k-123"})

    assert response.status_code == 400
    assert state.engine is working, "a failed reconfigure must not disable a working LLM"


def test_config_never_echoes_the_key_in_an_error(client, monkeypatch):
    """Provider SDKs sometimes embed the key in error text; it must be redacted."""
    secret = "AIzaSy-super-secret-key-value"

    def leaky(_settings):
        from app.llm.base import LLMError

        raise LLMError(f"401 from https://api.example/v1?key={secret}")

    monkeypatch.setattr("app.api.main.build_provider", leaky)
    response = client.post("/config/llm", json={"provider": "gemini", "api_key": secret})

    assert response.status_code == 400
    assert secret not in response.text
    assert "redacted" in response.json()["detail"]


def test_config_unblocks_query(client, monkeypatch):
    """The whole point: no key at startup, then a key, then questions work."""
    state.engine = None
    state.llm_error = "GEMINI_API_KEY is not set."
    assert client.post("/query", json={"question": "How many tickets?"}).status_code == 503

    # First scripted reply is consumed by the health probe inside /config/llm.
    monkeypatch.setattr(
        "app.api.main.build_provider",
        lambda s: FakeLLM(["ok", sql_reply("SELECT COUNT(*) AS n FROM tickets"), "500 tickets."]),
    )
    assert client.post("/config/llm", json={"provider": "gemini", "api_key": "k"}).status_code == 200

    body = client.post("/query", json={"question": "How many tickets?"}).json()
    assert body["rows"] == [{"n": 500}]
