"""Shared fixtures. Nothing here touches the network — a scripted fake stands in
for the LLM so the suite runs offline and deterministically."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import load_settings
from app.data.loader import TicketStore
from app.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    """Returns queued responses in order; records every prompt it was given."""

    name = "fake"
    model = "fake-1"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        if not self.responses:
            return "No more scripted responses."
        return self.responses.pop(0)


def sql_reply(sql: str, confidence: str = "high") -> str:
    return json.dumps({"sql": sql, "explanation": "test", "confidence": confidence})


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def store(settings, tmp_path_factory) -> TicketStore:
    db = tmp_path_factory.mktemp("db") / "test.db"
    return TicketStore.from_csv(settings, db_path=Path(db))
