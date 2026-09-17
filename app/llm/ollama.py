"""Ollama provider — fully local, no API key, no network cost.

This is the fallback that lets an evaluator run the system without any
credentials of their own.
"""

from __future__ import annotations

import httpx

from app.llm.base import LLMError, LLMProvider


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 120.0):
        self.model = model
        self._host = host.rstrip("/")
        self._timeout = timeout

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = httpx.post(
                f"{self._host}/api/chat", json=payload, timeout=self._timeout
            )
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._host}. Start it with `ollama serve` "
                f"and pull the model with `ollama pull {self.model}`."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:200]
            if exc.response.status_code == 404:
                raise LLMError(
                    f"Ollama has no model named {self.model!r}. "
                    f"Run: ollama pull {self.model}"
                ) from exc
            raise LLMError(f"Ollama returned {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        text = (response.json().get("message", {}).get("content") or "").strip()
        if not text:
            raise LLMError("Ollama returned an empty response.")
        return text
