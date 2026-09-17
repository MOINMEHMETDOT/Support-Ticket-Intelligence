"""Groq provider (free tier). Listed as an allowed backend in the brief."""

from __future__ import annotations

from app.llm.base import LLMError, LLMProvider


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and add it to .env."
            )
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError("The groq package is not installed. Run: pip install groq") from exc

        self.model = model
        self._client = Groq(api_key=api_key)

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001 — SDK raises a wide error family
            message = str(exc)
            transient = "429" in message or "rate limit" in message.lower() or any(
                code in message for code in ("500", "502", "503", "504")
            )
            raise LLMError(f"Groq call failed: {message}", retryable=transient) from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise LLMError("Groq returned an empty response.")
        return text
