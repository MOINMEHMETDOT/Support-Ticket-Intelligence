"""Google Gemini provider (free tier via Google AI Studio)."""

from __future__ import annotations

from app.llm.base import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env, "
                "or set LLM_PROVIDER=ollama to run without a key."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "The google-genai package is not installed. Run: pip install google-genai"
            ) from exc

        self.model = model
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        from google.genai import types

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
        except Exception as exc:  # noqa: BLE001 — SDK raises a wide error family
            raise LLMError(self._explain(exc)) from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError(
                "Gemini returned an empty response (often a safety block or a "
                "hit token limit). Try rephrasing the question."
            )
        return text

    def _explain(self, exc: Exception) -> str:
        """Turn a raw SDK error into one actionable sentence.

        The SDK embeds a full JSON error document in str(exc); surfacing that
        verbatim fills the UI with noise that tells the user nothing about what
        to do next.
        """
        message = str(exc)
        lowered = message.lower()

        if "api_key_invalid" in lowered or "api key not valid" in lowered:
            return (
                "Gemini rejected the API key. Check it was copied in full from "
                "https://aistudio.google.com/apikey — keys start with 'AIza'."
            )
        if "not found" in lowered or "404" in message:
            return (
                f"Model {self.model!r} was not found for this API key. "
                "Run `python scripts/list_models.py` to see the exact ids your "
                "key can use, then set GEMINI_MODEL."
            )
        if "permission" in lowered or "401" in message or "403" in message:
            return "Gemini refused the request — the key exists but lacks access to this model."
        if "429" in message or "quota" in lowered or "resource_exhausted" in lowered:
            return (
                "Gemini free-tier rate limit hit. Wait a moment and retry, or "
                "switch the provider to groq/ollama."
            )
        return f"Gemini call failed: {_first_line(message)}"


def _first_line(message: str, limit: int = 160) -> str:
    """Collapse a multi-line SDK error to a single readable clause."""
    line = " ".join(message.split())
    return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
