"""Selects the configured provider. The only place that knows all three exist."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMError, LLMProvider

SUPPORTED = ("gemini", "groq", "ollama")


def build_provider(settings: Settings) -> LLMProvider:
    provider = settings.provider

    if provider == "gemini":
        from app.llm.gemini import GeminiProvider

        return GeminiProvider(settings.gemini_api_key, settings.gemini_model)

    if provider == "groq":
        from app.llm.groq import GroqProvider

        return GroqProvider(settings.groq_api_key, settings.groq_model)

    if provider == "ollama":
        from app.llm.ollama import OllamaProvider

        return OllamaProvider(settings.ollama_host, settings.ollama_model)

    raise LLMError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported: {', '.join(SUPPORTED)}."
    )
