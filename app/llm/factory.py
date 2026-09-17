"""Selects the configured provider. The only place that knows all three exist."""

from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMError, LLMProvider
from app.llm.retry import RetryingProvider

SUPPORTED = ("gemini", "groq", "ollama")


def build_provider(settings: Settings, *, retries: bool = True) -> LLMProvider:
    """Construct the configured provider, wrapped with backoff by default.

    Retries are applied here rather than inside each provider so all three get
    identical behaviour. Pass retries=False in tests that assert on raw
    provider errors.
    """
    provider = _build_bare(settings)
    return RetryingProvider(provider) if retries else provider


def _build_bare(settings: Settings) -> LLMProvider:
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
