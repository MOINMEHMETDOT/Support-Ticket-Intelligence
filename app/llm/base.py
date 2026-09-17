"""Provider-agnostic LLM interface.

The rest of the system talks to this and nothing else, so swapping Gemini for
Groq or a local Ollama model is a config change rather than a code change. That
matters here for a practical reason: the API key in .env is the author's, and an
evaluator without one can still run everything on Ollama at zero cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised when a provider is misconfigured or the upstream call fails.

    `retryable` separates transient conditions — rate limits, upstream 5xx —
    from permanent ones like a bad key or a missing model. Retrying the latter
    just turns a fast, clear failure into a slow one.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class LLMProvider(ABC):
    """A minimal single-turn text completion interface.

    Deliberately not a chat interface: every LLM call in this system is a
    stateless transformation (question -> SQL, result -> sentence). Keeping it
    stateless makes each call independently retryable and testable.
    """

    name: str
    model: str

    @abstractmethod
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        """Return the model's text response, or raise LLMError."""

    def health(self) -> tuple[bool, str]:
        """Cheap liveness probe used by GET /health."""
        try:
            reply = self.complete("Reply with the single word: ok", "ping")
        except LLMError as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001 — providers raise varied SDK errors
            return False, f"{type(exc).__name__}: {exc}"
        return True, reply.strip()[:80]
