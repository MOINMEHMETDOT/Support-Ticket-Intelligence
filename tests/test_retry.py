"""Backoff behaviour. Sleep is injected, so these run instantly."""

from __future__ import annotations

import pytest

from app.llm.base import LLMError, LLMProvider
from app.llm.retry import RetryingProvider


class Flaky(LLMProvider):
    """Fails `failures` times with the given error, then succeeds."""

    name = "flaky"
    model = "flaky-1"

    def __init__(self, failures: int, *, retryable: bool = True):
        self.failures = failures
        self.retryable = retryable
        self.calls = 0

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMError(f"boom {self.calls}", retryable=self.retryable)
        return "ok"


def wrap(inner, **kw):
    slept: list[float] = []
    provider = RetryingProvider(inner, sleep=slept.append, base_delay=1.0, **kw)
    return provider, slept


def test_succeeds_without_sleeping_when_the_first_call_works():
    provider, slept = wrap(Flaky(failures=0))
    assert provider.complete("s", "u") == "ok"
    assert slept == []


def test_retries_transient_failures_until_one_succeeds():
    inner = Flaky(failures=2)
    provider, slept = wrap(inner)
    assert provider.complete("s", "u") == "ok"
    assert inner.calls == 3
    assert len(slept) == 2


def test_backoff_grows_between_attempts():
    provider, slept = wrap(Flaky(failures=3), attempts=4)
    provider.complete("s", "u")
    assert slept[0] < slept[1] < slept[2], f"expected growing delays, got {slept}"


def test_permanent_errors_are_not_retried():
    """A bad key must fail fast — retrying only makes the error slower."""
    inner = Flaky(failures=1, retryable=False)
    provider, slept = wrap(inner)
    with pytest.raises(LLMError, match="boom 1"):
        provider.complete("s", "u")
    assert inner.calls == 1
    assert slept == []


def test_gives_up_after_the_attempt_budget_and_raises_the_last_error():
    inner = Flaky(failures=99)
    provider, slept = wrap(inner, attempts=3)
    with pytest.raises(LLMError, match="boom 3"):
        provider.complete("s", "u")
    assert inner.calls == 3
    assert len(slept) == 2, "no sleep after the final attempt"


def test_delay_is_capped():
    provider, slept = wrap(Flaky(failures=9), attempts=8, max_delay=4.0)
    with pytest.raises(LLMError):
        provider.complete("s", "u")
    assert max(slept) <= 4.0 * 1.25 + 1e-9, "jitter may exceed the cap only slightly"


def test_identity_is_transparent_to_callers():
    inner = Flaky(failures=0)
    provider, _ = wrap(inner)
    assert (provider.name, provider.model) == (inner.name, inner.model)
    assert provider.inner is inner


def test_rate_limit_errors_from_gemini_are_marked_retryable():
    from app.llm.gemini import GeminiProvider

    probe = GeminiProvider.__new__(GeminiProvider)  # no network, no key
    probe.model = "m"
    assert probe._as_error(Exception("429 RESOURCE_EXHAUSTED")).retryable is True
    assert probe._as_error(Exception("API key not valid")).retryable is False
