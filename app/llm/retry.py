"""Exponential backoff for transient provider failures.

Added because the eval suite could not complete a run: 40 questions at two LLM
calls each saturates a free-tier per-minute quota in about two minutes, and
every subsequent case failed with a 429 that had nothing to do with answer
quality. Rate limits are an expected operating condition on a free tier, not an
error — the system should absorb them and be slower, not wrong.

Wrapping at the factory means every provider inherits this; none of them
implement it themselves.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

from app.llm.base import LLMError, LLMProvider

logger = logging.getLogger(__name__)


class RetryingProvider(LLMProvider):
    def __init__(
        self,
        inner: LLMProvider,
        *,
        attempts: int = 4,
        base_delay: float = 2.0,
        max_delay: float = 32.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._inner = inner
        self._attempts = max(1, attempts)
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self.name = inner.name
        self.model = inner.model

    @property
    def inner(self) -> LLMProvider:
        return self._inner

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        for attempt in range(self._attempts):
            try:
                return self._inner.complete(system, user, temperature=temperature)
            except LLMError as exc:
                last_attempt = attempt == self._attempts - 1
                if not exc.retryable or last_attempt:
                    raise
                delay = min(self._base_delay * (2**attempt), self._max_delay)
                # Jitter so concurrent callers don't retry in lockstep.
                delay += random.uniform(0, delay * 0.25)
                logger.warning(
                    "Transient LLM failure (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    self._attempts,
                    delay,
                    exc,
                )
                self._sleep(delay)

        raise LLMError("Retry loop exited without a result.")  # pragma: no cover
