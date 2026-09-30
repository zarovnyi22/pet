"""A second provider behind the first: same LLMClient interface, chosen by LLM_FALLBACK_PROVIDER.

The primary gets its full retries first (post_json). Only if they run out on 503, or the
primary answers 429, does the same call go to the fallback. After a fallback the primary is
skipped for PRIMARY_COOLDOWN_SECONDS: in a /reformulate run a second call would otherwise
spend the same ~30 s of retries again and miss the run's 60 s deadline.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from app.llm.base import LLMClient, LLMError, LLMResponse, Message, ToolSpec

logger = logging.getLogger("app.llm")

FALLBACK_ON = ("llm_unavailable", "llm_rate_limited")
PRIMARY_COOLDOWN_SECONDS = 60.0


class FallbackLLMClient(LLMClient):
    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self.provider = primary.provider
        self._primary_down_until = 0.0

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        return await self._call(lambda llm: llm.complete(messages, json_mode=json_mode))

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        return await self._call(lambda llm: llm.complete_with_tools(messages, tools))

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.fallback.aclose()

    async def _call[T](self, call: Callable[[LLMClient], Awaitable[T]]) -> T:
        if time.monotonic() < self._primary_down_until:
            self._log("primary cooling down")
            return await call(self.fallback)
        try:
            return await call(self.primary)
        except LLMError as exc:
            if exc.code not in FALLBACK_ON:
                raise
            self._primary_down_until = time.monotonic() + PRIMARY_COOLDOWN_SECONDS
            self._log(exc.code)
            return await call(self.fallback)

    def _log(self, reason: str) -> None:
        logger.warning(
            "llm fallback",
            extra={
                "provider": self.primary.provider,
                "fallback": self.fallback.provider,
                "reason": reason,
            },
        )
