"""Provider-neutral LLM interface.

Callers build `Message`/`ToolSpec` objects; each provider translates them to its own wire
format. Switching providers is LLM_PROVIDER=gemini|groq, nothing else.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.errors import AppError

logger = logging.getLogger("app.llm")


class LLMError(AppError):
    def __init__(self, message: str, *, code: str = "llm_error", status_code: int = 502) -> None:
        super().__init__(status_code, code, message)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque provider data that must be echoed back on the next turn
    # (e.g. Gemini's thoughtSignature on function-call parts).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # role="assistant"
    tool_call_id: str | None = None  # role="tool"
    name: str | None = None  # role="tool": which tool produced this result


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(ABC):
    provider: str

    @abstractmethod
    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        """Plain completion. json_mode asks the provider to emit a single JSON object."""

    @abstractmethod
    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        """One model turn: either tool calls to execute, or a final text answer."""

    async def aclose(self) -> None:  # noqa: B027 — optional hook, no-op by default
        pass


# Token usage in one shape for both providers; output includes reasoning ("thinking") tokens,
# since those count against a tokens-per-minute limit too.
TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")
UsageParser = Callable[[dict[str, Any]], dict[str, int]]

# HTTP attempts of the LLM call in progress, for callers that record usage (the pipeline's
# trace). A ContextVar, not a client attribute: one client serves every concurrent request.
_attempts: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_attempts", default=None)


@contextmanager
def track_attempts() -> Iterator[list[dict[str, Any]]]:
    """Collect every HTTP attempt (retries included) made inside the block."""
    attempts: list[dict[str, Any]] = []
    token = _attempts.set(attempts)
    try:
        yield attempts
    finally:
        _attempts.reset(token)


def summarize_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """One LLM call's usage: who answered, how many HTTP attempts, tokens summed over them."""
    providers = list(dict.fromkeys(a["provider"] for a in attempts))
    return {
        "provider": ",".join(providers) or None,
        "http_attempts": len(attempts),
        **{f: sum(a.get(f, 0) for a in attempts) for f in TOKEN_FIELDS},
    }


# Transient provider failures (overload 503, a hung request, an unreachable host) are retried
# with growing pauses plus jitter: live Gemini answered "model is currently experiencing high
# demand" several times a day, usually for seconds, sometimes longer. All attempts together stay
# within RETRY_BUDGET_SECONDS; inside /reformulate the run's own deadline cuts them shorter
# (the pipeline awaits every call through bounded()). 429 is not retried: a per-minute quota
# will not recover in seconds; neither will an invalid key or another 4xx.
RETRYABLE = ("llm_unavailable", "llm_timeout")
RETRY_DELAYS_SECONDS = (2.0, 4.0, 8.0, 16.0)
RETRY_JITTER_SECONDS = 1.0
RETRY_BUDGET_SECONDS = 60.0
_sleep = asyncio.sleep  # replaced in tests


@dataclass
class Endpoint:
    """Where and how to call one provider: key_env names the API key variable (as in
    require_key) for the invalid-key message; usage reads token counts from a response."""

    provider: str
    key_env: str
    url: str
    headers: dict[str, str]
    usage: UsageParser


async def post_json(http: httpx.AsyncClient, endpoint: Endpoint, body: dict) -> dict:
    """POST to a provider API, mapping transport/HTTP failures to LLMError; transient ones are
    retried (see RETRY_DELAYS_SECONDS) and the last error is raised when the retries run out."""
    started = time.monotonic()
    for delay in (*RETRY_DELAYS_SECONDS, None):
        try:
            return await _post_once(http, endpoint, body)
        except LLMError as exc:
            if exc.code not in RETRYABLE or delay is None:
                raise
            pause = delay + random.uniform(0, RETRY_JITTER_SECONDS)
            if time.monotonic() - started + pause > RETRY_BUDGET_SECONDS:
                raise
            logger.warning(
                "llm retry",
                extra={"provider": endpoint.provider, "reason": exc.code, "pause_s": pause},
            )
            await _sleep(pause)
    raise AssertionError("unreachable")


async def _post_once(http: httpx.AsyncClient, endpoint: Endpoint, body: dict) -> dict:
    provider = endpoint.provider
    started = time.monotonic()
    try:
        resp = await http.post(endpoint.url, headers=endpoint.headers, json=body)
    except httpx.HTTPError as exc:
        _attempt({"provider": provider, "status": None, "error": type(exc).__name__})
        if isinstance(exc, httpx.TimeoutException):
            raise LLMError(
                f"{provider} request timed out", code="llm_timeout", status_code=504
            ) from None
        if isinstance(exc, httpx.ConnectError):  # DNS or refused connection: may recover
            raise LLMError(
                f"{provider} is unreachable: {type(exc).__name__}",
                code="llm_unavailable",
                status_code=503,
            ) from None
        raise LLMError(f"{provider} request failed: {type(exc).__name__}") from None
    data = resp.json() if resp.status_code < 400 else None
    attempt = {
        "provider": provider,
        "status": resp.status_code,
        "duration_ms": int((time.monotonic() - started) * 1000),
        **(endpoint.usage(data) if data is not None else {}),
    }
    logger.info("llm request", extra=attempt)
    _attempt(attempt)
    if resp.status_code == 401 or (resp.status_code == 400 and "API_KEY_INVALID" in resp.text):
        # Groq: 401 invalid_api_key; Gemini: 400 with reason API_KEY_INVALID. A configuration
        # problem, like a missing key: 503 with the variable to fix, not a provider failure.
        # The provider's body is not echoed: it adds nothing and may quote the key.
        raise LLMError(
            f"{endpoint.key_env} is invalid (LLM_PROVIDER={provider})",
            code="llm_invalid_key",
            status_code=503,
        )
    if resp.status_code == 429:
        raise LLMError(
            f"{provider} rate limit or quota exceeded", code="llm_rate_limited", status_code=503
        )
    if resp.status_code == 503:
        # Provider overload ("model is currently experiencing high demand"): transient.
        raise LLMError(
            f"{provider} is temporarily unavailable: {resp.text[:300]}",
            code="llm_unavailable",
            status_code=503,
        )
    if resp.status_code >= 400:
        # Body only, never request headers: those carry the API key.
        raise LLMError(f"{provider} returned HTTP {resp.status_code}: {resp.text[:300]}")
    return data


def _attempt(entry: dict[str, Any]) -> None:
    attempts = _attempts.get()
    if attempts is not None:
        attempts.append(entry)


def require_key(provider: str, env_var: str, key: str) -> None:
    if not key:
        raise LLMError(
            f"{env_var} is not set (LLM_PROVIDER={provider})",
            code="llm_not_configured",
            status_code=503,
        )


def get_llm_client(settings) -> LLMClient:
    """LLM_PROVIDER's client, wrapped with LLM_FALLBACK_PROVIDER's when that is set."""
    primary = _provider_client(settings.llm_provider, settings)
    if not settings.llm_fallback_provider:
        return primary
    from app.llm.fallback import FallbackLLMClient

    return FallbackLLMClient(primary, _provider_client(settings.llm_fallback_provider, settings))


def _provider_client(name: str, settings) -> LLMClient:
    # Imported here so the fake and tests never pull in provider modules.
    if name == "gemini":
        from app.llm.gemini import GeminiClient

        return GeminiClient(
            settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds
        )
    if name == "groq":
        from app.llm.groq import GroqClient

        return GroqClient(
            settings.groq_api_key,
            settings.groq_model,
            settings.llm_timeout_seconds,
            settings.groq_reasoning_effort,
        )
    raise ValueError(f"unknown LLM provider: {name}")
