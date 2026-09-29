"""Provider-neutral LLM interface.

Callers build `Message`/`ToolSpec` objects; each provider translates them to its own wire
format. Switching providers is LLM_PROVIDER=gemini|groq, nothing else.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.errors import AppError


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


async def post_json(
    http: httpx.AsyncClient, provider: str, url: str, headers: dict, body: dict
) -> dict:
    """POST to a provider API, mapping transport/HTTP failures to LLMError."""
    try:
        resp = await http.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        raise LLMError(
            f"{provider} request timed out", code="llm_timeout", status_code=504
        ) from None
    except httpx.HTTPError as exc:
        raise LLMError(f"{provider} request failed: {type(exc).__name__}") from None
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
    return resp.json()


def require_key(provider: str, env_var: str, key: str) -> None:
    if not key:
        raise LLMError(
            f"{env_var} is not set (LLM_PROVIDER={provider})",
            code="llm_not_configured",
            status_code=503,
        )


def get_llm_client(settings) -> LLMClient:
    # Imported here so the fake and tests never pull in provider modules.
    if settings.llm_provider == "gemini":
        from app.llm.gemini import GeminiClient

        return GeminiClient(
            settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_seconds
        )
    if settings.llm_provider == "groq":
        from app.llm.groq import GroqClient

        return GroqClient(settings.groq_api_key, settings.groq_model, settings.llm_timeout_seconds)
    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider}")
