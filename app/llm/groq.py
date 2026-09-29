"""Groq via its OpenAI-compatible chat completions REST API, no SDK."""

import json
from typing import Any

import httpx

from app.llm.base import (
    LLMClient,
    LLMError,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    post_json,
    require_key,
)

URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient(LLMClient):
    provider = "groq"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self._api_key = api_key
        self._model = model
        self._http = httpx.AsyncClient(timeout=timeout)

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        body = self._body(messages)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return (await self._chat(body)).text

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        body = self._body(messages)
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        return await self._chat(body)

    async def aclose(self) -> None:
        await self._http.aclose()

    def _body(self, messages: list[Message]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [_to_openai(m) for m in messages],
            "temperature": 0.2,
        }

    async def _chat(self, body: dict[str, Any]) -> LLMResponse:
        require_key(self.provider, "GROQ_API_KEY", self._api_key)
        data = await post_json(
            self._http, self.provider, URL, {"Authorization": f"Bearer {self._api_key}"}, body
        )
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise LLMError("groq returned no choices") from None

        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                raise LLMError(
                    f"groq returned invalid tool arguments for {tc['function']['name']}"
                ) from None
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls)


def _to_openai(m: Message) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in m.tool_calls
        ]
    return out
