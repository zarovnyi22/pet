"""Gemini via the REST API (generateContent), no SDK: one httpx call per turn."""

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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self._api_key = api_key
        self._model = model
        self._http = httpx.AsyncClient(timeout=timeout)

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        body = self._body(messages)
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        return (await self._generate(body)).text

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        body = self._body(messages)
        body["tools"] = [
            {
                "functionDeclarations": [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in tools
                ]
            }
        ]
        return await self._generate(body)

    async def aclose(self) -> None:
        await self._http.aclose()

    def _body(self, messages: list[Message]) -> dict[str, Any]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        body: dict[str, Any] = {
            "contents": _to_contents(messages),
            "generationConfig": {"temperature": 0.2},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    async def _generate(self, body: dict[str, Any]) -> LLMResponse:
        require_key(self.provider, "GEMINI_API_KEY", self._api_key)
        data = await post_json(
            self._http,
            self.provider,
            f"{BASE_URL}/{self._model}:generateContent",
            {"x-goog-api-key": self._api_key},
            body,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise LLMError(f"gemini returned no answer: {reason}")
        parts = candidates[0].get("content", {}).get("parts", [])

        texts, calls = [], []
        for i, part in enumerate(parts):
            if "functionCall" in part:
                fc = part["functionCall"]
                extra = (
                    {"thoughtSignature": part["thoughtSignature"]}
                    if "thoughtSignature" in part
                    else {}
                )
                calls.append(
                    ToolCall(
                        id=fc.get("id") or f"call_{i}",
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                        extra=extra,
                    )
                )
            elif "text" in part and not part.get("thought"):
                texts.append(part["text"])
        return LLMResponse(text="".join(texts), tool_calls=calls)


def _to_contents(messages: list[Message]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            role, parts = "user", [{"text": m.content}]
        elif m.role == "assistant":
            role, parts = "model", [{"text": m.content}] if m.content else []
            for call in m.tool_calls:
                part: dict[str, Any] = {"functionCall": {"name": call.name, "args": call.arguments}}
                part.update(call.extra)
                parts.append(part)
        else:  # tool result
            role = "user"
            parts = [{"functionResponse": {"name": m.name, "response": _as_object(m.content)}}]
        # Gemini wants alternating turns: parallel tool results go into one user turn.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})
    return contents


def _as_object(content: str) -> dict[str, Any]:
    # functionResponse.response must be a JSON object.
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        value = content
    return value if isinstance(value, dict) else {"result": value}
