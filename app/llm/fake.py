"""Scripted LLM for tests: returns queued responses, never touches the network."""

from app.llm.base import LLMClient, LLMResponse, Message, ToolSpec


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, responses: list[str | LLMResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[list[Message]] = []  # every prompt received, for assertions

    async def complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        response = self._next(messages)
        return response.text if isinstance(response, LLMResponse) else response

    async def complete_with_tools(
        self, messages: list[Message], tools: list[ToolSpec]
    ) -> LLMResponse:
        response = self._next(messages)
        return response if isinstance(response, LLMResponse) else LLMResponse(text=response)

    def _next(self, messages: list[Message]) -> str | LLMResponse:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLM: no scripted responses left")
        return self.responses.pop(0)
