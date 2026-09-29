"""Tool-calling loop, no framework: LLM turn -> run the requested tools -> repeat, until the
model returns a final answer that validates, or the iteration/time budget runs out."""

import asyncio
import json
import time
from collections.abc import Awaitable
from typing import Any, Protocol

from pydantic import ValidationError

from app.agent.common import (
    AgentError,
    OutOfTime,
    bounded,
    cap_confidence,
    format_validation,
    log_step,
    shorten,
)
from app.agent.prompts import FORCE_FINAL, RETRY_INVALID, system_prompt
from app.agent.tools import TOOL_SPECS
from app.llm.base import LLMClient, LLMError, Message, ToolCall
from app.schemas import (
    NutritionPer100g,
    ReformulateIn,
    ReformulateOut,
    ReformulationAnswer,
    TraceStep,
)

# Rounding slack when matching the answer's nutrition against calc_nutrition results.
NUTRITION_TOLERANCE = 0.05


class ToolRunner(Protocol):
    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class AgentLoop:
    def __init__(
        self, llm: LLMClient, tools: ToolRunner, *, max_iterations: int, timeout_seconds: float
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.trace: list[TraceStep] = []
        self._iteration = 0
        self._deadline = 0.0
        self._calc_results: list[NutritionPer100g] = []
        self._request: ReformulateIn | None = None

    async def run(self, request: ReformulateIn) -> ReformulateOut:
        self._deadline = time.monotonic() + self.timeout_seconds
        self._request = request
        try:
            answer = await self._run(request)
        except OutOfTime:
            message = f"no final answer within {self.timeout_seconds:g} s"
            self._record("error", message=message)
            raise AgentError(504, "agent_timeout", message, self.trace) from None
        except LLMError as exc:
            self._record("error", message=f"{exc.code}: {exc.message}")
            raise AgentError(exc.status_code, exc.code, exc.message, self.trace) from None
        return ReformulateOut(**answer.model_dump(), trace=self.trace)

    async def _run(self, request: ReformulateIn) -> ReformulationAnswer:
        messages = [
            Message(role="system", content=system_prompt()),
            Message(role="user", content=f"Reformulate this recipe:\n{request.model_dump_json()}"),
        ]
        results: dict[str, dict[str, Any]] = {}  # call key -> result, reused for repeats
        last_key: str | None = None
        force_final = retried = False

        for iteration in range(1, self.max_iterations + 1):
            self._iteration = iteration
            if force_final:
                # No tools offered: the model can only answer.
                text = await self._bounded(self.llm.complete(messages, json_mode=True))
                calls: list[ToolCall] = []
            else:
                response = await self._bounded(self.llm.complete_with_tools(messages, TOOL_SPECS))
                text, calls = response.text, response.tool_calls

            if calls:
                messages.append(Message(role="assistant", content=text, tool_calls=calls))
                repeated, last_key = await self._run_tools(calls, results, messages, last_key)
                if repeated:
                    self._record("loop_guard", message="same tool call twice in a row")
                    messages.append(Message(role="user", content=FORCE_FINAL))
                    force_final = True
                continue

            try:
                return self._parse_final(text)
            except ValueError as exc:
                self._record("validation_error", message=str(exc))
                if retried:
                    raise AgentError(
                        502, "agent_invalid_output", f"invalid final answer: {exc}", self.trace
                    ) from None
                retried = True
                messages.append(Message(role="assistant", content=text))
                messages.append(Message(role="user", content=RETRY_INVALID.format(error=exc)))

        message = f"no final answer after {self.max_iterations} iterations"
        self._record("error", message=message)
        raise AgentError(504, "agent_timeout", message, self.trace)

    async def _run_tools(
        self,
        calls: list[ToolCall],
        results: dict[str, dict[str, Any]],
        messages: list[Message],
        last_key: str | None,
    ) -> tuple[bool, str | None]:
        """Run all calls of one model turn concurrently; returns (repeat detected, last key)."""
        keys = [_call_key(c) for c in calls]
        pending = {k: c for k, c in zip(keys, calls, strict=True) if k not in results}
        timed = await self._bounded(asyncio.gather(*(self._timed(c) for c in pending.values())))
        fresh: dict[str, int] = {}  # key -> duration_ms, for calls actually run this turn
        for key, (result, ms) in zip(pending, timed, strict=True):
            results[key], fresh[key] = result, ms

        repeated = False
        for call, key in zip(calls, keys, strict=True):
            repeated = repeated or key == last_key
            last_key = key
            result = results[key]
            if call.name == "calc_nutrition" and "per_100g" in result:
                self._calc_results.append(NutritionPer100g(**result["per_100g"]))
            # pop: a duplicate later in the same turn counts as a reuse, not a second run.
            ms = fresh.pop(key, None)
            self._record(
                "tool_call",
                tool=call.name,
                arguments=call.arguments,
                result=shorten(result),
                message=None if ms is not None else "reused result of an identical earlier call",
                duration_ms=ms,
            )
            messages.append(
                Message(
                    role="tool",
                    content=tool_content(result),
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
        return repeated, last_key

    async def _timed(self, call: ToolCall) -> tuple[dict[str, Any], int]:
        started = time.monotonic()
        result = await self.tools.execute(call.name, call.arguments)
        return result, int((time.monotonic() - started) * 1000)

    async def _bounded[T](self, awaitable: Awaitable[T]) -> T:
        return await bounded(awaitable, self._deadline)

    def _parse_final(self, text: str) -> ReformulationAnswer:
        raw = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if not raw:
            raise ValueError("empty answer, expected a JSON object")
        try:
            answer = ReformulationAnswer.model_validate(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"not valid JSON: {exc}") from None
        except ValidationError as exc:
            raise ValueError(format_validation(exc)) from None
        self._check_nutrition(answer)
        self._check_goal(answer)
        self._cap_confidence(answer)
        return answer

    def _check_nutrition(self, answer: ReformulationAnswer) -> None:
        # Rule 3 enforced in code: before/after must be numbers calc_nutrition actually returned.
        if not self._calc_results:
            raise ValueError("nutrition_per_100g must come from calc_nutrition, never called")
        for label, value in answer.nutrition_per_100g:
            if not any(_close(value, calc) for calc in self._calc_results):
                raise ValueError(
                    f"nutrition_per_100g.{label} does not match any calc_nutrition result; "
                    "copy the values from calc_nutrition"
                )

    def _check_goal(self, answer: ReformulationAnswer) -> None:
        # The one goal with a number the code can verify: sugar per 100 g must drop enough.
        if self._request is None or self._request.goal != "reduce_sugar":
            return
        percent = self._request.goal_params.percent
        before = answer.nutrition_per_100g.before.sugar_g
        after = answer.nutrition_per_100g.after.sugar_g
        target = round(before * (1 - percent / 100), 2)
        if after > target + NUTRITION_TOLERANCE:
            cut = (1 - after / before) * 100 if before else 0
            raise ValueError(
                f"reduce_sugar needs sugar_g to drop by at least {percent}%: "
                f"{before} -> at most {target} g per 100 g, but after is {after} "
                f"(only {cut:.1f}% less). Cut more sugar and run calc_nutrition again"
            )

    def _cap_confidence(self, answer: ReformulationAnswer) -> None:
        # Fixed in place rather than rejected: a retry would cost an iteration for a fix the
        # code can make itself.
        for message in cap_confidence(answer.substitutions):
            self._record("correction", message=message)

    def _record(self, type_: str, **fields: Any) -> None:
        self.trace.append(
            TraceStep(step=len(self.trace) + 1, iteration=self._iteration, type=type_, **fields)
        )
        log_step(self.trace[-1])


def tool_content(result: dict[str, Any]) -> str:
    # Compact separators: every result is re-sent on each later turn.
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def _call_key(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"


def _close(a: NutritionPer100g, b: NutritionPer100g) -> bool:
    return all(
        abs(getattr(a, f) - getattr(b, f)) <= NUTRITION_TOLERANCE
        for f in NutritionPer100g.model_fields
    )
