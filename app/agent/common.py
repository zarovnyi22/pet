"""Pieces shared by both orchestrations of /reformulate: the fixed pipeline (production path,
app/agent/pipeline.py) and the free tool-calling loop (first attempt, app/agent/loop.py)."""

import asyncio
import time
from collections.abc import Awaitable
from typing import Any

from pydantic import ValidationError

from app.errors import AppError
from app.schemas import Substitution, TraceStep

TRACE_STRING_LIMIT = 300
# Only trial reports prove a substitution works in a real product; specs and OFF only suggest.
HIGH_CONFIDENCE_PREFIX = "trial-"


class AgentError(AppError):
    """A failed run. Carries the trace gathered so far, for the response and the run log."""

    def __init__(self, status_code: int, code: str, message: str, trace: list[TraceStep]) -> None:
        super().__init__(status_code, code, message)
        self.trace = trace


class OutOfTime(Exception):
    pass


async def bounded[T](awaitable: Awaitable[T], deadline: float) -> T:
    """Await within the run's wall-clock budget (time.monotonic() deadline)."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        elif isinstance(awaitable, asyncio.Future):
            awaitable.cancel()  # a gather() whose tool tasks are already scheduled
        raise OutOfTime
    try:
        return await asyncio.wait_for(awaitable, remaining)
    except TimeoutError:
        raise OutOfTime from None


def cap_confidence(substitutions: list[Substitution]) -> list[str]:
    """Lower "high" to "medium" unless a trial report backs it; returns what was changed."""
    changed = []
    for sub in substitutions:
        if sub.confidence == "high" and not any(
            s.startswith(HIGH_CONFIDENCE_PREFIX) for s in sub.sources
        ):
            sub.confidence = "medium"
            changed.append(
                f"confidence of {sub.original!r} -> {sub.replacement!r} lowered from "
                f"high to medium: no trial report among sources {sub.sources}"
            )
    return changed


def format_validation(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'answer'}: {e['msg']}"
        for e in exc.errors(include_url=False)
    )


def shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > TRACE_STRING_LIMIT:
        return value[:TRACE_STRING_LIMIT] + "…"
    if isinstance(value, dict):
        return {k: shorten(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v) for v in value]
    return value
