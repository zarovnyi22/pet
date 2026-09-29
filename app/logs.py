"""Structured JSON logs: one formatter for every logger, with the current request's id.

The request id lives in a ContextVar, so it follows the request into everything it awaits -
routers, the agent pipeline, asyncio.gather() tool calls and asyncio.to_thread() work - without
being passed around by hand.
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Attributes every LogRecord has; anything else came from `extra=` and is logged as a field
# (except uvicorn's color_message: the same text with ANSI codes).
_STANDARD = set(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        entry |= {k: v for k, v in vars(record).items() if k not in _STANDARD}
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Route every logger, uvicorn's included, through one JSON handler on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    # The request middleware logs every request as JSON with its id: no plain-text duplicate.
    logging.getLogger("uvicorn.access").disabled = True
    # httpx logs every outgoing URL at INFO; app.llm already logs each LLM call with its timing.
    logging.getLogger("httpx").setLevel(logging.WARNING)
