import json
import logging

import pytest

from app.llm.fake import FakeLLM
from app.logs import JsonFormatter, request_id_var
from app.main import app
from tests.test_pipeline import YOGURT, choice, plan
from tests.test_pipeline import knowledge_base as knowledge_base  # fixture specs + OFF


class Captured(logging.Handler):
    """Collects log lines exactly as they would be printed: formatted JSON."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.lines: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(json.loads(self.format(record)))


@pytest.fixture
def logs():
    handler = Captured()
    logging.getLogger().addHandler(handler)
    try:
        yield handler.lines
    finally:
        logging.getLogger().removeHandler(handler)


def test_formatter_writes_one_json_object_with_request_id_and_extras():
    token = request_id_var.set("abc123")
    try:
        record = logging.makeLogRecord(
            {"name": "app.agent", "levelname": "INFO", "msg": "agent %s", "args": ("step",)}
        )
        record.duration_ms = 42
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)

    assert line["message"] == "agent step"
    assert (line["request_id"], line["duration_ms"], line["logger"]) == ("abc123", 42, "app.agent")
    assert line["ts"].endswith("+00:00")


async def test_every_request_gets_an_id_in_header_and_log(db_client, logs):
    resp = await db_client.get("/health")
    request_id = resp.headers["X-Request-ID"]

    [line] = [x for x in logs if x["message"] == "request"]
    assert line["request_id"] == request_id
    assert (line["path"], line["status"]) == ("/health", 200)


async def test_client_request_id_is_reused_only_if_well_formed(db_client):
    ok = await db_client.get("/health", headers={"X-Request-ID": "demo-42"})
    assert ok.headers["X-Request-ID"] == "demo-42"
    forged = await db_client.get("/health", headers={"X-Request-ID": 'x", "level": "ERROR'})
    assert forged.headers["X-Request-ID"] != 'x", "level": "ERROR'


@pytest.mark.usefixtures("knowledge_base")
async def test_agent_steps_are_logged_under_the_request_id(db_client, logs):
    app.state.llm = FakeLLM([plan(), choice()])
    app.state.embedder = app.state.http = None
    resp = await db_client.post("/reformulate", json=YOGURT.model_dump())
    assert resp.status_code == 200
    request_id = resp.headers["X-Request-ID"]

    agent = [x for x in logs if x["logger"] == "app.agent"]
    # One log line per trace step, all tagged with the request that caused them.
    assert len(agent) == len(resp.json()["trace"])
    assert {x["request_id"] for x in agent} == {request_id}
    assert {"agent llm_call", "agent tool_call"} <= {x["message"] for x in agent}
    [run] = [x for x in logs if x["message"] == "reformulation run"]
    assert (run["request_id"], run["status"]) == (request_id, "ok")
