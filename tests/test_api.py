import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.db import apply_migrations, create_pool
from app.llm.fake import FakeLLM
from app.main import app
from tests.test_pipeline import YOGURT, choice, plan
from tests.test_pipeline import knowledge_base as knowledge_base  # autouse: fixture DB + OFF

requires_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set (docker compose up db)"
)


def make_client() -> AsyncClient:
    # ASGITransport does not run the lifespan: tests put their own deps into app.state,
    # so no embedding model is loaded and no real LLM client is ever created.
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def db_client():
    pool = await create_pool(os.environ["DATABASE_URL"])
    await apply_migrations(pool)
    app.state.pool = pool
    app.state.llm = FakeLLM()
    try:
        async with make_client() as client:
            yield client
    finally:
        await pool.close()


@requires_db
async def test_health_ok(db_client):
    resp = await db_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "db": "ok",
        "llm_provider": get_settings().llm_provider,
    }


class UnreachablePool:
    def acquire(self, timeout=None):
        raise ConnectionRefusedError


async def test_health_reports_db_down():
    app.state.pool = UnreachablePool()
    async with make_client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"
    assert resp.json()["db"] == "error: ConnectionRefusedError"


async def fetch_run(run_id: str) -> dict:
    pool = app.state.pool
    row = await pool.fetchrow("SELECT * FROM reformulation_runs WHERE id = $1", int(run_id))
    return {**row, "trace": json.loads(row["trace"])}


@requires_db
async def test_reformulate_returns_answer_and_logs_the_run(db_client):
    app.state.llm = FakeLLM([plan(), choice()])
    app.state.embedder = app.state.http = None  # search and lookups are fixtures
    resp = await db_client.post("/reformulate", json=YOGURT.model_dump())

    assert resp.status_code == 200
    body = resp.json()
    assert body["allergens_after"] == ["soybeans"]
    assert [s["type"] for s in body["trace"] if s["type"] == "llm_call"] == ["llm_call"] * 2
    run = await fetch_run(resp.headers["X-Run-Id"])
    assert run["status"] == "ok"
    assert json.loads(run["response"])["substitutions"][0]["replacement"] == "соєвий напій"
    assert run["trace"] == body["trace"] and run["duration_ms"] >= 0


@requires_db
async def test_reformulate_invalid_answer_is_502_with_trace_and_logged(db_client):
    app.state.llm = FakeLLM([plan(), "not json", "still not json"])
    app.state.embedder = app.state.http = None
    resp = await db_client.post("/reformulate", json=YOGURT.model_dump())

    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["code"] == "agent_invalid_output"
    assert [s["type"] for s in body["trace"]].count("validation_error") == 2
    run = await fetch_run(resp.headers["X-Run-Id"])
    assert (run["status"], run["response"]) == ("agent_invalid_output", None)
    assert run["trace"] == body["trace"]


async def test_reformulate_rejects_params_that_do_not_match_the_goal():
    async with make_client() as client:
        resp = await client.post(
            "/reformulate",
            json=YOGURT.model_dump() | {"goal": "make_vegan"},  # still has allergen=milk
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
