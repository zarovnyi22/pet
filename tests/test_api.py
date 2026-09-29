import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.db import apply_migrations, create_pool
from app.llm.fake import FakeLLM
from app.main import app

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
