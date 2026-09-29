"""Shared fixtures: an isolated test database and a fake embedder.

Tests never load the real embedding model and never call a live LLM or Open Food Facts.
"""

import hashlib
import math
import os
import re
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from app.db import apply_migrations, create_pool
from app.embeddings import EMBEDDING_DIM
from app.llm.fake import FakeLLM
from app.main import app

WORD = re.compile(r"\S+")


class FakeEmbedder:
    """Stands in for app.embeddings.Embedder: whitespace tokens and bag-of-words vectors.

    Each word is hashed into one of 384 dimensions, so texts sharing words have a higher
    cosine similarity: enough for retrieval tests to be meaningful, with no model download.
    """

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in WORD.finditer(text)]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()  # stable across runs, unlike hash()
            vector[int.from_bytes(digest[:4], "big") % EMBEDDING_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def make_client() -> AsyncClient:
    # ASGITransport does not run the lifespan: tests put their own deps into app.state,
    # so no embedding model is loaded and no real LLM client is ever created.
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set (docker compose run --rm test sets it)")
    name = urlsplit(url).path.lstrip("/")
    if not name.endswith("_test"):
        # The fixtures truncate every table: never point them at the demo database.
        pytest.exit(f"DATABASE_URL must name a *_test database, got {name!r}", returncode=2)
    return url


async def _ensure_database(url: str) -> None:
    parts = urlsplit(url)
    name = parts.path.lstrip("/")
    admin = await asyncpg.connect(urlunsplit(parts._replace(path="/postgres")))
    try:
        if not await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()


@pytest.fixture
async def db_pool(database_url):
    await _ensure_database(database_url)
    pool = await create_pool(database_url)
    await apply_migrations(pool)
    await pool.execute("TRUNCATE documents, chunks, reformulation_runs RESTART IDENTITY CASCADE")
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def db_client(db_pool):
    app.state.pool = db_pool
    app.state.llm = FakeLLM()
    app.state.embedder = FakeEmbedder()
    app.state.http = None  # no test may reach Open Food Facts
    async with make_client() as client:
        yield client
