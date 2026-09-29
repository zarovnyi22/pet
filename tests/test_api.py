import json

import pytest

from app.config import get_settings
from app.llm.fake import FakeLLM
from app.main import app
from app.routers.ask import NO_DATA_ANSWER
from tests.conftest import make_client
from tests.test_pipeline import YOGURT, choice, plan
from tests.test_pipeline import knowledge_base as knowledge_base  # fixture specs + OFF

EGG_SPEC = {
    "doc_id": "spec-whole-egg",
    "title": "Whole Egg",
    "doc_type": "ingredient_spec",
    "content": (
        "# Whole Egg\n\nEggs bind, aerate and emulsify cake batters. Aquafaba replaces one "
        "egg with 45 g chickpea water.\n\n## Nutrients per 100 g\n\n| Nutrient | Value |\n"
        "|---|---|\n| Energy | 143 kcal |\n| Protein | 12.6 g |\n| Fat | 9.5 g |\n"
        "| Carbohydrates | 0.7 g |\n| of which sugars | 0.4 g |\n"
    ),
}
SUGAR_GUIDE = {
    "doc_id": "guideline-sugar-reduction",
    "title": "Sugar reduction",
    "doc_type": "guideline",
    "content": "Cut sucrose in steps of 10% and replace the mass with polydextrose.",
}


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


# --- POST /documents -------------------------------------------------------------------------


async def test_document_is_chunked_and_stored(db_client):
    resp = await db_client.post("/documents", json=EGG_SPEC)

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "spec-whole-egg" and body["chunks_created"] >= 1
    pool = app.state.pool
    assert await pool.fetchval("SELECT count(*) FROM chunks") == body["chunks_created"]
    table = await pool.fetchval(
        "SELECT nutrients_per_100g FROM documents WHERE doc_id = 'spec-whole-egg'"
    )
    assert json.loads(table)["Value"]["protein_g"] == 12.6  # parsed at ingest


async def test_reingest_replaces_chunks_instead_of_adding(db_client):
    first = (await db_client.post("/documents", json=EGG_SPEC)).json()["chunks_created"]
    shorter = EGG_SPEC | {"content": "Eggs bind batters."}
    second = (await db_client.post("/documents", json=shorter)).json()["chunks_created"]

    pool = app.state.pool
    assert first >= 1 and second == 1
    assert await pool.fetchval("SELECT count(*) FROM chunks") == 1
    assert await pool.fetchval("SELECT count(*) FROM documents") == 1
    assert await pool.fetchval("SELECT nutrients_per_100g FROM documents") is None


async def test_invalid_document_uses_the_common_error_format(db_client):
    resp = await db_client.post("/documents", json=EGG_SPEC | {"doc_type": "recipe"})
    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "validation_error" and "doc_type" in error["message"]


# --- POST /ask ---------------------------------------------------------------------------------


async def test_ask_on_empty_knowledge_base_is_409(db_client):
    resp = await db_client.post("/ask", json={"question": "What replaces eggs?"})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "empty_knowledge_base"


async def test_ask_answers_from_the_retrieved_chunks_and_cites_them(db_client):
    await db_client.post("/documents", json=EGG_SPEC)
    await db_client.post("/documents", json=SUGAR_GUIDE)
    answer = {
        "found": True,
        "answer": "Aquafaba: 45 g per egg [spec-whole-egg].",
        "cited_doc_ids": ["spec-whole-egg"],
    }
    app.state.llm = FakeLLM([json.dumps(answer)])

    resp = await db_client.post("/ask", json={"question": "What replaces one egg?", "top_k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == answer["answer"]
    assert {s["doc_id"] for s in body["sources"]} == {"spec-whole-egg"}  # only cited ones
    # Retrieval ran: the egg chunk is the best match and reached the LLM prompt.
    prompt = app.state.llm.calls[0][-1].content
    assert "Aquafaba replaces one egg" in prompt
    assert prompt.index("spec-whole-egg") < prompt.index("guideline-sugar-reduction")


async def test_ask_says_no_data_instead_of_guessing(db_client):
    await db_client.post("/documents", json=SUGAR_GUIDE)
    app.state.llm = FakeLLM([json.dumps({"found": False})])

    resp = await db_client.post("/ask", json={"question": "Shelf life of butter?"})

    assert resp.json() == {"answer": NO_DATA_ANSWER, "sources": []}


async def test_ask_with_invalid_llm_answer_is_502(db_client):
    await db_client.post("/documents", json=SUGAR_GUIDE)
    app.state.llm = FakeLLM(["Sure! Use polydextrose."])

    resp = await db_client.post("/ask", json={"question": "How to cut sugar?"})

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "llm_error"


# --- POST /reformulate ---------------------------------------------------------------------------


async def fetch_run(run_id: str) -> dict:
    pool = app.state.pool
    row = await pool.fetchrow("SELECT * FROM reformulation_runs WHERE id = $1", int(run_id))
    return {**row, "trace": json.loads(row["trace"])}


@pytest.mark.usefixtures("knowledge_base")
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


@pytest.mark.usefixtures("knowledge_base")
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
