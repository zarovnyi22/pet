import json

import pytest

from app.llm.base import LLMError
from app.llm.fake import FakeLLM
from app.main import app
from app.routers import ask as ask_mod
from app.routers.ask import parse_answer, select_sources, to_search_query
from app.schemas import Source
from tests.conftest import make_client


def src(doc_id: str) -> Source:
    return Source(doc_id=doc_id, title=doc_id, chunk_text="...", score=0.5)


RETRIEVED = [src("spec-whole-egg"), src("spec-aquafaba"), src("spec-whole-egg")]


def test_keeps_only_cited_sources():
    assert select_sources(RETRIEVED, ["spec-aquafaba"]) == [RETRIEVED[1]]


def test_unknown_doc_id_is_ignored():
    picked = select_sources(RETRIEVED, ["spec-aquafaba", "made-up-doc"])
    assert picked == [RETRIEVED[1]]


def test_only_unknown_doc_ids_fall_back_to_all_retrieved():
    assert select_sources(RETRIEVED, ["made-up-doc"]) == RETRIEVED
    assert select_sources(RETRIEVED, []) == RETRIEVED


@pytest.mark.parametrize(
    ("cited", "expected"),
    [(["spec-aquafaba"], ["spec-aquafaba"]), ([1, 3], ["1", "3"]), (None, [])],
)
def test_sloppy_citation_list_does_not_fail(cited, expected):
    raw = json.dumps({"found": True, "answer": "a", "cited_doc_ids": cited})
    assert parse_answer(raw).cited_doc_ids == expected


def test_json_in_code_fence_is_accepted():
    assert parse_answer('```json\n{"found": false}\n```').found is False


def test_non_json_answer_is_llm_error():
    with pytest.raises(LLMError):
        parse_answer("Sure! Here is the answer: eggs.")


# --- non-English questions are searched in English ------------------------------------------


@pytest.fixture
def searched(monkeypatch):
    """/ask without a database: records what it searched for, returns one aquafaba chunk."""
    queries = []

    async def search(pool, embedder, query, top_k):
        queries.append(query)
        return [src("spec-aquafaba")]

    async def has_chunks(pool):
        return True

    monkeypatch.setattr(ask_mod, "search", search)
    monkeypatch.setattr(ask_mod, "has_chunks", has_chunks)
    app.state.pool = app.state.embedder = None
    return queries


ANSWER = json.dumps({"found": True, "answer": "45 г аквафаби [spec-aquafaba]", "cited_doc_ids": []})


async def test_ukrainian_question_is_searched_by_its_translation(searched):
    question = "Чим замінити яйце в бісквіті?"
    app.state.llm = llm = FakeLLM(['"egg replacement in sponge cake"', ANSWER])
    async with make_client() as client:
        resp = await client.post("/ask", json={"question": question})

    assert resp.status_code == 200
    assert searched == ["egg replacement in sponge cake"]  # quotes stripped
    assert llm.calls[0][-1].content == question  # the translation request
    assert llm.calls[1][-1].content.endswith(f"Question: {question}")  # answer: the original


async def test_english_question_is_searched_as_is_without_a_translation_call(searched):
    question = "What replaces egg in a sponge cake?"
    app.state.llm = llm = FakeLLM([ANSWER])
    async with make_client() as client:
        resp = await client.post("/ask", json={"question": question})

    assert resp.status_code == 200
    assert searched == [question]
    assert len(llm.calls) == 1  # the answer only


async def test_to_search_query_keeps_the_question_if_the_translation_is_empty():
    assert await to_search_query(FakeLLM(["  "]), "Що таке аквафаба?") == "Що таке аквафаба?"
