import json

import pytest

from app.llm.base import LLMError
from app.routers.ask import parse_answer, select_sources
from app.schemas import Source


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
