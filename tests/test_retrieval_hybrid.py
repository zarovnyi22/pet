"""Hybrid search: RRF on plain data, and vector + full-text search on the test database."""

from pathlib import Path

import pytest

from app.ingest import ingest_document, parse_markdown
from app.retrieval import fulltext_ranking, rrf, search
from tests.conftest import FakeEmbedder

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
DOCS = ["spec-soy-drink", "spec-stevia", "spec-milk-2-5", "spec-erythritol", "spec-wheat-flour"]
E418 = "Which of our ingredients contains E418, and what is E418?"
STEVIA = "Above what dosage does stevia give a bitter liquorice aftertaste?"


def test_rrf_sums_reciprocal_ranks_and_keeps_vector_order_on_ties():
    fused = rrf([[1, 2, 3], [3, 4]], k=60)
    assert [item for item, _ in fused] == [3, 1, 2, 4]  # 3 is in both; 2 and 4 tie, 2 seen first
    assert fused[0][1] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_with_one_empty_ranking_is_that_ranking():
    assert [item for item, _ in rrf([[5, 6, 7], []])] == [5, 6, 7]


@pytest.fixture
async def corpus(db_pool):
    embedder = FakeEmbedder()
    for doc_id in DOCS:
        await ingest_document(
            db_pool, embedder, parse_markdown((CORPUS / f"{doc_id}.md").read_text())
        )
    return db_pool, embedder


async def doc_of(pool, chunk_id: int) -> str:
    return await pool.fetchval("SELECT doc_id FROM chunks WHERE id = $1", chunk_id)


async def test_every_chunk_gets_its_tsvector(corpus):
    pool, _ = corpus
    assert await pool.fetchval("SELECT count(*) FROM chunks WHERE tsv IS NULL") == 0
    assert await pool.fetchval("SELECT count(*) FROM chunks") >= len(DOCS)


async def test_exact_term_is_found_by_full_text(corpus):
    pool, _ = corpus
    ranking = await fulltext_ranking(pool, E418, 20)
    assert await doc_of(pool, ranking[0]) == "spec-soy-drink"


async def test_long_question_matches_with_or_where_and_matches_nothing(corpus):
    pool, _ = corpus
    and_matches = await pool.fetchval(
        "SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', $1)", STEVIA
    )
    ranking = await fulltext_ranking(pool, STEVIA, 20)

    assert and_matches == 0  # every word of the question in one chunk: none
    assert await doc_of(pool, ranking[0]) == "spec-stevia"


async def test_hybrid_search_marks_full_text_hits(corpus):
    pool, embedder = corpus
    sources = await search(pool, embedder, E418, 5, hybrid=True)

    soy = [s for s in sources if s.doc_id == "spec-soy-drink"]
    assert soy and "fulltext" in soy[0].matched_by
    assert all(-1 <= s.score <= 1 for s in sources)  # a real cosine, for full-text hits too


async def test_hybrid_off_is_vector_search_as_before(corpus):
    pool, embedder = corpus
    sources = await search(pool, embedder, E418, 5, hybrid=False)

    assert len(sources) == 5
    assert all(s.matched_by == ["vector"] for s in sources)
    scores = [s.score for s in sources]
    assert scores == sorted(scores, reverse=True)  # plain cosine order, as before


async def test_common_words_alone_propose_nothing(corpus):
    # In most chunks of these specs: a stop word for this corpus, so no full-text proposal
    # that would only echo the vector search.
    pool, _ = corpus
    assert await fulltext_ranking(pool, "ingredient per 100 g", 20) == []
