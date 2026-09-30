"""Vector search over chunks in pgvector (shared by /ask and the agent's KB tool)."""

import asyncio
import json
from typing import Any

import asyncpg

from app.config import get_settings
from app.embeddings import Embedder
from app.ingest import to_pgvector
from app.schemas import Source


async def has_chunks(pool: asyncpg.Pool) -> bool:
    return await pool.fetchval("SELECT EXISTS (SELECT 1 FROM chunks)")


async def doc_nutrients(
    pool: asyncpg.Pool, doc_ids: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Parsed nutrient tables of these documents: {doc_id: {column: {nutrient: value}}}."""
    rows = await pool.fetch(
        """
        SELECT doc_id, nutrients_per_100g FROM documents
        WHERE doc_id = ANY($1::text[]) AND nutrients_per_100g IS NOT NULL
        """,
        doc_ids,
    )
    return {r["doc_id"]: json.loads(r["nutrients_per_100g"]) for r in rows}


async def doc_allergens(
    pool: asyncpg.Pool, doc_ids: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Parsed allergens tables: {doc_id: {column: {"allergens": [...], "vegan": bool | None}}},
    stored at ingest (or backfilled on start). A document without one is left out: its
    allergen status is unknown, and the pipeline treats unknown as not safe."""
    rows = await pool.fetch(
        """
        SELECT doc_id, allergens FROM documents
        WHERE doc_id = ANY($1::text[]) AND allergens IS NOT NULL
        """,
        doc_ids,
    )
    return {r["doc_id"]: json.loads(r["allergens"]) for r in rows}


async def spec_catalog(pool: asyncpg.Pool) -> list[tuple[str, str]]:
    """(doc_id, title) of every document with a nutrient table, for mapping ingredients."""
    rows = await pool.fetch(
        "SELECT doc_id, title FROM documents WHERE nutrients_per_100g IS NOT NULL ORDER BY doc_id"
    )
    return [(r["doc_id"], r["title"]) for r in rows]


# Hybrid search: each side proposes this many chunks, Reciprocal Rank Fusion picks top_k.
CANDIDATES = 20
RRF_K = 60  # the constant from the RRF paper: dampens the weight of the very first ranks
# The question's lexemes (stemmed, stop words dropped). OR, not AND: plainto_tsquery /
# websearch_to_tsquery need every word of a long question in one chunk and usually match
# nothing. But an OR ranked by ts_rank_cd alone has no IDF: for "Which of our ingredients
# contains E418?" chunks full of "ingredient"/"contain" beat the one chunk with the rare
# "e418". So a chunk scores the sum of ln(N / chunks with the lexeme) over the lexemes it
# contains (BM25 without term frequency), and ts_rank_cd breaks ties. ts_stat reads every
# tsvector per query: fine for a corpus of ~100 chunks; at scale, keep lexeme counts in a table.
#
# Only rare lexemes take part. With every lexeme, the OR matched a dozen chunks for any question
# (common words: "ingredient", "contain", "kcal") and overlapped the vector list; RRF then
# ranked chunks sitting in both lists at middling ranks (1/65 + 1/68) above the one chunk the
# full-text search put first (1/61): E418 and "380 kcal" ended 8th and 11th, and a trial ID
# the vector search had found was pushed out. A lexeme in more than a tenth of the chunks is a
# stop word for this corpus; the result was the same for any cut-off from 5% to 20%.
RARE_TERM_MAX_SHARE = 0.10
FULLTEXT_SQL = """
WITH terms AS (
  SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english', $1))) AS word
), stats AS (
  SELECT s.word, ln((SELECT count(*) FROM chunks)::float8 / s.ndoc) AS idf
  FROM ts_stat('SELECT tsv FROM chunks') AS s JOIN terms USING (word)
  WHERE s.ndoc <= $3::float8 * (SELECT count(*) FROM chunks)
), any_term AS (
  SELECT replace(plainto_tsquery('english', $1)::text, ' & ', ' | ')::tsquery AS q
)
SELECT c.id
FROM chunks c
JOIN stats ON c.tsv @@ quote_literal(stats.word)::tsquery
CROSS JOIN any_term
GROUP BY c.id, c.tsv, any_term.q
ORDER BY sum(stats.idf) DESC, ts_rank_cd(c.tsv, any_term.q) DESC, c.id
LIMIT $2
"""


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: sum of 1 / (k + rank) over the rankings an id appears in.
    Ties keep first-seen order (vector ranking first)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])  # sorted() is stable


async def vector_ranking(pool: asyncpg.Pool, vector: str, limit: int) -> list[int]:
    # `<=>` is cosine distance, the operator the hnsw index (vector_cosine_ops) serves.
    rows = await pool.fetch(
        "SELECT id FROM chunks ORDER BY embedding <=> $1::vector LIMIT $2", vector, limit
    )
    return [r["id"] for r in rows]


async def fulltext_ranking(pool: asyncpg.Pool, query: str, limit: int) -> list[int]:
    rows = await pool.fetch(FULLTEXT_SQL, query, limit, RARE_TERM_MAX_SHARE)
    return [r["id"] for r in rows]


async def search(
    pool: asyncpg.Pool,
    embedder: Embedder,
    query: str,
    top_k: int,
    *,
    hybrid: bool | None = None,
) -> list[Source]:
    """Top chunks for the query: vector search, plus full-text search fused by RRF when
    HYBRID_SEARCH is on (default). Shared by /ask and the agent's knowledge-base tool."""
    hybrid = get_settings().hybrid_search if hybrid is None else hybrid
    [embedding] = await asyncio.to_thread(embedder.embed, [query])
    vector = to_pgvector(embedding)
    by_vector = await vector_ranking(pool, vector, CANDIDATES if hybrid else top_k)
    by_text = await fulltext_ranking(pool, query, CANDIDATES) if hybrid else []
    chosen = [item for item, _ in rrf([by_vector, by_text])][:top_k]

    # Vectors are normalized, so 1 - cosine distance is cosine similarity in [-1, 1].
    rows = await pool.fetch(
        """
        SELECT c.id, c.doc_id, d.title, c.text, 1 - (c.embedding <=> $1::vector) AS score
        FROM chunks c JOIN documents d USING (doc_id)
        WHERE c.id = ANY($2::bigint[])
        """,
        vector,
        chosen,
    )
    by_id = {r["id"]: r for r in rows}
    in_vector, in_text = set(by_vector), set(by_text)
    return [
        Source(
            doc_id=by_id[i]["doc_id"],
            title=by_id[i]["title"],
            chunk_text=by_id[i]["text"],
            score=round(by_id[i]["score"], 4),
            matched_by=[
                name for name, found in (("vector", in_vector), ("fulltext", in_text)) if i in found
            ],
        )
        for i in chosen
        if i in by_id
    ]
