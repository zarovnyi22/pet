"""Vector search over chunks in pgvector (shared by /ask and the agent's KB tool)."""

import asyncio
import json

import asyncpg

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


async def spec_catalog(pool: asyncpg.Pool) -> list[tuple[str, str]]:
    """(doc_id, title) of every document with a nutrient table, for mapping ingredients."""
    rows = await pool.fetch(
        "SELECT doc_id, title FROM documents WHERE nutrients_per_100g IS NOT NULL ORDER BY doc_id"
    )
    return [(r["doc_id"], r["title"]) for r in rows]


async def search(pool: asyncpg.Pool, embedder: Embedder, query: str, top_k: int) -> list[Source]:
    [vector] = await asyncio.to_thread(embedder.embed, [query])
    # `<=>` is cosine distance, the operator the hnsw index (vector_cosine_ops) serves.
    # Vectors are normalized, so score = 1 - distance is cosine similarity in [-1, 1].
    rows = await pool.fetch(
        """
        SELECT c.doc_id, d.title, c.text, 1 - (c.embedding <=> $1::vector) AS score
        FROM chunks c
        JOIN documents d USING (doc_id)
        ORDER BY c.embedding <=> $1::vector
        LIMIT $2
        """,
        to_pgvector(vector),
        top_k,
    )
    return [
        Source(
            doc_id=r["doc_id"], title=r["title"], chunk_text=r["text"], score=round(r["score"], 4)
        )
        for r in rows
    ]
