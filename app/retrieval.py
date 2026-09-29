"""Vector search over chunks in pgvector (shared by /ask and the agent's KB tool)."""

import asyncio

import asyncpg

from app.embeddings import Embedder
from app.ingest import to_pgvector
from app.schemas import Source


async def has_chunks(pool: asyncpg.Pool) -> bool:
    return await pool.fetchval("SELECT EXISTS (SELECT 1 FROM chunks)")


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
