"""Ingest pipeline shared by POST /documents and the CLI.

CLI: python -m app.ingest data/corpus/
"""

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

from app.chunking import chunk_text
from app.config import get_settings
from app.db import apply_migrations, create_pool
from app.embeddings import Embedder
from app.schemas import DocumentIn


def to_pgvector(vector: list[float]) -> str:
    # pgvector text format; avoids a pgvector-python dependency for one cast.
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


async def ingest_document(pool: asyncpg.Pool, embedder: Embedder, doc: DocumentIn) -> int:
    settings = get_settings()
    # CPU-bound model work runs before the transaction and off the event loop.
    chunks = await asyncio.to_thread(
        chunk_text,
        doc.content,
        embedder.token_spans,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    embeddings = await asyncio.to_thread(embedder.embed, chunks)

    async with pool.acquire() as conn, conn.transaction():
        # The upsert row-locks the document, so concurrent re-ingests of one doc_id
        # serialize here instead of interleaving their chunks.
        await conn.execute(
            """
            INSERT INTO documents (doc_id, title, doc_type, content)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (doc_id) DO UPDATE
              SET title = EXCLUDED.title,
                  doc_type = EXCLUDED.doc_type,
                  content = EXCLUDED.content
            """,
            doc.doc_id,
            doc.title,
            doc.doc_type,
            doc.content,
        )
        await conn.execute("DELETE FROM chunks WHERE doc_id = $1", doc.doc_id)
        await conn.executemany(
            """
            INSERT INTO chunks (doc_id, chunk_index, text, embedding)
            VALUES ($1, $2, $3, $4::vector)
            """,
            [
                (doc.doc_id, i, text, to_pgvector(vector))
                for i, (text, vector) in enumerate(zip(chunks, embeddings, strict=True))
            ],
        )
    return len(chunks)


def parse_markdown(raw: str) -> DocumentIn:
    """Split `---` frontmatter (flat `key: value` lines) from the markdown body."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing frontmatter")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        raise ValueError("unterminated frontmatter") from None

    meta = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip()
    return DocumentIn(
        doc_id=meta.get("doc_id", ""),
        title=meta.get("title", ""),
        doc_type=meta.get("doc_type", ""),
        content=body,
    )


async def ingest_folder(folder: Path) -> int:
    paths = sorted(folder.glob("*.md"))
    if not paths:
        print(f"no .md files in {folder}", file=sys.stderr)
        return 1

    settings = get_settings()
    embedder = Embedder(settings.embedding_model)
    pool = await create_pool(settings.database_url)
    failed = 0
    try:
        await apply_migrations(pool)
        for path in paths:
            try:
                doc = parse_markdown(path.read_text(encoding="utf-8"))
                n = await ingest_document(pool, embedder, doc)
                print(f"{doc.doc_id}: {n} chunks")
            except Exception as exc:  # one bad file must not stop the rest
                failed += 1
                print(f"{path.name}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        docs = await pool.fetchval("SELECT count(*) FROM documents")
        chunks = await pool.fetchval("SELECT count(*) FROM chunks")
        ok = len(paths) - failed
        print(f"done: {ok}/{len(paths)} files; db has {docs} documents, {chunks} chunks")
    finally:
        await pool.close()
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of markdown documents.")
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    sys.exit(asyncio.run(ingest_folder(args.folder)))


if __name__ == "__main__":
    main()
