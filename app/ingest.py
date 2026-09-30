"""Ingest pipeline shared by POST /documents and the CLI.

CLI: python -m app.ingest data/corpus/
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import asyncpg

from app.allergens import parse_allergens_table
from app.chunking import chunk_text
from app.config import get_settings
from app.db import apply_migrations, create_pool
from app.embeddings import Embedder
from app.schemas import DocumentIn

logger = logging.getLogger("app.ingest")

# Row label in a spec's nutrient table -> our nutrient key. Other rows (fibre) are ignored.
NUTRIENT_ROWS = {
    "energy": "kcal",
    "protein": "protein_g",
    "fat": "fat_g",
    "carbohydrates": "carbs_g",
    "of which sugars": "sugar_g",
    # Not a nutrient: calc_nutrition only weighs the five above. The pipeline uses it for the
    # sucrose equivalent of sugars and sweeteners.
    "relative sweetness (sucrose = 1)": "sweetness",
}
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_nutrients_table(content: str) -> dict[str, dict[str, float]] | None:
    """The markdown table under "## Nutrients per 100 g" (the heading may carry a note, e.g.
    "(blend without hydrocolloid)"), as {column header: {nutrient: value}}.

    Most specs have one value column ("Value"); some compare variants side by side
    (e.g. bulk vs freeze-dried starter), so every column is kept.
    """
    section = re.search(r"^## Nutrients per 100 g\b[^\n]*$(.*?)(?=^## |\Z)", content, re.M | re.S)
    if not section:
        return None
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.group(1).splitlines()
        if line.strip().startswith("|") and not set(line.strip()) <= set("|-: ")
    ]
    if len(rows) < 2:
        return None
    columns = rows[0][1:]
    table: dict[str, dict[str, float]] = {column: {} for column in columns}
    for label, *cells in rows[1:]:
        key = NUTRIENT_ROWS.get(label.lower())
        if key is None:
            continue
        for column, cell in zip(columns, cells, strict=False):
            number = NUMBER.search(cell)  # "100 g (polyols)" -> 100
            if number:
                table[column][key] = float(number.group())
    table = {column: values for column, values in table.items() if values}
    return table or None


def to_pgvector(vector: list[float]) -> str:
    # pgvector text format; avoids a pgvector-python dependency for one cast.
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


async def ingest_document(pool: asyncpg.Pool, embedder: Embedder, doc: DocumentIn) -> int:
    settings = get_settings()
    # Parsed first: a malformed allergens table rejects the document before any model work.
    allergens = parse_allergens_table(doc.content)
    # CPU-bound model work runs before the transaction and off the event loop.
    chunks = await asyncio.to_thread(
        chunk_text,
        doc.content,
        embedder.token_spans,
        settings.chunk_size_tokens,
        settings.chunk_overlap_tokens,
    )
    embeddings = await asyncio.to_thread(embedder.embed, chunks)
    nutrients = parse_nutrients_table(doc.content)

    async with pool.acquire() as conn, conn.transaction():
        # The upsert row-locks the document, so concurrent re-ingests of one doc_id
        # serialize here instead of interleaving their chunks.
        await conn.execute(
            """
            INSERT INTO documents
              (doc_id, title, doc_type, content, nutrients_per_100g, allergens)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb)
            ON CONFLICT (doc_id) DO UPDATE
              SET title = EXCLUDED.title,
                  doc_type = EXCLUDED.doc_type,
                  content = EXCLUDED.content,
                  nutrients_per_100g = EXCLUDED.nutrients_per_100g,
                  allergens = EXCLUDED.allergens
            """,
            doc.doc_id,
            doc.title,
            doc.doc_type,
            doc.content,
            json.dumps(nutrients) if nutrients else None,
            json.dumps(allergens) if allergens else None,
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


async def backfill_allergens(pool: asyncpg.Pool) -> None:
    """Fill documents.allergens where it is NULL from the stored content, with the ingest
    parser; chunks and embeddings are left alone. Runs on every start, after the migrations,
    so rows ingested before 003_allergens get their facts without a manual re-ingest.

    Only content that has the table can be filled: a spec stored before the table was added
    to the corpus stays NULL (unknown, so the pipeline refuses it) until it is re-ingested.
    """
    rows = await pool.fetch(
        "SELECT doc_id, content FROM documents WHERE allergens IS NULL AND doc_type = $1",
        "ingredient_spec",
    )
    stale = []
    for row in rows:
        try:
            facts = parse_allergens_table(row["content"])
        except ValueError as exc:
            logger.warning(
                "allergens table invalid", extra={"doc_id": row["doc_id"], "error": str(exc)}
            )
            continue
        if facts is None:
            stale.append(row["doc_id"])
            continue
        await pool.execute(
            "UPDATE documents SET allergens = $2::jsonb WHERE doc_id = $1 AND allergens IS NULL",
            row["doc_id"],
            json.dumps(facts),
        )
    if stale:
        logger.warning(
            "specs without an allergens table: re-ingest them (make ingest)",
            extra={"doc_ids": stale},
        )


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
