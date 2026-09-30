"""recall@5 of the /ask search on eval/questions.jsonl, English vs Ukrainian.

A question is a hit when its expected_doc_id is among the documents of the top 5 chunks,
the same search and top_k /ask uses. Questions carry a `set`: "base" (general questions) and
"exact_terms" (the answer hinges on an exact term: E-code, trial ID, a number from a table);
recall is reported per set. Runs inside the api container, which has the embedding
model, the database and the LLM key (./eval is mounted there, like ./data):

    make eval             # en and uk as is: search only, no LLM calls
    make eval-translate   # plus uk translated by /ask's to_search_query: 1 LLM call per question
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.db import create_pool
from app.embeddings import Embedder
from app.llm.base import get_llm_client
from app.retrieval import has_chunks, search
from app.routers.ask import to_search_query

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
TOP_K = 5


async def main(translate: bool) -> int:
    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        if not await has_chunks(pool):
            print("knowledge base is empty: run `make ingest` first", file=sys.stderr)
            return 1
        embedder = Embedder(settings.embedding_model)
        llm = get_llm_client(settings) if translate else None

        async def hits(queries: list[str]) -> list[bool]:
            found = []
            for query, q in zip(queries, questions, strict=True):
                docs = {s.doc_id for s in await search(pool, embedder, query, TOP_K)}
                found.append(q["expected_doc_id"] in docs)
            return found

        rows = {
            "en": await hits([q["question_en"] for q in questions]),
            "uk (as is)": await hits([q["question_uk"] for q in questions]),
        }
        if llm is not None:
            translated = [await to_search_query(llm, q["question_uk"]) for q in questions]
            await llm.aclose()
            rows["uk → en (LLM)"] = await hits(translated)
            print("Translations:")
            for q, query in zip(questions, translated, strict=True):
                print(f"  {q['expected_doc_id']}: {query}")
            print()
    finally:
        await pool.close()

    print(report(questions, rows))
    return 0


def report(questions: list[dict], rows: dict[str, list[bool]]) -> str:
    """recall@5 per question set ("base", "exact_terms", ...) and per query language, plus
    the questions each row missed."""
    sets = list(dict.fromkeys(q.get("set", "base") for q in questions))
    header = "| Queries | " + " | ".join(
        f"{name} ({sum(q.get('set', 'base') == name for q in questions)})" for name in sets
    )
    lines = [header + " |", "|---" * (len(sets) + 1) + "|"]
    for label, found in rows.items():
        cells = []
        for name in sets:
            mine = [
                ok for q, ok in zip(questions, found, strict=True) if q.get("set", "base") == name
            ]
            cells.append(f"{sum(mine)}/{len(mine)} ({sum(mine) / len(mine):.0%})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    for label, found in rows.items():
        missed = [
            f"{q['expected_doc_id']} [{q.get('set', 'base')}]"
            for q, ok in zip(questions, found, strict=True)
            if not ok
        ]
        lines.append(f"missed, {label}: {', '.join(missed) or '-'}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--translate", action="store_true", help="also translate uk questions (live LLM calls)"
    )
    sys.exit(asyncio.run(main(parser.parse_args().translate)))
