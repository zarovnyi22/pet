import json
from typing import Annotated, Any

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, BeforeValidator, ValidationError

from app.errors import AppError
from app.llm.base import LLMClient, LLMError, Message
from app.retrieval import has_chunks, search
from app.schemas import AskIn, AskOut, Source

router = APIRouter()

NO_DATA_ANSWER = "The knowledge base has no data to answer this question."

SYSTEM_PROMPT = """You answer questions for food R&D technologists using ONLY the \
sources provided. Do not use outside knowledge, do not guess.

Reply with a single JSON object and nothing else:
{"found": true|false, "answer": "...", "cited_doc_ids": ["..."]}

- found=false if the sources do not contain the answer; then answer="" and cited_doc_ids=[].
- found=true: answer concisely, keep concrete numbers (dosages, grams, percentages) from the \
sources, and put the doc_id in square brackets after each claim, e.g. [spec-egg].
- cited_doc_ids lists every doc_id you used.
- Write the answer in the language of the question (sources are in English), in its standard \
literary form: do not mix languages or use words from another language (e.g. no Russian words \
in a Ukrainian answer)."""

TRANSLATE_PROMPT = """Translate the user's question into a short English search query for a \
food R&D knowledge base. Keep ingredient names, numbers and units. Return only the query: \
no quotes, no explanations."""


def _loose_id_list(value: Any) -> Any:
    # Citations only filter sources, so a sloppy list (null, [1, 3]) must not fail the request.
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else value


class LLMAnswer(BaseModel):
    found: bool
    answer: str = ""
    cited_doc_ids: Annotated[list[str], BeforeValidator(_loose_id_list)] = []


async def to_search_query(llm: LLMClient, question: str) -> str:
    """The question as an English search query: the corpus and all-MiniLM-L6-v2 are English,
    so an untranslated Ukrainian question barely matches anything. One LLM call, and only
    when the question has non-ASCII letters; an English question is searched as is."""
    if not any(ch.isalpha() and not ch.isascii() for ch in question):
        return question
    messages = [
        Message(role="system", content=TRANSLATE_PROMPT),
        Message(role="user", content=question),
    ]
    query = (await llm.complete(messages)).strip().strip('"').strip()
    return query or question


def build_prompt(question: str, sources: list[Source]) -> list[Message]:
    # Sources are labelled by doc_id only: numbered labels get cited as [1] instead of doc_id.
    context = "\n\n".join(
        f"<source doc_id={s.doc_id!r} title={s.title!r}>\n{s.chunk_text}\n</source>"
        for s in sources
    )
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=f"Sources:\n\n{context}\n\nQuestion: {question}"),
    ]


def parse_answer(raw: str) -> LLMAnswer:
    # Some models wrap JSON in ``` fences even in JSON mode.
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return LLMAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError(f"LLM answer is not valid JSON: {type(exc).__name__}") from None


def select_sources(sources: list[Source], cited_doc_ids: list[str]) -> list[Source]:
    """Chunks the answer actually cites. Ids not among the retrieved chunks are ignored;
    if none match, fall back to everything the model was shown."""
    cited = set(cited_doc_ids)
    return [s for s in sources if s.doc_id in cited] or sources


# Swagger (/docs) offers these in a dropdown: an English question is searched as is, a
# Ukrainian one is translated for the search first and answered in Ukrainian.
ASK_EXAMPLES = {
    "english": {
        "summary": "English question",
        "value": {"question": "What can replace eggs in a sponge cake and at what dosage?"},
    },
    "ukrainian": {
        "summary": "Українське питання (переклад для пошуку)",
        "value": {"question": "Чим замінити яйце в бісквіті?", "top_k": 5},
    },
}


@router.post("/ask", response_model=AskOut)
async def ask(
    body: Annotated[AskIn, Body(openapi_examples=ASK_EXAMPLES)], request: Request
) -> AskOut:
    state = request.app.state
    if not await has_chunks(state.pool):
        raise AppError(409, "empty_knowledge_base", "No documents are indexed yet.")

    query = await to_search_query(state.llm, body.question)
    sources = await search(state.pool, state.embedder, query, body.top_k)
    # The answer sees the original question, so it replies in the asker's language.
    raw = await state.llm.complete(build_prompt(body.question, sources), json_mode=True)
    result = parse_answer(raw)

    if not result.found or not result.answer.strip():
        return AskOut(answer=NO_DATA_ANSWER, sources=[])
    return AskOut(answer=result.answer, sources=select_sources(sources, result.cited_doc_ids))
