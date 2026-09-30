from fastapi import APIRouter, Request

from app.errors import AppError
from app.ingest import ingest_document
from app.schemas import DocumentIn, DocumentOut

router = APIRouter()


@router.post("/documents", response_model=DocumentOut)
async def create_document(doc: DocumentIn, request: Request) -> DocumentOut:
    try:
        n = await ingest_document(request.app.state.pool, request.app.state.embedder, doc)
    except ValueError as exc:  # a malformed allergens table: rejected, never stored as "none"
        raise AppError(422, "invalid_allergens_table", str(exc)) from None
    return DocumentOut(doc_id=doc.doc_id, chunks_created=n)
