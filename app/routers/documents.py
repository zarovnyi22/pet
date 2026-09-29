from fastapi import APIRouter, Request

from app.ingest import ingest_document
from app.schemas import DocumentIn, DocumentOut

router = APIRouter()


@router.post("/documents", response_model=DocumentOut)
async def create_document(doc: DocumentIn, request: Request) -> DocumentOut:
    n = await ingest_document(request.app.state.pool, request.app.state.embedder, doc)
    return DocumentOut(doc_id=doc.doc_id, chunks_created=n)
