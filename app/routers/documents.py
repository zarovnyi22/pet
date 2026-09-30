from typing import Annotated

from fastapi import APIRouter, Body, Request

from app.errors import AppError
from app.ingest import ingest_document
from app.schemas import DocumentIn, DocumentOut

router = APIRouter()


DOCUMENT_EXAMPLES = {
    "guideline": {
        "summary": "A short guideline note",
        "value": {
            "doc_id": "note-demo",
            "title": "Demo note",
            "doc_type": "guideline",
            "content": "Reduce sucrose in steps of 10% and replace the mass with polydextrose.",
        },
    },
    "ingredient_spec": {
        "summary": "An ingredient spec with nutrient and allergen tables (parsed at ingest)",
        "value": {
            "doc_id": "spec-demo-rice-drink",
            "title": "Rice Drink (demo) — Ingredient Specification",
            "doc_type": "ingredient_spec",
            "content": (
                "# Rice Drink\n\nPlant-based milk replacement; low protein.\n\n"
                "## Allergens\n\n| | Value |\n|---|---|\n| Allergens | none |\n"
                "| Vegan | yes |\n\n## Nutrients per 100 g\n\n| Nutrient | Value |\n"
                "|---|---|\n| Energy | 47 kcal |\n| Protein | 0.1 g |\n| Fat | 1.0 g |\n"
                "| Carbohydrates | 9.4 g |\n| of which sugars | 4.0 g |\n"
            ),
        },
    },
}


@router.post("/documents", response_model=DocumentOut)
async def create_document(
    doc: Annotated[DocumentIn, Body(openapi_examples=DOCUMENT_EXAMPLES)], request: Request
) -> DocumentOut:
    try:
        n = await ingest_document(request.app.state.pool, request.app.state.embedder, doc)
    except ValueError as exc:  # a malformed allergens table: rejected, never stored as "none"
        raise AppError(422, "invalid_allergens_table", str(exc)) from None
    return DocumentOut(doc_id=doc.doc_id, chunks_created=n)
