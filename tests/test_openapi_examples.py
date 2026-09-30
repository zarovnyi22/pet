"""Request examples shown in Swagger (/docs): present in the schema and valid requests."""

import pytest

from app.allergens import parse_allergens_table
from app.ingest import parse_nutrients_table
from app.main import app
from app.routers.ask import ASK_EXAMPLES
from app.routers.documents import DOCUMENT_EXAMPLES
from app.routers.reformulate import REFORMULATE_EXAMPLES
from app.schemas import AskIn, DocumentIn, ReformulateIn


@pytest.mark.parametrize(
    ("path", "names"),
    [
        ("/documents", {"guideline", "ingredient_spec"}),
        ("/ask", {"english", "ukrainian"}),
        ("/reformulate", {"remove_allergen", "reduce_sugar", "make_vegan"}),
    ],
)
def test_swagger_shows_the_examples(path, names):
    body = app.openapi()["paths"][path]["post"]["requestBody"]["content"]["application/json"]
    assert set(body["examples"]) == names


@pytest.mark.parametrize(
    ("model", "examples"),
    [(DocumentIn, DOCUMENT_EXAMPLES), (AskIn, ASK_EXAMPLES), (ReformulateIn, REFORMULATE_EXAMPLES)],
)
def test_every_example_is_a_valid_request(model, examples):
    for example in examples.values():
        model.model_validate(example["value"])


def test_demo_spec_has_tables_the_ingest_parses():
    content = DOCUMENT_EXAMPLES["ingredient_spec"]["value"]["content"]
    assert parse_nutrients_table(content)["Value"]["kcal"] == 47
    assert parse_allergens_table(content) == {"Value": {"allergens": [], "vegan": True}}


def test_ukrainian_ask_example_is_the_one_from_the_feedback():
    assert ASK_EXAMPLES["ukrainian"]["value"]["question"] == "Чим замінити яйце в бісквіті?"
