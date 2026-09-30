from pathlib import Path

import pytest

from app.allergens import from_off, parse_allergens_table
from app.ingest import parse_nutrients_table

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
SPECS = sorted(CORPUS.glob("spec-*.md"))
BULK, DVS = "Bulk starter (fermented milk 2.5%)", "Freeze-dried plant-based DVS"


def facts_of(doc_id: str) -> dict | None:
    return parse_allergens_table((CORPUS / f"{doc_id}.md").read_text())


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_every_spec_states_allergens_and_vegan_for_each_nutrient_column(path):
    content = path.read_text()
    facts = parse_allergens_table(content)
    assert facts is not None
    assert list(facts) == list(parse_nutrients_table(content))
    assert all(column["vegan"] is not None for column in facts.values())


def test_starter_columns_differ_bulk_is_milk():
    assert facts_of("spec-yogurt-starter") == {
        BULK: {"allergens": ["milk"], "vegan": False},
        DVS: {"allergens": [], "vegan": True},
    }


@pytest.mark.parametrize(
    ("doc_id", "allergens", "vegan"),
    [
        ("spec-milk-2-5", ["milk"], False),
        ("spec-whole-milk", ["milk"], False),
        ("spec-butter", ["milk"], False),
        ("spec-whole-egg", ["eggs"], False),
        ("spec-soy-drink", ["soybeans"], True),
        ("spec-oat-drink", ["gluten"], True),
        ("spec-wheat-flour", ["gluten"], True),  # "may contain" traces are not in the table
        ("spec-gluten-free-blend", [], True),
        ("spec-aquafaba", [], True),
        ("spec-strawberry-frozen", [], True),
        ("spec-sucrose", [], True),
        ("spec-erythritol", [], True),  # maize glucose, no wheat: no new allergen
        ("spec-polydextrose", [], True),
        ("spec-stevia", [], True),
    ],
)
def test_single_column_specs(doc_id, allergens, vegan):
    assert facts_of(doc_id) == {"Value": {"allergens": allergens, "vegan": vegan}}


def table(allergens: str, vegan: str) -> str:
    rows = f"| Allergens | {allergens} |\n| Vegan | {vegan} |\n"
    return "## Allergens\n\n| | Value |\n|---|---|\n" + rows


def test_aliases_become_eu_names_and_unknown_vegan_is_none():
    assert parse_allergens_table(table("Soy, egg, wheat", "unknown")) == {
        "Value": {"allergens": ["eggs", "gluten", "soybeans"], "vegan": None}
    }


@pytest.mark.parametrize(
    ("allergens", "vegan"),
    [("chickpea", "yes"), ("", "yes"), ("milk", "maybe"), ("none", "")],
)
def test_bad_cells_fail_instead_of_reading_as_safe(allergens, vegan):
    with pytest.raises(ValueError):
        parse_allergens_table(table(allergens, vegan))


def test_table_without_vegan_row_fails():
    with pytest.raises(ValueError):
        parse_allergens_table("## Allergens\n\n| | Value |\n|---|---|\n| Allergens | milk |\n")


def test_documents_without_a_table():
    assert parse_allergens_table("# Trial\n\nNo allergens section.") is None
    assert parse_allergens_table("## Allergens\n\nContains **milk**.") is None


def test_off_tags_map_to_eu_names_and_drop_the_rest():
    facts = from_off(["milk", "gellan-gum-allergy", "sesame-seeds", "coconut"], "no")
    assert facts == {"allergens": ["milk", "sesame"], "vegan": False}
    assert from_off([], "unknown") == {"allergens": [], "vegan": None}
