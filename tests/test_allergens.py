from pathlib import Path

import pytest

from app.allergens import from_off, from_spec

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.mark.parametrize(
    ("doc_id", "allergens", "vegan"),
    [
        ("spec-milk-2-5", ["milk"], False),  # "Contains **milk** ... Not suitable for vegan"
        ("spec-butter", ["milk"], False),
        ("spec-whole-egg", ["eggs"], False),
        ("spec-soy-drink", ["soybeans"], None),  # "**adds another**" is not an allergen
        ("spec-wheat-flour", ["gluten"], None),  # "**gluten (wheat)**"; traces not bold
        ("spec-oat-drink", ["gluten"], None),  # "**cereals containing gluten**"
        ("spec-aquafaba", [], None),  # "**not** one of the 14"; peanut/lupin not bold
        ("spec-yogurt-starter", [], None),  # "**certified dairy-free culture**" is not milk
        ("spec-strawberry-frozen", [], True),  # "Suitable for vegan products"
        ("spec-sucrose", [], None),
    ],
)
def test_spec_allergens_come_from_bold_names_only(doc_id, allergens, vegan):
    facts = from_spec((CORPUS / f"{doc_id}.md").read_text())
    assert facts == {"allergens": allergens, "vegan": vegan}


def test_off_tags_map_to_eu_names_and_drop_the_rest():
    facts = from_off(["milk", "gellan-gum-allergy", "sesame-seeds", "coconut"], "no")
    assert facts == {"allergens": ["milk", "sesame"], "vegan": False}
    assert from_off([], "unknown") == {"allergens": [], "vegan": None}
