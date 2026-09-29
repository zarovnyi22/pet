from pathlib import Path

import pytest

from app.ingest import parse_markdown, parse_nutrients_table

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def table_of(doc_id: str) -> dict | None:
    return parse_nutrients_table(parse_markdown((CORPUS / f"{doc_id}.md").read_text()).content)


def test_single_column_table():
    assert table_of("spec-milk-2-5") == {
        "Value": {"kcal": 53, "protein_g": 2.9, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}
    }


def test_notes_after_the_number_are_ignored():
    assert table_of("spec-erythritol")["Value"]["carbs_g"] == 100  # "100 g (polyols)"
    assert table_of("spec-polydextrose")["Value"]["kcal"] == 100  # "100 kcal (declared ...)"


def test_variant_columns_are_kept_apart():
    table = table_of("spec-yogurt-starter")
    assert set(table) == {"Bulk starter (fermented milk 2.5%)", "Freeze-dried plant-based DVS"}
    assert table["Bulk starter (fermented milk 2.5%)"]["kcal"] == 55
    assert table["Freeze-dried plant-based DVS"]["carbs_g"] == 90


def test_documents_without_a_table():
    assert table_of("trial-cookies-sugar-30") is None
    assert parse_nutrients_table("# Title\n\nNo table here.") is None
    assert parse_nutrients_table("## Nutrients per 100 g\n\nto be measured\n") is None


@pytest.mark.parametrize("path", sorted(CORPUS.glob("spec-*.md")), ids=lambda p: p.stem)
def test_every_spec_in_the_corpus_has_all_five_nutrients(path):
    table = parse_nutrients_table(parse_markdown(path.read_text()).content)
    assert table, f"{path.name}: no nutrient table parsed"
    for column, values in table.items():
        assert set(values) == {"kcal", "protein_g", "fat_g", "carbs_g", "sugar_g"}, column
