"""EU allergens and vegan status from source data, so the agent's allergen claims are checked
against specs and Open Food Facts instead of being taken from the model's own list."""

import re
from typing import Any

# Names the allergens table may use -> the canonical EU name (schemas.EUAllergen).
_CANONICAL = {
    "soy": "soybeans",
    "soya": "soybeans",
    "soybean": "soybeans",
    "egg": "eggs",
    "wheat": "gluten",
    "peanut": "peanuts",
    "crustacean": "crustaceans",
    "mollusc": "molluscs",
}
# Open Food Facts allergen tags (without "en:") that differ from the EU names.
_OFF_TAGS = {"sesame-seeds": "sesame", "sulphur-dioxide-and-sulphites": "sulphites"}
EU_ALLERGENS = frozenset(
    "gluten crustaceans eggs fish peanuts soybeans milk nuts celery mustard sesame sulphites "
    "lupin molluscs".split()
)
_VEGAN = {"yes": True, "no": False, "unknown": None}


def parse_allergens_table(content: str) -> dict[str, dict[str, Any]] | None:
    """The table at the top of a spec's "## Allergens" section, with the same columns as its
    nutrient table: {column: {"allergens": [canonical EU names], "vegan": bool | None}}.

    Rows "Allergens" (EU names separated by commas, or "none") and "Vegan" (yes / no /
    unknown). Any other value raises ValueError: a spec must fail at ingest, not be read as
    allergen-free. None if the section has no table.
    """
    section = re.search(r"^## Allergens\b[^\n]*$(.*?)(?=^## |\Z)", content, re.M | re.S)
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
    cells = {label.lower(): values for label, *values in rows[1:]}
    if set(cells) != {"allergens", "vegan"}:
        raise ValueError(f"allergens table needs rows Allergens and Vegan, got {sorted(cells)}")
    table = {}
    for i, column in enumerate(columns):
        allergens_cell = cells["allergens"][i].lower() if i < len(cells["allergens"]) else ""
        vegan_cell = cells["vegan"][i].lower() if i < len(cells["vegan"]) else ""
        if vegan_cell not in _VEGAN:
            raise ValueError(f"{column!r}: Vegan must be yes / no / unknown, got {vegan_cell!r}")
        table[column] = {
            "allergens": _parse_allergens(column, allergens_cell),
            "vegan": _VEGAN[vegan_cell],
        }
    return table


def _parse_allergens(column: str, cell: str) -> list[str]:
    if cell == "none":
        return []
    names = {_canonical(name.strip()) for name in cell.split(",") if name.strip()}
    unknown = names - EU_ALLERGENS
    if not names or unknown:
        raise ValueError(f"{column!r}: Allergens must be EU allergen names or 'none', got {cell!r}")
    return sorted(names)


def from_off(tags: list[str], vegan: str) -> dict[str, Any]:
    """Allergens and vegan status of an Open Food Facts product (tags already without "en:")."""
    found = {_OFF_TAGS.get(t, t) for t in tags} & EU_ALLERGENS
    return {"allergens": sorted(found), "vegan": {"yes": True, "no": False}.get(vegan)}


def _canonical(word: str) -> str:
    word = word.lower()
    return _CANONICAL.get(word, word)
