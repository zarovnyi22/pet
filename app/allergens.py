"""EU allergens and vegan status from source data, so the agent's allergen claims are checked
against specs and Open Food Facts instead of being taken from the model's own list."""

import re
from typing import Any

# Words that name an EU Annex II allergen -> its canonical name (schemas.EUAllergen).
# Deliberately no "dairy": "**certified dairy-free culture**" must not read as milk.
_SPEC_WORDS = re.compile(
    r"\b(milk|soya?|soybeans?|eggs?|gluten|wheat|peanuts?|nuts|fish|crustaceans?|molluscs?"
    r"|celery|mustard|sesame|sulphites|lupin)\b",
    re.IGNORECASE,
)
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


def from_spec(content: str) -> dict[str, Any]:
    """Allergens and vegan status stated in a spec's "## Allergens" section.

    Allergens are the bold ones ("Contains **milk**", "**cereals containing gluten**");
    unbolded mentions are cross-reactions or "may contain" traces, not ingredients.
    """
    section = re.search(r"^## Allergens\b[^\n]*$(.*?)(?=^## |\Z)", content, re.M | re.S)
    text = section.group(1) if section else ""
    found = {
        _canonical(word)
        for bold in re.findall(r"\*\*(.+?)\*\*", text)
        for word in _SPEC_WORDS.findall(bold)
    }
    lowered = text.lower()
    vegan = (
        False
        if "not suitable for vegan" in lowered
        else True
        if "suitable for vegan" in lowered
        else None
    )
    return {"allergens": sorted(found), "vegan": vegan}


def from_off(tags: list[str], vegan: str) -> dict[str, Any]:
    """Allergens and vegan status of an Open Food Facts product (tags already without "en:")."""
    found = {_OFF_TAGS.get(t, t) for t in tags} & EU_ALLERGENS
    return {"allergens": sorted(found), "vegan": {"yes": True, "no": False}.get(vegan)}


def _canonical(word: str) -> str:
    word = word.lower()
    return _CANONICAL.get(word, word)
