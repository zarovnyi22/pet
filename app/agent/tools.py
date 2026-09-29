"""The agent's three tools: their JSON schemas for the LLM and their implementations.

Tools never raise into the agent loop: bad arguments and upstream failures come back as
{"error": "..."} so the model can see what went wrong and adjust.
"""

from typing import Any

import asyncpg
import httpx
from pydantic import BaseModel, Field, ValidationError

from app.embeddings import Embedder
from app.llm.base import ToolSpec
from app.retrieval import search
from app.schemas import NutritionPer100g

NUTRIENTS = tuple(NutritionPer100g.model_fields)  # kcal, protein_g, fat_g, carbs_g, sugar_g

# search-a-licious, not the legacy cgi/search.pl: the latter is heavily rate-limited and
# answers 503 "temporarily unavailable" often enough to break a demo.
OFF_SEARCH_URL = "https://search.openfoodfacts.org/search"
OFF_TIMEOUT_SECONDS = 5.0
OFF_MAX_PRODUCTS = 3
# Open Food Facts asks API clients to identify themselves.
OFF_USER_AGENT = "ReformulationAssistant/0.1 (pet project)"
# Our nutrient name -> Open Food Facts `nutriments` key.
OFF_NUTRIENT_KEYS = {
    "kcal": "energy-kcal_100g",
    "protein_g": "proteins_100g",
    "fat_g": "fat_100g",
    "carbs_g": "carbohydrates_100g",
    "sugar_g": "sugars_100g",
}

NUTRIENTS_SCHEMA = {
    "type": "object",
    "properties": {n: {"type": "number", "minimum": 0} for n in NUTRIENTS},
    "description": "Nutrients per 100 g of this ingredient; omit the ones you do not know.",
}

TOOL_SPECS = [
    ToolSpec(
        name="search_knowledge_base",
        description=(
            "Vector search over internal R&D documents (ingredient specs, trial reports, "
            "guidelines). Returns chunks with doc_id, title, text and similarity score. "
            "Internal specs take priority over Open Food Facts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in English."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_product",
        description=(
            "Search Open Food Facts for a commercial product or ingredient. Returns up to "
            f"{OFF_MAX_PRODUCTS} matches with nutrients per 100 g, allergens and ingredients. "
            "Cite a match as its `source` value."
        ),
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "e.g. 'coconut milk'"}},
            "required": ["name"],
        },
    ),
    ToolSpec(
        name="calc_nutrition",
        description=(
            "Nutrients per 100 g of a whole recipe: mass-weighted average of its ingredients. "
            "This is the raw mix before fermentation or baking (e.g. lactose that cultures "
            "convert to lactic acid is still counted as sugar). "
            "Always use this for nutrition numbers, never compute them yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "grams": {"type": "number", "minimum": 0},
                            "nutrients_per_100g": NUTRIENTS_SCHEMA,
                        },
                        "required": ["name", "grams", "nutrients_per_100g"],
                    },
                }
            },
            "required": ["ingredients"],
        },
    ),
]


# --- calc_nutrition -----------------------------------------------------------------------


class NutritionIngredient(BaseModel):
    name: str
    grams: float = Field(ge=0)
    nutrients_per_100g: dict[str, float] = {}


class NutritionResult(BaseModel):
    total_grams: float
    per_100g: NutritionPer100g
    # nutrient -> ingredients that did not report it (counted as 0, so the value is a floor).
    missing: dict[str, list[str]] = {}


def calc_nutrition(ingredients: list[NutritionIngredient]) -> NutritionResult:
    """Per-100 g nutrients of the mix: sum(grams_i * value_i) / total_grams.

    Models the raw mix, not processing: no fermentation (lactose -> lactic acid), no water
    loss on baking. Hence the spec's yogurt example (78 kcal, 12.1 g sugar) differs from this.

    Each ingredient's value is already per 100 g of that ingredient, so the weighted average
    of per-100 g values is directly the per-100 g value of the mix.
    """
    total = sum(i.grams for i in ingredients)
    if total <= 0:
        raise ValueError("total mass of ingredients is zero")

    per_100g: dict[str, float] = {}
    missing: dict[str, list[str]] = {}
    for nutrient in NUTRIENTS:
        weighted = 0.0
        for ing in ingredients:
            value = ing.nutrients_per_100g.get(nutrient)
            if value is None:
                missing.setdefault(nutrient, []).append(ing.name)
            else:
                weighted += ing.grams * value
        per_100g[nutrient] = round(weighted / total, 2)
    return NutritionResult(
        total_grams=round(total, 2), per_100g=NutritionPer100g(**per_100g), missing=missing
    )


# --- lookup_product -----------------------------------------------------------------------


def _untag(tag: str) -> str:
    # "en:sea-salt" -> "sea-salt"; non-English tags keep their prefix ("fr:gellane").
    return tag.removeprefix("en:")


def _join(value: list[str] | str | None) -> str:
    return ", ".join(value) if isinstance(value, list) else value or ""


def _vegan_status(analysis_tags: list[str]) -> str:
    if "en:vegan" in analysis_tags:
        return "yes"
    if "en:non-vegan" in analysis_tags:
        return "no"
    return "unknown"


def _off_product(p: dict[str, Any]) -> dict[str, Any]:
    nutriments = p.get("nutriments") or {}
    return {
        "source": f"off:{p.get('code', '')}",
        "product_name": p.get("product_name") or "",
        "brands": _join(p.get("brands")),
        "nutrients_per_100g": {
            ours: nutriments[key] for ours, key in OFF_NUTRIENT_KEYS.items() if key in nutriments
        },
        "allergens": [_untag(t) for t in p.get("allergens_tags") or []],
        "ingredients": [_untag(t) for t in p.get("ingredients_tags") or []][:30],
        "vegan": _vegan_status(p.get("ingredients_analysis_tags") or []),
    }


async def lookup_product(http: httpx.AsyncClient, name: str) -> dict[str, Any]:
    params = {
        "q": name,
        "page_size": 10,
        "fields": "code,product_name,brands,nutriments,allergens_tags,"
        "ingredients_tags,ingredients_analysis_tags",
    }
    try:
        resp = await http.get(
            OFF_SEARCH_URL,
            params=params,
            headers={"User-Agent": OFF_USER_AGENT},
            timeout=OFF_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        products = resp.json().get("hits") or []
    except httpx.TimeoutException:
        return {"error": f"Open Food Facts timed out after {OFF_TIMEOUT_SECONDS:.0f}s"}
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"Open Food Facts request failed: {type(exc).__name__}"}

    # Products without per-100 g energy are useless for calc_nutrition; skip them.
    usable = [
        _off_product(p) for p in products if "energy-kcal_100g" in (p.get("nutriments") or {})
    ]
    return {"query": name, "products": usable[:OFF_MAX_PRODUCTS]}


# --- dispatch -----------------------------------------------------------------------------


class Toolbox:
    """Tool implementations bound to one agent run: DB pool, embedder, HTTP client, and the
    Open Food Facts cache, which lives exactly as long as the run."""

    def __init__(self, pool: asyncpg.Pool, embedder: Embedder, http: httpx.AsyncClient) -> None:
        self.pool = pool
        self.embedder = embedder
        self.http = http
        self._off_cache: dict[str, dict[str, Any]] = {}

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "search_knowledge_base":
                return await self._search_knowledge_base(**arguments)
            if name == "lookup_product":
                return await self._lookup_product(**arguments)
            if name == "calc_nutrition":
                return self._calc_nutrition(**arguments)
        except (TypeError, ValueError, ValidationError) as exc:
            # TypeError: unexpected/missing kwargs; ValueError covers pydantic and zero mass.
            return {"error": f"invalid arguments for {name}: {exc}"}
        return {"error": f"unknown tool: {name}"}

    async def _search_knowledge_base(self, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        top_k = max(1, min(int(top_k), 10))
        chunks = await search(self.pool, self.embedder, query, top_k)
        return {"chunks": [c.model_dump() for c in chunks]}

    async def _lookup_product(self, name: str) -> dict[str, Any]:
        key = name.strip().lower()
        if not key:
            raise ValueError("name must not be empty")
        if key in self._off_cache:
            return self._off_cache[key]
        result = await lookup_product(self.http, name.strip())
        if "error" not in result:  # retrying a timeout later in the run may succeed
            self._off_cache[key] = result
        return result

    def _calc_nutrition(self, ingredients: list[dict[str, Any]]) -> dict[str, Any]:
        parsed = [NutritionIngredient.model_validate(i) for i in ingredients]
        return calc_nutrition(parsed).model_dump()
