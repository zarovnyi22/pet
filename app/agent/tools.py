"""The agent's three tools: their JSON schemas for the LLM and their implementations.

Tools never raise into the agent loop: bad arguments and upstream failures come back as
{"error": "..."} so the model can see what went wrong and adjust.
"""

import json
import re
from typing import Any

import asyncpg
import httpx
from pydantic import BaseModel, Field, ValidationError

from app import allergens
from app.embeddings import Embedder
from app.llm.base import ToolSpec
from app.retrieval import doc_nutrients, search
from app.schemas import NutritionPer100g, Source

NUTRIENTS = tuple(NutritionPer100g.model_fields)  # kcal, protein_g, fat_g, carbs_g, sugar_g

# search-a-licious, not the legacy cgi/search.pl: the latter is heavily rate-limited and
# answers 503 "temporarily unavailable" often enough to break a demo.
OFF_SEARCH_URL = "https://search.openfoodfacts.org/search"
OFF_TIMEOUT_SECONDS = 5.0
OFF_MAX_PRODUCTS = 3
# Context budget per model turn: every tool result is re-sent on each later turn, so it adds up
# (a run used ~31K input tokens before these limits; Groq's free tier allows 8K per minute).
AGENT_TOP_K = 3
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CHUNK_TEXT_LIMIT = 600
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

OFF_MAIN_NUTRIENTS = ("energy-kcal_100g", "proteins_100g", "fat_100g", "carbohydrates_100g")

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
            "guidelines). Returns chunks with doc_id, title, text and similarity score, plus "
            "nutrients_per_100g for every returned spec that has a nutrient table (cite that "
            "doc_id as nutrients_source). Internal specs take priority over Open Food Facts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in English."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": AGENT_TOP_K,
                    "default": AGENT_TOP_K,
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="lookup_product",
        description=(
            "Search Open Food Facts for a commercial product or ingredient. Returns up to "
            f"{OFF_MAX_PRODUCTS} matches with nutrients per 100 g, allergens and vegan status. "
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
            "Always use this for nutrition numbers, never compute them yourself. "
            "Each ingredient's nutrients must name their nutrients_source: a doc_id from "
            "search_knowledge_base or an off:<code> from lookup_product in this run, and the "
            "values must be copied exactly from that source. Values the source does not "
            "contain are ignored and reported under `rejected` and `missing`."
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
                            "nutrients_source": {
                                "type": "string",
                                "description": "doc_id or off:<code> the nutrients come from",
                            },
                        },
                        "required": ["name", "grams", "nutrients_per_100g", "nutrients_source"],
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
    nutrients_source: str | None = None


class RejectedNutrients(BaseModel):
    ingredient: str
    nutrients_source: str | None
    nutrients: list[str]  # the dropped ones; the ingredient's other nutrients still count
    reason: str


class NutritionResult(BaseModel):
    total_grams: float
    per_100g: NutritionPer100g
    # nutrient -> ingredients that did not report it (counted as 0, so the value is a floor).
    missing: dict[str, list[str]] = {}
    # Nutrients dropped because the cited source does not back them up.
    rejected: list[RejectedNutrients] = []


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
        # OFF stores floats like 3.4000000953674; rounding saves tokens and loses nothing.
        "nutrients_per_100g": {
            ours: round(nutriments[key], 2)
            for ours, key in OFF_NUTRIENT_KEYS.items()
            if isinstance(nutriments.get(key), int | float)
        },
        "allergens": [_untag(t) for t in p.get("allergens_tags") or []],
        "vegan": _vegan_status(p.get("ingredients_analysis_tags") or []),
    }


async def lookup_product(http: httpx.AsyncClient, name: str) -> dict[str, Any]:
    params = {
        "q": name,
        "page_size": 10,
        "fields": "code,product_name,nutriments,allergens_tags,ingredients_analysis_tags",
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

    usable, skipped = [], {}
    for p in products:
        reason = _incomplete(p.get("nutriments") or {})
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        elif len(usable) < OFF_MAX_PRODUCTS:
            usable.append(_off_product(p))
    result: dict[str, Any] = {"query": name, "products": usable}
    if skipped:
        result["skipped"] = skipped  # so the trace shows why a search came back thin
    return result


def _incomplete(nutriments: dict[str, Any]) -> str | None:
    """Why a product's nutrients are useless for calc_nutrition, or None. All-zero main
    nutrients are missing data (a honey at 0 kcal); one zero is real (erythritol: 0 kcal,
    100 g carbohydrates)."""
    if "energy-kcal_100g" not in nutriments:
        return "no kcal per 100 g"
    if not any(nutriments.get(key) for key in OFF_MAIN_NUTRIENTS):
        return "kcal, protein, fat and carbs all zero or missing"
    return None


# --- dispatch -----------------------------------------------------------------------------


class Toolbox:
    """Tool implementations bound to one agent run: DB pool, embedder, HTTP client, and the
    per-run state: the Open Food Facts cache and every source the run has actually seen."""

    def __init__(self, pool: asyncpg.Pool, embedder: Embedder, http: httpx.AsyncClient) -> None:
        self.pool = pool
        self.embedder = embedder
        self.http = http
        self._off_cache: dict[str, dict[str, Any]] = {}
        # What the model was actually shown in this run, per source: the only values
        # calc_nutrition accepts. off:<code> -> nutrients; doc_id -> its parsed nutrient table
        # (specs) and, for documents without one, the numbers in its chunk texts.
        self._product_nutrients: dict[str, dict[str, float]] = {}
        self._product_facts: dict[str, dict[str, Any]] = {}  # off:<code> -> allergens, vegan
        self._product_names: dict[str, str] = {}  # off:<code> -> product_name
        self._doc_tables: dict[str, dict[str, dict[str, float]]] = {}
        self._doc_numbers: dict[str, set[float]] = {}

    @property
    def seen_sources(self) -> set[str]:
        return set(self._product_nutrients) | set(self._doc_tables) | set(self._doc_numbers)

    def remember_product(
        self,
        source: str,
        nutrients: dict[str, float],
        allergen_tags: list[str] | None = None,
        vegan: str = "unknown",
        name: str = "",
    ) -> None:
        self._product_nutrients[source] = nutrients
        self._product_facts[source] = allergens.from_off(allergen_tags or [], vegan)
        if name:
            self._product_names[source] = name

    def product_name(self, source: str) -> str | None:
        return self._product_names.get(source)

    def columns_of(self, doc_id: str) -> list[str]:
        return list(self._doc_tables.get(doc_id, {}))

    def product_facts(self, source: str) -> dict[str, Any] | None:
        return self._product_facts.get(source)

    def remember_table(self, doc_id: str, table: dict[str, dict[str, float]]) -> None:
        self._doc_tables[doc_id] = table

    def nutrients_of(self, source: str, column: str | None = None) -> dict[str, float]:
        """Nutrients of a source this run has seen, for code that builds a recipe itself."""
        if source in self._product_nutrients:
            return self._product_nutrients[source]
        table = self._doc_tables.get(source)
        if table is None:
            raise ValueError(f"{source!r} has no nutrient data in this run")
        if column is None and len(table) == 1:
            return next(iter(table.values()))
        if column not in table:
            raise ValueError(f"{source!r} has several columns, pick one of {sorted(table)}")
        return table[column]

    def remember_text(self, doc_id: str, text: str) -> None:
        numbers = {float(n) for n in NUMBER_RE.findall(text)}
        self._doc_numbers.setdefault(doc_id, set()).update(numbers)

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

    async def _search_knowledge_base(self, query: str, top_k: int = AGENT_TOP_K) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        top_k = max(1, min(int(top_k), AGENT_TOP_K))
        chunks = await search(self.pool, self.embedder, query, top_k)
        compacted = [_compact_chunk(c) for c in chunks]
        for chunk in compacted:  # the truncated text: only what the model actually saw
            self.remember_text(chunk["doc_id"], chunk["chunk_text"])
        # A spec's nutrient table rarely lands in the top chunks, so it comes structured with
        # any chunk of that spec: the model never has to find (or guess) the numbers.
        tables = await doc_nutrients(self.pool, sorted({c.doc_id for c in chunks}))
        for doc_id, table in tables.items():
            self.remember_table(doc_id, table)
        result: dict[str, Any] = {"chunks": compacted}
        if tables:
            result["nutrients_per_100g"] = {d: _flatten(t) for d, t in tables.items()}
        return result

    async def _lookup_product(self, name: str) -> dict[str, Any]:
        key = name.strip().lower()
        if not key:
            raise ValueError("name must not be empty")
        if key in self._off_cache:
            return self._off_cache[key]
        result = await lookup_product(self.http, name.strip())
        if "error" not in result:  # retrying a timeout later in the run may succeed
            self._off_cache[key] = result
            for product in result["products"]:
                self.remember_product(
                    product["source"],
                    product["nutrients_per_100g"],
                    product.get("allergens"),
                    product.get("vegan", "unknown"),
                    product.get("product_name", ""),
                )
        return result

    def _calc_nutrition(self, ingredients: list[dict[str, Any]] | str) -> dict[str, Any]:
        parsed = [NutritionIngredient.model_validate(i) for i in _unstringify(ingredients)]
        rejected = []
        for ing in parsed:
            dropped, reason = self._unbacked(ing)
            if dropped:
                rejected.append(
                    RejectedNutrients(
                        ingredient=ing.name,
                        nutrients_source=ing.nutrients_source,
                        nutrients=dropped,
                        reason=reason,
                    )
                )
                # Counted as missing, never as supplied.
                for nutrient in dropped:
                    del ing.nutrients_per_100g[nutrient]
        result = calc_nutrition(parsed)
        result.rejected = rejected
        return result.model_dump()

    def _unbacked(self, ing: NutritionIngredient) -> tuple[list[str], str]:
        """Nutrients the cited source does not back up, and why."""
        supplied = list(ing.nutrients_per_100g)
        source = ing.nutrients_source
        if not supplied:
            return [], ""
        if not source:
            return supplied, (
                "no nutrients_source given; cite the doc_id or off:<code> the values come from"
            )
        if source in self._product_nutrients:
            # Open Food Facts: must equal what lookup_product returned for this product.
            known = self._product_nutrients[source]
            dropped = [
                n
                for n in supplied
                if n not in known or not _close(ing.nutrients_per_100g[n], known[n])
            ]
            return dropped, (
                f"values differ from {source}: {known}. Copy these exact values, or cite the "
                "product the values came from"
            )
        if source in self._doc_tables:
            # Spec with a parsed table: must match that nutrient in one of its columns.
            columns = self._doc_tables[source].values()
            dropped = [
                n
                for n in supplied
                if not any(n in c and _close(ing.nutrients_per_100g[n], c[n]) for c in columns)
            ]
            return dropped, (
                f"values differ from the nutrient table of {source}: "
                f"{_flatten(self._doc_tables[source])}. Copy these exact values"
            )
        if source in self._doc_numbers:
            # Knowledge base: the number must appear in the chunk text the model was shown.
            numbers = self._doc_numbers[source]
            dropped = [
                n
                for n in supplied
                if not any(_close(ing.nutrients_per_100g[n], x) for x in numbers)
            ]
            return dropped, (
                f"values not found in the text of {source} shown in this run; cite a source "
                "that states them, or omit them"
            )
        return supplied, (
            f"{source!r} was not returned by search_knowledge_base or lookup_product in this "
            "run; search for it first, or cite a source whose values you were shown"
        )


def _flatten(table: dict[str, dict[str, float]]) -> dict[str, Any]:
    # The common single-column table ("Value") is shown flat; variant tables keep their columns.
    return next(iter(table.values())) if len(table) == 1 else table


def _close(a: float, b: float) -> bool:
    # 1% or 0.01 absolute: tolerates rounding (53.975 -> 54) but not a different product.
    return abs(a - b) <= max(0.01, 0.01 * abs(b))


def _compact_chunk(chunk: Source) -> dict[str, Any]:
    text = chunk.chunk_text
    if len(text) > CHUNK_TEXT_LIMIT:
        text = text[:CHUNK_TEXT_LIMIT] + "…"
    return chunk.model_dump() | {"chunk_text": text}


def _unstringify(value: Any) -> list[Any]:
    """Models sometimes send JSON as strings: the whole list, one object per string, or several
    comma-separated objects in one string (seen from Gemini). Parse instead of failing the call,
    which would cost the agent one of its 6 iterations."""
    if isinstance(value, str):
        value = _loads_objects(value)
    if not isinstance(value, list):
        value = [value]
    items: list[Any] = []
    for item in value:
        parsed = _loads_objects(item) if isinstance(item, str) else item
        items.extend(parsed if isinstance(parsed, list) else [parsed])
    return items


def _loads_objects(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # '{...},{...}' is not JSON by itself, but is once wrapped in brackets.
        try:
            return json.loads(f"[{text}]")
        except json.JSONDecodeError:
            raise ValueError(
                "ingredients must be JSON objects, got an unparseable string"
            ) from None
