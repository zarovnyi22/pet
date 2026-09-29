import json

import pytest

from app.agent.tools import NutritionIngredient, Toolbox, calc_nutrition

# Per-100 g reference values for the yogurt from docs/SPEC.md ("Що будуємо").
# Milk 2.5%: typical label values.
MILK_25 = {"kcal": 53, "protein_g": 2.9, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}
SUGAR = {"kcal": 400, "protein_g": 0, "fat_g": 0, "carbs_g": 100, "sugar_g": 100}  # spec-sucrose
# USDA FoodData Central 09090, strawberries, frozen, unsweetened.
STRAWBERRY = {"kcal": 35, "protein_g": 0.43, "fat_g": 0.11, "carbs_g": 9.13, "sugar_g": 4.56}


def ing(name: str, grams: float, nutrients: dict[str, float]) -> NutritionIngredient:
    return NutritionIngredient(name=name, grams=grams, nutrients_per_100g=nutrients)


def yogurt() -> list[NutritionIngredient]:
    return [
        ing("молоко 2.5%", 800, MILK_25),
        ing("цукор", 90, SUGAR),
        ing("полуниця заморожена", 100, STRAWBERRY),
        ing("закваска", 10, {}),  # no nutrient data: 1% of the mass
    ]


def test_spec_yogurt():
    # kcal = (800*53 + 90*400 + 100*35) / 1000 = 81.9
    # sugar = (800*4.7 + 90*100 + 100*4.56) / 1000 = 13.216
    result = calc_nutrition(yogurt())
    assert result.total_grams == 1000
    assert result.per_100g.model_dump() == {
        "kcal": 81.9,
        "protein_g": 2.36,
        "fat_g": 2.01,
        "carbs_g": 13.67,
        "sugar_g": 13.22,
    }
    assert result.missing == {n: ["закваска"] for n in MILK_25}


def test_single_ingredient_keeps_its_values():
    result = calc_nutrition([ing("молоко 2.5%", 250, MILK_25)])
    assert result.per_100g.model_dump() == MILK_25
    assert result.missing == {}


def test_weighted_average_by_mass():
    # 900 g milk + 100 g sugar: sugar = (900*4.7 + 100*100) / 1000 = 14.23 g/100 g
    result = calc_nutrition([ing("молоко 2.5%", 900, MILK_25), ing("цукор", 100, SUGAR)])
    assert result.per_100g.sugar_g == pytest.approx(14.23)
    assert result.per_100g.kcal == pytest.approx(87.7)
    assert result.per_100g.protein_g == pytest.approx(2.61)


def test_zero_gram_ingredient_does_not_dilute():
    result = calc_nutrition([ing("молоко 2.5%", 500, MILK_25), ing("цукор", 0, SUGAR)])
    assert result.per_100g.model_dump() == MILK_25


def test_zero_total_mass_raises():
    with pytest.raises(ValueError, match="zero"):
        calc_nutrition([ing("молоко 2.5%", 0, MILK_25), ing("цукор", 0, SUGAR)])
    with pytest.raises(ValueError):
        calc_nutrition([])


def test_missing_nutrients_count_as_zero_and_are_reported():
    # Strawberry without sugar data still adds mass, contributes 0 sugar, and is flagged.
    no_sugar = {k: v for k, v in STRAWBERRY.items() if k != "sugar_g"}
    result = calc_nutrition([ing("молоко 2.5%", 50, MILK_25), ing("полуниця", 50, no_sugar)])
    assert result.per_100g.sugar_g == pytest.approx(2.35)
    assert result.missing == {"sugar_g": ["полуниця"]}


def test_no_nutrients_at_all():
    result = calc_nutrition([ing("закваска", 10, {})])
    assert result.per_100g.kcal == 0
    assert set(result.missing) == set(MILK_25)


async def test_tool_returns_errors_instead_of_raising():
    tools = Toolbox(pool=None, embedder=None, http=None)
    zero = await tools.execute(
        "calc_nutrition",
        {"ingredients": [{"name": "x", "grams": 0, "nutrients_per_100g": MILK_25}]},
    )
    assert "zero" in zero["error"]
    negative = await tools.execute(
        "calc_nutrition", {"ingredients": [{"name": "x", "grams": -5, "nutrients_per_100g": {}}]}
    )
    assert "error" in negative
    assert "error" in await tools.execute("calc_nutrition", {"items": []})
    assert "unknown tool" in (await tools.execute("nope", {}))["error"]


async def test_tool_accepts_ingredients_sent_as_json_strings():
    # Shapes seen from LLMs: one string per object, several objects in one string (Gemini),
    # and the whole list as a string. All must give the same result as proper objects.
    items = [
        {
            "name": n.name,
            "grams": n.grams,
            "nutrients_per_100g": n.nutrients_per_100g,
            "nutrients_source": "spec-whole-milk",
        }
        for n in yogurt()
    ]
    tools = seeded_toolbox()
    expected = await tools.execute("calc_nutrition", {"ingredients": items})
    assert expected["per_100g"]["kcal"] == 81.9  # nutrients accepted, not dropped

    one_per_string = [json.dumps(i, ensure_ascii=False) for i in items]
    all_in_one_string = [",\n".join(one_per_string)]
    whole_list = json.dumps(items, ensure_ascii=False)
    for ingredients in (one_per_string, all_in_one_string, whole_list):
        assert await tools.execute("calc_nutrition", {"ingredients": ingredients}) == expected

    broken = await tools.execute("calc_nutrition", {"ingredients": ['{"name": "молоко"']})
    assert "unparseable string" in broken["error"]


# --- nutrient provenance ------------------------------------------------------------------


def doc_text(*nutrients: dict) -> str:
    """A knowledge-base chunk that states these values, e.g. "kcal 53 | protein_g 2.9"."""
    return " | ".join(f"{k} {v}" for d in nutrients for k, v in d.items())


def seeded_toolbox() -> Toolbox:
    """As if search_knowledge_base had shown a doc with all the yogurt values."""
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_text("spec-whole-milk", doc_text(MILK_25, SUGAR, STRAWBERRY))
    return tools


def item(name: str, grams: float, nutrients: dict, source: str | None) -> dict:
    return {"name": name, "grams": grams, "nutrients_per_100g": dict(nutrients)} | (
        {"nutrients_source": source} if source else {}
    )


async def calc(tools: Toolbox, *items: dict) -> dict:
    return await tools.execute("calc_nutrition", {"ingredients": list(items)})


async def test_values_backed_by_their_sources_are_used():
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_text("spec-milk-2-5", "| Energy | 53 kcal |\n| Protein | 2.9 g | Fat 2.5 g 4.7")
    tools.remember_product("off:3017620422003", SUGAR)
    result = await calc(
        tools,
        item("молоко 2.5%", 800, MILK_25, "spec-milk-2-5"),
        item("цукор", 90, SUGAR, "off:3017620422003"),
    )
    assert result["rejected"] == [] and result["missing"] == {}
    assert result["per_100g"]["sugar_g"] == pytest.approx(14.34)


async def test_unseen_or_missing_source_drops_all_nutrients():
    tools = seeded_toolbox()
    result = await calc(
        tools,
        item("молоко 2.5%", 800, MILK_25, "spec-whole-milk"),
        item("цукор", 90, SUGAR, "spec-sucrose"),  # real doc_id, but not seen this run
        item("полуниця заморожена", 100, STRAWBERRY, None),  # from memory
        item("закваска", 10, {}, None),  # honestly unknown: missing, not rejected
    )
    # Only the milk counts: sugar = 800 * 4.7 / 1000.
    assert result["per_100g"]["sugar_g"] == pytest.approx(3.76)
    assert result["missing"]["sugar_g"] == ["цукор", "полуниця заморожена", "закваска"]
    sugar, strawberry = result["rejected"]
    assert (sugar["ingredient"], sugar["nutrients"]) == ("цукор", list(SUGAR))
    assert "not returned by search_knowledge_base or lookup_product" in sugar["reason"]
    assert strawberry["nutrients_source"] is None
    assert strawberry["reason"].startswith("no nutrients_source given; cite the doc_id")


async def test_values_that_differ_from_the_cited_product_are_dropped():
    # Seen live: brown-sugar values (375 kcal, 100 g sugar) cited as a 30 kcal "Sugar" product.
    tools = Toolbox(pool=None, embedder=None, http=None)
    cited = {"kcal": 30, "protein_g": 0, "fat_g": 0, "carbs_g": 8, "sugar_g": 0}
    tools.remember_product("off:0070090304104", cited)
    brown = {"kcal": 375, "protein_g": 0, "fat_g": 0, "carbs_g": 100, "sugar_g": 100}
    result = await calc(tools, item("цукор", 90, brown, "off:0070090304104"))

    [rejected] = result["rejected"]
    assert rejected["nutrients"] == ["kcal", "carbs_g", "sugar_g"]  # the zeros do match
    assert "values differ from off:0070090304104" in rejected["reason"]
    assert result["per_100g"]["sugar_g"] == 0 and result["missing"]["sugar_g"] == ["цукор"]


async def test_number_absent_from_the_doc_text_is_dropped_alone():
    # Seen live: milk cited as spec-milk-2-5 (53 kcal) but sent as 52 kcal from memory.
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_text("spec-milk-2-5", doc_text(MILK_25))
    result = await calc(tools, item("молоко 2.5%", 100, MILK_25 | {"kcal": 52}, "spec-milk-2-5"))

    [rejected] = result["rejected"]
    assert rejected["nutrients"] == ["kcal"]
    assert "not found in the text of spec-milk-2-5" in rejected["reason"]
    assert result["per_100g"]["protein_g"] == 2.9  # the backed values still count
    assert result["missing"] == {"kcal": ["молоко 2.5%"]}


async def test_rounding_is_tolerated():
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_product("off:1", MILK_25 | {"kcal": 53.975})
    result = await calc(tools, item("молоко 2.5%", 100, MILK_25 | {"kcal": 54}, "off:1"))
    assert result["rejected"] == []


async def test_search_and_lookup_record_their_sources(monkeypatch):
    from app.agent import tools as tools_mod
    from app.schemas import Source

    asked_top_k = []

    async def fake_search(pool, embedder, query, top_k):
        asked_top_k.append(top_k)
        # 400 kcal is stated only past the 600-character cut, so the model never saw it.
        text = "Energy 380 kcal. " + "y" * 700 + " Energy 400 kcal."
        return [Source(doc_id="spec-sucrose", title="Sucrose", chunk_text=text, score=0.7)]

    async def fake_lookup(http, name):
        product = {"source": "off:123", "product_name": name, "nutrients_per_100g": SUGAR}
        return {"query": name, "products": [product]}

    async def no_tables(pool, doc_ids):
        return {}

    monkeypatch.setattr(tools_mod, "search", fake_search)
    monkeypatch.setattr(tools_mod, "doc_nutrients", no_tables)
    monkeypatch.setattr(tools_mod, "lookup_product", fake_lookup)
    tools = Toolbox(pool=None, embedder=None, http=None)

    found = await tools.execute("search_knowledge_base", {"query": "sugar", "top_k": 10})
    await tools.execute("lookup_product", {"name": "sugar"})

    assert tools.seen_sources == {"spec-sucrose", "off:123"}
    assert asked_top_k == [tools_mod.AGENT_TOP_K]  # the model asked for 10
    # Context budget: long chunk texts are cut for the model.
    assert len(found["chunks"][0]["chunk_text"]) == tools_mod.CHUNK_TEXT_LIMIT + 1
    assert (await calc(tools, item("цукор", 90, SUGAR, "off:123")))["rejected"] == []
    only_seen = await calc(tools, item("цукор", 90, {"kcal": 380}, "spec-sucrose"))
    assert only_seen["rejected"] == []
    unseen = await calc(tools, item("цукор", 90, {"kcal": 400}, "spec-sucrose"))
    assert unseen["rejected"][0]["nutrients"] == ["kcal"]


STARTER_TABLE = {
    "Bulk starter": {"kcal": 55, "protein_g": 3.0, "fat_g": 2.5, "carbs_g": 4.0, "sugar_g": 3.6},
    "Plant DVS": {"kcal": 380, "protein_g": 2.0, "fat_g": 0.5, "carbs_g": 90, "sugar_g": 5.0},
}


async def test_search_attaches_spec_nutrient_tables(monkeypatch):
    from app.agent import tools as tools_mod
    from app.schemas import Source

    async def fake_search(pool, embedder, query, top_k):
        # Neither chunk contains the table: it is further down each document.
        return [
            Source(doc_id="spec-sucrose", title="Sucrose", chunk_text="Function...", score=0.7),
            Source(doc_id="spec-yogurt-starter", title="Starter", chunk_text="Forms", score=0.6),
            Source(doc_id="trial-x", title="Trial", chunk_text="Results", score=0.5),
        ]

    async def fake_tables(pool, doc_ids):
        assert doc_ids == ["spec-sucrose", "spec-yogurt-starter", "trial-x"]
        return {"spec-sucrose": {"Value": SUGAR}, "spec-yogurt-starter": STARTER_TABLE}

    monkeypatch.setattr(tools_mod, "search", fake_search)
    monkeypatch.setattr(tools_mod, "doc_nutrients", fake_tables)
    tools = Toolbox(pool=None, embedder=None, http=None)
    found = await tools.execute("search_knowledge_base", {"query": "sugar starter"})

    # Single-column tables are shown flat, variant tables keep their columns.
    assert found["nutrients_per_100g"] == {
        "spec-sucrose": SUGAR,
        "spec-yogurt-starter": STARTER_TABLE,
    }
    result = await calc(
        tools,
        item("цукор", 90, SUGAR, "spec-sucrose"),  # the table was never in a chunk text
        item("закваска", 10, STARTER_TABLE["Plant DVS"], "spec-yogurt-starter"),
    )
    assert result["rejected"] == []


async def test_values_not_in_the_spec_table_are_dropped_with_the_table():
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_table("spec-sucrose", {"Value": SUGAR})
    result = await calc(tools, item("цукор", 90, SUGAR | {"kcal": 387}, "spec-sucrose"))

    [rejected] = result["rejected"]
    assert rejected["nutrients"] == ["kcal"]
    # The fix is in the message: the exact values to copy.
    assert "nutrient table of spec-sucrose" in rejected["reason"]
    assert "'kcal': 400" in rejected["reason"] and "Copy these exact values" in rejected["reason"]


async def test_a_value_may_come_from_any_column_of_a_variant_table():
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_table("spec-yogurt-starter", STARTER_TABLE)
    mixed = STARTER_TABLE["Bulk starter"] | {"kcal": 380}
    assert (await calc(tools, item("закваска", 10, mixed, "spec-yogurt-starter")))["rejected"] == []
