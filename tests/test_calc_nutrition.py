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
