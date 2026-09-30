"""lookup_product against a mocked Open Food Facts search: which products are shown."""

import httpx

from app.agent.tools import lookup_product


def hit(code: str, **nutriments) -> dict:
    return {"code": code, "product_name": code, "nutriments": nutriments}


HITS = [
    hit("honey-empty", **{"energy-kcal_100g": 0, "proteins_100g": 0, "carbohydrates_100g": 0}),
    hit("no-energy", **{"sugars_100g": 80}),
    hit("erythritol", **{"energy-kcal_100g": 0, "carbohydrates_100g": 100, "fat_100g": 0}),
    hit("honey", **{"energy-kcal_100g": 304, "carbohydrates_100g": 82.4, "sugars_100g": 82.1}),
]


async def search(hits: list[dict]) -> dict:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"hits": hits}))
    async with httpx.AsyncClient(transport=transport) as http:
        return await lookup_product(http, "honey")


async def test_products_with_no_usable_nutrients_are_skipped_and_counted():
    result = await search(HITS)

    # Erythritol's 0 kcal is real (100 g carbohydrates); the all-zero honey is missing data.
    assert [p["source"] for p in result["products"]] == ["off:erythritol", "off:honey"]
    assert result["skipped"] == {
        "kcal, protein, fat and carbs all zero or missing": 1,
        "no kcal per 100 g": 1,
    }


async def test_nothing_skipped_means_no_skipped_field():
    result = await search(HITS[2:])
    assert "skipped" not in result and len(result["products"]) == 2
