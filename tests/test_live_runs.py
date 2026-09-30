"""The checks of scripts/live_runs.py on stored answers: no service, no LLM."""

import pytest

from scripts.live_runs import check, starter_problems, usage

DVS = "Freeze-dried plant-based DVS"


def answer(starter_subs: list[tuple[str, float, str | None]], allergens_after=("soybeans",)):
    """A remove_allergen answer: milk -> soy, and the given starter subs (name, g, column)."""
    subs = [{"original": "молоко 2.5%", "replacement": "соєвий напій", "grams": 800}]
    subs += [{"original": "закваска", "replacement": r, "grams": g} for r, g, _ in starter_subs]
    chosen = [
        s
        | {
            "nutrients_source": "spec-yogurt-starter" if column else "spec-soy-drink",
            "nutrients_column": column,
        }
        for s, (_, _, column) in zip(subs[1:], starter_subs, strict=True)
    ]
    usage_ = {"provider": "gemini", "total_tokens": 3000}
    return {
        "substitutions": subs,
        "allergens_after": list(allergens_after),
        "warnings": [],
        "trace": [
            {"type": "llm_call", "tool": "plan", "result": {}, "usage": usage_},
            {
                "type": "llm_call",
                "tool": "choose",
                "result": {"substitutions": chosen},
                "usage": usage_,
            },
        ],
    }


GOOD = [("рослинна закваска", 0.2, DVS), ("соєвий напій", 9.8, None)]


def test_dvs_plus_base_passes():
    assert check("remove_allergen", 200, answer(GOOD)) == []
    assert check("make_vegan", 200, answer(GOOD)) == []


@pytest.mark.parametrize(
    ("subs", "problem"),
    [
        ([], "starter not replaced"),
        ([("рослинна закваска", 10, DVS)], "DVS 10 g > 0.5 g"),
        ([("рослинна закваска", 0.2, DVS)], "starter subs total 0.2 g, not 10"),
        ([("соєвий напій", 10, None)], "no DVS culture among the starter subs"),
    ],
)
def test_starter_failures_are_named(subs, problem):
    assert problem in starter_problems(answer(subs))


@pytest.mark.parametrize("goal", ["remove_allergen", "make_vegan"])
def test_milk_left_in_allergens_fails_both_milk_goals(goal):
    body = answer(GOOD, allergens_after=["milk"])
    assert check(goal, 200, body) == ["milk in allergens_after"]


@pytest.mark.parametrize("goal", ["remove_allergen", "make_vegan"])
def test_milk_not_replaced_fails_both_milk_goals(goal):
    body = answer(GOOD)
    body["substitutions"] = [s for s in body["substitutions"] if s["original"] != "молоко 2.5%"]
    assert check(goal, 200, body) == ["milk not replaced"]


def sugar(before: float, after: float, warnings=()) -> dict:
    nutrition = {"before": {"sugar_g": before}, "after": {"sugar_g": after}}
    return {"nutrition_per_100g": nutrition, "warnings": list(warnings), "trace": []}


def test_reduce_sugar_needs_30_percent_and_no_sweetness_rise():
    assert check("reduce_sugar", 200, sugar(13.26, 9.15)) == []
    assert check("reduce_sugar", 200, sugar(13.26, 9.5)) == ["sugar 13.26 -> 9.5 g, not -30%"]
    rises = sugar(13.26, 9.15, ["Sweetness rises by 40%: ..."])
    assert check("reduce_sugar", 200, rises) == ["sweetness rises"]


def test_errors_and_failed_requests():
    assert check("make_vegan", 502, {"error": {"code": "agent_invalid_output"}}) == [
        "HTTP 502 agent_invalid_output"
    ]
    assert check("make_vegan", 0, {"error": {"code": "ReadTimeout"}}) == [
        "request failed: ReadTimeout"
    ]


def test_usage_shows_fallback_provider_calls_and_tokens():
    body = answer(GOOD)
    body["trace"][1]["usage"] = {"provider": "gemini,groq", "total_tokens": 2000}
    assert usage(body) == {"provider": "gemini,groq", "calls": 2, "tokens": 5000}
