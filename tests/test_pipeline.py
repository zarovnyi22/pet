import asyncio
import json
from pathlib import Path

import pytest

from app.agent import pipeline as pipeline_mod
from app.agent import tools as tools_mod
from app.agent.common import AgentError
from app.agent.pipeline import ReformulationPipeline
from app.agent.tools import NutritionIngredient, Toolbox, calc_nutrition
from app.allergens import parse_allergens_table
from app.ingest import parse_nutrients_table
from app.llm.fake import FakeLLM
from app.schemas import GoalParams, ReformulateIn, Source

# Every test here runs against the fixture knowledge base below; other modules opt in explicitly.
pytestmark = pytest.mark.usefixtures("knowledge_base")

YOGURT = ReformulateIn(
    product_name="Полуничний йогурт 2.5%",
    ingredients=[
        {"name": "молоко 2.5%", "grams": 800},
        {"name": "цукор", "grams": 90},
        {"name": "полуниця заморожена", "grams": 100},
        {"name": "закваска", "grams": 10},
    ],
    goal="remove_allergen",
    goal_params={"allergen": "milk"},
)


BULK, DVS = "Bulk starter (fermented milk 2.5%)", "Freeze-dried plant-based DVS"
# Parsed from the corpus by the ingest parsers, never typed by hand: a hand-written fixture
# hid the dairy-starter bug (its facts were missing, so the check never saw bulk = milk).
CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
SPECS = {path.stem: path.read_text() for path in sorted(CORPUS.glob("spec-*.md"))}
TABLES = {doc_id: parse_nutrients_table(content) for doc_id, content in SPECS.items()}
FACTS = {doc_id: parse_allergens_table(content) for doc_id, content in SPECS.items()}
OFF_STRAWBERRY = {
    "source": "off:5010251784173",
    "product_name": "Strawberries",
    "nutrients_per_100g": {
        "kcal": 32,
        "protein_g": 0.67,
        "fat_g": 0,
        "carbs_g": 7.63,
        "sugar_g": 4.66,
    },
    "allergens": [],
    "vegan": "yes",
}


@pytest.fixture
def knowledge_base(monkeypatch):
    """The DB and Open Food Facts, replaced by the corpus values above."""

    async def catalog(pool):
        return [(doc_id, doc_id.removeprefix("spec-")) for doc_id in TABLES]

    async def tables(pool, doc_ids):
        return {d: TABLES[d] for d in doc_ids if d in TABLES}

    async def allergens(pool, doc_ids):
        return {d: FACTS[d] for d in doc_ids if FACTS.get(d)}

    async def search(pool, embedder, query, top_k):
        text = "Erythritol replaced 30% of sucrose; texture equal to control."
        return [Source(doc_id="trial-cookies-sugar-30", title="Trial", chunk_text=text, score=0.6)]

    async def lookup(http, name):
        return {"query": name, "products": [OFF_STRAWBERRY]}

    monkeypatch.setattr(pipeline_mod, "spec_catalog", catalog)
    monkeypatch.setattr(pipeline_mod, "doc_nutrients", tables)
    monkeypatch.setattr(pipeline_mod, "doc_allergens", allergens)
    monkeypatch.setattr(tools_mod, "doc_nutrients", tables)
    monkeypatch.setattr(tools_mod, "search", search)
    monkeypatch.setattr(tools_mod, "lookup_product", lookup)


def plan(strawberry_spec: str | None = "spec-strawberry-frozen") -> str:
    return json.dumps(
        {
            "ingredients": [
                {"name": "молоко 2.5%", "english": "milk 2.5%", "spec": "spec-milk-2-5"},
                {"name": "цукор", "english": "sugar", "spec": "spec-sucrose"},
                {
                    "name": "полуниця заморожена",
                    "english": "frozen strawberries",
                    "spec": strawberry_spec,
                },
                {"name": "закваска", "english": "yogurt starter", "spec": "spec-yogurt-starter"},
            ],
            "candidates": [
                {"english": "soy drink", "spec": "spec-soy-drink"},
                {"english": "erythritol", "spec": "spec-erythritol"},
                {"english": "oat drink", "spec": "spec-oat-drink"},
            ],
            "queries": ["milk-free yogurt trial"],
        },
        ensure_ascii=False,
    )


ORIGINALS = [
    {"name": "молоко 2.5%", "nutrients_source": "spec-milk-2-5"},
    {"name": "цукор", "nutrients_source": "spec-sucrose"},
    {"name": "полуниця заморожена", "nutrients_source": "spec-strawberry-frozen"},
    {"name": "закваска", "nutrients_source": "spec-yogurt-starter", "nutrients_column": BULK},
]
SOY = {
    "original": "молоко 2.5%",
    "replacement": "соєвий напій",
    "grams": 800,
    "nutrients_source": "spec-soy-drink",
    "sources": ["spec-soy-drink"],
    "rationale": "1:1 by weight",
    "confidence": "medium",
}
PLANT_STARTER = {
    "original": "закваска",
    "replacement": "рослинна закваска",
    "grams": 10,
    "nutrients_source": "spec-yogurt-starter",
    "nutrients_column": DVS,
    "sources": ["spec-yogurt-starter"],
    "rationale": "dairy-free culture",
    "confidence": "medium",
}
ERYTHRITOL = {
    "original": "цукор",
    "replacement": "еритрит",
    "grams": 20,  # only a proportion: code computes the dose
    "nutrients_source": "spec-erythritol",
    "sources": ["spec-erythritol", "trial-cookies-sugar-30"],
    "rationale": "bulk without sugar",
    "confidence": "high",
}


def choice(**overrides) -> str:
    answer = {
        "original_nutrients": ORIGINALS,
        "substitutions": [SOY, PLANT_STARTER],
        "allergens_before": ["milk"],
        "allergens_after": ["soy"],
        "warnings": [],
    } | overrides
    return json.dumps(answer, ensure_ascii=False)


def run(llm: FakeLLM, request: ReformulateIn = YOGURT, timeout: float = 5):
    tools = Toolbox(pool=None, embedder=None, http=None)
    return ReformulationPipeline(llm, tools, timeout_seconds=timeout).run(request)


def per_100g(*items: tuple[str, float, dict]) -> dict:
    parsed = [NutritionIngredient(name=n, grams=g, nutrients_per_100g=v) for n, g, v in items]
    return calc_nutrition(parsed).per_100g.model_dump()


def value(doc_id: str, column: str = "Value") -> dict:
    return TABLES[doc_id][column]


async def test_remove_allergen_takes_every_number_from_the_sources():
    llm = FakeLLM([plan(), choice()])
    out = await run(llm)

    assert len(llm.calls) == 2  # plan + choose, nothing else
    assert out.nutrition_per_100g.before.model_dump() == per_100g(
        ("milk", 800, value("spec-milk-2-5")),
        ("sugar", 90, value("spec-sucrose")),
        ("strawberry", 100, value("spec-strawberry-frozen")),
        ("starter", 10, value("spec-yogurt-starter", BULK)),
    )
    assert out.nutrition_per_100g.after.model_dump() == per_100g(
        ("soy", 800, value("spec-soy-drink")),
        ("sugar", 90, value("spec-sucrose")),
        ("strawberry", 100, value("spec-strawberry-frozen")),
        ("starter", 10, value("spec-yogurt-starter", DVS)),
    )
    assert (out.allergens_before, out.allergens_after) == (["milk"], ["soybeans"])  # canonical
    calcs = [s for s in out.trace if s.tool == "calc_nutrition"]
    assert [c.result["rejected"] for c in calcs] == [[], []]
    assert [s.type for s in out.trace if s.type == "llm_call"] == ["llm_call"] * 2


async def test_reduce_sugar_dose_is_solved_by_code():
    request = YOGURT.model_copy(
        update={"goal": "reduce_sugar", "goal_params": GoalParams(percent=30)}
    )
    llm = FakeLLM([plan(), choice(substitutions=[ERYTHRITOL], allergens_after=["milk"])])
    out = await run(llm, request)

    before = out.nutrition_per_100g.before.sugar_g
    after = out.nutrition_per_100g.after.sugar_g
    # 30% asked, code aims at 31%: met, and not overshot (lactose and fruit sugar included).
    assert before * 0.69 - 0.02 <= after <= before * 0.70
    # (132.56 g sugar - 91.47 g target) / (100 - 0) g sugar per g -> 41.1 g erythritol.
    assert out.substitutions[0].grams == 41.1
    [dose] = [s for s in out.trace if s.type == "correction" and "sugar dose" in s.message]
    assert "41.1 g" in dose.message and "proposed 20.0 g" in dose.message
    [calc_after] = [s for s in out.trace if s.tool == "calc_nutrition"][1:]
    assert calc_after.result["total_grams"] == 1000  # mass compensated
    assert out.substitutions[0].confidence == "high"  # backed by a trial report


async def test_unreachable_sugar_goal_is_retried_with_the_reason():
    request = YOGURT.model_copy(
        update={"goal": "reduce_sugar", "goal_params": GoalParams(percent=30)}
    )
    no_better = ERYTHRITOL | {"nutrients_source": "spec-oat-drink"}  # 4 g sugar: fine
    same_sugar = ERYTHRITOL | {"nutrients_source": "spec-sucrose"}  # 100 g: not less sugary
    llm = FakeLLM(
        [
            plan(),
            choice(substitutions=[same_sugar], allergens_after=["milk"]),
            choice(substitutions=[no_better], allergens_after=["milk"]),
        ]
    )
    out = await run(llm, request)

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "not less sugary" in error.message
    assert out.nutrition_per_100g.after.sugar_g <= out.nutrition_per_100g.before.sugar_g * 0.7


async def test_source_without_data_is_retried_with_a_hint():
    unseen = SOY | {"nutrients_source": "spec-coconut"}
    llm = FakeLLM([plan(), choice(substitutions=[unseen, PLANT_STARTER]), choice()])
    out = await run(llm)

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "'spec-coconut' has no nutrient data in this run" in error.message
    assert "Your answer is invalid" in llm.calls[-1][-1].content
    assert out.substitutions[0].replacement == "соєвий напій"


async def test_multi_column_spec_needs_a_column():
    no_column = [o | {"nutrients_column": None} for o in ORIGINALS]
    llm = FakeLLM([plan(), choice(original_nutrients=no_column), choice()])
    out = await run(llm)
    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "several columns" in error.message and BULK in error.message


async def test_second_invalid_choice_is_502_with_trace():
    llm = FakeLLM([plan(), "not json", choice(allergens_after=["milk"])])
    with pytest.raises(AgentError) as err:
        await run(llm)

    assert (err.value.status_code, err.value.code) == (502, "agent_invalid_output")
    errors = [s.message for s in err.value.trace if s.type == "validation_error"]
    assert len(errors) == 2 and "allergens_after still contains 'milk'" in errors[1]


async def test_ingredient_without_spec_gets_open_food_facts_nutrients():
    originals = [
        o | {"nutrients_source": OFF_STRAWBERRY["source"]}
        if o["name"] == "полуниця заморожена"
        else o
        for o in ORIGINALS
    ]
    llm = FakeLLM([plan(strawberry_spec=None), choice(original_nutrients=originals)])
    out = await run(llm)

    [lookup] = [s for s in out.trace if s.tool == "lookup_product"]
    assert lookup.arguments == {"name": "frozen strawberries"}
    assert OFF_STRAWBERRY["source"] in llm.calls[1][-1].content  # offered as a candidate
    assert (
        out.nutrition_per_100g.before.kcal
        == per_100g(
            ("milk", 800, value("spec-milk-2-5")),
            ("sugar", 90, value("spec-sucrose")),
            ("strawberry", 100, OFF_STRAWBERRY["nutrients_per_100g"]),
            ("starter", 10, value("spec-yogurt-starter", BULK)),
        )["kcal"]
    )


async def test_plan_must_cover_every_ingredient():
    partial = json.loads(plan())
    partial["ingredients"] = partial["ingredients"][:2]
    llm = FakeLLM([json.dumps(partial, ensure_ascii=False), plan(), choice()])
    out = await run(llm)
    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "ingredients missing from the plan" in error.message


async def test_code_fixes_sources_confidence_and_warnings():
    oat = SOY | {
        "replacement": "вівсяний напій",
        "nutrients_source": "spec-oat-drink",
        "sources": ["trial-made-up"],  # never seen: dropped
        "confidence": "high",
    }
    llm = FakeLLM([plan(), choice(substitutions=[oat, PLANT_STARTER])])
    out = await run(llm)

    sub = out.substitutions[0]
    assert sub.sources == ["spec-oat-drink"]  # the nutrient source is always cited
    assert sub.confidence == "medium"  # "high" needs a trial report
    assert any(w.startswith("Protein drops from") for w in out.warnings)  # 2.4 -> ~0.9 g


async def test_timeout_is_504_with_trace():
    class SlowLLM(FakeLLM):
        async def complete(self, messages, *, json_mode=False):
            await asyncio.sleep(1)

    with pytest.raises(AgentError) as err:
        await run(SlowLLM(), timeout=0.05)
    assert (err.value.status_code, err.value.code) == (504, "agent_timeout")
    assert err.value.trace[-1].type == "error"


@pytest.mark.parametrize("duplicate", ["цукор", " Цукор "])
def test_duplicate_ingredient_names_are_rejected(duplicate):
    # The pipeline handles the recipe by name: a duplicate would merge and lose 45 g.
    data = YOGURT.model_dump() | {
        "ingredients": [*YOGURT.model_dump()["ingredients"], {"name": duplicate, "grams": 45}]
    }
    with pytest.raises(ValueError, match="duplicate ingredient names"):
        ReformulateIn.model_validate(data)


async def test_allergens_the_model_left_out_are_added_from_the_sources():
    llm = FakeLLM([plan(), choice(allergens_before=[], allergens_after=[])])
    out = await run(llm)

    assert (out.allergens_before, out.allergens_after) == (["milk"], ["soybeans"])
    fixes = [s.message for s in out.trace if s.type == "correction" and "allergens" in s.message]
    assert fixes == [
        "allergens_before: added ['milk'] stated by the sources of ['закваска', 'молоко 2.5%']",
        "allergens_after: added ['soybeans'] stated by the sources of ['соєвий напій']",
    ]


async def test_remove_allergen_rejects_a_replacement_that_contains_it():
    cream = SOY | {"replacement": "вершки", "nutrients_source": "spec-milk-2-5"}
    llm = FakeLLM(
        [plan(), choice(substitutions=[cream, PLANT_STARTER], allergens_after=[]), choice()]
    )
    out = await run(llm)  # the model claimed milk-free; the source says otherwise

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert error.message.startswith("['вершки'] still contain 'milk' according to their sources")
    assert out.substitutions[0].replacement == "соєвий напій"


async def test_make_vegan_rejects_an_ingredient_its_source_calls_non_vegan():
    request = YOGURT.model_copy(update={"goal": "make_vegan", "goal_params": GoalParams()})
    only_starter = choice(substitutions=[PLANT_STARTER], allergens_after=["milk"])
    llm = FakeLLM([plan(), only_starter, choice()])
    out = await run(llm, request)

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "['молоко 2.5%'] are not vegan according to their sources" in error.message
    assert out.allergens_after == ["soybeans"]


# --- unknown allergen / vegan status is not safe ----------------------------------------------


@pytest.fixture
def soy_without_allergens_table(knowledge_base, monkeypatch):
    """spec-soy-drink as a row ingested before its allergens table existed (NULL)."""

    async def allergens(pool, doc_ids):
        return {d: FACTS[d] for d in doc_ids if FACTS.get(d) and d != "spec-soy-drink"}

    monkeypatch.setattr(pipeline_mod, "doc_allergens", allergens)


@pytest.mark.usefixtures("soy_without_allergens_table")
async def test_spec_without_allergen_data_is_refused_then_warned():
    out = await run(FakeLLM([plan(), choice(), choice()]))

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "status unknown for соєвий напій (spec-soy-drink has no allergens table)" in (
        error.message
    )
    assert "unknown is not safe" in error.message
    assert any("соєвий напій" in w and "not verified" in w for w in out.warnings)
    assert out.substitutions[0].confidence == "low"


@pytest.mark.usefixtures("soy_without_allergens_table")
async def test_retry_with_a_known_source_has_no_warning():
    oat = SOY | {"replacement": "вівсяний напій", "nutrients_source": "spec-oat-drink"}
    fixed = choice(substitutions=[oat, PLANT_STARTER], allergens_after=["gluten"])
    out = await run(FakeLLM([plan(), choice(), fixed]))

    assert out.substitutions[0].replacement == "вівсяний напій"
    assert out.substitutions[0].confidence == "medium"
    assert not [w for w in out.warnings if "status" in w]


def off_drink(vegan: str) -> dict:
    return OFF_STRAWBERRY | {
        "source": "off:123",
        "product_name": "Coconut drink",
        "vegan": vegan,
    }


def plan_with_off_candidate() -> str:
    data = json.loads(plan())
    data["candidates"].append({"english": "coconut drink", "spec": None})
    return json.dumps(data, ensure_ascii=False)


COCONUT = SOY | {
    "replacement": "кокосовий напій",
    "nutrients_source": "off:123",
    "sources": ["off:123"],
}
VEGAN = YOGURT.model_copy(update={"goal": "make_vegan", "goal_params": GoalParams()})


async def test_off_product_without_vegan_status_is_warned_and_low(monkeypatch):
    async def lookup(http, name):
        return {"query": name, "products": [off_drink("unknown")]}

    monkeypatch.setattr(tools_mod, "lookup_product", lookup)
    answer = choice(substitutions=[COCONUT, PLANT_STARTER], allergens_after=[])
    out = await run(FakeLLM([plan_with_off_candidate(), answer]), VEGAN)

    assert not [s for s in out.trace if s.type == "validation_error"]  # warned, not refused
    assert (
        "allergen/vegan status not verified: Open Food Facts data incomplete for "
        "кокосовий напій (off:123: vegan status not given)"
    ) in out.warnings
    assert out.substitutions[0].confidence == "low"


async def test_off_product_stated_non_vegan_is_refused(monkeypatch):
    async def lookup(http, name):
        return {"query": name, "products": [off_drink("no")]}

    monkeypatch.setattr(tools_mod, "lookup_product", lookup)
    answer = choice(substitutions=[COCONUT, PLANT_STARTER], allergens_after=[])
    out = await run(FakeLLM([plan_with_off_candidate(), answer, choice()]), VEGAN)

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "['кокосовий напій'] are not vegan according to their sources" in error.message
    assert out.substitutions[0].replacement == "соєвий напій"


async def test_reduce_sugar_warns_about_unknown_status_instead_of_refusing():
    request = YOGURT.model_copy(
        update={"goal": "reduce_sugar", "goal_params": GoalParams(percent=30)}
    )
    no_strawberry_source = [o for o in ORIGINALS if o["name"] != "полуниця заморожена"]
    answer = choice(
        original_nutrients=no_strawberry_source, substitutions=[ERYTHRITOL], allergens_after=[]
    )
    out = await run(FakeLLM([plan(), answer]), request)

    assert not [s for s in out.trace if s.type == "validation_error"]
    assert "Allergen status unknown for полуниця заморожена (no source): check the label." in (
        out.warnings
    )
