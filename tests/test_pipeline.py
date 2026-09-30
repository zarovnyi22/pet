import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.agent import pipeline as pipeline_mod
from app.agent import tools as tools_mod
from app.agent.common import AgentError
from app.agent.pipeline import ReformulationPipeline
from app.agent.tools import NutritionIngredient, Toolbox, calc_nutrition
from app.allergens import parse_allergens_table
from app.ingest import parse_nutrients_table
from app.llm.fake import FakeLLM
from app.llm.gemini import GeminiClient
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
                {"english": "polydextrose", "spec": "spec-polydextrose"},
                {"english": "stevia", "spec": "spec-stevia"},
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
    "grams": 0.2,  # max 0.05% of the product (spec-yogurt-starter)
    "nutrients_source": "spec-yogurt-starter",
    "nutrients_column": DVS,
    "sources": ["spec-yogurt-starter"],
    "rationale": "dairy-free culture",
    "confidence": "medium",
}


def starter(base: dict = SOY) -> list[dict]:
    """Bulk starter 10 g -> plant DVS culture 0.2 g + 9.8 g of the plant base, as the
    Substitutes table of spec-yogurt-starter says."""
    extra = {"original": "закваска", "grams": 9.8, "rationale": "starter mass as plant base"}
    return [PLANT_STARTER, base | extra]


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
        "substitutions": [SOY, *starter()],
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
        ("starter", 0.2, value("spec-yogurt-starter", DVS)),
        ("starter base", 9.8, value("spec-soy-drink")),
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
    llm = FakeLLM([plan(), choice(substitutions=[unseen, *starter()]), choice()])
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
    llm = FakeLLM([plan(), choice(substitutions=[oat, *starter(oat)])])
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
    llm = FakeLLM([plan(), choice(substitutions=[cream, *starter()], allergens_after=[]), choice()])
    out = await run(llm)  # the model claimed milk-free; the source says otherwise

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert error.message.startswith("['вершки'] still contain 'milk' according to their sources")
    assert out.substitutions[0].replacement == "соєвий напій"


async def test_make_vegan_rejects_an_ingredient_its_source_calls_non_vegan():
    request = YOGURT.model_copy(update={"goal": "make_vegan", "goal_params": GoalParams()})
    only_starter = choice(substitutions=starter(), allergens_after=["milk"])
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
    fixed = choice(substitutions=[oat, *starter(oat)], allergens_after=["gluten"])
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
    answer = choice(substitutions=[COCONUT, *starter(COCONUT)], allergens_after=[])
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
    answer = choice(substitutions=[COCONUT, *starter(COCONUT)], allergens_after=[])
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


# --- sweetness ------------------------------------------------------------------------------

SUGAR_30 = YOGURT.model_copy(update={"goal": "reduce_sugar", "goal_params": GoalParams(percent=30)})
POLYDEXTROSE = ERYTHRITOL | {
    "replacement": "полідекстроза",
    "nutrients_source": "spec-polydextrose",
    "sources": ["spec-polydextrose"],
}
STEVIA = ERYTHRITOL | {
    "replacement": "стевія",
    "grams": 0.1,
    "nutrients_source": "spec-stevia",
    "sources": ["spec-stevia"],
}


def sweetness_warnings(out) -> list[str]:
    return [w for w in out.warnings if "weetness" in w]


async def test_erythritol_and_polydextrose_half_and_half_warn_about_sweetness():
    answer = choice(substitutions=[ERYTHRITOL, POLYDEXTROSE], allergens_after=["milk"])
    out = await run(FakeLLM([plan(), answer]), SUGAR_30)

    # 90 g sucrose -> 48.8 g sucrose + 20.6 g erythritol x 0.65 + 20.6 g polydextrose x 0.05.
    assert [s.grams for s in out.substitutions] == [20.6, 20.6]
    assert sweetness_warnings(out) == [
        "Sweetness drops by 30%: sucrose equivalent 90.0 g -> 63.2 g per batch. About "
        "0.09-0.11 g steviol glycosides (0.011% of the product) would close the 26.8 g gap "
        "(spec-stevia: 0.01% Reb A ~ 2.5-3% sucrose); not added to the recipe."
    ]
    assert [s.replacement for s in out.substitutions] == ["еритрит", "полідекстроза"]  # advice only


async def test_stevia_chosen_by_the_model_is_dosed_by_code():
    too_much = STEVIA | {"grams": 0.4}  # live run 85: sweetness 90 -> ~164 g (+80%)
    answer = choice(substitutions=[ERYTHRITOL, POLYDEXTROSE, too_much], allergens_after=["milk"])
    out = await run(FakeLLM([plan(), answer]), SUGAR_30)

    # Bulk as before (20.6 + 20.6 g). Gap: 90 - (48.8 + 20.6 x 0.65 + 20.6 x 0.05) = 26.8 g of
    # sucrose equivalent; each gram of stevia instead of sugar adds 250 - 1 -> 0.11 g.
    assert [s.grams for s in out.substitutions] == [20.6, 20.6, 0.11]
    [fix] = [s.message for s in out.trace if s.type == "correction" and "stevia dose" in s.message]
    assert fix == (
        "stevia dose computed by code: 0.11 g of 'стевія' for a 26.8 g sucrose-equivalent gap "
        "(the model proposed 0.4 g)"
    )
    assert sweetness_warnings(out) == []  # neither a drop nor a rise


async def test_stevia_dose_is_capped_at_its_max_dosage(monkeypatch):
    monkeypatch.setitem(TABLES["spec-stevia"]["Value"], "max_dosage_pct", 0.005)  # 0.05 g
    answer = choice(substitutions=[ERYTHRITOL, POLYDEXTROSE, STEVIA], allergens_after=["milk"])
    out = await run(FakeLLM([plan(), answer]), SUGAR_30)

    assert out.substitutions[2].grams == 0.05
    assert (
        "стевія capped at the 0.005% limit of spec-stevia (0.05 g): it closes 12.5 of the 26.8 g "
        "sucrose-equivalent gap."
    ) in out.warnings
    assert any(w.startswith("Sweetness drops by") for w in out.warnings)  # the rest of the gap


async def test_intense_sweetener_alone_cannot_replace_sugar():
    only_stevia = choice(substitutions=[STEVIA], allergens_after=["milk"])
    fixed = choice(substitutions=[ERYTHRITOL, STEVIA], allergens_after=["milk"])
    out = await run(FakeLLM([plan(), only_stevia, fixed]), SUGAR_30)

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "['стевія'] add sweetness but no bulk" in error.message


async def test_sweetness_rise_is_warned_too():
    # Outside reduce_sugar the model's stevia dose is not recomputed: 0.4 g is far too much.
    sugar_swap = [ERYTHRITOL | {"grams": 45}, STEVIA | {"grams": 0.4}]
    answer = choice(substitutions=[SOY, *starter(), *sugar_swap])
    out = await run(FakeLLM([plan(), answer]))

    assert sweetness_warnings(out) == [
        "Sweetness rises by 93%: sucrose equivalent 90.0 g -> 173.8 g per batch; the product "
        "will taste sweeter than the original."
    ]


async def test_sugar_replacement_without_sweetness_data_is_flagged():
    oat = ERYTHRITOL | {"replacement": "вівсяний напій", "nutrients_source": "spec-oat-drink"}
    out = await run(
        FakeLLM([plan(), choice(substitutions=[oat], allergens_after=["milk"])]), SUGAR_30
    )

    assert sweetness_warnings(out) == [
        "Sweetness of вівсяний напій is unknown (no relative sweetness in its source): the "
        "change in sweetness was not checked."
    ]


async def test_sweetness_passes_the_provenance_check_and_stays_out_of_nutrition():
    answer = choice(substitutions=[ERYTHRITOL, POLYDEXTROSE], allergens_after=["milk"])
    out = await run(FakeLLM([plan(), answer]), SUGAR_30)

    calcs = [s for s in out.trace if s.tool == "calc_nutrition"]
    sugar = calcs[0].arguments["ingredients"][1]
    assert sugar["nutrients_per_100g"]["sweetness"] == 1.0  # sent to calc_nutrition...
    assert [c.result["rejected"] for c in calcs] == [[], []]  # ...and backed by the spec
    assert "sweetness" not in out.nutrition_per_100g.after.model_dump()


async def test_toolbox_backs_a_copied_sweetness_and_rejects_an_invented_one():
    tools = Toolbox(pool=None, embedder=None, http=None)
    tools.remember_table("spec-erythritol", TABLES["spec-erythritol"])
    copied = TABLES["spec-erythritol"]["Value"]
    ingredients = [
        {
            "name": "a",
            "grams": 50,
            "nutrients_per_100g": copied,
            "nutrients_source": "spec-erythritol",
        },
        {
            "name": "b",
            "grams": 50,
            "nutrients_per_100g": copied | {"sweetness": 0.9},
            "nutrients_source": "spec-erythritol",
        },
    ]
    result = await tools.execute("calc_nutrition", {"ingredients": ingredients})

    assert [(r["ingredient"], r["nutrients"]) for r in result["rejected"]] == [("b", ["sweetness"])]
    assert result["per_100g"]["carbs_g"] == 100  # nutrients themselves are unaffected


# --- token usage in the trace -----------------------------------------------------------------


async def test_llm_calls_record_provider_tokens_and_start_time(monkeypatch):
    answers = iter([plan(), choice()])

    def gemini_answer(request):
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": next(answers)}]}}],
                "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200},
            },
        )

    llm = GeminiClient("key", "gemini-test", 5)
    await llm.aclose()
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(gemini_answer))
    out = await run(llm)
    await llm.aclose()

    calls = [s for s in out.trace if s.type == "llm_call"]
    assert [c.tool for c in calls] == ["plan", "choose"]
    for call in calls:
        assert call.usage | {"at_ms": 0} == {
            "provider": "gemini",
            "http_attempts": 1,
            "input_tokens": 1000,
            "output_tokens": 200,
            "reasoning_tokens": 0,
            "total_tokens": 1200,
            "at_ms": 0,
        }
    assert 0 <= calls[0].usage["at_ms"] <= calls[1].usage["at_ms"]


async def test_failed_llm_call_is_in_the_trace_with_its_attempts():
    llm = GeminiClient("key", "gemini-test", 5)
    await llm.aclose()
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(AgentError) as err:
        await run(llm)
    await llm.aclose()

    assert err.value.code == "llm_rate_limited"
    [call] = [s for s in err.value.trace if s.type == "llm_call"]
    assert (call.tool, call.message) == ("plan", "llm_rate_limited")
    assert call.usage["http_attempts"] == 1 and call.usage["total_tokens"] == 0


# --- dosage limits and the model's numbers ------------------------------------------------------


async def test_dry_culture_above_its_max_dosage_is_refused_with_a_hint():
    whole_mass = PLANT_STARTER | {"grams": 10}  # live runs: 10 g of a 0.05% culture
    out = await run(FakeLLM([plan(), choice(substitutions=[SOY, whole_mass]), choice()]))

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert error.message == (
        "'рослинна закваска': 10 g is 1.00% of the product, spec-yogurt-starter allows max 0.05% "
        "= 0.50 g; replace the rest of 'закваска''s mass with the base ingredient as a second "
        "substitution"
    )
    # The retry (0.2 g culture + 9.8 g soy base) passes the mass check: 10 g of 10 g replaced.
    assert [(s.replacement, s.grams) for s in out.substitutions[1:]] == [
        ("рослинна закваска", 0.2),
        ("соєвий напій", 9.8),
    ]


async def test_model_warnings_with_doses_or_percentages_are_dropped():
    wrong = "Use erythritol at 1.08% (10.8 g) of the product."  # live run: it was 2.04%
    process = "Ferment the soy base longer than milk: the gel sets more slowly."
    out = await run(FakeLLM([plan(), choice(warnings=[wrong, process])]))

    assert wrong not in out.warnings and process in out.warnings
    [fix] = [s.message for s in out.trace if s.type == "correction" and "dropped" in s.message]
    assert fix.endswith(f"{[wrong]}")
