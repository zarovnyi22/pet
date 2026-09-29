import asyncio
import json

import pytest

from app.agent.loop import AgentError, AgentLoop
from app.agent.prompts import FORCE_FINAL
from app.agent.tools import NutritionIngredient, Toolbox, calc_nutrition
from app.llm.base import LLMError, LLMResponse, ToolCall
from app.llm.fake import FakeLLM
from app.schemas import GoalParams, ReformulateIn, ReformulationAnswer

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

MILK = {"kcal": 53, "protein_g": 2.9, "fat_g": 2.5, "carbs_g": 4.7, "sugar_g": 4.7}
OAT = {"kcal": 46, "protein_g": 1.0, "fat_g": 1.5, "carbs_g": 6.7, "sugar_g": 4.0}
SUGAR = {"kcal": 400, "protein_g": 0, "fat_g": 0, "carbs_g": 100, "sugar_g": 100}
BEFORE = [
    {"name": "молоко 2.5%", "grams": 800, "nutrients_per_100g": MILK, "nutrients_source": "off:1"},
    {"name": "цукор", "grams": 90, "nutrients_per_100g": SUGAR, "nutrients_source": "spec-sucrose"},
]
AFTER = [
    {
        "name": "вівсяний напій",
        "grams": 800,
        "nutrients_per_100g": OAT,
        "nutrients_source": "spec-oat-drink",
    },
    BEFORE[1],
]


def seed(tools: Toolbox) -> None:
    """As if earlier searches/lookups in the run had shown these sources and their values."""
    tools.remember_product("off:1", MILK)
    for doc_id, nutrients in [("spec-sucrose", SUGAR), ("spec-oat-drink", OAT)]:
        tools.remember_text(doc_id, " ".join(f"{k} {v}" for k, v in nutrients.items()))


def per_100g(ingredients: list[dict]) -> dict:
    parsed = [NutritionIngredient.model_validate(i) for i in ingredients]
    return calc_nutrition(parsed).per_100g.model_dump()


def final_answer(**overrides) -> str:
    answer = {
        "substitutions": [
            {
                "original": "молоко 2.5%",
                "replacement": "вівсяний напій",
                "grams": 800,
                "rationale": "1:1 by weight",
                "sources": ["spec-oat-drink"],
                "confidence": "medium",
            }
        ],
        "allergens_before": ["milk"],
        "allergens_after": [],
        "nutrition_per_100g": {"before": per_100g(BEFORE), "after": per_100g(AFTER)},
        "warnings": [],
    } | overrides
    return json.dumps(answer, ensure_ascii=False)


def call(name: str, id_: str = "c1", **arguments) -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=arguments)


def turn(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(text="", tool_calls=list(calls))


CALCS = turn(
    call("calc_nutrition", "calc-before", ingredients=BEFORE),
    call("calc_nutrition", "calc-after", ingredients=AFTER),
)


class FakeTools:
    """Knowledge-base search is canned; calc_nutrition is the real arithmetic."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.delay = delay
        self._real = Toolbox(pool=None, embedder=None, http=None)
        seed(self._real)

    async def execute(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        await asyncio.sleep(self.delay)
        if name == "calc_nutrition":
            return await self._real.execute(name, arguments)
        chunk = {"doc_id": "spec-oat-drink", "title": "Oat drink", "chunk_text": "x" * 1000}
        return {"chunks": [chunk | {"score": 0.8}]}


def agent(llm: FakeLLM, tools: FakeTools | None = None, **limits) -> AgentLoop:
    limits = {"max_iterations": 6, "timeout_seconds": 5} | limits
    return AgentLoop(llm, tools or FakeTools(), **limits)


async def test_tool_call_then_final_answer():
    llm = FakeLLM([turn(call("search_knowledge_base", query="oat drink")), CALCS, final_answer()])
    tools = FakeTools()
    out = await agent(llm, tools).run(YOGURT)

    assert out.substitutions[0].replacement == "вівсяний напій"
    assert tools.calls[0] == ("search_knowledge_base", {"query": "oat drink"})
    assert [(s.iteration, s.type, s.tool) for s in out.trace] == [
        (1, "tool_call", "search_knowledge_base"),
        (2, "tool_call", "calc_nutrition"),
        (2, "tool_call", "calc_nutrition"),
    ]
    # The model sees the tool result on its next turn.
    last = llm.calls[1][-1]
    assert (last.role, last.name, last.tool_call_id) == ("tool", "search_knowledge_base", "c1")


async def test_all_calls_of_one_turn_run_in_one_iteration():
    llm = FakeLLM([CALCS, final_answer()])
    out = await agent(llm, max_iterations=2).run(YOGURT)

    assert [s.iteration for s in out.trace] == [1, 1]
    tool_messages = [m for m in llm.calls[1] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["calc-before", "calc-after"]
    assert json.loads(tool_messages[1].content)["per_100g"] == per_100g(AFTER)


async def test_iteration_limit_gives_agent_timeout_with_trace():
    llm = FakeLLM([turn(call("search_knowledge_base", query=f"q{i}")) for i in range(3)])
    with pytest.raises(AgentError) as err:
        await agent(llm, max_iterations=3).run(YOGURT)

    assert (err.value.status_code, err.value.code) == (504, "agent_timeout")
    assert [s.type for s in err.value.trace] == ["tool_call"] * 3 + ["error"]
    assert "3 iterations" in err.value.trace[-1].message


async def test_time_limit_gives_agent_timeout():
    llm = FakeLLM([turn(call("search_knowledge_base", query="slow"))])
    with pytest.raises(AgentError) as err:
        await agent(llm, FakeTools(delay=5), timeout_seconds=0.1).run(YOGURT)

    assert err.value.code == "agent_timeout"
    assert "0.1 s" in err.value.trace[-1].message


async def test_same_call_twice_in_a_row_forces_final_answer():
    # calc-after closes turn 1 and is requested again in turn 2: a consecutive repeat.
    repeat = turn(call("calc_nutrition", "again", ingredients=AFTER))
    llm = FakeLLM([CALCS, repeat, final_answer()])
    tools = FakeTools()
    out = await agent(llm, tools).run(YOGURT)

    assert len(tools.calls) == 2  # the repeat reused the cached result
    assert [s.type for s in out.trace] == ["tool_call"] * 3 + ["loop_guard"]
    assert out.trace[2].message == "reused result of an identical earlier call"
    # The forced turn is a plain completion (no tools offered) ending with the stop message.
    assert llm.calls[-1][-1].content == FORCE_FINAL


async def test_non_consecutive_repeat_is_not_a_loop():
    search = turn(call("search_knowledge_base", query="oat drink"))
    llm = FakeLLM([search, CALCS, search, final_answer()])
    out = await agent(llm).run(YOGURT)
    assert "loop_guard" not in [s.type for s in out.trace]


async def test_invalid_json_is_retried_once_with_the_error():
    llm = FakeLLM([CALCS, "Here is the answer: oat drink", final_answer()])
    out = await agent(llm).run(YOGURT)

    assert out.trace[-1].type == "validation_error"
    assert "not valid JSON" in out.trace[-1].message
    retry_prompt = llm.calls[-1][-1]
    assert retry_prompt.role == "user" and "not valid JSON" in retry_prompt.content


async def test_second_invalid_answer_fails():
    llm = FakeLLM([CALCS, "nope", json.dumps({"substitutions": []})])
    with pytest.raises(AgentError) as err:
        await agent(llm).run(YOGURT)

    assert (err.value.status_code, err.value.code) == (502, "agent_invalid_output")
    errors = [s.message for s in err.value.trace if s.type == "validation_error"]
    assert len(errors) == 2
    assert "allergens_before: Field required" in errors[1]


async def test_nutrition_not_from_calc_nutrition_is_rejected():
    made_up = {"before": MILK, "after": OAT}
    llm = FakeLLM([CALCS, final_answer(nutrition_per_100g=made_up), final_answer()])
    out = await agent(llm).run(YOGURT)
    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "does not match any calc_nutrition result" in error.message


async def test_answer_without_calc_nutrition_is_rejected():
    llm = FakeLLM([final_answer(), final_answer()])
    with pytest.raises(AgentError) as err:
        await agent(llm).run(YOGURT)
    assert "calc_nutrition" in err.value.message


def test_unsourced_substitution_must_be_low_confidence_with_warning():
    base = json.loads(final_answer())
    sub = base["substitutions"][0] | {"sources": []}
    with pytest.raises(ValueError, match="confidence must be 'low'"):
        ReformulationAnswer.model_validate(base | {"substitutions": [sub]})
    low = sub | {"confidence": "low"}
    with pytest.raises(ValueError, match="warnings"):
        ReformulationAnswer.model_validate(base | {"substitutions": [low]})
    ok = base | {"substitutions": [low], "warnings": ["no source for oat drink"]}
    assert ReformulationAnswer.model_validate(ok).substitutions[0].confidence == "low"


async def test_trace_shortens_long_results_but_the_model_sees_them_in_full():
    llm = FakeLLM([turn(call("search_knowledge_base", query="oat")), CALCS, final_answer()])
    out = await agent(llm).run(YOGURT)

    traced = out.trace[0].result["chunks"][0]["chunk_text"]
    assert len(traced) == 301 and traced.endswith("…")
    sent = json.loads(llm.calls[1][-1].content)["chunks"][0]["chunk_text"]
    assert len(sent) == 1000


async def test_llm_failure_keeps_the_trace():
    class FailingLLM(FakeLLM):
        async def complete_with_tools(self, messages, tools):
            if self.responses:
                return await super().complete_with_tools(messages, tools)
            raise LLMError("quota", code="llm_rate_limited", status_code=503)

    llm = FailingLLM([turn(call("search_knowledge_base", query="oat"))])
    with pytest.raises(AgentError) as err:
        await agent(llm).run(YOGURT)

    assert (err.value.status_code, err.value.code) == (503, "llm_rate_limited")
    assert [s.type for s in err.value.trace] == ["tool_call", "error"]


async def test_high_confidence_without_trial_report_is_lowered_in_code():
    subs = json.loads(final_answer())["substitutions"]
    spec_only = subs[0] | {"confidence": "high", "sources": ["spec-oat-drink", "off:1"]}
    with_trial = subs[0] | {
        "original": "цукор",
        "confidence": "high",
        "sources": ["spec-erythritol", "trial-cookies-sugar-30"],
    }
    llm = FakeLLM([CALCS, final_answer(substitutions=[spec_only, with_trial])])
    out = await agent(llm).run(YOGURT)

    assert [s.confidence for s in out.substitutions] == ["medium", "high"]
    [fix] = [s for s in out.trace if s.type == "correction"]
    assert "'молоко 2.5%'" in fix.message and "high to medium" in fix.message
    # Corrected, not retried: no validation error, no extra LLM turn.
    assert "validation_error" not in [s.type for s in out.trace]
    assert len(llm.calls) == 2


# reduce_sugar: sugar per 100 g must drop by the requested percent. BEFORE has 14.34 g.
SUGAR_60 = [BEFORE[0], BEFORE[1] | {"grams": 60}]  # 11.35 g: -20.9%
SUGAR_30 = [BEFORE[0], BEFORE[1] | {"grams": 30}]  # 8.14 g: -43.2%


def reduce_sugar(percent: int) -> ReformulateIn:
    return YOGURT.model_copy(
        update={"goal": "reduce_sugar", "goal_params": GoalParams(percent=percent)}
    )


def sugar_answer(after: list[dict]) -> str:
    sub = {
        "original": "цукор",
        "replacement": "цукор",
        "grams": after[1]["grams"],
        "rationale": "less sugar",
        "sources": ["spec-sucrose"],
        "confidence": "medium",
    }
    nutrition = {"before": per_100g(BEFORE), "after": per_100g(after)}
    return final_answer(substitutions=[sub], nutrition_per_100g=nutrition, allergens_after=["milk"])


def calcs(after: list[dict]) -> LLMResponse:
    return turn(
        call("calc_nutrition", "before", ingredients=BEFORE),
        call("calc_nutrition", "after", ingredients=after),
    )


async def test_sugar_goal_met_is_accepted():
    llm = FakeLLM([calcs(SUGAR_60), sugar_answer(SUGAR_60)])
    out = await agent(llm).run(reduce_sugar(20))
    assert out.nutrition_per_100g.after.sugar_g == per_100g(SUGAR_60)["sugar_g"]


async def test_sugar_goal_missed_is_retried_with_the_numbers():
    llm = FakeLLM(
        [calcs(SUGAR_60), sugar_answer(SUGAR_60), calcs(SUGAR_30), sugar_answer(SUGAR_30)]
    )
    out = await agent(llm).run(reduce_sugar(30))

    [error] = [s for s in out.trace if s.type == "validation_error"]
    assert "at least 30%" in error.message and "only 20.9% less" in error.message
    assert out.nutrition_per_100g.after.sugar_g == per_100g(SUGAR_30)["sugar_g"]


async def test_sugar_goal_is_not_checked_for_other_goals():
    llm = FakeLLM([calcs(SUGAR_60), sugar_answer(SUGAR_60)])
    out = await agent(llm).run(YOGURT)  # remove_allergen: any sugar change is fine
    assert "validation_error" not in [s.type for s in out.trace]
