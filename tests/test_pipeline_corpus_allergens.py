"""PM regression (30.09.2026): a bulk yogurt starter is fermented milk, so a yogurt that keeps
it must never come out milk-free or vegan. Facts are parsed from data/corpus here, not
taken from test_pipeline's fixture, so this file tests the corpus and the parser too."""

from pathlib import Path

import pytest

from app import allergens
from app.agent import pipeline as pipeline_mod
from app.agent.common import AgentError
from app.llm.fake import FakeLLM
from app.schemas import GoalParams
from tests.test_pipeline import (  # noqa: F401
    PLANT_STARTER,
    SOY,
    YOGURT,
    choice,
    knowledge_base,
    plan,
    run,
)

CORPUS = Path(__file__).parent.parent / "data" / "corpus"
VEGAN = YOGURT.model_copy(update={"goal": "make_vegan", "goal_params": GoalParams()})


@pytest.fixture(autouse=True)
def facts_from_corpus(knowledge_base, monkeypatch):  # noqa: F811
    async def from_corpus(pool, doc_ids):
        return {
            d: allergens.parse_allergens_table((CORPUS / f"{d}.md").read_text()) for d in doc_ids
        }

    monkeypatch.setattr(pipeline_mod, "doc_allergens", from_corpus)


async def test_dairy_starter_blocks_milk_free_claim():
    forgot_starter = choice(substitutions=[SOY], allergens_after=["soy"])
    with pytest.raises(AgentError) as err:
        await run(FakeLLM([plan(), forgot_starter, forgot_starter]))
    assert "['закваска'] still contain 'milk'" in err.value.message


async def test_make_vegan_keeping_the_dairy_starter_fails():
    forgot_starter = choice(substitutions=[SOY], allergens_after=["soy"])
    with pytest.raises(AgentError) as err:
        await run(FakeLLM([plan(), forgot_starter, forgot_starter]), VEGAN)
    assert "['закваска'] are not vegan" in err.value.message


async def test_make_vegan_with_soy_drink_and_plant_starter_passes():
    out = await run(FakeLLM([plan(), choice(substitutions=[SOY, PLANT_STARTER])]), VEGAN)

    assert [s.replacement for s in out.substitutions] == ["соєвий напій", "рослинна закваска"]
    assert out.allergens_after == ["soybeans"]
    assert not [w for w in out.warnings if "status" in w]
