from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DocType = Literal["ingredient_spec", "trial_report", "guideline"]


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str = Field(min_length=1)
    doc_type: DocType
    content: str

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        return value


class DocumentOut(BaseModel):
    doc_id: str
    chunks_created: int


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class Source(BaseModel):
    doc_id: str
    title: str
    chunk_text: str
    # Cosine similarity of the chunk to the query, for every chunk, also one found only by
    # full-text search (then it is low: found by the term, not by meaning). Results are ordered
    # by Reciprocal Rank Fusion, not by this score.
    score: float
    # Which search found it: ["vector"], ["fulltext"] or both.
    matched_by: list[Literal["vector", "fulltext"]] = []


class AskOut(BaseModel):
    answer: str
    sources: list[Source]


# --- POST /reformulate ---------------------------------------------------------------------

# EU Regulation 1169/2011, Annex II.
EUAllergen = Literal[
    "gluten",
    "crustaceans",
    "eggs",
    "fish",
    "peanuts",
    "soybeans",
    "milk",
    "nuts",
    "celery",
    "mustard",
    "sesame",
    "sulphites",
    "lupin",
    "molluscs",
]

Goal = Literal["remove_allergen", "reduce_sugar", "make_vegan"]
Confidence = Literal["high", "medium", "low"]


class RecipeIngredient(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    grams: float = Field(gt=0)


class GoalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allergen: EUAllergen | None = None
    percent: int | None = Field(default=None, ge=10, le=50)


class ReformulateIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    # The example from docs/SPEC.md, "Що будуємо".
                    "product_name": "Полуничний йогурт 2.5%",
                    "ingredients": [
                        {"name": "молоко 2.5%", "grams": 800},
                        {"name": "цукор", "grams": 90},
                        {"name": "полуниця заморожена", "grams": 100},
                        {"name": "закваска", "grams": 10},
                    ],
                    "goal": "remove_allergen",
                    "goal_params": {"allergen": "milk"},
                }
            ]
        }
    )

    product_name: str = Field(min_length=1, max_length=200)
    ingredients: list[RecipeIngredient] = Field(min_length=1, max_length=50)
    goal: Goal
    goal_params: GoalParams = Field(default_factory=GoalParams)

    @field_validator("ingredients")
    @classmethod
    def names_are_unique(cls, ingredients: list[RecipeIngredient]) -> list[RecipeIngredient]:
        # The recipe is handled by name: duplicates would silently merge and lose their mass.
        seen, duplicates = set(), []
        for ingredient in ingredients:
            key = ingredient.name.strip().casefold()
            if key in seen:
                duplicates.append(ingredient.name)
            seen.add(key)
        if duplicates:
            raise ValueError(f"duplicate ingredient names {duplicates}: merge them into one line")
        return ingredients

    @model_validator(mode="after")
    def params_match_goal(self) -> "ReformulateIn":
        p = self.goal_params
        if self.goal == "remove_allergen":
            if p.allergen is None or p.percent is not None:
                raise ValueError("goal remove_allergen requires goal_params.allergen only")
        elif self.goal == "reduce_sugar":
            if p.percent is None or p.allergen is not None:
                raise ValueError("goal reduce_sugar requires goal_params.percent (10-50) only")
        elif p.allergen is not None or p.percent is not None:
            raise ValueError("goal make_vegan takes no goal_params")
        return self


class Substitution(BaseModel):
    original: str
    replacement: str
    grams: float = Field(ge=0)
    rationale: str
    # doc_id from the knowledge base, or "off:<barcode>" for an Open Food Facts product.
    sources: list[str] = []
    confidence: Confidence


class NutritionPer100g(BaseModel):
    kcal: float
    protein_g: float
    fat_g: float
    carbs_g: float
    sugar_g: float


class NutritionBeforeAfter(BaseModel):
    before: NutritionPer100g
    after: NutritionPer100g


class TraceStep(BaseModel):
    step: int
    iteration: int
    type: Literal["llm_call", "tool_call", "loop_guard", "validation_error", "correction", "error"]
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    # Tool result as the model saw it, with long strings (chunk texts) shortened for readability.
    result: Any = None
    message: str | None = None
    duration_ms: int | None = None
    # llm_call: provider, http_attempts, input/output/reasoning/total tokens, and at_ms (start,
    # ms since the run began) - what a tokens-per-minute limit sees.
    usage: dict[str, Any] | None = None


class ReformulationAnswer(BaseModel):
    """What the model must return as its final answer; the loop adds the trace."""

    substitutions: list[Substitution]
    allergens_before: list[str]
    allergens_after: list[str]
    nutrition_per_100g: NutritionBeforeAfter
    warnings: list[str] = []

    @model_validator(mode="after")
    def unsourced_means_low_confidence(self) -> "ReformulationAnswer":
        for sub in self.substitutions:
            if not sub.sources and sub.confidence != "low":
                raise ValueError(
                    f"substitution {sub.original!r} has no sources, so confidence must be 'low'"
                )
        if any(not sub.sources for sub in self.substitutions) and not self.warnings:
            raise ValueError("a substitution without sources must be explained in warnings")
        return self


class ReformulateOut(ReformulationAnswer):
    trace: list[TraceStep] = []
