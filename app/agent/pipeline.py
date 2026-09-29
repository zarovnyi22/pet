"""Fixed reformulation pipeline: the production path of POST /reformulate.

The fallback from CLAUDE.md. The free tool-calling loop (app/agent/loop.py) did not
stabilize within 6 iterations: every number the model carried by hand could be wrong, and
every check that caught it cost another iteration. Here code calls the tools in a fixed
order and the LLM does only what needs language:

  1. LLM: plan - English names, ingredient -> spec mapping, candidate replacements, queries.
  2. code: spec nutrient tables, lookup_product for the rest, search_knowledge_base for context.
  3. LLM: choose - nutrient sources and substitutions, from the given candidates only.
  4. code: before/after recipes, calc_nutrition, the reduce_sugar dose, checks, warnings.

Two LLM calls (plus at most one retry each), and the model never supplies a number.
"""

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, ValidationError

from app.agent.common import (
    AgentError,
    OutOfTime,
    bounded,
    cap_confidence,
    format_validation,
    log_step,
    shorten,
)
from app.agent.prompts import CHOOSE_PROMPT, PLAN_PROMPT
from app.agent.tools import Toolbox
from app.llm.base import LLMClient, LLMError, Message
from app.retrieval import doc_allergens, doc_nutrients, spec_catalog
from app.schemas import (
    Confidence,
    EUAllergen,
    NutritionBeforeAfter,
    NutritionPer100g,
    ReformulateIn,
    ReformulateOut,
    Substitution,
    TraceStep,
)

# Aim one point past the requested sugar cut, so rounding grams can never miss the goal.
SUGAR_MARGIN_POINTS = 1.0
PROTEIN_DROP_WARNING = 0.20
EXCERPT_CHARS = 400

# Common non-canonical names models use for the 14 EU allergens.
ALLERGEN_ALIASES = {
    "soy": "soybeans",
    "soya": "soybeans",
    "egg": "eggs",
    "peanut": "peanuts",
    "nut": "nuts",
    "tree nuts": "nuts",
    "sulfites": "sulphites",
    "sulphur dioxide": "sulphites",
    "crustacean": "crustaceans",
    "mollusc": "molluscs",
    "dairy": "milk",
    "lactose": "milk",
    "wheat": "gluten",
    "oats": "gluten",
}


def _canonical_allergens(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    names = [ALLERGEN_ALIASES.get(str(v).strip().lower(), str(v).strip().lower()) for v in value]
    return list(dict.fromkeys(names))


Allergens = Annotated[list[EUAllergen], BeforeValidator(_canonical_allergens)]


# --- step 1: plan ---------------------------------------------------------------------------


class PlannedIngredient(BaseModel):
    name: str
    english: str
    spec: str | None = None


class Candidate(BaseModel):
    english: str
    spec: str | None = None


class Plan(BaseModel):
    ingredients: list[PlannedIngredient]
    candidates: list[Candidate] = Field(default=[], max_length=6)
    queries: list[str] = Field(default=[], max_length=3)


# --- step 3: choice -------------------------------------------------------------------------


class OriginalNutrients(BaseModel):
    name: str
    nutrients_source: str | None = None
    nutrients_column: str | None = None


class ChosenSubstitution(BaseModel):
    original: str
    replacement: str
    grams: float = Field(gt=0)
    nutrients_source: str
    nutrients_column: str | None = None
    sources: list[str] = []
    rationale: str
    confidence: Confidence


class Choice(BaseModel):
    original_nutrients: list[OriginalNutrients]
    substitutions: list[ChosenSubstitution] = Field(min_length=1)
    allergens_before: Allergens
    allergens_after: Allergens
    warnings: list[str] = []


CHOICE_SCHEMA = json.dumps(Choice.model_json_schema(), separators=(",", ":"))


class _Computed(BaseModel):
    """A choice that passed every check, with the nutrition code computed for it."""

    choice: Choice
    before: dict[str, Any]
    after: dict[str, Any]


class ReformulationPipeline:
    def __init__(self, llm: LLMClient, tools: Toolbox, *, timeout_seconds: float) -> None:
        self.llm = llm
        self.tools = tools
        self.timeout_seconds = timeout_seconds
        self.trace: list[TraceStep] = []
        self._stage = 0
        self._deadline = 0.0

    async def run(self, request: ReformulateIn) -> ReformulateOut:
        self._deadline = time.monotonic() + self.timeout_seconds
        try:
            return await self._run(request)
        except OutOfTime:
            message = f"no answer within {self.timeout_seconds:g} s"
            self._record("error", message=message)
            raise AgentError(504, "agent_timeout", message, self.trace) from None
        except LLMError as exc:
            self._record("error", message=f"{exc.code}: {exc.message}")
            raise AgentError(exc.status_code, exc.code, exc.message, self.trace) from None

    async def _run(self, request: ReformulateIn) -> ReformulateOut:
        self._stage = 1
        catalog = dict(await self._bounded(spec_catalog(self.tools.pool)))
        plan = await self._ask(
            "plan",
            PLAN_PROMPT,
            _plan_input(request, catalog),
            lambda data: self._parse_plan(data, request, catalog),
        )

        self._stage = 2
        candidates = await self._gather(plan, catalog)

        self._stage = 3
        computed = await self._ask(
            "choose",
            f"{CHOOSE_PROMPT}\n\nRESPONSE SCHEMA:\n{CHOICE_SCHEMA}",
            _choose_input(request, plan, candidates),
            lambda data: self._compute(data, request),
        )

        self._stage = 4
        return self._finish(computed)

    # --- LLM steps --------------------------------------------------------------------------

    async def _ask[T](
        self, purpose: str, system: str, user: str, parse: Callable[[Any], Awaitable[T]]
    ) -> T:
        """One JSON completion, checked by `parse`; one retry with the error, as in the loop."""
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        for attempt in (1, 2):
            started = time.monotonic()
            text = await self._bounded(self.llm.complete(messages, json_mode=True))
            self._record(
                "llm_call",
                tool=purpose,
                result=shorten(_parse_json_or_text(text)),
                duration_ms=int((time.monotonic() - started) * 1000),
            )
            try:
                return await parse(_parse_json(text))
            except ValueError as exc:  # includes JSONDecodeError and pydantic's ValidationError
                error = format_validation(exc) if isinstance(exc, ValidationError) else str(exc)
                self._record("validation_error", tool=purpose, message=error)
                if attempt == 2:
                    raise AgentError(
                        502,
                        "agent_invalid_output",
                        f"invalid {purpose} answer: {error}",
                        self.trace,
                    ) from None
                messages += [
                    Message(role="assistant", content=text),
                    Message(
                        role="user",
                        content=f"Your answer is invalid: {error}\n"
                        "Return only the corrected JSON object.",
                    ),
                ]
        raise AssertionError("unreachable")

    async def _parse_plan(self, data: Any, request: ReformulateIn, catalog: dict[str, str]) -> Plan:
        plan = Plan.model_validate(data)
        planned = {p.name for p in plan.ingredients}
        missing = [i.name for i in request.ingredients if i.name not in planned]
        if missing:
            raise ValueError(f"ingredients missing from the plan: {missing}; use the recipe names")
        for item in [*plan.ingredients, *plan.candidates]:
            if item.spec and item.spec not in catalog:
                self._record("correction", message=f"{item.spec!r} is not in the catalog: ignored")
                item.spec = None
        return plan

    # --- step 2: code gathers the data ------------------------------------------------------

    async def _gather(self, plan: Plan, catalog: dict[str, str]) -> str:
        """Fetch everything the choice may cite; returns it as the CANDIDATE SOURCES text."""
        items = [*plan.ingredients, *plan.candidates]
        spec_ids = sorted({i.spec for i in items if i.spec})
        tables = await self._bounded(doc_nutrients(self.tools.pool, spec_ids))
        for doc_id, table in tables.items():
            self.tools.remember_table(doc_id, table)
        self._record(
            "tool_call", tool="spec_tables", arguments={"doc_ids": spec_ids}, result=tables
        )

        unmapped = list(dict.fromkeys(i.english for i in items if not i.spec))
        calls = [("lookup_product", {"name": name}) for name in unmapped]
        calls += [("search_knowledge_base", {"query": q}) for q in plan.queries]
        results = await self._tools(calls)

        lines = [
            f"{doc_id} | {catalog[doc_id]} | nutrients per 100 g: {_flat(table)}"
            for doc_id, table in tables.items()
        ]
        for (name, _), result in zip(calls, results, strict=True):
            if name == "lookup_product":
                for p in result.get("products", []):
                    lines.append(
                        f"{p['source']} | {p['product_name']} (Open Food Facts) | nutrients per "
                        f"100 g: {p['nutrients_per_100g']} | allergens: {p['allergens']} | "
                        f"vegan: {p['vegan']}"
                    )
            else:
                attached = result.get("nutrients_per_100g", {})
                for chunk in result.get("chunks", []):
                    table = (
                        f" | nutrients per 100 g: {attached[chunk['doc_id']]}"
                        if (chunk["doc_id"] in attached)
                        else ""
                    )
                    excerpt = " ".join(chunk["chunk_text"].split())[:EXCERPT_CHARS]
                    lines.append(
                        f"{chunk['doc_id']} | {chunk['title']}{table} | excerpt: {excerpt}"
                    )
        return "\n".join(dict.fromkeys(lines))

    # --- step 4: code computes and checks ---------------------------------------------------

    async def _compute(self, data: Any, request: ReformulateIn) -> _Computed:
        choice = Choice.model_validate(data)
        recipe = {i.name: i.grams for i in request.ingredients}

        originals: dict[str, tuple[str, dict[str, float]]] = {}
        for o in choice.original_nutrients:
            if o.name not in recipe:
                raise ValueError(f"original_nutrients: {o.name!r} is not a recipe ingredient")
            if o.nutrients_source:
                nutrients = self.tools.nutrients_of(o.nutrients_source, o.nutrients_column)
                originals[o.name] = (o.nutrients_source, nutrients)

        replacements: list[tuple[ChosenSubstitution, dict[str, float]]] = []
        for sub in choice.substitutions:
            if sub.original not in recipe:
                raise ValueError(f"substitution: {sub.original!r} is not a recipe ingredient")
            nutrients = self.tools.nutrients_of(sub.nutrients_source, sub.nutrients_column)
            replacements.append((sub, nutrients))
        for name, grams in recipe.items():
            replaced = sum(s.grams for s in choice.substitutions if s.original == name)
            if replaced > grams + 0.01:
                raise ValueError(
                    f"substitutions replace {replaced} g of {name!r}, recipe has {grams}"
                )

        self._fix_sources(choice)
        if (
            request.goal == "remove_allergen"
            and request.goal_params.allergen in choice.allergens_after
        ):
            raise ValueError(
                f"allergens_after still contains {request.goal_params.allergen!r}: replace every "
                "ingredient that contains it"
            )

        before_items = _recipe_items(recipe, originals, [])
        before = await self._calc(before_items)
        if request.goal == "reduce_sugar":
            self._solve_sugar_dose(recipe, originals, replacements, before, request)
        after_items = _recipe_items(recipe, originals, replacements)
        after = await self._calc(after_items)

        if request.goal == "reduce_sugar":
            target = _sugar_target(before, request.goal_params.percent, margin=0)
            if after["per_100g"]["sugar_g"] > target + 0.01:  # only if sugar data is missing
                raise ValueError(
                    f"sugar_g after is {after['per_100g']['sugar_g']}, the goal needs at most "
                    f"{target}; substitute a sugary ingredient whose sugar is known"
                )
        await self._check_allergens(choice, request, before_items, after_items)
        return _Computed(choice=choice, before=before, after=after)

    async def _check_allergens(
        self,
        choice: Choice,
        request: ReformulateIn,
        before_items: list[dict[str, Any]],
        after_items: list[dict[str, Any]],
    ) -> None:
        """Hold the model's allergen lists to what the sources say about each ingredient."""
        sources = {i["nutrients_source"] for i in before_items + after_items} - {None}
        docs = sorted(s for s in sources if not s.startswith("off:"))
        facts = await self._bounded(doc_allergens(self.tools.pool, docs))
        facts |= {s: f for s in sources if (f := self.tools.product_facts(s))}

        def by_allergen(items: list[dict[str, Any]]) -> dict[str, list[str]]:
            found: dict[str, list[str]] = {}
            for item in items:
                for allergen in facts.get(item["nutrients_source"], {}).get("allergens", []):
                    found.setdefault(allergen, []).append(item["name"])
            return found

        data_before, data_after = by_allergen(before_items), by_allergen(after_items)
        target = request.goal_params.allergen
        if request.goal == "remove_allergen" and target in data_after:
            raise ValueError(
                f"{data_after[target]} still contain {target!r} according to their sources: "
                "replace them too"
            )
        if request.goal == "make_vegan":
            non_vegan = [
                i["name"]
                for i in after_items
                if facts.get(i["nutrients_source"], {}).get("vegan") is False
            ]
            if non_vegan:
                raise ValueError(
                    f"{non_vegan} are not vegan according to their sources: replace them too"
                )
        # Allergens the sources state but the model left out are added, not retried.
        for field, data in (("allergens_before", data_before), ("allergens_after", data_after)):
            listed = getattr(choice, field)
            added = [a for a in sorted(data) if a not in listed]
            if added:
                listed.extend(added)
                self._record(
                    "correction",
                    message=f"{field}: added {added} stated by the sources of "
                    f"{sorted({n for a in added for n in data[a]})}",
                )

    def _solve_sugar_dose(
        self,
        recipe: dict[str, float],
        originals: dict[str, tuple[str, dict[str, float]]],
        replacements: list[tuple[ChosenSubstitution, dict[str, float]]],
        before: dict[str, Any],
        request: ReformulateIn,
    ) -> None:
        """Set the grams of the added-sugar replacement so sugar per 100 g meets the goal.

        With mass compensation, replacing R grams of the sugar ingredient (s0 g sugar/100 g)
        by the chosen blend (s1 g/100 g) lowers total sugar linearly: S(R) = S(0) - R(s0-s1)/100.
        Solving S(R) = target gives the dose directly, lactose and fruit sugar included,
        instead of the model's trial and error.
        """
        sugary = [
            name
            for name in recipe
            if name in originals and any(s.original == name for s, _ in replacements)
        ]
        if not sugary:
            raise ValueError(
                "reduce_sugar: substitute part of the added sugar, and give its original_nutrients"
            )
        name = max(sugary, key=lambda n: originals[n][1].get("sugar_g", 0))
        s0 = originals[name][1].get("sugar_g", 0)
        blend = [(s, n) for s, n in replacements if s.original == name]
        if any("sugar_g" not in n for _, n in blend):
            raise ValueError(f"reduce_sugar: a replacement for {name!r} has no sugar value")
        proposed = sum(s.grams for s, _ in blend)
        s1 = sum(s.grams / proposed * n["sugar_g"] for s, n in blend)
        if s1 >= s0:
            raise ValueError(f"reduce_sugar: the replacements for {name!r} are not less sugary")

        in_blend = {id(s) for s, _ in blend}
        others = [(s, n) for s, n in replacements if id(s) not in in_blend]
        at_zero = _recipe_items(recipe, originals, others)  # the sugar ingredient kept whole
        total_mass = sum(recipe.values())
        sugar_at_zero = sum(
            i["grams"] * i["nutrients_per_100g"].get("sugar_g", 0) / 100 for i in at_zero
        )
        target_total = _sugar_target(before, request.goal_params.percent) * total_mass / 100
        dose = max(0.0, math.ceil((sugar_at_zero - target_total) * 100 / (s0 - s1) * 10) / 10)
        if dose > recipe[name]:
            raise ValueError(
                f"reduce_sugar: even replacing all {recipe[name]} g of {name!r} does not reach "
                f"the goal; also substitute another sugary ingredient or a less sugary replacement"
            )
        for sub, _ in blend:
            sub.grams = round(dose * sub.grams / proposed, 1)
        self._record(
            "correction",
            message=f"sugar dose computed by code: {dose} g of {name!r} replaced "
            f"(the model proposed {proposed} g)",
        )

    def _fix_sources(self, choice: Choice) -> None:
        # Cheap fixes made in place instead of costing a retry.
        seen = self.tools.seen_sources
        for sub in choice.substitutions:
            unknown = [s for s in sub.sources if s not in seen]  # incl. invented trial ids
            if unknown:
                sub.sources = [s for s in sub.sources if s not in unknown]
                self._record("correction", message=f"dropped unseen sources {unknown}")
            if sub.nutrients_source not in sub.sources:
                sub.sources.append(sub.nutrients_source)

    async def _calc(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        [result] = await self._tools([("calc_nutrition", {"ingredients": items})])
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def _finish(self, computed: _Computed) -> ReformulateOut:
        choice, before, after = computed.choice, computed.before, computed.after
        substitutions = [
            Substitution(
                original=s.original,
                replacement=s.replacement,
                grams=s.grams,
                rationale=s.rationale,
                sources=s.sources,
                confidence=s.confidence,
            )
            for s in choice.substitutions
        ]
        for message in cap_confidence(substitutions):
            self._record("correction", message=message)

        warnings = list(choice.warnings)
        p0, p1 = before["per_100g"]["protein_g"], after["per_100g"]["protein_g"]
        if p0 and (p0 - p1) / p0 > PROTEIN_DROP_WARNING:
            warnings.append(f"Protein drops from {p0} to {p1} g per 100 g ({(p0 - p1) / p0:.0%}).")
        for label, result in (("before", before), ("after", after)):
            for nutrient, names in result["missing"].items():
                warnings.append(f"No {nutrient} data for {', '.join(names)} ({label}).")

        return ReformulateOut(
            substitutions=substitutions,
            allergens_before=choice.allergens_before,
            allergens_after=choice.allergens_after,
            nutrition_per_100g=NutritionBeforeAfter(
                before=NutritionPer100g(**before["per_100g"]),
                after=NutritionPer100g(**after["per_100g"]),
            ),
            warnings=warnings,
            trace=self.trace,
        )

    # --- plumbing -------------------------------------------------------------------------

    async def _tools(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        async def timed(name: str, args: dict[str, Any]) -> tuple[dict[str, Any], int]:
            started = time.monotonic()
            result = await self.tools.execute(name, args)
            return result, int((time.monotonic() - started) * 1000)

        timed_results = await self._bounded(asyncio.gather(*(timed(n, a) for n, a in calls)))
        for (name, args), (result, ms) in zip(calls, timed_results, strict=True):
            self._record(
                "tool_call", tool=name, arguments=args, result=shorten(result), duration_ms=ms
            )
        return [result for result, _ in timed_results]

    async def _bounded[T](self, awaitable: Awaitable[T]) -> T:
        return await bounded(awaitable, self._deadline)

    def _record(self, type_: str, **fields: Any) -> None:
        self.trace.append(
            TraceStep(step=len(self.trace) + 1, iteration=self._stage, type=type_, **fields)
        )
        log_step(self.trace[-1])


def _plan_input(request: ReformulateIn, catalog: dict[str, str]) -> str:
    specs = "\n".join(f"{doc_id}: {title}" for doc_id, title in catalog.items())
    return f"RECIPE:\n{request.model_dump_json()}\n\nSPEC CATALOG:\n{specs}"


def _choose_input(request: ReformulateIn, plan: Plan, candidates: str) -> str:
    mapping = "\n".join(
        f"{p.name} = {p.english} -> {p.spec or 'no spec'}" for p in plan.ingredients
    )
    return (
        f"RECIPE:\n{request.model_dump_json()}\n\nINGREDIENTS:\n{mapping}\n\n"
        f"CANDIDATE SOURCES (cite only these ids):\n{candidates}"
    )


def _recipe_items(
    recipe: dict[str, float],
    originals: dict[str, tuple[str, dict[str, float]]],
    replacements: list[tuple[ChosenSubstitution, dict[str, float]]],
) -> list[dict[str, Any]]:
    """calc_nutrition ingredients: what is left of each original, then every replacement.

    Nutrients come from the sources the choice named, fetched by code - never typed by the
    model - so calc_nutrition's provenance check always passes."""
    items = []
    for name, grams in recipe.items():
        rest = grams - sum(s.grams for s, _ in replacements if s.original == name)
        if rest > 0.01:
            source, nutrients = originals.get(name, (None, {}))
            items.append(_item(name, rest, source, nutrients))
    for sub, nutrients in replacements:
        if sub.grams > 0:
            items.append(_item(sub.replacement, sub.grams, sub.nutrients_source, nutrients))
    return items


def _item(name: str, grams: float, source: str | None, nutrients: dict[str, float]) -> dict:
    return {
        "name": name,
        "grams": round(grams, 2),
        "nutrients_per_100g": dict(nutrients),
        "nutrients_source": source,
    }


def _sugar_target(
    before: dict[str, Any], percent: int, margin: float = SUGAR_MARGIN_POINTS
) -> float:
    return before["per_100g"]["sugar_g"] * (1 - (percent + margin) / 100)


def _flat(table: dict[str, dict[str, float]]) -> dict[str, Any]:
    return next(iter(table.values())) if len(table) == 1 else table


def _parse_json(text: str) -> Any:
    raw = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def _parse_json_or_text(text: str) -> Any:
    try:
        return _parse_json(text)
    except ValueError:
        return text
