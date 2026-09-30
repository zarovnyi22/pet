"""FallbackLLMClient and retries inside /reformulate: FakeLLM and MockTransport, no network."""

import asyncio
import time

import httpx
import pytest
from pydantic import ValidationError

from app.agent.common import AgentError
from app.config import Settings
from app.llm import base
from app.llm.base import LLMError, Message, get_llm_client
from app.llm.fake import FakeLLM
from app.llm.fallback import FallbackLLMClient
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient
from tests.test_pipeline import choice, knowledge_base, plan, run  # noqa: F401

pytestmark = pytest.mark.usefixtures("knowledge_base")
HI = [Message(role="user", content="hi")]


class FailingLLM(FakeLLM):
    provider = "gemini"

    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code

    async def complete(self, messages, *, json_mode=False):
        self.calls.append(messages)
        raise LLMError("primary failed", code=self.code, status_code=503)


@pytest.fixture(autouse=True)
def no_retry_pauses(monkeypatch):
    async def sleep(seconds):
        pass

    monkeypatch.setattr(base, "_sleep", sleep)


@pytest.mark.parametrize("code", ["llm_unavailable", "llm_rate_limited"])
async def test_fallback_answers_when_the_primary_is_down(code, caplog):
    primary, backup = FailingLLM(code), FakeLLM(["from groq"])
    llm = FallbackLLMClient(primary, backup)

    assert await llm.complete(HI) == "from groq"
    [record] = [r for r in caplog.records if r.getMessage() == "llm fallback"]
    assert (record.provider, record.fallback, record.reason) == ("gemini", "fake", code)


@pytest.mark.parametrize("code", ["llm_error", "llm_invalid_key", "llm_not_configured"])
async def test_other_errors_are_not_hidden_by_the_fallback(code):
    backup = FakeLLM(["never"])
    with pytest.raises(LLMError) as err:
        await FallbackLLMClient(FailingLLM(code), backup).complete(HI)
    assert err.value.code == code and backup.calls == []


async def test_primary_is_skipped_while_cooling_down_then_tried_again():
    primary, backup = FailingLLM("llm_unavailable"), FakeLLM(["1", "2", "3"])
    llm = FallbackLLMClient(primary, backup)

    await llm.complete(HI)
    await llm.complete(HI)  # no second round of retries on a primary that just failed
    assert (len(primary.calls), len(backup.calls)) == (1, 2)

    llm._primary_down_until = time.monotonic() - 1  # PRIMARY_COOLDOWN_SECONDS have passed
    await llm.complete(HI)
    assert len(primary.calls) == 2


def test_settings_build_the_wrapper_only_when_asked():
    plain = Settings(_env_file=None, llm_provider="gemini")
    assert isinstance(get_llm_client(plain), GeminiClient)

    wrapped = get_llm_client(Settings(_env_file=None, llm_fallback_provider="groq"))
    assert isinstance(wrapped, FallbackLLMClient)
    assert isinstance(wrapped.primary, GeminiClient) and isinstance(wrapped.fallback, GroqClient)

    with pytest.raises(ValidationError, match="must differ"):
        Settings(_env_file=None, llm_provider="groq", llm_fallback_provider="groq")


# --- through the pipeline ---------------------------------------------------------------------


def client_with(cls, handler):
    llm = cls("key", "gpt-oss-test" if cls is GroqClient else "gemini-test", 5)
    llm._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return llm


async def test_reformulate_finishes_on_groq_when_gemini_is_overloaded():
    answers = iter([plan(), choice()])

    def groq_answer(request):
        usage = {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200}
        body = {"choices": [{"message": {"content": next(answers)}}], "usage": usage}
        return httpx.Response(200, json=body)

    gemini = client_with(GeminiClient, lambda r: httpx.Response(503, text="high demand"))
    llm = FallbackLLMClient(gemini, client_with(GroqClient, groq_answer))
    out = await run(llm)
    await llm.aclose()

    plan_call, choose_call = [s.usage for s in out.trace if s.type == "llm_call"]
    # Plan: Gemini's 5 attempts (4 retries), then Groq. Choose: straight to Groq (cooldown).
    assert (plan_call["provider"], plan_call["http_attempts"]) == ("gemini,groq", 6)
    assert (choose_call["provider"], choose_call["http_attempts"]) == ("groq", 1)
    assert choose_call["total_tokens"] == 1200


async def test_run_deadline_cuts_the_retries_short(monkeypatch):
    monkeypatch.setattr(base, "_sleep", asyncio.sleep)  # real pauses here
    monkeypatch.setattr(base, "RETRY_DELAYS_SECONDS", (1.0, 1.0, 1.0, 1.0))
    gemini = client_with(GeminiClient, lambda r: httpx.Response(503, text="high demand"))

    started = time.monotonic()
    with pytest.raises(AgentError) as err:
        await run(gemini, timeout=0.3)
    await gemini.aclose()

    assert (err.value.status_code, err.value.code) == (504, "agent_timeout")
    assert time.monotonic() - started < 1.0  # not the 4 s of retries


def test_groq_reasoning_effort_defaults_to_low_and_empty_turns_it_off():
    default = get_llm_client(Settings(_env_file=None, llm_provider="groq"))
    assert default._reasoning_effort == "low"
    off = Settings(_env_file=None, llm_provider="groq", groq_reasoning_effort="")
    assert get_llm_client(off)._reasoning_effort == ""


# --- the fallback runs out too (live: Groq's 8K tokens a minute, run 3 of a live batch) -------


class RateLimitedGroq(FailingLLM):
    provider = "groq"


def cooling_down(primary, backup) -> FallbackLLMClient:
    llm = FallbackLLMClient(primary, backup)
    llm._primary_down_until = time.monotonic() + 60  # the primary failed a moment ago
    return llm


async def test_rate_limited_fallback_during_cooldown_tries_the_primary_once():
    primary, backup = FakeLLM(["from gemini", "again gemini"]), RateLimitedGroq("llm_rate_limited")
    llm = cooling_down(primary, backup)

    assert await llm.complete(HI) == "from gemini"
    assert len(backup.calls) == 1 and len(primary.calls) == 1
    await llm.complete(HI)  # it answered: the cooldown is over, no detour through the fallback
    assert len(backup.calls) == 1 and len(primary.calls) == 2


async def test_both_exhausted_is_one_clear_error():
    primary, backup = FailingLLM("llm_unavailable"), RateLimitedGroq("llm_rate_limited")
    with pytest.raises(LLMError) as err:
        await cooling_down(primary, backup).complete(HI)

    assert (err.value.code, err.value.status_code) == ("llm_rate_limited", 503)
    assert err.value.message == "groq rate limited and gemini still failing (llm_unavailable)"
    assert len(primary.calls) == 1  # one try, not a loop


async def test_other_fallback_errors_during_cooldown_are_not_retried_on_the_primary():
    primary, backup = FakeLLM(["never"]), RateLimitedGroq("llm_invalid_key")
    with pytest.raises(LLMError) as err:
        await cooling_down(primary, backup).complete(HI)
    assert err.value.code == "llm_invalid_key" and primary.calls == []
