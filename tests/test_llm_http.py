import json

import httpx
import pytest

from app.llm import base, gemini, groq
from app.llm.base import (
    Endpoint,
    LLMError,
    Message,
    post_json,
    summarize_attempts,
    track_attempts,
)
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient

ENDPOINT = Endpoint("gemini", "GEMINI_API_KEY", "https://llm.test", {}, gemini.usage)


@pytest.fixture(autouse=True)
def pauses(monkeypatch) -> list[float]:
    """Retry pauses are recorded instead of slept; no jitter, so they are exact."""
    slept = []

    async def sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(base, "_sleep", sleep)
    monkeypatch.setattr(base, "RETRY_JITTER_SECONDS", 0)
    return slept


async def post_with_status(status: int, body: str = "{}") -> dict:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    async with httpx.AsyncClient(transport=transport) as http:
        return await post_json(http, ENDPOINT, {})


async def post_sequence(*outcomes) -> tuple[dict | LLMError, int]:
    """Provider answers with these outcomes in turn (a status code or an exception)."""
    calls = []

    def handler(request):
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, text='{"ok": true}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        try:
            return await post_json(http, ENDPOINT, {}), len(calls)
        except LLMError as exc:
            return exc, len(calls)


@pytest.mark.parametrize(
    ("status", "code", "http_status"),
    [
        (429, "llm_rate_limited", 503),
        (503, "llm_unavailable", 503),  # "model is currently experiencing high demand"
        (400, "llm_error", 502),
        (500, "llm_error", 502),
    ],
)
async def test_provider_errors_map_to_llm_error_codes(status, code, http_status):
    with pytest.raises(LLMError) as err:
        await post_with_status(status, '{"error": {"message": "high demand"}}')
    assert (err.value.code, err.value.status_code) == (code, http_status)


async def test_provider_success_returns_json():
    assert await post_with_status(200, '{"ok": true}') == {"ok": True}


@pytest.mark.parametrize("first", [503, httpx.ReadTimeout("hung")], ids=["overloaded", "timeout"])
async def test_transient_failure_is_retried(first):
    result, calls = await post_sequence(first, 200)
    assert (result, calls) == ({"ok": True}, 2)


async def test_retries_back_off_until_the_provider_recovers(pauses):
    result, calls = await post_sequence(503, 503, 200)
    assert (result, calls, pauses) == ({"ok": True}, 3, [2.0, 4.0])


async def test_persistent_overload_gives_up_after_four_retries(pauses):
    result, calls = await post_sequence(503, 503, 503, 503, 503)
    assert (result.code, calls, pauses) == ("llm_unavailable", 5, [2.0, 4.0, 8.0, 16.0])


async def test_retries_stop_at_the_time_budget(monkeypatch, pauses):
    monkeypatch.setattr(base, "RETRY_BUDGET_SECONDS", 3)  # room for the 2 s pause, not the 4 s
    result, calls = await post_sequence(503, 503, 200)
    assert (result.code, calls, pauses) == ("llm_unavailable", 2, [2.0])


async def test_jitter_is_added_to_each_pause(monkeypatch, pauses):
    monkeypatch.setattr(base, "RETRY_JITTER_SECONDS", 1.0)
    await post_sequence(503, 503, 200)
    assert 2.0 <= pauses[0] <= 3.0 and 4.0 <= pauses[1] <= 5.0


async def test_unreachable_host_is_unavailable_and_retried():
    result, calls = await post_sequence(httpx.ConnectError("refused"), 200)
    assert (result, calls) == ({"ok": True}, 2)
    result, calls = await post_sequence(*[httpx.ConnectError("refused")] * 5)
    assert (result.code, result.status_code, calls) == ("llm_unavailable", 503, 5)


@pytest.mark.parametrize("status", [429, 400, 404, 500])
async def test_rate_limit_and_other_errors_are_not_retried(status, pauses):
    result, calls = await post_sequence(status, 200)
    assert (calls, pauses) == (1, [])


# What each provider answers to a wrong key (bodies trimmed from real responses).
GEMINI_BAD_KEY = (
    400,
    '{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", '
    '"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}',
)
GROQ_BAD_KEY = (
    401,
    '{"error": {"message": "Invalid API Key", "type": "invalid_request_error", '
    '"code": "invalid_api_key"}}',
)


@pytest.mark.parametrize(
    ("client", "answer", "env_var"),
    [
        (GeminiClient("wrong-key", "gemini-test", 5), GEMINI_BAD_KEY, "GEMINI_API_KEY"),
        (GroqClient("wrong-key", "groq-test", 5), GROQ_BAD_KEY, "GROQ_API_KEY"),
    ],
    ids=["gemini", "groq"],
)
async def test_invalid_key_is_503_llm_invalid_key_and_not_retried(client, answer, env_var):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(answer[0], text=answer[1])

    await client.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as err:
        await client.complete([Message(role="user", content="hi")])
    await client.aclose()

    assert (err.value.code, err.value.status_code) == ("llm_invalid_key", 503)
    assert err.value.message == f"{env_var} is invalid (LLM_PROVIDER={client.provider})"
    assert len(requests) == 1  # a wrong key does not fix itself: no retry


# --- token usage ------------------------------------------------------------------------------

# Trimmed from real responses: Gemini with thinking, Groq gpt-oss with reasoning tokens.
GEMINI_USAGE = {
    "usageMetadata": {
        "promptTokenCount": 3120,
        "candidatesTokenCount": 410,
        "thoughtsTokenCount": 250,
        "totalTokenCount": 3780,
    }
}
GROQ_USAGE = {
    "usage": {
        "prompt_tokens": 3300,
        "completion_tokens": 900,
        "total_tokens": 4200,
        "completion_tokens_details": {"reasoning_tokens": 520},
    }
}


def test_gemini_usage_counts_thoughts_as_output():
    assert gemini.usage(GEMINI_USAGE) == {
        "input_tokens": 3120,
        "output_tokens": 660,  # 410 answer + 250 thinking
        "reasoning_tokens": 250,
        "total_tokens": 3780,
    }
    no_thinking = {"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}
    assert gemini.usage(no_thinking) == {
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 0,
        "total_tokens": 15,
    }
    assert gemini.usage({}) == {}


def test_groq_usage_keeps_reasoning_inside_completion():
    assert groq.usage(GROQ_USAGE) == {
        "input_tokens": 3300,
        "output_tokens": 900,
        "reasoning_tokens": 520,
        "total_tokens": 4200,
    }
    plain = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    assert groq.usage(plain)["reasoning_tokens"] == 0
    assert groq.usage(plain)["total_tokens"] == 15
    assert groq.usage({}) == {}


async def test_every_http_attempt_is_tracked_with_its_tokens():
    answers = iter(
        [httpx.Response(503, text="high demand"), httpx.Response(200, json=GEMINI_USAGE)]
    )
    transport = httpx.MockTransport(lambda request: next(answers))
    async with httpx.AsyncClient(transport=transport) as http:
        with track_attempts() as attempts:
            await post_json(http, ENDPOINT, {})

    assert [a["status"] for a in attempts] == [503, 200]
    assert summarize_attempts(attempts) == {
        "provider": "gemini",
        "http_attempts": 2,  # the retry counts: it hits the provider's rate limits too
        "input_tokens": 3120,
        "output_tokens": 660,
        "reasoning_tokens": 250,
        "total_tokens": 3780,
    }


async def test_llm_request_log_carries_the_tokens(caplog):
    caplog.set_level("INFO", logger="app.llm")
    await post_with_status(200, json.dumps(GEMINI_USAGE))
    [record] = [r for r in caplog.records if r.getMessage() == "llm request"]
    assert (record.input_tokens, record.output_tokens, record.total_tokens) == (3120, 660, 3780)


async def test_attempts_are_not_collected_outside_a_tracker():
    await post_with_status(200, json.dumps(GEMINI_USAGE))  # no tracker: nothing to append to
    with track_attempts() as attempts:
        pass
    assert attempts == []


# --- Groq reasoning_effort ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "sent"),
    [
        ("openai/gpt-oss-120b", "low", "low"),
        ("openai/gpt-oss-120b", "", None),  # not set: the provider's default, as before
        ("qwen/qwen3.8-27b", "low", None),  # other models take other values: never sent
    ],
)
async def test_groq_sends_reasoning_effort_only_for_gpt_oss(model, effort, sent):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = GroqClient("key", model, 5, reasoning_effort=effort)
    await client.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await client.complete([Message(role="user", content="hi")]) == "ok"
    await client.aclose()
    assert bodies[0].get("reasoning_effort") == sent
