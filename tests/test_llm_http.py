import httpx
import pytest

from app.llm import base
from app.llm.base import LLMError, Message, post_json
from app.llm.gemini import GeminiClient
from app.llm.groq import GroqClient


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(base, "RETRY_DELAY_SECONDS", 0)


async def post_with_status(status: int, body: str = "{}") -> dict:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    async with httpx.AsyncClient(transport=transport) as http:
        return await post_json(http, "gemini", "GEMINI_API_KEY", "https://llm.test", {}, {})


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
            return await post_json(
                http, "gemini", "GEMINI_API_KEY", "https://llm.test", {}, {}
            ), len(calls)
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
async def test_transient_failure_is_retried_once(first):
    result, calls = await post_sequence(first, 200)
    assert (result, calls) == ({"ok": True}, 2)


async def test_second_transient_failure_gives_up():
    result, calls = await post_sequence(503, 503)
    assert (result.code, calls) == ("llm_unavailable", 2)


async def test_rate_limit_is_not_retried():
    result, calls = await post_sequence(429, 200)
    assert (result.code, calls) == ("llm_rate_limited", 1)


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
