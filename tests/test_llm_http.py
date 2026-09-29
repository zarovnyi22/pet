import httpx
import pytest

from app.llm import base
from app.llm.base import LLMError, post_json


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch):
    monkeypatch.setattr(base, "RETRY_DELAY_SECONDS", 0)


async def post_with_status(status: int, body: str = "{}") -> dict:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    async with httpx.AsyncClient(transport=transport) as http:
        return await post_json(http, "gemini", "https://llm.test", {}, {})


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
            return await post_json(http, "gemini", "https://llm.test", {}, {}), len(calls)
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
