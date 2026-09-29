import httpx
import pytest

from app.llm.base import LLMError, post_json


async def post_with_status(status: int, body: str = "{}") -> dict:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=body))
    async with httpx.AsyncClient(transport=transport) as http:
        return await post_json(http, "gemini", "https://llm.test", {}, {})


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
