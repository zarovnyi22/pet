"""Proof, not a promise, that the suite needs no LLM keys and no internet."""

import os
import socket

import pytest

from app.config import get_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_NETWORK_ISOLATED") != "1",
    reason="only meaningful inside `docker compose run --rm test` (offline network)",
)


def test_no_llm_keys_in_the_environment():
    settings = get_settings()
    assert settings.gemini_api_key == "" and settings.groq_api_key == ""


@pytest.mark.parametrize(
    "host",
    ["generativelanguage.googleapis.com", "api.groq.com", "search.openfoodfacts.org"],
)
def test_external_apis_are_unreachable(host):
    # DNS failure or no route - either way, a test that tried a live call would fail loudly.
    with pytest.raises(OSError):
        socket.create_connection((host, 443), timeout=3).close()
