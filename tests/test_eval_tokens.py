"""The token summary script (eval/tokens.py) on a stored run: no database, no LLM."""

import json
from datetime import UTC, datetime

from eval.tokens import peak_in_window, summarize


def call(at_ms: int, total: int, provider: str = "groq") -> dict:
    return {
        "type": "llm_call",
        "usage": {
            "provider": provider,
            "http_attempts": 1,
            "input_tokens": total - 100,
            "output_tokens": 100,
            "reasoning_tokens": 40,
            "total_tokens": total,
            "at_ms": at_ms,
        },
    }


def test_peak_counts_only_calls_inside_one_60_s_window():
    assert peak_in_window([(0, 3000), (20_000, 2000), (70_000, 4000)]) == 6000  # 20 s + 70 s
    assert peak_in_window([(0, 3000), (59_999, 2000)]) == 5000
    assert peak_in_window([]) == 0


def test_run_summary_sums_calls_and_keeps_retries_visible():
    trace = [call(0, 3000), {"type": "validation_error"}, call(9000, 2500), call(15000, 2600)]
    row = {
        "id": 7,
        "request": json.dumps({"goal": "make_vegan"}),
        "trace": json.dumps(trace),
        "status": "ok",
        "duration_ms": 20000,
        "created_at": datetime(2026, 9, 30, tzinfo=UTC),
    }
    summary = summarize(row)

    assert {k: summary[k] for k in ("provider", "goal", "calls", "http", "input", "output")} == {
        "provider": "groq",
        "goal": "make_vegan",
        "calls": 3,  # the choose retry after invalid JSON is a call of its own
        "http": 3,
        "input": 7800,
        "output": 300,
    }
    assert (summary["reasoning"], summary["total"], summary["peak"]) == (120, 8100, 8100)


def test_run_without_usage_is_marked_unmeasured():
    row = {
        "id": 1,
        "request": json.dumps({"goal": "reduce_sugar"}),
        "trace": json.dumps([{"type": "llm_call"}]),
        "status": "ok",
        "duration_ms": 1000,
        "created_at": datetime(2026, 9, 30, tzinfo=UTC),
    }
    assert summarize(row)["measured"] is False
