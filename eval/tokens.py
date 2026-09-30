"""Token usage of recent /reformulate runs, from the trace in reformulation_runs.

Per run: provider, goal, LLM calls (retries after invalid JSON are calls of their own; HTTP
retries show in "HTTP"), input / output / total tokens (output includes reasoning, shown
separately), and the peak tokens in any 60 s window - what a tokens-per-minute limit sees.
Runs in the api container (./eval is mounted there):

    make tokens                 # the last 6 runs
    make tokens ARGS="--last 3"
"""

import argparse
import asyncio
import json
import sys
from datetime import timedelta
from typing import Any

from app.config import get_settings
from app.db import create_pool

WINDOW_MS = 60_000


def peak_in_window(events: list[tuple[float, int]]) -> int:
    """Largest token sum over any WINDOW_MS window; events are (start ms, tokens)."""
    events = sorted(events)
    return max(
        (sum(t for at, t in events if start <= at < start + WINDOW_MS) for start, _ in events),
        default=0,
    )


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    trace = json.loads(row["trace"])
    calls = [s for s in trace if s["type"] == "llm_call"]
    usage = [c["usage"] for c in calls if c.get("usage")]

    def total(field: str) -> int:
        return sum(u.get(field, 0) for u in usage)

    return {
        "id": row["id"],
        "provider": ",".join(dict.fromkeys(u["provider"] for u in usage if u.get("provider")))
        or "-",
        "goal": json.loads(row["request"])["goal"],
        "status": row["status"],
        "calls": len(calls),
        "http": total("http_attempts"),
        "input": total("input_tokens"),
        "output": total("output_tokens"),
        "reasoning": total("reasoning_tokens"),
        "total": total("total_tokens"),
        "peak": peak_in_window([(u.get("at_ms", 0), u.get("total_tokens", 0)) for u in usage]),
        "measured": bool(usage),
        # For the peak across runs: when each call started, in wall-clock ms.
        "events": [
            (
                (row["created_at"] - timedelta(milliseconds=row["duration_ms"] or 0)).timestamp()
                * 1000
                + u.get("at_ms", 0),
                u.get("total_tokens", 0),
            )
            for u in usage
        ],
    }


async def main(last: int) -> int:
    pool = await create_pool(get_settings().database_url)
    try:
        rows = await pool.fetch(
            "SELECT id, request, trace, status, duration_ms, created_at FROM reformulation_runs "
            "ORDER BY id DESC LIMIT $1",
            last,
        )
    finally:
        await pool.close()
    if not rows:
        print("no runs in reformulation_runs yet", file=sys.stderr)
        return 1

    runs = [summarize(dict(r)) for r in reversed(rows)]
    print(
        "| run | provider | goal | status | LLM calls | HTTP | input | output (reasoning) "
        "| total | peak / 60 s |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        if not r["measured"]:
            print(
                f"| {r['id']} | - | {r['goal']} | {r['status']} | {r['calls']} | "
                "no usage recorded (run before token logging) | | | | |"
            )
            continue
        print(
            f"| {r['id']} | {r['provider']} | {r['goal']} | {r['status']} | {r['calls']} | "
            f"{r['http']} | {r['input']} | {r['output']} ({r['reasoning']}) | {r['total']} | "
            f"{r['peak']} |"
        )
    measured = [r for r in runs if r["measured"]]
    if measured:
        events = [e for r in measured for e in r["events"]]
        print(f"\npeak over any 60 s across these runs: {peak_in_window(events)} tokens")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--last", type=int, default=6, help="how many recent runs (default 6)")
    sys.exit(asyncio.run(main(parser.parse_args().last)))
