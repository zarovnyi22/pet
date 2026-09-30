"""Live /reformulate runs: the demo yogurt (docs/SPEC.md), each goal N times, checked.

Runs against a started service with a real LLM key, so it is never part of the tests:

    make live-runs                       # 3 goals x 5 runs, 20 s pause
    make live-runs ARGS="--n 2 --pause 30"

A run succeeds only with HTTP 200 and every check of its goal passing; a failed check is
reported as the run's reason, never relaxed. Writes the tables to docs/live_runs.md and every
raw response to docs/live_runs/<goal>-<i>.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
MILK = "молоко 2.5%"
STARTER = "закваска"
STARTER_GRAMS = 10.0
DVS_MAX_GRAMS = 0.5  # spec-yogurt-starter: max 0.05% of the 1000 g product
YOGURT = {
    "product_name": "Полуничний йогурт 2.5%",
    "ingredients": [
        {"name": MILK, "grams": 800},
        {"name": "цукор", "grams": 90},
        {"name": "полуниця заморожена", "grams": 100},
        {"name": STARTER, "grams": STARTER_GRAMS},
    ],
}
GOALS = {
    "remove_allergen": {"allergen": "milk"},
    "reduce_sugar": {"percent": 30},
    "make_vegan": {},
}
HTTP_TIMEOUT = 150.0  # the run's own budget is 60 s; a fallback run can take ~40 s of it


# --- checks (pure, unit-tested) -----------------------------------------------------------------


def starter_problems(body: dict[str, Any]) -> list[str]:
    """The bulk starter (fermented milk) must be replaced: a DVS culture within its 0.05% limit
    plus a base, 10 g in total. The column is not in the answer, so it is read from the model's
    accepted choice in the trace (the last `choose` call); code keeps the starter grams as is."""
    subs = [s for s in body.get("substitutions", []) if s["original"] == STARTER]
    if not subs:
        return ["starter not replaced"]
    problems = []
    replaced = sum(s["grams"] for s in subs)
    if abs(replaced - STARTER_GRAMS) > 0.01:
        problems.append(f"starter subs total {replaced:g} g, not {STARTER_GRAMS:g}")
    choices = [
        t["result"]
        for t in body.get("trace", [])
        if t["type"] == "llm_call" and t.get("tool") == "choose" and isinstance(t["result"], dict)
    ]
    chosen = choices[-1].get("substitutions", []) if choices else []
    dvs = [
        s
        for s in chosen
        if s.get("original") == STARTER
        and s.get("nutrients_source") == "spec-yogurt-starter"
        and "DVS" in (s.get("nutrients_column") or "")
    ]
    if not dvs:
        problems.append("no DVS culture among the starter subs")
    elif (grams := sum(s["grams"] for s in dvs)) > DVS_MAX_GRAMS:
        problems.append(f"DVS {grams:g} g > {DVS_MAX_GRAMS:g} g")
    return problems


def check(goal: str, status: int, body: dict[str, Any]) -> list[str]:
    """Why this run failed; empty = success."""
    if status != 200:
        code = body.get("error", {}).get("code", "?")
        return [f"request failed: {code}" if status == 0 else f"HTTP {status} {code}"]
    problems = []
    if goal in ("remove_allergen", "make_vegan"):
        # Both goals must take the milk out: the milk base and the bulk starter (fermented milk).
        if "milk" in body["allergens_after"]:
            problems.append("milk in allergens_after")
        if not any(s["original"] == MILK for s in body.get("substitutions", [])):
            problems.append("milk not replaced")
        problems += starter_problems(body)
    if goal == "reduce_sugar":
        nutrition = body["nutrition_per_100g"]
        before, after = nutrition["before"]["sugar_g"], nutrition["after"]["sugar_g"]
        # 0.005: both values are rounded to 0.01 g in the answer.
        if after > before * 0.7 + 0.005:
            problems.append(f"sugar {before} -> {after} g, not -30%")
        if any(w.startswith("Sweetness rises") for w in body["warnings"]):
            problems.append("sweetness rises")
    return problems


def usage(body: dict[str, Any]) -> dict[str, Any]:
    """Provider(s), LLM calls (retries included) and tokens, from the trace."""
    calls = [t for t in body.get("trace", []) if t["type"] == "llm_call"]
    providers = [
        p for t in calls for p in ((t.get("usage") or {}).get("provider") or "").split(",")
    ]
    return {
        "provider": ",".join(dict.fromkeys(p for p in providers if p)) or "-",
        "calls": len(calls),
        "tokens": sum((t.get("usage") or {}).get("total_tokens", 0) for t in calls),
    }


# --- report -----------------------------------------------------------------------------------


def summary_table(runs: list[dict[str, Any]], n: int) -> str:
    rows = [
        "| Ціль | Успішних | Середня тривалість | Помилки | Закваску замінено | Провайдер "
        "| LLM-викликів (min–max) | Токенів у середньому |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for goal in GOALS:
        mine = [r for r in runs if r["goal"] == goal]
        ok = sum(not r["problems"] for r in mine)
        errors: dict[str, int] = {}
        for r in mine:
            for p in r["problems"]:
                errors[p] = errors.get(p, 0) + 1
        starter = (
            f"{sum(not starter_problems(r['body']) for r in mine if r['status'] == 200)}/{n}"
            if goal != "reduce_sugar"
            else "—"
        )
        calls = [r["calls"] for r in mine]
        rows.append(
            f"| {goal} | {ok}/{n} | {mean(r['seconds'] for r in mine):.1f} с | "
            f"{'; '.join(f'{k} ×{v}' for k, v in errors.items()) or '—'} | {starter} | "
            f"{', '.join(sorted({r['provider'] for r in mine}))} | "
            f"{mean(calls):.1f} ({min(calls)}–{max(calls)}) | "
            f"{mean(r['tokens'] for r in mine):.0f} |"
        )
    return "\n".join(rows)


def runs_table(runs: list[dict[str, Any]]) -> str:
    rows = [
        "| # | Ціль | HTTP | Результат | Тривалість | Провайдер | LLM-викликів | Токенів "
        "| run_id |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        result = "OK" if not r["problems"] else "FAIL: " + "; ".join(r["problems"])
        rows.append(
            f"| {r['i']} | {r['goal']} | {r['status']} | {result} | {r['seconds']:.1f} с | "
            f"{r['provider']} | {r['calls']} | {r['tokens']} | {r['run_id']} |"
        )
    return "\n".join(rows)


def main(base_url: str, n: int, pause: float) -> int:
    raw_dir = ROOT / "docs" / "live_runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_dir.glob("*.json"):
        old.unlink()
    total = n * len(GOALS)
    runs: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, timeout=HTTP_TIMEOUT) as http:
        health = http.get("/health").json()
        print(f"service: {health}", flush=True)
        for i in range(1, n + 1):
            for goal, params in GOALS.items():
                if runs:
                    time.sleep(pause)
                started = time.monotonic()
                try:
                    resp = http.post(
                        "/reformulate", json=YOGURT | {"goal": goal, "goal_params": params}
                    )
                    status, body = resp.status_code, resp.json()
                    run_id = resp.headers.get("X-Run-Id", "-")
                except (httpx.HTTPError, ValueError) as exc:
                    status, body, run_id = 0, {"error": {"code": type(exc).__name__}}, "-"
                seconds = time.monotonic() - started
                (raw_dir / f"{goal}-{i}.json").write_text(
                    json.dumps(body, ensure_ascii=False, indent=2)
                )
                run = {
                    "i": i,
                    "goal": goal,
                    "status": status,
                    "seconds": seconds,
                    "run_id": run_id,
                    "body": body,
                    "problems": check(goal, status, body),
                    **usage(body),
                }
                runs.append(run)
                verdict = "OK" if not run["problems"] else "FAIL " + "; ".join(run["problems"])
                print(
                    f"[{len(runs)}/{total}] {goal} #{i}: {verdict} | {seconds:.1f} s | "
                    f"{run['provider']} | {run['calls']} calls | {run['tokens']} tokens",
                    flush=True,
                )

    report = (
        f"# Живі прогони /reformulate\n\n{time.strftime('%Y-%m-%d %H:%M')}, {n} запусків на "
        f"ціль, пауза {pause:g} с, сервіс: `{json.dumps(health)}`.\n\n"
        f"{summary_table(runs, n)}\n\n## Усі запуски\n\n{runs_table(runs)}\n"
    )
    (ROOT / "docs" / "live_runs.md").write_text(report)
    print("\n" + summary_table(runs, n))
    print("\nwritten: docs/live_runs.md, docs/live_runs/*.json")
    return 0 if all(not r["problems"] for r in runs) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--n", type=int, default=5, help="runs per goal (default 5)")
    parser.add_argument("--pause", type=float, default=20, help="seconds between runs (20)")
    args = parser.parse_args()
    sys.exit(main(args.base_url, args.n, args.pause))
