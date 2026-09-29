import json
import logging
import time
from typing import Any

import asyncpg
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.agent.common import AgentError
from app.agent.pipeline import ReformulationPipeline
from app.agent.tools import Toolbox
from app.config import get_settings
from app.schemas import ReformulateIn, ReformulateOut, TraceStep

router = APIRouter()
logger = logging.getLogger(__name__)

AGENT_ERRORS = {
    502: {"description": "LLM failure or invalid final answer after one retry; includes trace"},
    504: {"description": "agent_timeout: time limit exceeded; includes trace"},
}


@router.post("/reformulate", response_model=ReformulateOut, responses=AGENT_ERRORS)
async def reformulate(body: ReformulateIn, request: Request, response: Response) -> Any:
    state = request.app.state
    settings = get_settings()
    # The fixed pipeline, not the free AgentLoop: see app/agent/pipeline.py and the README.
    agent = ReformulationPipeline(
        state.llm,
        Toolbox(state.pool, state.embedder, state.http),
        timeout_seconds=settings.agent_timeout_seconds,
    )
    started = time.monotonic()
    try:
        out = await agent.run(body)
    except AgentError as exc:
        run_id = await save_run(state.pool, body, None, exc.trace, exc.code, started)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": exc.code, "message": exc.message},
                "trace": [t.model_dump(mode="json") for t in exc.trace],
            },
            headers=run_id_header(run_id),
        )
    except Exception:
        # A bug, not an agent outcome: still log the run so the trace is not lost.
        await save_run(state.pool, body, None, agent.trace, "internal_error", started)
        raise

    answer = out.model_dump(mode="json", exclude={"trace"})
    run_id = await save_run(state.pool, body, answer, out.trace, "ok", started)
    response.headers.update(run_id_header(run_id))
    return out


def run_id_header(run_id: int | None) -> dict[str, str]:
    return {"X-Run-Id": str(run_id)} if run_id is not None else {}


async def save_run(
    pool: asyncpg.Pool,
    request: ReformulateIn,
    answer: dict[str, Any] | None,
    trace: list[TraceStep],
    status: str,
    started: float,
) -> int | None:
    """Log the run to reformulation_runs. A logging failure must not cost the user the answer."""
    try:
        return await pool.fetchval(
            """
            INSERT INTO reformulation_runs (request, response, trace, status, duration_ms)
            VALUES ($1::jsonb, $2::jsonb, $3::jsonb, $4, $5)
            RETURNING id
            """,
            request.model_dump_json(),
            json.dumps(answer, ensure_ascii=False) if answer is not None else None,
            json.dumps([t.model_dump(mode="json") for t in trace], ensure_ascii=False),
            status,
            int((time.monotonic() - started) * 1000),
        )
    except Exception:
        logger.exception("failed to save reformulation run")
        return None
    finally:
        logger.info(
            "reformulation run",
            extra={"status": status, "goal": request.goal, "trace_steps": len(trace)},
        )
