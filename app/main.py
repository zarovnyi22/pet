import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.config import get_settings
from app.db import apply_migrations, check_db, create_pool
from app.embeddings import Embedder
from app.errors import AppError, error_response
from app.llm.base import get_llm_client
from app.routers import ask, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = await create_pool(settings.database_url)
    await apply_migrations(app.state.pool)
    app.state.embedder = await asyncio.to_thread(Embedder, settings.embedding_model)
    # A missing API key does not block startup: /health stays up, LLM calls return 503.
    app.state.llm = get_llm_client(settings)
    yield
    await app.state.llm.aclose()
    await app.state.pool.close()


app = FastAPI(title="Reformulation Assistant", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(ask.router)


@app.exception_handler(AppError)
async def app_error(request: Request, exc: AppError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    message = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
    )
    return error_response(422, "validation_error", message)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return error_response(exc.status_code, "http_error", str(exc.detail))


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    db = await check_db(request.app.state.pool)
    ok = db == "ok"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={
            "status": "ok" if ok else "degraded",
            "db": db,
            "llm_provider": get_settings().llm_provider,
        },
    )
