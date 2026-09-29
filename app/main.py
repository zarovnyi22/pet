import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.config import get_settings
from app.db import apply_migrations, check_db, create_pool
from app.embeddings import Embedder
from app.routers import documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = await create_pool(settings.database_url)
    await apply_migrations(app.state.pool)
    app.state.embedder = await asyncio.to_thread(Embedder, settings.embedding_model)
    yield
    await app.state.pool.close()


app = FastAPI(title="Reformulation Assistant", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router)


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


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
