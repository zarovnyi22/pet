from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db import apply_migrations, check_db, create_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = await create_pool(settings.database_url)
    await apply_migrations(app.state.pool)
    yield
    await app.state.pool.close()


app = FastAPI(title="Reformulation Assistant", version="0.1.0", lifespan=lifespan)


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
