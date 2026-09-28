from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
# Arbitrary constant: serializes migrations if several app instances start at once.
MIGRATION_LOCK_ID = 384_001


async def create_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=1, max_size=10)


async def apply_migrations(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1)", MIGRATION_LOCK_ID)
        try:
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                await conn.execute(path.read_text(encoding="utf-8"))
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", MIGRATION_LOCK_ID)


async def check_db(pool: asyncpg.Pool) -> str:
    try:
        async with pool.acquire(timeout=2) as conn:
            await conn.fetchval("SELECT 1")
        return "ok"
    except Exception as exc:  # health must report the failure, not raise
        return f"error: {type(exc).__name__}"
